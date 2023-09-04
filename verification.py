import os
import sys
import pandas as pd
from torchvision import datasets, transforms
from functions import *

activation_list = ['relu']
layer_list = [2,3,4]
neuron_list = [5,10]
form_list = ['exact','no_exact']        # exact{exact: exacto, no_exact: formulaciones alternas o envolturas, prop: modelo para calcular las cotas solo con propagacion}
apply_bounds_list = [False,True]
type_bounds_list = ['prop','mix']
minutes = 15
save_image = False
apply_softmax = False

set_initial_sol = True
print_output = True
save_results = True
real_output = 1
target_output = 7
input_lb =0 
input_ub = 1
tols_list = [0.01,0.05]

if len(sys.argv) > 1:
    activation_list = [sys.argv[1]]
    if len(sys.argv) >= 3:
        layer_list = [int(sys.argv[2])]
    if len(sys.argv) >= 4:
        neuron_list = [int(sys.argv[3])]
    if len(sys.argv) >= 5:
        form_list = [sys.argv[4]]
    if len(sys.argv) >= 6:
        apply_bounds_list = [bool(sys.argv[5])]
    if len(sys.argv) >= 7:
        type_bounds_list = [sys.argv[6]]
    if len(sys.argv) >= 8:
        save_image = bool(sys.argv[7])
    if len(sys.argv) >= 9:    
        apply_softmax = bool(sys.argv[10])

## Se cargan las imagenes
test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(),download=True)
test_loader  = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)
## Se selecciona un solo par de entrada y salida correspondiente del conjunto de datos
input_example, output_example = next(iter(test_loader))
## Se transforma el input en una lista
image_list = input_example[output_example == real_output][0].view(-1,784).tolist()[0]

## Por cada activacion
for activation in activation_list:
    ## Se recorren las capas
    for n_layers in layer_list:
        ## Se recorren las neuronas 
        for n_neurons in neuron_list:
            ## Se recorren las formulaciones
            for exact in form_list:
                file_name = 'verif_results/{}/datos_verificacion_{}_{}como{}.xlsx'.format(exact,activation,real_output,target_output)
                if activation != 'relu' and exact == 'no_exact':
                    break
                for tol_distance in tols_list:
                    ## Lista de info a guardar 
                    new_line = [n_layers,n_neurons,tol_distance]
                    aux_list = apply_bounds_list
                    if exact == 'no_exact':
                        aux_list = [True]
                    ## Se aplica o no se aplican las cotas
                    for apply_bounds in aux_list:
                        aux_bounds_list = type_bounds_list
                        if not apply_bounds:
                            aux_bounds_list = ['-']
                        ## Se recorren los tipos de cotas
                        for type_bounds in aux_bounds_list:
                            ## Caso donde se deben aplicar las cotas
                            if apply_bounds:
                                if type_bounds == 'prop':
                                    bounds_file = 'nn_bounds/{}_prop_bounds_L{}_n{}.txt'.format(activation,n_layers,n_neurons)
                                else:
                                    bounds_file = 'nn_bounds/{}_bounds_L{}_n{}.txt'.format(activation,n_layers,n_neurons)
                                ## Se cargan las cotas de la red
                                if os.path.exists(bounds_file):
                                    bounds = read_bounds(n_layers,n_neurons,activation,bounds_file)
                                else:
                                    print('archivo no existe')
                                    print(bounds_file)
                                    break
                            else:
                                bounds = OrderedDict()
                            ## Se crea la instancia de la red neuronal
                            net = neural_network(n_neurons,n_layers,activation)
                            ## Se cargan los parámetros de la red
                            net.load_state_dict(torch.load('nn_parameters/{}_model_weights_L{}_n{}.pth'.format(activation,n_layers, n_neurons)))
                            params = net.state_dict()
                            ## Se filtran los parametros
                            filtered_params = filter_params(params)
                            ## Busqueda de ejemplo adversarial
                            print('\n===== Capas {} Neuronas {} =====\n'.format(n_layers,n_neurons))
                            print('Tolerancia: {} '.format(tol_distance))
                            adv_ex = False
                            ## Se crea el modelo de verificacion
                            verif_model,all_vars = create_verification_model(params,bounds,activation,tol_distance,apply_softmax,image_list,target_output,real_output,exact,apply_bounds)
                            ## Se añade la solucion inicial
                            if set_initial_sol:
                                initial_sol,image_vars = create_initial_sol(verif_model,params,image_list,exact,activation,apply_softmax)
                                accepted = verif_model.addSol(initial_sol)
                                print('\n Solucion inicial aceptada:',accepted,'\n')
                            if print_output:
                                verif_model.redirectOutput()
                            else:
                                verif_model.hideOutput()
                            ## Se generan las variables correspondientes a la imagen entregada al modelo
                            
                            ## Se limita el tiempo de resolucion
                            verif_model.setParam('limits/time', int(60*minutes))
                            ## Se aumenta la tolerancia de factibilidad
                            verif_model.setParam('numerics/feastol', 1E-5)
                            ## Se optimiza el modelo en busca del ejemplo adversarial
                            t0 = time.time()
                            try:
                                aux_t = time.time()
                                verif_model.optimize()
                                dt = time.time() - aux_t
                                model_status = verif_model.getStatus()
                            except:
                                dt = time.time() - t0
                                model_status = 'problem'
                            ## Caso solucion optima
                            if model_status == 'optimal':
                                gap = 0.0
                                solution = [verif_model.getVal(all_vars['h{},{}'.format(-1,i)]) for i in range(len(image_list))]
                                obj_val  = verif_model.getObjVal()
                                if obj_val > 0:
                                    ## Se encontro un ejemplo adversarial
                                    adv_ex = True
                                    ## Caso en que se debe guardar la imagen
                                    if save_image:
                                        color_map = 'gray'
                                        png_name  = '{}_adv_ex.png'.format(activation) 
                                        generate_png(solution,image_list,color_map,png_name,input_lb,input_ub)
                                        output,soft_output = calculate_probs(net,solution)
                                        print('\n Softmax output: ',soft_output,'\n')
                            ## Caso no alcanzo el tiempo
                            else:
                                try:
                                    primalb = verif_model.getDualbound()
                                    dualb = verif_model.getPrimalbound()
                                    if (primalb == 1e+20) or (dualb == -1e+20):
                                        gap = '-'
                                    else:
                                        gap = (primalb-dualb)/np.abs(dualb)
                                    ## Caso en que no existe ejemplo adversarial
                                    if primalb < 0:
                                        obj_val = '<0'
                                    ## Caso en que si existe un ejemplo adversarial
                                    elif dualb > 0:
                                        obj_val = '>0'
                                        adv_ex = True
                                except:
                                    gap = '-'
                                    obj_val = '-'
                            if save_results:
                                ## Se genera la nueva linea del df
                                adv_aux = 'No'
                                if adv_ex:
                                    adv_aux = 'Si'
                                new_line += [type_bounds,dt,gap,adv_aux,model_status]
                    if save_results:
                        ## Se lee el df existente o se genera uno nuevo
                        df = read_df(file_name)
                        ## Se añade la linea al df
                        df = df._append(pd.Series(new_line), ignore_index=True)
                        ## Se intenta guardar el nuevo df actualizado
                        save_df(df,file_name)            