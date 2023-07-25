import os
import sys
import pandas as pd
from torchvision import datasets, transforms
from functions import *

activation_list = ['relu']
layer_list = [2]
neuron_list = [10]
form_list = ['exact']        # exact{exact: exacto, no_exact: formulaciones alternas o envolturas, prop: modelo para calcular las cotas solo con propagacion}
apply_bounds_list = [True]
type_bounds_list = ['prop']
minutes = 1
save_image = True
apply_softmax = False

print_output = False
real_output = 1
target_output = 8
input_lb =0 
input_ub = 1
tol_0 = 0.01
tol_f = 0.05
tol_step = 0.005

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
                file_name = 'verif_results/{}/datos_verificacion_{}.xlsx'.format(exact,activation)
                if activation != 'relu' and exact == 'no_exact':
                    break
                aux_list = apply_bounds_list
                if exact == 'no_exact':
                    aux_list = [True]
                ## Se aplica o no se aplican las cotas
                for apply_bounds in aux_list:
                    if not apply_bounds:
                        type_bounds_list = ['-']
                    ## Se recorren los tipos de cotas
                    for type_bounds in type_bounds_list:
                        if True:
                            ## Se crea la instancia de la red neuronal
                            net = neural_network(n_neurons,n_layers,activation)
                            ## Se cargan los parámetros de la red
                            net.load_state_dict(torch.load('nn_parameters/{}_model_weights_L{}_n{}.pth'.format(activation,n_layers, n_neurons)))
                            params = net.state_dict()
                            ## Se filtran los parametros
                            filtered_params = filter_params(params)
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
                                    break
                            else:
                                bounds = OrderedDict()
                            ## Datos para el ciclo
                            adv_ex = False
                            tol_distance = tol_0
                            ## Ciclo para busqueda de ejemplo adversarial
                            print('\n===== Capas {} Neuronas {} =====\n'.format(n_layers,n_neurons))
                            while not adv_ex and tol_distance <= tol_f:
                                ## Se crea el modelo de verificacion
                                verif_model,all_vars = create_verification_model(params,bounds,activation,tol_distance,apply_softmax,image_list,target_output,real_output,exact,apply_bounds)
                                if print_output:
                                    verif_model.redirectOutput()
                                else:
                                    verif_model.hideOutput()
                                ## Se limita el tiempo de resolucion
                                verif_model.setParam('limits/time', int(60*minutes))
                                ## Se optimiza el modelo en busca del ejemplo adversarial
                                auxt = time.time()
                                verif_model.optimize()
                                dt = time.time() - auxt 
                                ## Caso solucion optima
                                if verif_model.getStatus() == 'optimal':
                                    gap = 0.0
                                    solution      = [verif_model.getVal(all_vars['h{},{}'.format(-1,i)]) for i in range(len(image_list))]
                                    output,probs  = calculate_probs(net, solution)
                                    real_prob     = probs[real_output]
                                    target_prob   = probs[target_output]
                                    obj_val       = verif_model.getObjVal()
                                    if obj_val > 0:
                                        ## Se encontro un ejemplo adversarial
                                        adv_ex = True
                                        ## Caso en que se debe guardar la imagen
                                        if save_image:
                                            color_map = 'gray'
                                            png_name  = '{}_adv_ex.png'.format(activation) 
                                            generate_png(solution,image_list,color_map,png_name,input_lb,input_ub)
                                        break
                                ## Caso no alcanzo el tiempo
                                else:
                                    primalb = verif_model.getDualbound()
                                    dualb = verif_model.getPrimalbound()
                                    gap = (dualb-primalb)/np.abs(dualb)
                                    target_prob,real_prob = '-','-'
                                    ## Caso en que no existe ejemplo adversarial
                                    if primalb < 0:
                                        obj_val = '<0'
                                    ## Caso en que si existe un ejemplo adversarial
                                    elif dualb > 0:
                                        obj_val = '>0'
                                        adv_ex = True
                                        break
                                ## Se aumenta la tolerancia
                                print('Nuevo intento')
                                tol_distance += tol_step
                            
                            ## Se lee el df existente o se genera uno nuevo
                            df = read_df(file_name)
                            ## Se genera la nueva linea del df                  
                            new_line = [n_layers,n_neurons,type_bounds,tol_distance,dt,gap,adv_ex,obj_val]
                            ## Se añade la linea al df
                            df = df._append(pd.Series(new_line), ignore_index=True)
                            ## Se intenta guardar el nuevo df actualizado
                            file_name = 'datos_verificacion_{}.xlsx'.format(activation)
                            save_df(df,file_name)
                        else:
                            print('error')