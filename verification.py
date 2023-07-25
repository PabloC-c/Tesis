import os
import sys
import pandas as pd
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from functions import *

activation_list = ['relu']
layer_list = [2]
neuron_list = [50]
exact = 'no_exact'        # exact{prop: propagacion, exact: exacto, no_exact: formulaciones o envolturas}
apply_bounds = True
minutes = 15
save_image = False
apply_softmax = False
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
        exact = sys.argv[4]
    if len(sys.argv) >= 6:
        apply_bounds = bool(sys.argv[5])
    if len(sys.argv) >= 7:
        save_image = bool(sys.argv[6])
    if len(sys.argv) >= 8:    
        apply_softmax = bool(sys.argv[7])

## Se cargan las imagenes
test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(),download=True)
test_loader  = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)
## Se selecciona un solo par de entrada y salida correspondiente del conjunto de datos
input_example, output_example = next(iter(test_loader))
## Se transforma el input en una lista
image_list = input_example[output_example == real_output][0].view(-1,784).tolist()[0]

## Por cada activacion
for activation in activation_list:
    ## Se lee el archivo con los datos previos, en caso de haber
    if os.path.exists('datos_verificacion_{}.xlsx'.format(activation)):
        df = pd.read_excel('datos_verificacion_{}.xlsx'.format(activation),header=None)
    ## Caso contrario se crea un df vacio
    else:
        df = pd.DataFrame()
    ## Se recorren las capas
    for n_layers in layer_list:
    ## Se recorren las neuronas 
        for n_neurons in neuron_list:
            try:
                ## Se crea la instancia de la red neuronal
                net = neural_network(n_neurons,n_layers,activation)
                ## Se cargan los parámetros de la red
                net.load_state_dict(torch.load('nn_parameters/{}_model_weights_L{}_n{}.pth'.format(activation,n_layers, n_neurons)))
                params = net.state_dict()
                ## Se filtran los parametros
                filtered_params = filter_params(params)
                ## Caso donde se deben aplicar las cotas
                if apply_bounds:
                    if exact == 'prop':
                        file_name = 'nn_bounds/{}_prop_bounds_L{}_n{}.txt'.format(activation,n_layers,n_neurons)
                    else:
                        file_name = 'nn_bounds/{}_bounds_L{}_n{}.txt'.format(activation,n_layers,n_neurons)
                    ## Se cargan las cotas de la red
                    if os.path.exists(file_name):
                        bounds = read_bounds(n_layers,n_neurons,activation)
                    else:
                        break
                ## Datos para el ciclo
                adv_ex = False
                tol_distance = tol_0
                ## Ciclo para busqueda de ejemplo adversarial
                while not adv_ex and tol_distance <= tol_f:
                    ## Se crea el modelo de verificacion
                    verif_model,all_vars = create_verification_model(params,bounds,activation,tol_distance,apply_softmax,image_list,target_output,real_output,exact,apply_bounds)
                    #verif_model.redirectOutput()
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
                                png_name  = '{}_adv_exp.png'.format(activation) 
                                generate_png(solution,image_list,color_map,png_name)
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
                            adv_ex = False
                        ## Caso en que si existe un ejemplo adversarial
                        elif dualb > 0:
                            obj_val = '>0'
                            adv_ex = True
                            break
                    ## Se aumenta la tolerancia
                    tol_distance += tol_step
                
                ## Se genera la nueva linea del df                  
                new_line = [n_layers,n_neurons,tol_distance,dt,gap,adv_ex,obj_val]
                ## Se añade la linea al df
                df = df._append(pd.Series(new_line), ignore_index=True)
                ## Se intenta guardar el nuevo df actualizado
                written = False
                while not written:
                    try:
                        df.to_excel('datos_verificacion_{}.xlsx'.format(activation), header = False, index = False)
                        written = True
                    except:
                        time.sleep(5) 
                
            except:
                print('error')
