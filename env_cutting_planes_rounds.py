import os
import sys
import pandas as pd
from torchvision import datasets, transforms
from functions import *

activation_list = ['sigmoid']
layer_list = [2]
neuron_list = [5]
exact = 'env_cut'      # multidim_env   exact{exact: exacto, no_exact: formulaciones alternas o envolturas, prop: modelo para calcular las cotas solo con propagacion}
apply_bounds = True
type_cuts_list = ['R_H,f,i']
minutes = 1
save_image = False
apply_softmax = False
tols_list = [0.01,0.05]

root_node_only = False
add_verif_bounds = True
set_initial_sol = False
print_output = True
save_results = True
real_output = 1
target_output = 2
input_lb = 0 
input_ub = 1
n_input = 784

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

print('Hola1')

## Por cada activacion
for activation in activation_list:
    print('Hola2')
    ## Se recorren las capas
    for n_layers in layer_list:
        print('Hola3')
        ## Se recorren las neuronas 
        for n_neurons in neuron_list:
            print('Hola4')
            for tol_distance in tols_list:
                print('Hola5')
                ## Lista de info a guardar 
                new_line = [n_layers,n_neurons,tol_distance]
                for type_cut in type_cuts_list:
                    print('Tolerancia: {} '.format(tol_distance))
                    print('Tipo de corte ',type_cut)
                    ## Nombre del archivo xlsx donde se guardan los resultados de los experimentos
                    file_name = calculate_verif_file_name(exact,activation,real_output,target_output,root_node_only)
                    ## Se crea la instancia de la red neuronal
                    net = neural_network(n_neurons,n_layers,activation)
                    ## Se cargan los parámetros de la red
                    net.load_state_dict(torch.load('nn_parameters/{}_model_weights_L{}_n{}.pth'.format(activation,n_layers, n_neurons)))
                    ## Se filtran los parametros
                    params = filter_params(net.state_dict())
                    ## Diccionario para las cotas
                    bounds = OrderedDict()
                    ## Se inicializa el modelo de verificacion
                    bounds_model,bounds,inpt,all_vars = initialize_neuron_model(bounds,add_verif_bounds,tol_distance,image_list,n_input)
                    ## Tiempo y cortes de la red
                    net_time = 0
                    net_cuts = 0
                    h_cuts = 0
                    ## Capa de la red
                    for l in range(n_layers):
                        ## Parametros capa l
                        weight,bias = get_w_b_names(l)
                        W = params[weight]
                        ## Lista para guardar las cotas de la capa
                        layer_bounds_list = []
                        ## Neurona de la red
                        for i in range(n_neurons):
                            ## Lista para las cotas de la neurona
                            neuron_bounds = []
                            ## Sentido del primer problema a resolver
                            sense = 'minimize'
                            ## Calculo de cotas
                            for q in range(2):
                                ## Solucion encontrada
                                done = False
                                ## Cantidad de rondas
                                round_count = 0
                                ## Cantidad de cortes añadidos
                                neuron_cuts = 0
                                ## Solucion objetivo previa
                                prev_obj = None
                                ## Tiempo que ha demorado
                                neuron_time = 0
                                print('\n ===== Red: {}x{}, Capa: {}, Neurona: {} ===== \n'.format(n_layers,n_neurons,l,i))
                                while not done:
                                    print('\n === Ronda {} === \n'.format(round_count))
                                    ## Se fija la funcion objetivo
                                    bounds_model = set_objective_function(bounds_model,inpt,params,bounds,l,i,sense)
                                    ## Se calcula la cota 
                                    sol_list,dt = solve_neuron_model(bounds_model,sense,params,bounds,l,i,exact,minutes,print_output)
                                    ## Primera capa oculta
                                    if l == 0 or type_cut == 'R_H':
                                        ## Ya se tiene la solucion 
                                        done = True
                                        ## Se libera el modelo
                                        bounds_model.freeTransform()
                                    else:
                                        bounds_model.writeLP('bounds_model')
                                        ## Se verifica si la solucion converge
                                        if prev_obj is None:
                                            prev_obj = bounds_model.getObjVal()
                                        else:
                                            obj = bounds_model.getObjVal()
                                            if np.abs(obj-prev_obj) < 1E-09:
                                                done = True
                                            else:
                                                prev_obj = obj
                                        ## Se obtiene la solucion
                                        lp_sol = get_bounds_model_lpsol(l,n_input,n_neurons,bounds_model,all_vars)
                                        ## Se libera el modelo
                                        bounds_model.freeTransform()
                                        ## Se intenta añadir cortes
                                        bounds_model,mdenv_count = env_cut_verif_model_lp_sol(l,n_input,n_neurons,activation,params,bounds,bounds_model,all_vars,lp_sol)
                                        ## Caso en que no se añadieron mas cortes
                                        if mdenv_count == 0:
                                            print('{} cortes añadidos'.format(mdenv_count))
                                            done = True
                                        else:
                                            print('{} cortes añadidos'.format(mdenv_count))
                                            neuron_cuts += mdenv_count
                                    ## Se actualiza el tiempo
                                    neuron_time += dt
                                    ## Caso en que se ha acabado el tiempo
                                    if neuron_time >= minutes*60:
                                        done = True
                                    ## Se actualiza la ronda
                                    round_count += 1
                                ## Se guarda la cota
                                neuron_bounds.append(sol_list[1])
                                ## Se actualiza el tiempo de la red
                                net_time += neuron_time
                                ## Se actualizan los cortes de la red
                                net_cuts += neuron_cuts
                                ## Sentido del segundo problema a resolver
                                sense = 'maximize'
                            ## Se añaden las cotas de la neurona a las cotas de la capa
                            layer_bounds_list.append((neuron_bounds[0],neuron_bounds[1]))
                        bounds[l] = layer_bounds_list
                        bounds_model,inpt,all_vars,h_cuts = update_neuron_model(bounds_model,inpt,all_vars,params,bounds,l,h_cuts,activation,exact)
                    ## Se actualiza la informacion de la red
                    new_line.append(net_time,net_cuts,h_cuts)