import os
import sys
import pandas as pd
from torchvision import datasets, transforms
from functions import *

activation_list = ['sigmoid']
layer_list = [2,3,4]
neuron_list = [5,10]
exact = 'env_cut'      # multidim_env   exact{exact: exacto, no_exact: formulaciones alternas o envolturas, prop: modelo para calcular las cotas solo con propagacion}
apply_bounds = True
type_cuts_list = ['R_H','R_H,f','R_H,f,i']
minutes = 10
save_image = False
apply_softmax = False
tols_list = [0.01,0.05]
max_rounds = 20

root_node_only = False
add_verif_bounds = False
set_initial_sol = False
print_output = False
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

## Por cada activacion
for activation in activation_list:
    ## Archivo xlsx
    file_name = 'bounds_calculation/{}_rounds_{}_minutes_{}.xlsx'.format(activation,max_rounds,minutes)
    header = ['L','n']
    for k in type_cuts_list:
        for l in range(max(layer_list)):
            header += ['{}:time'.format(l),'{}:rounds'.format(l),'{}:cuts'.format(l),'{}:lb gap [%]'.format(l),'{}:ub gap [%]'.format(l)]
    if os.path.exists(file_name) and save_results:
        read = False
        while not read:
            try:
                df = pd.read_excel(file_name,header = 0)
                read = True
            except:
                time.sleep(10)
    else:
        df = pd.DataFrame(columns=header)
    ## Se recorren las capas
    for n_layers in layer_list:
        ## Se recorren las neuronas 
        for n_neurons in neuron_list:
            ## Caso sin verificacion
            if not add_verif_bounds:
                tols_list_range = [tols_list[0]]
            else:
                tols_list_range = tols_list
            for tol_distance in tols_list_range:
                ## Lista de info a guardar 
                new_line = [n_layers,n_neurons]
                bounds_list = []
                for type_cut in type_cuts_list:
                    print('Tolerancia: {} '.format(tol_distance))
                    print('Tipo de corte ',type_cut)
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
                    ## Cotas de verificacion
                    if add_verif_bounds:
                        sol_dict = {}
                        sol_dict,layer_input = update_bounds_vars_by_image(-1,params,sol_dict,image_list,exact,activation)
                        accepted = add_bounds_vars_by_image(bounds_model,sol_dict)
                    ## Tiempo y cortes de la red
                    net_time = 0
                    net_cuts = 0
                    h_cuts = 0
                    ## Tiempos, rondas, cortes e intervalos promedio de las capas
                    layers_times = []
                    layers_rounds = []
                    layers_cuts = []
                    layers_lb_gaps = []
                    layers_ub_gaps = []
                    ## Capa de la red
                    for l in range(n_layers):
                        ## Parametros capa l
                        weight,bias = get_w_b_names(l)
                        W = params[weight]
                        ## Neuronas de la capa 
                        layer_neurons = W.size()[0]
                        ## Lista para guardar las cotas de la capa
                        layer_bounds_list = []
                        ## Datos de la capa
                        layer_time = 0
                        layer_rounds = 0
                        layer_cuts = 0 
                        layer_lb_gap = 0
                        layer_ub_gap = 0
                        ## Neurona de la red
                        for i in range(layer_neurons):
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
                                ## Razon de termino
                                reason = None
                                print('\n ===== Red: {}x{}, Capa: {}, Neurona: {} ===== \n'.format(n_layers,n_neurons,l,i))
                                print(' Sentido:',sense)
                                while not done and round_count < max_rounds and neuron_time <= minutes*60:
                                    print('\n === Ronda {} === \n'.format(round_count+1))
                                    ## Tiempo de la ronda
                                    round_time0 = time.time()
                                    ## Se fija la funcion objetivo
                                    bounds_model = set_objective_function(bounds_model,inpt,params,bounds,l,i,sense)
                                    ## Se calcula la cota
                                    sol_list,dt = solve_neuron_model(bounds_model,sense,params,bounds,l,i,exact,minutes,print_output)
                                    ## Primera capa oculta
                                    if l < 1 or type_cut == 'R_H':
                                        if l == 0:
                                            reason = 'Primera capa oculta'
                                        else:
                                            reason = 'Caso solo cortes H'
                                        ## Ya se tiene la solucion
                                        done = True
                                        ## Se libera el modelo
                                        bounds_model.freeTransform()
                                        if add_verif_bounds:
                                            accepted = add_bounds_vars_by_image(bounds_model,sol_dict)
                                    else:
                                        ## Se verifica si la solucion converge
                                        if prev_obj is None:
                                            prev_obj = bounds_model.getObjVal()
                                        else:
                                            obj = bounds_model.getObjVal()
                                            if np.abs(obj-prev_obj) < 1E-09:
                                                reason = 'La solucion converge'
                                                done = True
                                            else:
                                                prev_obj = obj
                                        ## Se obtiene la solucion
                                        lp_sol = get_bounds_model_lpsol(l,bounds_model,all_vars,params)
                                        ## Se libera el modelo
                                        bounds_model.freeTransform()
                                        if add_verif_bounds:
                                            accepted = add_bounds_vars_by_image(bounds_model,sol_dict)
                                        ## Se intenta añadir cortes
                                        print('Añadiendo cortes')
                                        bounds_model,mdenv_count = env_cut_verif_model_lp_sol(l,activation,params,bounds,bounds_model,all_vars,lp_sol,type_cut)
                                        ## Caso en que no se añadieron mas cortes
                                        if mdenv_count == 0:
                                            reason = 'No se añadieron cortes'
                                            done = True
                                        else:
                                            print('\n {} cortes añadidos'.format(mdenv_count))
                                            neuron_cuts += mdenv_count
                                    ## Se actualiza el tiempo
                                    neuron_time += time.time()-round_time0
                                    ## Se actualiza la ronda
                                    round_count += 1
                                if not reason is None:
                                    print('\n',reason,'\n')
                                elif round_count >= max_rounds:
                                    print('\n Se alcanzo el limite de rondas\n')
                                else:
                                    print('\n Se alcanzo el limite de tiempo\n')
                                ## Se guarda la cota
                                neuron_bounds.append(sol_list[1])
                                ## Se actualiza el tiempo de la red
                                net_time += neuron_time
                                ## Se actualizan los cortes de la red
                                net_cuts += neuron_cuts
                                ## Se guardan los datos de la capa
                                layer_time += neuron_time
                                layer_rounds += round_count
                                layer_cuts += neuron_cuts
                                ## Sentido del segundo problema a resolver
                                sense = 'maximize'
                            ## Se añaden las cotas de la neurona a las cotas de la capa
                            layer_bounds_list.append((neuron_bounds[0],neuron_bounds[1]))
                            ## Se actualizan los datos de la capa
                            if len(bounds_list) == 0:
                                layer_lb_gap += 0
                                layer_ub_gap += 0
                            else:
                                layer_lb_gap += np.abs(bounds_list[0][l][i][0] - neuron_bounds[0])/np.abs(bounds_list[0][l][i][0])
                                layer_ub_gap += np.abs(bounds_list[0][l][i][1] - neuron_bounds[1])/np.abs(bounds_list[0][l][i][1])
                        ## Se guardan las cotas de la capa
                        bounds[l] = layer_bounds_list 
                        ## Se actualiza el modelo
                        bounds_model,inpt,all_vars,h_cuts = update_neuron_model(bounds_model,inpt,all_vars,params,bounds,l,h_cuts,activation,exact)
                        if add_verif_bounds:
                            sol_dict,layer_input = update_bounds_vars_by_image(l,params,sol_dict,layer_input,exact,activation)
                            accepted = add_bounds_vars_by_image(bounds_model,sol_dict)
                        ## Se guardan los datos de la capa
                        layers_times.append(layer_time/layer_neurons)
                        layers_rounds.append(layer_rounds/layer_neurons)
                        layers_cuts.append(layer_cuts/layer_neurons)
                        layers_lb_gaps.append(100*layer_lb_gap/layer_neurons)
                        layers_ub_gaps.append(100*layer_ub_gap/layer_neurons)
                        ## Se actualiza la fila del df
                        new_line += [layers_times[-1],layers_rounds[-1],layers_cuts[-1],layers_lb_gaps[-1],layers_ub_gaps[-1]]
                    ## Se actualiza la informacion de la red
                    if n_layers < max(layer_list):
                        for k in range(max(layer_list)-n_layers):
                            new_line += ['-','-','-','-','-']
                    ## Se guardan las cotas
                    bounds_file = 'nn_bounds/{}_bounds_L{}_n{}_{}_cut_rounds_{}_minutes_{}_.txt'.format(activation,n_layers,n_neurons,type_cut,max_rounds,minutes)
                    bounds_writen = write_bounds(bounds,n_layers,n_neurons,activation,bounds_file)
                    bounds_list.append(bounds)
                print('header',header)
                print(len(header))
                print('new line',new_line)
                print(len(new_line))
                df.loc[len(df)] = new_line
                if save_results:
                    writen = False
                    while not writen:
                        try:
                            df.to_excel(file_name,header = header, index = False)
                            writen = True
                        except:
                            time.sleep(10)