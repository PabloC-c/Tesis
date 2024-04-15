from functions import *
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

## Arquitectura de la red
activation_list = ['relu']
layer_list = [2]
neuron_list = [5]
tols_list = [0.01]
max_iter = 50
n_input = 784
n_output = 10
n_clusters = 2
policy = 'path'
eta = True
## Clase real y clase target
real_label = 1
target_label = 7
## Se cargan las imagenes
test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(),download=True)
test_loader  = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)
## Se selecciona un solo par de entrada y salida correspondiente del conjunto de datos
input_example, output_example = next(iter(test_loader))
## Se transforma el input en una lista
hat_x = input_example[output_example == real_label][0].view(-1,784).tolist()[0]
## Heuristicas a utilizar
heu_methods = ['master_prop','pricing_prop','pricing_zfixed']
## Tipo de cotas para el modelo de verificacion estandar
type_bounds = 'verif_bounds'

## Por cada activacion
for activation in activation_list:
    ## Se recorren las capas
    for n_layers in layer_list:
        ## Se recorren las neuronas 
        for n_neurons in neuron_list:
            ## Se recorren las tolerancias para el input
            for eps in tols_list:
                ## Se crea la red
                net = neural_network(n_neurons,n_layers,activation,n_input,n_output)
                ## Archivo de los parametros
                filename = 'nn_parameters/{}_model_weights_L{}_n{}.pth'.format(activation,n_layers,n_neurons)
                ## Se cargan los parametros de la red
                net.load_state_dict(torch.load(filename))
                params = net.state_dict()
                filtered_params = filter_params(params)
                ##  MIP
                ## Nombre del archivo de las cotas
                bounds_file = calculate_bounds_file_name(type_bounds,activation,n_layers,n_neurons,eps,real_label)
                ## Se cargan las cotas del modelo
                bounds = read_bounds(True,n_layers,n_neurons,activation,bounds_file)
                ## Se cre el modelo MIP
                verif_model,_,mdenv_count = create_verification_model(params = filtered_params,
                                                                      bounds = bounds,
                                                                      activation = activation,
                                                                      tol_distance = eps,
                                                                      apply_softmax = False,
                                                                      image_list = hat_x,
                                                                      output_target = target_label,
                                                                      real_output = real_label,
                                                                      exact = 'no_exact',
                                                                      apply_bounds = True)
                ## Setup
                verif_model.hideOutput()
                verif_model.setPresolve(SCIP_PARAMSETTING.OFF)
                verif_model.setHeuristics(SCIP_PARAMSETTING.OFF)
                verif_model.disablePropagation()
                verif_model.optimize()
                ip_obj = verif_model.getObjVal()
                ## Relajacion lineal
                lp_verif_model,_,mdenv_count = create_verification_model(params = filtered_params,
                                                                                bounds = bounds,
                                                                                activation = activation,
                                                                                tol_distance = eps,
                                                                                apply_softmax = False,
                                                                                image_list = hat_x,
                                                                                output_target = target_label,
                                                                                real_output = real_label,
                                                                                exact = 'no_exact',
                                                                                lp_sol_file = '',
                                                                                apply_bounds = True,
                                                                                lp_relax = True)
                lp_verif_model.hideOutput()
                lp_verif_model.setPresolve(SCIP_PARAMSETTING.OFF)
                lp_verif_model.setHeuristics(SCIP_PARAMSETTING.OFF)
                lp_verif_model.disablePropagation()
                lp_verif_model.optimize()
                lp_obj = lp_verif_model.getObjVal()
                ## Column generation
                ## Particion
                partition = create_disjoint_partition(n_input,
                                                      n_output,
                                                      n_neurons,
                                                      n_layers,
                                                      n_clusters,
                                                      filtered_params,
                                                      policy,
                                                      use_bias = False)
                ## Aristas a relajar 
                edges_p = create_edges_p(partition,n_input,n_neurons,n_layers)
                ## Parametros lambda
                lambda_params = {}
                for edge in edges_p:
                    lambda_params[edge] = 0
                ## Se generan 3 columnas iniciales
                ref0 = hat_x
                ref1 = [min(max(val-eps,0),1) for val in hat_x]
                ref2 = [min(max(val+eps,0),1) for val in hat_x]
                columns = [propagate_input_to_column(ref0,filtered_params,edges_p,activation),
                           propagate_input_to_column(ref1,filtered_params,edges_p,activation),
                           propagate_input_to_column(ref2,filtered_params,edges_p,activation)]
                ## Arrelos para guardar las soluciones y valores obejtivos
                all_sols = {'master':[],'pricing':[]}
                best_sol = {'master':{},'pricing':{}}
                for heu in heu_methods:
                    all_sols[heu] = []
                    best_sol[heu] = {}
                ## Valores iniciales
                master_model = None
                cons_dict = {}
                theta = []
                master_times = []
                pricing_times = []
                pricing_models = []
                pricing_vars = {}
                try:
                    ## Iteraciones
                    counter = 1
                    while counter <= max_iter:
                        print(f'\n ========== Iteracion {counter} ========== \n')
                        ## Modelo maestro
                        max_value = max([lambda_params[llave] for llave in lambda_params])
                        min_value = min([lambda_params[llave] for llave in lambda_params])
                        print(f' Mayor lambda: {max_value}')
                        print(f' Menor lambda: {min_value}')
                        
                        master_model,master_obj,master_sol,lambda_params,theta,cons_dict,master_times = master_iteration(columns,
                                                                                                                         edges_p,
                                                                                                                         real_label,
                                                                                                                         target_label,
                                                                                                                         filtered_params,
                                                                                                                         master_times,
                                                                                                                         master_model = None,
                                                                                                                         theta = [],
                                                                                                                         cons_dict = {})
                        print(' master resuelto')
                        ## Se guarda el mejor valor objetivo del master
                        if len(all_sols['master']) == 0: 
                            all_sols['master'].append(master_obj)
                            best_sol['master'] = master_sol
                        else:
                            if master_obj < all_sols['master'][-1]:
                                all_sols['master'].append(master_obj)
                                best_sol['master'] = master_sol
                            else:
                                all_sols['master'].append(all_sols['master'][-1])
                        ## Pricing
                        pricing_models,pricing_obj,pricing_sol,pricing_vars,pricing_times = pricing_iteration(n_clusters,
                                                                                                              partition,
                                                                                                              edges_p,
                                                                                                              lambda_params,
                                                                                                              bounds,
                                                                                                              filtered_params,
                                                                                                              hat_x,
                                                                                                              real_label,
                                                                                                              target_label,
                                                                                                              eps,
                                                                                                              pricing_times,
                                                                                                              activation,
                                                                                                              pricing_models,
                                                                                                              pricing_vars,
                                                                                                              eta)
                        print(' pricing resuelto')
                        ## Se guarda el mejor valor objetivo del pricing
                        if len(all_sols['pricing'])==0: 
                            all_sols['pricing'].append(pricing_obj)
                            best_sol['pricing'] = pricing_sol
                        else:
                            if pricing_obj > all_sols['pricing'][-1]:
                                all_sols['pricing'].append(pricing_obj)
                                best_sol['pricing'] = pricing_sol
                            else:
                                all_sols['pricing'].append(all_sols['pricing'][-1])
                        ## Heuristicas
                        all_sols,best_sol = create_heuristic_sols(heu_methods,
                                                                  all_sols,
                                                                  best_sol,
                                                                  master_sol,
                                                                  master_obj,
                                                                  pricing_sol,
                                                                  pricing_obj,
                                                                  filtered_params,
                                                                  bounds,
                                                                  real_label,
                                                                  target_label,
                                                                  pricing_models,
                                                                  pricing_vars,
                                                                  partition,
                                                                  activation = 'relu')
                        is_master_feasible,infactibility = check_feasiblity(master_sol,filtered_params,bounds,activation)
                        print(f' Cota master: {master_obj}, Factibilidad: {is_master_feasible}')
                        print(f' Cota pricing: {pricing_obj}')
                        ## Se añade la columna correspondiente 
                        columns.append(get_pricing_column(pricing_models,edges_p))
                        ## Se aumenta el contador
                        counter +=1
                except:
                    print(' Iteraciones finalizadas')
                
                ## Excel de tiempos
                while len(master_times) > len(pricing_times):
                    pricing_times.append('-')
                while len(pricing_times) > len(master_times):
                    master_times.append('-') 
                df_times = pd.DataFrame({'master': master_times,'pricing':pricing_times})
                #df_times.to_excel(f'graphs/time_L{n_layers}_n{n_neurons}_eps{int(100*eps)}_policy_{policy}_clusters{n_clusters}_iters{max_iter}.xlsx', index=None)
                ## Total de iteraciones
                total_iteration = max([len(all_sols[method]) for method in all_sols])
                
                ## Grafico
                lp_list = [lp_obj for i in range(total_iteration)]
                ip_list = [ip_obj for i in range(total_iteration)]

                ## Colores
                colors = ['k','b','g','c','m','r','y']
                ## Lista de etiquetas
                list_labels = ['MILP',
                               'LP',
                               'Master',
                               'Pricing',
                               'Master propagation',
                               'Pricing propagation',
                               'Pricing z-fixed']
                ## Lista con toda la informacion
                all_info = [ip_list,
                            lp_list,
                            all_sols['master'],
                            all_sols['pricing']]
                for heu in heu_methods:
                    all_info.append(all_sols[heu])

                ## Eje x
                x_axis = list(range(1, total_iteration + 1))
                
                plt.figure(figsize=(25, 12))
                ## Grafica cada lista con su respectivo color y etiqueta
                for aux_lista, aux_color, aux_label in zip(all_info,colors,list_labels):
                    if aux_label in ['MILP','LP']:
                        plt.plot(x_axis,aux_lista,'-',color=aux_color,label=aux_label)
                    elif aux_label == 'Pricing z-fixed':
                        if len(aux_lista) > 0:
                            while len(aux_lista) < total_iteration:
                                aux_lista.append(aux_lista[-1])
                            plt.plot(x_axis,aux_lista,'|--',color=aux_color,label=aux_label,dashes=(3,4),alpha=1)
                    else: 
                        while len(aux_lista) < total_iteration:
                            aux_lista.append(aux_lista[-1])
                        plt.plot(x_axis,aux_lista,'|--',color=aux_color,label=aux_label,dashes=(3,4),alpha=1)
                plt.xticks(x_axis)
                plt.legend()
                #plt.savefig(f'graphs/L{n_layers}_n{n_neurons}_eps{int(100*eps)}_policy_{policy}_clusters{n_clusters}_iters{max_iter}.png')
                #plt.show()