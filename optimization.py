import os
import sys
import time
import torch
import pandas as pd
import torch.nn as nn
from functions import *
from collections import OrderedDict
from pyscipopt import Model,quicksum
from torchvision import datasets, transforms

activation_list = ['relu','softplus','sigmoid']
layer_list = [2,3,4] 
neuron_list = [5,10]
exact = 'exact'
minutes = 10
filter_tol = 1e-5
add_verif_bounds = True
print_output=True

tol_distance = 0.05
real_output = 1

if len(sys.argv) > 1:
    activation_list = [sys.argv[1]]
    if len(sys.argv) >= 3:
        layer_list = [int(sys.argv[2])]
    if len(sys.argv) >= 4:
        neuron_list = [int(sys.argv[3])]
    if len(sys.argv) >= 5:
        exact = sys.argv[4]
    if len(sys.argv) >= 6:
        minutes = float(sys.argv[5])    
    if len(sys.argv) >= 7:
        filter_tol = float(sys.argv[6])
        
if add_verif_bounds:
    ## Se cargan las imagenes
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(),download=True)
    test_loader  = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)
    ## Se selecciona un solo par de entrada y salida correspondiente del conjunto de datos
    input_example, output_example = next(iter(test_loader))
    ## Se transforma el input en una lista
    image_list = input_example[output_example == real_output][0].view(-1,784).tolist()[0]
else:
    image_list = []

for activation in activation_list:
    ## Caso cotas de verificacion
    if add_verif_bounds:
        ## Caso propagacion
        if exact == 'prop':
            data_file = 'datos_{}_verifbounds_target{}_tolper{}_prop.xlsx'.format(activation,real_output,int(tol_distance*100))
        ## Caso mixo
        else:
            data_file = 'datos_{}_verifbounds_target{}_tolper{}_mix.xlsx'.format(activation,real_output,int(tol_distance*100))
    ## Caso cotas iniciales
    else:
        ## Caso propagacion
        if exact == 'prop':
            data_file = 'datos_{}_prop.xlsx'.format(activation)
        ## Caso mixo
        else:
            data_file = 'datos_{}_mix.xlsx'.format(activation)
    ## Caso donde ya existe un df con datos previos
    if os.path.exists(data_file):
        df = pd.read_excel(data_file,header=None)
    ## Caso en que no hay datos previos
    else:
        df = pd.DataFrame()
    ## Se recorren las neuronas y la capas
    for n_neurons in neuron_list:
        for n_layers in layer_list:
            ## Se define el archivo donde se guardaran las cotas
            ## Caso solo propagacion
            if add_verif_bounds:
                ## Caso propagacion
                if exact == 'prop':
                    bounds_file = 'nn_bounds/{}_verifbounds_target{}_tolper{}_prop_L{}_n{}.txt'.format(activation,real_output,int(tol_distance*100),n_layers,n_neurons)
                ## Caso mixto
                else:
                    bounds_file = 'nn_bounds/{}_verifbounds_target{}_tolper{}_L{}_n{}.txt'.format(activation,real_output,int(tol_distance*100),n_layers,n_neurons)
            ## Caso cotas de verificacion
            else:
                ## Caso propagacion
                if exact == 'prop':
                    bounds_file = 'nn_bounds/{}_bounds_prop_L{}_n{}.txt'.format(activation,n_layers,n_neurons)
                ## Caso mixto
                else:
                    bounds_file = 'nn_bounds/{}_bounds_L{}_n{}.txt'.format(activation,n_layers,n_neurons)
            ## Se calculan las cotas en caso de no haber
            if True:#not os.path.exists(bounds_file):
                print('\n Capas: ',n_layers,' Neuronas: ',n_neurons,'\n')
                ## Se crea la instancia de la red neuronal
                net = neural_network(n_neurons,n_layers)
                ## Se cargan los parámetros de la red
                net.load_state_dict(torch.load('nn_parameters/{}_model_weights_L{}_n{}.pth'.format(activation,n_layers,n_neurons)))
                ## Se guardan en params los parametos de la red
                params = net.state_dict()
                ## Se filtran los parametros de la red
                filtered_params = filter_params(params,filter_tol)
                ## Filtro personal
                if exact == 'prop':
                    type_model = exact
                else:
                    ## Se asume que exact == 'exact'
                    if activation == 'relu':
                        type_model = 'no_exact'
                    else:
                        type_model = exact
                ## Se calculan las cotas
                bounds,layers_time,net_model,input_var,output_var,all_vars = calculate_bounds(filtered_params,activation,type_model,minutes,add_verif_bounds,tol_distance,image_list,print_output)
                ## Se obtiene informacion con respecto a las cotas
                avg_width,stables = analysis_bounds(bounds)
                ## Se guardan las cotas en el archivo txt correspondiente
                write_bounds(bounds,n_layers,n_neurons,activation,bounds_file)
                print('Tiempos: ',layers_time)
                print('Tamaños: ',avg_width)
                print('Estables: ',sum(stables))
                ## Se guarda la informacion de la instancia
                ## Caso 2 capas ocultas
                if n_layers == 2:
                    new_line = [n_layers,n_neurons,layers_time[0],layers_time[1],'-','-',avg_width[0],avg_width[1],'-','-']
                ## Caso 3 capas ocultas
                elif n_layers == 3:
                    new_line = [n_layers,n_neurons,layers_time[0],layers_time[1],layers_time[2],'-',avg_width[0],avg_width[1],avg_width[2],'-']
                ## Caso 4 capas ocultas
                else:
                    new_line = [n_layers,n_neurons,layers_time[0],layers_time[1],layers_time[2],layers_time[3],avg_width[0],avg_width[1],avg_width[2],avg_width[3]]
                ## Se añade la informacion al df
                df = df._append(pd.Series(new_line), ignore_index=True)
                ## Se intenta escribir en el archivo del df
                written = False
                while not written:
                    try:
                        df.to_excel(data_file, header = False, index = False)
                        written = True
                    except:
                        time.sleep(5)
            ## Caso en que las cotas ya fueron calculadas previamente
            else:
                print('Cotas previamente calculadas')
        
        
