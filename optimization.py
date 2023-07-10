import time
import torch
import torch.nn as nn
import pandas as pd
from pyscipopt import Model,quicksum
from collections import OrderedDict
from functions import *

activation_list = ['softplus']
neuron_list = [10,25,50]
layer_list  = [2,3,4]
filter_tol = 1e-5
print('Tolerancia del filtro: ',filter_tol)
data = OrderedDict()
df = pd.DataFrame()

for activation in activation_list:
    dfs_list = []
    for n_neurons in neuron_list:
        for n_layers in layer_list:
            print('\n Capas: ',n_layers,' Neuronas: ',n_neurons,'\n')
            ## Crear la instancia de la red neuronal
            net = neural_network(n_neurons,n_layers)
            ## Cargar los parámetros de la red
            net.load_state_dict(torch.load('nn_parameters/'+activation+"_"+"model_weights_L{}_n{}.pth".format(n_layers, n_neurons)))
            ## Guardar los parametos de la red
            params = net.state_dict()
            filtered_params = filter_params(params,filter_tol)
            bounds,layers_time,net_model,input_var,output_var,all_vars = calculate_bounds(filtered_params,activation)
            avg_width,stables = analysis_bounds(bounds)
            data[(n_layers,n_neurons)] = [bounds,layers_time,avg_width,stables]
            write_bounds(bounds,n_layers,n_neurons,activation)
            print('Tiempos: ',layers_time)
            print('Tamaños: ',avg_width)
            print('Estables: ',sum(stables))
            if n_layers == 2:
                new_line = [n_layers,n_neurons,layers_time[0],layers_time[1],'-','-',avg_width[0],avg_width[1],'-','-']
            elif n_layers == 3:
                new_line = [n_layers,n_neurons,layers_time[0],layers_time[1],layers_time[2],'-',avg_width[0],avg_width[1],avg_width[2],'-']
            else:
                new_line = [n_layers,n_neurons,layers_time[0],layers_time[1],layers_time[2],layers_time[3],avg_width[0],avg_width[1],avg_width[2],avg_width[3]]
            dfs_list.append(pd.DataFrame([new_line]))
            df = pd.concat(dfs_list,ignore_index=True)
            written = False
            while not written:
                try:
                    df.to_excel('datos_{}.xlsx'.format(activation), header = False, index = False)
                    written = True
                except:
                    time.sleep(5)
