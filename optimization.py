import torch
import torch.nn as nn
import pandas as pd
from pyscipopt import Model,quicksum
from collections import OrderedDict
from functions import *
                
neuron_list = [10]
layer_list  = [3]
activation= 'softplus'
filter_tol = 1e-3
print('Tolerancia del filtro: ',filter_tol)
data = OrderedDict()
df = pd.DataFrame()

for n_neurons in neuron_list:
    for n_layers in layer_list if n_neurons != 50 else [2]:
        print('\n Capas: ',n_layers,' Neuronas: ',n_neurons,'\n')
        ## Crear la instancia de la red neuronal
        net = neural_network(n_neurons,n_layers)
        ## Cargar los parámetros de la red
        net.load_state_dict(torch.load(activation+"_"+"model_weights_L{}_n{}.pth".format(n_layers, n_neurons)))
        ## Guardar los parametos de la red
        params = net.state_dict()
        filtered_params = filter_params(params,filter_tol)
        bounds,layers_time,net_model,input_var,output_var,all_vars = calculate_bounds(filtered_params,activation)
        avg_width,stables = analysis_bounds(bounds)
        data[(n_layers,n_neurons)] = [bounds,layers_time,avg_width,stables]
        print('Tiempos: ',layers_time)
        print('Tamaños: ',avg_width)
        print('Estables: ',sum(stables))
        if n_layers == 2:
            nueva_fila = [n_neurons,0.005,n_layers,layers_time[0],layers_time[1],'-',avg_width[0],avg_width[1],'-',sum(stables)]
        else:
            nueva_fila = [n_neurons,0.005,n_layers,layers_time[0],layers_time[1],layers_time[2],avg_width[0],avg_width[1],avg_width[2],sum(stables)]
        df = df.append(pd.Series(nueva_fila), ignore_index=True)
df.to_excel('datos_{}.xlsx'.format(activation), header = False, index = False)
