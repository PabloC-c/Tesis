import os
import sys
import time
import torch
import torch.nn as nn
import pandas as pd
from pyscipopt import Model,quicksum
from collections import OrderedDict
from functions import *

if len(sys.argv) == 5:
  ## Activacion, capas, neuronas, tolerancia del filtro
  _, activation, n_layers, n_neurons, filter_tol = sys.argv
  activation_list = [activation]
  layer_list = [int(n_layers)]
  neuron_list = [int(n_neurons)]
  filter_tol = float(filter_tol)
elif len(sys.argv) == 2:
  ## Activacion
  _, activation = sys.argv
  activation_list = [activation]
  layer_list    = [2,3,4]
  neuron_list   = [10,25,50]
  filter_tol = 1e-5
  
else:
  activation_list = ['sigmoid']
  layer_list  = [2,3,4]  
  neuron_list = [10,25,50]
  filter_tol = 1e-5
  
for activation in activation_list:
    if not os.path.exists('datos_{}.xlsx'.format(activation)):
        df = pd.read_excel('datos_{}.xlsx'.format(activation))
    else:
        df = pd.DataFrame()
    data = OrderedDict()
    for n_neurons in neuron_list:
        for n_layers in layer_list:
            if not os.path.exists('nn_bounds/{}_bounds_L{}_n{}.pth'.format(activation,n_layers,n_neurons)):
                print('\n Capas: ',n_layers,' Neuronas: ',n_neurons,'\n')
                ## Crear la instancia de la red neuronal
                net = neural_network(n_neurons,n_layers)
                ## Cargar los parámetros de la red
                net.load_state_dict(torch.load('nn_parameters/{}_model_weights_L{}_n{}.pth'.format(activation,n_layers,n_neurons)))
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
                df = df._append(pd.Series(new_line), ignore_index=True)
                written = False
                while not written:
                    try:
                        df.to_excel('datos_{}.xlsx'.format(activation), header = False, index = False)
                        written = True
                    except:
                        time.sleep(5)
            else:
                print('Cotas previamente calculadas')
        
        
