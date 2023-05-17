import torch
import torch.nn as nn
from pyscipopt import Model,quicksum
from collections import OrderedDict
from functions import *
                
neuron_list = [10,25,50]
layer_list  = [2,3]
activation= 'softplus'
data = OrderedDict()

for n_neurons in neuron_list:
    for n_layers in layer_list if n_neurons != 50 else [2]:
        print('\n Capas: ',n_layers,' Neuronas: ',n_neurons,'\n')
        ##
        if activation == 'relu':
            ## Crear la instancia de la red neuronal
            net = relu_net(n_neurons,n_layers)
    
            ## Cargar los parámetros de la red
            net.load_state_dict(torch.load(activation+"_"+"model_weights_L{}_n{}.pth".format(n_layers, n_neurons)))
            
            ## Guardar los parametos de la red
            params = net.state_dict()
                        
            filtered_params = filter_params(params)
            
            bounds,layer_times = calculate_bounds(filtered_params)
            
            avg_width,stables = analysis_bounds(bounds)
                
        else:
            ## Crear la instancia de la red neuronal
            net = nl_net(activation,n_neurons,n_layers)
    
            ## Cargar los parámetros de la red
            net.load_state_dict(torch.load(activation+"_"+"model_weights_L{}_n{}.pth".format(n_layers, n_neurons)))
        
            ## Guardar los parametos de la red
            params = net.state_dict()
            
            filtered_params = filter_params(params)    
            bounds,layer_times = calculate_bounds_no_linear_nl(filtered_params,activation)
        
            avg_width,stables = analysis_bounds(bounds)
            
        data[(n_layers,n_neurons)] = [bounds,layer_times,avg_width,stables]
