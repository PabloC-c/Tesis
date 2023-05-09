import torch
import torch.nn as nn
from pyscipopt import Model,quicksum
from collections import OrderedDict
from functions import *
                
n_neurons = 100
n_layers  = 0
# Crear la instancia de la red neuronal
net = Relu_net(n_neurons,n_layers)

# Cargar los parámetros de la red
net.load_state_dict(torch.load("model_weights_L{}_n{}.pth".format(n_layers, n_neurons)))

# Guardar los parametos de la red
params = net.state_dict()

filtered_params = filter_params(params)

bounds,layer_times = calculate_bounds(filtered_params)

avg_width,stables = analysis_bounds(bounds)