##### En este archivo se encuentran las funciones para training.py y optimizacion.py

#### Librerias

import torch
import torch.nn as nn
import torch
import torch.nn as nn
from pyscipopt import Model,quicksum
from collections import OrderedDict
import time

#### Funciones

### Funcion que crea una clase de NN's que aprende sobre los datos de Mnist, contiene 2 capas ocultas, con funciones de activacion relus.
### El parametro k que recibe la funcion corresponde a la cantidad de neuronas por capa que tendrá la red

class Relu_net(nn.Module):
    def __init__(self,n_neurons,n_layers):
        super(Relu_net, self).__init__()
        self.n_layers = n_layers
        self.fc_input  = nn.Linear(in_features = 784, out_features = n_neurons)
        self.fc_hidden = nn.ModuleList(
            [nn.Linear(in_features = n_neurons, out_features = n_neurons) for i in range(n_layers)])
        self.fc_output = nn.Linear(in_features = n_neurons, out_features = 10) 

    def forward(self, x):
        x = x.view(-1, 784) # Aplanar la entrada
        x = torch.relu(self.fc_input(x))
        for i in range(self.n_layers):
            x = torch.relu(self.fc_hidden[i](x))
        x = self.fc_output(x)
        return x

###
###

def get_w_b_names(l,n_layers):
    if l == 0:
        weight = 'fc_input.weight'
        bias   = 'fc_input.bias'
    elif l == n_layers-1:
        weight = 'fc_output.weight'
        bias   = 'fc_output.bias'
    else:
        weight = 'fc_hidden.{}.weight'.format(l-1)
        bias   = 'fc_hidden.{}.bias'.format(l-1)
    return weight,bias
    


### Funcion que genera el problema de optimziacion asociado a a evaluacion de una neurona de una red
### El parametro params contiene los pesos y sesgos de la red, bounds contiene las cotas conocidas de la red, l indica la capa de la red, i es la neurona y sense indica el sentido del problema de optimizacion

def create_neuron_model(params,bounds,l,i,sense):
    ## Se genera el modelo de optimizacion
    neuron_model = Model()
    ## Se crean las variables de input del modelo
    n_layers = int(len(params)/2)
    n_input = 784
    inpt = [neuron_model.addVar(lb = bounds[0][k][0], ub = bounds[0][k][1], name = 'h{},{}'.format(0,k)) for k in range(n_input)]
    ## Se recorren las capas hasta la l-1
    for j in range(l):
        ## Parametros capa j
        weight,bias = get_w_b_names(j,n_layers)
        W,b = params[weight],params[bias]
        ## Neuronas capa anterior
        n_input = W.size()[1]
        ## Neuronas de capa j
        n_neurons = W.size()[0]
        ## Arreglo para variables h
        aux_h = []
        ## Se recorren las neuronas de la capa
        for q in range(n_neurons):
            ## Variable de la evaluacion del input en la funcion lineal
            z = neuron_model.addVar(lb = None, ub = None,vtype = 'C', name = 'z{},{}'.format(j,q))
            ## Variable de la parte positiva de la evaluacion
            h = neuron_model.addVar(lb = 0, ub = bounds[j+1][q][1], vtype = 'C', name = 'h{},{}'.format(j,q))
            aux_h.append(h)
            ## Variable de la parte negativa de la evaluacion
            not_h = neuron_model.addVar(lb = 0, ub = bounds[j+1][q][0], vtype = 'C', name = 'not_h{},{}'.format(j,q))
            ## Variable binaria que indica la activacion de la neurona
            u = neuron_model.addVar(vtype = 'B', name = 'u{},{}'.format(j+1,q))
            ## Restriccion de evaluacion con la funcion lineal
            neuron_model.addCons(quicksum(float(W[q,k])*inpt[k] for k in range(n_input)) + float(b[q]) == z, name = 'eval{},{}'.format(j,q))
            ## Restriccion de igualdad de variables
            neuron_model.addCons(z == h - not_h, name = 'vequal{},{}'.format(j,q))
            ## Restricciones big-M
            neuron_model.addCons(h <= bounds[j+1][q][1]*u, name = 'active{},{}'.format(j,q))
            neuron_model.addCons(not_h <= bounds[j+1][q][0]*(1-u), name = 'not_active{},{}'.format(j,q))
        ## Se actualiza el input de la siguiente capa
        inpt = aux_h
        n_input = n_neurons
    ## Parametros de la capa l   
    weight,bias = get_w_b_names(l,n_layers)
    W,b = params[weight],params[bias]
    ## Se genera la funcion objetivo
    neuron_model.setObjective(quicksum(float(W[i,k])*inpt[k] for k in range(n_input)) + float(b[i]), sense)
    ## Se entrega el modelo
    return neuron_model

### Funcion que optimiza el problema de optimizacion asociado a la evaluacion de una neurona de la red
### El parametro neuron_model es el modelo de la neurona, sense es el sentido del problema de optimziacion, tol es la holgura que se añade a las cotas duales

def solve_neuron_model(neuron_model,sense,tol = 1e-05, minutes = 10):
    #neuron_model.setRealParam('limits/time', minutes*60.0)
    ## Se resuelve el problema
    t0 = time.time()
    neuron_model.optimize()
    dt = time.time() - t0
    ## Caso de solucion optima
    if neuron_model.getStatus() == 'optimal':
        ## Se entrega el valor objetivo optimo
        obj_val = neuron_model.getObjVal()
        sol = [True,obj_val]
    ## Caso contrario
    else:
        ## Se entrega la cota dual
        dual_val = neuron_model.getDualbound()
        sol = [False,dual_val]
    ## Para el caso de minimizacion
    if sense == 'minimize':
        ## Se entrega el valor absoluto del valor encontrado
        sol[1] = -1*sol[1]
    ## Caso en que la solucion es muy pequeña
    if sol[1] < tol:
        ## Se entrega 0
        sol[1] = 0
    ## Caso en que se entrega la cota dual
    if not sol[0]:
        ## Se añade la tolerancia
        sol[1] = sol[1] + tol
    ## Se entrega la solucion
    return sol,dt

### Funcion que calcula las cotas de las neuronas de la red neuronal
### El parametro params contiene los pesos y sesgos de la red

def calculate_bounds(params):
    ## Calcular cantidad de capas
    n_layers = int(len(params)/2)
    ## Crear arreglo para guardar cotas de las capas
    ## Inicia con las cotas del input
    bounds = [[(0,1) for i in range(784)]]
    layer_time = []
    ## Se recorren las capas
    for l in range(n_layers):
        ## Parametros capa l
        weight,bias = get_w_b_names(l,n_layers)
        W = params[weight]
        ## Cantidad de neuronas en la capa l
        n_neurons = W.size()[0]
        ## Arreglo para guardar las cotas de la capa l
        aux = []
        tiempo = 0
        ## Se recorren las neuronas de la capa l
        for i in range(n_neurons):
            ## Se determina el valor maximo de la neurona i
            neuron_model_max = create_neuron_model(params,bounds,l,i,'maximize')
            sol_max,dt1      = solve_neuron_model(neuron_model_max,'maximize')
            ## Se determina el minimo de la neurona i
            neuron_model_min = create_neuron_model(params,bounds,l,i,'minimize')
            sol_min,dt2      = solve_neuron_model(neuron_model_min,'minimize')
            ## Se añaden las cotas de la neurona al arreglo de la capa
            aux.append((sol_min[1],sol_max[1]))
            tiempo += dt1 + dt2
        ## Se anaden las cotas de la capa al arreglo bounds
        bounds.append(aux)
        layer_time.append(tiempo/n_neurons)
    ## Se entregan las cotas
    return bounds,layer_time

### Funcion que filtra los pesos y sesgos de la red neuronal
###

def filter_params(params,tol = 1e-05):
    ## Se genera un diccionario para los parametros filtrados 
    filtered_params = OrderedDict()
    ## Calcular cantidad de capas
    n_layers = int(len(params)/2)
    ## Se recorren las capas de la red
    for l in range(n_layers):
        ## Parametros de la capa
        weight,bias = get_w_b_names(l,n_layers)
        W,b = params[weight].clone(),params[bias].clone()
        ## Neuronas capa anterior
        n_input = W.size()[1]
        ## Neuronas de capa l
        n_neurons = W.size()[0]
        ## Se recorren las filas de los parametros
        for i in range(n_neurons):
            ## Se filtra el sesgo de la neurona
            if -tol < b[i] and b[i] < tol:
                b[i] = 0
            ## Se recorren las columnas de los parametros
            for j in range(n_input):
                if -tol < W[i,j] and W[i,j] < tol:
                    W[i,j] = 0
        ## Se añaden al diccionario los parametros filtrados
        filtered_params[weight] = W   
        filtered_params[bias] = b
    ## Se entregan los paramtros filtrados
    return filtered_params

### 
###

def analysis_bounds(bounds):
    n_layers  = len(bounds)
    avg_width = []
    stables   = []
    for l in range(n_layers-1):
        l_bounds = bounds[l+1]
        aux_width  = 0
        aux_stable = 0
        for i in range(len(l_bounds)):
            lb = l_bounds[i][0]
            ub = l_bounds[i][1]
            aux_width += lb + ub
            if lb < 0 and ub > 0:
                aux_stable += 1
            if lb > 0 and ub < 0:
                aux_stable += 1
        avg_width.append(aux_width/len(l_bounds))
        stables.append(aux_stable)
    return avg_width,stables