"""
En este archivo se encuentran las funciones para training.py y optimizacion.py
"""

#### Librerias

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pyscipopt as pso
import pyscipopt.scip as scip
from pyscipopt import Model,quicksum
from collections import OrderedDict
import time


#### Funciones generales ############################################################################################################################################################

###
###

def training(activation,n_layers,n_neurons,trainset,trainloader,testset,testloader):
    if activation == 'relu':
        net = relu_net(n_neurons,n_layers)
    else:
        net = nl_net(activation,n_neurons,n_layers)
    # Definir la función de pérdida y el optimizador
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, weight_decay=0.001)

    # Definir el factor de regularización L1
    l1_lambda = 0.005

    # Entrenar la red neuronal
    for epoch in range(50):  # Correr el conjunto de datos de entrenamiento 10 veces
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # Obtener los datos de entrada y las etiquetas
            inputs, labels = data

            # Reestablecer los gradientes a cero
            optimizer.zero_grad()

            # Propagar hacia adelante, calcular la pérdida y retropropagar el error
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            
            # Calcular la penalización L1 y agregarla a la pérdida
            l1_reg = 0
            for param in net.parameters():
                l1_reg += torch.sum(torch.abs(param))
            loss += l1_lambda * l1_reg
            
            loss.backward()
            optimizer.step()

            # Imprimir estadísticas de entrenamiento
            running_loss += loss.item()
            if i % 100 == 99:    # Imprimir cada 100 mini-lotes procesados
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

    print('Entrenamiento finalizado')

    # Evaluar la precisión de la red neuronal en el conjunto de datos de prueba
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Precisión en el conjunto de prueba: %d %%' % (100 * correct / total))
    return net

#### Funciones caso relu ############################################################################################################################################################

### Funcion que crea una clase de NN's que aprende sobre los datos de Mnist, contiene 2 capas ocultas, con funciones de activacion relus.
### El parametro k que recibe la funcion corresponde a la cantidad de neuronas por capa que tendrá la red

class relu_net(nn.Module):
    def __init__(self,n_neurons,n_layers,n_input = 784,n_output = 10):
        super(relu_net, self).__init__()
        self.n_layers = n_layers
        self.fc_hidden = nn.ModuleList(
            [nn.Linear(in_features  = n_neurons if i!= 0 else n_input,
                       out_features = n_neurons if i!= (n_layers-1) else n_output) for i in range(n_layers)])

    def forward(self, x):
        x = x.view(-1, 784) # Aplanar la entrada
        for i in range(self.n_layers):
            x = torch.relu(self.fc_hidden[i](x))
        return x

###
###

def get_w_b_names(l):
    if True:
        weight = 'fc_hidden.{}.weight'.format(l)
        bias   = 'fc_hidden.{}.bias'.format(l)
    return weight,bias

### Funcion que genera el problema de optimziacion asociado a a evaluacion de una neurona de una red
### El parametro params contiene los pesos y sesgos de la red, bounds contiene las cotas conocidas de la red, l indica la capa de la red, i es la neurona y sense indica el sentido del problema de optimizacion

def create_neuron_model(params,bounds,l,i,sense):
    ## Se genera el modelo de optimizacion
    neuron_model = Model()
    ## Se crean las variables de input del modelo
    n_input = 784
    inpt = [neuron_model.addVar(lb = bounds[-1][k][0], ub = bounds[-1][k][1], name = 'h{},{}'.format(-1,k)) for k in range(n_input)]
    ## Se recorren las capas hasta la l-1
    for j in range(l):
        ## Parametros capa j
        weight,bias = get_w_b_names(j)
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
            h = neuron_model.addVar(lb = 0, ub = bounds[j][q][1], vtype = 'C', name = 'h{},{}'.format(j,q))
            aux_h.append(h)
            ## Variable de la parte negativa de la evaluacion
            not_h = neuron_model.addVar(lb = 0, ub = bounds[j][q][0], vtype = 'C', name = 'not_h{},{}'.format(j,q))
            ## Variable binaria que indica la activacion de la neurona
            u = neuron_model.addVar(vtype = 'B', name = 'u{},{}'.format(j,q))
            ## Restriccion de evaluacion con la funcion lineal
            neuron_model.addCons(quicksum(float(W[q,k])*inpt[k] for k in range(n_input)) + float(b[q]) == z, name = 'eval{},{}'.format(j,q))
            ## Restriccion de igualdad de variables
            neuron_model.addCons(z == h - not_h, name = 'vequal{},{}'.format(j,q))
            ## Restricciones big-M
            neuron_model.addCons(h <= bounds[j][q][1]*u, name = 'active{},{}'.format(j,q))
            neuron_model.addCons(not_h <= bounds[j][q][0]*(1-u), name = 'not_active{},{}'.format(j,q))
        ## Se actualiza el input de la siguiente capa
        inpt = aux_h
        n_input = n_neurons
    ## Parametros de la capa l   
    weight,bias = get_w_b_names(l)
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
    if sol[1] >= 0 and sol[1] < tol:
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
    bounds     = OrderedDict()
    bounds[-1] = [(0,1) for i in range(784)]
    layer_time = []
    ## Se recorren las capas
    for l in range(n_layers):
        ## Parametros capa l
        weight,bias = get_w_b_names(l)
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
        bounds[l] = aux
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
        weight,bias = get_w_b_names(l)
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
        l_bounds = bounds[l]
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

#### Funciones caso relu, implementeacion modelo unico ########################################################################################################################################################

### Funcion que inicializa el modelo de la red, solo genera las variables de input
###

def initialize_neuron_model(bounds,n_input = 784):
    neuron_model = Model()
    inpt = [neuron_model.addVar(lb = bounds[-1][k][0], ub = bounds[-1][k][1], name = 'h{},{}'.format(-1,k)) for k in range(n_input)]
    return neuron_model,inpt

### Funcion que dada una neurona i en una capa l, fija la funcion objetivo correspondiente 
###

def set_objective_function(neuron_model,inpt,params,bounds,l,i,sense):
    ## Tamaño del input de la neurona
    n_input = len(inpt)
    ## Parametros de la capa l   
    weight,bias = get_w_b_names(l)
    W,b = params[weight],params[bias]
    ## Se genera la funcion objetivo
    neuron_model.setObjective(quicksum(float(W[i,k])*inpt[k] for k in range(n_input)) + float(b[i]), sense)
    return neuron_model

### Funcion que dada una capa l y las cotas de sus neuronas, genera las restricciones correspondientes
###

def update_neuron_model(neuron_model,inpt,params,bounds,l):
    n_input = len(inpt)
    ## Parametros de la capa l   
    weight,bias = get_w_b_names(l)
    W,b = params[weight],params[bias]
    ## Neuronas capa l
    n_neurons = W.size()[0]
    ## Arreglo auxiliar para guardar el input de la siguiente capa
    aux_input = []
    for i in range(n_neurons):
        ## Variable de la evaluacion del input en la funcion lineal
        z = neuron_model.addVar(lb = None, ub = None,vtype = 'C', name = 'z{},{}'.format(l,i))
        ## Variable de la parte positiva de la evaluacion
        h = neuron_model.addVar(lb = 0, ub = bounds[l][i][1], vtype = 'C', name = 'h{},{}'.format(l,i))
        ## Se guarda la variable h, para el input de la siguiente capa
        aux_input.append(h)
        ## Variable de la parte negativa de la evaluacion
        not_h = neuron_model.addVar(lb = 0, ub = bounds[l][i][0], vtype = 'C', name = 'not_h{},{}'.format(l,i))
        ## Variable binaria que indica la activacion de la neurona
        u = neuron_model.addVar(vtype = 'B', name = 'u{},{}'.format(l,i))
        ## Restriccion de evaluacion con la funcion lineal
        neuron_model.addCons(quicksum(float(W[i,k])*inpt[k] for k in range(n_input)) + float(b[i]) == z, name = 'eval{},{}'.format(l,i))
        ## Restriccion de igualdad de variables
        neuron_model.addCons(z == h - not_h, name = 'vequal{},{}'.format(l,i))
        ## Restricciones big-M
        neuron_model.addCons(h <= bounds[l][i][1]*u, name = 'active{},{}'.format(l,i))
        neuron_model.addCons(not_h <= bounds[l][i][0]*(1-u), name = 'not_active{},{}'.format(l,i))
    return neuron_model,aux_input


###
###

def calculate_bounds2(params):
    ## Calcular cantidad de capas
    n_layers = int(len(params)/2)
    ## Crear arreglo para guardar cotas de las capas
    ## Inicia con las cotas del input
    bounds     = OrderedDict()
    bounds[-1] = [(0,1) for i in range(784)]
    ## Se inicializa el modelo
    neuron_model,inpt = initialize_neuron_model(bounds)
    layer_time = []
    ## Se recorren las capas
    for l in range(n_layers):
        ## Parametros capa l
        weight,bias = get_w_b_names(l)
        W = params[weight]
        ## Cantidad de neuronas en la capa l
        n_neurons = W.size()[0]
        ## Arreglo para guardar las cotas de la capa l
        aux = []
        tiempo = 0
        ## Se recorren las neuronas de la capa l
        for i in range(n_neurons):
            ## Se determina el valor maximo de la neurona i
            neuron_model = set_objective_function(neuron_model,inpt,params,bounds,l,i,'maximize')
            sol_max,dt1  = solve_neuron_model(neuron_model,'maximize')
            ## Se determina el minimo de la neurona i
            neuron_model = set_objective_function(neuron_model,inpt,params,bounds,l,i,'minimize')
            sol_min,dt2  = solve_neuron_model(neuron_model,'minimize')
            ## Se añaden las cotas de la neurona al arreglo de la capa
            aux.append((sol_min[1],sol_max[1]))
            tiempo += dt1 + dt2
        ## Se anaden las cotas de la capa al arreglo bounds
        bounds[l] = aux
        layer_time.append(tiempo/n_neurons)
        ## Se actualiza el modelo con las cotas de la capa l
        neuron_model,inpt = update_neuron_model(neuron_model,inpt,params,bounds,l)
    ## Se entregan las cotas
    return bounds,layer_time

#### Funciones caso no lineal ########################################################################################################################################################

###
###

class nl_net(nn.Module):
    def __init__(self,activation,n_neurons,n_layers,n_input = 784,n_output = 10):
        super(nl_net, self).__init__()
        self.n_layers = n_layers
        self.activation = activation
        self.fc_hidden = nn.ModuleList(
            [nn.Linear(in_features  = n_neurons if i!= 0 else n_input,
                       out_features = n_neurons if i!= (n_layers-1) else n_output) for i in range(n_layers)])
        if activation == 'softplus':
            self.softplus = nn.Softplus()

    def forward(self, x):
        x = x.view(-1, 784) # Aplanar la entrada
        if self.activation == 'softplus':
            for i in range(self.n_layers):
                x = self.softplus(self.fc_hidden[i](x))
        return x

###
###

def create_neuron_model_nl(params,bounds,l,i,sense,activation = 'softplus'):
    ## Se genera el modelo de optimizacion
    neuron_model = Model()
    ## Se crean las variables de input del modelo
    n_input = 784
    inpt = [neuron_model.addVar(lb = bounds[-1][k][0], ub = bounds[-1][k][1], name = 'h{},{}'.format(-1,k)) for k in range(n_input)]
    ## Se recorren las capas hasta la l-1
    for j in range(l+1):
        ## Parametros capa j
        weight,bias = get_w_b_names(j)
        W,b = params[weight],params[bias]
        ## Neuronas capa anterior
        n_input = W.size()[1]
        ## Neuronas de capa j
        n_neurons = W.size()[0]
        ## Arreglo para guardar el input de la ultima capa
        aux_inpt = []
        ## Se recorren las neuronas de la capa
        for q in range(n_neurons) if j < l else [i]:
            ## Variable de la evaluacion del input en la funcion lineal
            z = neuron_model.addVar(lb = None, ub = None,vtype = 'C', name = 'z{},{}'.format(j,q))
            ## Variable de evaluacion en la funcion de activacion
            if j < l:
                a = neuron_model.addVar(lb = -bounds[j][q][0], ub = bounds[j][q][1],vtype = 'C', name = 'z{},{}'.format(j,q))
            else:
                a = neuron_model.addVar(lb = None, ub = None,vtype = 'C', name = 'a{},{}'.format(j,q))
            aux_inpt.append(a)
            ## Restriccion de evaluacion con la funcion lineal
            neuron_model.addCons(quicksum(float(W[q,k])*inpt[k] for k in range(n_input)) + float(b[q]) == z, name = 'eval{},{}'.format(j,q))
            ## Restriccion de evaluacion en la funcion de activacion
            if activation == 'softplus':
                #nl_contraint = NonlinearConstraint(softplus_constraint, lb = 0, ub = 0, name = 'actv{},{}'.format(j,q), vars=[z, a])
                neuron_model.addCons(scip.log(1+scip.exp(z)) - a  == 0, name = 'actv{},{}'.format(j,q))
        ## Se actualiza el input de la siguiente capa
        inpt = aux_inpt
    ## Se genera la funcion objetivo
    neuron_model.setObjective(a, sense)
    ## Se entrega el modelo
    return neuron_model

###
###

def solve_neuron_model_nl(neuron_model,sense,tol = 1e-05, minutes = 10):
    neuron_model.setParam('limits/time', int(60*minutes))
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
    if sol[1] >= 0 and sol[1] < tol:
        ## Se entrega 0
        sol[1] = 0
    ## Caso en que se entrega la cota dual
    if not sol[0]:
        ## Se añade la tolerancia
        sol[1] = sol[1] + tol
    ## Se entrega la solucion
    return sol,dt

###
###

def calculate_bounds_no_linear_nl(params,activation):
    ## Calcular cantidad de capas
    n_layers = int(len(params)/2)
    ## Crear arreglo para guardar cotas de las capas
    ## Inicia con las cotas del input
    bounds     = OrderedDict()
    bounds[-1] = [(0,1) for i in range(784)]
    layer_time = []
    ## Se recorren las capas
    for l in range(n_layers):
        ## Parametros capa l
        weight,bias = get_w_b_names(l)
        W = params[weight]
        ## Cantidad de neuronas en la capa l
        n_neurons = W.size()[0]
        ## Arreglo para guardar las cotas de la capa l
        aux = []
        tiempo = 0
        ## Se recorren las neuronas de la capa l
        for i in range(n_neurons):
            ## Se determina el valor maximo de la neurona i
            neuron_model_max = create_neuron_model_nl(params,bounds,l,i,'maximize')
            sol_max,dt1      = solve_neuron_model_nl(neuron_model_max,'maximize')
            ## Se determina el minimo de la neurona i
            neuron_model_min = create_neuron_model_nl(params,bounds,l,i,'minimize')
            sol_min,dt2      = solve_neuron_model_nl(neuron_model_min,'minimize')
            ## Se añaden las cotas de la neurona al arreglo de la capa
            aux.append((sol_min[1],sol_max[1]))
            tiempo += dt1 + dt2
        ## Se anaden las cotas de la capa al arreglo bounds
        bounds[l] = aux
        layer_time.append(tiempo/n_neurons)
    ## Se entregan las cotas
    return bounds,layer_time
