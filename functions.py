"""
En este archivo se encuentran las funciones para training.py y optimizacion.py
"""

#### Librerias

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pyscipopt as pso
import pyscipopt.scip as scip
from pyscipopt import Model,quicksum
from collections import OrderedDict
import time


#### Funciones generales ############################################################################################################################################################

###
###

def training(activation,n_layers,n_neurons,trainset,trainloader,testset,testloader):
    net = neural_network(n_neurons,n_layers,activation)
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

###
###

def get_w_b_names(l):
    if True:
        weight = 'fc_hidden.{}.weight'.format(l)
        bias   = 'fc_hidden.{}.bias'.format(l)
    return weight,bias


#### Funciones de NN's ############################################################################################################################################################

### 
###

class neural_network(nn.Module):
    def __init__(self,n_neurons,n_layers,activation = 'relu',n_input = 784,n_output = 10):
        super(neural_network, self).__init__()
        self.activation = activation
        self.n_layers = n_layers
        self.fc_hidden = nn.ModuleList(
            [nn.Linear(in_features  = n_neurons if i!= 0 else n_input,
                       out_features = n_neurons if i!= (n_layers-1) else n_output) for i in range(n_layers)])
        if activation == 'softplus': 
            self.softplus = nn.Softplus()

    def forward(self, x):
        x = x.view(-1, 784) # Aplanar la entrada
        for i in range(self.n_layers):
            if self.activation == 'relu':
                x = torch.relu(self.fc_hidden[i](x))
            elif self.activation == 'softplus':
                x = self.softplus(self.fc_hidden[i](x))
        #x = nn.functional.softmax(x, dim=1)  # Aplicar Softmax al output    
        return x

### Funcion que inicializa el modelo de la red, solo genera las variables de input
###

def initialize_neuron_model(bounds,n_input = 784):
    neuron_model = Model()
    inpt = [neuron_model.addVar(lb = bounds[-1][k][0], ub = bounds[-1][k][1], name = 'h{},{}'.format(-1,k)) for k in range(n_input)]
    all_vars = {}
    for i in range(n_input):
        all_vars['h{},{}'.format(-1,i)] = inpt[i]
    return neuron_model,inpt,all_vars

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

def update_neuron_model(neuron_model,inpt,all_vars,params,bounds,l,activation = 'relu'):
    n_input = len(inpt)
    ## Parametros de la capa l   
    weight,bias = get_w_b_names(l)
    W,b = params[weight],params[bias]
    ## Neuronas capa l
    n_neurons = W.size()[0]
    ## Arreglo auxiliar para guardar el input de la siguiente capa
    aux_input = []
    for i in range(n_neurons):
        if activation == 'relu':
            ## Variable de la evaluacion del input en la funcion lineal
            z = neuron_model.addVar(lb = None, ub = None,vtype = 'C', name = 'z{},{}'.format(l,i))
            all_vars['z{},{}'.format(l,i)] = z
            ## Variable de la parte positiva de la evaluacion
            h = neuron_model.addVar(lb = 0, ub = max(0,bounds[l][i][1]), vtype = 'C', name = 'h{},{}'.format(l,i))
            all_vars['h{},{}'.format(l,i)] = h
            ## Se guarda la variable h, para el input de la siguiente capa
            aux_input.append(h)
            ## Variable de la parte negativa de la evaluacion
            not_h = neuron_model.addVar(lb = 0, ub = max(0,bounds[l][i][0]), vtype = 'C', name = 'not_h{},{}'.format(l,i))
            all_vars['not_h{},{}'.format(l,i)] = not_h
            ## Variable binaria que indica la activacion de la neurona
            u = neuron_model.addVar(vtype = 'B', name = 'u{},{}'.format(l,i))
            all_vars['u{},{}'.format(l,i)] = u
            ## Restriccion de evaluacion con la funcion lineal
            neuron_model.addCons(quicksum(float(W[i,k])*inpt[k] for k in range(n_input)) + float(b[i]) == z, name = 'eval{},{}'.format(l,i))
            ## Restriccion de igualdad de variables
            neuron_model.addCons(z == h - not_h, name = 'vequal{},{}'.format(l,i))
            ## Restricciones big-M
            neuron_model.addCons(h <= bounds[l][i][1]*u, name = 'active{},{}'.format(l,i))
            neuron_model.addCons(not_h <= bounds[l][i][0]*(1-u), name = 'not_active{},{}'.format(l,i))
        else:
            ## Variable de la evaluacion del input en la funcion lineal
            z = neuron_model.addVar(lb = -bounds[l][i][0], ub = bounds[l][i][1],vtype = 'C', name = 'z{},{}'.format(l,i))
            all_vars['z{},{}'.format(l,i)] = z
            ## Variable de evaluacion en la funcion de activacion
            a = neuron_model.addVar(lb = 0, ub = None,vtype = 'C', name = 'a{},{}'.format(l,i))
            all_vars['a{},{}'.format(l,i)] = a
            ## Se guarda la variable a, para el input de la siguiente capa
            aux_input.append(a)
            ## Restriccion de evaluacion con la funcion lineal
            neuron_model.addCons(quicksum(float(W[i,k])*inpt[k] for k in range(n_input)) + float(b[i]) == z, name = 'eval{},{}'.format(l,i))
            ## Restriccion de evaluacion en la funcion de activacion
            if activation == 'softplus':
                neuron_model.addCons(scip.log(1+scip.exp(z)) == a, name = 'actv{},{}'.format(l,i)) 
        
    return neuron_model,aux_input,all_vars


### Funcion que optimiza el problema de optimizacion asociado a la evaluacion de una neurona de la red
### El parametro neuron_model es el modelo de la neurona, sense es el sentido del problema de optimziacion, tol es la holgura que se añade a las cotas duales

def solve_neuron_model(neuron_model,sense,params,bounds,l,i,tol = 1e-03, minutes = 10, print_output = False, digits = 4):
    neuron_model.setParam('limits/time', int(60*minutes))
    ## Se resuelve el problema
    if print_output:
        neuron_model.redirectOutput()
    t0 = time.time()
    try:
        aux_t = time.time()
        neuron_model.optimize()
        aux_dt = time.time() - aux_t
        model_status = neuron_model.getStatus()
    except:
        aux_dt = time.time() - t0
        model_status = 'problem'
    dt = aux_dt    
    print('\n Status',neuron_model.getStatus(),'\n')
    ## Caso de solucion optima
    if model_status == 'optimal':
        ## Se entrega el valor objetivo optimo
        obj_val = neuron_model.getObjVal()
        sol = [True,obj_val]
    elif model_status in ['infeasible','unbounded','inforunbd','problem']:
        dual_val = calculate_aprox_bound(params,bounds,l,i,sense)
        sol = [False,dual_val]
    ## Caso contrario    
    else:
        ## Se entrega la cota dual
        dual_val = neuron_model.getDualbound()
        sol = [False,dual_val]
    ## Para el caso de minimizacion
    if sense == 'minimize':
        ## Se entrega el del valor encontrado por -1
        sol[1] = -1*sol[1]
    ## Caso en que la solucion es muy pequeña
    if sol[1] >= 0 and sol[1] < tol:
        ## Se entrega 0
        sol[1] = 0
    ## Se añade la tolerancia
    sol[1] = sol[1] + tol
    ## Se entrega la solucion
    sol[1] = np.around(sol[1],digits)
    return sol,dt

###
###

def calculate_aprox_bound(params,bounds,l,i,sense):
    weight,bias = get_w_b_names(l)
    W,b = params[weight],params[bias]
    input_bounds = bounds[l-1]
    aprox_bound  = float(b[i])
    for j in range(len(input_bounds)):
        lb,ub = -input_bounds[j][0],input_bounds[j][1]
        if float(W[i,j]) >= 0:
            if sense == 'maximize':
                aprox_bound += float(W[i,j])*ub
            else:
                aprox_bound += float(W[i,j])*lb
        else:
            if sense == 'maximize':
                aprox_bound += float(W[i,j])*lb
            else:
                aprox_bound += float(W[i,j])*ub
    return aprox_bound

###
###

def calculate_bounds(params,activation = 'relu'):
    ## Calcular cantidad de capas
    n_layers = int(len(params)/2)
    ## Crear arreglo para guardar cotas de las capas
    ## Inicia con las cotas del input
    bounds     = OrderedDict()
    bounds[-1] = [(0,1) for i in range(784)]
    ## Se inicializa el modelo
    neuron_model,inpt,all_vars = initialize_neuron_model(bounds)
    input_var  = inpt
    layers_time = []
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
            print('\n ===== Capa {}, neurona {} ====='.format(l,i))
            sol_max,dt1  = solve_neuron_model(neuron_model,'maximize',params,bounds,l,i)
            neuron_model.freeTransform()
            ## Se determina el minimo de la neurona i
            neuron_model = set_objective_function(neuron_model,inpt,params,bounds,l,i,'minimize')
            sol_min,dt2  = solve_neuron_model(neuron_model,'minimize',params,bounds,l,i)
            neuron_model.freeTransform()
            ## Se añaden las cotas de la neurona al arreglo de la capa
            aux.append((sol_min[1],sol_max[1]))
            tiempo += dt1 + dt2
        ## Se anaden las cotas de la capa al arreglo bounds
        bounds[l] = aux
        layers_time.append(tiempo/n_neurons)
        ## Se actualiza el modelo con las cotas de la capa l
        neuron_model,inpt,all_vars = update_neuron_model(neuron_model,inpt,all_vars,params,bounds,l,activation)
    output_var = inpt
    ## Se entregan las cotas
    return bounds,layers_time,neuron_model,input_var,output_var,all_vars


#### Verificacion ########################################################################################################################################################

###
###

def calculate_variables(net_model,input_value,params,all_vars,activation = 'relu'):
    ## Arreglo solucion
    solution = net_model.createSol()
    ## Calcular la cantidad de capas
    n_layers = int(len(params)/2)
    ## Cantidad de neuronas iniciales
    n_input = len(input_value)
    ## Variable para guardar el input de la capa
    input_layer = input_value
    aux = []
    ## Se recorren las capas
    for l in range(-1,n_layers):
        ## Caso capa input
        if l == -1:
            n_neurons = n_input
            for i in range(n_neurons):
                net_model.setSolVal(solution, all_vars['h{},{}'.format(-1,i)], input_value[i])
            aux = input_layer
        ## Resto de capas
        else:
            ## Parametros de la capa l
            weight,bias = get_w_b_names(l)
            W = params[weight]
            b = params[bias]
            n_neurons = W.size()[0]
            for i in range(n_neurons):
                ## Variable z de evaluacion en la funcion lineal
                z = sum(float(W[i,j])*input_layer[j] for j in range(n_input)) + float(b[i])
                net_model.setSolVal(solution, all_vars['z{},{}'.format(l,i)], z)
                if activation == 'relu': 
                    h = 0
                    not_h = 0
                    if z == 0:
                        u = 0
                    elif z > 0:
                        h = z
                        u = 1
                    else:
                        not_h = -z
                        u = 0
                    net_model.setSolVal(solution, all_vars['h{},{}'.format(l,i)], h)
                    net_model.setSolVal(solution, all_vars['not_h{},{}'.format(l,i)], not_h)
                    net_model.setSolVal(solution, all_vars['u{},{}'.format(l,i)], u)
                    aux.append(h)
                elif activation == 'softplus':
                    a = np.log(1+np.exp(z))
                    net_model.setSolVal(solution, all_vars['a{},{}'.format(l,i)], a)
                    aux.append(a)
        input_layer = aux
        n_input = n_neurons
        aux = []
    return solution

###
###

def create_verification_model(net_model,net_input_var,net_output_var,input_value,real_output,output_value,output_target,params,bounds,tol_distance = 0.5, apply_softmax = True):
    ## Cantidad de neuronas del input
    n_input = len(net_input_var)
    ## Cantidad de neuronas del output
    n_output = len(net_output_var)
    ## Calcular cantidad de capas
    n_layers = int(len(params)/2)
    ## Restriccion de proximidad
    for i in range(n_input):
        net_model.addCons( net_input_var[i] - input_value[i] <= tol_distance, name = 'inpt_dist{},1'.format(i))
        net_model.addCons( net_input_var[i] - input_value[i] >= -tol_distance, name = 'inpt_dist{},2'.format(i))
    if apply_softmax:
        ## Se crean las nuevas variables para aplicar softmax
        aux_list = []
        for i in range(n_output):
            soft_output = net_model.addVar(vtype = 'C', name = 'output{}'.format(i))
            net_model.addCons(soft_output == scip.exp(net_output_var[i])/quicksum(scip.exp(net_output_var[k]) for k in range(n_output)),
                              name = 'soft{}'.format(i))
            aux_list.append(soft_output)
        net_output_var = aux_list
    ## Se genera la restriccion correspondiente a la funcion objetivo
    objective = net_model.addVar(lb = None, ub = None,vtype = 'C', name = 'obj_val')
    net_model.addCons(net_output_var[output_target] - output_value
                      >= objective, name = 'obj_cons')
    net_model.setObjective(objective, 'maximize')
    return net_model
