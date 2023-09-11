"""
En este archivo se encuentran las funciones para training.py y optimizacion.py
"""

#### Librerias

import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pyscipopt.scip as scip
from pyscipopt import Model,quicksum
from collections import OrderedDict
import time


#### Funciones generales ############################################################################################################################################################

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

###
###

def calculate_verif_file_name(exact,activation,real_output,target_output,root_node_only):
    if root_node_only:
         file_name = 'root_node/{}/datos_verificacion_{}_{}como{}.xlsx'.format(exact,activation,real_output,target_output)
    else:
        file_name = 'verif_results/{}/datos_verificacion_{}_{}como{}.xlsx'.format(exact,activation,real_output,target_output)
    return file_name

###
###

def calculate_bounds_file_name(type_bounds,activation,n_layers,n_neurons):
    if type_bounds == 'prop':
        bounds_file = 'nn_bounds/{}_prop_bounds_L{}_n{}.txt'.format(activation,n_layers,n_neurons)
    else:
        bounds_file = 'nn_bounds/{}_bounds_L{}_n{}.txt'.format(activation,n_layers,n_neurons)
    return bounds_file 

###
###

def write_bounds(bounds,n_layers,n_neurons,activation,file_name):
    with open(file_name,'w') as bounds_file:
        for layer,layer_bounds in bounds.items():
            for lb,ub in layer_bounds:
                line = f"{lb} {ub}"
                bounds_file.write(line + '\n')
            bounds_file.write('\n')
    bounds_file.close()        
    return True

###
###

def read_bounds(apply_bounds,n_layers,n_neurons,activation,bounds_file):
    bounds = OrderedDict()
    if os.path.exists(bounds_file) and apply_bounds:
        with open(bounds_file, 'r') as bf:
            layer = -1
            value = []
            for line in bf:
                if line.strip() == "":
                    bounds[layer] = value
                    layer += 1
                    value = []
                else:
                    lb,ub = line.strip().split()
                    value.append((float(lb), float(ub)))
                bounds[layer] = value
    return bounds
        

###
###

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

    def forward(self, x):
        x = x.view(-1, 784) # Aplanar la entrada
        for i in range(self.n_layers):
            x = self.fc_hidden[i](x)
            if self.activation == 'relu':
                x = F.relu(x)
            elif self.activation == 'softplus':
                x = F.softplus(x)
            elif self.activation == 'sigmoid':
                x = F.sigmoid(x)
        return x

### Funcion que inicializa el modelo de la red, solo genera las variables de input
###

def initialize_neuron_model(bounds):
    neuron_model = Model()
    if len(bounds) == 0:
        bounds[-1] = [(0,1) for i in range(784)]
    n_input = len(bounds[-1])
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

def update_neuron_model(neuron_model,inpt,all_vars,params,bounds,l,activation = 'relu',exact = 'no_exact', apply_bounds = True):
    k = 4
    n_input = len(inpt)
    ## Parametros de la capa l   
    weight,bias = get_w_b_names(l)
    W,b = params[weight],params[bias]
    ## Neuronas capa l
    n_neurons = W.size()[0]
    ## Arreglo auxiliar para guardar el input de la siguiente capa
    aux_input = []
    for i in range(n_neurons):
        ## Formulacion exacta
        if exact == 'exact':
            ## Variable de la evaluacion del input en la funcion lineal
            if apply_bounds:
                z = neuron_model.addVar(lb = -bounds[l][i][0], ub = bounds[l][i][1],vtype = 'C', name = 'z{},{}'.format(l,i))
            else:
                z = neuron_model.addVar(lb = None, ub = None,vtype = 'C', name = 'z{},{}'.format(l,i))
            all_vars['z{},{}'.format(l,i)] = z
            ## Variable de evaluacion en la funcion de activacion
            a = neuron_model.addVar(lb = None, ub = None,vtype = 'C', name = 'a{},{}'.format(l,i))
            all_vars['a{},{}'.format(l,i)] = a
            ## Se guarda la variable a, para el input de la siguiente capa
            aux_input.append(a)
            ## Restriccion de evaluacion con la funcion lineal
            neuron_model.addCons(quicksum(float(W[i,k])*inpt[k] for k in range(n_input)) + float(b[i]) == z, name = 'eval{},{}'.format(l,i))
            ## Restriccion de evaluacion en la funcion de activacion
            if activation == 'relu':
                neuron_model.addCons((z+abs(z))/2 == a, name = 'actv{},{}'.format(l,i))
            if activation == 'softplus':
                neuron_model.addCons(scip.log(1+scip.exp(z)) == a, name = 'actv{},{}'.format(l,i))
            if activation == 'sigmoid':
                neuron_model.addCons((1/(1+scip.exp(-z))) == a, name = 'actv{},{}'.format(l,i))
        ## Formulacion alternativa/relajacion
        else:
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
                a = neuron_model.addVar(lb = None, ub = None,vtype = 'C', name = 'a{},{}'.format(l,i))
                all_vars['a{},{}'.format(l,i)] = a
                ## Se guarda la variable a, para el input de la siguiente capa
                aux_input.append(a)
                ## Restriccion de evaluacion con la funcion lineal
                neuron_model.addCons(quicksum(float(W[i,k])*inpt[k] for k in range(n_input)) + float(b[i]) == z, name = 'eval{},{}'.format(l,i))
                ## Restriccion de evaluacion en la funcion de activacion
                if activation == 'softplus':
                    neuron_model.addCons(scip.log(1+scip.exp(z)) == a, name = 'actv{},{}'.format(l,i))
                if activation == 'sigmoid':
                    neuron_model.addCons((1/(1+scip.exp(-z))) == a, name = 'actv{},{}'.format(l,i))
                ## Se obtienen las cotas de la neurona 
                bounds_l_i = bounds[l][i]
                ## Se obtienen las tuplas correspondientes a la envoltura
                cv_env,cc_env = get_activation_env_list(activation,bounds_l_i, k)
                ## Se añaden los planos cortantes convexos
                for q in range(len(cv_env)):
                    x0,f_x0,m = cv_env[q]
                    neuron_model.addCons(f_x0+m*(z-x0) <= a, name = 'cv_env{}_{},{}'.format(q,l,i))
                ## Se añaden los planos cortantes concavos
                for q in range(len(cc_env)):
                    x0,f_x0,m = cc_env[q]
                    neuron_model.addCons(f_x0+m*(z-x0) >= a, name = 'cc_env{}_{},{}'.format(q,l,i))
    return neuron_model,aux_input,all_vars


### Funcion que optimiza el problema de optimizacion asociado a la evaluacion de una neurona de la red
### El parametro neuron_model es el modelo de la neurona, sense es el sentido del problema de optimziacion, tol es la holgura que se añade a las cotas duales

def solve_neuron_model(neuron_model,sense,params,bounds,l,i,exact = 'no_exact',minutes = 10,tol = 1e-03,print_output = False,digits = 4):
    ## Se calcula la cota con el metodo de propagacion
    propb = calculate_aprox_bound(params,bounds,l,i,sense)
    if exact in ['exact','no_exact']:
        neuron_model.setParam('limits/time', int(60*minutes))
        ## Se resuelve el problema
        if print_output:
            neuron_model.redirectOutput()
        else:
            neuron_model.hideOutput()
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
    else:
        model_status = 'problem'
    ## Caso de solucion optima
    if model_status == 'optimal':
        ## Se entrega el valor objetivo optimo
        obj_val = neuron_model.getObjVal()
        if (sense == 'maximize' and propb < obj_val) or (sense == 'minimize' and propb > obj_val):
            obj_val = propb
        sol = [True,obj_val]
        aprox_bound = calculate_aprox_bound(params,bounds,l,i,sense)
        print('\t Caso sol optima \n')
        print('\t problema: {}, valor obj : {}, valor prop: {}'.format(sense,obj_val,aprox_bound))
    elif model_status in ['infeasible','unbounded','inforunbd','problem']:
        if exact == 'prop':
            aux_t = time.time()
            dual_val = propb
            dt = time.time() - aux_t
        else:
            try:
                dual_val = neuron_model.getDualbound()
            except:
                dual_val = propb
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

def calculate_aprox_bound(params,bounds,l,i,sense,activation = 'relu',tol = 1e-05):
    weight,bias = get_w_b_names(l)
    W,b = params[weight],params[bias]
    input_bounds = bounds[l-1]
    aprox_bound  = float(b[i])
    for j in range(len(input_bounds)):
        lb,ub = -input_bounds[j][0],input_bounds[j][1]
        if activation == 'relu' and l > 0:
            if lb < 1e-05:
                lb = 0
            if ub < 1e-05:
                ub = 0
        elif activation == 'softplus' and l > 0:
            lb = np.log(1 + np.exp(lb))
            ub = np.log(1 + np.exp(ub))
        elif activation == 'sigmoid'  and l > 0:
            lb = 1/(1+np.exp(-lb))
            ub = 1/(1+np.exp(-ub))
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

def calculate_bounds(params,activation = 'relu',exact = 'no_exact',minutes = 10):
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
            print('\n ===== Capa {}, neurona {} ====='.format(l,i))
            ## Caso solo propagacion
            if exact == 'prop':
                t_aux   = time.time()
                sol_max = calculate_aprox_bound(params,bounds,l,i,'maximize',activation)
                sol_min = calculate_aprox_bound(params,bounds,l,i,'minimize',activation)
                tiempo += (time.time() - t_aux)
                aux.append((-1*sol_min,sol_max))
            ## Caso modelo de optimizacion
            else:
                ## Se determina el valor maximo de la neurona i
                neuron_model = set_objective_function(neuron_model,inpt,params,bounds,l,i,'maximize')
                sol_max,dt1  = solve_neuron_model(neuron_model,'maximize',params,bounds,l,i,exact,minutes)
                neuron_model.freeTransform()
                ## Se determina el minimo de la neurona i
                neuron_model = set_objective_function(neuron_model,inpt,params,bounds,l,i,'minimize')
                sol_min,dt2  = solve_neuron_model(neuron_model,'minimize',params,bounds,l,i,exact,minutes)
                neuron_model.freeTransform()
                ## Se añaden las cotas de la neurona al arreglo de la capa
                aux.append((sol_min[1],sol_max[1]))
                tiempo += dt1 + dt2
        ## Se anaden las cotas de la capa al arreglo bounds
        bounds[l] = aux
        layers_time.append(tiempo/n_neurons)
        ## Se actualiza el modelo con las cotas de la capa l
        if exact != 'prop':
            neuron_model,inpt,all_vars = update_neuron_model(neuron_model,inpt,all_vars,params,bounds,l,activation,exact)
    output_var = inpt
    ## Se entregan las cotas
    return bounds,layers_time,neuron_model,input_var,output_var,all_vars

#### Verificacion ########################################################################################################################################################

###
###

def set_verification_model(net_model,net_input_var,net_output_var,input_value,real_output,output_value,output_target,params,bounds,tol_distance = 0.1, apply_softmax = True):
    ## Cantidad de neuronas del input
    n_input = len(net_input_var)
    ## Cantidad de neuronas del output
    n_output = len(net_output_var)
    ## Restriccion de proximidad
    for i in range(n_input):
        net_model.addCons( net_input_var[i] - input_value[i] <= tol_distance, name = 'inpt_dist_{},1'.format(i))
        net_model.addCons( net_input_var[i] - input_value[i] >= -tol_distance, name = 'inpt_dist_{},2'.format(i))
    if apply_softmax:
        ## Se crean las nuevas variables para aplicar softmax
        aux_list = []
        for i in range(n_output):
            soft_output = net_model.addVar(vtype = 'C', name = 'soft_output_{}'.format(i))
            net_model.addCons(soft_output == scip.exp(net_output_var[i])/quicksum(scip.exp(net_output_var[k]) for k in range(n_output)),
                              name = 'soft_cons_{}'.format(i))
            aux_list.append(soft_output)
        net_output_var = aux_list
    ## Se genera la restriccion correspondiente a la funcion objetivo
    net_model.setObjective(net_output_var[output_target] - net_output_var[real_output], 'maximize')
    return net_model

###
###

def create_verification_model(params,bounds,activation,tol_distance,apply_softmax,image_list,output_target,real_output,exact = 'exact',apply_bounds = True):
    ## Se calcula la cantidad de capas ocultas
    n_layers = int(len(params)/2)
    ## Se inicializa el modelo de verificacion
    verif_model,inpt,all_vars = initialize_neuron_model(bounds)
    ## Se crean las restricciones de proximidad en el input
    for i in range(len(inpt)):
        verif_model.addCons( inpt[i] - image_list[i] <= tol_distance, name = 'inpt_dist_{},1'.format(i))
        verif_model.addCons( inpt[i] - image_list[i] >= -tol_distance, name = 'inpt_dist_{},2'.format(i))
    ## Se crean la evaluacion de la red
    ## Se recorren las capas
    for l in range(n_layers):
        ## Se crean las restricicones y variables
        verif_model,aux_input,all_vars = update_neuron_model(verif_model,inpt,all_vars,params,bounds,l,activation,exact,apply_bounds)
        inpt = aux_input
    ## Caso en el que se aplica softmax
    if apply_softmax:
        ## Lista para guardar las nuevas variables
        aux_list = []
        for i in range(len(inpt)):
            ## Variable post aplicacion de softmax
            soft_output = verif_model.addVar(vtype = 'C', name = 'soft_output_{}'.format(i))
            ## Restriccion de evaluacion en funcion softmax
            verif_model.addCons(soft_output == scip.exp(inpt[i])/quicksum(scip.exp(inpt[k]) for k in range(len(inpt))),
                                name = 'soft_cons_{}'.format(i))
            ## Se guarda la variable ya evaluada
            aux_list.append(soft_output)
        ## Se guarda el output
        output = aux_list
    ## Caso en el que no se aplica softmax
    else:
        ## Se guarda el output
        output = inpt
    ## Se genera la funcion objetivo
    verif_model.setObjective(output[output_target] - output[real_output], 'maximize')
    ## Se retorna el modelo de verificacion
    return verif_model,all_vars

###
###

def calculate_softmax(vector):
    aux = [np.exp(vector[i]) for i in range(len(vector))]
    aux_sum = sum(aux)
    soft = [aux[i]/aux_sum for i in range(len(vector))]
    return soft

###
###

def generate_vars_by_image(params,image_list,exact,activation = 'relu',apply_softmax = 'False'):
    ## Se genera el diccionario que contenga los valores de la solucion inicial
    sol_dict = {} 
    for i in range(len(image_list)):
        sol_dict['h{},{}'.format(-1,i)] = image_list[i]
    ## Se calcula la cantidad de capas ocultas
    n_layers = int(len(params)/2)
    ## Se calculan las variables de propagacion
    aux_input = image_list[:]
    for l in range(n_layers):
        ## Parametros de la capa l
        weight,bias = get_w_b_names(l)
        W = params[weight]
        b = params[bias]
        n_neurons = W.size()[0]
        aux_list = []
        for i in range(n_neurons):
            z = b[i] + sum(float(W[i,j])*aux_input[j] for j in range(len(aux_input)))
            sol_dict['z{},{}'.format(l,i)] = z
            if exact == 'exact':
                if activation == 'relu':
                    if z > 0:
                        a = z
                    else:
                        a = 0
                elif activation == 'softplus':
                    a = np.log(1 + np.exp(z))
                elif activation == 'sigmoid':
                    a = 1/(1+np.exp(-z))
                sol_dict['a{},{}'.format(l,i)] = a
            else:
                if activation == 'relu':
                    if z > 0:
                        h     = z
                        not_h = 0
                        u     = 1
                    else:
                        h     = 0
                        not_h = -1.0*z
                        u     = 0 
                    sol_dict['h{},{}'.format(l,i)] = h
                    sol_dict['not_h{},{}'.format(l,i)] = not_h
                    sol_dict['u{},{}'.format(l,i)] = u
                    a = h
                else:
                    if activation == 'softplus':
                        a = np.log(1 + np.exp(z))
                    elif activation == 'sigmoid':
                        a = 1/(1+np.exp(-z))
                    sol_dict['a{},{}'.format(l,i)] = a
            aux_list.append(a)
        aux_input = aux_list[:]
    if apply_softmax:
        output = calculate_softmax(aux_input)
        for i in range(len(output)):
            sol_dict['soft_output_{}'.format(i)] = output[i]
    return sol_dict

###
###

def create_initial_sol(model,params,image_list,exact,activation = 'relu',apply_softmax = 'False'):
    initial_sol = model.createSol()
    model_vars  = model.getVars()
    image_vars  = generate_vars_by_image(params,image_list,exact,activation,apply_softmax)
    for i in range(len(model_vars)):
        var_i = model_vars[i]
        var_name = var_i.name
        var_val  = image_vars[var_name]
        model.setSolVal(initial_sol, var_i, var_val)
    return initial_sol,image_vars

###
###

def calculate_probs(net,image_list,xpix = 28,ypix = 28):
    ## Se convierte la lista en un tensor de torch
    image = torch.tensor(image_list).view(1, xpix, ypix)
    ## Se calcula el output de la red
    output = net(image).tolist()[0]
    ## Se aplica softmax
    soft_output = calculate_softmax(output)
    return output,soft_output
    
###
###

def generate_png(solution,image_list,color_map,png_name,input_lb,input_ub):
  image_solution = np.array(solution).reshape(28, 28)
  image_input    = np.array(image_list).reshape(28, 28)
  ## Crea una figura con los subplots
  fig, axs = plt.subplots(1, 2)
  axs[0].imshow(image_input, vmin = input_lb, vmax = input_ub,cmap=color_map)
  axs[0].axis('off')
                        
  axs[1].imshow(image_solution, vmin = input_lb, vmax = input_ub, cmap=color_map)
  axs[1].axis('off')
                        
  #axs[2].imshow(np.abs(image_solution-image_input), vmin = input_lb, vmax = input_ub, cmap=color_map) #np.abs(image_solution-image_input)
  #axs[2].axis('off')
                        
  ## Ajusta el espaciado entre los subplots
  plt.tight_layout()
                            
  ## Guarda la figura con las imágenes en un archivo
  plt.savefig(png_name, dpi=300, bbox_inches='tight')
                    
  # Muestra la figura con las dos imágenes
  #plt.show()
  
###
###

def save_df(df,file_name,header = False,index = False):
    ## Se intenta guardar el df
    while True:
        try:
            df.to_excel(file_name,header = header, index = index)
            break
        except:
            time.sleep(5)
            
###
###

def read_df(file_name,header = None):
    ## Caso en que ya existe el df
    if os.path.exists(file_name):
        ## Se intenta leer el df
        while True:
            try:
                df = pd.read_excel(file_name,header = header)
                break
            except:
                time.sleep(5)
    ## Caso contrario se crea un df vacio
    else:
        df = pd.DataFrame()
    return df

###
###

def calculate_list_perturbation(real_list,generated_list):
    perturbations = [np.abs(generated_list[i]-real_list[i]) for i in range(len(real_list))]
    minp = min(perturbations)
    maxp = max(perturbations)
    return minp,maxp

###
###

def calculate_inflec_point(activation):
    if activation == 'sigmoid':
        inflec_point = 0
    return inflec_point

###
###

def get_activ_func(activation):
    if activation == 'relu':
        f = lambda x: (x+np.abs(x))/2
    elif activation == 'softplus':
        f = lambda x: np.log(1 + np.exp(x))
    elif activation == 'sigmoid':
        f = lambda x: 1/(1+np.exp(-x))
    return f

###
###

def get_activ_derv(activation):
    if activation == 'sigmoid':
        df = lambda x: np.exp(-x)/(np.power((1+np.exp(-x)),2))
    elif activation == 'softplus':
        df = lambda x: 1/(1+np.exp(-x))
    return df

### Funcion que calcula el cv point de la funcion. Se requiere que el ub de la cota sea mayor al punto de inflexion de la funcion de activacion.
###

def calculate_cv_point(activation,bounds,tol = 1E-05):
    ## Se genera la funcion de activacion y su derivada
    f  = get_activ_func(activation)
    df = get_activ_derv(activation)
    ## Se obtienen las cotas correspondientes
    lb,ub  = -bounds[0],bounds[1]
    ## Se determina el cv point
    aux_lb = lb
    aux_ub = calculate_inflec_point(activation)
    f_ub   = f(ub)
    ## Busqueda binaria
    while True:
        ## Se calcula el punto medio y el valor de la funcion en ese punto
        aux   = (aux_lb+aux_ub)/2
        f_aux = f(aux)
        ## Pendiente de aux con respecto al ub
        m = (f_ub-f_aux)/(ub-aux)
        ## Derivada en aux
        df_aux = df(aux)
        ## Diferencia entre la pendiente y la derivada
        dif = m-df_aux
        ## Caso cv point
        if -tol <= dif and dif <= tol:
            break
        ## Caso a la derecha del cv point
        elif -tol >= dif:
            aux_ub = aux
        ## Caso a la izquierda del cv point
        else:
            aux_lb = aux
        ## Caso en que no existe el cv point o este es el lb
        if aux-lb <= tol:
            aux = lb
            break
    ## Se retorna el cv point
    return aux

### Funcion que calcula el cc point de la funcion. Se requiere que el lb de la cota sea menor al punto de inflexion de la funcion de activacion.
###

def calculate_cc_point(activation,bounds,tol = 1E-05):
    ## Se genera la funcion de activacion y su derivada
    f  = get_activ_func(activation)
    df = get_activ_derv(activation)
    ## Se obtienen las cotas correspondientes
    lb,ub  = -bounds[0],bounds[1]
    ## Se determina el cc point
    aux_lb = calculate_inflec_point(activation)
    aux_ub = ub
    f_lb   = f(lb)
    ## Busqueda binaria
    while True:
        ## Se calcula el punto medio y el valor de la funcion en ese punto
        aux   = (aux_lb+aux_ub)/2
        f_aux = f(aux)
        ## Pendiente de aux con respecto al lb
        m = (f_aux-f_lb)/(aux-lb)
        ## Derivada en aux
        df_aux = df(aux)
        ## Diferencia entre la pendiente y la derivada
        dif = m-df_aux
        ## Caso cc point
        if -tol <= dif and dif <= tol:
            break
        ## Caso a la izquierda del cc point
        elif -tol >= dif:
            aux_lb = aux
        ## Caso a la derecha del cc point    
        else:
            aux_ub = aux
        ## Caso en que no existe el cc point o este es el ub
        if ub-aux <= tol:
            aux = ub
            break
    ## Se retorna el cc point    
    return aux

###
###

def get_tan_func(activation,x0,aux = None):
    ## Se genera la funcion de activacion y su derivada
    f  = get_activ_func(activation)
    df = get_activ_derv(activation)
    ## Se evalua la funcione en el punto de referencia
    f_x0 = f(x0)
    ## Caso en que se debe calcular la recta con respecto a un segundo punto
    if not aux == None:
        f_aux = f(aux)
        m = (f_x0-f_aux)/(x0-aux)
    ## Caso en el que solo se utiliza la derivada
    else:
        m = df(x0)
    ## Funcion lambda correspondiente a la tangente
    tan = lambda x:  f_x0 + m*(x-x0)
    ## Se retorna la funcion tangente
    return tan,m

###
###

def calculate_k_points(activation,k,lb,ub,tol = 1E-05):
    flag = True
    if flag:
        ## Se genera la funcion de activacion y su derivada
        f  = get_activ_func(activation)
        df = get_activ_derv(activation)
        ## Lista donde se guardaran los k puntos para generar las tangentes de la funcion 
        k_points = []
        ## Se calcula el valor de las derivadas en lb y ub
        df_lb = df(lb)
        df_ub = df(ub)
        ## Se determina si en [lb,ub] la funcion es convexa o concava
        ## Caso intervalo concavo
        if df_lb > df_ub:
            ## Se generan los puntos desde el mayor hasta el menor
            x0 = ub
            xf = lb
        ## Caso intervalo convexo
        else:
            x0 = lb
            xf = ub
        ## Se determina a partir de que valor la funcion comienza a cambiar su pendiente
        aux_x0 = x0
        aux_xf = xf
        ## Busqueda binaria
        while np.abs(aux_x0-aux_xf) >= tol:
            ## Se calcula el punto medio y el valor de la funcion y su derivada
            aux    = (aux_x0+aux_xf)/2
            f_aux  = f(aux)
            df_aux = df(aux)
            ## Pendiente de aux con respecto al x0
            m = (f_aux-f(x0))/(aux-x0)
            ## Diferencia con la pendiente inicial
            dif = m - df_aux
            ## Caso en que la pendiente aun se parece
            if -tol <= dif and dif <= tol:
                aux_x0 = aux
            ## Caso en que la pendiente aumento mucho
            else:
                aux_xf = aux
        ## Se verifica si es util el nuevo punto inicial
        if np.abs(aux_xf - x0) > np.abs(xf-x0)/k:
            x0 = aux_xf
        else:
            flag = False
    else:
        x0 = lb
        xf = ub
    ## Se generan los k puntos
    if flag:
        ## Paso para calcular los k puntos
        step = (xf-x0)/k
        k_points = [x0+1.05*i*step for i in range(k)]
    else:
        ## Paso para calcular los k puntos
        step = (xf-x0)/(k+1)
        k_points = [x0+1.05*(i+1)*step for i in range(k)]
    ## Se ordenan los puntos de menos a mayor
    k_points.sort()
    ## Se entregan los k puntos
    return k_points

###
###

def is_convexoconcave(activation,bounds):
    cv_list = ['softplus']
    cc_list = []
    if activation in cv_list:
        convexoconcave = 'convex'
    elif activation in cc_list:
        convexoconcave = 'concave'
    else:
        lb,ub = -bounds[0],bounds[1]
        inflec_point = calculate_inflec_point(activation)
        if ub <= inflec_point:
            convexoconcave = 'convex'
        elif lb >= inflec_point:
            convexoconcave = 'concave'
        else:
            convexoconcave = 'convexoconcave'
    return convexoconcave

###
###

def get_activation_env_list(activation,bounds,k):
    ## Se determina si la funcion es convex, concave o convexoconcave
    convexoconcave = is_convexoconcave(activation, bounds)
    ## Se obtiene la funcion de activacion
    f = get_activ_func(activation)
    ## Se obtienen las cotas
    lb,ub = -bounds[0],bounds[1]
    ## Lista para guardar las tuplas (x0,inter.,pend.) de la env convexa y concava
    cv_env = []
    cc_env = []
    ## Caso en que la funcion es convex o concave
    if convexoconcave in ['convex','concave']:
        ## Se genera la envoltura
        func = (lb,f(lb),get_tan_func(activation,lb,ub)[1])
        ## Caso convex
        if convexoconcave == 'convex':
            cc_env.append(func)
        ## Caso cocave
        else:
            cv_env.append(func)
    else:
        ## Se obtiene los cv y cc point
         cv = calculate_cv_point(activation, bounds)
         cc = calculate_cc_point(activation, bounds)
         ## Se añaden las envolturas de las cv y cc point
         cv_env.append((cv,f(cv),get_tan_func(activation,cv,ub)[1]))
         cc_env.append((lb,f(lb),get_tan_func(activation,lb,cc)[1]))
         ## Se añaden las demas envolturas convexas
         cv_points = calculate_k_points(activation,k,lb,cv)
         for i in range(k):
             aux = cv_points[i]
             cv_env.append((aux,f(aux),get_tan_func(activation,aux)[1]))
         ## Se añaden las demas envolturas concavas
         cc_points = calculate_k_points(activation,k,cc,ub)
         for i in range(k):
             aux = cc_points[i]
             cc_env.append((aux,f(aux),get_tan_func(activation,aux)[1]))
    ## Se retorna las listas con las funciones de la envoltura
    return cv_env,cc_env
