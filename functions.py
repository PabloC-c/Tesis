"""
En este archivo se encuentran las funciones para training.py y optimizacion.py
"""

#### Librerias

import os
import csv
import time
import torch
import random
import numpy as np
import pandas as pd
import torch.nn as nn
import networkx as nx
import pyscipopt.scip as scip
from itertools import product
import matplotlib.pyplot as plt
import torch.nn.functional as F
from collections import OrderedDict
from pyscipopt import Model,quicksum,SCIP_PARAMSETTING,Eventhdlr, SCIP_EVENTTYPE
from concave_env import *

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

def calculate_verif_file_name(exact,activation,real_output,target_output,root_node_only,multidim_env = False,initial_sol = None,type_bounds = None):
    if root_node_only:
         file_name = 'root_node/{}/datos_verificacion_{}_{}como{}.xlsx'.format(exact,activation,real_output,target_output)
         if multidim_env:
             file_name = 'root_node/{}/datos_verificacion_{}_{}como{}_{}_{}_.xlsx'.format(exact,activation,real_output,target_output,initial_sol,type_bounds)
    else:
        file_name = 'verif_results/{}/datos_verificacion_{}_{}como{}.xlsx'.format(exact,activation,real_output,target_output)        
    return file_name

###
###

def calculate_bounds_file_name(type_bounds,activation,n_layers,n_neurons,tol_distance,real_output):
    if type_bounds == 'verif_bounds':
        bounds_file = 'nn_bounds/{}_verifbounds_target{}_tolper{}_L{}_n{}.txt'.format(activation,real_output,int(tol_distance*100),n_layers,n_neurons)
    elif type_bounds == 'verif_bounds_prop':
        bounds_file = 'nn_bounds/{}_verifbounds_target{}_tolper{}_prop_L{}_n{}.txt'.format(activation,real_output,int(tol_distance*100),n_layers,n_neurons)
    elif type_bounds == 'mix':
        bounds_file = 'nn_bounds/{}_bounds_L{}_n{}.txt'.format(activation,n_layers,n_neurons)
    elif type_bounds == 'prop':
        bounds_file = 'nn_bounds/{}_prop_bounds_L{}_n{}.txt'.format(activation,n_layers,n_neurons)
    else:
        bounds_file = '-1'
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
                    if len(value) > 0:
                        bounds[layer] = value
                    layer += 1
                    value = []
                else:
                    lb,ub = line.strip().split()
                    value.append((float(lb), float(ub)))
                if len(value) > 0:
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
        self.n_input = n_input
        self.fc_hidden = nn.ModuleList(
            [nn.Linear(in_features  = n_neurons if i!= 0 else n_input,
                       out_features = n_neurons if i!= (n_layers-1) else n_output) for i in range(n_layers)])

    def forward(self, x):
        ## Se aplana el input
        if self.n_input == 784:
            x = x.view(-1, self.n_input)
        for l in range(self.n_layers):
            x = self.fc_hidden[l](x)
            if self.activation == 'relu':
                x = F.relu(x)
            elif self.activation == 'softplus':
                x = F.softplus(x)
            elif self.activation == 'sigmoid':
                x = F.sigmoid(x)
        return x

### Funcion que inicializa el modelo de la red, solo genera las variables de input
###

def initialize_neuron_model(bounds,add_verif_bounds,tol_distance,image_list,n_input = 784): ######### CAMBIAR A 784 ################
    neuron_model = Model()
    neuron_model.data = {}
    if len(bounds) == 0:
        if add_verif_bounds:
            bounds[-1] = [(-(max(image_list[i]-tol_distance,0)),min(image_list[i]+tol_distance,1)) for i in range(len(image_list))]
        else:
            bounds[-1] = [(0,1) for i in range(n_input)] 
    ## Tamaño del input
    n_input = len(bounds[-1])
    ## Se generan las variables de input
    inpt = [neuron_model.addVar(lb = -bounds[-1][k][0], ub = bounds[-1][k][1], name = 'h{},{}'.format(-1,k)) for k in range(n_input)]
    all_vars = {}
    for i in range(n_input):
        all_vars['h{},{}'.format(-1,i)] = inpt[i]
    return neuron_model,bounds,inpt,all_vars


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

def update_neuron_model(neuron_model,inpt,all_vars,params,bounds,l,mdenv_count,activation = 'relu',form = 'no_exact',lp_sol_file = '',apply_bounds = True,lp_relax = False):
    ## Numeros de cortes -1 a añadir en las envolturas 1 dimensional
    k = 4
    ## Numero de variables de input para la capa l
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
        if form == 'exact':
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
        ## Formulacion con envoltura
        elif form == 'no_exact':
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
                if not lp_relax:
                    u = neuron_model.addVar(vtype = 'B', name = 'u{},{}'.format(l,i))
                else:
                    u = neuron_model.addVar(lb = 0,ub = 1,vtype = 'C', name = 'u{},{}'.format(l,i))
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
        elif form == 'multidim_env':
            if not 'multidim_env_count' in neuron_model.data:
                neuron_model.data['multidim_env_count'] = {}
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
                ## Para las capas despues de la 0 (la 0 recibe la capa de input)
                succes = False
                if l > 0:
                    ## Se determina si es posible añadir la envoltura multidimensional
                    succes,cc_or_cv,c,d = calculate_hyperplane(l,i,bounds,activation,params,n_input,lp_sol_file)
                ## Caso en que es posible
                if succes:
                    ## Se aumenta el contador de cortes añadidos
                    mdenv_count += 1
                    ## Caso en que la funcion es concava
                    if cc_or_cv == 'cc':
                        ## Se añade la restriccion correspondiente
                        neuron_model.addCons(quicksum(c[k]*inpt[k] for k in range(n_input)) + d - a <= 0, name = 'cv_multidim_env0_{},{}'.format(l,i))
                    ## Caso en que la funcion es convexa
                    else:
                        neuron_model.addCons(a + quicksum(-1*c[k]*inpt[k] for k in range(n_input)) - d <= 0, name = 'cv_multidim_env0_{},{}'.format(l,i))
                    ## La neurona tiene un corte multidimensional
                    neuron_model.data['multidim_env_count'][(l,i)] = 1
                else:
                    ## La neurona tiene 0 cortes multidimensionales
                    neuron_model.data['multidim_env_count'][(l,i)] = 0
        elif form == 'env_cut':
            if not 'multidim_env_count' in neuron_model.data:
                neuron_model.data['multidim_env_count'] = {}
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
                ## Restriccion H concava
                ## Funcion y su derivada
                sigma,sigma_der = get_activ_func(activation),get_activ_derv(activation)
                ## Parametros
                neuron_w = np.array([float(W[i,k]) for k in range(n_input)])
                neuron_b = float(b[i]) 
                L = [-bounds[l-1][k][0] for k in range(n_input)]
                U = [bounds[l-1][k][1] for k in range(n_input)]
                ## Pesos re escalados
                cc_w,cc_b = concave_re_scale_0_1_box(neuron_w,neuron_b,L,U)
                ## Calculo de z_hat
                z_hat = compute_z_hat(cc_b, cc_b+np.sum(cc_w), sigma, sigma_der)
                ## Caso en que la funcion no es concava
                if np.abs(z_hat-cc_b) > 1E-05:
                    slope = (sigma(z_hat)-sigma(cc_b))/(z_hat-cc_b) 
                    inter = sigma(cc_b)
                    neuron_model.addCons(inter+slope*(quicksum(neuron_w[k]*inpt[k] for k in range(n_input))+neuron_b-cc_b) >= a , name = 'h_cc{},{}'.format(l,i))
                    mdenv_count += 1
                ## Restriccion H convexa
                ## Funcion y su derivada
                minus_sigma,minus_sigma_der = get_activ_func('-'+activation),get_activ_derv('-'+activation)
                ## Parametros
                neuron_w = np.array([float(W[i,k]) for k in range(n_input)])
                neuron_b = float(b[i]) 
                L = [-bounds[l-1][k][0] for k in range(n_input)]
                U = [bounds[l-1][k][1] for k in range(n_input)]
                ## Pesos re escalados
                cv_w,cv_b = convex_re_scale_0_1_box(neuron_w,neuron_b,L,U)
                ## Calculo de z_hat
                z_hat = compute_z_hat(cv_b, cv_b+np.sum(cv_w), minus_sigma, minus_sigma_der)
                ## Caso en que la funcion no es convexa
                if np.abs(z_hat-cv_b) > 1E-05:
                    slope = (sigma(z_hat)-sigma(cv_b))/(z_hat-cv_b) 
                    inter = sigma(cv_b)
                    neuron_model.addCons(inter+slope*(quicksum(neuron_w[k]*inpt[k] for k in range(n_input))+neuron_b-cv_b) <= a , name = 'h_cv{},{}'.format(l,i))
                    mdenv_count += 1
                neuron_model.data['multidim_env_count'][(l,i)] = 0
    return neuron_model,aux_input,all_vars,mdenv_count


### Funcion que optimiza el problema de optimizacion asociado a la evaluacion de una neurona de la red
### El parametro neuron_model es el modelo de la neurona, sense es el sentido del problema de optimziacion, tol es la holgura que se añade a las cotas duales

def solve_neuron_model(neuron_model,sense,params,bounds,l,i,exact = 'no_exact',minutes = 10,print_output = False,tol = 1E-3,feas_tol = 1E-5,digits = 4):
    ## Se calcula la cota con el metodo de propagacion
    propb = calculate_aprox_bound(params,bounds,l,i,sense)
    if exact in ['exact','no_exact','env_cut']:
        ## Limite de tiempo
        neuron_model.setParam('limits/time', int(60*minutes))
        ## Tolerancia de factibilidad
        neuron_model.setParam('numerics/feastol', feas_tol)
        ## Se resuelve el problema
        if print_output:
            neuron_model.redirectOutput()
            print('\nCaso',sense,'\n')
        else:
            neuron_model.hideOutput()
        t0 = time.time()
        try:
            aux_t = time.time()
            print('optimizando')
            neuron_model.optimize()
            print('optimizado')
            aux_dt = time.time() - aux_t
            model_status = neuron_model.getStatus()
        except:
            aux_dt = time.time() - t0
            model_status = 'problem'
        dt = aux_dt    
        #print('\n Status',neuron_model.getStatus(),'\n')
    else:
        model_status = 'problem'
    ## Caso de solucion optima
    if model_status == 'optimal':
        ## Se entrega el valor objetivo optimo
        obj_val = neuron_model.getObjVal()
        #if (sense == 'maximize' and propb < obj_val) or (sense == 'minimize' and propb > obj_val):
        #    obj_val = propb
        sol = [True,obj_val]
        aprox_bound = calculate_aprox_bound(params,bounds,l,i,sense)
        #print('\t Caso sol optima \n')
        #print('\t problema: {}, valor obj : {}, valor prop: {}'.format(sense,obj_val,aprox_bound))
    elif model_status in ['infeasible','unbounded','inforunbd','problem']:
        if exact == 'prop':
            aux_t = time.time()
            dual_val = propb
            dt = time.time() - aux_t
        else:
            try:
                dual_val = neuron_model.getDualbound()
                #dual_val = propb
                #print('bounds capa previa: ',bounds[l-1])
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

def update_bounds_vars_by_image(l,params,sol_dict,layer_input,exact,activation = 'relu'):
    ## Caso capa inicial
    if l == -1:
        for i in range(len(layer_input)):
            sol_dict['h{},{}'.format(-1,i)] = layer_input[i]
            aux_input = layer_input
    ## Para el resto de capas
    else:
        ## Parametros de la capa l
        weight,bias = get_w_b_names(l)
        W = params[weight]
        b = params[bias]
        n_neurons = W.size()[0]
        aux_input = []
        for i in range(n_neurons):
            ## Evaluacion en la funcion lineal
            z = b[i] + sum(float(W[i,j])*layer_input[j] for j in range(len(layer_input)))
            ## Se añade la variable
            sol_dict['z{},{}'.format(l,i)] = z
            ## Caso modelo exacto
            if exact == 'exact':
                ## Caso Relu
                if activation == 'relu':
                    if z > 0:
                        a = z
                    else:
                        a = 0
                ## Caso softplus
                elif activation == 'softplus':
                    a = np.log(1 + np.exp(z))
                ## Caso sigmoide
                elif activation == 'sigmoid':
                    a = 1/(1+np.exp(-z))
                ## Se añade la evaluacion en la activacion
                sol_dict['a{},{}'.format(l,i)] = a
            ## Caso envolturas convexas
            else:
                ## Caso Relu
                if activation == 'relu':
                    if z > 0:
                        h     = z
                        not_h = 0
                        u     = 1
                    else:
                        h     = 0
                        not_h = -1.0*z
                        u     = 0 
                    ## Se añaden las variables correspondientes
                    sol_dict['h{},{}'.format(l,i)] = h
                    sol_dict['not_h{},{}'.format(l,i)] = not_h
                    sol_dict['u{},{}'.format(l,i)] = u
                    a = h
                ## Caso no lineal
                else:
                    if activation == 'softplus':
                        a = np.log(1 + np.exp(z))
                    elif activation == 'sigmoid':
                        a = 1/(1+np.exp(-z))
                    ## Se añade la variable correspondiente
                    sol_dict['a{},{}'.format(l,i)] = a
            aux_input.append(a)
    return sol_dict,aux_input

###
###

def add_bounds_vars_by_image(model,sol_dict):
    initial_sol = model.createSol()
    model_vars  = model.getVars()
    for i in range(len(model_vars)):
        var_i = model_vars[i]
        var_name = var_i.name
        var_val  = sol_dict[var_name]
        model.setSolVal(initial_sol, var_i, var_val)
    accepted = model.addSol(initial_sol)
    return accepted

###
###

def calculate_bounds(params,activation = 'relu',exact = 'no_exact',minutes = 10,add_verif_bounds = False,tol_distance = 0,image_list = [],print_output=False,n_input = 784):
    ## Contador de cortes multidimensionales añadidos
    mdenv_count = 0
    ## Calcular cantidad de capas
    n_layers = int(len(params)/2)
    ## Crear arreglo para guardar cotas de las capas
    ## Inicia con las cotas del input
    bounds     = OrderedDict()
    ## Se inicializa el modelo
    neuron_model,bounds,inpt,all_vars = initialize_neuron_model(bounds,add_verif_bounds,tol_distance,image_list,n_input)
    ## Caso cotas de verificacion
    if add_verif_bounds:
        sol_dict = {}
        sol_dict,layer_input = update_bounds_vars_by_image(-1,params,sol_dict,image_list,exact,activation)
        accepted = add_bounds_vars_by_image(neuron_model,sol_dict)
    ## Variables iniciales
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
            #print('\n ===== Capa {}, neurona {} ====='.format(l,i))
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
                sol_max,dt1  = solve_neuron_model(neuron_model,'maximize',params,bounds,l,i,exact,minutes,print_output)
                neuron_model.freeTransform()
                if add_verif_bounds:
                    accepted = add_bounds_vars_by_image(neuron_model,sol_dict)    
                ## Se determina el minimo de la neurona i
                neuron_model = set_objective_function(neuron_model,inpt,params,bounds,l,i,'minimize')
                sol_min,dt2  = solve_neuron_model(neuron_model,'minimize',params,bounds,l,i,exact,minutes,print_output)
                neuron_model.freeTransform()
                if add_verif_bounds:
                    accepted = add_bounds_vars_by_image(neuron_model,sol_dict)
                ## Se añaden las cotas de la neurona al arreglo de la capa
                aux.append((sol_min[1],sol_max[1]))
                tiempo += dt1 + dt2
        ## Se anaden las cotas de la capa al arreglo bounds
        bounds[l] = aux
        layers_time.append(tiempo/n_neurons)
        ## Se actualiza el modelo con las cotas de la capa l
        if exact != 'prop':
            neuron_model,inpt,all_vars,mdenv_count = update_neuron_model(neuron_model,inpt,all_vars,params,bounds,l,mdenv_count,activation,exact)
        ## Se actualizan las variables de las cotas de verificacion
        if add_verif_bounds:
            sol_dict,layer_input = update_bounds_vars_by_image(l,params,sol_dict,layer_input,exact,activation)
            accepted = add_bounds_vars_by_image(neuron_model,sol_dict)
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

def create_verification_model(params,bounds,activation,tol_distance,apply_softmax,image_list,output_target,real_output,exact = 'exact',lp_sol_file = '',apply_bounds = True,lp_relax = False):
    ## Contador de cortes multidimensionales añadidos
    mdenv_count = 0
    ## Se calcula la cantidad de capas ocultas
    n_layers = int(len(params)/2)
    ## Se inicializa el modelo de verificacion
    verif_model,bounds,inpt,all_vars = initialize_neuron_model(bounds,False,0,[])
    ## Se crean las restricciones de proximidad en el input
    for i in range(len(inpt)):
        verif_model.addCons( inpt[i] - image_list[i] <= tol_distance, name = 'inpt_dist_{},1'.format(i))
        verif_model.addCons( inpt[i] - image_list[i] >= -tol_distance, name = 'inpt_dist_{},2'.format(i))
    ## Se crean la evaluacion de la red
    ## Se recorren las capas
    for l in range(n_layers):
        ## Se crean las restricicones y variables
        verif_model,aux_input,all_vars,mdenv_count = update_neuron_model(verif_model,inpt,all_vars,params,bounds,l,mdenv_count,activation,exact,lp_sol_file,apply_bounds,lp_relax)
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
    verif_model.setObjective(output[real_output] - output[output_target], 'minimize')
    ## Se retorna el modelo de verificacion
    return verif_model,all_vars,mdenv_count

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
    elif activation == '-relu':
        f = lambda x: -(x+np.abs(x))/2
    elif activation == 'softplus':
        f = lambda x: np.log(1 + np.exp(x))
    elif activation == '-softplus':
        f = lambda x: -np.log(1 + np.exp(x))
    elif activation == 'sigmoid':
        f = lambda x: 1/(1+np.exp(-x)) if np.abs(x) < 100 else (0.0 if x < 0 else 1.0) 
    elif activation == '-sigmoid':
        f = lambda x: -1/(1+np.exp(-x)) if np.abs(x) < 100 else (0.0 if x < 0 else -1.0) 
    return f

###
###

def get_activ_derv(activation):
    if activation == 'sigmoid':
        df = lambda x: np.exp(-x)/(np.power((1+np.exp(-x)),2)) if np.abs(x) < 100 else 0.0
    elif activation == '-sigmoid':
        df = lambda x: -np.exp(-x)/(np.power((1+np.exp(-x)),2))  if np.abs(x) < 100 else 0.0
    elif activation == 'softplus':
        df = lambda x: 1/(1+np.exp(-x))
    elif activation == '-softplus':
        df = lambda x: -1/(1+np.exp(-x))
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
         ## Se añaden las demas envolturas convexas en caso de ser necesarias
         if cv - lb >= 1E-6:
             cv_points = calculate_k_points(activation,k,lb,cv)
             for i in range(k):
                 aux = cv_points[i]
                 cv_env.append((aux,f(aux),get_tan_func(activation,aux)[1]))
         ## Se añaden las demas envolturas concavas en caso de ser necesarias
         if ub - cc >= 1E-6:
             cc_points = calculate_k_points(activation,k,cc,ub)
             for i in range(k):
                 aux = cc_points[i]
                 cc_env.append((aux,f(aux),get_tan_func(activation,aux)[1]))
    ## Se retorna las listas con las funciones de la envoltura
    return cv_env,cc_env

###
###

def sol_to_bigM(sol_file,model,params,apply_softmax):    
    sol_dict = {}
    with open(sol_file, newline='\n') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        next(reader)  # skip header
        for line in reader:
            line_info = line[0].split()
            if line_info[0][0] != 'a':
                sol_dict[line_info[0]] = float(line_info[1])
    weight,bias = get_w_b_names(0)
    n_input = params[weight].size()[1]
    for i in range(n_input):
        input_name = 'h{},{}'.format(-1,i)
        if not input_name in sol_dict:
            sol_dict[input_name] = 0.0
    ## Se calcula la cantidad de capas ocultas
    n_layers = int(len(params)/2)
    for l in range(n_layers):
        ## Parametros de la capa l
        weight,bias = get_w_b_names(l)
        W = params[weight]
        n_neurons = W.size()[0]
        if l == n_layers-1:
            aux_input = []
        for i in range(n_neurons):
            z_name = 'z{},{}'.format(l,i)
            if z_name in sol_dict:
                z = sol_dict[z_name]
            else:
                z = 0.0
                sol_dict[z_name] = z
            if z > 0:
                h = z
                not_h = 0
                u = 1
            else:
                h = 0
                not_h = -1.0*z
                u     = 0 
            sol_dict['h{},{}'.format(l,i)] = h
            sol_dict['not_h{},{}'.format(l,i)] = not_h
            sol_dict['u{},{}'.format(l,i)] = u
            if l == n_layers-1:
                aux_input.append(h)
    if apply_softmax:
        output = calculate_softmax(aux_input)
        for i in range(len(output)):
            soft_name = 'soft_output_{}'.format(i)
            if not soft_name in sol_dict:
                sol_dict[soft_name] = output[i]
    return sol_dict

###
###

def set_bigM_deafult_sol(sol_file,model,params,apply_softmax):
    initial_sol = model.createSol()
    model_vars  = model.getVars()
    default_sol = sol_to_bigM(sol_file,model,params,apply_softmax)
    for i in range(len(model_vars)):
        var_i = model_vars[i]
        var_name = var_i.name
        var_val  = default_sol[var_name]
        model.setSolVal(initial_sol, var_i, var_val)
    return initial_sol,default_sol

###
###

def read_sol_file(sol_file):
    sol_dict = {}
    with open(sol_file, newline='\n') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        next(reader)  # skip header
        for line in reader:
            line_info = line[0].split()
            sol_dict[line_info[0]] = float(line_info[1])
    return sol_dict

###
###

def get_multidim_env_points(l,bounds,activation):
    ## Cotas de la capa anterior
    aux = bounds[l-1][:]
    ## Se codifican y se aplica la funcion de activacion
    activ_f = get_activ_func(activation)
    input_bounds = [(activ_f(-aux[i][0]),activ_f(aux[i][1])) for i in range(len(aux))]
    points_list  = list(product(*input_bounds))
    return points_list

###
###

def generate_hyperplane_model_lpsol(l,i,n_input,activation,lp_sol_file):
    ## Solucion lp a cortar
    sol_dict = read_sol_file(lp_sol_file)
    ## Lista para guardar las variables de input para el modelo del hiperplano
    lp_sol = []
    for j in range(n_input+1):
        ## Indices de la capa y la neurona correspondientes
        layer_idx  = l-1
        neuron_idx = j
        if j == n_input:
            layer_idx  = l
            neuron_idx = i
        ## Nombre de la variable
        var_name = 'a{},{}'.format(layer_idx,neuron_idx)
        ## Caso en que la variable esta en la solucion lp
        if var_name in sol_dict:
            ## Valor de la variable
            var_value = sol_dict[var_name]
        ## Caso contrario
        else:
            ## Nombre de la variable z asociada
            z_name = 'z{},{}'.format(layer_idx,neuron_idx)
            ## Funcion de activacion correspondiente
            activ_f = get_activ_func(activation)
            ## Caso en que la variable z esta en la solucion lp
            if z_name in sol_dict:
                ## Valor de la variable z
                z_val = sol_dict[z_name]
            ## Caso contrario
            else:
                ## Valor de la variable z
                z_val = 0 
            ## Calculo de la variable a partir de z
            var_value = activ_f(z_val)
        ## Se añade la variable a la lista de las variables del input
        lp_sol.append(var_value)
    ## Se retorna la lista de las variables del input
    return lp_sol

###
###

def create_hyperplane_model(l,i,params,bounds,lp_sol,points_list,activation,cc_or_cv):
    ## Cantidad de variables del hiperplano
    n = len(bounds[l-1])
    ## Se genera el modelo
    hp_model = Model()
    ## Se añaden las variables del hiperplano
    c = [hp_model.addVar(lb = -1, ub = 1, name = 'c{}'.format(i)) for i in range(n)]
    ## Se añade la variable constante del hiperplano
    d = hp_model.addVar(lb = None, ub = None,name = 'd')
    ## Se añaden las restricciones
    for idx in range(len(points_list)):
        ## Se selecciona un punto
        point = points_list[idx]
        ## Parametros de la capa
        weight,bias = get_w_b_names(l)
        W = params[weight]
        b = params[bias]
        ## Se evalua el punto en la funcion lineal
        z_val = sum(float(W[i,k])*point[k] for k in range(n)) + float(b[i])
        ## Se evalua en la funcion de activacion
        activ_f = get_activ_func(activation)
        a_val = activ_f(z_val)
        ## Se añade la restriccion correspondiente 
        ## Caso dominio convexo
        if cc_or_cv == 'cv':
          hp_model.addCons(a_val + quicksum(-1*c[k]*point[k] for k in range(n)) - d <= 0, name = 'point_cons_{}'.format(idx))
        ## Caso dominio convexo
        elif cc_or_cv == 'cc':
          hp_model.addCons(quicksum(c[k]*point[k] for k in range(n)) + d - a_val <= 0, name = 'point_cons_{}'.format(idx))
    ## Se setea la funcion objetivo
    if cc_or_cv == 'cv':
        hp_model.setObjective(lp_sol[-1] + quicksum(-1*c[k]*lp_sol[k] for k in range(n)) - d, 'maximize')
    elif cc_or_cv == 'cc':
        hp_model.setObjective(quicksum(c[k]*lp_sol[k] for k in range(n)) + d - lp_sol[-1], 'maximize')
    ## Se retorna el modelo
    return hp_model,c,d

###
###

def calculate_hyperplane(l,i,bounds,activation,params,n_input,lp_sol_file):
    ## Funcion de activacion
    activ_f = get_activ_func(activation)
    ## Cotas de la neurona
    lb,ub = -bounds[l][i][0],bounds[l][i][1]
    ## Se determina si el dominio de la neurona se encuentra en la parte concava o convexa
    cc_or_cv = ''
    if activation in ['relu','softplus']:
        cc_or_cv = 'cv'
    else:
        inflec_point = calculate_inflec_point(activation)
        ## Caso concavo
        if lb >= inflec_point-1E-6 and ub > inflec_point+1E-6:
            cc_or_cv = 'cc'
        ## Caso convexo
        elif lb < inflec_point-1E-6 and ub <= inflec_point+1E-6:
            cc_or_cv = 'cv'
    succes = False
    if cc_or_cv in ['cc','cv']:
        ## Se calculan los vertices de la region del input de la capa l
        points_list = get_multidim_env_points(l,bounds,activation)
        ## Se mapea la solucion lp a cortar al modelo del hiperplano
        lp_sol = generate_hyperplane_model_lpsol(l,i,n_input,activation,lp_sol_file)
        ## Se crea el modelo del hiperplano
        hp_model,c_var,d_var = create_hyperplane_model(l,i,params,bounds,lp_sol,points_list,activation,cc_or_cv)
        ## Se resuelve el problema
        try:
            hp_model.hideOutput()
            hp_model.optimize()
        except:
            return succes,cc_or_cv,[],None
        ## Se determina el valor objetivo del problema
        obj_val = hp_model.getObjVal()
        ## Se determina si la solucion lp es cortada
        if obj_val > 1E-3:
            succes = True
            c = [hp_model.getVal(var) for var in c_var]
            d = hp_model.getVal(d_var)
        else:
            c = []
            d = None
    else:
        c = []
        d = None
    return succes,cc_or_cv,c,d

###
###

def get_bounds_model_lpsol(neuron_l,n_input,n_neurons,bounds_model,all_vars):
    ## Lista para guardar las variables de input para el modelo del hiperplano
    lp_sol = {}
    ## Se recorren las capas
    for l in (-1,neuron_l):
        ## Caso capa de entrada
        n = n_input
        name_var = 'h{},{}'
        ## Resto de capas
        if l > -1:
            n = n_neurons
            name_var = 'a{},{}'
        ## Se guarda la solucion
        for i in range(n):
            lp_sol[(l,i)] = bounds_model.getVal(all_vars[name_var.format(l,i)])
    return lp_sol
    

###
###

def cut_verif_model_lp_sol(n_layers,n_neurons,activation,params,bounds,verif_model,all_vars,lp_sol_file):
    ## Contador de cortes multidimensionales añadidos
    mdenv_count = 0
    ## Se recorren las capas
    for l in range(1,n_layers):
        ## Output de la capa anterior
        layer_inpt = [all_vars['a{},{}'.format(l-1,j)] for j in range(n_neurons)]
        ## Se recorren las neuronas
        for i in range(n_neurons):
            ## Variable de output de la neurona
            a = all_vars['a{},{}'.format(l,i)]
            succes = False
            ## Se determina si es posible añadir la envoltura multidimensional
            succes,cc_or_cv,c,d = calculate_hyperplane(l,i,bounds,activation,params,n_neurons,lp_sol_file)
            ## Caso en que es posible
            if succes:
                ## Se aumenta el contador de cortes añadidos
                mdenv_count += 1
                ## Caso en que la funcion es concava
                n_cuts = verif_model.data['multidim_env_count'][(l,i)]
                if cc_or_cv == 'cc':
                    ## Se añade la restriccion correspondiente
                    verif_model.addCons(quicksum(c[k]*layer_inpt[k] for k in range(n_neurons)) + d - a <= 0, name = 'cv_multidim_env{}_{},{}'.format(n_cuts,l,i))
                ## Caso en que la funcion es convexa
                else:
                    verif_model.addCons(a + quicksum(-1*c[k]*layer_inpt[k] for k in range(n_neurons)) - d <= 0, name = 'cc_multidim_env{}_{},{}'.format(n_cuts,l,i))
                ## Se aumenta la cantidad de cortes multidimensionales que tiene 
                verif_model.data['multidim_env_count'][(l,i)] += 1
    return verif_model,mdenv_count

###
###

def env_cut_verif_model_lp_sol(neuron_l,n_input,n_neurons,activation,params,bounds,model,all_vars,lp_sol,type_cut):
    ## Contador de cortes multidimensionales añadidos
    mdenv_count = 0
    ## Se recorren las capas
    for l in range(neuron_l):
        ## Input de la capa
        var_name = 'h{},{}'
        n = n_input
        if l > 0:
            var_name = 'a{},{}'
            n = n_neurons
        layer_inpt = [all_vars[var_name.format(l-1,j)] for j in range(n)]
        sol_tocut = np.array([lp_sol[var_name.format(l-1,j)] for j in range(n)]) 
        ## Parametros de la capa
        weight,bias = get_w_b_names(l)
        W,b = params[weight],params[bias]
        ## Se recorren las neuronas
        for i in range(n_neurons):
            ## Variable de output de la neurona
            a = all_vars['a{},{}'.format(l,i)]
            ## Valor a cortar
            z_tocut = lp_sol['a{},{}'.format(l,i)]
            ## Numero de cortes totales de la neurona
            n_cuts = model.data['multidim_env_count'][(l,i)]
            ## Parametros de la neurona
            neuron_w = np.array([float(W[i,k]) for k in range(n)])
            neuron_b = float(b[i]) 
            L = [-bounds[l-1][k][0] for k in range(n)]
            U = [bounds[l-1][k][1] for k in range(n)]
            ## Solucion a cortar
            sigma = get_activ_func(activation)
            ## Valor de la funcion en la solucion a cortar
            f_sol_tocut = sigma(sum(neuron_w*sol_tocut)+neuron_b)
            ## Corte envoltura concava
            if f_sol_tocut - z_tocut < -1E-01:
                ## Pesos re escalados
                cc_w,cc_b = concave_re_scale_0_1_box(neuron_w,neuron_b,L,U)
                ## Funcion de activacion y su derivada
                sigma_der = get_activ_derv(activation)
                ## Solucion re escalada
                rescaled_sol = concave_re_scale_vector(sol_tocut,neuron_w,L,U)
                ## Se calcula z_hat
                z_hat = compute_z_hat(cc_b, cc_b+np.sum(cc_w), sigma, sigma_der)
                ## Se identifica la region del vector
                R_f,R_l = vector_in_region(cc_b,cc_b+np.sum(cc_w),np.dot(cc_w,rescaled_sol),cc_b,z_hat)
                if (type_cut == 'R_H,f' and R_f) or (type_cut == 'R_H,f,i' and not (R_f or R_l)):
                    ## Constante del plano
                    z_env0 = concave_envelope(rescaled_sol, cc_w, cc_b, sigma, sigma_der)
                    ## Derivada de la solucion
                    der = concave_envelope_derivate(rescaled_sol, cc_w, cc_b, sigma, sigma_der)
                    der = concave_scale_der_by_w(der,neuron_w,U,L)
                    ## Restriccion plano cortante de la envoltura concava
                    model.addCons(z_env0+quicksum(der[k]*(layer_inpt[k]-sol_tocut[k]) for k in range(n)) - a >= -1E-09, name = 'cc_env_cut{}_{},{}'.format(n_cuts,l,i))
                    ## Se aumenta la cantidad de cortes multidimensionales de la neurona 
                    model.data['multidim_env_count'][(l,i)] += 1
                    ## Se aumenta la cantidad de cortes totales
                    mdenv_count += 1
            ## Corte envoltura convexa
            elif f_sol_tocut - z_tocut > 1E-01:
                ## Pesos re escalados
                cv_w,cv_b = convex_re_scale_0_1_box(neuron_w,neuron_b,L,U)
                ## Funcion de activacion y su derivada
                sigma,sigma_der = get_activ_func('-'+activation),get_activ_derv('-'+activation)
                ## Solucion re escalada
                rescaled_sol = convex_re_scale_vector(sol_tocut,neuron_w,L,U)
                ## Se calcula z_hat
                z_hat = compute_z_hat(cv_b, cv_b+np.sum(cv_w), sigma, sigma_der)
                ## Se identifica la region del vector
                R_f,R_l = vector_in_region(cv_b,cv_b+np.sum(cv_w),np.dot(cv_w,rescaled_sol),cv_b,z_hat)
                if (type_cut == 'R_H,f' and R_f) or (type_cut == 'R_H,f,i' and not (R_f or R_l)):
                    ## Constante del plano
                    z_env0 = -concave_envelope(rescaled_sol, cv_w, cv_b, sigma, sigma_der)
                    ## Derivada de la solucion
                    der = concave_envelope_derivate(rescaled_sol, cv_w, cv_b, sigma, sigma_der)
                    der = -convex_scale_der_by_w(der,neuron_w,U,L)
                    ## Restriccion plano cortante de la envoltura concava
                    model.addCons(z_env0+quicksum(der[k]*(layer_inpt[k]-sol_tocut[k]) for k in range(n)) - a <= 1E-09, name = 'cv_env_cut{}_{},{}'.format(n_cuts,l,i))
                    ## Se aumenta la cantidad de cortes multidimensionales de la neurona 
                    model.data['multidim_env_count'][(l,i)] += 1
                    ## Se aumenta la cantidad de cortes totales
                    mdenv_count += 1
    return model,mdenv_count

#### Generacion de columnas ########################################################################################################################################################

###
###

def get_partition(partition,l,i): 
    for k in range(len(partition)): 
        if (l,i) in partition[k]:
            return k

### Funcion que para una red y una particion, genera todos los modelos de pricing asociados
###

def create_pricing_models(n_clusters,partition,edges_p,lambda_params,bounds,params,hat_x,real_label,target_label,eps,activation='relu',pricing_models=[],all_vars={},eta=False):
    """
    Parametros
    ----------
    n_clusters : int
        Cantidad de cluster a considerar.
    partition : list
        Cada entrada contiene una lista con las duplas (layer,neuron) de la particion correspondiente.
    edges_p : list
        Cada entrada es una tupla (layer,neuron,previous_neuron) que conecta dos particiones diferentes. 
    lambda_params : dict
        Llaves: entradas de edges_p. Valores: valor lambda correspondiente.
    bounds : dict
        Llaves: capas de -1 a L-1. Valores: lista con duplas de cotas (-l,u) para cada neurona.
    params : dict
        Contiene los parametros W y b de la red.
    hat_x : list
        Lista correspondiente al input de referencia.
    real_label : int
        Indice de la neurona de la capa de output, corresponde a la verdadera clase de hat_x.
    target_label : int
        Indice de una neurona de la capa de output, diferente a real_label.
    eps : float
        Distancia maxima entre la variable de input y el input de referencia.
    activation : string
        Funcion de activacion correspondiente a la red.
    Returns
    -------
    pricing_models : list
        Contiene los modelos de pricing asociados a las particiones.
    all_vars : dict
        LLaves: nombre de todas las variables de los K modelos. Valores: variable correspondiente
    """
    ## Caso inicial
    if len(pricing_models) == 0:
        ## Arreglos para guardar los modelos de pricing
        pricing_models = []
        models_dict = {}
        ## Diccionario para guardar las variables
        all_vars = {}
        ## Diccionario para guardar las variables x de cada neurona
        x_dict = {}
        ## Diccionario para guardar las variables y duplicadas
        y_dict = {}
        ## Se recorren las capas
        for l in range(-1,len(bounds)-1):
            ## Numero de neuronas de la capa
            n_neurons = len(bounds[l])
            ## Se recorren las neuronas de la capa
            for i in range(n_neurons):
                ## Se determina la particion a la que pertenece
                k = get_partition(partition,l,i)
                ## Modelo de la particion
                if not k in models_dict:
                    models_dict[k] = Model()        
                k_model = models_dict[k]
                ## Se agregan las varaibles de la neurona
                k_model,x,y_list,x_dict,y_dict,all_vars = set_dupla_vars(k_model,l,i,edges_p,x_dict,y_dict,all_vars,bounds,hat_x,eps,activation,partition[k])
                ## Restricciones de la funcion de activacion
                if l > -1:
                    k_model,all_vars = set_activfunction_cons(k_model,l,i,x,y_list,all_vars,bounds,params,activation)
        ## Se añaden las funciones objetivo, y se genera la lista de modelos
        for k in range(n_clusters):
            ## Modelo k
            k_model = models_dict[k]
            ## Funcion objetivo
            k_model,all_vars = set_partition_objective(k,k_model,x_dict,y_dict,edges_p,lambda_params,partition[k],real_label,target_label,params,all_vars,eta)
            ## Se agrega el modelo a la lista
            pricing_models.append(k_model)
    else:
        for k in range(n_clusters):
            k_model = pricing_models[k]
            k_model.freeTransform()
            ## Caso eta
            if eta:
                ## Se agrega la penalizacion
                k_model = add_eta_penaltycons(k,k_model,edges_p,lambda_params,partition[k],all_vars)
                pricing_models[k] = k_model
            else:
                ## Se actualiza la funcion objetivo
                k_model,all_vars = set_partition_objective(k,k_model,{},{},edges_p,lambda_params,partition[k],real_label,target_label,params,all_vars,eta)
    return pricing_models,all_vars

### Funcion que añade las varibles de una neurona en especifico al modelo de su particion correspondiente
###

def set_dupla_vars(k_model,l,i,edges_p,x_dict,y_dict,all_vars,bounds,hat_x,eps,activation,partition_k = None):
    """
    Parametros
    ----------
    k_model : scip model
        Modelo asociado a la particion k.
    l : int
        Indice de la capa de la dupla.
    i : int
        Indice de la neurona de la dupla. 
    edges_p : list
        Cada entrada es una tupla (layer,neuron,previous_neuron) que conecta dos particiones diferentes. 
    x_dict : dict
        Llaves: dupla correspondiente. Valores: variable correspondiente.
    y_dict : dict
        Llaves: tupla correspondiente. Valores: variable correspondiente.
    all_vars : dict
        LLaves: nombre de un variable. Valores: variable correspondiente.
    bounds : dict
        Llaves: capas de -1 a L-1. Valores: lista con duplas de cotas (-l,u) para cada neurona.
    hat_x : list
        Lista correspondiente al input de referencia.
    eps : float
        Distancia maxima entre la variable de input y el input de referencia.
    activation : string
        Funcion de activacion correspondiente a la red.
    Returns
    -------
    k_model : scip model
        Modelo asociado a la particion k.
    x: scip var
        Variable x asociada a la dupla.
    y_list : list
        Lista con las variables duplicadas de la capa anterior.
    x_dict : dict
        Llaves: dupla correspondiente. Valores: variable correspondiente.
    y_dict : dict
        Llaves: tupla correspondiente. Valores: variable correspondiente.
    all_vars : dict
        LLaves: nombre de un variable. Valores: variable correspondiente.
    """
    ## Cotas correspondiente
    lb,ub = -bounds[l][i][0],bounds[l][i][1]
    ## Lista para guardar las variables duplicadas de la capa anterior
    y_list = []
    ## Caso neurona de input
    if l == -1 and not (l,i) in x_dict:
        ## Se agrega la variable de input
        x = k_model.addVar(lb = max(lb,hat_x[i]-eps),
                           ub = min(ub,hat_x[i]+eps),
                           vtype = 'C',
                           name = 'x^{}_{}'.format(l,i))
    ## Caso contrario
    else:
        ## Se agrega la variable de output de la neurona
        x = k_model.addVar(lb = None,
                           ub = None,
                           vtype = 'C',
                           name = 'x^{}_{}'.format(l,i))
        ## Se recorren las neuronas de la capa anterior
        for j in range(len(bounds[l-1])):
            ## Caso en que la relacion de las nuronas j,i es relajada
            if (l,i,j) in edges_p:
                ## Cotas de la neurona j
                lb_aux,ub_aux = -bounds[l-1][j][0],bounds[l-1][j][1]
                ## Caso en que la capa anterior es el input
                if l == 0:
                    ## Se agrega las variable duplicadas del input j de la neurona i
                    y = k_model.addVar(lb = max(lb_aux,hat_x[j]-eps),
                                       ub = min(ub_aux,hat_x[j]+eps),
                                       vtype = 'C',
                                       name = 'y^{},{}_{}'.format(l-1,i,j))
                ## Para las demas capas
                else:
                    ## Caso relu
                    if activation == 'relu':
                        ## Se ajustan las cotas correspondientes
                        lb_aux,ub_aux = max(0,lb_aux),max(0,ub_aux)
                    ## Se agrega la variable duplicada del input j de la neurona i
                    y = k_model.addVar(lb = lb_aux,
                                       ub = ub_aux,
                                       vtype = 'C',
                                       name = 'y^{},{}_{}'.format(l-1,i,j))
            ## Caso en que la relacion no es relajada
            else:
                ## Se agrega las variable duplicadas del input j de la neurona i
                y = k_model.addVar(lb = None,
                                   ub = None,
                                   vtype = 'C',
                                   name = 'y^{},{}_{}'.format(l-1,i,j))
                ## Variable que y duplica
                x_aux = x_dict[(l-1,j)]
                ## Restriccion de igualdad
                k_model.addCons(y - x_aux == 0, name = 'duplicate_{},{},{}'.format(l,i,j))
            ## Se agrega la variable y a su lista
            y_list.append(y)
            ## Se agrega la variable y a su diccionario
            y_dict[(l-1,i,j)] = y
            ## Se agrega la variable y en el diccionario de variables
            all_vars['y^{},{}_{}'.format(l-1,i,j)] = y
    ## Se agrega la variable x en su diccionario
    x_dict[(l,i)] = x
    ## Se agrega la variable x en el diccionario de variables     
    all_vars['x^{}_{}'.format(l,i)] = x
    return k_model,x,y_list,x_dict,y_dict,all_vars

### Funcion que agrega las restricciones asociadas a la funcion de activacion de una neurona
###

def set_activfunction_cons(k_model,l,i,x,y_list,all_vars,bounds,params,activation):
    """
    Parametros
    ----------
    k_model : scip model
        Modelo asociado a la particion k.
    l : int
        Indice de la capa de la dupla.
    i : int
        Indice de la neurona de la dupla. 
    x: scip var
        Variable x asociada a la dupla.
    y_list : list
        Lista con las variables duplicadas de la capa anterior.
    all_vars : dict
        LLaves: nombre de un variable. Valores: variable correspondiente.
    bounds : dict
        Llaves: capas de -1 a L-1. Valores: lista con duplas de cotas (-l,u) para cada neurona.
    params : dict
        Contiene los parametros W y b de la red.
    activation : string
        Funcion de activacion correspondiente.

    Returns
    -------
    k_model : scip model
        Modelo asociado a la particion k.
    all_vars : dict
        LLaves: nombre de un variable. Valores: variable correspondiente.

    """
    ## Cotas correspondientes
    lb,ub = -bounds[l][i][0],bounds[l][i][1]
    ## Parametros de la capa l
    w_name,b_name = get_w_b_names(l)
    W,b = params[w_name],params[b_name]
    ## Caso ReLU
    if activation == 'relu':
        ## Se agrega la variable binaria
        z = k_model.addVar(vtype = 'B',
                           name = 'z^{}_{}'.format(l,i))
        ## Termino lineal de la neurona
        lineal_term = b[i]+quicksum(W[i,j]*y_list[j] for j in range(len(y_list)))
        ## Restricciones big-m
        k_model.addCons(0 <= x, name = f'bigm_upper1_{l},{i}')
        k_model.addCons(x <= ub*z, name = f'bigm_upper2_{l},{i}')
        k_model.addCons(0 <= x-lineal_term, name = f'bigm_lower1_{l},{i}')
        k_model.addCons(x-lineal_term <= lb*(z-1), name = f'bigm_lower2_{l},{i}')
        ## Se guarda la variable binaria en el diccionario
        all_vars['z^{}_{}'.format(l,i)] = z
    return k_model,all_vars

### Funcion que agrega la funcion objetivo de la particion k
###

def set_partition_objective(k,k_model,x_dict,y_dict,edges_p,lambda_params,k_partition,real_label,target_label,params,all_vars,eta):
    """
    Parametros
    ----------
    k_model : scip model
        Modelo asociado a la particion k.
    x_dict : dict
        Llaves: dupla correspondiente. Valores: variable correspondiente.
    y_dict : dict
        Llaves: tupla correspondiente. Valores: variable correspondiente.
    edges_p : list
        Cada entrada es una tupla (layer,neuron,previous_neuron) que conecta dos particiones diferentes. 
    lambda_params : dict
        Llaves: entradas de edges_p. Valores: valor lambda correspondiente.
    k_partition : list
        Contiene duplas de la forma (layer,neuron) de la particion k.
    real_label : int
        Indice de la neurona de la capa de output, corresponde a la verdadera clase de hat_x.
    target_label : int
        Indice de una neurona de la capa de output, diferente a real_label.
    params : dict
        Contiene los parametros W y b de la red.

    Returns
    -------
    k_model : scip model
        Modelo asociado a la particion k.
    """
    ## Cantidad total de capas
    n_layers = int(len(params)/2)
    ## Se identifican las variables y en caso de ser necesario
    if len(y_dict) == 0:
        for (l,i,j) in edges_p:
            if (l,i) in k_partition:
                y_dict[(l-1,i,j)] = all_vars[f'y^{l-1},{i}_{j}']
    ## Se identifican las variables x en caso de ser necesario
    if len(x_dict) == 0:
        for (l,i,j) in edges_p:
            if (l-1,j) in k_partition:
                x_dict[(l-1,j)] = all_vars[f'x^{l-1}_{j}']
    if (n_layers-1,real_label) in k_partition:
        x_dict[(n_layers-1,real_label)] = all_vars[f'x^{n_layers-1}_{real_label}']
    if (n_layers-1,target_label) in k_partition:
        x_dict[(n_layers-1,target_label)] = all_vars[f'x^{n_layers-1}_{target_label}']
    ## Termino para la funcion objetivo
    penalty = 0
    ## Se suman las variables y
    penalty += quicksum(lambda_params[(l,i,j)]*y_dict[(l-1,i,j)] for (l,i,j) in edges_p if (l,i) in k_partition)
    ## Se suman las variables x
    penalty += quicksum(-lambda_params[(l,i,j)]*x_dict[(l-1,j)] for (l,i,j) in edges_p if (l-1,j) in k_partition)
    ## Funcion objetivo
    objective = 0
    ## Caso agregar eta
    if eta:
        ## Se agrega la variable eta en caso de ser necesario
        if not  f'eta_{k}' in all_vars:
            eta_var = k_model.addVar(lb=None,ub=None,vtype='C',name=f'eta_{k}')
            all_vars[f'eta_{k}'] = eta_var
        else:
            eta_var = all_vars[f'eta_{k}']
        ## Se añade la restriccion
        k_model.addCons(eta_var >= penalty ,name = 'eta_penaltycons')
        ## Funcion objetivo
        objective += eta_var
    ## Caso sin eta
    else:
        objective += penalty
    ## Se suma el termino de la clase real
    if (n_layers-1,real_label) in k_partition:
        objective += x_dict[(n_layers-1,real_label)]
    ## Se suma el termino de la clase target
    if (n_layers-1,target_label) in k_partition:
        objective += -x_dict[(n_layers-1,target_label)]
    ## Se setea la funcion objetivo
    #print(f'funcion objetivo pricing: {objective}')
    k_model.setObjective(objective,'minimize')
    return k_model,all_vars

###
###

def add_eta_penaltycons(k,k_model,edges_p,lambda_params,k_partition,all_vars):
    ## Variable eta
    eta_var = all_vars[f'eta_{k}']
    ## Se identifican las variables y corrrespondientes
    y_dict = {}
    for (l,i,j) in edges_p:
        if (l,i) in k_partition:
            y_dict[(l-1,i,j)] = all_vars[f'y^{l-1},{i}_{j}']
    ## Se identifican las variables x correspondientes
    x_dict = {}
    for (l,i,j) in edges_p:
        if (l-1,j) in k_partition:
            x_dict[(l-1,j)] = all_vars[f'x^{l-1}_{j}']
    ## Termino de la penalizacion
    penalty = 0
    ## Se suman las variables y
    penalty += quicksum(lambda_params[(l,i,j)]*y_dict[(l-1,i,j)] for (l,i,j) in edges_p if (l,i) in k_partition)
    ## Se suman las variables x
    penalty += quicksum(-lambda_params[(l,i,j)]*x_dict[(l-1,j)] for (l,i,j) in edges_p if (l-1,j) in k_partition)
    ## Se agrega la restriccion
    k_model.addCons(eta_var >= penalty ,name='eta_penaltycons')
    return k_model 

###
###

def get_pricing_column(pricing_models,edges_p):
    ## Diccionario para la columna
    column = {}
    ## Se recorren los pricing
    for model in pricing_models:
        ## Se recorren las variables
        for var in model.getVars():
            ## Se añaden todas las variables x
            if var.name[0] == 'x':
                ## Se añade la variable a la columna
                column[var.name] = model.getVal(var)
            ## Para las variables y
            if var.name[0] == 'y':
                ## Se determinan los valores (l,i,j) asociados
                parts = var.name.split(',')
                l = int(parts[0].split('^')[1])+1
                i = int(parts[1].split('_')[0])
                j = int(parts[1].split('_')[1])
                ## Si la conexion es relajada
                if (l,i,j) in edges_p:
                    ## Se añade la variable y
                    column[var.name] = model.getVal(var)
    return column
###
###

def create_master_model(columns,edges_p,real_label,target_label,params,master_model = None,theta = [],cons_dict = {}):
    ## Numero de capas
    n_layers = int(len(params)/2)
    ## Caso inicial
    if master_model == None:
        ## Se crea el modelo maestro
        master_model = Model()
        ## Diccionario para guardar las restricciones
        cons_dict = {}
        ## Variables de la combinacion convexa
        theta = [master_model.addVar(lb = 0, ub = 1, vtype = 'C', name = f'theta_{q}') for q in range(len(columns))]
        ## Restriccion convexa
        cons = master_model.addCons(quicksum(theta[q] for q in range(len(columns))) == 1, name = 'convex_comb')
        cons_dict['conv'] = cons
        ## Restriccion sobre las relaciones relajadas
        for (l,i,j) in edges_p:
            ## Nombre de la variable x e y
            x_name = f'x^{l-1}_{j}'
            y_name = f'y^{l-1},{i}_{j}'
            ## Sumatoria de la restriccion
            cons_lhs = quicksum(theta[q]*(columns[q][y_name]-columns[q][x_name]) for q in range(len(columns)))
            ## Se agrega la restriccion
            cons = master_model.addCons(cons_lhs == 0, name = f'relaxed_cons_{l},{i},{j}')
            ## Se guarda la restriccion
            cons_dict[(l,i,j)] = cons
        ## Nombre de las variables de la funcion objetivo
        xr_name = f'x^{n_layers-1}_{real_label}'
        xt_name = f'x^{n_layers-1}_{target_label}'
        ## Funcion objetivo
        objective = quicksum(theta[q]*(columns[q][xr_name]-columns[q][xt_name]) for q in range(len(columns)))
        ## Se setea la funcion objetivo
        master_model.setObjective(objective,'minimize')
    ## Caso donde se añade una columna
    else:
        ## Se setea el modelo
        master_model.freeTransform()
        ## Funcion objetivo
        obj_function = master_model.getObjective()
        ## Se recorren las nuevas columnas
        for q in range(len(master_model.getVars()),len(columns)):
            ## Se añade la nueva variable
            new_theta = master_model.addVar(lb = 0, ub = 1, vtype = 'C', name = f'theta_{q}')
            theta.append(new_theta)
            ## Se actualiza la restriccion de convexidad
            cons = cons_dict['conv']
            master_model.addConsCoeff(cons,new_theta,1)
            cons_dict['conv'] = master_model.getConss()[0]
            ## Restricciones de las relaciones relajadas
            counter = 1
            for (l,i,j) in edges_p:
                ## Nombre de la variable x e y
                x_name = f'x^{l-1}_{j}'
                y_name = f'y^{l-1},{i}_{j}'
                ## Coeficiente de la nueva variable
                coef = columns[q][y_name]-columns[q][x_name]
                ## Restriccion de la relacion
                cons = cons_dict[(l,i,j)]
                ## Se agrega el termino de la nueva columna
                master_model.addConsCoeff(cons, new_theta, coef)
                ## Se actualiza el diccionario
                cons_dict[(l,i,j)] = master_model.getConss()[counter]
                counter+=1
            ## Nombre de las variables de la funcion objetivo
            xr_name = f'x^{n_layers-1}_{real_label}'
            xt_name = f'x^{n_layers-1}_{target_label}'
            ## Coeficiente de la nueva variable
            obj_coef = columns[q][xr_name]-columns[q][xt_name]
            ## Funcion objetivo actualizada
            obj_function += new_theta*obj_coef
        ## Se setea la funcion objetivo
        master_model.setObjective(obj_function, sense='minimize')
    return master_model,theta,cons_dict

###
### 

def propagate_input_to_column(ref_input,params,edges_p,activation = 'relu'):
    ## Diccionario para guardar la columna
    column = {}
    ## Se añaden las variables de entrada
    for i in range(len(ref_input)):
        column[f'x^{-1}_{i}'] = ref_input[i]
    ## Numero de capas
    n_layers = int(len(params)/2)
    ## Se recorren las capas
    for l in range(n_layers):
        ## Parametros de la capa
        wname,bname = get_w_b_names(l)
        W,b = params[wname],params[bname]
        ## Neuronas de la capa 
        n_neurons = W.size()[0]
        ## Neuronas de la capa previa
        n_input = W.size()[1]
        for i in range(n_neurons):
            ## Valor de salida de la neurona
            neuron_output = float(b[i])
            ## Se recorren las neuronas de entrada
            for j in range(n_input):
                ## Se suma el termino correspondiente a la salida
                neuron_output += float(W[i,j])*column[f'x^{l-1}_{j}']
                ## Si la relacion es relajada
                if (l,i,j) in edges_p:
                    column[f'y^{l-1},{i}_{j}'] = column[f'x^{l-1}_{j}']
            ## Se añade la variable de salida de la neurona
            if activation == 'relu':
                column[f'x^{l}_{i}'] = max(0,neuron_output)
    return column

###
###

def get_master_lambdas(master_model,edges_p,cons_dict,columns,n_layers,real_label,target_label,lambda_params = None):
    ## Diccionario para los parametros lambda 
    if lambda_params is None:
        lambda_params = {}
        for edge in edges_p:
            lambda_params[edge] = 0
    ## Se recorren las aristas relajadas
    for edge in edges_p:
        ## Restriccion correspondiente
        cons = cons_dict[edge]
        ## Valor dual
        dual_value = master_model.getDualsolLinear(cons)#master_model.getDualSolVal(cons)
        ## Se actualiza lambda
        lambda_params[edge] = -dual_value
    ## Se reescalan los lambda
    #print(f'Lambda previos: {lambda_params}')
    lambda_params = re_scale_lambda(master_model,cons_dict,lambda_params,edges_p,columns,n_layers,real_label,target_label)
    #print(f'Lambda reescalados: {lambda_params}')
    return lambda_params

###
###

def make_theta_solution(master_model,theta,columns):
    sol = {}
    for q in range(len(theta)):
        theta_var = theta[q]
        theta_val = master_model.getVal(theta_var)
        for vname in columns[q]:
            if vname[0] == 'x':
                if not vname in sol:
                    sol[vname] = theta_val*columns[q][vname]
                else:
                    sol[vname] += theta_val*columns[q][vname]
    return sol

###
###

def compute_pricing_sol_obj(pricing_models,pricing_vars,partition):
    ## Diccionario para guardar la solucion
    pricing_sol = {}
    ## Se recorren las variables
    for vname in pricing_vars:
        ## Variables x
        if vname[0] == 'x':
            ## Se identifica la neurona de la variable
            parts = vname.split('^')
            l = int(parts[1].split('_')[0])
            i = int(parts[1].split('_')[1])
            ## Particion a la que pertenece
            k = get_partition(partition,l,i)
            ## Variable de salida de la neurona
            var = pricing_vars[vname]
            ## Valor de la variable
            val = pricing_models[k].getVal(var)
            ## Se agrega la variable
            pricing_sol[vname] = val
    ## Valor objetivo
    pricing_obj = 0
    ## Se recorren las particiones
    for model in pricing_models:
        pricing_obj += model.getObjVal()
    return pricing_obj,pricing_sol

###
###

def propagation_heuristic(sol,params,real_label,target_label,activation = 'relu'):
    ## Diccionario para guarda la solucion
    heu_sol = {}
    ## Numero de capas
    n_layers = int(len(params)/2)
    ## Tamaño del input
    wname,bname = get_w_b_names(0)
    W = params[wname]
    n_input = W.size()[1]
    ## Lista donde guardar las variables asociadas al input
    input_list = []
    ## Se recorren las variables de input
    for i in range(n_input):
        ## Se añaden a la lista
        input_list.append(sol[f'x^{-1}_{i}'])
        heu_sol[f'x^{-1}_{i}'] = sol[f'x^{-1}_{i}']
    ## Se recorren las capas
    for l in range(n_layers):
        ## Parametros de la capa
        wname,bname = get_w_b_names(l)
        W,b = params[wname],params[bname]
        ## Neuronas de la capa previa
        n_input = W.size()[1]
        ## Neuronas de la capa
        n_neurons = W.size()[0]
        ## Lista para guardar el output de la capa
        aux_list = []
        ## Se recorren las neuronas de la capa
        for i in range(n_neurons):
            ## Evaluacion en la funcion lineal
            val = float(b[i]) + sum(float(W[i,j])*input_list[j] for j in range(n_input))
            if activation == 'relu':
                val = max(0,val)
            ## Se guarda el output de la neurona
            aux_list.append(val)
            heu_sol[f'x^{l}_{i}'] = val
        ## Se actualiza el input de la siguiente capa   
        input_list = aux_list
    ## Valor objetivo
    heu_obj = aux_list[real_label]-aux_list[target_label]
    return heu_obj,heu_sol

def pricing_zfixed_heuristic(pricing_models,pricing_vars,partition,params,real_label,target_label,activation = 'relu'):
    ## Diccionario para la solucion
    heu_sol = {}
    ## Numero de capas
    n_layers = int(len(params)/2)
    ## Tamaño del input
    wname,bname = get_w_b_names(0)
    W = params[wname]
    n_input = W.size()[1]
    ## Lista donde guardar las variables asociadas al input
    input_list = []
    ## Se recorren las variables de input
    for i in range(n_input):
        ## Se identifica a que particion pertenece
        k = get_partition(partition, -1, i)
        ## Nombre de la variable
        var_name = f'x^{-1}_{i}'
        ## Variable asociada
        var = pricing_vars[var_name]
        ## Valor del pricing
        val = pricing_models[k].getVal(var)
        ## Se añade el valor
        input_list.append(val)
        heu_sol[var_name] = val
    ## Se recorren las capas
    for l in range(n_layers):
        ## Parametros de la capa
        wname,bname = get_w_b_names(l)
        W,b = params[wname],params[bname]
        ## Neuronas de la capa previa
        n_input = W.size()[1]
        ## Neuronas de la capa
        n_neurons = W.size()[0]
        ## Lista para guardar el output de la capa l
        aux_list = []
        ## Se recorren las neuronas de la capa
        for i in range(n_neurons):
            ## Particion de la neurona
            k = get_partition(partition, l, i)
            ## Variable z
            zvar = pricing_vars[f'z^{l}_{i}']
            ## Valor de z
            z = pricing_models[k].getVal(zvar)
            ## Neurona inactiva
            if z == 0:
                val = 0
            ## Neurona supuestamente activa
            else:
                val = float(b[i]) + sum(float(W[i,j])*input_list[j] for j in range(n_input))
            ## Se guarda el output de la neurona
            aux_list.append(val)
            heu_sol[f'x^{l}_{i}'] = val
        ## Se actualiza el input de la siguiente capa   
        input_list = aux_list
    ## Valor objetivo
    heu_obj = aux_list[real_label] - aux_list[target_label]
    return heu_obj,heu_sol

###
###

def create_disjoint_partition(n_input,n_output,n_neurons,n_layers,n_clusters,params,policy = 'equal_lvl',use_bias = False):
    ## Lista para guardar las particiones
    partition = [[] for k in range(n_clusters)]
    ## Politica para la particion
    if policy == 'equal_lvl':
        ## Se recorren las capas
        for l in range(-1,n_layers):
            ## Cantidad de neuronas
            n = n_neurons
            ## Para la capa de entrada
            if l == -1:
                n = n_input
            ## Para la capa de output
            elif l == n_layers-1:
                n = n_output
            ## Duplas de la capa
            layer_duplas = [(l,i) for i in range(n)]           
            ## Cantidad de neuronas a asignar por particion
            size = n//n_clusters
            ## Resto de la division
            remaining = n%n_clusters
            ## Se recorren las particiones
            for k in range(n_clusters):
                ## Indice de incio
                start_idx = k*size+min(k,remaining)
                ## Indice de termino
                end_idx = start_idx + size + (1 if k<remaining else 0)
                ## Se agregan las duplas a la particion 
                partition[k] += layer_duplas[start_idx:end_idx]
    elif policy == 'layer':
        ## Se calcula el tamaño de cada particion
        size = (n_layers+1)//n_clusters
        ## Resto de la division
        remaining = (n_layers+1)%n_clusters
        ## Contador de particiones y capas agregadas
        k = 0
        aux_size = size + (0 if k <n_clusters-remaining else 1)
        ## Se recorren las capas
        for l in range(-1,n_layers):
            ## Neuronas de la capa
            n = n_neurons
            ## Para la capa de entrada
            if l == -1:
                n = n_input
            ## Para la capa de saluda
            elif l == n_layers-1:
                n = n_output
            ## Duplas de la capa
            layer_duplas = [(l,i) for i in range(n)]
            ## Se añaden las duplas a la particion correspondiente
            partition[k] += layer_duplas
            ## Se actualizan los valores
            aux_size -= 1
            if aux_size == 0:
                k += 1
                aux_size = size + (0 if k <n_clusters-remaining else 1)
    elif policy == 'path':
        ## Menor cantidad de neuronas de una capa
        n_min = min(n_input,n_neurons,n_output)
        ## Cantidad de neuronas por particion
        size = (n_min*(n_layers+1))//n_clusters
        ## Grafo
        graph = nx.DiGraph()
        ## Se recorren las capas
        for l in range(-1,n_layers):
            ## Para la capa de entrada
            if l == -1:
                ## Se añaden las aristas del vertice de inicio
                for i in range(n_input):
                    graph.add_edge('s',(l,i),weight = 1)
            ## Para la capa de salida
            if l == n_layers-1:
                ## Se añaden las aristas del vertice de termino
                for i in range(n_output):
                    graph.add_edge((l,i),'t',weight = 1)
            ## Para las capas ocultas
            if -1 < l:
                ## Parametros de la capa
                wname,bname = get_w_b_names(l)
                W,b = params[wname],params[bname]
                ## Neuronas de la capa
                n = W.size()[0]
                ## Neuronas de la capa previa
                n_prev = W.size()[1]
                ## Se recorren las neuronas
                for i in range(n):
                    ## Se recorren las neuronas de la capa previa
                    for j in range(n_prev):
                        ## Caso donde no se considera el bias
                        if not use_bias:
                            distance = float(np.abs(W[i,j]))
                        ## Caso donde se considera el bias
                        else:
                            distance = float(np.abs(W[i,j])) + float(np.abs(b[i]))
                        ## Se añade la arista
                        graph.add_edge((l-1,j),(l,i),weight = distance)
        ## Particion
        k = 0
        ## Se recorre el grafo
        while graph.number_of_nodes() > 2:
            ## Se encuentra el camino mas largo
            longest_path = nx.dag_longest_path(graph,weight='weight')
            ## Se añaden los vertices a la particion
            for node in longest_path:
                if not node in ['s','t']:
                    (l,i) = node
                    partition[k].append((l,i))
                    graph.remove_node((l,i))
            if len(partition[k]) >= size and k<n_clusters-1:
                k += 1 
            
    elif policy == 'greedy':
        ## Se calcula el tamaño de cada particion
        size = (n_input+n_output+(n_layers-1)*n_neurons)//n_clusters
        ## Resto
        remaining = (n_input+n_output+(n_layers-1)*n_neurons)%n_clusters
        ## Lista para guardar las neuronas consideradas
        used = []
        ## Diccionario para guardar los pesos por capa
        edges_dict = {}
        ## Se recorren las capas
        for l in range(n_layers):
            ## Parametros de la capa
            wname,bname = get_w_b_names(l)
            W,b = params[wname],params[bname]
            ## Neuronas de la capa
            n = W.size()[1]
            ## Neuronas de la capa anterior
            n_prev = W.size()[0]
            ## Se recorren las neuronas 
            for i in range(n):
                ## Se recorren las neuronas de la capa anterior
                for j in range(n_prev):
                    ## Se guarda el peso de la arista
                    edges_dict[(l,i,j)] = np.abs(float(W[i,j]))
        ## Se ordena la lista descendentemente
        edges_dict = dict(sorted(edges_dict.items(), key=lambda item: item[1]))
        ## Contador de particiones y neuronas agregadas
        k = n_clusters-1
        aux_size = size + (1 if k<remaining else 0)
        ## Se recorren las aristas y sus valores
        for (l,i,j) in edges_dict:
            ## Se verifica cual neurona ha sido usada
            if not (l,i) in used:
                partition[k] += [(l,i)]
                used.append((l,i))
                aux_size -= 1
            if aux_size > 0 and not (l-1,j) in used:    
                partition[k] += [(l-1,j)]
                used.append((l-1,j))
                aux_size -= 1
            if aux_size == 0:
                k -= 1
                aux_size = size + (1 if k<remaining else 0)
        ## Se arreglan las particiones disconexas
        ...            
    return partition

###
###

def create_edges_p(partition,n_input,n_neurons,n_layers):
    ## Lista para guardar las aristas
    edges_p = []
    ## Se recorren las particiones
    for k in range(len(partition)):
        ## Se recorren las duplas de la particion
        for (l,i) in partition[k]:
            ## Para las capas posteiores a la entrada
            if l > -1:
                ## Neuronas de la capa anterior
                n = n_neurons
                ## Caso primer capa oculta
                if l == 0:
                    n = n_input
                ## Se recorren las neuronas de la capa anterior
                for j in range(n):
                    ## Si la neurona no se encuentra en la misma particion
                    if not (l-1,j) in partition[k]:
                        ## Se agrega la arista
                        edges_p.append((l,i,j))
    return edges_p

###
###

def check_feasiblity(sol,params,bounds,activation = 'relu',tol = 1E-5):
    ## Variable que indica si la solucion es factible
    is_feasible = True
    ## Lista de neuronas que violan o la activacion, o las cotas
    infactibility = {}
    ## Lista para guardar la entrada de la solucion
    input_list = []
    ## Numero de capas
    n_layers = int(len(params)/2)
    ## Numero de neuronas de input
    wname,bname = get_w_b_names(0)
    W,b = params[wname],params[bname]
    n_input = W.size()[1]
    ## Se recorren las variables de entrada
    for i in range(n_input):
        ## Se añade el valor de la variable
        input_list.append(sol[f'x^{-1}_{i}'])
        ## Cotas de la variable
        lb,ub = -bounds[-1][i][0],bounds[-1][i][1]
        ## Se verifican las cotas
        if sol[f'x^{-1}_{i}'] < lb or sol[f'x^{-1}_{i}'] > ub:
            infactibility[(-1,i)] = 'bounds'
            is_feasible = False
    ## Se recorren las capas
    for l in range(n_layers):
        ## Parametros de la capa
        wname,bname = get_w_b_names(l)
        W,b = params[wname],params[bname]
        ## Neuronas de la capa
        n_neurons = W.size()[0]
        ## Neuronas de la capa previa
        n_input = W.size()[1]
        ## Lista para guardar la salida de la capa
        aux_list = []
        ## Se recorren las capas
        for i in range(n_neurons):
            ## Se evalua la funcion lineal
            val = float(b[i]) + sum(float(W[i,j])*input_list[j] for j in range(n_input))
            ## Cotas de la neurona
            lb,ub = -bounds[l][i][0],bounds[l][i][1]
            ## Funcion de activacion
            if activation == 'relu':
                flb,fub = max(0,lb),max(0,ub)
                val = max(0,val)
            ## Se verifica si el valor de la solucion cumple las cotas
            if sol[f'x^{l}_{i}'] < flb or sol[f'x^{l}_{i}'] > fub:
                infactibility[(l,i)] = ['bounds']
                is_feasible = False
            ## Se verifica si el valor de la solucion es correcto
            if np.abs(val-sol[f'x^{l}_{i}']) > tol:
                if (l,i) in infactibility:
                    infactibility[(l,i)].append('activation')
                else:
                    infactibility[(l,i)] = ['activation']
                is_feasible = False
            ## Se añade el valor a la lista auxiliar
            aux_list.append(val)
        ## Se actualiza la entrada de la siguiente capa
        input_list = aux_list
    return is_feasible,infactibility

###
###

def master_iteration(columns,edges_p,real_label,target_label,params,master_times,master_model,theta,cons_dict):
    ## Cantidad de capas
    n_layers = int(len(params)/2)
    ## Se crea el modelo maestro
    master_model,theta,cons_dict = create_master_model(columns,edges_p,real_label,target_label,params,master_model,theta,cons_dict)
    ## Setup del optimizador
    master_model.setPresolve(SCIP_PARAMSETTING.OFF)
    master_model.setHeuristics(SCIP_PARAMSETTING.OFF)
    master_model.disablePropagation()
    master_model.hideOutput()
    ## Se optimiza el modelo maestro
    aux_time = time.time()
    master_model.optimize()
    iter_time = time.time()-aux_time
    ## Se obtiene la solucion objetivo
    print(' Master Status: ',master_model.getStatus())
    master_obj = master_model.getObjVal()
    ## Se obtienen los valores lambda correspondientes
    lambda_params = get_master_lambdas(master_model,edges_p,cons_dict,columns,n_layers,real_label,target_label,lambda_params = None)
    ## Solucion que se obtiene a partir del modelo maestro
    master_sol = make_theta_solution(master_model,theta,columns)
    ## Se añade el tiempo de la iteracion
    master_times.append(iter_time)
    ## Se retorna el modelo maestro, valor objetivo, solucion que implica, los valores lambda
    return master_model,master_obj,master_sol,lambda_params,theta,cons_dict,master_times

###
###

def pricing_iteration(n_clusters,partition,edges_p,lambda_params,bounds,params,hat_x,real_label,target_label,eps,pricing_times,activation='relu',pricing_models=[],pricing_vars={},eta=False):
    ## Se crean los modelos de pricing
    pricing_models,pricing_vars = create_pricing_models(n_clusters,
                                                        partition,
                                                        edges_p,
                                                        lambda_params,
                                                        bounds,
                                                        params,
                                                        hat_x,
                                                        real_label,
                                                        target_label,
                                                        eps,
                                                        activation,
                                                        pricing_models,
                                                        pricing_vars,
                                                        eta)
    ## Tiempo inicial
    iter_time = 0
    ## Se resuelven los modelos
    for model in pricing_models:
        ## Setup
        model.hideOutput()
        model.setPresolve(SCIP_PARAMSETTING.OFF)
        model.setHeuristics(SCIP_PARAMSETTING.OFF)
        model.disablePropagation()
        ## Se optimiza el modelo
        aux_time = time.time()
        model.optimize()
        iter_time += (time.time() - aux_time) 
    ## Se obtiene la solucion general a partir de los modelos de pricing
    pricing_obj,pricing_sol = compute_pricing_sol_obj(pricing_models,pricing_vars,partition)
    ## Se añade el tiempo de la iteracion
    pricing_times.append(iter_time)
    ## Se retornan los modelos de pricing, valor objetivo, solucion que implican, variables de los modelos
    return pricing_models,pricing_obj,pricing_sol,pricing_vars,pricing_times

###
###

def create_heuristic_sols(heu_methods,all_sols,best_sol,master_sol,master_obj,pricing_sol,pricing_obj,params,bounds,real_label,target_label,pricing_models,pricing_vars,partition,activation = 'relu'):
    if 'master_prop' in heu_methods:
        ## Master propagation
        m_prop_obj,m_prop_sol = propagation_heuristic(master_sol,params,real_label,target_label,activation)
        ## Se guarda el mejor valor objetivo de la heuristica master prop
        if len(all_sols['master_prop'])==0: 
            all_sols['master_prop'].append(m_prop_obj)
            best_sol['master_prop'] = m_prop_sol
        else:
            if m_prop_obj < all_sols['master_prop'][-1]:
                all_sols['master_prop'].append(m_prop_obj)
                best_sol['master_prop'] = m_prop_sol
            else:
                all_sols['master_prop'].append(all_sols['master_prop'][-1])
    if 'pricing_prop' in heu_methods:
        ## Pricing propagation
        p_prop_obj,p_prop_sol = propagation_heuristic(pricing_sol,params,real_label,target_label,activation)
        ## Se guarda el valor objetivo de la heuristica pricing propagation
        if len(all_sols['pricing_prop']) == 0:
            all_sols['pricing_prop'].append(p_prop_obj)
            best_sol['pricing_prop'] = p_prop_sol
        else:
            if p_prop_obj < all_sols['pricing_prop'][-1]:
                all_sols['pricing_prop'].append(p_prop_obj)
            else:
                all_sols['pricing_prop'].append(all_sols['pricing_prop'][-1])
    if 'pricing_zfixed' in heu_methods:
        ## Pricing z fixed
        p_zfixed_obj,p_zfixed_sol = pricing_zfixed_heuristic(pricing_models,pricing_vars,partition,params,real_label,target_label,activation)
        ## Se verifica la factibilidad de la solucion
        is_feasible,infactibility = check_feasiblity(p_zfixed_sol,params,bounds,activation)
        ## En caso de ser factible
        if is_feasible:
            ## Se guarda el valor objetivo de la heuristica pricing z-fixed
            if len(all_sols['pricing_zfixed']) == 0:
                all_sols['pricing_zfixed'].append(p_zfixed_obj)
                best_sol['pricing_zfixed'] = p_zfixed_sol
            else:
                if p_zfixed_obj < all_sols['pricing_zfixed'][-1]:
                    all_sols['pricing_zfixed'].append(p_zfixed_obj)
                else:
                    all_sols['pricing_zfixed'].append(all_sols['pricing_zfixed'][-1])
    return all_sols,best_sol

###
###

def re_scale_lambda(master_model,cons_dict,lambda_params,edges_p,columns,n_layers,real_label,target_label):
    ## True si se reescanalan aquellas variables que tienen coeficientes de un solo signo o 0
    flag = False
    ## Diccionario para los nuevos lambda
    new_lambda = {}
    ## Restriccion convexa
    dual_conv = master_model.getDualSolVal(cons_dict['conv'])
    print(f'Dual restriccion conv: {dual_conv}')
    ## Diccionario para los conjuntos de signos
    sign_dict = {'+-':[],'+':[],'-':[]}
    if flag:
        ## Se identifican los conjuntos
        for (l,i,j) in lambda_params:
            ## Nombre de la variable x e y
            x_name = f'x^{l-1}_{j}'
            y_name = f'y^{l-1},{i}_{j}'
            ## Signo de la restriccion
            sign = ''
            ## Se recorren las columnas
            for q in range(len(columns)):
                ## Coeficiente de la columna
                coef = columns[q][y_name]-columns[q][x_name]
                ## Caso variable dual se va a menos infinito
                if coef > 0:
                    if sign in ['','-']:
                        sign = '-'
                    else:
                        sign = '+-'
                        break
                ## Caso variable dual se va a mas infinito
                if coef < 0:
                    if sign in ['','+']:
                        sign = '+'
                    else:
                        sign = '+-'
                        break
            if sign == '':
                sign = '+-'
            ## Se añade la restriccion a un conjunto
            sign_dict[sign].append((l,i,j))
            ## Se setea el valor de la nueva solucion
            if sign == '+-':
                new_lambda[(l,i,j)] = -lambda_params[(l,i,j)]
            else:
                new_lambda[(l,i,j)] = 0 
    else:
        for edge in lambda_params:
            value = -lambda_params[edge]
            ## Se verifica si el valor es mas o menos infinito
            if value + 1 == value:
                if value > 0:
                    sign_dict['+'].append(edge)
                elif value < 0:
                    sign_dict['-'].append(edge)
                new_lambda[edge] = 0
            else:
                new_lambda[edge] = value
    #print(sign_dict)
    ## Se setean las variables duales que se van a mas infinito
    for (l,i,j) in sign_dict['+']:
        ## Valores candidatos para la variable dual
        J_values = []
        ## Se recorren las variables del primal
        for q in range(len(columns)):
            lhs = dual_conv
            ## Se recorren las restricciones originales
            for (aux_l,aux_i,aux_j) in lambda_params:
                ## Nombre de la variable x e y
                x_name = f'x^{aux_l-1}_{aux_j}'
                y_name = f'y^{aux_l-1},{aux_i}_{aux_j}'
                ## Coeficiente de la variable en la restriccion
                coef = columns[q][y_name]-columns[q][x_name]
                lhs += coef*new_lambda[(aux_l,aux_i,aux_j)]
            ## Coeficiente del objetivo de la variable
            xr_name = f'x^{n_layers-1}_{real_label}'
            xt_name = f'x^{n_layers-1}_{target_label}'
            obj_coef = columns[q][xr_name] - columns[q][xt_name]
            ## Caso en que la restriccion es violada
            if lhs > obj_coef:
                ## Coeficiente de la variable primal en la restriccion
                x_name = f'x^{l-1}_{j}'
                y_name = f'y^{l-1},{i}_{j}'
                coef = columns[q][y_name] - columns[q][x_name]
                if coef < 0:
                    value = (obj_coef-(lhs))/coef
                    J_values.append(value)
        ## Se actualiza el valor de lambda
        if len(J_values) > 0:
            new_lambda[(l,i,j)] = max(J_values)
    ## Se setean las variables duales que se van a menos infinito
    for (l,i,j) in sign_dict['-']:
        ## Valores candidatos para la variable dual
        J_values = []
        ## Se recorren las variables del primal
        for q in range(len(columns)):
            lhs = dual_conv
            ## Se recorren las restricciones originales
            for (aux_l,aux_i,aux_j) in lambda_params:
                ## Nombre de la variable x e y
                x_name = f'x^{aux_l-1}_{aux_j}'
                y_name = f'y^{aux_l-1},{aux_i}_{aux_j}'
                ## Coeficiente de la variable en la restriccion
                coef = columns[q][y_name]-columns[q][x_name]
                lhs += coef*new_lambda[(aux_l,aux_i,aux_j)]
            ## Coeficiente del objetivo de la variable
            xr_name = f'x^{n_layers-1}_{real_label}'
            xt_name = f'x^{n_layers-1}_{target_label}'
            obj_coef = columns[q][xr_name] - columns[q][xt_name]
            ## Caso en que la restriccion es violada
            if lhs > obj_coef:
                ## Coeficiente de la variable primal en la restriccion
                x_name = f'x^{l-1}_{j}'
                y_name = f'y^{l-1},{i}_{j}'
                coef = columns[q][y_name] - columns[q][x_name]
                if coef > 0:
                    value = (obj_coef-(lhs))/coef
                    J_values.append(value)
        ## Se actualiza el valor de lambda
        if len(J_values) > 0:
            new_lambda[(l,i,j)] = min(J_values)
    for edge in new_lambda:
        new_lambda[edge] = -new_lambda[edge]
    return new_lambda
