import os
import sys
import time
import torch
import torch.nn as nn
import pandas as pd
from pyscipopt import Model,quicksum
from collections import OrderedDict
from torchvision import datasets, transforms
from functions import *

activation_list = ['sigmoid']
layer_list = [2,3,4] 
neuron_list = [5,10]
exact = 'no_exact'
apply_bounds = True
type_bounds = 'verif_bounds_prop'
minutes = 10
filter_tol = 1e-5

tol_distance = 0.01

if len(sys.argv) > 1:
    activation_list = [sys.argv[1]]
    if len(sys.argv) >= 3:
        layer_list = [int(sys.argv[2])]
    if len(sys.argv) >= 4:
        neuron_list = [int(sys.argv[3])]
    if len(sys.argv) >= 5:
        exact = sys.argv[4]
    if len(sys.argv) >= 6:
        minutes = float(sys.argv[5])    
    if len(sys.argv) >= 7:
        filter_tol = float(sys.argv[6])

def create_layer_bound_problem(l,params,bounds,activation,exact,apply_bounds,add_verif_bounds,tol_distance,image_list):
    ## Se inicializa el modelo de verificacion
    layer_model,bounds,inpt,all_vars = initialize_neuron_model(bounds,add_verif_bounds,tol_distance,image_list)
    ## Se genera la evaluacion de la red
    ## Se recorren las capas
    for j in range(l):
        ## Se crean las restricicones y variables
        layer_model,aux_input,all_vars = update_neuron_model(layer_model,inpt,all_vars,params,bounds,j,activation,exact,apply_bounds)
        inpt = aux_input
    ## Parametros de la capa l   
    weight,bias = get_w_b_names(l)
    W,b = params[weight],params[bias]
    ## Neuronas capa l
    n_neurons = W.size()[0]
    ## Arreglo para guardar la evaluacion lineal de la capa
    layer_lin_eval = []
    ## Se recorren las neuronas de la capa l
    for i in range(n_neurons):
        ## Variable de la evaluacion del input en la funcion lineal
        z = layer_model.addVar(lb = None, ub = None,vtype = 'C', name = 'z{},{}'.format(l,i))
        ## Restriccion de evaluacion con la funcion lineal
        layer_model.addCons(quicksum(float(W[i,k])*inpt[k] for k in range(len(inpt))) + float(b[i]) == z, name = 'eval{},{}'.format(l,i))
        layer_lin_eval.append(z)
    ## Se genera la funcion objetivo
    layer_model.setObjective(quicksum(layer_lin_eval[i] for i in range(n_neurons)), 'maximize')
    ## Se retorna el modelo de verificacion
    return layer_model,all_vars

if type_bounds in ['verif_bounds','verif_bounds_prop']:
    add_verif_bounds = True
else:
    add_verif_bounds = False

if add_verif_bounds:
    real_output = 1
    ## Se cargan las imagenes
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(),download=True)
    test_loader  = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)
    ## Se selecciona un solo par de entrada y salida correspondiente del conjunto de datos
    input_example, output_example = next(iter(test_loader))
    ## Se transforma el input en una lista
    image_list = input_example[output_example == real_output][0].view(-1,784).tolist()[0]
else:
    image_list = []
       
for activation in activation_list:
    if type_bounds in ['verif_bounds','verif_bounds_prop']:
        data_file = 'layers_bounds/{}_{}_{}_tolper{}.xlsx'.format(activation,exact,type_bounds,int(100*tol_distance))
    else:
        data_file = 'layers_bounds/{}_{}_{}.xlsx'.format(activation,exact,type_bounds)
    ## Caso donde ya existe un df con datos previos
    if os.path.exists(data_file):
        df = pd.read_excel(data_file,header=None)
    ## Caso en que no hay datos previos
    else:
        df = pd.DataFrame()
    ## Se recorren las neuronas y la capas
    for n_neurons in neuron_list:
        for n_layers in layer_list:
            new_line = [n_neurons,n_layers]
            ## Se calculan las cotas en caso de no haber
            if True:#not os.path.exists(bounds_file):
                print('\n Capas: ',n_layers,' Neuronas: ',n_neurons,'\n')
                ## Se crea la instancia de la red neuronal
                net = neural_network(n_neurons,n_layers)
                ## Se cargan los parámetros de la red
                net.load_state_dict(torch.load('nn_parameters/{}_model_weights_L{}_n{}.pth'.format(activation,n_layers,n_neurons)))
                ## Se guardan en params los parametos de la red
                params = net.state_dict()
                ## Se filtran los parametros de la red
                filtered_params = filter_params(params,filter_tol)
                ## Archivo que contiene las cotas
                bounds_file = calculate_bounds_file_name(type_bounds,activation,n_layers,n_neurons,tol_distance,1)
                ## Se pre-cargan las cotas en caso de necesitarlas
                bounds = read_bounds(apply_bounds,n_layers,n_neurons,activation,bounds_file)
                real_bounds = read_bounds(True,n_layers,n_neurons,activation,bounds_file) 
                ## Lista para guardar la cota de cada capa
                layers_bounds = []
                new_line = []
                ## Se recorren las capas de la red
                for l in range(n_layers):
                    layer_model,all_vars = create_layer_bound_problem(l,params,bounds,activation,exact,apply_bounds,add_verif_bounds,tol_distance,image_list)
                    ## Se limita el tiempo de resolucion
                    layer_model.setParam('limits/time', int(60*minutes))
                    ## Se aumenta la tolerancia de factibilidad
                    layer_model.setParam('numerics/feastol', 1E-5)
                    ## Se optimiza el modelo en busca del ejemplo adversarial
                    t0 = time.time()
                    try:
                        aux_t = time.time()
                        layer_model.optimize()
                        dt = time.time() - aux_t
                    except:
                        dt = time.time() - t0
                    model_status = layer_model.getStatus()
                    print('\n Status final del modelo:',model_status,'\n')
                    if model_status == 'optimal':
                        obj_val = layer_model.getObjVal()
                    else:
                        obj_val = layer_model.getDualbound()
                    bounds_sum = sum(bounds[l][i][1] for i in range(len(bounds[l])))
                    layers_bounds.append([obj_val,bounds_sum])
                    new_line.append(obj_val)
                    new_line.append(bounds_sum)
                if len(layer_list) > 1:
                    while len(new_line) < (2+2*max(layer_list)):
                        new_line.append('-')
                df = df._append(pd.Series(new_line), ignore_index=True)
                ## Se intenta escribir en el archivo del df
                written = False
                while not written:
                    try:
                        df.to_excel(data_file, header = False, index = False)
                        written = True
                    except:
                        time.sleep(5)
            