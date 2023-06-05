import torch
import os
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from functions import *

## Transformación para normalizar los datos
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

## Cargar los datos de entrenamiento y validación de MNIST
trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

## Obtener un solo par de entrada y salida correspondiente del conjunto de datos de entrenamiento
input_example, output_example = next(iter(trainloader))

## Acceder al tensor de entrada con los valores de los píxeles
input_value = input_example[0].view(-1,784).tolist()[0]#.view(-1, 784).tolist()

## Acceder al valor de salida correspondiente
output_value = output_example[0]#.item()#.item()

## Crear la instancia de la red neuronal
activation = 'relu'
net = neural_network(10,2,activation)

## Cargar los parámetros de la red
net.load_state_dict(torch.load(activation+"_"+"model_weights_L{}_n{}.pth".format(2, 10)))
params = net.state_dict()
filtered_params = filter_params(params)

bounds,layers_time,net_model,net_input_var,net_output_var,all_vars = calculate_bounds(params,activation)

solution = calculate_variables(net_model,input_value,filtered_params,all_vars,activation)

#flag = net_model.trySol(solution)

#net_model = create_verification_model(net_model,net_input_var,net_output_var,input_value,output_value,params,bounds,tol_distance = 1e-6)
#net_model.redirectOutput()
#net_model.optimize()