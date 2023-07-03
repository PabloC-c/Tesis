import torch
import os
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from scipy.special import softmax 
from functions import *

## Transformación para normalizar los datos
transform = transforms.Compose([
    transforms.ToTensor()
])

## Cargar los datos de entrenamiento y validación de MNIST
trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64)#, shuffle=True)

testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64)#, shuffle=True)

## Obtener un solo par de entrada y salida correspondiente del conjunto de datos de entrenamiento
input_example, output_example = next(iter(trainloader))

## Acceder al tensor de entrada con los valores de los píxeles
choosen_digit = 1
input_value = input_example[output_example == choosen_digit][0].view(-1,784).tolist()[0]#input_example[0].view(-1,784).tolist()[0]
image_input = np.array(input_value).reshape(28, 28)
output_target = 9

## Crear la instancia de la red neuronal
n_neurons = 10
n_layers = 2
activation = 'relu'
net = neural_network(n_neurons,n_layers,activation)

## Cargar los parámetros de la red
net.load_state_dict(torch.load(activation+"_"+"model_weights_L{}_n{}.pth".format(n_layers, n_neurons)))
params = net.state_dict()
filtered_params = filter_params(params)

bounds,layers_time,net_model,net_input_var,net_output_var,all_vars = calculate_bounds(params,activation)

#solution = calculate_variables(net_model,input_value,filtered_params,all_vars,activation)

#flag = net_model.trySol(solution)
image = torch.tensor(input_value).view(1, 28, 28)
aux   = net(image).tolist()[0]
apply_softmax = False
tol_distance = 0.209
if apply_softmax:
    aux   = softmax(np.array(aux))
output_value = aux[choosen_digit]

net_model = create_verification_model(net_model,net_input_var,net_output_var,input_value,choosen_digit,output_value,output_target,params,bounds,tol_distance, apply_softmax)
net_model.redirectOutput()
net_model.optimize()

solution = [net_model.getVal(all_vars['h{},{}'.format(-1,i)]) for i in range(len(input_value))]
image_solution = np.array(solution).reshape(28, 28)

# Crea una figura con dos subplots
fig, axs = plt.subplots(1, 3)
color_map = 'gray'
input_lb=0 
input_ub= 1

axs[0].imshow(image_input, vmin = input_lb, vmax = input_ub,cmap=color_map)
axs[0].axis('off')

axs[1].imshow(image_solution, vmin = input_lb, vmax = input_ub, cmap=color_map)
axs[1].axis('off')

axs[2].imshow(np.abs(image_solution-image_input), vmin = input_lb, vmax = input_ub, cmap=color_map) #np.abs(image_solution-image_input)
axs[2].axis('off')

# Ajusta el espaciado entre los subplots
plt.tight_layout()

# Muestra la figura con las dos imágenes
plt.show()
