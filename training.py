import torch
import os
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from functions import *

# Transformación para normalizar los datos
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Cargar los datos de entrenamiento y validación de MNIST
trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

# Crear la instancia de la red neuronal
neuron_list = [10,25,50]
layer_list  = [2,3]
activation = 'relu'
for n_neurons in neuron_list:
    for n_layers in layer_list if n_neurons != 50 else [2]:
        if True:#not os.path.exists(activation + "_" + "model_weights_L{}_n{}.pth".format(n_layers, n_neurons)):
            print('\n Capas: ',n_layers,' Neuronas: ',n_neurons,'\n')
            net = training(activation,n_layers,n_neurons,trainset,trainloader,testset,testloader)
            # Guardar los parámetros de la red
            torch.save(net.state_dict(), activation + "_" + "model_weights_L{}_n{}.pth".format(n_layers, n_neurons)) 
