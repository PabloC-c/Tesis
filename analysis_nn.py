import os
import torch
from torchvision import datasets, transforms
from functions import *

def calculate_accuracy(test_loader,model):
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        acc = correct / total
    return acc

## Redes a chequear
neuron_list = [10]
layer_list  = [2]
activation = 'relu'

batch_size = 64

for n_neurons in neuron_list:
        for n_layers in layer_list:
            if os.path.exists('nn_parameters/{}_model_weights_L{}_n{}.pth'.format(activation,n_layers, n_neurons)):
                ## Se crea la red
                net = neural_network(n_neurons,n_layers,activation)
                ## Cargar los parámetros de la red
                net.load_state_dict(torch.load('nn_parameters/{}_model_weights_L{}_n{}.pth'.format(activation,n_layers, n_neurons)))
                params = net.state_dict()
                filtered_params = filter_params(params)
                ## Se cargan los datos de testeo
                test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())
                test_loader  = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
                acc = calculate_accuracy(test_loader, net)
                print('Capas {}, Neuronas {}, Precision {:.4f}%'.format(n_layers,n_neurons,100*acc))
                print(params)