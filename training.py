import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
from torchvision import datasets, transforms
from functions import *

#torch.manual_seed(5)
#torch.cuda.manual_seed_all(5)

def trainer(model, device, train_loader, optimizer, num_epochs, print_loss = False,regul_L = 'L1',L_lambda = 0.005):
    criterion = nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            ## Se envian los datos al dispositivo de entrenamiento
            data, target = data.to(device), target.to(device)
            ## Se resetean los gradientes de los parametros
            optimizer.zero_grad()
            ## Se calculan los respectivos outputs
            output = model(data)
            ## Se calcula la perdida
            loss = criterion(output, target)
            ## Se calculan los gradientes
            loss.backward()
            ## Se actualizan los parametros de la red
            optimizer.step()
            
            if print_loss and batch_idx % 100 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')


def tester(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({accuracy:.2f}%)')
    return accuracy

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

## Configuración del entrenamiento
batch_size = 64
max_batch_size = 512
learning_rate = 0.05
num_epochs = 100      ##ESTABA EN 50
weight_decay = 0.0001 ##ESTABA EN 0.005
print_loss = False
regul_L = 'L2'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## Redes a entrenar
neuron_list = [2]
layer_list  = [2]
activation_list = ['relu']

## Se descargan los datos de MNIST
mnist = False
if mnist:
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())
else:
    ## Tamaño de entrada y salida
    n_input = 2
    n_output = 2
    ## Cantidad de instancias
    dataset_size = 15000
    ## Porcentaje de datos para entrenamiento
    train_proportion = 0.6
    ## Se generan las instancias
    data = torch.rand(dataset_size, 2)
    ## Se generan las etiquetas de los datos
    labels = (data[:, 0] + data[:, 1] <1).long()
    print(labels.float().mean())
    ## Se crea el dataset
    dataset = torch.utils.data.TensorDataset(data, labels)
    ## Tamaño de los conjuntos de entrenamiento y prueba
    train_size = int(train_proportion * dataset_size)
    test_size = dataset_size - train_size
    ## Se dividen los conjuntos
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
net_dict = {}
batch_size_og = batch_size
for activation in activation_list:
    for n_neurons in neuron_list:
        for n_layers in layer_list:
            if mnist:
                if regul_L == 'L1':
                    filename = "nn_parameters_L1/{}_model_weights_L{}_n{}.pth".format(activation,n_layers, n_neurons)
                else:
                    filename = "nn_parameters/{}_model_weights_L{}_n{}.pth".format(activation,n_layers, n_neurons)
            else:
                if regul_L == 'L1':
                    filename = "nn_parameters_L1/{}_toy_ex_L{}_n{}.pth".format(activation,n_layers, n_neurons)
                else:
                    filename = "nn_parameters/{}_toy_ex_L{}_n{}.pth".format(activation,n_layers, n_neurons)
            if True:#os.path.exists(filename):
                if os.path.exists(filename):
                    ## Se crea la red
                    if mnist:
                        net = neural_network(n_neurons,n_layers,activation)
                    else:
                        net = neural_network(n_neurons,n_layers,activation,n_input,n_output)
                    ## Cargar los parámetros de la red
                    net.load_state_dict(torch.load(filename))
                    params = net.state_dict()
                    filtered_params = filter_params(params)
                    ## Se cargan los datos de testeo
                    test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())
                    test_loader  = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=256, shuffle=False)
                    acc = calculate_accuracy(test_loader, net)
                    if acc >= 0.95:
                        break
                    elif acc > 0.55:
                        og_acc = acc
                        re_train = True
                        net_dict[(n_layers,n_neurons)] = (net,acc)
                    else:
                        re_train = False
                        acc = 0
                else:
                    acc = 0
                    re_train = False
                print('\n ===== Capas: ',n_layers,' Neuronas: ',n_neurons,' Activacion: ',activation,'===== \n')
                while (acc<0.9 and batch_size <= max_batch_size ):
                    print('intento con batch size :',batch_size)
                    ## Se crean los dataloaders para el manejo de los datos durante el entrenamiento
                    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
                    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
                    ## Se genera el modelo
                    if mnist:
                        net = neural_network(n_neurons,n_layers,activation)
                    else:
                        net = neural_network(n_neurons,n_layers,activation,n_input,n_output)
                    if re_train:
                        net.load_state_dict(torch.load(filename))
                    net.to(device)
                    ## Se setea el optimizador
                    if regul_L == 'L1':
                        optimizer = optim.Adam(net.parameters(), lr=learning_rate)
                    else:
                        optimizer = optim.AdamW(net.parameters(), lr=learning_rate, weight_decay = weight_decay)
                    ## Se entrena la red
                    try:
                        trainer(net, device, train_loader, optimizer, num_epochs, print_loss = print_loss,regul_L = regul_L,L_lambda = weight_decay)
                        acc = tester(net, device, test_loader)
                        if acc > 0.9:
                            break
                        trained = True
                    except:
                        print('\n \t Siguiente size  \t')
                        trained = False
                        acc = 0
                    if not (n_layers,n_neurons) in net_dict:
                        net_dict[(n_layers,n_neurons)] = (net,acc)                        
                    elif acc > net_dict[(n_layers,n_neurons)][1]:
                            net_dict[(n_layers,n_neurons)] = (net,acc)
                    ## Se mueve el modelo a la CPU
                    net = net.to("cpu")
                    batch_size = batch_size * 2
                print('\t Precisión en el conjunto de prueba: {} %'.format(100 * acc))
                # Guardar los parámetros de la red
                if not re_train or (re_train and (og_acc < net_dict[(n_layers,n_neurons)][1])):
                    final_net = net_dict[(n_layers,n_neurons)][0]
                    torch.save(final_net.state_dict(), filename)
                    print('Parametros guardados')
                batch_size = batch_size_og