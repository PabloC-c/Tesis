import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from functions import *

def trainer(model, device, train_loader, optimizer, epoch, print_loss = False,regul_L = 'L1',L_lambda = 0.005):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        ## Se envian los datos al dispositivo de entrenamiento
        data, target = data.to(device), target.to(device)
        ## Se resetean los gradientes de los parametros
        optimizer.zero_grad()
        ## Se calculan los respectivos outputs
        output = model(data)
        ## Se calcula la perdida
        loss = F.cross_entropy(output, target)
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
batch_size = 16
learning_rate = 0.05
epochs = 50
weight_decay = 0.005
print_loss = False
regul_L = 'L2'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## Redes a entrenar
neuron_list = [5,10]
layer_list  = [2,3,4]
activation_list = ['sigmoid']

## Se descargan los datos de MNIST
train_dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

net_dict = {}
batch_size_og = batch_size
for activation in activation_list:
    for n_neurons in neuron_list:
        for n_layers in layer_list:
            if regul_L == 'L1':
                filename = "nn_parameters_L1/{}_model_weights_L{}_n{}.pth".format(activation,n_layers, n_neurons)
            else:
                filename = "nn_parameters/{}_model_weights_L{}_n{}.pth".format(activation,n_layers, n_neurons)
            if True:#os.path.exists(filename):
                if os.path.exists(filename):
                    ## Se crea la red
                    net = neural_network(n_neurons,n_layers,activation)
                    ## Cargar los parámetros de la red
                    net.load_state_dict(torch.load('nn_parameters/{}_model_weights_L{}_n{}.pth'.format(activation,n_layers, n_neurons)))
                    params = net.state_dict()
                    filtered_params = filter_params(params)
                    ## Se cargan los datos de testeo
                    test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())
                    test_loader  = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=256, shuffle=False)
                    acc = calculate_accuracy(test_loader, net)
                    if acc >= 0.85:
                        break
                print('\n ===== Capas: ',n_layers,' Neuronas: ',n_neurons,' Activacion: ',activation,'===== \n')
                acc = 0
                while (acc<0.9 and batch_size < 2048 ):
                    print('intento con batch size :',batch_size)
                    ## Se crean los dataloaders para el manejo de los datos durante el entrenamiento
                    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
                    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=256, shuffle=False)
                    ## Se genera el modelo
                    net = neural_network(n_neurons, n_layers, activation).to(device)
                    ## Se setea el optimizador
                    if regul_L == 'L1':
                        optimizer = optim.Adam(net.parameters(), lr=0.001)
                    else:
                        optimizer = optim.AdamW(net.parameters(), lr=0.001, weight_decay=0.005)
                    # optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
                    ## Se entrena la red
                    try:
                        for epoch in range(1, epochs + 1):
                            trainer(net, device, train_loader, optimizer, epoch,print_loss = print_loss,regul_L = regul_L,L_lambda = 0.005)
                            if epoch%10 == 0:
                                acc = tester(net, device, test_loader)
                                if acc > 0.9:
                                    break
                        trained = True
                    except:
                        trained = False
                        acc = 0
                    if batch_size == batch_size_og:
                        net_dict[(n_layers,n_neurons)] = (net,acc)                        
                    elif acc > net_dict[(n_layers,n_neurons)][1]:
                            net_dict[(n_layers,n_neurons)] = (net,acc)
                    batch_size = batch_size * 2
                print('\t Precisión en el conjunto de prueba: {} %'.format(100 * acc))
                # Guardar los parámetros de la red
                ## Se mueve el modelo a la CPU
                net = net.to("cpu")
                torch.save(net.state_dict(), filename)
                batch_size = batch_size_og