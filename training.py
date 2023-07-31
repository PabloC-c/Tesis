import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from functions import *

def trainer(model, device, train_loader, optimizer, epoch, l1_lambda=0.001, print_loss = False):
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

        ## Se aplica regularización L1 a todos los parámetros (pesos y sesgos) excepto las capas de salida
        l1_reg = torch.tensor(0., device=device)
        for param in model.parameters():
            if param.requires_grad and len(param.shape) > 1:  # Excluir capas de salida
                l1_reg += torch.norm(param, p=1)
        
        ## Se suma la penalizacion
        loss += l1_lambda * l1_reg
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


## Configuración del entrenamiento
batch_size = 16
learning_rate = 0.001
epochs = 40
weight_decay = 0.005
print_loss = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## Redes a entrenar
neuron_list = [5]
layer_list  = [4]
activation_list = ['relu','softplus','sigmoid']

## Se descargan los datos de MNIST
train_dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

net_dict = {}
batch_size_og = batch_size
for activation in activation_list:
    for n_neurons in neuron_list:
        for n_layers in layer_list:
            if os.path.exists("nn_parameters_L1/{}_model_weights_L{}_n{}.pth".format(activation,n_layers, n_neurons)):
                print('\n ===== Capas: ',n_layers,' Neuronas: ',n_neurons,' Activacion: ',activation,'===== \n')
                acc = 0
                while (acc<0.9 and batch_size <= 1024 ):
                    print('intento con batch size :',batch_size)
                    ## Se crean los dataloaders para el manejo de los datos durante el entrenamiento
                    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
                    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=128, shuffle=False)
                    ## Se genera el modelo
                    net = neural_network(n_neurons, n_layers, activation).to(device)
                    ## Se setea el optimizador
                    optimizer = optim.Adam(net.parameters(), lr=0.001)
                    # optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
                    ## Se entrena la red
                    try:
                        for epoch in range(1, epochs + 1):
                            trainer(net, device, train_loader, optimizer, epoch,l1_lambda = 0.001,print_loss = print_loss)
                            if epoch%10 == 0:
                                acc = tester(net, device, test_loader)
                                if acc >= 0.95:
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
                torch.save(net.state_dict(), "nn_parameters_L1/{}_model_weights_L{}_n{}.pth".format(activation,n_layers, n_neurons))
                batch_size = batch_size_og