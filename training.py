import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from functions import *

def trainer(n_neurons,n_layers,activation,batch_size,learning_rate,epochs,weight_decay = 0.005,print_training_loss = True):
    ## Se verifica si se puede utilizar la GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ## Se descargan los datos de MNIST
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

    ## Se crean los dataloaders para el manejo de los datos durante el entrenamiento
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    ## Se crea el modelo
    model = neural_network(n_neurons,n_layers,activation).to(device)

    ## Se define la función de pérdida y el optimizador con regularización L2
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    ## Entrenamiento
    total_step = len(train_loader)
    for epoch in range(epochs):
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            ## Se envian los datos al dispositivo de entrenamiento
            images = images.to(device)
            labels = labels.to(device)

            ## Forward
            outputs = model(images)
            loss = criterion(outputs, labels)

            ## Backward y optimización
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            ## Calculo de perdida
            if (i+1) % 100 == 0 and print_training_loss:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch+1, epochs, i+1, total_step, loss.item()))
                
    ## Se mueve el modelo a la CPU
    model = model.to("cpu")
    ## Se evalua en el conjunto de prueba
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        acc = correct / total
    return model,acc

## Configuración del entrenamiento
batch_size = 16
learning_rate = 0.001
epochs = 40
weight_decay = 0.005
print_training_loss = True

## Redes a entrenar
neuron_list = [10,25,50]
layer_list  = [2,3,4]
activation = 'sigmoid'

acc_dict = {}
batch_size_og = batch_size
for n_neurons in neuron_list:
    for n_layers in layer_list:
        if not os.path.exists('nn_parameters/'+activation + "_" + "model_weights_L{}_n{}.pth".format(n_layers, n_neurons)):
            acc = 0
            while (acc<0.8 and batch_size <= 1024 ):
                print('\n ===== Capas: ',n_layers,' Neuronas: ',n_neurons,'===== \n')
                net,acc = trainer(n_neurons,n_layers,activation,batch_size,learning_rate,epochs,weight_decay,print_training_loss)
                if acc < 0.9:
                    batch_size = batch_size * 2
            acc_dict[(n_layers,n_neurons)] = acc
            print('\t Precisión en el conjunto de prueba: {} %'.format(100 * acc))
            # Guardar los parámetros de la red
            torch.save(net.state_dict(), 'nn_parameters/' + activation + "_" + "model_weights_L{}_n{}.pth".format(n_layers, n_neurons))
            batch_size = batch_size_og
