import torch
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
n_neurons = 100
n_layers  = 0
net = Relu_net(n_neurons,n_layers)

# Definir la función de pérdida y el optimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, weight_decay=0.001)

# Definir el factor de regularización L1
l1_lambda = 0.005

# Entrenar la red neuronal
for epoch in range(10):  # Correr el conjunto de datos de entrenamiento 10 veces
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # Obtener los datos de entrada y las etiquetas
        inputs, labels = data

        # Reestablecer los gradientes a cero
        optimizer.zero_grad()

        # Propagar hacia adelante, calcular la pérdida y retropropagar el error
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        
        # Calcular la penalización L1 y agregarla a la pérdida
        l1_reg = 0
        for param in net.parameters():
            l1_reg += torch.sum(torch.abs(param))
        loss += l1_lambda * l1_reg
        
        loss.backward()
        optimizer.step()

        # Imprimir estadísticas de entrenamiento
        running_loss += loss.item()
        if i % 100 == 99:    # Imprimir cada 100 mini-lotes procesados
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

print('Entrenamiento finalizado')

# Evaluar la precisión de la red neuronal en el conjunto de datos de prueba
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        inputs, labels = data
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Precisión en el conjunto de prueba: %d %%' % (100 * correct / total))

# Guardar los parámetros de la red
torch.save(net.state_dict(), "model_weights_L{}_n{}.pth".format(n_layers, n_neurons))

# Cargar los parámetros de la red
net.load_state_dict(torch.load("model_weights_L{}_n{}.pth".format(n_layers, n_neurons)))