import torch
import numpy as np
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from scipy.special import softmax 
from functions import *

test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(),download=True)
test_loader  = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

## Obtener un solo par de entrada y salida correspondiente del conjunto de datos de entrenamiento
input_example, output_example = next(iter(test_loader))

## Acceder al tensor de entrada con los valores de los píxeles
choosen_digit = 1
image_list = input_example[output_example == choosen_digit][0].view(-1,784).tolist()[0]#input_example[0].view(-1,784).tolist()[0]
image_input = np.array(image_list).reshape(28, 28)
output_target = 9

## Crear la instancia de la red neuronal
n_neurons = 10
n_layers = 2
activation = 'relu'
exact = 'exact' #prop: propagacion, exact: exacto, no_exact: formulaciones o envolturas
apply_softmax = False 
tol_distance = 0.1
apply_bounds = True
minutes = 15
net = neural_network(n_neurons,n_layers,activation)

## Cargar los parámetros de la red
net.load_state_dict(torch.load('nn_parameters/'+activation+"_"+"model_weights_L{}_n{}.pth".format(n_layers, n_neurons)))
params = net.state_dict()
filtered_params = filter_params(params)

## Se cargan las cotas de la red
bounds = read_bounds(n_layers,n_neurons,activation)

image = torch.tensor(image_list).view(1, 28, 28)
aux   = net(image).tolist()[0]

verif_model,all_vars = create_verification_model(params,bounds,activation,tol_distance,apply_softmax,image_list,output_target,choosen_digit,exact,apply_bounds)
verif_model.redirectOutput()
verif_model.setParam('limits/time', int(60*minutes))
verif_model.optimize()


solution = [verif_model.getVal(all_vars['h{},{}'.format(-1,i)]) for i in range(len(image_list))]
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
