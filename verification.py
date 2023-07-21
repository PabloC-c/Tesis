import torch
import sys
import numpy as np
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from scipy.special import softmax 
from functions import *

if len(sys.argv) == 9:
  ## Capas, neuronas, activacion, tolerancia, formulacion, applicar bounds, aplicar softmax, minutos
  _, n_layers, n_neurons, activation, tol_distance, exact, apply_bounds, apply_softmax, minutes = sys.argv
  apply_softmax = bool(apply_softmax)
  minutes = float(minutes)
elif len(sys.argv) == 7:
  ## Capas, neuronas, activacion, tolerancia, formulacion, applicar bounds
  _, n_layers, n_neurons, activation, tol_distance, exact, apply_bounds = sys.argv
  apply_softmax, minutes = False, 15
else:
  n_layers = 2
  n_neurons = 10
  activation = 'relu'
  tol_distance = '0.05'
  exact = 'no_exact'        # exact{prop: propagacion, exact: exacto, no_exact: formulaciones o envolturas}
  apply_bounds = True
  apply_softmax = False
  minutes = 15
  
if len(sys.argv) > 0:
  n_layers = int(n_layers)
  n_neurons = int(n_neurons)
  tol_distance = float(tol_distance)
  apply_bounds = bool(apply_bounds)

## Crear la instancia de la red neuronal
net = neural_network(n_neurons,n_layers,activation)

test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(),download=True)
test_loader  = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

## Obtener un solo par de entrada y salida correspondiente del conjunto de datos de entrenamiento
input_example, output_example = next(iter(test_loader))

## Acceder al tensor de entrada con los valores de los píxeles
choosen_digit = 1
image_list = input_example[output_example == choosen_digit][0].view(-1,784).tolist()[0]#input_example[0].view(-1,784).tolist()[0]
image_input = np.array(image_list).reshape(28, 28)
output_target = 9

## Cargar los parámetros de la red
net.load_state_dict(torch.load('nn_parameters/{}_model_weights_L{}_n{}.pth'.format(activation,n_layers, n_neurons)))
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

# Guarda la figura con las imágenes en un archivo
plt.savefig('imagen_resultado.png', dpi=300, bbox_inches='tight')

# Muestra la figura con las dos imágenes
#plt.show()
