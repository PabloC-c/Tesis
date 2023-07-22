import os
import sys
import pandas as pd
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from functions import *

activation_list = ['relu']
layer_list = [2]
neuron_list = [50]
exact = 'no_exact'        # exact{prop: propagacion, exact: exacto, no_exact: formulaciones o envolturas}
apply_bounds = True
minutes = 15
save_image = False
apply_softmax = False

if len(sys.argv) > 1:
    activation_list = [sys.argv[1]]
    if len(sys.argv) >= 3:
        layer_list = [int(sys.argv[2])]
    if len(sys.argv) >= 4:
        neuron_list = [int(sys.argv[3])]
    if len(sys.argv) >= 5:
        exact = sys.argv[4]
    if len(sys.argv) >= 6:
        apply_bounds = bool(sys.argv[5])
    if len(sys.argv) >= 7:
        save_image = bool(sys.argv[6])
    if len(sys.argv) >= 8:    
        apply_softmax = bool(sys.argv[7])

## Se cargan las imagenes
test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(),download=True)
test_loader  = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

## Obtener un solo par de entrada y salida correspondiente del conjunto de datos de entrenamiento
input_example, output_example = next(iter(test_loader))

## Acceder al tensor de entrada con los valores de los píxeles
real_output = 1
target_output = 8
image_list = input_example[output_example == real_output][0].view(-1,784).tolist()[0]#input_example[0].view(-1,784).tolist()[0]
image_input = np.array(image_list).reshape(28, 28)
input_lb =0 
input_ub = 1

tol_0 = 0.01
tol_f = 0.05
tol_step = 0.005

steps = int((tol_f-tol_0)/tol_step)

for activation in activation_list:
    if os.path.exists('datos_verificacion_{}.xlsx'.format(activation)):
        df = pd.read_excel('datos_verificacion_{}.xlsx'.format(activation),header=None)
    else:
        df = pd.DataFrame()
    for n_layers in layer_list:
        for n_neurons in neuron_list:
            try:
                ## Crear la instancia de la red neuronal
                net = neural_network(n_neurons,n_layers,activation)

                ## Cargar los parámetros de la red
                net.load_state_dict(torch.load('nn_parameters/{}_model_weights_L{}_n{}.pth'.format(activation,n_layers, n_neurons)))
                params = net.state_dict()
                filtered_params = filter_params(params)
                    
                ## Se cargan las cotas de la red
                bounds = read_bounds(n_layers,n_neurons,activation)
                for i in range(0,steps,1):
                    tol_distance = tol_0 + tol_step*i

                    verif_model,all_vars = create_verification_model(params,bounds,activation,tol_distance,apply_softmax,image_list,target_output,real_output,exact,apply_bounds)
                    #verif_model.redirectOutput()
                    verif_model.setParam('limits/time', int(60*minutes))
                    auxt = time.time()
                    verif_model.optimize()
                    dt = time.time() - auxt 
                    
                    if verif_model.getStatus() == 'optimal':
                        gap = 0.0
                        solution = [verif_model.getVal(all_vars['h{},{}'.format(-1,i)]) for i in range(len(image_list))]
                        output,probs = calculate_probs(net, solution, real_output)
                        real_prob = probs[real_output]
                        target_prob = probs[target_output]
                        if save_image:
                            image_solution = np.array(solution).reshape(28, 28)
                            # Crea una figura con dos subplots
                            fig, axs = plt.subplots(1, 2)
                            color_map = 'gray'
                        
                            axs[0].imshow(image_input, vmin = input_lb, vmax = input_ub,cmap=color_map)
                            axs[0].axis('off')
                        
                            axs[1].imshow(image_solution, vmin = input_lb, vmax = input_ub, cmap=color_map)
                            axs[1].axis('off')
                        
                            #axs[2].imshow(np.abs(image_solution-image_input), vmin = input_lb, vmax = input_ub, cmap=color_map) #np.abs(image_solution-image_input)
                            #axs[2].axis('off')
                        
                            # Ajusta el espaciado entre los subplots
                            plt.tight_layout()
                            
                            # Guarda la figura con las imágenes en un archivo
                            plt.savefig('imagen_resultado.png', dpi=300, bbox_inches='tight')
                    
                            # Muestra la figura con las dos imágenes
                            #plt.show()
                        if verif_model.getObjVal() > 0:    
                            break
                    
                    else:
                        primalb = verif_model.getDualbound()
                        dualb = verif_model.getPrimalbound()
                        gap = (dualb-primalb)/np.abs(dualb)
                        target_prob,real_prob = '-','-'
                        if dualb > 0:
                            break              
                new_line = [n_layers,n_neurons,tol_distance,dt,gap,target_prob,real_prob]
                df = df._append(pd.Series(new_line), ignore_index=True)
                written = False
                while not written:
                    try:
                        df.to_excel('datos_verificacion_{}.xlsx'.format(activation), header = False, index = False)
                        written = True
                    except:
                        time.sleep(5) 
                
            except:
                print('error')
