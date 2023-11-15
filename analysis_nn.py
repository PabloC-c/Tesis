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
neuron_list = [5,10]
layer_list  = [2,3,4]
activation = 'sigmoid'

batch_size = 64

calculate_acc = False

if calculate_acc:
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
                #print(params)

## Redes
activation = 'sigmoid'
neuron_list = [5,10]
layer_list  = [2,3,4]
distance_list = [0.01,0.05]
type_bounds = 'verif_bounds'
real_output = 1
df = pd.DataFrame()

calculate_convexorconcav = True

inflec_point = calculate_inflec_point(activation)

if calculate_convexorconcav:
    for tol_distance in distance_list:
        df = pd.DataFrame()
        for n_neurons in neuron_list:
            for n_layers in layer_list:
                new_line = [n_layers,n_neurons]
                print('\n \t Caso {} capas, {} neuronas \n'.format(n_layers,n_neurons))
                bounds_file = calculate_bounds_file_name(type_bounds,activation,n_layers,n_neurons,tol_distance,real_output)
                bounds      = read_bounds(True, n_layers, n_neurons, activation, bounds_file)
                l = -1
                while l < n_layers:
                    layer_bounds = bounds[l]
                    s_convexorconcav = 0
                    for i in range(len(layer_bounds)):
                        lb,ub = -layer_bounds[i][0],layer_bounds[i][1]
                        if lb > ub:
                            print('Capa {}, neurona {}, cotas cruzadas'.format(l,i))
                        if l >=0:
                            activ_f = get_activ_func(activation)
                            lb,ub = activ_f(lb),activ_f(ub)
                        elif (lb < inflec_point-1E-6 and ub <= inflec_point+1E-6) or (lb >= inflec_point-1E-6 and ub > inflec_point+1E-6):
                            s_convexorconcav += 1
                    p_convexorconcav = 100*(s_convexorconcav/len(layer_bounds))
                    new_line.append(p_convexorconcav)
                    print('Capa {}: {}% de neuronas convex or concave'.format(l,p_convexorconcav))
                    l+=1
                while len(new_line) < 2+1+max(layer_list):
                    new_line.append('-')
                df = df._append(pd.Series(new_line), ignore_index=True)
                ## Se intenta escribir en el archivo del df
                data_file = 'data_{}_{}_signs_tolper{}.xlsx'.format(activation,type_bounds,int(100*tol_distance))
                written = False
                while not written:
                    try:
                        df.to_excel(data_file, header = False, index = False)
                        written = True
                    except:
                        time.sleep(5)

activation = 'sigmoid'
neuron_list = [5,10]
layer_list  = [2,3,4]
distance_list = [0.01,0.05]
type_bounds = 'verif_bounds'
exact = 'no_exact'
real_output = 1
target_output = 7
df = pd.DataFrame()

check_tightness = False

def read_lpsol_check_tightness(lp_sol_file,n_layers,bounds,new_line,tight_tol = 0.1,tol = 1E-6):
    sol_dict = {}
    try:
        with open(lp_sol_file, newline='\n') as csvfile:
            reader = csv.reader(csvfile, delimiter='\t')
            next(reader)  # skip header
            for line in reader:
                line_info = line[0].split()
                if line_info[0][0] == 'z' or line_info[0][0] == 'h':
                    sol_dict[line_info[0]] = float(line_info[1])
    except:
        return sol_dict,new_line
    input_bounds = bounds[-1]
    s_tight = 0
    for i in range(len(input_bounds)):
        lb,ub = -input_bounds[i][0],input_bounds[i][1]
        var_name = 'h-1,{}'.format(i)
        if var_name in sol_dict:
            var_value = sol_dict[var_name]
        else:
            var_value = 0.0
        bounds_range = np.linalg.norm(ub-lb)
        var_distance = min(np.linalg.norm(var_value-lb),np.linalg.norm(var_value-ub))
        per_distance = var_distance/bounds_range
        if bounds_range <= tol or per_distance <= tight_tol:
            s_tight += 1
    p_tight = s_tight/len(input_bounds)
    new_line.append(p_tight)
    for l in range(n_layers):
        layer_bounds = bounds[l]
        s_tight = 0
        for i in range(len(layer_bounds)):
            lb,ub = -layer_bounds[i][0],layer_bounds[i][1]
            var_name = 'z{},{}'.format(l,i)
            if var_name in sol_dict:
                var_value = sol_dict[var_name]
            else:
                var_value = 0.0
            bounds_range = np.linalg.norm(ub-lb)
            var_distance = min(np.linalg.norm(var_value-lb),np.linalg.norm(var_value-ub))
            per_distance = var_distance/bounds_range
            if bounds_range <= tol or per_distance <= tight_tol:
                s_tight += 1
        p_tight = s_tight/len(layer_bounds)
        new_line.append(p_tight*100)
    return sol_dict,new_line

if check_tightness:
    for tol_distance in distance_list:
        df = pd.DataFrame()
        for n_neurons in neuron_list:
            for n_layers in layer_list:
                new_line = [n_layers,n_neurons]
                print('\n \t Caso {} capas, {} neuronas \n'.format(n_layers,n_neurons))
                bounds_file = calculate_bounds_file_name(type_bounds,activation,n_layers,n_neurons,tol_distance,real_output)
                lp_sol_file = 'sols/{}_{}_{}_sol_L{}_n{}_1como{}_tolper{}.txt'.format(activation,exact,type_bounds,n_layers,n_neurons,target_output,int(100*tol_distance))
                bounds      = read_bounds(True, n_layers, n_neurons, activation, bounds_file)
                sol_dict,new_line = read_lpsol_check_tightness(lp_sol_file,n_layers,bounds,new_line)
                if len(sol_dict) == 0:
                    continue
                while len(new_line) < 2+1+max(layer_list):
                    new_line.append('-')
                df = df._append(pd.Series(new_line), ignore_index=True)
                ## Se intenta escribir en el archivo del df
                data_file = 'data_{}_{}_{}_tightness_1como{}_tolper{}.xlsx'.format(activation,exact,type_bounds,target_output,int(100*tol_distance))
                written = False
                while not written:
                    try:
                        df.to_excel(data_file, header = False, index = False)
                        written = True
                    except:
                        time.sleep(5)
        
