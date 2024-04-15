from functions import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
np.random.seed(5)
## Arquitectura de la red
activation = 'relu'
n_neurons = 2
n_layers = 2
n_input = 2
n_output = 2
## Se crea la red
#net = neural_network(n_neurons,n_layers,activation,n_input,n_output)
## Archivo de los parametros
#filename = "nn_parameters/{}_toy_ex_L{}_n{}.pth".format(activation,n_layers, n_neurons)
## Se cargan los parametros de la red
#net.load_state_dict(torch.load(filename))
#params = net.state_dict()
#filtered_params = filter_params(params)

###############################################################################################################################

## Funcion que crea el diccionario de parametros dada un arreglo con matrices y vectores
def create_params(matrix_array, bias_array):
    params = OrderedDict()
    for i in range(len(matrix_array)):
        params[f'fc_hidden.{i}.weight'] = torch.tensor(matrix_array[i])
        params[f'fc_hidden.{i}.bias'] = torch.tensor(bias_array[i])
    return params

## Caso "interesante" para ambos problemas lambda1 positivo y lambda2 negativo
#matrix_array = [[[3,4],[1,2]],[[1.5,-2],[2,3]]] 
#bias_array = [[-1,-0.5],[1,-2]]

## Testeo
matrix_array = [[[-0.9,0.3],[0.9,-0.2]],[[0.2,-0.3],[-0.7,0.1]]] 
bias_array = [[0.5,0.1],[0.2,0,5]]

#matrix_array = []
#bias_array = []
#multiplier = 1
#for l in range(n_layers):
#    if l == 0:
#        matrix_array.append((multiplier*np.random.rand(n_neurons,n_input)).tolist())
#        bias_array.append((multiplier*np.random.rand(n_neurons)).tolist())
#    elif l == n_layers - 1:
#        matrix_array.append((multiplier*np.random.rand(n_output,n_neurons)).tolist())
#        bias_array.append((multiplier*np.random.rand(n_output)).tolist())
#    else:
#        matrix_array.append((multiplier*np.random.rand(n_neurons,n_neurons)).tolist())
#        bias_array.append((multiplier*np.random.rand(n_neurons)).tolist())

#random_layers = np.random.choice(np.arange(0, n_layers), n_layers//2, replace=False).astype(int)

#for l in random_layers:
#    matrix = matrix_array[l]
#    n = n_neurons
#    n_prev = n_neurons
#    if l == 0:
#        n_prev = n_input
#    elif l == n_layers-1:
#        n = n_output
#    random_neurons = np.random.choice(np.arange(0, n), int(n*0.5), replace=False).astype(int)
#    random_prev = np.random.choice(np.arange(0, n_prev), int(n_prev*0.5), replace=False).astype(int)
#    for i in random_neurons:
#        for j in random_prev:
#            matrix[i][j] = matrix[i][j]*500
#            print(matrix[i][j])
#    matrix_array[l] = matrix
    
filtered_params = create_params(matrix_array,bias_array)

bounds,layers_time,neuron_model,input_var,output_var,all_vars = calculate_bounds(filtered_params,activation = 'relu',exact = 'no_exact',minutes = 10,add_verif_bounds = False,tol_distance = 0,image_list = [],print_output=False,n_input=n_input)

###############################################################################################################################

def f1(x,y,params,bounds,lambda_param):
    l,u = -bounds[0][1][0],bounds[0][1][1]
    u = max(0,u)
    weight,bias = get_w_b_names(1)
    W2,b2 = params[weight],params[bias]
    c = W2[0,1]
    weight,bias = get_w_b_names(0)
    W1,b1 = params[weight],params[bias]
    alpha = b2[0]+W2[0,0]*(b1[0]+W1[0,0]*x+W1[0,1]*y)
    min_obj = 0
    case = ''
    optim_sol = 0
    if c>0 and not (-alpha/c)>u:
        if (lambda_param+c)>0:
            sol = max(0,(-alpha/c))
        else:
            sol = u
        output = (lambda_param+c)*sol
        if case == '':
            min_obj = output
            case = 'caso 1'
            optim_sol = sol          
    if c>0 and not alpha>0:
        if lambda_param > 0:
            sol = 0
        else:
            sol = min((-alpha/c),u)
        output = lambda_param*sol
        if case == '':
            min_obj = output
            case = 'caso 2'
            optim_sol = sol
        elif output == min_obj:
            if np.abs(sol) < np.abs(optim_sol) and (lambda_param+c) >= 0:
                min_obj = output
                case = 'caso 2'
                optim_sol = sol
        elif output < min_obj:
            min_obj = output
            case = 'caso 2'
            optim_sol = sol
    if c<0 and not alpha<0:
        if (lambda_param+c) > 0:
            sol = 0
        else:
            sol = min((-alpha/c),u)
        output = (lambda_param+c)*sol
        if case == '':
            min_obj = output
            case = 'caso 3'
            optim_sol = sol
        elif output == min_obj:
            if np.abs(sol) < np.abs(optim_sol) and (lambda_param+c) >= 0:
                min_obj = output
                case = 'caso 3'
                optim_sol = sol
        elif output < min_obj:
            min_obj = output
            case = 'caso 3'
            optim_sol = sol
    if c<0 and not (-alpha/c)>u:
        if lambda_param > 0:
            sol = max(0,(-alpha/c))
        else:
            sol = u
        output = lambda_param*sol
        if case == '':
            min_obj = output
            case = 'caso 4'
            optim_sol = sol
        elif output == min_obj:
            if np.abs(sol) < np.abs(optim_sol) and (lambda_param+c) >= 0:
                min_obj = output
                case = 'caso 4'
                optim_sol = sol
        elif output < min_obj:
            min_obj = output
            case = 'caso 4'
            optim_sol = sol
    return alpha+min_obj,case,optim_sol

def f2(x,y,params,bounds,lambda_param):
    l,u = -bounds[0][0][0],bounds[0][0][1]
    u = max(0,u)
    weight,bias = get_w_b_names(1)
    W2,b2 = params[weight],params[bias]
    c = W2[1,0]
    weight,bias = get_w_b_names(0)
    W1,b1 = params[weight],params[bias]
    alpha = b2[1]+W2[1,1]*(b1[1]+W1[1,1]*x+W1[1,0]*y)
    min_obj = 0
    case = ''
    optim_sol = 0
    if c>0 and not (-alpha/c)>u:
        if (lambda_param+c)>0:
            sol = max(0,(-alpha/c))
        else:
            sol = u
        output = (lambda_param+c)*sol
        if case == '':
            min_obj = output
            case = 'caso 1'
            optim_sol = sol          
    if c>0 and not alpha>0:
        if lambda_param > 0:
            sol = 0
        else:
            sol = min((-alpha/c),u)
        output = lambda_param*sol
        if case == '':
            min_obj = output
            case = 'caso 2'
            optim_sol = sol
        elif output == min_obj:
            if np.abs(sol) < np.abs(optim_sol) and (lambda_param+c) >= 0:
                min_obj = output
                case = 'caso 2'
                optim_sol = sol
        elif output < min_obj:
            min_obj = output
            case = 'caso 2'
            optim_sol = sol
    if c<0 and not alpha<0:
        if (lambda_param+c) > 0:
            sol = 0
        else:
            sol = min((-alpha/c),u)
        output = (lambda_param+c)*sol
        if case == '':
            min_obj = output
            case = 'caso 3'
            optim_sol = sol
        elif output == min_obj:
            if np.abs(sol) < np.abs(optim_sol) and (lambda_param+c) >= 0:
                min_obj = output
                case = 'caso 3'
                optim_sol = sol
        elif output < min_obj:
            min_obj = output
            case = 'caso 3'
            optim_sol = sol
    if c<0 and not (-alpha/c)>u:
        if lambda_param > 0:
            sol = max(0,(-alpha/c))
        else:
            sol = u
        output = lambda_param*sol
        if case == '':
            min_obj = output
            case = 'caso 4'
            optim_sol = sol
        elif output == min_obj:
            if np.abs(sol) < np.abs(optim_sol) and (lambda_param+c) >= 0:
                min_obj = output
                case = 'caso 4'
                optim_sol = sol
        elif output < min_obj:
            min_obj = output
            case = 'caso 4'
            optim_sol = sol
    return alpha+min_obj,case,optim_sol

###############################################################################################################################

to_plot = False

if to_plot:

    f1_vectorized = np.vectorize(lambda x, y: f1(x, y, filtered_params, bounds, lambda_param1)[0])

    x = np.linspace(hat_x[0]-eps, hat_x[0]+eps, 100)
    y = np.linspace(hat_x[1]-eps, hat_x[1]+eps, 100)

    X, Y = np.meshgrid(x, y)
    Z = f1_vectorized(X, Y) 

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none') #Greys

    # Agrega una barra de colores que mapea los valores de Z a colores.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D contour')
    # Añadir un punto al gráfico
    z1 = f1_vectorized(x1,y1)
    ax.scatter([x1], [y1], [z1], color="r")

    # Mostrar el gráfico
    plt.show()

    f2_vectorized = np.vectorize(lambda x, y: f2(x, y, filtered_params, bounds, lambda_param2)[0])

    x = np.linspace(hat_x[1]-eps, hat_x[1]+eps, 100)
    y = np.linspace(hat_x[0]-eps, hat_x[0]+eps, 100)

    X, Y = np.meshgrid(x, y)
    Z = f2_vectorized(X, Y) 

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none') #Greys

    # Agrega una barra de colores que mapea los valores de Z a colores.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D contour')
    # Añadir un punto al gráfico
    z2 = f2_vectorized(x2,y2)
    ax.scatter([x2], [y2], [z2], color="r")

    # Mostrar el gráfico
    plt.show()

###############################################################################################################################

policy = 'equal_lvl'
eps = 0.1
n_clusters = 2
if n_input == 2:
    hat_x = [0.3,0.7]
else:
    hat_x = np.random.rand(n_input)

partition = create_disjoint_partition(n_input,
                                      n_output,
                                      n_neurons,
                                      n_layers,
                                      n_clusters,
                                      filtered_params,
                                      policy,
                                      use_bias = False)
## Aristas a relajar 
edges_p = create_edges_p(partition,n_input,n_neurons,n_layers)
## Parametros lambda
lambda_params = {}
for edge in edges_p:
    lambda_params[edge] = 0
real_label = 0
target_label = 1

## Relajacion lineal
lp_verif_model,_,mdenv_count = create_verification_model(params = filtered_params,
                                                                bounds = bounds,
                                                                activation = activation,
                                                                tol_distance = eps,
                                                                apply_softmax = False,
                                                                image_list = hat_x,
                                                                output_target = target_label,
                                                                real_output = real_label,
                                                                exact = 'no_exact',
                                                                lp_sol_file = '',
                                                                apply_bounds = True,
                                                                lp_relax = True)
lp_verif_model.hideOutput()
lp_verif_model.setPresolve(SCIP_PARAMSETTING.OFF)
lp_verif_model.setHeuristics(SCIP_PARAMSETTING.OFF)
lp_verif_model.disablePropagation()
lp_verif_model.optimize()
lp_obj = lp_verif_model.getObjVal()
print(f' Valor objetivo relajacion lineal: {lp_obj}')

## Modelo exacto
verif_model,_,mdenv_count = create_verification_model(params = filtered_params,
                                                             bounds = bounds,
                                                             activation = activation,
                                                             tol_distance = eps,
                                                             apply_softmax = False,
                                                             image_list = hat_x,
                                                             output_target = target_label,
                                                             real_output = real_label,
                                                             exact = 'no_exact')
verif_model.hideOutput()
verif_model.setPresolve(SCIP_PARAMSETTING.OFF)
verif_model.setHeuristics(SCIP_PARAMSETTING.OFF)
verif_model.disablePropagation()
verif_model.optimize()
ip_obj = verif_model.getObjVal()
print(f' Valor objetivo real: {ip_obj}')
    
###############################################################################################################################

heu_methods = ['master_prop','pricing_prop','pricing_zfixed']

## Se generan 3 columnas iniciales
ref0 = hat_x
ref1 = [min(max(val-eps,0),1) for val in hat_x]
ref2 = [min(max(val+eps,0),1) for val in hat_x]

columns = [propagate_input_to_column(ref0,filtered_params,edges_p,activation='relu'),
           propagate_input_to_column(ref1,filtered_params,edges_p,activation='relu'),
           propagate_input_to_column(ref2,filtered_params,edges_p,activation='relu')]

all_sols = {'master':[],'pricing':[]}
best_sol = {'master':{},'pricing':{}}

for heu in heu_methods:
    all_sols[heu] = []
    best_sol[heu] = {}

max_iter = 10
counter = 1
master_model = None
cons_dict = {}
theta = []
master_times = []
pricing_times = []
pricing_models = []
pricing_vars = []
eta = True
lambda_params = {}
for edge in edges_p:
    lambda_params[edge] = 1
while counter <= max_iter:
    print(f'\n ========== Iteracion {counter} ========== \n')
    ## Pricing
    pricing_models,pricing_obj,pricing_sol,pricing_vars,pricing_times = pricing_iteration(n_clusters,
                                                                                          partition,
                                                                                          edges_p,
                                                                                          lambda_params,
                                                                                          bounds,
                                                                                          filtered_params,
                                                                                          hat_x,
                                                                                          real_label,
                                                                                          target_label,
                                                                                          eps,
                                                                                          pricing_times,
                                                                                          activation,
                                                                                          pricing_models,
                                                                                          pricing_vars,
                                                                                          eta)
    ## Se añade la columna correspondiente 
    columns.append(get_pricing_column(pricing_models,edges_p))
    ## Modelo maestro
    master_model,master_obj,master_sol,lambda_params,theta,cons_dict,master_times = master_iteration(columns,
                                                                                                     edges_p,
                                                                                                     real_label,
                                                                                                     target_label,
                                                                                                     filtered_params,
                                                                                                     master_times,
                                                                                                     master_model = None,
                                                                                                     theta = [],
                                                                                                     cons_dict = {})
    print(f'Master status: {master_model.getStatus()}')
    if counter == 1:
        master_model.writeProblem('master_model.lp')
    ## Se guarda el mejor valor objetivo del master
    if len(all_sols['master']) == 0: 
        all_sols['master'].append(master_obj)
        best_sol['master'] = master_sol
    else:
        if master_obj < all_sols['master'][-1]:
            all_sols['master'].append(master_obj)
            best_sol['master'] = master_sol
        else:
            all_sols['master'].append(all_sols['master'][-1])
    print(f' Lambda: {lambda_params}')
    ## Se guarda el mejor valor objetivo del pricing
    if len(all_sols['pricing'])==0: 
        all_sols['pricing'].append(pricing_obj)
        best_sol['pricing'] = pricing_sol
    else:
        if pricing_obj > all_sols['pricing'][-1]:
            all_sols['pricing'].append(pricing_obj)
            best_sol['pricing'] = pricing_sol
        else:
            all_sols['pricing'].append(all_sols['pricing'][-1])
    ## Heuristicas
    all_sols,best_sol = create_heuristic_sols(heu_methods,all_sols,best_sol,master_sol,master_obj,pricing_sol,pricing_obj,filtered_params,bounds,real_label,target_label,pricing_models,pricing_vars,partition,activation = 'relu')
    print(f' Cota master: {master_obj}')
    print(f' Cota pricing: {pricing_obj}')
    ## Se aumenta el contador
    counter +=1

###############################################################################################################################

lp_list = [lp_obj for i in range(max_iter)]
ip_list = [ip_obj for i in range(max_iter)]

## Colores
colors = ['k','b','g','c','m','r','y']
## Lista de etiquetas
list_labels = ['MILP',
               'LP',
               'Master',
               'Pricing',
               'Master propagation',
               'Pricing propagation',
               'Pricing z-fixed']
## Lista con toda la informacion
all_info = [ip_list,
            lp_list,
            all_sols['master'],
            all_sols['pricing']]
for heu in heu_methods:
    all_info.append(all_sols[heu])

## Eje x
x_axis = list(range(1, max_iter + 1))

## Grafica cada lista con su respectivo color y etiqueta
for aux_lista, aux_color, aux_label in zip(all_info,colors,list_labels):
    if aux_label in ['MILP','LP']:
        plt.plot(x_axis,aux_lista,'-',color=aux_color,label=aux_label)
    elif aux_label == 'Pricing z-fixed':
        if len(aux_lista) > 0:
            while len(aux_lista) < max_iter:
                aux_lista.append(aux_lista[-1])
            plt.plot(x_axis,aux_lista,'|--',color=aux_color,label=aux_label,dashes=(3,4),alpha=1)
    else: 
        plt.plot(x_axis,aux_lista,'|--',color=aux_color,label=aux_label,dashes=(3,4),alpha=1)
plt.xticks(x_axis)
plt.legend()
plt.show()
