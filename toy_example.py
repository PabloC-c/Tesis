from functions import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
## Arquitectura de la red
activation = 'relu'
n_neurons = 2
n_layers = 2
n_input = 2
n_output = 2
## Se crea la red
net = neural_network(n_neurons,n_layers,activation,n_input,n_output)
## Archivo de los parametros
filename = "nn_parameters/{}_toy_ex_L{}_n{}.pth".format(activation,n_layers, n_neurons)
## Se cargan los parametros de la red
net.load_state_dict(torch.load(filename))
params = net.state_dict()
filtered_params = filter_params(params)

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

filtered_params = create_params(matrix_array,bias_array)

bounds,layers_time,neuron_model,input_var,output_var,all_vars = calculate_bounds(filtered_params,activation = 'relu',exact = 'no_exact',minutes = 10,add_verif_bounds = False,tol_distance = 0,image_list = [],print_output=False)

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

hat_x = [0.3,0.7]
eps = 0.1
lambda_param1 = 0
lambda_param2 = 0
n_clusters = 2
partition = [[(-1,0),(0,0),(1,0)],[(-1,1),(0,1),(1,1)]]
edges_p = [(0,0,1),(0,1,0),(1,0,1),(1,1,0)]
lambda_params = {(0,0,1):0,(0,1,0):0,(1,0,1):lambda_param1,(1,1,0):lambda_param2}
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

lambda_method = 'dual'

## Se generan 3 columnas iniciales
ref0 = hat_x
ref1 = [val-eps for val in hat_x]
ref2 = [val+eps for val in hat_x]

columns = [propagate_input_to_column(ref0,filtered_params,edges_p,activation='relu'),
           propagate_input_to_column(ref1,filtered_params,edges_p,activation='relu'),
           propagate_input_to_column(ref2,filtered_params,edges_p,activation='relu')]

master_list = []
m_prop_list = []
pricing_list = []
p_prop_list = []
p_zfixed_list = []

master_bestsol = {}
m_prop_bestsol = {}
pricing_bestsol = {}
p_prop_bestsol = {}
p_zfixed_bestsol = {}


max_iter = 8
counter = 1
while counter <= max_iter:
    print(f'\n ========== Iteracion {counter} ========== \n')
    ## Modelo maestro
    master_model,theta,cons_dict = create_master_model(columns,edges_p,real_label,target_label,filtered_params)
    master_model.setPresolve(SCIP_PARAMSETTING.OFF)
    master_model.setHeuristics(SCIP_PARAMSETTING.OFF)
    master_model.disablePropagation()
    master_model.hideOutput()
    master_model.optimize()
    master_obj = master_model.getObjVal()
    ## Actualizacion de lambdas
    if lambda_method == 'dual':
        lambda_params = get_master_lambdas(master_model,edges_p,cons_dict,lambda_params = None)
    ## Pricing
    pricing_models,pricing_vars = create_pricing_models(n_clusters,
                                                        partition,
                                                        edges_p,
                                                        lambda_params,
                                                        bounds,
                                                        filtered_params,
                                                        hat_x,
                                                        real_label,
                                                        target_label,
                                                        eps,
                                                        activation)
    for model in pricing_models:
        model.hideOutput()
        model.setPresolve(SCIP_PARAMSETTING.OFF)
        model.setHeuristics(SCIP_PARAMSETTING.OFF)
        model.disablePropagation()
        model.optimize()
    pricing_obj,pricing_sol = compute_pricing_sol_obj(pricing_models,pricing_vars,partition)
    ## Heuristicas
    ## Master propagation
    master_sol = make_theta_solution(master_model,theta,columns)
    m_prop_obj,m_prop_sol = propagation_heuristic(master_sol,filtered_params,real_label,target_label,activation)
    ## Pricing propagation
    p_prop_obj,p_prop_sol = propagation_heuristic(pricing_sol,filtered_params,real_label,target_label,activation)
    ## Pricing z fixed
    p_zfixed_obj,p_zfixed_sol = pricing_zfixed_heuristic(pricing_models,pricing_vars,partition,filtered_params,real_label,target_label,activation)
    ## Se guarda el mejor valor objetivo del master
    if len(master_list) == 0: 
        master_list.append(master_obj)
        master_bestsol = master_sol
    else:
        if master_obj < master_list[-1]:
            master_list.append(master_obj)
            master_bestsol = master_sol
        else:
            master_list.append(master_list[-1])
    ## Se guarda el mejor valor objetivo del pricing
    if len(pricing_list)==0: 
        pricing_list.append(pricing_obj)
        pricing_bestsol = pricing_sol
    else:
        if pricing_obj > pricing_list[-1]:
            pricing_list.append(pricing_obj)
            pricing_bestsol = pricing_sol
        else:
            pricing_list.append(pricing_list[-1])
    ## Se guarda el mejor valor objetivo de la heuristica master prop
    if len(m_prop_list)==0: 
        m_prop_list.append(m_prop_obj)
        m_prop_bestsol = m_prop_sol
    else:
        if m_prop_obj < m_prop_list[-1]:
            m_prop_list.append(m_prop_obj)
            m_prop_bestsol = m_prop_sol
        else:
            m_prop_list.append(m_prop_list[-1])
    ## Se guarda el valor objetivo de la heuristica pricing propagation
    if len(p_prop_list) == 0 or pricing_obj<p_prop_obj<master_obj:
        p_prop_list.append(p_prop_obj)
        p_prop_bestsol = p_prop_sol
    else:
        p_prop_list.append(p_prop_list[-1])
    ## Se guarda el valor objetivo de la heuristica pricing z-fixed
    if len(p_zfixed_list) == 0 or pricing_obj<p_zfixed_obj<master_obj:
        p_zfixed_list.append(p_zfixed_obj)
        p_zfixed_bestsol = p_zfixed_sol
    else:
        p_zfixed_list.append(p_zfixed_list[-1])
    print(f' Cota master: {master_obj}')
    print(f' Cota pricing: {pricing_obj}')
    columns.append(get_pricing_column(pricing_models,edges_p))
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
            master_list,
            pricing_list,
            m_prop_list,
            p_prop_list,
            p_zfixed_list]
## Eje x
x_axis = list(range(1, max_iter + 1))

## Grafica cada lista con su respectivo color y etiqueta
for aux_lista, aux_color, aux_label in zip(all_info,colors,list_labels):
    if aux_label in ['MILP','LP']:
        plt.plot(x_axis,aux_lista,'-',color=aux_color,label=aux_label)
    else: 
        plt.plot(x_axis,aux_lista,'|--',color=aux_color,label=aux_label,dashes=(3,4),alpha=1)

plt.xticks(x_axis)
plt.legend()
plt.show()
