from functions import *

activation = 'sigmoid'
exact = 'exact'
apply_bounds = True
type_bounds = 'verif_bounds'
n_layers = 4
n_neurons = 10
real_output = 1
target_output = 2
tol_distance = 0.05

l = 2
i = 6

if type_bounds == '-':
    lp_sol_file = 'sols/{}_{}_{}_sol_L{}_n{}_1como{}_tolper{}.txt'.format(activation,exact,'unbound',n_layers,n_neurons,target_output,int(100*tol_distance))
else:
    lp_sol_file = 'sols/{}_{}_{}_sol_L{}_n{}_1como{}_tolper{}.txt'.format(activation,exact,type_bounds,n_layers,n_neurons,target_output,int(100*tol_distance))

bounds_file = calculate_bounds_file_name(type_bounds,activation,n_layers,n_neurons,tol_distance,real_output)
bounds = read_bounds(apply_bounds,n_layers,n_neurons,activation,bounds_file)

points_list = get_multidim_env_points(l,bounds,activation)

net = neural_network(n_neurons,n_layers,activation)
net.load_state_dict(torch.load('nn_parameters/{}_model_weights_L{}_n{}.pth'.format(activation,n_layers, n_neurons)))
filtered_params = filter_params(net.state_dict())

n_input = len(bounds[l-1])

lp_sol = generate_hyperplane_model_lpsol(l,i,n_input,activation,lp_sol_file)

hp_model,c,d = create_hyperplane_model(l,i,filtered_params,bounds,lp_sol,points_list,activation)

succes,cc_or_cv,c,d = calculate_hyperplane(l,i,bounds,activation,filtered_params,n_input,lp_sol_file)