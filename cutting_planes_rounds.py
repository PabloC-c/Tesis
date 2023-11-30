import os
import sys
import pandas as pd
from torchvision import datasets, transforms
from functions import *

activation_list = ['softplus']
layer_list = [4]
neuron_list = [10]
exact = 'multidim_env'      # exact{exact: exacto, no_exact: formulaciones alternas o envolturas, prop: modelo para calcular las cotas solo con propagacion}
apply_bounds = True
type_bounds_list = ['verif_bounds']
minutes = 15
save_image = False
apply_softmax = False

root_node_only = True
set_initial_sol = False
print_output = False
save_results = False
real_output = 1
target_output = 2
input_lb =0 
input_ub = 1
tol_distance = 0.05


if len(sys.argv) > 1:
    activation_list = [sys.argv[1]]
    if len(sys.argv) >= 3:
        layer_list = [int(sys.argv[2])]
    if len(sys.argv) >= 4:
        neuron_list = [int(sys.argv[3])]
    if len(sys.argv) >= 5:
        form_list = [sys.argv[4]]
    if len(sys.argv) >= 6:
        apply_bounds_list = [bool(sys.argv[5])]
    if len(sys.argv) >= 7:
        type_bounds_list = [sys.argv[6]]
    if len(sys.argv) >= 8:
        save_image = bool(sys.argv[7])
    if len(sys.argv) >= 9:    
        apply_softmax = bool(sys.argv[10])

class LPstatEventhdlr(Eventhdlr):
    """PySCIPOpt Event handler to collect data on LP events."""

    varlist = {}

    def collectNodeInfo(self):
        objval = self.model.getSolObjVal(None)
        if abs(objval) >= self.model.infinity(): #LP no acotado
            return

        LPsol = {}
        if self.varlist == {}:
            self.varlist = self.model.getVars(transformed=False)
        for var in self.varlist:
            solval = self.model.getSolVal(None, var)
            # store only solution values above 1e-6
            if abs(solval) > 1e-6:
                LPsol[var.name] = solval
        self.LPsol = LPsol
        #print("\nSeen sol\n", eventhdlr.LPsol)
        
    def eventexec(self, event):
        if event.getType() == SCIP_EVENTTYPE.FIRSTLPSOLVED or event.getType() == SCIP_EVENTTYPE.LPSOLVED:
            self.collectNodeInfo()
        else:
            print("unexpected event:" + str(event))
        return {}

    def eventinit(self):
        self.model.catchEvent(SCIP_EVENTTYPE.LPEVENT, self)

## Se cargan las imagenes
test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(),download=True)
test_loader  = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)
## Se selecciona un solo par de entrada y salida correspondiente del conjunto de datos
input_example, output_example = next(iter(test_loader))
## Se transforma el input en una lista
image_list = input_example[output_example == real_output][0].view(-1,784).tolist()[0]

done = False

round_count = 1
while not done:
    ## Por cada activacion
    for activation in activation_list:
        ## Se recorren las capas
        for n_layers in layer_list:
            ## Se recorren las neuronas 
            for n_neurons in neuron_list:
                for type_bounds in type_bounds_list:
                    print('Cotas ',type_bounds)
                    ## Nombre del archivo xlsx donde se guardan los resultados de los experimentos
                    file_name = calculate_verif_file_name(exact,activation,real_output,target_output,root_node_only)
                    ## Nombre del archivo de las cotas
                    bounds_file = calculate_bounds_file_name(type_bounds,activation,n_layers,n_neurons,tol_distance,1)
                    ## Se cargan las cotas del modelo
                    bounds = read_bounds(True,n_layers,n_neurons,activation,bounds_file)
                    ## Se crea la instancia de la red neuronal
                    net = neural_network(n_neurons,n_layers,activation)
                    ## Se cargan los parámetros de la red
                    net.load_state_dict(torch.load('nn_parameters/{}_model_weights_L{}_n{}.pth'.format(activation,n_layers, n_neurons)))
                    ## Se filtran los parametros
                    filtered_params = filter_params(net.state_dict())
                    ## Busqueda de ejemplo adversarial
                    print('\n===== Capas {} Neuronas {} =====\n'.format(n_layers,n_neurons))
                    print('Tolerancia: {} '.format(tol_distance))
                    adv_ex = False
                    ## Se ajustan los parametros en el caso root_node_only
                    if root_node_only:
                        sol_file = 'default_sols/exact/{}/{}_default_verif_sol_L{}_n{}_1como{}.sol'.format(tol_distance,activation,n_layers,n_neurons,target_output)
                        default_run = False
                        if not os.path.exists(sol_file):
                            default_run = True
                            if apply_bounds:
                                apply_bounds = False
                    ## Nombre archivo de la lp sol para la envoltura multidimensional
                    if exact == 'multidim_env':
                        lp_sol_file = 'sols/{}_{}_{}_sol_L{}_n{}_1como{}_tolper{}.txt'.format(activation,exact,type_bounds,n_layers,n_neurons,target_output,int(100*tol_distance))
                        if not os.path.exists(lp_sol_file):        
                            lp_sol_file = 'sols/{}_{}_{}_sol_L{}_n{}_1como{}_tolper{}.txt'.format(activation,'no_exact',type_bounds,n_layers,n_neurons,target_output,int(100*tol_distance))
                            if not os.path.exists(lp_sol_file):
                                lp_sol_file = ''
                    ## Se crea el modelo de verificacion
                    if (exact != 'multidim_env') or lp_sol_file != '':
                        verif_model,all_vars,mdenv_count = create_verification_model(filtered_params,bounds,activation,tol_distance,apply_softmax,image_list,target_output,real_output,exact,lp_sol_file,apply_bounds)
                    else:
                        continue
                    print('ola')
                    ## Se verifica si se añadieron nuevos planos cortantes
                    if mdenv_count == 0:
                        done = True
                    ## Cantidad de cortes añadidos en la ronda
                    print('\n Ronda {}, {} cortes añadidos \n'.format(round_count,mdenv_count))
                    ## Se crea y añade el event handler
                    eventhdlr       = LPstatEventhdlr()
                    eventhdlr.LPsol = {}
                    verif_model.includeEventhdlr(eventhdlr, "LPrec", "rec LP sol after every LP event")
                    ## Se añade la solucion inicial
                    if set_initial_sol:
                        initial_sol,image_vars = create_initial_sol(verif_model,filtered_params,image_list,exact,activation,apply_softmax)
                        accepted = verif_model.addSol(initial_sol)
                    ## Node root only
                    if root_node_only:
                        if not default_run:
                            verif_model.setParam('limits/totalnodes',1)
                            verif_model.setParam('branching/random/priority',1000000)
                            ## Se lee la solucion default
                            if activation == 'relu' and exact == 'no_exact':
                                initial_sol,default_vars = set_bigM_deafult_sol(sol_file,verif_model,filtered_params,apply_softmax)
                                accepted = verif_model.addSol(initial_sol)
                            else:
                                verif_model.readSol(sol_file)
                            verif_model.setHeuristics(SCIP_PARAMSETTING.OFF)
                    if print_output:
                        if root_node_only:
                            if not default_run:
                                verif_model.redirectOutput()
                        else:
                            verif_model.redirectOutput()
                    else:
                        verif_model.hideOutput()
                    ## Se limita el tiempo de resolucion
                    verif_model.setParam('limits/time', int(60*minutes))
                    ## Se aumenta la tolerancia de factibilidad
                    verif_model.setParam('numerics/feastol', 1E-5)
                    ## Se optimiza el modelo en busca del ejemplo adversarial
                    aux_t = time.time()
                    verif_model.optimize()
                    dt = time.time() - aux_t
                    if type_bounds == '-':
                        new_sol_file = 'sols/{}_{}_{}_sol_L{}_n{}_1como{}_tolper{}.txt'.format(activation,exact,'unbound',n_layers,n_neurons,target_output,int(100*tol_distance))
                    else:
                        new_sol_file = 'sols/{}_{}_{}_sol_L{}_n{}_1como{}_tolper{}.txt'.format(activation,exact,type_bounds,n_layers,n_neurons,target_output,int(100*tol_distance))
                    print('Status ',verif_model.getStatus())
                    LPsol_dict = eventhdlr.LPsol
                    if len(LPsol_dict) > 0:
                        max_len = max(len(key) for key in LPsol_dict)
                        with open(new_sol_file, "w+") as f:
                            for key,value in LPsol_dict.items():
                                file_line = f"{key:<{max_len}} {value}\n"
                                f.write(file_line)
                        print('\t Solucion guardada, archivo {}'.format(new_sol_file))
    round_count += 1