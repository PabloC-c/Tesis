import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
from scipy.integrate import dblquad
import time
from functions import *

def calculate_inflex_point(activation):
    if activation in ['sigmoid','selu','-sigmoid','-selu'] :
        inflex_point = 0
    else:
        inflex_point = None
    return inflex_point

def concave_re_scale_0_1_box(w,b,L,U):
    n = len(w)
    new_b = b + sum(w[i]*L[i] if w[i] >= 0 else w[i]*U[i] for i in range(n))
    new_w = np.zeros(n)
    for i in range(n):
        if w[i] >= 0:
            new_w[i] = w[i]*(U[i]-L[i])
        else:
            new_w[i] = w[i]*(L[i]-U[i])
    return new_w,new_b

def convex_re_scale_0_1_box(w,b,L,U):
    n = len(w)
    new_b = b + sum(w[i]*U[i] if w[i] >= 0 else w[i]*L[i] for i in range(n))
    new_w = np.zeros(n)
    for i in range(n):
        if w[i] >= 0:
            new_w[i] = w[i]*(L[i]-U[i])
        else:
            new_w[i] = w[i]*(U[i]-L[i])
    return new_w,new_b

def concave_re_scale_vector(x,w,L,U):
    n = len(w)
    new_x = np.zeros(n)
    for i in range(n):
        if np.abs(L[i]-U[i])<1E-03:
            new_x[i] = 1
        elif w[i] >= 0:
            new_x[i] = (x[i]-L[i])/(U[i]-L[i])
        else:
            new_x[i] = (x[i]-U[i])/(L[i]-U[i])
        if len(x) < -1:
            print('entrada:',i)
            print('vector:',x)
            print('pesos:',w)
            print('Lb:',L)
            print('Ub:',U)
            print('nuevo valor:',new_x[i])
    return new_x

def convex_re_scale_vector(x,w,L,U):
    n = len(w)
    new_x = np.zeros(n)
    for i in range(n):
        if np.abs(L[i]-U[i])<1E-03:
            new_x[i] = 1
        elif w[i] <= 0:
            new_x[i] = (x[i]-L[i])/(U[i]-L[i])
        else:
            new_x[i] = (x[i]-U[i])/(L[i]-U[i])
    return new_x

def concave_scale_der_by_w(vector,w,U,L):
    n = len(vector)
    scaled_vector = np.zeros(n)
    for i in range(n):
        if np.abs(L[i]-U[i])<1E-03:
            scaled_vector[i] = 0
        elif w[i] >= 0:
            scaled_vector[i] = vector[i]/(U[i]-L[i])
        else:
            scaled_vector[i] = vector[i]/(L[i]-U[i])
    return scaled_vector

def convex_scale_der_by_w(vector,w,U,L):
    n = len(vector)
    scaled_vector = np.zeros(n)
    for i in range(n):
        if np.abs(L[i]-U[i])<1E-03:
            scaled_vector[i] = 0
        elif w[i] >= 0:
            scaled_vector[i] = vector[i]/(L[i]-U[i])
        else:
            scaled_vector[i] = vector[i]/(U[i]-L[i])
    return scaled_vector

## Activation function σ that satisfies the STFE property
def sigma(z):
    return 1/(1 + np.exp(-z))

## Activation function σ that satisfies the STFE property derivate's
def sigma_der(z):
    return np.exp(-z)/np.power((1 + np.exp(-z)),2)

def minus_sigma(z):
    return -1/(1 + np.exp(-z))

def minus_sigma_der(z):
    return -np.exp(-z)/np.power((1 + np.exp(-z)),2)

# Function to compute the tie point z_hat for a given interval [L, U]
def compute_z_hat(L0, U0, activation, sigma, sigma_der):
    aux_t = time.time()
    L = L0
    U = U0
    if L0 > U0:
        L = -L
        U = -U
    inflex = calculate_inflex_point(activation)
    if inflex is None:
        z_hat = U
    else:
        if L >= inflex:
            z_hat = L + 1E-05
        elif U <= inflex:
            z_hat = U - 1E-05
        else:
            aux_L = min(U,max(L,inflex))
            aux_U = U
            while True:
                z_hat = (aux_L+aux_U)/2
                if L0 < U0:
                    der   = sigma_der(z_hat)
                    slope = (sigma(z_hat) - sigma(L))/(z_hat-L)
                else:
                    der = -sigma_der(-z_hat)
                    slope = (sigma(-z_hat) - sigma(-L))/(z_hat-L)
                if np.abs(der - slope) <= 1E-5:
                    break
                elif der > slope:
                    if np.abs(z_hat-U) <= 1E-5:
                        break
                    else:
                        aux_L = z_hat
                else:
                    if np.abs(z_hat-max(L,0)) <= 1E-5:
                        break
                    else:
                        aux_U = z_hat
                if time.time()-aux_t >= 10:
                    print('\nCaso lento')
                    print('L0:',L0)
                    print('U0:',U0)
                    print('L:',aux_L)
                    print('U:',aux_U)
                    print('tie point:',z_hat)
    if L0 > U0:
        z_hat = -z_hat
    return z_hat      
      
#
def vector_in_region(L,U,x,w,b,z_hat):
    wx = wx = np.dot(w, x)
    if L < U:
        R_f = wx + b >= z_hat
        R_l = wx + b < z_hat and wx + b*np.linalg.norm(x, ord=np.inf) + 1E-05 >= z_hat*np.linalg.norm(x, ord=np.inf)
    else:
        R_f = wx + b<= z_hat
        R_l = wx + b > z_hat and wx + b*np.linalg.norm(x, ord=np.inf) - 1E-05 <= z_hat*np.linalg.norm(x, ord=np.inf)
    return R_f,R_l

# Recursive function for computing the concave envelope
def concave_envelope(x, w, b, activation, sigma, sigma_der, depth=0):
    """
    Recursive function to compute the concave envelope of f(x) = σ(w^T x + b).
    
    Parameters:
    - w: weight vector (numpy array)
    - b: bias term (scalar)
    - x: input vector (numpy array)
    - z_hat: tie point (scalar) for the function σ over [b, sum(w)+b]
    - depth: recursion depth, initially set to 0
    """
    
    if min(x) < 0:
        print('Envelope: Revursive numerical error',np.min(x))
        x = np.maximum(x,0)
    if max(x) > 1:
        print('Envelope: Revursive numerical error :',np.max(x))
        x = np.minimum(x,1)
    
    # Compute w^Tx
    wx = np.dot(w, x)
    
    # Define the current interval for this recursive step
    L = b
    U = np.dot(w, np.ones_like(x)) + b
    
    # Update z_hat for the current interval
    z_hat = compute_z_hat(L, U, activation, sigma, sigma_der)
    
    # Define the regions
    R_f,R_l = vector_in_region(L,U,x,w,b,z_hat)
    
    # Case 0
    #if np.linalg.norm(x, ord=np.inf) < 1E-03:
    #    R_f = False
    #    R_l = True
    
    # Region R_f: the envelope equals the function
    if R_f:
        return [sigma(wx+b)],[z_hat]
    
    # Region R_l: the envelope is a linear function
    elif R_l:
        return [sigma(b) + ((sigma(z_hat) - sigma(b)) / (z_hat - b)) *(wx)],[z_hat]
    
    # Region R_i: recursion on the lower dimension
    else:
        # Find the index i where xi is the largest
        i = np.argmax(x)
        x_minus_i = np.delete(x, i)
        w_minus_i = np.delete(w, i)
        
        # Recursive call for lower-dimensional function
        f_list,hat_list = concave_envelope(x_minus_i/x[i],w_minus_i, b + w[i], activation, sigma, sigma_der, depth + 1)
        f_minus_i = f_list[-1]
        f_list.append(sigma(b) + x[i]*(f_minus_i-sigma(b)))
        hat_list.append(z_hat)
            
        # Compute the perspective of the concave envelope for the lower dimension
        return f_list,hat_list


def compute_z_hat_vertex(w,b,z_hat):
    ## First vertex
    x1,y1 = None,None
    ## Case x1 == 1
    if 0 <= (z_hat-b-w[0])/w[1] and (z_hat-b-w[0])/w[1] <= 1:
        x1 = 1
        y1 = (z_hat-b-w[0])/w[1]
    ## Case x1 == 0
    elif 0 <= (z_hat-b)/w[1] and (z_hat-b)/w[1] <= 1:
        x1 = 0
        y1 = (z_hat-b)/w[1]
    else:
        x1 = (z_hat-b-w[1])/w[0]
        y1 = 1
        x2 = (z_hat-b)/w[0]
        y2 = 0
        return [[x1,y1],[x2,y2]]
    ## Second vertex
    x2,y2 = None,None
    ## Case y2 == 1
    if 0 <= (z_hat-b-w[1])/w[0] and (z_hat-b-w[1])/w[0] <= 1:
        x2 = (z_hat-b-w[1])/w[0]
        y2 = 1
    ## Case y2 == 0
    elif 0 <= (z_hat-b)/w[0] and (z_hat-b)/w[0] <= 1:
        x2 = (z_hat-b)/w[0]
        y2 = 0
    else:
        if x1 == 1:
            x2 = 0
            y2 = (z_hat-b)/w[1]
        else:
            x2 = 1
            y2 = (z_hat-b-w[0])/w[1]
    return [[x1,y1],[x2,y2]]

def monte_carlo_integration(f, n, num_samples=10000):
    """
    Perform Monte Carlo integration of a function f over [0,1]^n.
    
    Parameters:
    f: Function to integrate, takes a vector x as input.
    n: Dimension of the integration space.
    num_samples: Number of random samples to use.
    
    Returns:
    Estimated value of the integral.
    """
    # Generate random samples uniformly in [0,1]^n
    samples = np.random.rand(num_samples, n)
    
    # Evaluate the function at the random sample points
    function_values = np.apply_along_axis(f, 1, samples)
    
    # Compute the mean of the function values
    integral_estimate = np.mean(function_values)
    
    # Since the volume of the integration domain [0,1]^n is 1^n = 1, 
    # no need to multiply by the volume.
    return integral_estimate


def concave_envelope_derivate(x, w, b, sigma, sigma_der, f_list = [], z_list = [], depth=0):
    """
    Recursive function to compute the concave envelope of f(x) = σ(w^T x + b).
    
    Parameters:
    - w: weight vector (numpy array)
    - b: bias term (scalar)
    - x: input vector (numpy array)
    - z_hat: tie point (scalar) for the function σ over [b, sum(w)+b]
    - depth: recursion depth, initially set to 0
    """
    
    if min(x) < 0:
        print('Derivative: Revursive numerical error',np.min(x))
        x = np.maximum(x,0)
    if max(x) > 1:
        print('Derivative: Revursive numerical error :',np.max(x))
        x = np.minimum(x,1)
    
    # Compute w^Tx
    wx = float(np.dot(w, x))
    
    # Define the current interval for this recursive step
    L = b
    U = float(np.dot(w, np.ones_like(x))) + b
    
    # Update z_hat for the current interval
    z_hat = z_list[-1-depth]
    
    # Define the regions
    R_f,R_l = vector_in_region(L,U,x,w,b,z_hat)
    
    # Case 0
    #if np.linalg.norm(x, ord=np.inf) < 1E-03:
    #    R_f = False
    #    R_l = True
    
    # Region R_f: the envelope equals the function
    if R_f:
        return w*float(sigma_der(wx+b))
    
    # Region R_l: the envelope is a linear function
    elif R_l:
        return w*((sigma(z_hat) - sigma(b)) / (z_hat - b))
    
    # Region R_i: recursion on the lower dimension
    else:
        # Find the index i where xi is the largest
        i = np.argmax(x)
        x_minus_i = np.delete(x, i)
        w_minus_i = np.delete(w, i)
                
        # Recursive call for lower-dimensional function
        #f_minus_i = concave_envelope(x_minus_i/x[i],w_minus_i, b + w[i], sigma, sigma_der, z_hat, depth + 1)
        f_minus_i = f_list[-2-depth]

        der = np.zeros(len(x))
        der_minus_i = concave_envelope_derivate(x_minus_i/x[i],w_minus_i,b + w[i], sigma, sigma_der, f_list, z_list, depth + 1)
        der[:i] = der_minus_i[:i]
        der[i] = f_minus_i-sigma(b)-(1/x[i])*np.dot(der_minus_i,x_minus_i)
        der[i+1:] = der_minus_i[i:]
        
        return der

def concave_env_derivate_plane(x0,x,w,b,og_w,og_b,L,U,activation,sigma,sigma_der):
    x0_rescaled = concave_re_scale_vector(x0,og_w,L,U)
    f_list,z_list = concave_envelope(x0_rescaled, w, b, activation, sigma, sigma_der)
    f0   = f_list[-1]
    der  = concave_envelope_derivate(x0_rescaled, w, b, sigma, sigma_der, f_list, z_list, 0)
    der  = concave_scale_der_by_w(der,og_w,U,L)
    diff = x-x0
    mult = np.dot(diff,der)
    return f0+mult

def concave_check(n_samples,w,b,og_w,og_b,L,U,sigma,sigma_der):
    samples = np.random.uniform(L, U, (n_samples, len(L)))
    for i in range(len(samples)):
        der_plane = concave_env_derivate_plane(samples[i],samples,w,b,og_w,og_b,L,U,activation,sigma,sigma_der)
        env = []
        for j in range(len(samples)):
            sample_rescaled = concave_re_scale_vector(samples[j],og_w,L,U)
            env_eval = concave_envelope(sample_rescaled, w, b, activation, sigma, sigma_der)[0][-1]
            env.append(env_eval)
        env = np.array(env)
        cutting_plane = np.all(der_plane - env >= -1E010)
        if not cutting_plane:
            idx = np.where((der_plane - env >= -1E010) == False)[0]
            return [samples[i],samples,idx,der_plane,env]
    return [True]

def convex_env_derivate_plane(x0,x,w,b,og_w,og_b,L,U,activation,sigma,sigma_der):
    x0_rescaled = convex_re_scale_vector(x0,og_w,L,U)
    f_list,z_list = concave_envelope(x0_rescaled, w, b, activation, sigma, sigma_der)
    f_list = -np.array(f_list)
    f0   = f_list[-1] 
    der  = concave_envelope_derivate(x0_rescaled, w, b, sigma, sigma_der, -f_list, z_list, 0)
    der  = -convex_scale_der_by_w(der,og_w,U,L)
    diff = x-x0
    mult = np.dot(diff,der)
    return f0+mult

def convex_check(n_samples,w,b,og_w,og_b,L,U,sigma,sigma_der):
    samples = np.random.rand(n_samples, len(w))
    for i in range(len(samples)):
        der_plane = convex_env_derivate_plane(samples[i],samples,w,b,og_w,og_b,L,U,activation,sigma,sigma_der)
        env = []
        for j in range(len(samples)):
            sample_rescaled = concave_re_scale_vector(samples[j],og_w,L,U)
            env_eval = -(concave_envelope(sample_rescaled, w, b, activation, sigma, sigma_der)[0][-1])
            env.append(env_eval)
        env = np.array(env)
        cutting_plane = np.all(env - der_plane >= -1E010)
        if not cutting_plane:
            idx = np.where((env - der_plane >= -1E010) == False)[0]
            return [samples[i],samples,idx,der_plane,env]
    return [True]

if __name__ == '__main__':
    # Example parameters
    
    activation = 'selu'
    sigma,sigma_der = get_activ_func(activation),get_activ_derv(activation)
    minus_sigma,minus_sigma_der = get_activ_func('-'+activation),get_activ_derv('-'+activation)
    
    # Interesting to plot in sigmoid: w = [7,7], b = -2.5 | w = [3,3], b = -0.5 | w = [7,3], b = -0.5 | w = [1,1], b = -0.5
    
    b = -1
    w = [0.5,0.5]
    L = [-3,-3]
    U = [3,3]
    n = len(w)
    
    mnist = True
    
    if mnist:
        n_neurons = 10
        n_layers = 6
        layer_idx = 0
        neuron_idx = 7
        ## Se crea la instancia de la red neuronal
        net = neural_network(n_neurons,n_layers,activation)
        ## Se cargan los parámetros de la red
        net.load_state_dict(torch.load('nn_parameters/{}_model_weights_L{}_n{}.pth'.format(activation,n_layers, n_neurons)))
        ## Se filtran los parametros
        params = filter_params(net.state_dict())
        weight,bias = get_w_b_names(layer_idx)
        n = params[weight].size()[1]
        w = [float(params[weight][neuron_idx,k]) for k in range(n)]
        b = float(params[bias][neuron_idx])
        L = [0 for i in range(n)]
        U = [1 for i in range(n)]
        x = [0.55 for i in range(n)]
        cc_w,cc_b = concave_re_scale_0_1_box(w,b,L,U)
        f_list,z_list = concave_envelope(x, cc_w, cc_b, activation, sigma, sigma_der)
        der = concave_envelope_derivate(x, cc_w, cc_b, sigma, sigma_der, f_list, z_list)
        
    ## Parameters rescaling
    og_b = b
    og_w = w[:]
    
    w,b = concave_re_scale_0_1_box(w,b,L,U)

    z_hat = compute_z_hat(b, np.sum(w) + b, activation, sigma, sigma_der)  # Compute initial z_hat for the interval [b, sum(w) + b]
    
    def f(x,w,b):
        return sigma(np.dot(w,x)+b)
    
    def h(x,w,b,z_hat):
        if f(x,w,b) > z_hat:
            return f(x,w,b)
        else:
            return sigma(b)+((sigma(z_hat) - sigma(b)) / (z_hat - b)) *(np.dot(w,x))
    
    #Number of samples for Monte Carlo integration
    n_samples = int(1E+2)

    # Compute the integral
    aux_t = time.time()
    integral_f   = monte_carlo_integration(lambda x: f(x,w,b), n, n_samples)
    t_f = time.time() - aux_t
    
    aux_t = time.time()
    integral_env = monte_carlo_integration(lambda x: concave_envelope(x,w, b, activation, sigma, sigma_der, depth=0)[0][-1], n, n_samples)
    t_env = time.time() - aux_t
    
    aux_t = time.time()
    integral_h   = monte_carlo_integration(lambda x:  h(x,w,b,z_hat), n, n_samples)
    t_h = time.time() - aux_t

    outputcc = concave_check(n_samples,w,b,og_w,og_b,L,U,sigma,sigma_der)
    if len(outputcc) > 1:
        x0,samples,idx,der_plane,env_cc = outputcc
    else:
        outputcc = outputcc[0]
    
    new_w,new_b = convex_re_scale_0_1_box(og_w,og_b,L,U)
    
    outputcv = convex_check(n_samples,new_w,new_b,og_w,og_b,L,U,minus_sigma,minus_sigma_der)
    if len(outputcv) > 1:
        x0,samples,idx,der_plane,env_cv = outputcv
    else:
        outputcv = outputcv[0]     
        
    plot_flag = True
    cutting_plane = False

    if plot_flag and n == 2:
        
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.set_xlim([L[0],U[0]])
        ax.set_ylim([L[1],U[1]])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('f')
        
        x = np.linspace(L[0], U[0], 200)
        y = np.linspace(L[1], U[1], 200)
        X, Y = np.meshgrid(x, y)
        
        ## Function
        f = np.vectorize(lambda x, y: sigma(np.dot(og_w,np.array([x,y]))+og_b))
        Z = f(X, Y)
        ax.set_zlim([np.min(Z)-0.1, np.max(Z)+0.1])
        ax.plot_surface(X, Y, Z, cmap='coolwarm' , edgecolor='none',alpha = 1)
        
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.set_xlim([L[0],U[0]])
        ax.set_ylim([L[1],U[1]])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('f')
        
        x = np.linspace(L[0], U[0], 200)
        y = np.linspace(L[1], U[1], 200)
        X, Y = np.meshgrid(x, y)
        
        ## Function
        f = np.vectorize(lambda x, y: sigma(np.dot(og_w,np.array([x,y]))+og_b))
        Z = f(X, Y)
        ax.set_zlim([np.min(Z)-0.1, np.max(Z)+0.1])
        ax.plot_surface(X, Y, Z, cmap='coolwarm' , edgecolor='none',alpha = 1)
        
        ## Concave env
        
        w,b = concave_re_scale_0_1_box(og_w,og_b,L,U)
        
        if og_w[0] <= 0:
            scaled_x = (x-U[0])/(L[0]-U[0])
        else:
            scaled_x = (x-L[0])/(U[0]-L[0])
        if og_w[1] <= 0:
            scaled_y = (y-U[1])/(L[1]-U[1])
        else:
            scaled_y = (y-L[1])/(U[1]-L[1])
        
        scaled_X,scaled_Y = np.meshgrid(scaled_x,scaled_y)
        env  = np.vectorize(lambda x, y: concave_envelope(np.array([x,y]), w, b, activation, sigma, sigma_der, depth=0)[0][-1])
        Z_env = env(scaled_X, scaled_Y)
        ax.plot_surface(X, Y, Z_env, cmap='Reds', edgecolor='none', alpha = 0.6)
        
        ## Convex env
        new_w,new_b = convex_re_scale_0_1_box(og_w,og_b,L,U)
        
        if og_w[0] >= 0:
            scaled_x = (x-U[0])/(L[0]-U[0])
        else:
            scaled_x = (x-L[0])/(U[0]-L[0])
        if og_w[1] >= 0:
            scaled_y = (y-U[1])/(L[1]-U[1])
        else:
            scaled_y = (y-L[1])/(U[1]-L[1])
        
        scaled_X,scaled_Y = np.meshgrid(scaled_x,scaled_y)
        env = np.vectorize(lambda x, y: -(concave_envelope(np.array([x,y]), new_w, new_b, activation, minus_sigma, minus_sigma_der)[0][-1]))
        Z_env = env(scaled_X, scaled_Y)
        ax.plot_surface(X, Y, Z_env, cmap='Blues', edgecolor='none', alpha = 0.9)
        
        ## Concave z hat
        if np.sum(w)+b > z_hat:
            [[x1,y1],[x2,y2]] = compute_z_hat_vertex(w,b,z_hat)
            
            if og_w[0] <= 0:
                x1 = x1*(L[0]-U[0])+U[0]
                x2 = x2*(L[0]-U[0])+U[0]
            else:
                x1 = x1*(U[0]-L[0])+L[0]
                x2 = x2*(U[0]-L[0])+L[0]
            if og_w[1] <= 0:
                y1 = y1*(L[1]-U[1])+U[1]
                y2 = y2*(L[1]-U[1])+U[1]
            else:
                y1 = y1*(U[1]-L[1])+L[1]
                y2 = y2*(U[1]-L[1])+L[1] 
            
            ax.plot([x1,x2], [y1,y2], [sigma(z_hat),sigma(z_hat)], color = 'black',linewidth = 5)
            
        ## Cutting plane
        if cutting_plane:
            scaled_x0 = [0.3,0.1]
            x0 = np.zeros(n)
            
            if og_w[0] <= 0:
                x0[0] = scaled_x0[0]*(L[0]-U[0])+U[0]
            else:
                x0[0] = scaled_x0[0]*(U[0]-L[0])+L[0]
            if og_w[1] <= 0:
                x0[1] = scaled_x0[1]*(L[1]-U[1])+U[1]
            else:
                x0[1] = scaled_x0[1]*(U[1]-L[1])+L[1] 
                
            f_list,z_list = concave_envelope(scaled_x0, w, b, activation, sigma, sigma_der)
            der = concave_envelope_derivate(scaled_x0, w, b, sigma, sigma_der, f_list, z_list, 0)
            
            der = concave_scale_der_by_w(der,og_w,U,L)
            
            plane = np.vectorize(lambda x, y: concave_envelope(scaled_x0, w, b, activation, sigma, sigma_der, z_hat)[0][-1]+np.dot(der,np.array([x,y])-x0))
            Z_plane = plane(X,Y)
            ax.plot_surface(X, Y, Z_plane, cmap='Blues', edgecolor='none', alpha = 0.6)
            ax.scatter(x0[0],x0[1],concave_envelope(scaled_x0, w, b, activation, sigma, sigma_der,z_hat)[-1],color='black',marker='o',s = 100)
