import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
from scipy.integrate import dblquad
from concave_env import *

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

def convex_sigma(z):
    return -sigma(z)

def convex_sigma_der(z):
    return -sigma_der(z)

if __name__ == '__main__':
    L = [-1,-1]
    U = [1,1]
    w = [-7,7]
    b = -1.5
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlim([L[0],U[0]])
    ax.set_ylim([L[1],U[1]])
    ax.set_zlim([-1.1,1.1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('f')
    
    x = np.linspace(L[0], U[0], 200)
    y = np.linspace(L[1], U[1], 200)   
    X, Y = np.meshgrid(x, y)
    
    f = np.vectorize(lambda x, y: sigma(np.dot(w,np.array([x,y]))+b))
    Z = f(X, Y)
    ax.plot_surface(X, Y, Z, cmap='coolwarm' , edgecolor='none')
    
    new_w,new_b = convex_re_scale_0_1_box(w,b,L,U)
    
    if w[0] >= 0:
        x_rescaled = (x-U[0])/(L[0]-U[0])
    else:
        x_rescaled = (x-L[0])/(U[0]-L[0])
    if w[1] >= 0:
        y_rescaled = (y-U[1])/(L[1]-U[1])
    else:
        y_rescaled = (y-L[1])/(U[1]-L[1])
    
    X_rescaled, Y_rescaled = np.meshgrid(x_rescaled, y_rescaled)
    
    env = np.vectorize(lambda x, y: -concave_envelope(np.array([x,y]), new_w, new_b, minus_sigma, minus_sigma_der))
    Z_env = env(X_rescaled,Y_rescaled)
    ax.plot_surface(X, Y, Z_env, cmap='Blues', edgecolor='none', alpha = 0.9)
    
    x0 = [0.75,-0.75]
    x0_rescaled = [0,0]
    
    if w[0] >= 0:
        x0_rescaled[0] = (x0[0]-U[0])/(L[0]-U[0])
    else:
        x0_rescaled[0] = (x0[0]-L[0])/(U[0]-L[0])
    if w[1] >= 0:
        x0_rescaled[1] = (x0[1]-U[1])/(L[1]-U[1])
    else:
        x0_rescaled[1] = (x0[1]-L[1])/(U[1]-L[1])
    
    der = concave_envelope_derivate(x0_rescaled, new_w, new_b, minus_sigma, minus_sigma_der)
    der = -convex_scale_der_by_w(der,w,U,L)
    
    plane = np.vectorize(lambda x, y:np.dot(der,np.array([x,y])-x0)-concave_envelope(x0_rescaled, new_w, new_b, minus_sigma, minus_sigma_der))
    Z_plane = plane(X,Y)
    ax.plot_surface(X, Y, Z_plane, cmap='Blues', edgecolor='none', alpha = 0.6)
    ax.scatter(x0[0],x0[1],-concave_envelope(x0_rescaled, new_w, new_b, minus_sigma, minus_sigma_der),color='black',marker='o',s = 100)