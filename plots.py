import os
import sys
import pandas as pd
from torchvision import datasets, transforms
from functions import *

## Funcion de activacion concavaconvexa
activation = 'sigmoid'
f  = get_activ_func(activation)
df = get_activ_derv(activation)

## Bounds correspondientes 
bounds = (16,1)
lb,ub = -bounds[0],bounds[1]

## Se calculan los cc y cv points
cc = calculate_cc_point(activation,bounds)
cv = calculate_cv_point(activation,bounds)

## Se evaluan los puntos en la funcion 
f_lb = f(lb)
f_ub = f(ub)
f_cc = f(cc)
f_cv = f(cv)

## Se generan las tangentes
tan_cc = get_tan_func(activation,cc,lb)[0]
tan_cv = get_tan_func(activation,cv,ub)[0]

## Se generan los vectores del eje x
q = 10000
x    = [(lb)+i*(ub-lb)/q for i in range(q+1)]
x_cc = [(lb)+i*(cc+2-lb)/q for i in range(q+1)] 
x_cv = [(cv-2)+i*(ub-cv+2)/q for i in range(q+1)] 

## Se generan los plots de las funciones
plt.plot(x,[f(xi) for xi in x])
plt.plot(x_cc,[tan_cc(xi) for xi in x_cc],'--',color = 'orange')
plt.plot(x_cv,[tan_cv(xi) for xi in x_cv],'--',color = 'green')

## Se generan los plots de los puntos
plt.plot(lb, f_lb, marker="o", markersize=5, markeredgecolor="orange", markerfacecolor="orange")
plt.plot(ub, f_ub, marker="o", markersize=5, markeredgecolor="green", markerfacecolor="green")
plt.plot(cc, f_cc, marker="o", markersize=5, markeredgecolor="orange", markerfacecolor="orange")
plt.plot(cv, f_cv, marker="o", markersize=5, markeredgecolor="green", markerfacecolor="green")

## Se generan los puntos de las tangentes a añadir en caso de ser necesarios
k = 4
aux = time.time()
## Se añaden las demas envolturas convexas en caso de ser necesarias
if cv - lb >= 1E-6:
    cv_points = calculate_k_points(activation,k,lb,cv)
else:
    cv_points = []
## Se añaden las demas envolturas concavas en caso de ser necesarias
if ub - cc >= 1E-6:
    cc_points = calculate_k_points(activation,k,cc,ub)
else:
    cc_points = []
    
dt = time.time()-aux
print('Tiempo de calculo de puntos',dt)

## Plots de los puntos concavos
x_list = [lb,lb+3]
tan_lb = get_tan_func(activation, lb)[0]
#plt.plot(x_list,[tan_lb(xi) for xi in x_list],'--',color = 'green')
if len(cc_points) > 0:
    for i in range(k):
        plt.plot(cc_points[i], f(cc_points[i]), marker="o", markersize=5, markeredgecolor="orange", markerfacecolor="orange")
        x_list = [cc_points[i]-2,cc_points[i]+2]
        tan_i  = get_tan_func(activation, cc_points[i])[0]
        plt.plot(x_list,[tan_i(xi) for xi in x_list],'--',color = 'orange')
    
## Plots de los puntos convexos
x_list = [ub-3,ub]
tan_ub = get_tan_func(activation, ub)[0]
#plt.plot(x_list,[tan_ub(xi) for xi in x_list],'--',color = 'orange')
if len(cv_points) > 0:
    for i in range(k):
        plt.plot(cv_points[i], f(cv_points[i]), marker="o", markersize=5, markeredgecolor="green", markerfacecolor="green")
        x_list = [cv_points[i]-2,cv_points[i]+2]
        tan_i  = get_tan_func(activation, cv_points[i])[0]
        plt.plot(x_list,[tan_i(xi) for xi in x_list],'--',color = 'green')

## Mostrar ejes
#plt.axhline(0, color="black", linewidth = 1.5)
#plt.axvline(0, color="black", linewidth = 1.5)

## Limites de los ejes
plt.xlim(lb-0.1, ub+0.1)
#plt.xlim(0.5, 1.5)
plt.ylim(max(-0.05,f_lb-0.05), min(1.05,f_ub+0.05))
#plt.ylim(0.5,1.5)

## Plot y/o guardar como imagen
plt.savefig('activacion.png', dpi=500, bbox_inches='tight')
plt.show()