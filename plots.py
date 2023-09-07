import os
import sys
import pandas as pd
from torchvision import datasets, transforms
from functions import *

activation = 'sigmoid'

bounds = (2.5,2.5)
lb,ub = -bounds[0],bounds[1]

cc = calculate_cc_point(activation,bounds)
cv = calculate_cv_point(activation,bounds)

f_lb = calculate_activ_func(activation, lb)
f_ub = calculate_activ_func(activation, ub)
f_cc = calculate_activ_func(activation, cc)
f_cv = calculate_activ_func(activation, cv)

k = 300
x    = [(lb)+i*(ub-lb)/k for i in range(k+1)]
x_cc = [(lb)+i*(cc-lb)/k for i in range(k+1)] 
x_cv = [(cv)+i*(ub-cv)/k for i in range(k+1)] 

plt.plot(x,[calculate_activ_func(activation, xi) for xi in x])
plt.plot(x_cc,[calculate_tan_func(activation,xi,cc) for xi in x_cc],'--')
plt.plot(x_cv,[calculate_tan_func(activation,xi,cv,) for xi in x_cv],'--')

plt.plot(lb, f_lb, marker="o", markersize=5, markeredgecolor="orange", markerfacecolor="orange")
plt.plot(ub, f_ub, marker="o", markersize=5, markeredgecolor="green", markerfacecolor="green")
plt.plot(cc, f_cc, marker="o", markersize=5, markeredgecolor="orange", markerfacecolor="orange")
plt.plot(cv, f_cv, marker="o", markersize=5, markeredgecolor="green", markerfacecolor="green")

plt.axhline(0, color="black", linewidth = 1.5)
#plt.axvline(0, color="black", linewidth = 1.5)

plt.xlim(lb-0.1, ub+0.1)
plt.ylim(max(-0.05,f_lb-0.05), min(1.05,f_ub+0.05))

#plt.savefig('activacion.png', dpi=300, bbox_inches='tight')
plt.show()