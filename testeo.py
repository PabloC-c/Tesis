from functions import *

activation = 'sigmoid'

vectorlb = [-1,-1]
vectorub = [2,2]
w = [2,0.5]
b = 1
bounds = (-(b+sum(w[i]*vectorlb[i] if w[i] >= 0 else w[i]*vectorub[i]  for i in range(len(w)))),b+sum(w[i]*vectorub[i] if w[i] >=0 else w[i]*vectorlb[i] for i in range(len(w))))
activ = get_activ_func(activation)
print(bounds)
cc = calculate_cc_point(activation,bounds)