from functions import *
from concave_envGPT import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.style.use('seaborn-poster')

activation = 'sigmoid'

vectorlb = [-1,-1]
vectorub = [1,1]

## Caso interesante: w = [7,7], b = -2.5

n = 2
w = [1,1]
b = -0.5


bounds = (-(b+sum(w[i]*vectorlb[i] if w[i] >= 0 else w[i]*vectorub[i]  for i in range(len(w)))),b+sum(w[i]*vectorub[i] if w[i] >=0 else w[i]*vectorlb[i] for i in range(len(w))))
activ = get_activ_func(activation)
cc = calculate_cc_point(activation,bounds)
cv = calculate_cv_point(activation,bounds)

def activ_compo(activ,w,b,vector):
    aux = b+sum(w[i]*vector[i] for i in range(len(vector)))
    return activ(aux)

def derv_compo(derv,w_var,b,var):
    aux = b+w_var*var
    return w_var*derv(aux)

def edge_cc(inflex,ub,activ,derv,w_var,b,var0,z0):
    auxlb = (inflex-b)/w_var
    auxub = ub
    while True:
        cc    = (auxlb+auxub)/2
        slope =  (activ_compo(activ,[w_var],b,[cc])-z0)/(cc-var0)
        #slope = (activ(b+w[0]*x4+w[1]*y4)-activ(b+w[0]*x3+w[1]*y3))/(y4-y3)
        derv_cc = derv_compo(derv,w_var,b,cc)
        #der = w[1]*deriv(b+w[0]*x4+w[1]*y4)
        if np.abs(slope-derv_cc) < 1e-4:
            return cc
        if slope < derv_cc:
            if np.abs(cc-ub) < 1e-4:
                return cc
            else:
                auxlb = cc
        else:
            auxub = cc

def oned_env_triangle_eval(zlb,zcc,x,y,bounds):
    return zlb+((zcc-zlb)/(cc+bounds[0]))*(b+w[0]*x+w[1]*y+bounds[0])

def edge_proyection(x0,y0,x,y,triangle_points):
    for i in range(2):
        if triangle_points[0][i] == triangle_points[1][i]:
            break
    if i == 0:
        x_proy = triangle_points[0][i]
        y_proy = y0+((y-y0)/(x-x0))*(x_proy-x0)
    else:
        y_proy = triangle_points[0][i]
        x_proy = x0+((x-x0)/(y-y0))*(y_proy-y0)
    return x_proy,y_proy
    

def edge_triangle_eval(x0,y0,triangle_points,x,y,activ,w,b,bounds):
    x_proy,y_proy = edge_proyection(x0,y0,x,y,triangle_points)
    edge = b+w[0]*x_proy+w[1]*y_proy
    x1,y1 = triangle_points[0]
    z1 = activ_compo(activ,w,b,[x1,y1])
    x2,y2 = triangle_points[1]
    z2 = activ_compo(activ,w,b,[x2,y2])
    if x1 == x2:
        z_proy = z1+((z2-z1)/(y2-y1))*(y_proy-y1)
    else:
        z_proy = z1+((z2-z1)/(x2-x1))*(x_proy-x1)
    
    z0 = activ_compo(activ,w,b,[x0,y0])
    return z0+((z_proy-z0)/(edge+bounds[0]))*(b+w[0]*x+w[1]*y+bounds[0])

def calculate_area(x0,y0,triangle_points):
    x1,y1 = triangle_points[0]
    x2,y2 = triangle_points[1]
    return 0.5 * np.abs(x0*(y1 - y2) + x1*(y2 - y0) + x2*(y0 - y1))

def is_point_inside_triangle(x0, y0,triangle_points, x, y):
    # Área del triángulo original
    area_original = calculate_area(x0,y0,triangle_points)
    
    x1,y1 = triangle_points[0]
    x2,y2 = triangle_points[1]
    
    # Áreas de los triángulos formados con el punto adicional
    area1 = calculate_area(x,y,[[x1,y1],[x2,y2]])
    area2 = calculate_area(x,y,[[x0,y0],[x2,y2]])
    area3 = calculate_area(x,y,[[x0,y0],[x1,y1]])

    # Suma de las áreas pequeñas
    area_sum = area1 + area2 + area3

    # Comprobar si el punto está dentro o en el borde del triángulo
    return np.abs(area_sum - area_original) < 1e-9  # Consideramos una pequeña tolerancia  

def env(x,y,vectorlb,vectorub,cc,w,b,bounds,triangle_list):
    xlb,ylb = vectorlb
    xub,yub = vectorub
    zcc = activ(cc)
    z = activ(b+w[0]*x+w[1]*y)
    if zcc<=z:
        return z
    zlb = activ(b+w[0]*xlb+w[1]*ylb)
    
    ## Triangulo envoltura 1D
    
    triangle_points = triangle_list[0]
    
    if is_point_inside_triangle(xlb, ylb, triangle_points, x, y):
        return oned_env_triangle_eval(zlb,zcc,x,y,bounds)
    
    if len(triangle_list) > 1:
        for triangle_points in triangle_list[1:]:
            if is_point_inside_triangle(xlb, ylb, triangle_points, x, y):
                return edge_triangle_eval(xlb,ylb,triangle_points,x,y,activ,w,b,bounds)
    if x>y:
        x_proy,y_proy = edge_proyection(xlb,ylb,x,y,[[xub,ylb],[xub,yub]])
        z_proy = activ_compo(activ,w,b,[x_proy,y_proy])
    else:
        x_proy,y_proy = edge_proyection(xlb,ylb,x,y,[[xlb,yub],[xub,yub]])
        z_proy = activ_compo(activ,w,b,[x_proy,y_proy])
    return zlb+((z_proy-zlb)/(b+w[0]*x_proy+w[1]*y_proy+bounds[0]))*(b+w[0]*x+w[1]*y+bounds[0])
            
def hcc(x,y,vectorlb,vectorub,cc,w,b,bounds):
    xlb,ylb = vectorlb
    xub,yub = vectorub
    zcc = activ(cc)
    z = activ(b+w[0]*x+w[1]*y)
    if zcc<z:
        return z
    else:
        zlb = activ(b+w[0]*xlb+w[1]*ylb)
        return zlb+((zcc-zlb)/(cc+bounds[0]))*(b+w[0]*x+w[1]*y+bounds[0])
    
def perspective(x,y,vectorub,vectorlb):
    if y == vectorlb[1]:
        return 0
    return activ((b+w[0]*x+w[1]*y)/((y-vectorlb[1])/(vectorub[1]-vectorlb[1])))*((y-vectorlb[1])/(vectorub[1]-vectorlb[1]))


plot_flag = True

if plot_flag:
    x = np.linspace(vectorlb[0], vectorub[0], 300)
    y = np.linspace(vectorlb[1], vectorub[1], 300)
    
    func = np.vectorize(lambda x, y: activ(w[0]*x+w[1]*y+b))
    
    X, Y = np.meshgrid(x, y)
    Z = func(X, Y)     
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlim([vectorlb[0],vectorub[0]])
    ax.set_ylim([vectorlb[1],vectorub[1]])
    ax.set_zlim([-0.1, 1.1])
    
    surf = ax.plot_surface(X, Y, Z, cmap='coolwarm', edgecolor='none') #Greys
    
    ## Triangulo envoltura 1D

    x1,y1 = None,None
    x2,y2 = None,None
    x3 = None
    x4 = None
    
    if activ(b+w[0]*vectorub[0]+w[1]*vectorlb[1]) <= activ(cc):
        x1 = vectorub[0]
    else:
        y1 = vectorlb[1]
    
    if activ(b+w[0]*vectorlb[0]+w[1]*vectorub[1]) <= activ(cc):
        y2 = vectorub[1]
    else:
        x2 = vectorlb[0]
    if x1 == None:
        x1 = (cc-w[1]*y1-b)/w[0]
    if y1 == None:
        y1 = (cc-w[0]*x1-b)/w[1]
    if x2 == None:
        x2 = (cc-w[1]*y2-b)/w[0]
    if y2 == None:
        y2 = (cc-w[0]*x2-b)/w[1]
    
    triangle_points = [[x1,y1],[x2,y2]]
    triangle_list   = [triangle_points] 
    
    ## Busqueda de triangulos
    
    derv = get_activ_derv(activation)
    x3,y3 = vectorub[0],vectorlb[1]
    if activ(b+w[0]*x3+w[1]*y3) < activ(0):
        triangle_points = [[x3,y3]]
        x4 = x3
        y4 = edge_cc(0,vectorub[1],activ,derv,w[1],b+w[0]*x4,y3,activ_compo(activ,w,b,[x3,y3]))
        triangle_points = [[x3,y3],[x4,y4]]
        triangle_list.append(triangle_points)
    x3,y3 = vectorlb[0],vectorub[1]
    if activ(b+w[0]*x3+w[1]*y3) < activ(0):
        y4 = y3
        x4 = edge_cc(0,vectorub[0],activ,derv,w[0],b+w[1]*y4,x3,activ_compo(activ,w,b,[x3,y3]))
        triangle_points = [[x3,y3],[x4,y4]]
        triangle_list.append(triangle_points)
    
    ## Grafico de puntos
    names = ['lb']
    x_list = [vectorlb[0]]
    y_list = [vectorlb[1]]
    z_list = [activ_compo(activ,w,b,[vectorlb[0],vectorlb[1]])]
    j = 1
    
    for points in triangle_list:
        for point in points:
            x_list.append(point[0])
            y_list.append(point[1])
            z_list.append(activ_compo(activ,w,b,[point[0],point[1]]))
            names.append(f'v{j}')
            j+=1
    
    ax.scatter(x_list,y_list,z_list,color='blue',marker='o',s=100)
    ## Etiquetas de los puntos
    for i in range(len(x_list)):
        ax.text(x_list[i], y_list[i], z_list[i], names[i], fontsize=12, color='red')
        
    ## P^cc
    
    ax.plot3D([x1,x2], [y1,y2], [activ(cc),activ(cc)], color = 'black', linewidth = 5)
    
    ## Recta de inflexion
    
    x1,y1 = None,None
    x2,y2 = None,None
    
    if activ(b+w[0]*vectorub[0]+w[1]*vectorlb[1]) <= activ(0):
        x1 = vectorub[0]
    else:
        y1 = vectorlb[1]
    
    if activ(b+w[0]*vectorlb[0]+w[1]*vectorub[1]) <= activ(0):
        y2 = vectorub[1]
    else:
        x2 = vectorlb[0]
    if x1 == None:
        x1 = (-w[1]*y1-b)/w[0]
    if y1 == None:
        y1 = (-w[0]*x1-b)/w[1]
    if x2 == None:
        x2 = (-w[1]*y2-b)/w[0]
    if y2 == None:
        y2 = (-w[0]*x2-b)/w[1]

    ax.plot3D([x1,x2], [y1,y2], [activ(0),activ(0)], color = 'b', linewidth = 5)    
    
    ## Grafico de triangulos
    
    xlb,ylb = vectorlb[0],vectorlb[1]
    
    for triangle_points in triangle_list:
        x1,y1 = triangle_points[0]
        x2,y2 = triangle_points[1]
        #ax.plot3D([xlb,x1], [ylb,y1], [activ_compo(activ,w,b,[xlb,ylb]),activ_compo(activ,w,b,[x1,y1])], color = 'black', linewidth = 5) 
        #ax.plot3D([xlb,x2], [ylb,y2], [activ_compo(activ,w,b,[xlb,ylb]),activ_compo(activ,w,b,[x2,y2])], color = 'black', linewidth = 5) 
        #ax.plot3D([x1,x2], [y1,y2], [activ_compo(activ,w,b,[x1,y1]),activ_compo(activ,w,b,[x2,y2])], color = 'black', linewidth = 5)  
    
    
    
    env_func = np.vectorize(lambda x, y: env(x,y,vectorlb,vectorub,cc,w,b,bounds,triangle_list))
    hcc_func = np.vectorize(lambda x, y: hcc(x,y,vectorlb,vectorub,cc,w,b,bounds))
    X, Y = np.meshgrid(x, y)
    Z = env_func(X, Y)
    Zhcc = hcc_func(X, Y)
    surf = ax.plot_surface(X, Y, Z, cmap='Reds', edgecolor='none',alpha = 0.4)
    #surf = ax.plot_surface(X, Y, Zhcc, cmap='Greys', edgecolor='none') #Greys
    # Mostrar el gráfico
    
    func_aux = np.vectorize(lambda x, y: perspective(x,y,vectorub,vectorlb))
    Z_aux = func_aux(X,Y)
    #surf = ax.plot_surface(X, Y, Z_aux, cmap='Greens', edgecolor='none')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('')
    
    ax.set_xlim([vectorlb[0],vectorub[0]])
    ax.set_ylim([vectorlb[1],vectorub[1]])
    ax.set_zlim([-0.1, 1.1])
    
    plt.show()