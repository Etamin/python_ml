import numpy as np
import random

def randData():
    x = np.arange(-1,1,0.02)
    y = [2*a+3 for a in x]  # 直线
    xa = []; ya = []
    for i in range(len(x)):
        d = np.float(random.randint(90,120))/100
        ya.append(y[i]*d)
        xa.append(x[i]*d)
    return xa,ya

def LS_line(X,Y, lam = 0.01):
    X = np.array(X)
    X = np.vstack((np.ones((len(X),)),X)) 
    X = np.mat(X).T     
    Y = np.mat(Y).T     
    M, N = X.shape
    I = np.eye(N, N)    
    
    theta = ((X.T * X + lam*I)**-1)*X.T*Y       
    theta = np.array(np.reshape(theta,len(theta)))[0]
    return theta    

x,y=randData()
print (LS_line(x,y))


