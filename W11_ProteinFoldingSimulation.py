"""
Authors: Idil Yaktubay and Souren Salehi
"""

import numpy as np
from random import random, randrange, seed
import matplotlib.pyplot as plt
'''

Q1a
Modification of the code from the textbook to run over different values of 
tau whilst taking different paths 
'''

#seed to keep the points constant 
seed(20)
N = 25 
R = 0.02

Tmax =10.0
Tmin=1e-3
taus = [5e3, 1e4, 5e4]
seeds = [1, 20, 30]


# Function to calculate the magnitude of a vector
def mag(x):
    return np.sqrt(x[0]**2+x[1]**2)

# Function to calculate the total length of the tour
def distance():
    s = 0.0
    for i in range(N):
        s+= mag(r[i+1]-r[i])
    return s 

# Choose N city locations and calculate the initial distance
r = np.empty([N+1,2],float)
for i in range(N):
    r[i,0] = random()
    r[i,1] = random()
r[N] = r[0]
D = distance()

Dlis =[]
#loop to run the problem over different time constants 
for q in taus:
    #loop to take different path
    for p in seeds:
        t = 0 
        T = Tmax 
        tau = q
        seed(p)

        while T>Tmin:
            t+=1
            #cooling function
            T = Tmax*np.exp(-t/tau)
            
            # Choose two cities to swap and make sure they are distinct
            i,j = randrange(1,N), randrange(1,N)
            while i==j:
                i,j = randrange(1,N),randrange(1,N)
             
            # Swap them and calculate the change in distance
            oldD = D
            r[i,0],r[j,0] = r[j,0],r[i,0]
            r[i,1],r[j,1] = r[j,1],r[i,1]
            D = distance()
            deltaD = D-oldD
            
            if random() > np.exp(-deltaD/T):
                r[i,0],r[j,0] = r[j,0],r[i,0]
                r[i,1],r[j,1] = r[j,1],r[i,1]
                D = oldD
                
        Dlis.append(D)
        plt.figure()
        plt.title(('solution with seed '+str(p)+ ' and tau '+ str(tau)+ ' distance '+str(np.round(D,3))), size = 18)
        plt.xlabel('x', size=18)
        plt.ylabel('y', size=18)
        plt.scatter(r[:, 0], r[:, 1], s=120)
        plt.plot(r[:, 0], r[:, 1])
        plt.show()
        

'''
Q1b
Using Simulated Annealing Optimization to minimise a function 

'''
    

#seed to get constant results
seed(4)
#setting the constants 
Tmax=1
Tmin=1e-5
tau = 1e4
sigma = 1

#defining the gaussian to get the x and y variations from
def gauss():
    r = np.sqrt(-2*(sigma**2)*np.log(1-random()))
    theta = 2*np.pi*random()
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    return x,y

#function that we want to minimise
def f(x,y):
    s = x**2 - np.cos(4*np.pi*x) +(y-1)**2
    return s 

#starting point for the loop
x=2
y=2
t=0
T = Tmax

#emplty list to record the evolution of the x and y values
xlis = []
ylis = []
tlis = []

#main loop
while T>Tmin:
    tlis.append(t)
    t+=1
    #cooling function
    T = Tmax*np.exp(-t/tau)
    
    #record the values of x and y during runtime
    xlis.append(x)
    ylis.append(y)
    
    #updating x and y values
    dx,dy = gauss()
    f0 = f(x,y)
    x1 = x+ dx
    y1 = y+ dy
    f1 = f(x1,y1)
    df =  f1 - f0
    if random() <= np.exp(-df/T):
        x = x1
        y=y1

print("for Q1b, the solutions are", 'x=', round(x,5), 'y=',round(y,5))
#plot of the evolution of x and y values
plt.scatter(tlis,xlis, s=7)
plt.suptitle("plot of x over the run time")
plt.xlabel('t values')
plt.ylabel('x values')

plt.show()

plt.scatter(tlis,ylis, s=7)
plt.suptitle("plot of y over the run time")
plt.xlabel('t values')
plt.ylabel('y values')

plt.show()


'''
Q1C 
We repeat the same method for the second function, with modifications for 
the range of solution

Author: Souren Salehi
'''

#function that we want to minimise
def f(x,y):
    s = np.cos(x) + np.cos(np.sqrt(2)*x) +np.cos(np.sqrt(3)*x) +(y-1)**2
    return s 

#starting point for the loop
x=15
y=2
t=0
T = Tmax

#emplty list to record the evolution of the x and y values
xlis = []
ylis = []

#main loop
while T>Tmin:
    t+=1
    #cooling function
    T = Tmax*np.exp(-t/tau)
    
    #record the values of x and y during runtime
    xlis.append(x)
    ylis.append(y)
    
    #updating x and y values
    dx,dy = gauss()
    f0 = f(x,y)
    x1 = x+ dx
    y1 = y+ dy
    #setting the range we require x and y to be within
    while not (0<x1<50 and -20<y1<20) :
        dx,dy = gauss()
        x1 = x+ dx
        y1 = y+ dy
        
    f1 = f(x1,y1)
    df =  f1 - f0
    if random() <= np.exp(-df/T):
        x = x1
        y=y1
      
print("for Q1c, the solutions are", 'x=', round(x,5), 'y=',round(y,5))
#plot of the evolution of x and y values
plt.scatter(tlis,xlis, s=7)
plt.suptitle("plot of x over the run time")
plt.xlabel('t values')
plt.ylabel('x values')

plt.show()

plt.scatter(tlis,ylis, s=7)
plt.suptitle("plot of y over the run time")
plt.xlabel('t values')
plt.ylabel('y values')

plt.show()

plt.show()
