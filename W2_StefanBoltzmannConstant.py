"""
Authors: Souren Salehi and Idil Yaktubay
"""

import numpy as np
from scipy import constants

'''
We're breaking the integration to 0-0.5 and 0.5-inf and using substitution for 
the latter to remove the infinite boundary

The integrals are done for 2 sets of steps (N and 2N) so that later
we may calculate the error using the practical method
'''

#setting the number of steps and boundaries for the integrals 
N1= 50
#as the integral is not defined at 0 we take it to be very small
a=1*10**(-6)
b=0.5

#we change the boundaries from 0.5-inf to 2-0 using the substitution 1/t
c=2
d=1*10**(-2)
N2=200


#finding the step sizes for each segment
h1 = (b-a)/N1
h2 = (c-d)/N2

#defining the function for the integral from 0 to 0.5
def f1(x):
    return (x**3)/(np.exp(x)-1)


#defining the composition of our function with 1/t for improper integral
def f2(x):
    return 1/((x**5)*(np.exp(1/x)-1))

#using the simpson rule of integration for the 0-0.5 segment of integration
s1 = f1(a)+f1(b)
for k in range(1,N1,2):
    s1 += 4*f1(a+k*h1)
    
for k in range(2,N1,2):
    s1+= 2*f1(a+k*h1)

I1 = (1/3)*h1*s1 

#using the simpson rule of integration for the 0.5-inf segment

s2 = f2(c)+f2(d)
for k in range(1,N2,2):
    s2 += 4*f2(d+k*h2)
    
for k in range(2,N2,2):
    s2+= 2*f2(d+k*h2)
I2 = (1/3)*h2*s2 


#repeating the process for N/2 steps to be able to calculate the error in 
#the integral we found above
N1= 25
a=1*10**(-6)
b=0.5
c=2
d=1*10**(-2)
N2=100

h1 = (b-a)/N1
h2 = (c-d)/N2

def f1(x):
    return (x**3)/(np.exp(x)-1)

def f2(x):
    return 1/((x**5)*(np.exp(1/x)-1))

s1 = f1(a)+f1(b)
for k in range(1,N1,2):
    s1 += 4*f1(a+k*h1)
    
for k in range(2,N1,2):
    s1+= 2*f1(a+k*h1)

I12 = (1/3)*h1*s1 


s2 = f2(c)+f2(d)

for k in range(1,N2,2):
    s2 += 4*f2(d+k*h2)
    
for k in range(2,N2,2):
    s2+= 2*f2(d+k*h2)

I22 = (1/3)*h2*s2 


#we may use the practical method of finding the error using the 2 integrals we 
#solved for above for each respective segment
e1 = (1/15)*(I1-I12)
e2= (1/15)*(I2-I22)
#using propagation of errors to find the sum for the total integral
et = abs(e1+e2)

#the total integral from 0-inf
I = I1+I2
print('The value of the integral is given by', np.round(I,4), '+/-', np.round(et,4))


#using the integral we found find the boltzman constant sigma
sigma = 2*(constants.pi)*(constants.k**4)*I/((constants.h**3)*(constants.c**2))

#using propagation of error we can find the error in our constant 
dsigma = 2*(constants.pi)*(constants.k**4)*et/((constants.h**3)*(constants.c**2))

print('calculated value of boltzmann constant is given by',np.round(sigma,13),'+/-',np.round(dsigma,13))
print('the true value of boltzmann constant is', constants.sigma)

#the function to find total energy per unit area given the temperature T
def W(T):
    return sigma*T**4
