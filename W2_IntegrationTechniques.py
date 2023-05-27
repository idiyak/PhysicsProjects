'''

PHY407F 
LAB #2
QUESTION #2

Authors: Souren Salehi and Idil Yaktubay
September 2022


'''
'''

Question 2(b) PSEUDOCODE

CODE THAT ESTIMATES AN INTEGRAL USING THE TRAPEZOIDAL AND SIMPSON'S RULE 
USING 4 SLICES

Author: Idil Yaktubay

'''

# Import numpy
# Import value of pi from math module

# Set the bounds of integration, the number of slices, and the width of slices
# Number of slices
# Lower bound
# Upper bound
# Width of slices

# Set the value of pi

# Define the integrand function

# Integration using the trapezoidal rule
# 1) Define the terms outside the summation in Trapezoidal formula
# 2) Add the terms inside the summation by iteration
# 3) Set the estimated integral
# 4) Print findings for Trapezoidal rule

# Integration using Simpson's rule
# 1) Define the terms outside the summations in Simpson's formula
# 2) Add the terms inside the summation for odd values of k by iteration
# 3) Add the terms inside the summation for even values of k by iteration
# 4) Set the estimated integral
# 5) Print findings for Simpson's rule

'''

Question 2(b) REAL PYTHON CODE

CODE THAT ESTIMATES AN INTEGRAL USING THE TRAPEZOIDAL AND SIMPSON'S RULE 
USING 4 SLICES

Author: Souren Salehi 

'''

# Import numpy
import numpy as np
# Import the value of pi from math module
from math import pi

# Set the bounds of integration, the number of slices, and the width of slices
n=2
N= 2**n # Number of slices
a=0 # Lower bound
b=1 # Upper bound
h=(b-a)/N # Width of slices

# Set the value of pi
pii = pi

# Define the integrand function
def f(x):
    return 4/(1+x**2)

# Integration using the trapezoidal rule
s = 0.5*f(a) + 0.5*f(b) # Terms outside the summation in Trapezoidal formula

# Add the terms inside the summation 
for k in range(1,N):
    s += f(a+k*h)
    
I1 = h*s # Estimated integral with Trapezoidal rule

# Print findings for Trapezoidal rule
print('QUESTION 2(B) \n')
print('Actual value of integral = ', pii, '...')
print('Integral for trapezoidal rule with 4 slices = ',I1)


#Integration using Simpson's rule
p = f(a)+f(b) # Terms outside the summation in Simpson's rule

for k in range(1,N,2): # Odd values of k
    p += 4*f(a+k*h)
    
for k in range(2,N,2): # Even values of k
    p+= 2*f(a+k*h)

I2 = (1/3)*h*p # Estimated integral with Simpson's rule
print("Integral with Simpson's rule with 4 slices = ", I2, '\n')

'''

Question 2(d) PSEUDOCODE

CODE THAT ESTIMATES THE ERROR OF A TRAPEZOIDAL INTEGRATION 
USING 16 AND 32 SLICES USING THE PRACTICAL METHOD

Author: Idil Yaktubay

'''
# Trapezoidal rule for the function using 16 slices

# Set the number of slices to 16

# Integration using the Trapezoidal rule
# 1) Set the terms outside the summation in Trapezoidal formula
# 2) Add the terms inside the summation by iteration
# 3) Set the estimated integral for 16 slices

# Repeat the trapezoidal rule for 32 slices

# Set the number of slices to 32

# Integration using the Trapezoidal rule
# 1) Set the terms outside the summation in Trapezoidal formula
# 2) Add the terms inside the summation by iteration
# 3) Set the estimated integral for 32 slices

# Find the error for the integration with 32 slices using 5.28 in the textbook

# Print findings

'''

Question 2(d) REAL PYTHON CODE

CODE THAT ESTIMATES THE ERROR OF A TRAPEZOIDAL INTEGRATION 
USING 16 AND 32 SLICES USING THE PRACTICAL METHOD

Author: Souren Salehi

'''

# Trapezoidal rule for the function using 16 slices
# Set the number of slices to 16
N = 16 
h = (b-a)/N # Width of the slices

s1 = 0.5*f(a) + 0.5*f(b) # Terms outside the summation in Trapezoidal formula
# Add the terms inside the summation
for k in range(1,N):
    s1 += f(a+k*h)
I16 = h*s1 # Set the estimated integral for 16 slices

# Repeat the trapezoidal rule with 32 slices
# Set the number of slices to 32
N=32
h = (b-a)/N # Width of the new slices

s2 = 0.5*f(a) + 0.5*f(b) # Terms outside the summation in Trapezoidal formula
# Add the terms inside the summation
for k in range(1,N):
    s2 += f(a+k*h)
I32 = h*s2 # Set the estimated integral for 32 slices

# Find the error using eqn 5.28 in the textbook
error = (1/3)*(I32-I16)

# Print findings
print("QUESTION 2(D)")
print("Practical estimation of error for Trapezoidal rule for 32 slices = ", \
      error)
