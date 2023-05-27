"""
PHY407H F 
LAB #10
QUESTION #2

Authors: Idil Yaktubay
November 2022
"""

"""
Question 2 PSEUDOCODE

CODE THAT ESTIMATES THE VOLUME OF A 10-D UNIT HYPERSPHERE USING MONTE CARLO 
INTEGRATION WITH THE VOLUME OF A 10-D UNIT HYPERCUBE
"""

# Import numpy
# Import necessary functions from random
# Import gamma function from scipy.special


# Define a function that gives the value of the hypersphere function
# Inside the function body:
	# Calculate length of 10-D position vector
	# If this length is less or equal to 1, return 1
	# Otherwise, return 0


# Set dimension to 10
# Set the number of random points to genarate to a million
# Set the side length of 10-D hypercube to 2 
# Set the volume of hypersphere to 2^10
# Set sphere radius to 1


# Set the sum of f(r_i) to zero to be updated
# Set the sum of f(r_i)^2 to zero to be updated
# Create an empty 10-D vector to place random points in


# Iterate for one million times, and with each iteration...
	# Assign random values for the components of vector r
	# Calculate hypersphere function value at this r
		# And add to f(r_i) sum
	# Calculate square of hypersphere function value at this r
		# And add to f(r_i)^2 sum


# Find and set variance of hypersphere function with the million points
# Find and set the standard deviation from the variance


# Calculate and set the Monte Carlo integral


# Calculate exact volume


# Print the volume estimation and std
# Print the exact volume


"""
Question 2 REAL PYTHON CODE

CODE THAT ESTIMATES THE VOLUME OF A 10-D UNIT HYPERSPHERE USING MONTE CARLO 
INTEGRATION WITH THE VOLUME OF A 10-D UNIT HYPERCUBE
"""


import numpy as np
from random import uniform
from scipy.special import gamma


def f(r):
    """
    

    Parameters
    ----------
    r : NUMPY.NDARRAY
        r is a position vector

    Returns
    -------
    f : TYPE
        f is the value of the hypersphere function

    """
    r_norm = np.linalg.norm(r) # Position vector length
    if r_norm**2 <= 1: 
        f = 1
    else:
        f = 0
    return f


n = 10  # Dimension
N = int(1e6) # Number of random points
L = 2 # Hypercube side length
V_approx = L**n # Hypercube volume
R = 1 # Hypersphere radius


r = np.empty(n) # 10-D position vector (to update)
sum_f = 0
sum_f_squared = 0
for i in range(N):
    for j in range(n):
        r[j] = uniform(-1, 1)
    sum_f += f(r)
    sum_f_squared += f(r)**2


var_f = sum_f_squared/N - (sum_f/N)**2 # Variance
std = V_approx*np.sqrt(var_f)/np.sqrt(N) # Standard deviation 


I = (V_approx/N)*sum_f # Estimated volume
V_exact = (R**n)*(np.pi**(n/2))/gamma(int(n/2) + 1) # Exact volume


print('The approximated volume of a 10D unit hypersphere is: ', I, \
      'with standard deviation', std)
print('The exact volume of a 10D unit hypersphere is: ', V_exact)
