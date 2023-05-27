"""

PHY407F 
LAB #6
QUESTION #1

Authors: Idil Yaktubay and Souren Salehi
October 2022

"""

"""

Question 1(b) PSEUDOCODE

CODE THAT USES THE VERLET METHOD TO UPDATE THE POSITIONS OF TWO PARTICLES IN
A LENNARD-JONES POTENTIAL FOR THREE SETS OF INITIAL CONDITIONS AND PLOTS 
THE TRAJECTORIES FOR THESE SETS OF INITIAL CONDITIONS

Author: Idil Yaktubay

"""


# Import necessary modules
	# Import numpy
	# Import matplotlib.pyplot


# Set relevant physical constants
	# Set Van der Waals radius
	# Set potential well depth
	# Set particle mass


# Define a function that calculates the accelaration of a particle due to
# the presence of another particle, given particle positions
# NOTE: This will be a VECTOR function (takes vectors and returns vectors) 
# for code efficiency
# Inside the function body:
	# Set x and y positions of particle one
	# Set x and y positions of particle two
	# From x,y components of positions, calculate and set separation 
	# distance
	# Using separation distance and input vectors, calculate and set
	# acceleration
	# Return acceleration vector


# Define a function that calculates Lennard-Jones potential for a 
# two-particle interaction, given a separation distance
# NOTE: This will be a SCALAR function (takes a scalar and returns
# a scalar)
# Inside the function body:
	# Using separation distance, calculate and set potential
	# Return potential


# Create time array
	# Set number of time steps to 100
	# Set size of time steps to 0.01
	# Set start time
	# Set end time
	# With above information, create time array


# Set initial conditions for both particles (for scenario i., ii., or iii.)


# VERLET METHOD
# 1) Set empty lists to accumulate positions and velocities for both
#    particles
# 2) Initialize Verlet variables that will be updates
#		-- Set initial position vector of particle 1
#		-- Set initial Verlet velocity (v(t + h/2)) for particle 1
#		-- Set initial velocity for particle 1
#		-- REPEAT above three steps for particle 2
# 3) Perform Verlet method by updating the initialized Verlet variables
#    using eq. 8-12 on Lab manual


# With the results of the Verlet method, plot trajectories


# ENERGY CALCULATIONS (To be able to answer part 1(c))
# Convert position and velocity lists to numpy arrays
# From the results of Verlet method, calculate and set separation distances 
# at each time point
# Calculate and set total kinetic energy at each time point
# Calculate and set total potential energy at each time point
# Calculate and print total energy of system at each time point


# RUN THE SAME CODE FOR THE REMAINING INITIAL CONDITIONS

"""

Question 1(b) REAL PYTHON CODE

CODE THAT USES THE VERLET METHOD TO UPDATE THE POSITIONS OF TWO PARTICLES IN
A LENNARD-JONES POTENTIAL FOR THREE SETS OF INITIAL CONDITIONS AND PLOTS 
THE TRAJECTORIES FOR THESE SETS OF INITIAL CONDITIONS 

NOTE: The last section of this code is for the purposes of question 1(c),
which the reader may ignore.

Authors: Idil Yaktubay and Souren Salehi

"""


# Import necessary modules
import numpy as np
import matplotlib.pyplot as plt


# Set relevant physical constants
sigma = 1.0 # Van der Waals radius
epsilon = 1.0 # Potential well depth
m = 1.0 # Particle mass


# Define a function that calculates acceleration
def acc(r1, r2):
    """
    Return accelaration acc of the particle at position r1 due to the 
    presence of the particle at position r2 in a 2-D molecular interaction

    Parameters
    ----------
    r1: NUMPY.NDARRAY
       2-D vector position of particle one
    r2: NUMPY.NDARRAY
        2-D vector position of particle two

    Returns
    -------
    acc: NUMPY.NDARRAY
        2-D vector accelaration of particle one due to particle two
        
    Preconditions
    -------------
        numpy.shape(r1) == numpy.shape(r2) == (2,) MUST be True

    """
    # Particle one position components
    x1 = r1[0]
    y1 = r1[1]
    
    # Particle two position components
    x2 = r2[0]
    y2 = r2[1]
    
    # Separation distance
    r = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    
    acc = (24*epsilon/m)*(sigma**6/r**6)*(1/r**2)*(2*(sigma**6/r**6) - 1) \
        *(r1 - r2)
        
    return acc


# Define a function that calculates Lennard-Jones potential
def V(r):
    """
    Return the Lennard-Jones potential V of a particle one at separation r from 
    a particle two in a 2-D molecular interaction
    
    Parameters
    ----------
    r: FLOAT
        Separation distance between particle one and particle two

    Returns
    -------
    V : FLOAT
        Lennard-Jones potential of particle one

    """
    V = 4*epsilon*((sigma/r)**12 - (sigma/r)**6)
    return V


# Set an array of time points
N = 100 # Number of time steps
dt = 0.01 # Size of time steps
a = 0.0 # Start time
b = N*dt + dt # End time (+ dt to ensure there are 100 steps and not 99)
t_points = np.arange(a, b, dt) # Time array


# Set initial velocities
v1_i = np.array([0.0, 0.0])
v2_i = np.array([0.0, 0.0])


# Set initial positions
r1_i = np.array([2., 3.])
r2_i = np.array([3.5, 4.4])


# Set empty lists to accumulate x, y, and v points for particle one
x1_points = []
y1_points = []
v1_corresponding = []


# Set empty lists to accumulate x, y, and v points for particle two
x2_points = []
y2_points = []
v2_corresponding = []


# Initialize variables for Verlet Method
# Particle one:
r1 = r1_i # r(t=0)
v1 = v1_i + (dt/2)*acc(r1_i, r2_i) # v(t=0 + h/2) for first step
v1_c = v1_i # v(t=0)
# Particle two:
r2 = r2_i # r(t=0)
v2 = v2_i + (dt/2)*acc(r2_i, r1_i) # v(t=0 + h/2) for first step
v2_c = v2_i # v(t=0)


# Perform Verlet Method
for i in range(t_points.size):
    # Record particle one position and velocity
    x1_points.append(r1[0])
    y1_points.append(r1[1])
    v1_corresponding.append(v1_c)
    
    # Record particle two position and velocity
    x2_points.append(r2[0])
    y2_points.append(r2[1])
    v2_corresponding.append(v2_c)
    
    # Update positions
    r1 = r1 + dt*v1
    r2 = r2 + dt*v2
    
    k1 = dt*acc(r1, r2)
    k2 = dt*acc(r2, r1)
    
    # Update velocities
    v1_c = v1 + 0.5*k1
    v2_c = v2 + 0.5*k2
    
    # Update Verlet velocities
    v1 = v1 + k1
    v2 = v2 + k2


# Convert everything to numpy.array to ease calculations
x1_points = np.array(x1_points)
y1_points = np.array(y1_points)
v1_corresponding = np.array(v1_corresponding)

x2_points = np.array(x2_points)
y2_points = np.array(y2_points)
v2_corresponding = np.array(v2_corresponding)


# Plot trajectories - UNCOMMENT TO GENERATE PLOTS
# plt.title('2-D Trajectories of Two Particles in a Lennard-Jones Potential', \
#           fontsize=13)
# plt.plot(x1_points, y1_points, '.', label='Particle 1')
# plt.plot(x2_points, y2_points, '.', label='Particle 2')
# plt.xlabel(r'$x$ (arbitary distance units)', fontsize=13)
# plt.ylabel(r'$y$ (arbitary distance units)', fontsize=13)
# plt.grid(linestyle='dotted')
# plt.legend(prop={'size': 13})


# PART 1(C)
# Find and print total energy of system at every time point
r_values = np.sqrt((x1_points - x2_points)**2 - (y1_points - y2_points)**2)
E_k = 0.5*m*(v1_corresponding[:, 0]**2 + v2_corresponding[:, 0]**2 \
    + v1_corresponding[:, 1]**2 + v2_corresponding[:, 1]**2) 
E_potential = V(r_values)
total_E = E_k + E_potential
# print('The total energy at each time point is: \n', total_E)
