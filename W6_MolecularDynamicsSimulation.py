"""

PHY407F 
LAB #6
QUESTION #2

Authors: Idil Yaktubay and Souren Salehi
October 2022

"""


"""

Question 2(a) PSEUDOCODE

CODE THAT USES THE VERLET METHOD TO UPDATE THE POSITIONS OF 16 PARTICLES IN
A LENNARD-JONES POTENTIAL AND PLOTS THE 2-D PARTICLE TRAJECTORIES 

Author: Idil Yaktubay

"""


# Import necessary modules
	# Import numpy
	# Import matplotlib.pyplot
	# Import acceleration and potential functions from Q1 file


# Set relevant physical constants
	# Set Van der Waals radius
	# Set potential well depth
	# Set particle mass


# Create time array
	# Set number of time steps to 1000
	# Set size of time steps to 0.01
	# Set start time
	# Set end time
	# With above information, create time array


# Set initial conditions for all 16 particles
	# Set number of particles
	# Set domain width
	# Set domain length
	# Set width of uniform particle separation	
	# Set length of uniform particle separation
	# Set possible x values
	# Set possible y values
	# Set array that contains all initial x positions
	# Set array that contains all initial y positions
	# Set initial positions and velocities as vectors


# Define a multi-purpose function that gives user-selected information
# about the molecular system
# ABOUT THIS FUNCTION:
#	- The user will provide the position vectors for ALL particles in 
#	the system at a given time and specify whether to obtain the 
#	acceleration vectors for ALL particles in the system at that time 
# 	or the separation distance from each particle to the remaining 
# 	particles at that time 
#	- The reason for defining this function in a multi-purpose manner
#	is because we can obtain separation distances AS we calculate 
#	accelerations, instead of passing through the data twice
# INSIDE THE FUNCTION BODY:
	# Set empty lists to accumulate separation distances and accelerations
	# Iterate over each particle to find its separation distance from 
	# remaining particles and acceleration due to remaining particles
	# Convert separation distance and acceleration lists to numpy arrays
	# Return appropriate array depending on the user's selection


# VERLET METHOD
# 1) Set empty lists to accumulate positions, velocities, and separation 
#    distances for ALL particles at ALL time points
# 2) Initialize vector variables for Verlet method (position, velocity, and 
#    Verlet velocity)
# 3) Perform Verlet method by updating the initialized Verlet variables using 
#    eq. 8-12 on Lab manual


# Plot particle trajectories 


"""

Question 2(a) REAL PYTHON CODE

CODE THAT USES THE VERLET METHOD TO UPDATE THE POSITIONS OF 16 PARTICLES IN
A LENNARD-JONES POTENTIAL AND PLOTS THE 2-D PARTICLE TRAJECTORIES 

Authors: Idil Yaktubay and Souren Salehi

"""


# Import necessary modules
import numpy as np
import matplotlib.pyplot as plt 
from LAB06_Q1 import acc, V


# Set relevant physical constants
sigma = 1.0 # Van der Waals radius
epsilon = 1.0 # Potential well depth
m = 1.0 # Particle mass


# Set an array of time points
N = 1000 # Number of time steps
dt = 0.01 # Size of time steps
a = 0.0 # Start time
b = N*dt + dt # End time (+ dt to ensure there are 1000 steps and not 999)
t_points = np.arange(a, b, dt) # Time array


# Set initial positions
n = 16  # Number of particles
Lx = 4.0 # Domain width
Ly = 4.0 # Domain length
dx = Lx/np.sqrt(n) # Width of particle separation
dy = Ly/np.sqrt(n) # Length of particle separation
x_grid = np.arange(dx/2, Lx, dx) # Possible x values
y_grid = np.arange(dy/2, Ly, dy) # Possible y values
xx_grid, yy_grid = np.meshgrid(x_grid, y_grid)
x_initial = xx_grid.flatten()
y_initial = yy_grid.flatten()


# Set initial conditions as vectors
v_initial = np.zeros([x_initial.size, 2]) 
r_initial = np.transpose(np.array([list(x_initial), list(y_initial)]))



def system_info(r, acc_or_dis):
    """
    
    POSSIBLE RETURNS:
    -----------------
        
    1) Return accelerations acc_i for ALL particles i given ALL position 
    vectors r and acc_or_dis is 'acc'
    
    2) Return separation distances r_all_i for ALL particles i given ALL 
    position vectors r and acc_or_dis is 'dis'
     
    Parameters
    ----------
    
    r : NUMPY.NDARRAY
        Contains the position vectors for each particle in a molecular 
        system
        
    acc_or_dis : STR
        Specifies whether the user wants accelerations or separation distances

    Returns
    -------
    
    1) acc_i: NUMPY.NDARRAY
       -- Contains the 2-D accelaration vector for each particle
       -- For example, acc_i[0] gives the 2-D acceleration of the 0th particle
          due to the presence of the remaining particles in the system
          
    2) r_all_i: NUMPY.NDARRAY
       -- Contains separation distances of all particles i to other particles
       -- For example, r_all_i[0] is an array whose elements give the distance
          between the 0th particle and each of the remaining particles in 
          the system
    
    Preconditions
    -------------
        r MUST contain 2-D vectors 
        acc_or_dis MUST either be the string 'acc' or the string 'dis'

    """
    
    # Set empty lists to accumulate separation distances and accelerations
    r_all_i = []
    acc_i = []
    
    
    for i in range(len(r)):
        
        # Set an empty list to accumulate separation distances for ith particle
        r_ij = [] 
        
        # Set zero acceleration vector of ith particle due to jth particles
        acc_ij = np.zeros(2) 
        
        for j in range(len(r)):
            if i != j:
                vec_dist = r[j] - r[i]
                mag_dist = np.sqrt(vec_dist[0]**2 + vec_dist[1]**2)
                r_ij.append(mag_dist) # Accumulate separation distances
                acc_ij += acc(r[i], r[j]) # Update acceleration vector
        r_all_i.append(r_ij) # Accumulate sep. dis. for ith particle
        acc_i.append(acc_ij) # Accumulate acceleration of ith particle
    
    # Convert results to numpy arrays 
    r_all_i = np.array(r_all_i)
    acc_i = np.array(acc_i)
    
    # Return appropriate values, as specified by the user
    if acc_or_dis == 'acc':
        return acc_i
    elif acc_or_dis == 'dis':
        return r_all_i


# Set empty lists to accumulate positions, velocities, and separation distances
# for ALL particles at ALL time points
x_points = []
y_points = []
v_corresponding = []
separation_distances = []


# Initialize vector variables for Verlet method
r_copy = r_initial 
v = v_initial + (dt/2)*system_info(r_initial, 'acc') # Verlet velocity vectors
v_c = v_initial # Velocity vectors


# Perform Verlet Method
for i in range(t_points.size):
    # Record position, velocity, and separation distance for EVERY particle
    x_points.append(r_copy[:, 0])
    y_points.append(r_copy[:, 1])
    v_corresponding.append(v_c)
    separation_distances.append(system_info(r_copy, 'dis'))
    
    # Update position, velocity, and Verlet velocity for EVERY particle
    r_copy = r_copy + dt*v
    k = dt*system_info(r_copy, 'acc')   
    v_c = v + 0.5*k
    v = v + k


# UNCOMMENT TO GENERATE TRAJECTORY PLOT
# plt.title('Trajectories of 16 Particles in a 2-D Molecular System due \nto a Varying Total Interaction Potential', \
#           fontsize=13)
# plt.plot(np.array(x_points), np.array(y_points), '.', markersize=1, c='gray')
# plt.plot(np.array(x_points)[-1][-1], np.array(y_points)[-1][-1], '.',\
#           c='gray', label='Particle Trajectories')
# plt.plot(x_initial, y_initial, '.', markersize=10, c='red', \
#           label='Initial positions of particles')
# plt.xlabel(r'$x$ (arbitrary distance units)', fontsize=13)
# plt.ylabel(r'$y$ (arbitrary distance units)', fontsize=13)
# plt.grid(linestyle='dotted')
# plt.legend()



"""

Question 2(b) PSEUDOCODE

CODE THAT CALCULATES THE TOTAL ENERGY OF THE ABOVE 2-D 16-PARTICLE SYSTEM AT 
EACH TIME POINT

Authors: Idil Yaktubay

"""


# Calculate TOTAL kinetic energy and TOTAL potential energy at all time points
	# Set an empty list to accumulate total kinetic energy at each time point
	# Iterate over x and y velocities of ALL particles to calculate 
	# total kinetic energy at each time point
	# Record each iteration in the empty list
	
	# REPEAT above steps for potential energy using V(r)/2 


# Calculate TOTAL energy of system by adding total kinetic and potential
# energies at all time points


"""

Question 2(b) REAL PYTHON CODE

CODE THAT CALCULATES THE TOTAL ENERGY OF THE ABOVE 2-D 16-PARTICLE SYSTEM AT 
EACH TIME POINT

Authors: Idil Yaktubay and Souren Salehi

"""

# Calculate total kinetic energy at ALL time points
E_k = []
for i in range(len(v_corresponding)):
    E_k_x = np.sum(0.5*m*v_corresponding[i][:, 0]**2)
    E_k_y = np.sum(0.5*m*v_corresponding[i][:, 1]**2)
    total_kenergy = E_k_x + E_k_y
    E_k.append(total_kenergy)

# Calculate total potential energy at ALL time points
E_potential = []
for i in range(len(separation_distances)):
    E_potential_i = 0
    for j in range(len(separation_distances[i])):
        jth_potential = np.sum(0.5*V(separation_distances[i][j]))
        E_potential_i += jth_potential
    E_potential.append(E_potential_i)


# Calculate total energy of system at ALL time points
total_E = np.array(E_k) + np.array(E_potential)
deviation_from_initial = abs((max(total_E) - min(total_E))/total_E[0])
print('The maximum deviation of total energy from the initial total energy', \
      'is', deviation_from_initial*100, '%.')


# UNCOMMENT TO GENERATE ENERGY PLOT
# plt.title('The Total Energy of a 2-D 16-Particle Molecular System \nWith Respect to Time')
# plt.plot(t_points, total_E, '.')
# plt.xlabel('Time (arbitrary time units)')
# plt.ylabel('Total Energy (arbitrary energy units)')
# plt.grid(linestyle='dotted')
