"""
PHY407F 
LAB #8
QUESTION #2

Authors: Idil Yaktubay and Souren Salehi
November 2022
"""


"""
Question 2(b) PSEUDOCODE

CODE THAT IMPLEMENTS A 1D SHALLOW-WATER SYSTEM WITH THE FTCS SCHEME TO GENERATE
PLOTS OF THE FREE WATER SURFACE WITH RESPECT TO POSITION ALONG X FOR TIMES 
t=0s, t=1s, AND t=4s

Author: Idil Yaktubay
"""


# Import numpy
# Import matplotlib.pyplot


# Set all relevant constants
	# Set system length L in m
	# Set equilibrium position H in m
	# Set flat bottom topography eta_b in m
	# Set gravitational acceleration g in ms^-2
	# Set initial function constant A in m
	# Set initial function constant mu in m
	# Set initial function constant sigma in m


# Set x domain of length L
	# Set spatial step size
	# Create spatial array up to L with appropriate step size


# Set time-related constants
	# Set t=0s 
	# Set t=1s
	# Set t=4s
	# Set time step size
	# Set a small number epsilon to check time of iteration
	# Create time array up to 4s with appropriate step size


# Define a function that returns the first component of flux-conservative 
# vector F given fluid velocity along x and free water altitude
# Inside the function body:
	# Calculate the first component
	# Return the first component


# Define a function that returns the second component of flux-conservative 
# vector F given fluid velocity along x and free water altitude
# Inside the function body:
	# Calculate the second component
	# Return the second component


# Perform FTCS scheme:
# Iterate over time points
	# For each time point, iterate over all positions along x
		# Set value of water altitude for x = 0
		# Set values of water altitude and water velocity for non-boundary x 
		# Set value of water altitude for x = L
	# Update the new values of water altitude and water velocity
	# Check if iteration corresponds to...
		# Time 0s; if yes, generate plot
		# Time 1s; if yes, generate plot
		# Time 4s; if yes, generate plot
		# Save plot
       
        
"""
Question 2(b) REAL PYTHON CODE

CODE THAT IMPLEMENTS A 1D SHALLOW-WATER SYSTEM WITH THE FTCS SCHEME TO GENERATE
PLOTS OF THE FREE WATER SURFACE WITH RESPECT TO POSITION ALONG X FOR TIMES 
t=0s, t=1s, AND t=4s

Authors: Idil Yaktubay and Souren Salehi
"""


# Import necessary modules
import numpy as np
import matplotlib.pyplot as plt


# Set all relevant constants
L = 1.0 # System length in m
H = 0.01 # Equilibrium position in m
eta_b = 0. # Flat bottom topography in m
g = 9.81 # Gravitational acceleration in ms^-2
A = 0.002 # in m
mu = 0.5 # in m
sigma = 0.05 # in m


# Set domain along x
dx = 0.02 # Spatial step size
x_domain = np.arange(0, L+dx, dx) # Spatial domain


# Set time-related constants and time domain
t_1 = 0.0
t_2 = 1.0
t_3 = 4.0
dt = 0.01 # Time step size
epsilon = dt/1000 # Small number to compare float times
t_domain = np.arange(0.0, t_3+dt, dt) # Time domain


# Set initial conditions
u = np.zeros(x_domain.size) # Fluid velocity
avg = np.average(A*np.e**(-(x_domain - mu)**2/sigma**2)) 
eta = H + A*np.e**(-(x_domain - mu)**2/sigma**2) - avg # Free water altitude


# Set zero arrays to contain future updated values for u and eta
u_new = np.zeros(x_domain.size)
eta_new = np.zeros(x_domain.size)


# Define functions that find the components of the flux-conservative vector 
# function F
def F_1(u, eta):
    """
    Return the first component F1 of the flux-conservative vector function F
    given fluid velocity u and free water altitude eta
    
    Parameters
    ----------
    u : float
        Fluid velocity in the x direction
    eta : float
        Altitude of the free water surface
        
    Returns
    -------
    F1 : float
        First component of F
    """
    F1 = 0.5*u**2 + g*eta
    return F1


def F_2(u, eta):
    """
    Return the second component F2 of the flux-conservative vector function F
    given fluid velocity u and free water altitude eta
    
    Parameters
    ----------
    u : float
        Fluid velocity in the x direction
    eta : float
        Altitude of the free water surface
        
    Returns
    -------
    F2 : float
        Second component of F
    """
    F2 = (eta - eta_b)*u
    return F2


# Perform FTCS scheme
for t in t_domain: # Iterate over time

    for i in range(x_domain.size): # Iterate over position
    
        if i == 0:
            eta_new[i] = eta[i] - (dt/dx)*(F_2(u[i+1], eta[i+1]) - \
                        F_2(u[i], eta[i]))
        elif 0 < i < x_domain.size-1:
            u_new[i] = u[i] - (dt/(2*dx))*(F_1(u[i+1], eta[i+1]) - \
                      F_1(u[i-1], eta[i-1]))
            eta_new[i] = eta[i] - (dt/(2*dx))*(F_2(u[i+1], eta[i+1]) - \
                        F_2(u[i-1], eta[i-1]))
        else:
            eta_new[i] = eta[i] - (dt/dx)*(F_2(u[i], eta[i]) - \
                        F_2(u[i-1], eta[i-1]))
                
    # Update velocity and altitude
    eta = np.copy(eta_new)
    u = np.copy(u_new)
    
    # Check times and generate plots
    if abs(t-t_1) < epsilon: # t=0s
        plt.plot(x_domain, eta, label='t = 0s')
        plt.title('Free Water Surface Altitude of a 1D Shallow-Water System as a Function of Position x',\
                  size=14)
        plt.xlabel('Position x (m)', size=14)
        plt.ylabel(r'Water Altitude $\eta$ (m)', size=14)
        plt.grid(linestyle='dotted')
    elif abs(t-t_2) < epsilon: # t=1s
        plt.plot(x_domain, eta, label='t = 1s')
    elif abs(t-t_3) < epsilon: # t=4s
        plt.plot(x_domain, eta, label='t = 4s')
        plt.legend(bbox_to_anchor=(1,0.5) , \
                   loc='center left', prop={'size': 14})
        plt.savefig('Q2_B.png', bbox_inches='tight')
