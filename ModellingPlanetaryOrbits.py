'''
PHY407F 
LAB #1
QUESTION #1

Authors: Idil Yaktubay and Souren Salehi
September 2022


'''

'''
Question 1(b) PSEUDOCODE

CODE THAT CALCULATES AMD PLOTS THE POSITION AND VELOCITY AS A FUNCTION OF TIME 
OF A PLANET UNDER A GRAVITY FORCE USING EULER-CROMER METHOD FOR GIVEN INITIAL
CONDITIONS

Author: Idil Yaktubay

'''

# Import matplotlib.pyplot
# Import numpy
# Import the time function from the time library


# Using the time function, define the time at the start of program execution


# Define the gravitational constant G times the mass of the sun M_s


# Set the initial velocities and positions for 2D planetary motion 


# Set the length of the time steps/increments
# Set the total length of time for the motion
# Using the previously set time step length, create a time array that plots
# up to the previously set total length of time
# Set the number of time steps including time zero


# Define a function that finds planetary acceleration as given by the equations
# from the lab file


# Create two 1-D velocity arrays the same length as the time array and full of
# zeroes, one for velocity along x and one for velocity along y


# Set the first element of each velocity array to the corresponding inital
# velocities


# Create two 1-D position arrays the same length as the time array and full of
# zeroes, one for the position along x and one for position along y

# Set the first element of each position array to the corresponding initial 
# positions


# Using the previously defined function for acceleration and the Euler-Cromer
# method, find the remaining elements of the velocity and position arrays by
# <length of time array> iterations.


# Using matplotlib, plot the components of velocity as a function of time and
# the 2-D orbit


# Using the time function, define the time at the end of the program execution


'''
Question 1(c) and 1(d) REAL PYTHON CODE

CODE THAT CALCULATES AMD PLOTS THE POSITION AND VELOCITY AS A FUNCTION OF TIME 
OF A PLANET UNDER THE NEWTONIAN AND RELATIVISTIC GRAVITY FORCES USING
EULER-CROMER METHOD FOR GIVEN INITIAL CONDITIONS

Author: Idil Yaktubay and Souren Salehi

'''

# Import matplotlib.pyplot
import matplotlib.pyplot as plt 
# Import numpy
import numpy as np 
# Import the time function from the time library
from time import time


# Define the time at the start of program execution
start = time()


# Define the gravitational constant G times the mass of the sun M_s 
C = 39.5 # in AU^3/yr^2

# Define the relativistic constant
a = 0.01 # in AU^2



# Set the initial velocity and position for 2D planetary motion
x_i = 0.47 # in AU
y_i = 0.0 # in AU
v_xi = 0.0 # in AU/yr
v_yi = 8.17 # in AU/yr


# Set the length of the time steps/increments 
dt = 0.0001 # in yr


# Set the total length of time 
total_t = 1 #in yr


# Using the previously set time step length, create a time array that plots
# up to the previously set total length of time
t = np.arange(0, total_t, dt)


# Set the number of time steps including time zero
n = len(t)


# Define functions that find gravitational acceleration due to Newtonian and 
# relativistic gravity forces, as given by the lab file

def find_aNewton(x, y):
    """Return the gravitational acceleration due to a Newtonian gravity force
    INPUT: 
        x [float] is distance from the source along x
        y [float] is distance from the source along y
    OUTPUT: 
        [float] is the acceleration
    """
    return -C*x/((x**2 + y**2)**(3/2))

def find_aRelat(x, y):
    """Return the gravitational acceleration due to a relativistic gravity 
    force
    INPUT: 
        x [float] is distance from the source along x
        y [float] is distance from the source along y
    OUTPUT: 
        [float] is the acceleration
    """
    return (-C*x/((x**2 + y**2)**(3/2)))*(1 + a/(x**2 + y**2))


# Create two 1-D velocity arrays the same length as the time array and full of
# zeroes, one for velocity along x and one for velocity along y

# Newtonian
v_x = np.zeros(n)
v_y = np.zeros(n)

# Relativistic
v_rx = np.zeros(n)
v_ry = np.zeros(n)

# Set the first element of each velocity array to the corresponding inital
# velocities

# Newtonian
v_x[0] = v_xi
v_y[0] = v_yi

# Relativistic
v_rx[0]= v_xi
v_ry[0]= v_yi

  

# Create two 1-D position arrays the same length as the time array and full of
# zeroes, one for the position along x and one for position along y

# Newtonian
x = np.zeros(n)
y = np.zeros(n)

# Realitivistic
x_r = np.zeros(n)
y_r = np.zeros(n)

# Set the first element of each position array to the corresponding initial 
# positions

# Newtonian
x[0] = x_i
y[0] = y_i

# Relativistic
x_r[0] = x_i
y_r[0] = y_i

# Create an array that gives angular momentum as a function of time

# Newtonian
L = np.zeros(n)

# Relativistic
L_r = np.zeros(n)


# Using the previously defined function for acceleration and the Euler-Cromer
# method, find the remaining elements of the velocity, position, and angular 
# momentum arrays

for i in range(0, n-1):
    
    # Newtonian
    v_x[i+1] = v_x[i] + find_aNewton(x[i], y[i])*dt
    v_y[i+1] = v_y[i] + find_aNewton(y[i], x[i])*dt
    
    x[i+1] = x[i] + v_x[i+1]*dt
    y[i+1] = y[i] + v_y[i+1]*dt
    
    L[i] = np.sqrt(x[i]**2 + y[i]**2)*np.sqrt(v_x[i]**2 + v_y[i]**2)
    
    # Relativistic
    v_rx[i+1] = v_rx[i] + find_aRelat(x_r[i], y_r[i])*dt
    v_ry[i+1] = v_ry[i] + find_aRelat(y_r[i], x_r[i])*dt
    
    x_r[i+1] = x_r[i] + v_rx[i+1]*dt
    y_r[i+1] = y_r[i] + v_ry[i+1]*dt
    
    L_r[i] = np.sqrt(x_r[i]**2 + y_r[i]**2)*np.sqrt(v_rx[i]**2 + v_ry[i]**2)
    
    i+=1

        
# Using matplotlib, plot the components of velocity as a function of time and
# the 2-D orbit

# Newtonian

# plt.plot(t, v_x, label='Velocity Along x')
# plt.plot(t, v_y, label='Velocity Along y')
# plt.title( \
#     "The Velocity of Mercury As a Function of Time Under Newtonian Gravity")
# plt.xlabel("Time (yr)")
# plt.ylabel("Velocity (AU/yr)")
# plt.grid(linestyle='dotted')

# plt.plot(x, y)
# plt.title("The Orbit of Mercury in Space Under Newtonian Gravity")
# plt.xlabel("Position Along x (AU)")
# plt.ylabel("Position Along y (AU)")
# plt.grid(linestyle='dotted')

# Relativistic

# plt.plot(t, v_rx, label='Velocity Along x')
# plt.plot(t, v_ry, label='Velocity Along y')
# plt.title( \
#     "The Velocity of Mercury As a Function of Time Under Relativistic Gravity")
# plt.xlabel("Time (yr)")
# plt.ylabel("Velocity (AU/yr)")
# plt.grid(linestyle='dotted')

# plt.plot(x_r, y_r)
# plt.title("The Orbit of Mercury in Space Under Relativistic Gravity")
# plt.xlabel("Position Along x (AU)")
# plt.ylabel("Position Along y (AU)")
# plt.grid(linestyle='dotted')


# plt.legend()
# plt.savefig('relativistic_velocity.png', bbox_inches='tight')

print("The Angular Momentum divided by Mercury's Mass for Newtonian Gravity with respect to time is...", \
      L)
print("The Angular Momentum divided by Mercury's Mass for Relativistic Gravity with respect to time is...", \
      L_r)

# Using the time function, define the time at the end of the program execution
end = time()
