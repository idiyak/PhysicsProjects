''

PHY407F 
LAB #1
QUESTION #2

Authors: Souren Salehi and Idil Yaktubay
September 2022


'''

'''
Question 2(a) PSEUDOCODE

CODE THAT CALCULATES AND PLOTS THE ORBITS OF THE EARTH AND JUPITER
IN A THREE-BODY PROBLEM

Author: Idil Yaktubay

'''

# Import matplotlib.pyplot
# Import numpy
# Import time

# Using the time function, define the time at the start of program execution

# Set the length of the time steps/increments
# Set the total length of time
# Using the previously set time step length, create a time array that plots
# up to the previously set total length of time
# Set the number of time steps including time zero

# Define the gravitational constant G times the mass of the sun M_s 

# Create 1-D arrays each with the same length as the time array and
# full of zeroes for Jupiter
# Jupiter's position along x
# Jupiter's position along y
# Jupiter's velocity along x
# Jupiter's velocity along y

# Set the initial conditions for Jupiter

# Using the Euler-Cromer method, find the remaining elements of the velocity
# and position arrays for Jupiter by iteration

# Create 1D arrays each with the same length as the time array and full of 
# zeroes for the Earth
# The Earth's distance to the Sun along x
# The Earth's distance to the Sun along y
# The Earth's velocity along x
# The Earth's velocity along y
# The distance between the Earth and Jupiter along x
# The distance between the Earth and Jupiter along y

# Set the initial conditions for the Earth

# Using the Euler-Cromer method, find the remaining elements of position and
# velocity for the Earth by iteration

# Using matplotlib, plot the orbit of the Earth and Jupiter in 2D space

# Save the plot


'''
Question 2(a) REAL PYTHON CODE

CODE THAT CALCULATES AND PLOTS THE ORBITS OF THE EARTH AND JUPITER
IN A THREE-BODY PROBLEM

Author: Souren Salehi

'''

# Import matplotlib.pyplot
import matplotlib.pyplot as plt
# Import numpy
import numpy as np
# Import time
from time import time

# Using the time function, define the time at the start of program execution
start = time()

# Set the length of the time steps/increments
dt = 0.0001 # in yr

# Set the total length of time
total_t = 20

# Using the previously set time step length, create a time array that plots
# up to the previously set total length of time
t = np.arange(0, total_t, dt)

# Set the number of time steps including time zero
n = len(t)

# Define the gravitational constant G times the mass of the sun M_s 
C = 39.5 # in AU^3/yr^2

# Create 1-D arrays each with the same length as the time array and
# full of zeroes for Jupiter
xj=np.zeros(n) # Jupiter's position along x
yj=np.zeros(n) # Jupiter's position along y
vxj=np.zeros(n) # Jupiter's velocity along x
vyj=np.zeros(n) # Jupiter's velocity along y

# Set the initial conditions for Jupiter
xj[0]=5.2
yj[0]=0
vxj[0]=0
vyj[0]=2.63

# Using the Euler-Cromer method, find the remaining elements of the velocity
# and position arrays for Jupiter by iteration
for i in np.arange(0,n-1):
    
    # Jupiter's velocity
    vxj[i+1] = vxj[i] - dt*C*xj[i]/((xj[i]**2+yj[i]**2)**(3/2))
    vyj[i+1] = vyj[i] - dt*C*yj[i]/((xj[i]**2+yj[i]**2)**(3/2))
    
    # Jupiter's position
    xj[i+1] = xj[i] + vxj[i+1]*dt
    yj[i+1] = yj[i] + vyj[i+1]*dt
    
    i=i+1

# Create 1D arrays each with the same length as the time array and full of 
# zeroes for the Earth
xs=np.zeros(n) # The Earth's distance to the Sun along x
ys=np.zeros(n) # The Earth's distance to the Sun along y
vxs=np.zeros(n) # The Earth's velocity along x
vys=np.zeros(n) # The Earth's velocity along y

xr=np.zeros(n) # The distance between the Earth and Jupiter along x
yr=np.zeros(n) # The distance between the Earth and Jupiter along y

# Set the initial conditions for the Earth
xs[0]= 1
ys[0]=0
vxs[0]=0
vys[0]=6.18

# Using the Euler-Cromer method, find the remaining elements of position and
# velocity for the Earth
for i in np.arange(0,n-1):
    xr[i] = xs[i] - xj[i]
    yr[i]=  ys[i] - yj[i]
    
    # The Earth's Velocity
    vxs[i+1] =  vxs[i] - (dt*C*xs[i]/((xs[i]**2+ys[i]**2)**(3/2))) - \
        (dt*C*10**(-3)*xr[i]/((xr[i]**2+yr[i]**2)**(3/2)))
    vys[i+1] =  vys[i] - (dt*C*ys[i]/((xs[i]**2+ys[i]**2)**(3/2))) - \
        (dt*C*10**(-3)*yr[i]/((xr[i]**2+yr[i]**2)**(3/2)))
        
    # The Earth's Position
    xs[i+1] = xs[i] + vxs[i+1]*dt 
    ys[i+1] = ys[i] + vys[i+1]*dt
    i=i+1
    
# Using matplotlib, plot the orbit of the Earth and Jupiter in 2D space
# plt.plot(xj,yj, label= 'The Orbit of Jupiter')
# plt.plot(xs,ys, label='The Orbit of the Earth')
# plt.title("The Orbits of the Earth and Jupiter Around the Sun")
# plt.xlabel("Position Along x (AU)")
# plt.ylabel("Position Along y (AU)")
# plt.grid(linestyle='dotted')
# plt.legend(loc=1)

# Save the plot
# plt.savefig('earth_and_jup_orbit.png', bbox_inches='tight')

