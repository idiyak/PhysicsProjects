"""
PHY407H F 
LAB #10
QUESTION #1

Authors: Idil Yaktubay and Nico Grisouard (See Grisouard's Brownian-start.py)
November 2022
"""


"""
Question 1(a) PSEUDOCODE

CODE THAT PLOTS THE 5000-STEP TRAJECTORY OF A PARTICLE UNDER BROWNIAN
MOTION INSIDE A 101X101 LATTICE (ASSUMING PARTICLE DOES NOT STICK TO WALLS)

Author: Idil Yaktubay 

"""


# Import numpy
# Import matplotlib.pyplot
# Import necessary functions from random


# Set the size of domain 
# Set the number of time steps


# Define a function that "randomly" chooses the particle's next move
# INSIDE FUNCTION BODY, if the particle is on...
	# 1) Left edge of lattice (excluding corners):
		# Move up OR
		# Move down OR
		# Move right
	# 2) Bottom edge of lattice (excluding corners):
		# Move up OR
		# Move right OR
		# Move left
	# 3) Right edge of the lattice (excluding corners):
		# Move up OR
		# Move down OR
		# Move left
	# 4) Top edge of the lattice (excluding corners):
		# Move left OR
		# Move right OR
		# Move down
	# 5) Top left corner of lattice:
		# Move down OR
		# Move right
	# 6) Bottom left corner of lattice:	
		# Move up OR
		# Move right
	# 7) Bottom right corner of lattice:
		# Move up OR
		# Move left 
	# 8) Top right corner of lattice:
		# Move down OR
		# Move left 
	# 9) The limits of lattice:
		# Move down OR
		# Move up OR
		# Move right OR
		# MOve left 
	# Return new particle coordinates


# Set the starting center point
# Assign this starting point to x and y coordinates (updated later)
# Copy starting x and y points to plot later (won't be updated)


# Set two lists (one for x, one for y) that only contain starting 
# points to accumulate remaining coordinates of the particle


# Iterate over the number of steps and with each iteration:
	# Update x and y coordinates using the defined function
	# Add new x coordinate to x list
	# Add new y coordinate to y list


# Set a time array for 5000 steps (5001 points) for future plots


# GENERATE PLOTS
	# x versus t plot
	# y versus t plot
	# y versus x plot 
	# y versus x plot zoomed in


"""
Question 1(a) REAL PYTHON CODE

CODE THAT PLOTS THE 5000-STEP TRAJECTORY OF A PARTICLE UNDER BROWNIAN
MOTION INSIDE A 101X101 LATTICE (ASSUMING PARTICLE DOES NOT STICK TO WALLS)

Authors: Idil Yaktubay and Nico Grisouard (See Grisouard's Brownian-start.py)

"""

# Import necessary modules
import numpy as np
import matplotlib.pyplot as plt
from random import randrange

Lp = 101  # Size of domain
Nt = 5000  # Number of time steps

def nextmove(x, y):
    """
    Return a new set of two-dimensional coordinates x and y when given a 
    current set of two dimensional coordinate x and y
    
    Parameters
    ----------
    x : INT
        x coordinate 
    y : INT
        y coordinate

    Returns
    -------
    x : INT
        new x coordinate
    y : TYPE
        new y coordinate

    """
    if x == 0 and y != 0 and y != Lp-1: # Left edge of lattice
        direction = randrange(3)
        if direction == 0: # Move up
            y += 1
        elif direction == 1: # Move down
            y -= 1
        elif direction == 2: # Move right
            x += 1
    elif y == 0 and x != 0 and x != Lp-1: # Bottom edge of lattice
        direction = randrange(3)
        if direction == 0: # Move up
            y += 1
        elif direction == 1: # Move right
            x += 1
        elif direction == 2: # Move left
            x -= 1
    elif x == Lp-1 and y != 0 and y != Lp-1: # Right edge of lattice
        direction = randrange(3)
        if direction == 0: # Move up
            y += 1
        elif direction == 1: # Move down
            y -= 1
        elif direction == 2:  # Move left
            x -= 1
    elif y == Lp-1 and x != 0 and x != Lp-1: # Top edge of lattice
        direction = randrange(3)  
        if direction == 0:  # Move left
            x -= 1
        elif direction == 1:  # Move right
            x += 1
        elif direction == 2:  # Move down
            y -= 1
    elif x == 0 and y == Lp-1: # Top left corner of lattice
        direction = randrange(2)
        if direction == 0: # Move right
            x += 1
        elif direction == 1: # Move down
            y -= 1
    elif x == 0 and y == 0: # Bottom left corner of lattice 
        direction = randrange(2)
        if direction == 0: # Move up
            y += 1
        elif direction == 1: # Move right
            x +=1
    elif x == Lp-1 and y == 0: # Bottom right corner of lattice
        direction = randrange(2)
        if direction == 0: # Move up
            y += 1
        elif direction == 1: # Move left
            x -= 1
    elif x == Lp-1 and y == Lp-1: # Top right corner of lattice
        direction = randrange(2)
        if direction == 0: # Move down
            y -= 1
        elif direction == 1: # Move left
            x -= 1 
    else: # Inside the limits of lattice
        direction = randrange(4) 
         
        if direction == 0:  # Move up
            y += 1
        elif direction == 1:  # Move down
            y -= 1
        elif direction == 2:  # Move right
            x += 1
        elif direction == 3:  # Move left
            x -= 1
        else:
            print("error: direction isn't 0-3")
    return x, y


centre_point = (Lp-1)//2  # Starting centre point of lattice
xp, yp = centre_point, centre_point # Starting x and y coordinates (to update)
xp_save, yp_save = xp, yp # Immutable starting x and y coordinates


x, y = [xp], [yp] # Accumulation lists for xy coordinates


for i in range(Nt): 
    xp, yp = nextmove(xp, yp) # Update xy coordinates
    x.append(xp) # Add to x list
    y.append(yp) # Add to y list


t = np.arange(0, len(x)) # Time array for future plot


# X VERSUS T PLOT
plt.figure()
plt.plot(t, np.array(x), 'o', markersize=0.8)
plt.title('Position Along x With Respect to Time for\na Two-Dimensional Brownian Particle',
          size=14)
plt.xlabel('Time (s)', size=14)
plt.ylabel('x (Distance Units)', size=14)
plt.grid(linestyle='dotted')


# Y VERSUS T PLOT
plt.figure()
plt.plot(t, np.array(y), 'o', markersize=0.8)
plt.title('Position Along y With Respect to Time for\na Two-Dimensional Brownian Particle',
          size=14)
plt.xlabel('Time (s)', size=14)
plt.ylabel('y (Distance Units)', size=14)
plt.grid(linestyle='dotted')


# Y VERSUS X PLOT 
plt.figure()
plt.plot(x, y, label='Trajectory')
plt.plot(xp_save, yp_save, 'o', color='red', markersize=5, \
          label='Initial Position')
plt.plot(x[-1], y[-1], 'o', color='magenta', markersize=5, \
         label='Final Position')
plt.title('The Brownian Trajectory of a Particle Confined to\na Two-Dimensional Box',
          size=14)
plt.xlim([0, Lp-1])
plt.ylim([0, Lp-1])
plt.xlabel('x (Distance Units)', size=14)
plt.ylabel('y (Distance Units)', size=14)
plt.grid(linestyle='dotted')
plt.legend(bbox_to_anchor=(0.5, -0.16) , \
            loc='upper center', prop={'size': 15})

    
# Y VERSUS T PLOT ZOOMED IN
plt.figure()
plt.plot(x, y, label='Trajectory')
plt.plot(xp_save, yp_save, 'o', color='red', markersize=5, 
          label='Initial Position')
plt.plot(x[-1], y[-1], 'o', color='magenta', markersize=5, \
         label='Final Position')
plt.title('The Brownian Trajectory of a Particle Confined to\na Two-Dimensional Box (Enlarged)',
          size=15)
plt.xlabel('x (Distance Units)', size=14)
plt.ylabel('y (Distance Units)', size=14)
plt.grid(linestyle='dotted')
plt.legend(bbox_to_anchor=(0.5, -0.16) , \
            loc='upper center', prop={'size': 15})
    

"""
Question 1(b) PSEUDOCODE

CODE THAT FINDS THE FINAL POSITIONS OF PARTICLES IN A TWO-DIMENSIONAL 
DIFFUSION-LIMITED AGGREGATION MODEL UNTIL THE CENTRE OF THE MODEL 
BECOMES OCCUPIED AND PLOTS THE DLA PATTERN 

Author: Idil Yaktubay

"""


# Set an empty list to accumulate final x,y positions of particles


# Iterate until the lattice center is filled...
	# 1) Start a particle at the lattice center
	# 2) Check if the neighboring positions are occupied by 
	#    anchored particles:
	#    If true, add the coordinate to the list
	# 3) Check if the current position is a lattice edge:
	#    If true, add the coordinate to the list
	# Check if the added coordinate is a center coordinate:
	# 	If true: stop the iteration
	#     If false: continue the iteration


# Generate plot of final particle positions

# Print info about the generated model


"""
Question 1(b) REAL PYTHON CODE

CODE THAT FINDS THE FINAL POSITIONS OF PARTICLES IN A TWO-DIMENSIONAL 
DIFFUSION-LIMITED AGGREGATION MODEL UNTIL THE CENTRE OF THE MODEL 
BECOMES OCCUPIED AND PLOTS THE DLA PATTERN 

Author: Idil Yaktubay

"""


stuck_particles = [] # Final position accumulation list


centre_full = False
while not centre_full:
    xp, yp = centre_point, centre_point # Set initial position
    stuck = False
    
    while not stuck:
        xp, yp = nextmove(xp, yp) # Update particle position
        
        # Check for neighboring anchored particles
        if [xp+1, yp] in stuck_particles or [xp, yp+1] in stuck_particles or \
            [xp-1, yp] in stuck_particles or [xp, yp-1] in stuck_particles: 
            stuck = True
            stuck_particles.append([xp, yp]) # Add anchored position to list
        
        # Check if current position is an edge
        elif xp == 0 or xp == Lp-1 or yp == 0 or yp == Lp-1:
            stuck = True
            stuck_particles.append([xp, yp]) # Add anchor position to list
    
    # Check if centre is occupied
    if [centre_point, centre_point] in stuck_particles:
        centre_full = True


# Generate DLA pattern
plt.figure()
plt.plot(np.array(stuck_particles)[:, 0], np.array(stuck_particles)[:, 1], \
         'o', markersize=2)
plt.title('Final Positions of Brownian Particles in a Two-Dimensional\nDiffusion-Limited Aggregation Model')
plt.xlabel('x (Position Units)')
plt.ylabel('y (Position Units)')
plt.grid(linestyle='dotted')

# Print particle information
print('It took ', len(stuck_particles), \
      ' particles to complete the DLA model!')
