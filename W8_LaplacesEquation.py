"""
PHY407F 
LAB #8
QUESTION #1

Authors: Souren Salehi and Idil Yaktubay
November 2022
"""

"""
Question 1(a) PSEUDOCODE

CODE THAT CALCULATES THE ELECTROSTATIC POTENTIAL AT EACH GRIDD POINT OF A 
SIMPLE 2D ELECTRONIC CAPACITOR USING THE GAUSS-SIEDEL METHOD WITHOUT 
OVER-RELAXATION 

Author: Idil Yaktubay
"""


# Import numpy
# Import matplotlib
# Import imshow, gray, show for contour plot
# Import time function to compare execution times


# Set grid dimension 
# Set absolute value of voltage at the plates
# Set target accuracy


# Create a 2D meshgrid with appropriate dimensions
    # Set grid points along x
    # Set grid points along y
    # Create meshgrid


# Create an array of zero voltage with meshgrid dimensions


# Set voltage to +1V and -1V at the plates
    # Iterate over points to assign plate voltages
		# Potential at plate 1
		# Potential at plate 2

# Set a value to be updated and compared to target accuracy


# Set start time
# Using Gauss-Siedel w/o over-relaxation, update values of V at grid points:
    # Iterate over each grid point until target accuracy is reached:
            # Update voltage for points along plates
            # Update voltage for points along boundaries
            # Update voltage for points interior to boundaries
                # Update accuracy comparison value
# Set end time


# Print relevant results (execution time and voltage values)


# Make a contour plot of the final result


"""
Question 1(a) REAL PYTHON CODE

CODE THAT CALCULATES THE ELECTROSTATIC POTENTIAL AT EACH GRIDD POINT OF A 
SIMPLE 2D ELECTRONIC CAPACITOR USING THE GAUSS-SIEDEL METHOD WITHOUT 
OVER-RELAXATION 

Author: Souren Salehi
"""


# Import necessary modules and functions
import numpy as np
import matplotlib.pyplot as plt
from pylab import gray
from time import time


M= 100 # Grid dimension
V0 = 1 # Absolute potential at plates
target = 1e-6 # Target accuracy


# Create appropriate meshgrid
x = np.linspace(0, 10, 101) 
y = np.linspace(0, 10, 101)
X, Y = np.meshgrid(x, y)


# Create an array of zero voltage with meshgrid dimensions
V = np.zeros([M+1, M+1], float)


# Set the voltages at the plates
for j in range(20,80):
    V[:,20][j]=+V0 # First plate
    V[:,80][j]=-V0 # Second plate


delta = 1.0 # Target accuracy

# Perform Gauss-Siedel without over-relaxation
start1 = time() # Start time of execution
while delta > target:
    delta = 0
    for i in range(M):
          for j in range(M):
              if j == 20 and 20 < i < 80: # Points along plate one
                  V[i,j] = V[i,j]
              elif j == 80 and 20 < i < 80: # Points along plate two
                  V[i,j] = V[i,j]
              elif i == 0 or j == 0 or i == M or j == M: # Boundary points
                  V[i,j] = V[i,j]
              else: # Interior points
                  dV = (1/4)*(V[i+1,j] + V[i-1,j] + V[i,j+1] + V[i,j-1]) - \
                      V[i,j]
                  V[i,j] += dV
                  if dV > delta:     
                      delta = dV
end1 = time() # End time of execution


# Print relevant results
print('Execution time for Gauss-Siedel w/o over-relation is: ', end1-start1, \
      's.')
print('The Voltage values for Gauss-Siedel w/o over-relaxation are: \n', \
      V)
    
    
# Generate plots
plt.figure()
plt.contourf(X,Y,V)
plt.colorbar(label='Potential (V)')
plt.title('The Electrostatic Potential due to an Electronic Capacitor in a Metal Box Calculated\nWith Gauss-Siedel Method Without Over-Relaxation', \
          size=13, y=1.08)
plt.xlabel('Position Along x (cm)', size=13)
plt.ylabel('Position Along y (cm)', size=13)
gray()


"""
Question 1(b) PSEUDOCODE

CODE THAT CALCULATES THE ELECTROSTATIC POTENTIAL AT EACH GRIDD POINT OF A 
SIMPLE 2D ELECTRONIC CAPACITOR USING THE GAUSS-SIEDEL METHOD WITH 
OVER-RELAXATION FOR omega=0.1 and omega=0.5

Author: Idil Yaktubay
"""

# Create an array of zero voltage with meshgrid dimensions for the w=0.1 case


# Set voltage to +1V and -1V at the plates for the w=0.1 case
	# Iterate over points to assign plate voltages
		# Potential at plate 1
		# Potential at plate 2
	

# Copy the iterated voltage as a separate array for w=0.5 case


# Set a value to be updated and compared to target accuracy


# Set over-relaxation constants
	# Set 0.1
	# Set 0.5


# Set start time for the w=0.1 case
# Use Gauss-Siedel w/ w=0.1 to update potential values at grid points:
      # Set a value to be updated and compared to target accuracy
      # Iterate over each grid point until target accuracy is reached:
            # Update voltage for points along plates
            # Update voltage for points along boundaries
            # Update voltage for points interior to boundaries
                # Update accuracy comparison value
# Set end time for the w=0.1 case


# Print relevant results for w=0.1 (execution time and voltage values)


# Make a contour plot of the final result for w=0.1


# Set start time for the w=0.5 case
# Use Gauss-Siedel w/ w=0.5 to update potential values at grid points:
      # Set a value to be updated and compared to target accuracy
      # Iterate over each grid point until target accuracy is reached:
            # Update voltage for points along plates
            # Update voltage for points along boundaries
            # Update voltage for points interior to boundaries
                # Update accuracy comparison value
# Set end time for the w=0.5 case


# Print relevant results for w=0.5 (execution time and voltage values)


# Make a contour plot of the final result for w=0.5



"""
Question 1(a) REAL PYTHON CODE

CODE THAT CALCULATES THE ELECTROSTATIC POTENTIAL AT EACH GRIDD POINT OF A 
SIMPLE 2D ELECTRONIC CAPACITOR USING THE GAUSS-SIEDEL METHOD WITH 
OVER-RELAXATION FOR omega=0.1 and omega=0.5

Author: Souren Salehi and Idil Yaktubay
"""


# Create an array of zero voltage with meshgrid dimensions for w=0.1
V1 = np.zeros([M+1, M+1], float)


# Set the plate potentials for w=0.1
for j in range(20,80):
    V1[:,20][j]=+V0 # First plate
    V1[:,80][j]=-V0 # Second plate


# Copy the same voltage array for w=0.5
V2 = np.copy(V1)


# Set over-relaxation constants
w1=0.1
w2=0.5

# Perform Gauss-Siedel with an over-relaxation constant w=0.1
delta1 = 1.0 # Target accuracy
start2 = time() # Start time of execution
while delta1 > target:
    delta1 = 0
    for i in range(M):
          for j in range(M):
              if j == 20 and 20 < i < 80: # Points along plate one
                  V1[i,j] = V1[i,j]
              elif j == 80 and 20 < i < 80: # Points along plate two
                  V1[i,j] = V1[i,j]
              elif i == 0 or j == 0 or i == M or j == M: # Boundary points
                  V1[i,j] = V1[i,j]
              else: # Interior points
                  dV = ((1+w1)/4)*(V1[i+1,j] + V1[i-1,j] + V1[i,j+1] + \
                                   V1[i,j-1]) - w1*V1[i,j] - V1[i,j]
                  V1[i,j] += dV
                  if dV > delta1:     
                      delta1 = dV 
end2 = time() # End time of execution


# Print relevant results corresponding to w=0.1
print('Execution time for Gauss-Siedel w/ over-relation (w=0.1) is: ',\
      end2-start2, 's.')
print('The Voltage values for Gauss-Siedel w/ over-relaxation (w=0.1) are: \n', \
      V1)


# Generate plots for w=0.1
plt.figure()
plt.contourf(X,Y,V1)
plt.colorbar(label='Potential (V)')
plt.title('The Electrostatic Potential due to an Electronic Capacitor in a Metal Box Calculated\nWith Gauss-Siedel Method With Over-Relaxation ($\omega$=0.1)', \
          size=13, y=1.08)
plt.xlabel('Position Along x (cm)', size=13)
plt.ylabel('Position Along y (cm)', size=13)
gray()    


# Perform Gauss-Siedel with an over-relaxation constant w=0.5
delta2 = 1.0 # Target accuracy
start3 = time() # Start time of execution
while delta2 > target:
    delta2 = 0
    for i in range(M):
          for j in range(M):
              if j == 20 and 20 < i < 80: # Points along plate one
                  V2[i,j] = V2[i,j]
              elif j == 80 and 20 < i < 80: # Points along plate two
                  V2[i,j] = V2[i,j]
              elif i == 0 or j == 0 or i == M or j == M: # Boundary points
                  V2[i,j] = V2[i,j]
              else: # Interior points
                  dV = ((1+w2)/4)*(V2[i+1,j] + V2[i-1,j] + V2[i,j+1] + 
                                   V2[i,j-1]) - w2*V2[i,j] - V2[i,j]
                  V2[i,j] += dV
                  if dV > delta2:     
                      delta2 = dV 
end3 = time() # End time of execution


# Print relevant results corresponding to w=0.5
print('Execution time for Gauss-Siedel w/ over-relation (w=0.5) is: ',\
      end3-start3, 's.')
print('The Voltage values for Gauss-Siedel w/ over-relaxation (w=0.5) are: \n', \
      V2)


# Generate plots for w=0.5
plt.figure()
plt.contourf(X,Y,V2)
plt.colorbar(label='Potential (V)')
plt.title('The Electrostatic Potential due to an Electronic Capacitor in a Metal Box Calculated\nWith Gauss-Siedel Method With Over-Relaxation ($\omega$=0.5)', \
          size=13, y=1.08)
plt.xlabel('Position Along x (cm)', size=13)
plt.ylabel('Position Along y (cm)', size=13)
gray()
