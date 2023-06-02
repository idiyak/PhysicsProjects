"""

PHY407F 
LAB #5
QUESTION #1

Authors: Souren Salehi and Idil Yaktubay
October 2022

"""

"""

Question 1(a) PSEUDOCODE

CODE THAT USES THE EULER-CROMER METHOD TO SIMULATE A RELATIVISTIC SPRING SYSTEM
OVER AT LEAST 10 OSCILLATIONS IN THE CASE OF ZERO INITIAL VELOCITY AND THREE
DIFFERENT INITIAL POSITIONS

Author: Idil Yaktubay


"""

# Import numpy
# Import constants from scipy to use for c
# Import matplotlib.pyplot
# Import numpy.fft

# Set the relevant constants
	# Set k
	# Set m
	# Set c using scipy.constants

# Create time arrays with different lengths to capture at least 10 oscillations
	# Set time step
	# Set first time array
	# Set the second, longer time array

# Define a function that returns acceleration based on position and velocity
# Inside the function body:
	# Set acceleration with appropriate formula
	# Return acceleration

# For EACH of the three initial positions of 1, xc, and 10xc:
	# Create empty arrays for the oscillation with only initial values present
		# Set empty array for velocity v
		# Set empty array for position x
		# Set initial x inside position array
		# Set initial v inside velocity array
	# Iterate using Euler-Cromer to find remaining positions and velocities

# For EACH of the three initial positions, plot oscillation wrt time


"""

Question 1(a) REAL PYTHON CODE 

CODE THAT USES THE EULER-CROMER METHOD TO SIMULATE A RELATIVISTIC SPRING SYSTEM
OVER AT LEAST 10 OSCILLATIONS IN THE CASE OF ZERO INITIAL VELOCITY AND THREE
DIFFERENT INITIAL POSITIONS

Author: Souren Salehi

"""


# Import the necessary modules
import numpy as np
from scipy import constants as cs
import matplotlib.pyplot as plt
import numpy.fft as ft

# Set relevant constants
k = 12
m = 1
c = cs.c

# Create time arrays with different lengths to capture at least 10 oscillations
dt = 0.0001
t = np.arange(0, 100, dt)
tc1 = np.arange(0, 150, dt)

def a(x,v):
    """
    Return acceleration a based on position x and velocity v

    Parameters
    ----------
    x : FLOAT
        Position x
    v : FLOAT
        Velocity v

    Returns
    -------
    a : FLOAT
       Acceleration a 

    """
    a = -(k/m)*x*(1-(v/c)**2)**(3/2)
    return a


# CALCULATIONS FOR x0=1 START
# Create empty arrays for x0=1 oscillation with only first elements present
v = np.zeros(len(t))
x = np.zeros(len(t))
x[0]= 1
v[0]=0
    
# Iterate over time using Euler-Cromer to find position and velocity
for n in range(len(t)-1):
    v[n+1] = v[n] + dt*a(x[n],v[n])
    x[n+1]= x[n] + v[n+1]*dt
    n+=1


# CALCULATIONS FOR x0=xc START
# Create empty arrays for x0=xc oscillation with only first elements present
vc = np.zeros(len(t))
xc = np.zeros(len(t))
x0 = np.sqrt(m/k)*c
xc[0]= x0
vc[0]=0

# Iterate over time using Euler-Cromer to find position and velocity
for n in range(len(t)-1):
    vc[n+1] = vc[n] + dt*a(xc[n],vc[n])
    xc[n+1]= xc[n] + vc[n+1]*dt
    n+=1


# CALCULATIONS FOR x0=10xc START
# Create empty arrays for x0=10xc oscillation with only first elements present
vc1 = np.zeros(len(tc1))
xc1 = np.zeros(len(tc1))
xc1[0]= 10*x0
vc1[0]=0

# Iterate over time using Euler-Cromer to find position and velocity
for n in range(len(tc1)-1):
    vc1[n+1] = vc1[n] + dt*a(xc1[n],vc1[n])
    xc1[n+1]= xc1[n] + vc1[n+1]*dt
    n+=1


# Produce plots for the three oscillations with different initial displacements

# PLOTS FOR INITIAL POSITION x0=1 - UNCOMMENT TO GENERATE PLOT
# plt.style.use('seaborn')
# plt.suptitle("Oscillation with initial displacement x0 = 1m", fontsize=20)
# plt.xlabel('time (s)', fontsize=20)
# plt.ylabel('position (m)', fontsize=20)
# plt.plot(t, x)
# plt.xlim(0, 30)
# plt.show()


# # PLOTS FOR INITIAL POSITION x0=xc - UNCOMMENT TO GENERATE PLOT
# plt.suptitle("Oscillation with initial displacement x0 = xc", fontsize=20)
# plt.xlabel('time (s)', fontsize=20)
# plt.ylabel('position (m)', fontsize=20)
# plt.plot(t, xc)
# plt.xlim(0, 30)
# plt.show()


# PLOTS FOR INITIAL POSITION x0=10xc - UNCOMMENT TO GENERATE PLOT
# plt.suptitle("Oscillation with initial displacement x0 = 10xc", fontsize=20)
# plt.xlabel('time (s)', fontsize=20)
# plt.ylabel('position (m)', fontsize=20)
# plt.plot(tc1, xc1)
# plt.show()


"""

Question 1(b) PSEUDOCODE

CODE THAT PLOTS THE FOURIER TRANSFORM OF THE PREVIOUS THREE POSITION 
FUNCTIONS WITH RESPECT TO THE FREQUENCY DOMAIN ON THE SAME GRAPH BY RESCALING  

Author: Idil Yaktubay

"""

# Calculate and set the frequency domains for the time segment for 
# x0 = 1, xc and the time segment for x0=10xc
	# Time segment for x0 = 1, xc
	# Time segment for x0 = 10 xc

# Perform Fast Fourier Transform on the three position functions to find
# Fourier coefficients 
	# For x0 = 1
	# For x0 = xc
	# For x0 = 10xc

# On the same figure, plot the relative Fourier coefficients with respect 
# to the frequencies by rescaling the coefficients as |x(omega)|/|x(omega)|_max 
	# Plot for x0 = 1 
	# Plot for x0 = xc
	# Plot for x0 = 10xc
   

"""

Question 1(b) REAL PYTHON CODE 

CODE THAT PLOTS THE FOURIER TRANSFORM OF THE PREVIOUS THREE POSITION 
FUNCTIONS ON THE SAME GRAPH BY RESCALING AND THUS FINDS ESTIMATES OF 
THE THREE PERIODS OF OSCILLATION

Author: Souren Salehi

"""


# Set the frequency domains for the different time segments
xf = ft.fftfreq(len(t), dt)
xfc = ft.fftfreq(len(tc1), dt)

# Perform fast fourier transform to find the coefficients for the three cases 
c = ft.fft(x)
cc = ft.fft(xc)
cc1 = ft.fft(xc1)


# Plot of the relative fourier coefficients with respect to frequency domain
# UNCOMMENT TO GENERATE PLOT
# plt.plot(xf, np.abs(c)/max(np.abs(c)), label=r'$x_0=1$')
# plt.plot(xf, np.abs(cc)/max(np.abs(cc)), label=r'$x_0=x_c$')
# plt.plot(xfc, np.abs(cc1)/max(np.abs(cc1)), label=r'$x_0=10x_c$')
# plt.legend(prop={'size': 16})
# plt.xlim(-2,2)
# plt.xlabel('Frequency (Hz)', fontsize=17)
# plt.ylabel('Relative Amplitude of Coefficients (m)', fontsize=17)
# plt.suptitle(\
#     'Relative Amplitudes of Fourier Coefficients for the 3 Oscillations', \
#     fontsize=20)
# plt.show()


"""

Question 1(c) PSEUDOCODE

CODE THAT COMPARES THE THREE SIMULATED FREQUENCIES TO THE FREQUENCIES ESTIMATED
BY GAUSSIAN QUADRATURE INTEGRALS FROM LAB3 QUESTION 2 BY PRODUCING OVERLAID
PLOTS

Author: Souren Salehi 


"""

# Set the periods found in lab3 question 2 with Gaussian quadratures
    # Set period for x0=1m
    # Set period for x0=xc
    # Set period for x0=10xc

# Set Gaussian quadrature frequencies by inverting the previously set periods
    # Set frequency for x0=1m
    # Set frequency for x0=xc
    # Set frequency for x0=10xc 
   
# For each of the three systems, plot Fourier coefficients versus frequency 
# with Gaussian quadrature frequency laid on top of each plot


"""

Question 1(c) REAL PYTHON CODE

CODE THAT COMPARES THE THREE SIMULATED FREQUENCIES TO THE FREQUENCIES ESTIMATED
BY GAUSSIAN QUADRATURE INTEGRALS FROM LAB3 QUESTION 2 BY PRODUCING OVERLAID
PLOTS

Author: Souren Salehi 

"""


# Set the periods found in lab3 question 2 
T = 1.7707
Tc = 2.08755
Tc1 = 11.628

# Set the corresponding frequencies
f = 1/T
fc = 1/Tc
fc1 = 1/Tc1

# Plot the Fourier coefficients for x0=1 overlaid with integral frequency
# UNCOMMENT TO GENERATE PLOT
# plt.xlabel('Frequency (Hz)', fontsize=17)
# plt.ylabel('Relative Amplitude of Coefficients (m)', fontsize=17)
# plt.suptitle('Relative Amplitude of Fourier Coefficients for x0=1', \
#              fontsize=20)
# plt.plot(xf, np.abs(c)/max(np.abs(c)), label='Fourier Coefficients')
# plt.xlim(-2,2)
# plt.plot([f,f],[0,1], label = 'Value from Gaussian\nQuadrature')
# plt.legend(loc='upper left', prop={'size': 12})
# plt.show()

# Plot the Fourier coefficients for x0=xc overlaid with integral frequency
# UNCOMMENT TO GENERATE PLOT
# plt.xlabel('Frequency (Hz)', fontsize=17)
# plt.ylabel('Relative Amplitude of Coefficients (m)', fontsize=17)
# plt.suptitle('Relative Amplitude of Fourier Coefficients for x0=xc', \
#              fontsize=20)
# plt.plot(xf, np.abs(cc)/max(np.abs(cc)), label='Fourier coefficients')
# plt.plot([fc,fc],[0,1], label = 'Value from Gaussian\nQuadrature')
# plt.xlim(-2,2)
# plt.legend(loc='upper left', prop={'size': 13})
# plt.show()

# Plot the Fourier coefficients for x0=10xc overlaid with integral frequency 
# UNCOMMENT TO GENERATE PLOT
# plt.xlabel('Frequency (Hz)', fontsize=17)
# plt.ylabel('Relative Amplitude of Coefficients (m)', fontsize=17)
# plt.suptitle('Relative Amplitude of Fourier Coefficients for x0=10xc', \
#              fontsize=20)
# plt.plot(xfc, np.abs(cc1)/max(np.abs(cc1)), label='Fourier Coefficients')
# plt.plot([fc1,fc1],[0,1], label = 'Values from Gaussian\nQuadratures')
# plt.xlim(-2,2)
# plt.legend(loc='upper left', prop={'size': 13})
# plt.show()







