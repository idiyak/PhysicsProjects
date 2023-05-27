'''

PHY407F 
LAB #1
QUESTION #3

Author: Souren Salehi
September 2022


'''
# Import matplotlib.pyplot
import matplotlib.pyplot as plt
# Import numpy
import numpy as np
# Import the time function from the time library
from time import time

# Create an array in intervals of 1 to use as the size of the matrix
Q = np.arange(2,200, 1)

# Create empty arrays to be used for the time of execution
diff = np.zeros(len(Q))
diff1 = np.zeros(len(Q))


# Run the fragment code to mulitply two square matrices and iterate over the
# Q array to find time over different matrix sizes
for r in range(len(Q)):
    N = Q[r]
    A = np.ones([N, N], float)*5
    B  = np.ones([N, N], float)*2
    C = np.zeros ([N,N], float)
    
    # Record time for each runtime of different matrix sizes
    start = time()
    for i in range(N):
        for j in range(N):
            for k in range(N):
                C[i,j] += A[i, k]*B[k,j]
    end = time()
    diff[r] = end-start
    r=r+1

print("times for explicit calculations are",diff)
# Plot the time taken for square matrices of different sizes to be multiplied
plt.title('Time for execution of explicit matrix multiplication')
plt.xlabel('Number of matrix entries')
plt.ylabel("Time[s]")
plt.scatter(Q,diff)
plt.grid(linestyle='dotted')
plt.show()

# Plot the time taken for the matrices to be multiplied in a cubic scale
plt.title('Time for execution of explicit matrix multiplication (cubic scale)')
plt.xlabel('Number of matrix entries cubed')
plt.ylabel("Time[s]")
plt.scatter(Q**3,diff)
plt.grid(linestyle='dotted')
plt.show()


# Iterate over the multiplication of matrices of different sizes using 
# the numpy dot function
for r in range(len(Q)):
    N=Q[r]
    start = time()
    A = np.ones([N, N], float)*5
    B  = np.ones([N, N], float)*2
    C = np.dot(A,B)
    end= time()
    diff1[r] = end-start
    r=r+1

print("times for numpy.dot are", diff1)
# Plot the time taken for square matrices of different sizes to be multiplied
# plt.title('Time for execution of numpy matrix multiplication')
# plt.xlabel('Number of matrix entries')
# plt.ylabel("Time[s]")
# plt.scatter(Q,diff1)
# plt.grid(linestyle='dotted')
# plt.show()

# Plot the time taken for the matrices to be multiplied in a cubic scale
# plt.title('Time for execution of numpy matrix multiplication (cubic scale)')
# plt.xlabel('Number of matrix entries qubed')
# plt.ylabel("Time[s]")
# plt.scatter(Q**3,diff1)
# plt.grid(linestyle='dotted')
# plt.show()
