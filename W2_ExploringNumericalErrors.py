'''

PHY407F 
LAB #2
QUESTION #1

Author: Idil Yaktubay
September 2022


'''


'''

Question 1(a) PSEUDOCODE

CODE THAT FINDS THE RELATIVE ERROR FOR AN ESTIMATION OF THE STANDARD DEVIATION
OF A GIVEN DATASET

Author: Idil Yaktubay

'''

# Import numpy

# Read data from a textfile 

# Using numpy, set the "correct" standard deviation
# Print the "correct" standard deviation for reference

# For equation (1), define a function that estimates the standard deviation
    # Pass through the data to calculate the mean
    # Pass through the data to calculate the standard deviation
    # Return the standard deviation


# For equation (2), define a function that estimates the standard deviation
    # Pass through the data once to calculate the sum xi^2 and the nx^2 terms
    # Print a warning for when the sum xi^2 - nx^2 is negative
    # Set a 'fake' standard deviation for the function to return for when 
    # the sum xi^2 - nx^2 is negative
    # Return the standard deviation otherwise

# Print the estimated standard deviations for reference

# Calculate and print the relative errors for both equations
# For equation (1)
# For equation (2)


'''

Question 1(b) REAL PYTHON CODE

CODE THAT FINDS THE RELATIVE ERROR FOR AN ESTIMATION OF THE STANDARD DEVIATION
OF A GIVEN DATASET

Author: Idil Yaktubay

'''

# Import numpy
import numpy as np

# Read data from a textfile 
c_data = np.loadtxt('cdata.txt')

# Using numpy, set the "correct" standard deviation
correct_std = np.std(c_data, ddof=1)
# Print the "correct" standard deviation for reference
print("QUESTION 1(B)\n")
print("The 'correct' standard deviation of Michelsen's data is: ", \
      correct_std, "x 10^3 km/s.")
    

# For equation (1), define a function that calculates sigma
def std_1(data, n):
    """Return an estimation of the standard deviation of data
    INPUT:
        data [numpy.darray] is a data set
        n [int] is the number of samples
    OUTPUT:
        std [float] is the estimated standard deviation
    """
    # Pass through the data to calculate the mean
    sum_xi = 0 # Sum of data elements xi
    for i in data:
        sum_xi += i
    mean = sum_xi/n
    
    # Pass through the data to calculate standard deviation
    sum_diff_sqr = 0 # Sum of (xi - x_mean)^2
    for i in data:
        sum_diff_sqr += (i - mean)**2
    std = np.sqrt((1/(n-1))*sum_diff_sqr)
    
    # Return the standard deviation
    return std
    
    
        
# For equation (2), define a function that estimates the standard deviation
def std_2(data, n):
    """Return an estimation of the standard deviation of data
    INPUT:
        data [numpy.darray] is a data set
        n [int] is the number of samples
    OUTPUT:
        std [float] is the estimated standard deviation
    """
    # Pass through the data once to calculate the sum xi^2 and the nx^2 terms
    sum_xi = 0 # Sum of the elements in data 
    sum_xi_2 = 0 # Sum of the square of the elements in data
    for i in data:
        sum_xi += i
        sum_xi_2+= i**2
    mean = sum_xi/n
    
    diff = sum_xi_2 - n*mean**2 
    std = 0.0 # Define "fake" std value for when sum xi^2 - nx^2 is negative
    
    # Print a warning for when the sum xi^2 - nx^2 is negative
    if diff < 0:
        print( \
        "WARNING: STD CALCULATION REQUIRES SQUARE ROOT OF A NEGATIVE NUMBER")
    # Return std value if sum xi^2 - nx^2 is positive
    else: 
        std = np.sqrt((1/(n-1))*(sum_xi_2 - n*mean**2))
        return std 
    
    # Return 'fake' std value for when sum xi^2 - nx^2 is negative
    return std

# Print the estimated standard deviations for reference
estimation_1 = std_1(c_data, c_data.size)
estimation_2 = std_2(c_data, c_data.size)
print("Standard deviation using equation 1 is: ", estimation_1, " x 10^3 km/s")
print("Standard deviation using equation 2 is: ", estimation_2, " x 10^3 km/s")


# Calculate and print the relative errors for both equations

# For equation (1)
rel_err_1 = (abs(estimation_1 - correct_std))/correct_std
print("The relative error for equation (1) is: ", rel_err_1*100, " %.")

# For equation (2)
rel_err_2 = (abs(estimation_2 - correct_std))/correct_std
print("The relative error for equation (2) is: ", rel_err_2*100, " %.\n")
