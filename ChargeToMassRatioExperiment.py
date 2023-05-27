# Idil Yaktubay and Maheen Mufti - Charge to Mass Ratio Experiment
"""
In this experiment, the behavior of a beam of electrons in uniform magnetic
fields perpendicular to the electron velocity was studied in order to 
determine the charge to mass ratio of an electron. By accelerating a 
beam of electrons through a constant potential difference, placing the 
beam in various uniform magnetic fields induced by different currents
through two Helmholtz Coils, and measuring the diameters of electron
orbits as well as the corresponding currents through Helmholtz coils, 
the charge to mass ratio of an electron was found to be (1.9 +- 0.3) x 10^11 C/kg.
This value was found to agree to within 6% of the theoretical value of 
1.8 x 10^11 C/kg. 
"""

import numpy as np 
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#===================SETTING THE VALUES OF RELEVANT PARAMETERS=================


mu_o = 4*np.pi*(10**(-7)) # Magnetic constant
R = 16.3/100 # Radius of the Helmholtz coils
R_uncertainty = 0.2/100 
fixed_voltage = 275 
fixed_voltage_err = 1
n = 130 # Number of turns of the Helmholtz coils 
k = (1/np.sqrt(2))*((4/5)**(3/2))*((mu_o*n)/R)
k_uncertainty = (R_uncertainty/R)*k
e = 1.602e-19 # Charge of an electron
m = 9.109e-31 # Mass of an electron

a = ((4/5)**(3/2))*mu_o*n


#============================LOADING THE DATA==================================


# Measurements with V fixed at 275 V

currents = np.loadtxt('275_volts.txt', usecols=0)
currents_uncertainty = np.loadtxt('275_volts.txt', usecols=2)

diameters = np.loadtxt('275_volts.txt', usecols=1)/100
diameters_err = np.zeros(15) + 0.5/100 

radii = diameters/2 # Measured radii
radii_err = (diameters_err/diameters)*radii

curvatures = 1/radii
curv_err = (radii_err/radii)*curvatures


#========================FINDING THE EXTERNAL FIELD============================


# Values of Helmholtz magnetic fields will be accumulated here
uncorrected_B_c = ((4/5)**(3/2))*(mu_o*n)*(currents/R)
B_c = []

for i in range(radii.size):
    if radii[i] < 0.2*R:
        B_c.append(uncorrected_B_c[i])
    else:
        b_c = (1 - ((radii[i]**4)/((R**4)*(0.6583 + \
                            0.29*(radii[i]**2/R**2))**2)))*uncorrected_B_c[i]
        B_c.append(b_c)
B_c = np.array(B_c)

# Partial derivatives to be used in error propagation
partial_I = (a/R) - ((a*radii**4)/((R**5)*(0.6583 + 0.29*(radii**2/R**2))**2))
partial_r = -(90800000000*currents*R*a*radii**3)/ \
    (841*(100*radii**2 + 227*R**2)**3)
partial_R = ((-currents*a)/R**2) + \
    (5*a*currents*radii**4)/((R**6)*(0.29*(radii**2/R**2) + 0.6583)**2) - \
    (29*currents*a*radii**6)/(25*((0.29*(radii**2/R**2) + 0.6583)**3)*R**8)

# Error propagation    
B_c_uncertainty = np.sqrt((partial_I*currents_uncertainty)**2 + \
                    (partial_R*R_uncertainty)**2 + (partial_r*radii_err)**2)   

# Model function to find B_e
def model_function(x, a, b):
    y = a*x + b
    return y

# Finding B_e
p_opt, p_cov = curve_fit(model_function, curvatures, B_c, \
                          sigma=B_c_uncertainty, absolute_sigma=True)
p_std = np.sqrt(np.diag(p_cov))
B_e = -p_opt[1]
B_c_fitted = model_function(curvatures, p_opt[0], p_opt[1])
print('FINDING THE EXTERNAL FIELD')
print('The optimized parameter b is', p_opt[1], '+-', p_std[1])
print('Therefore, the calculated value of the external field B_e is:', \
      B_e*1e9, 'nT +-', \
      p_std[1]*1e9, 'nT. \n')
    
    
#=======================FINDING CHARGE TO MASS RATIO===========================


def model_function_1(x, a, b):
    y = a*(x + b)
    return y

guesses = [k*np.sqrt(e/(m*fixed_voltage)), -B_e/k]
p_opt_1, p_cov_1 = curve_fit(model_function_1, currents, curvatures, \
                             p0=guesses, sigma=curv_err, absolute_sigma=True)
p_std_1 = np.sqrt(np.diag(p_cov_1))
ratio = (fixed_voltage*(p_opt_1[0]**2))/(k**2)
ratio_uncertainty = \
    np.sqrt((((2*p_opt_1[0]*fixed_voltage)/k**2)*(p_std_1[0]))**2 + \
            ((p_opt_1[0]**2/k**2)*(fixed_voltage_err))**2 + \
                (((-2*p_opt_1[0]**2*fixed_voltage)/k**3)*k_uncertainty)**2)
fitted_curvatures = model_function_1(currents, p_opt_1[0], p_opt_1[1])

print('FINDING THE CHARGE TO MASS RATIO')
print('The optimized value of parameter a is', p_opt_1[0], '+-', \
      p_std_1[0])
print('The charge to mass ratio of an electron is:', ratio, 'C/kg +-', \
      ratio_uncertainty, 'C/kg. \n')

    
#=======================CALCULATING CHI SQUARED VALUES=========================


n_parameters = 2
N = 15
def red_chisquared(data, fit, sigma):
    chisquared = (1/(N-n_parameters))*np.sum(((data-fit)/sigma)**2)
    return chisquared

# External field fit
chi_field = round(red_chisquared(B_c, B_c_fitted, B_c_uncertainty), 1)

# Charge to mass ratio fit
chi_ratio = round(red_chisquared(curvatures, fitted_curvatures, curv_err), 1)

print('Recuced chi squared for the fit used to calculate B_e is', chi_field)
print('Reduced chi squared for the fit used to calculate e/m is', chi_ratio)


#================================PLOTTING THE DATA=============================


# Plots related to the external field B_e
# plt.title(r'Graph of Helmholtz Magnetic Field $B_c$ versus Orbital Curvature $\frac{1}{r}$ at 275 Volts')
# plt.plot(curvatures, B_c_fitted, label='Model Curve')
# plt.scatter(curvatures, B_c, label='Experimental Data', color='green')
# plt.errorbar(curvatures, B_c, yerr=B_c_uncertainty, fmt='none', ecolor='red',\
#               capsize=3, label='Error in Experimental Data')
# plt.xlabel(r'Curvature of Orbit $\frac{1}{r} \, (m^{-1})$')
# plt.ylabel(r'Helmholtz Magnetic Field $B_c$ (T)')
# plt.grid(which='major', color='#DDDDDD', linewidth=0.8)
# plt.grid(which='minor', color='grey', linestyle=':', linewidth=0.5)
# plt.minorticks_on()
# plt.legend()

# Plots related to the charge to mass ratio 
# plt.title('Orbital Curvature of Electrons versus Current Through Helmholtz Coils at 275 V')
# plt.plot(currents, fitted_curvatures, label='Model Curve')
# plt.scatter(currents, curvatures, color='green', label='Experimental Data')
# plt.errorbar(currents, curvatures, yerr=curv_err, fmt='none', \
#               ecolor='red', capsize=3, label='Error in Experimental Data')
# plt.xlabel('Helmholtz Current I (A)')
# plt.ylabel(r'Orbital Curvature $\frac{1}{r} \, (m^{-1})$ ')
# plt.grid(which='major', color='#DDDDDD', linewidth=0.8)
# plt.grid(which='minor', color='grey', linestyle=':', linewidth=0.5)
# plt.minorticks_on()
# plt.legend()





















