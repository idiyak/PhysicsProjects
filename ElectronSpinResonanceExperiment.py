# Idil Yaktubay and Maheen Mufti - PHY224 Electron Spin Resonance Experiment
"""
In this experiment, the phenomenon of electron spin resonance
was studied in order to determine the Lande g factor of unpaired 
electrons in a diphenylpicryl hydrazyl (DPPH) sample. By mounting 
this sample in three copper coils of different sizes, generating a 
radio-frequency field inside each coil, placing each coil in 
an external magnetic field produced by 2 Helmholtz coils, 
and lastly taking measurements of resonance frequencies and resonance 
currents through the Helmholtz coils, the Lande g factors were 
found to be 2.9 +- 0.3, 2.0 +- 0.3, and 2.0 +- 0.3
for the small, medium, and large coils, respectively. 
"""

import numpy as np 
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#==============================ASSIGNING CONSTANTS============================


mu_o = (4*np.pi)*(10**(-7))
R = 7.5/100   # Radius of Helmholtz coils (m)
R_uncertainty = 0.5/100
n = 320 # Number of turns of each Helmholtz coil 
m = 9.109e-31 # Mass of an electron (kg)
e = 1.602e-19 # Charge of an electron (C)
ideal_gamma = e/(2*m)


#============================LOADING THE DATA=================================


# SMALL COIL

# Loading resonance frequencies (Hz)
freq_small = np.loadtxt('small_coil.txt', usecols=1)*1000000
# Loading resonance currents (A)
currents_small = np.loadtxt('small_coil.txt', usecols=0)/2
# Converting currents to magnetic field strengths (T)
B_small = ((4/5)**(3/2))*((mu_o*n*currents_small)/R)
# Uncertainty in each current measurement
currents_small_uc = currents_small*0.02
# Uncertainty in each magnetic field value
B_small_uc = B_small*(np.sqrt((currents_small_uc/currents_small)**2 \
            + (R_uncertainty/R)**2))



# MEDIUM COIL 

# Loading resonance frequencies (Hz)
freq_med = np.loadtxt('medium_coil.txt', usecols=1)*1000000
# Loading resonance currents (A)
currents_med = np.loadtxt('medium_coil.txt', usecols=0)/2
# Converting currents to magnetic field strengths (T)
B_med = ((4/5)**(3/2))*((mu_o*n*currents_med)/R)
# Uncertainty in each current measurement
currents_med_uc = currents_med*0.02
# Uncertainty in each magnetic field value
B_med_uc = B_med*(np.sqrt((currents_med_uc/currents_med)**2 \
            + (R_uncertainty/R)**2))


# LARGE COIL 

# Loading the resonance frequencies (Hz)
freq_large = np.loadtxt('large_coil.txt', usecols=1)*1000000
# Loading the resonance currents (A)
currents_large = np.loadtxt('large_coil.txt', usecols=0)/2
# Converting currents to magnetic field strengths (T)
B_large = ((4/5)**(3/2))*(mu_o*n*currents_large)/R
# Uncertainty in each current measurement
currents_large_uc = currents_large*0.02
# Uncertainty in each magnetic field value
B_large_uc = B_large*(np.sqrt((currents_large_uc/currents_large)**2 \
            + (R_uncertainty/R)**2))
    
    
#================================CURVE FITTING=================================

# CREATING THE MODEL FUNCTION TO USE WHEN CURVE FITTING
def model_function(x, a, b):
    return a*x + b

# SMALL COIL

guesses_small = (B_small[1] - B_small[0])/(freq_small[1] - freq_small[0])
p_opt_small, p_cov_small = curve_fit(model_function, freq_small, \
            B_small, p0=[guesses_small, 0], sigma=B_small_uc, \
                absolute_sigma=True)
p_std_small = np.sqrt(np.diag(p_cov_small))
gamma_small = (2*np.pi)/p_opt_small[0]
gamma_small_uc = (p_std_small[0]/p_opt_small[0])*gamma_small
g_small = gamma_small/ideal_gamma
g_small_uc = (gamma_small_uc/gamma_small)*g_small
fitted_small = model_function(freq_small, p_opt_small[0], p_opt_small[1])

print('SMALL COPPER COIL \n')
print('Slope of field vs frequency graph optimized by curve fitting is:',\
      p_opt_small[0], 'sT +-', p_std_small[0], 'sT. \n')
print('y-intercept of the same graph optimized by curve fitting is:', \
      p_opt_small[1], 'T +-', p_std_small[1], 'T. \n')
print('The gyromagnetic ratio is:', gamma_small, 's^-1 T^-2 +-',\
      gamma_small_uc, 's^-1 T^-1. \n')
print('Therefore, the Lande g factor is:', round(g_small, 3), \
      '+-', round(g_small_uc, 3), '. \n')


    
# MEDIUM COIL
guesses_med = (B_med[1] - B_med[0])/(freq_med[1] - freq_med[0])
p_opt_med, p_cov_med = curve_fit(model_function, freq_med, B_med, \
                                  p0=[guesses_med, 0], sigma=B_med_uc, \
                                      absolute_sigma=True)
p_std_med = np.sqrt(np.diag(p_cov_med))
gamma_med = (2*np.pi)/p_opt_med[0]
gamma_med_uc = (p_std_med[0]/p_opt_med[0])*gamma_med
g_med = gamma_med/ideal_gamma
g_med_uc = (gamma_med_uc/gamma_med)*g_med
fitted_med = model_function(freq_med, p_opt_med[0], p_opt_med[1])
print('MEDIUM COPPER COIL \n')
print('Slope of field vs frequency graph optimized by curve fitting is:',\
      p_opt_med[0], 'sT +-', p_std_med[0], 'sT. \n')
print('y-intercept of the same graph optimized by curve fitting is:', \
      p_opt_med[1], 'T +-', p_std_med[1], 'T. \n')
print('The gyromagnetic ratio is:', gamma_med, 's^-1 T^-2 +-',\
      gamma_med_uc, 's^-1 T^-1. \n')
print('Therefore, the Lande g factor is:', round(g_med, 3), \
      '+-', round(g_med_uc, 3), '. \n')


# LARGE COIL
guesses_large = (B_large[1] - B_large[0])/(freq_large[1] - freq_large[0])
p_opt_large, p_cov_large = curve_fit(model_function, freq_large, B_large, \
            p0=[guesses_large, 0], sigma=B_large_uc, absolute_sigma=True)
p_std_large = np.sqrt(np.diag(p_cov_large))
gamma_large = (2*np.pi)/p_opt_large[0]
gamma_large_uc = (p_std_large[0]/p_opt_large[0])*gamma_large
g_large = gamma_large/ideal_gamma
g_large_uc = (gamma_large_uc/gamma_large)*g_large
fitted_large = model_function(freq_large, p_opt_large[0], p_opt_large[1])
print('LARGE COPPER COIL \n')
print('Slope of field vs frequency graph optimized by curve fitting is:',\
      p_opt_large[0], 'sT +-', p_std_large[0], 'sT. \n')
print('y-intercept of the same graph optimized by curve fitting is:', \
      p_opt_large[1], 'T +-', p_std_large[1], 'T. \n')
print('The gyromagnetic ratio is:', gamma_large, 's^-1 T^-2 +-',\
      gamma_large_uc, 's^-1 T^-1. \n')
print('Therefore, the Lande g factor is:', round(g_large, 3), \
      '+-', round(g_large_uc, 3), '. \n')

#======================FINDING CHISQUARED FOR THE FITS=========================
n_parameters = 2
N = 15
def red_chisquared(data, fit, sigma):
    chisquared = (1/(N-n_parameters))*np.sum(((data-fit)/sigma)**2)
    return chisquared
chisquared_small = red_chisquared(B_small, fitted_small, B_small_uc)
chi_squared_med = red_chisquared(B_med, fitted_med, B_med_uc)
chi_squared_large = red_chisquared(B_large, fitted_large, B_large_uc)

print('REDUCED CHI-SQUARED VALUES \n')
print('Small copper coil:', chisquared_small)
print('Medium copper coil:', chi_squared_med)
print('Large copper coil:', chi_squared_large, '\n')

#==============================PLOTTING THE DATA===============================

# SMALL COIL
plt.scatter(freq_small, B_small, label='Experimental Data')
plt.errorbar(freq_small, B_small, yerr=B_small_uc, ecolor='red', fmt='none', \
              capsize=3, label='Error in Experimental Data')
plt.plot(freq_small, fitted_small, color='grey', label='Fitted Curve')
plt.grid(which='major', color='#DDDDDD', linewidth=0.8)
plt.grid(which='minor', color='grey', linestyle=':', linewidth=0.5)
plt.minorticks_on()
plt.title(r'Resonant Magnetic field $B_z$ vs Resonant Frequency $\nu$ for Small Copper Coil')
plt.xlabel(r'Frequency $\nu$ (Hz)')
plt.ylabel(r'Field $B_z$ (T)')
plt.legend()
    
# MEDIUM COIL 
plt.scatter(freq_med, B_med, label='Experimental Data')
plt.errorbar(freq_med, B_med, yerr=B_med_uc, ecolor='red', fmt='none', \
              capsize=3, label='Error in Experimental Data')
plt.plot(freq_med, fitted_med, color='grey', label= 'Fitted Curve')
plt.grid(which='major', color='#DDDDDD', linewidth=0.8)
plt.grid(which='minor', color='grey', linestyle=':', linewidth=0.5)
plt.minorticks_on()
plt.title(r'Resonant Magnetic field $B_z$ vs Resonant Frequency $\nu$ for Medium Copper Coil')
plt.xlabel(r'Frequency $\nu$ (Hz)')
plt.ylabel(r'Field $B_z$ (T)')
plt.legend()
    
# LARGE COIL 
# plt.scatter(freq_large, B_large, label='Experimental Data')
# plt.errorbar(freq_large, B_large, yerr=B_large_uc, ecolor='red', fmt='none', \
#               capsize=3, label='Error in Experimental Data')
# plt.plot(freq_large, fitted_large, color='grey', label= 'Fitted Curve')
# plt.grid(which='major', color='#DDDDDD', linewidth=0.8)
# plt.grid(which='minor', color='grey', linestyle=':', linewidth=0.5)
# plt.minorticks_on()
# plt.title(r'Resonant Magnetic field $B_z$ vs Resonant Frequency $\nu$ for Large Copper Coil')
# plt.xlabel(r'Frequency $\nu$ (Hz)')
# plt.ylabel(r'Field $B_z$ (T)')
# plt.legend()

    



