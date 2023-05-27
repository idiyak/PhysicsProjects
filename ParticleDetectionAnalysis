# Author: Idil Yaktubay
# February 7th 2022
# Particle Detection
"""
The goal of this project is to convert the raw data from a particle detection
experiment to an energy spectrum with computational techniques. 
The raw data is collected by a detector that converts the energy of 
incident particles into a signal of voltage with respect to time.
"""
import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import poisson, norm, chisquare, chi2
from scipy.optimize import curve_fit

with open('calibration_p3.pkl','rb') as file:
    data_from_file=pickle.load(file)

xx = np.linspace(0, 4095/1e3, 4096)
# plt.plot(xx, data_from_file['evt_15'], color='k', lw=0.5)
# plt.xlabel('Time (ms)')
# plt.ylabel('Readout Voltage (V)')

with open('noise_p3.pkl', 'rb') as file:
    noise_data=pickle.load(file)

# plt.plot(xx, noise_data['evt_0'], color='k', lw=0.5)
# plt.xlabel('Time (ms)')
# plt.ylabel('Readout Voltage (V)')

with open('signal_p3.pkl', 'rb') as file:
    signal_data=pickle.load(file)

#==========RECONSTRUCTING ENERGY ESTIMATORS FROM CALIBRATION DATA==============

amp_maxmin = [] # Amplitudes calcualted as max-min 
amp_maxbl = [] # Amplitudes calculted as max-baseline 
integ1 = [] # Integral of the pulse 
integ2 = [] # Integral with baseline subtraction
integ3 = [] # Integral with baseline subtraction and limited range

v_signal = [] 

# Finding the bounds of integ3
i = 999 
t1 = xx[i]
t_step_size = (4095./1e3)/4096
t2 = t1
pulse_time = 100*0.001 

while t2 <= t1 + pulse_time:
    t2 += t_step_size
    i += 1


for event in data_from_file.values():
    
    # max-min amplitudes
    amplitude_1 = max(event) - min(event)
    amp_maxmin.append(amplitude_1)
    
    # max-baseline amplitudes
    baseline = np.mean(event[:1000])
    amplitude_2 = max(event) - baseline
    amp_maxbl.append(amplitude_2)
    
    # Integrals
    integ1.append(np.sum(event))
    integ2.append(np.sum(event-baseline))
    integ3.append(np.sum(event[999: i]-baseline))

for signal in signal_data.values():
    baseline = np.mean(signal[:1000])
    v_signal.append(np.sum(signal[999:i] - baseline))
    
    
amp_maxmin = np.array(amp_maxmin) # Energy Estimator #1
amp_maxbl = np.array(amp_maxbl) # Energy Estimator #2
integ1 = np.array(integ1) # Energy estimator #3
integ2 = np.array(integ2) # Energy estimator #4
integ3 = np.array(integ3) # Energy estimator #5



#=======================FITTING AN ESTIMATE OF THE PULSES======================
def model_trace(t, a):
    t_fall = 80*0.001
    t_rise = 20*0.001
    c = ((t_fall/t_rise)**(-t_rise/(t_fall-t_rise)))*((t_rise-t_fall)/t_fall)
    return a*c*(np.exp(-(t-1)/t_rise) - np.exp(-(t-1)/t_fall)) 

std_trace = []
for noise in noise_data.values():
    std_trace.append(np.std(noise, ddof=1))
err_volt = np.zeros(len(xx)) + np.mean(np.array(std_trace))

optimal_amp = []
for event in data_from_file.values():
    popt, pcov = curve_fit(model_trace, xx[999:], event[999:], \
                           sigma=err_volt[999:], absolute_sigma=True)
    amp=max(model_trace(xx[999:], popt[0]))
    optimal_amp.append(amp)
    
# evt0 = data_from_file['evt_199']
# popt, pcov = curve_fit(model_trace, xx[999:], evt0[999:], sigma=err_volt[999:], \
#                         absolute_sigma=True)
# trace = model_trace(xx[999:], popt[0])
# plt.plot(xx[999:], evt0[999:], color='black', lw=0.5, label='Data')
# plt.plot(xx[999:], trace, label='Fit')
# plt.xlabel('Time (ms)')
# plt.ylabel('Amplitude (V)')
# plt.legend()
# plt.savefig('fit_example.png', bbox_inches='tight')

optimal_amp = np.array(optimal_amp)


#=========================CURVE FITTING THE HISTOGRAMS=========================
def model_distribution(x, s, mu, a, b):
    return a*np.exp(-np.power(x-mu, 2.)/(2*np.power(s, 2.))) + b
    

# EE1 
hist_maxmin = np.histogram(amp_maxmin, bins=70, density=False)
centers_maxmin = hist_maxmin[1][:-1] + np.diff(hist_maxmin[1])/2
err_maxmin = np.sqrt(hist_maxmin[0])
guesses_maxmin = [1.4230334463308138e-05, 0.0003047974114353422, \
                  96.25774595810068, 4.236278263554606]

popt_maxmin, pcov_maxmin = curve_fit(model_distribution, centers_maxmin, \
                            hist_maxmin[0], sigma=err_maxmin, \
                            p0=guesses_maxmin, absolute_sigma=True)
model_maxmin = model_distribution(centers_maxmin, popt_maxmin[0], \
                popt_maxmin[1], popt_maxmin[2], popt_maxmin[3])
    

# EE2 
hist_maxbl = np.histogram(amp_maxbl, bins=53, density=False)
centers_maxbl = hist_maxbl[1][:-1] + np.diff(hist_maxbl[1])/2
err_maxbl = np.sqrt(hist_maxbl[0])
guesses_maxbl = [1.255119000413482e-05, 0.00024033100489723533, \
                 140.57936804139968, 5.710704216464667] 

popt_maxbl, pcov_maxbl = curve_fit(model_distribution, centers_maxbl, \
                            hist_maxbl[0], sigma=err_maxbl, \
                            p0=guesses_maxbl, absolute_sigma=True)
model_maxbl = model_distribution(centers_maxbl, popt_maxbl[0], \
                popt_maxbl[1], popt_maxbl[2], popt_maxbl[3])

# EE3 
hist_1 = np.histogram(integ1, bins=15, density=False)
centers_1 = hist_1[1][:-1] + np.diff(hist_1[1])/2
err_1 = np.sqrt(hist_1[0])
guesses_1 = [0.041259006657985944, 0.02746822769455662, 187.1398818353271, \
             0.37415081698175584]

popt_1, pcov_1 = curve_fit(model_distribution, centers_1, \
                            hist_1[0], sigma=err_1, p0=guesses_1, \
                            absolute_sigma=True)
model_1 = model_distribution(centers_1, popt_1[0], \
                popt_1[1], popt_1[2], popt_1[3])

# EE4 
hist_2 = np.histogram(integ2, bins=22, density=False)
centers_2 = hist_2[1][:-1] + np.diff(hist_2[1])/2
err_2 = np.sqrt(hist_2[0])
guesses_2=[0.01166442914535712, 0.027216513573794603, \
138.05171087453886, 2.0407688333303518]

popt_2, pcov_2 = curve_fit(model_distribution, centers_2, \
                            hist_2[0], sigma=err_2, p0=guesses_2,\
                            absolute_sigma=True) 
model_2 = model_distribution(centers_2, popt_2[0], \
                popt_2[1], popt_2[2], popt_2[3])

# EE5 
hist_3 = np.histogram(integ3, bins=110, density=False)
centers_3 = hist_3[1][:-1] + np.diff(hist_3[1])/2
err_3 = np.sqrt(hist_3[0])
guesses_3=[0.0012384368027604734, 0.01677871845602672, \
132.0841154918348, 6.294674857813498]

popt_3, pcov_3 = curve_fit(model_distribution, centers_3, \
                            hist_3[0], sigma=err_3, p0=guesses_3,\
                            absolute_sigma=True) 
model_3 = model_distribution(centers_3, popt_3[0], \
                popt_3[1], popt_3[2], popt_3[3])
    
# EE6
hist_op = np.histogram(optimal_amp, bins=65, density=False)
centers_op = hist_op[1][:-1] + np.diff(hist_op[1])/2
err_op = np.sqrt(hist_op[0])
guesses_op = [1.8282053998882475e-05, 0.00021078757570835906, \
              81.3596661718331, 5.130660005912854]

popt_op, pcov_op = curve_fit(model_distribution, centers_op, \
                            hist_op[0], sigma=err_op, p0=guesses_op,\
                            absolute_sigma=True) 
model_op = model_distribution(centers_op, popt_op[0], \
                popt_op[1], popt_op[2], popt_op[3])

#==================================ENERGY CONVERSION===========================

# Calibration factors
maxmin_factor = 10/(popt_maxmin[1]*1000) 
maxbl_factor = 10/(popt_maxbl[1]*1000)
integ1_factor = 10/(popt_1[1]*1000)
integ2_factor = 10/(popt_2[1]*1000)
integ3_factor = 10/(popt_3[1]*1000)
op_factor = 10/(popt_op[1]*1000)

integ3_factor = 10/(popt_3[1]*1000) 
v_signal = np.array(v_signal)
e_signal = v_signal*1000*integ3_factor


# Conversion to energy
e_maxmin = amp_maxmin*1000*maxmin_factor
e_maxbl = amp_maxbl*1000*maxbl_factor
e_1 = integ1*1000*integ1_factor
e_2 = integ2*1000*integ2_factor
e_3 = integ3*1000*integ3_factor
e_op = optimal_amp*1000*op_factor

# E1
hist_emaxmin = np.histogram(e_maxmin, bins=70, density=False)
centers_emaxmin = hist_emaxmin[1][:-1] + np.diff(hist_emaxmin[1])/2
err_emaxmin = np.sqrt(hist_emaxmin[0])
guesses_emaxmin = [0.5668591056087535, 10.01543469875703, \
             381.28445293343606, 21.894993160538913]

popt_emaxmin, pcov_emaxmin = curve_fit(model_distribution, centers_emaxmin, \
                            hist_emaxmin[0], sigma=err_emaxmin, \
                            p0 = guesses_emaxmin, absolute_sigma=True) 
model_emaxmin = model_distribution(centers_emaxmin, popt_emaxmin[0], \
                popt_emaxmin[1], popt_emaxmin[2], popt_emaxmin[3])
    
# E2
hist_emaxbl = np.histogram(e_maxbl, bins=67, density=False)
centers_emaxbl = hist_emaxbl[1][:-1] + np.diff(hist_emaxbl[1])/2
err_emaxbl = np.sqrt(hist_emaxbl[0])
guesses_emaxbl = [0.5222498623404375, 10.000005381277239, \
              140.57847770086295, 140.57847770086295]

popt_emaxbl, pcov_emaxbl = curve_fit(model_distribution, centers_emaxbl, \
                            hist_emaxbl[0], sigma=err_emaxbl,\
                            p0=guesses_emaxbl, absolute_sigma=True) 
model_emaxbl = model_distribution(centers_emaxbl, popt_emaxbl[0], \
                popt_emaxbl[1], popt_emaxbl[2], popt_emaxbl[3])

# E3
hist_e1 = np.histogram(e_1, bins=15, density=False)
centers_e1 = hist_e1[1][:-1] + np.diff(hist_e1[1])/2
err_e1 = np.sqrt(hist_e1[0])
guesses_e1 = [15.020626465702422, 9.99999872379892, \
              187.13991593860544, 0.3741541860429061]

popt_e1, pcov_e1 = curve_fit(model_distribution, centers_e1, \
                            hist_e1[0], sigma=err_e1,\
                         p0=guesses_e1, absolute_sigma=True) 
model_e1 = model_distribution(centers_e1, popt_e1[0], \
                popt_e1[1], popt_e1[2], popt_e1[3])
    
# E4
hist_e2 = np.histogram(e_2, bins=22, density=False)
centers_e2 = hist_e2[1][:-1] + np.diff(hist_e2[1])/2
err_e2 = np.sqrt(hist_e2[0])
guesses_e2 = [4.264536559175409, 10.000000359228368, \
              120.27871219680378, 1.6839410924215559]

popt_e2, pcov_e2 = curve_fit(model_distribution, centers_e2, \
                            hist_e2[0], sigma=err_e2,\
                         p0=guesses_e2, absolute_sigma=True) 
model_e2 = model_distribution(centers_e2, popt_e2[0], \
                popt_e2[1], popt_e2[2], popt_e2[3])
    
# E5
hist_e3 = np.histogram(e_3, bins=100, density=False)
centers_e3 = hist_e3[1][:-1] + np.diff(hist_e3[1])/2
err_e3 = np.sqrt(hist_e3[0])
guesses_e3 = [4.264536559175409, 10.000000359228368, \
              120.27871219680378, 1.6839410924215559]

popt_e3, pcov_e3 = curve_fit(model_distribution, centers_e3, \
                            hist_e3[0], sigma=err_e3,\
                         p0=guesses_e3, absolute_sigma=True) 
model_e3 = model_distribution(centers_e3, popt_e3[0], \
                popt_e3[1], popt_e3[2], popt_e3[3])
 
# E6    
hist_eop = np.histogram(e_op, bins=62, density=False)
centers_eop = hist_eop[1][:-1] + np.diff(hist_eop[1])/2
err_eop = np.sqrt(hist_eop[0])

popt_eop, pcov_eop = curve_fit(model_distribution, centers_eop, \
                            hist_eop[0], sigma=err_eop, \
                            absolute_sigma=True) 
model_eop = model_distribution(centers_eop, popt_eop[0], \
                popt_eop[1], popt_eop[2], popt_eop[3])

#==============================SIGNAL CALCULATIONS=============================

def fun(x, a, b, c, d, s, mu, e, f):
    return a*(np.exp(-(x-d)/b)-np.exp(-(x-d)/c)) + \
        model_distribution(x, s, mu, e, f) 

guesses_sig = [130, 1.8, 1/2, -0.1, 0.3 , 6, 18, 0]
hist_sig = np.histogram(e_signal, bins=100, density=False)
centers_sig = hist_sig[1][:-1] + np.diff(hist_sig[1])/2
err_sig = np.sqrt(hist_sig[0])
for i in range(err_sig.size):
    if err_sig[i] == 0:
        err_sig[i] = 1.0
    

popt_sig, pcov_sig = curve_fit(fun, centers_sig, \
                            hist_sig[0], sigma=err_sig, \
                              p0 = guesses_sig ,absolute_sigma=True) 
uc_sig = np.sqrt(np.diag(pcov_sig))
model_sig = fun(centers_sig, popt_sig[0], \
                popt_sig[1], popt_sig[2], popt_sig[3], popt_sig[4], \
                popt_sig[5], popt_sig[6], popt_sig[7])

#============================CHI-SQUARED=======================================

def chi(data, fit, sigma):
    chisquared = np.sum(((data-fit)/sigma)**2)
    return chisquared

def dof(num_bins, num_parameters):
    dof = num_bins - num_parameters
    return dof

# Uncalibrated
chi_maxmin = chi(hist_maxmin[0], model_maxmin, err_maxmin)
dof_maxmin = dof(hist_maxmin[0].size, popt_maxmin.size)
p_maxmin = 1 - chi2.cdf(chi_maxmin, dof_maxmin)

chi_maxbl = chi(hist_maxbl[0], model_maxbl, err_maxbl)
dof_maxbl = dof(hist_maxbl[0].size, popt_maxbl.size)
p_maxbl = 1 - chi2.cdf(chi_maxbl, dof_maxbl)

chi_1 = chi(hist_1[0], model_1, err_1)
dof_1 = dof(hist_1[0].size, popt_1.size)
p_1 = 1 - chi2.cdf(chi_1, dof_1)

chi_2 = chi(hist_2[0], model_2, err_2)
dof_2 = dof(hist_2[0].size, popt_2.size)
p_2 = 1 - chi2.cdf(chi_2, dof_2)

chi_3 = chi(hist_3[0], model_3, err_3)
dof_3 = dof(hist_3[0].size, popt_3.size)
p_3 = 1 - chi2.cdf(chi_3, dof_3)

chi_op = chi(hist_op[0], model_op, err_op)
dof_op = dof(hist_op[0].size, popt_op.size)
p_op = 1 - chi2.cdf(chi_op, dof_op)

# Calibrated
chi_emaxmin = chi(hist_emaxmin[0], model_emaxmin, err_emaxmin)
dof_emaxmin = dof(hist_emaxmin[0].size, popt_emaxmin.size)
p_emaxmin = 1 - chi2.cdf(chi_emaxmin, dof_emaxmin)

chi_emaxbl = chi(hist_emaxbl[0], model_emaxbl, err_emaxbl)
dof_emaxbl = dof(hist_emaxbl[0].size, popt_emaxbl.size)
p_emaxbl = 1 - chi2.cdf(chi_emaxbl, dof_emaxbl)

chi_e1 = chi(hist_e1[0], model_e1, err_e1)
dof_e1 = dof(hist_e1[0].size, popt_e1.size)
p_e1 = 1 - chi2.cdf(chi_e1, dof_e1)

chi_e2 = chi(hist_e2[0], model_e2, err_e2)
dof_e2 = dof(hist_e2[0].size, popt_e2.size)
p_e2 = 1 - chi2.cdf(chi_e2, dof_e2)

chi_e3 = chi(hist_e3[0], model_e3, err_e3)
dof_e3 = dof(hist_e3[0].size, popt_e3.size)
p_e3 = 1 - chi2.cdf(chi_e3, dof_e3)

chi_eop = chi(hist_eop[0], model_eop, err_eop)
dof_eop = dof(hist_eop[0].size, popt_eop.size)
p_eop = 1 - chi2.cdf(chi_eop, dof_eop)

# signal
chi_sig = chi(hist_sig[0], model_sig, err_sig)
dof_sig = dof(hist_sig[0].size, popt_sig.size)
p_sig = 1 - chi2.cdf(chi_sig, dof_sig)



#============================UNCALIBRATED PLOTS================================

# E1 
# plt.hist(amp_maxmin*1000, bins=70, density=False, histtype='step', color='black', \
#           lw=0.7, label='Data')
# plt.errorbar(centers_maxmin*1000, hist_maxmin[0], yerr=err_maxmin, fmt='none', \
#               color='black')
# plt.plot(centers_maxmin*1000, model_maxmin, label= 'Fit')
# plt.xlabel('Amplitude (mV)')
# plt.ylabel(r'Events/0.01 mV')
# plt.legend()
# plt.figtext(.2, .8, r'$\mu$ = 0.31 mV')
# plt.figtext(.2, .76, r'$\sigma = 0.01$ mV')
# plt.figtext(.2, .72, r'$\chi^2/DOF$ = 152.26/66')
# plt.figtext(.2, .68, r'$\chi^2 prob. = 9.0 \times 10^{-9}$')
# plt.savefig('maxmin_unc', bbox_inches='tight')


# E2
# plt.hist(amp_maxbl*1000, bins=53, density=False, histtype='step', color='black', \
#           lw = 0.7, label='Data')
# plt.errorbar(centers_maxbl*1000, hist_maxbl[0], yerr=err_maxbl, fmt='none', \
#               color='black')
# plt.plot(centers_maxbl*1000, model_maxbl, label='Fit')
# plt.xlabel('Amplitude (mV)')
# plt.ylabel(r'Events/0.01 mV')
# plt.legend()
# plt.figtext(.2, .8, r'$\mu$ = 0.24 mV')
# plt.figtext(.2, .76, r'$\sigma$ = 0.01 mV')
# plt.figtext(.2, .72, r'$\chi^2/DOF$ = 151.10/49')
# plt.figtext(.2, .68, r'$\chi^2 prob. = 2 \times 10^{-12}$')
# plt.savefig('maxbl_unc.png', bbox_inches='tight')


# E3
# plt.hist(integ1*1000, bins=15, density=False, histtype='step', color='black', \
#           lw = 0.7, label='Data')
# plt.errorbar(centers_1*1000, hist_1[0], yerr=err_1, fmt='none', \
#               color='black')
# plt.plot(centers_1*1000, model_1, label='Fit')
# plt.xlabel(r'Amplitude (mV)')
# plt.ylabel(r'Events/0.02 mV')
# plt.legend()
# plt.figtext(.2, .8, r'$\mu$ = 27.47 mV')
# plt.figtext(.2, .76, r'$\sigma = 41.26$ mV')
# plt.figtext(.2, .72, r'$\chi^2/DOF$ = 9.7/11')
# plt.figtext(.2, .68, r'$\chi^2$ prob. = 0.55')
# plt.savefig('1_unc.png', bbox_inches='tight')


# E4
# plt.hist(integ2*1000, bins=22, density=False, histtype='step', color='black', \
#           lw = 0.7, label='Data')
# plt.errorbar(centers_2*1000, hist_2[0], yerr=err_2, fmt='none', \
#               color='black')
# plt.plot(centers_2*1000, model_2, label='Fit')
# plt.xlabel(r'Amplitude(mV)')
# plt.ylabel('Events/0.004 mV')
# plt.legend()
# plt.figtext(.2, .8, r'$\mu$ = 27.26 mV')
# plt.figtext(.2, .76, r'$\sigma = 11.63$ mV')
# plt.figtext(.2, .72, r'$\chi^2/DOF$ = 36.0/18')
# plt.figtext(.2, .68, r'$\chi^2$ prob. = 0.007')
# plt.savefig('2_unc.png', bbox_inches='tight')

# E5
# plt.hist(integ3*1000, bins=110, density=False, histtype='step', color='black', \
#           lw= 0.7, label='Data')
# plt.errorbar(centers_3*1000, hist_3[0], yerr=err_3, fmt='none', \
#               color='black')
# plt.plot(centers_3*1000, model_3, label='Fit')
# plt.xlabel(r'$Amplitude (mV)')
# plt.ylabel('Events/0.0003 mV')
# plt.legend()
# plt.figtext(.2, .8, r'$\mu$ = 16.78 mV')
# plt.figtext(.2, .76, r'\sigma = 0.43 mV')
# plt.figtext(.2, .72, r'$\chi^2/DOF$ = 122.06/106')
# plt.figtext(.2, .68, r'$\chi^2$ prob. = 0.14')
# plt.savefig('3_unc.png', bbox_inches='tight')

# E6
# plt.hist(optimal_amp*1000, bins=65, density=False, histtype='step', color='black',\
#           lw = 0.7, label= 'Data')
# plt.errorbar(centers_op*1000, hist_op[0], yerr=err_op, fmt='none', \
#               color='black')
# plt.plot(centers_op*1000, model_op, label='Fit')
# plt.xlabel('Amplitude (mV)')
# plt.ylabel(r'Events/0.01 mV')
# plt.legend()
# plt.figtext(.2, .8, r'$\mu$ = 0.21 mV')
# plt.figtext(.2, .76, r'$\sigma$ =  0.02 mV')
# plt.figtext(.2, .72, r'$\chi^2/DOF$ = 88.94/61')
# plt.figtext(.2, .68, r'$\chi^2$ prob. = 0.01')
# plt.savefig('op_unc.png', bbox_inches='tight')

#=================================CALIBRATED PLOTS=============================

# E1
# plt.hist(e_maxmin, bins=70, density=False, histtype='step', color='black',\
#           lw = 0.7, label= 'Data')
# plt.errorbar(centers_emaxmin, hist_emaxmin[0], yerr=err_emaxmin, fmt='none', \
#               color='black')
# plt.plot(centers_emaxmin, model_emaxmin, label='Fit')
# plt.xlabel('Energy (keV)')
# plt.ylabel(r'Events/0.2 keV')
# plt.legend()
# plt.figtext(.2, .8, r'$\mu$ = 10.00 keV')
# plt.figtext(.2, .76, r'$\sigma$ =  0.47 keV')
# plt.figtext(.2, .72, r'$\chi^2/DOF$ = 152.26/66')
# plt.figtext(.2, .68, r'$\chi^2 prob. = 9.0 \times 10^{-9}$')
# plt.savefig('maxmin_e.png', bbox_inches='tight')

# E2
# plt.hist(e_maxbl, bins=67, density=False, histtype='step', color='black',\
#           lw = 0.7, label= 'Data')
# plt.errorbar(centers_emaxbl, hist_emaxbl[0], yerr=err_emaxbl, fmt='none', \
#               color='black')
# plt.plot(centers_emaxbl, model_emaxbl, label='Fit')
# plt.xlabel('Energy (keV)')
# plt.ylabel(r'Events/0.3 keV')
# plt.legend()
# plt.figtext(.2, .8, r'$\mu$ = 10.00 keV')
# plt.figtext(.2, .76, r'$\sigma =  0.52$ keV ')
# plt.figtext(.2, .72, r'$\chi^2/DOF$ = 162.01/63')
# plt.figtext(.2, .68, r'$\chi^2 prob. = 1.2 \times 10^{-10}$')
# plt.savefig('maxbl_e.png', bbox_inches='tight')

# E3
# plt.hist(e_1, bins=15, density=False, histtype='step', color='black',\
#           lw = 0.7, label= 'Data')
# plt.errorbar(centers_e1, hist_e1[0], yerr=err_e1, fmt='none', \
#               color='black')
# plt.plot(centers_e1, model_e1, label='Fit')
# plt.xlabel('Energy (keV)')
# plt.ylabel(r'Events/7.1 keV')
# plt.legend()
# plt.figtext(.2, .8, r'$\mu$ = 10.00 keV')
# plt.figtext(.2, .76, r'$\sigma =  15.02$ keV ')
# plt.figtext(.2, .72, r'$\chi^2/DOF$ = 9.7/11')
# plt.figtext(.2, .68, r'$\chi^2$ prob. = 0.55')
# plt.savefig('1_e.png', bbox_inches='tight')

# E4
# plt.hist(e_2, bins=22, density=False, histtype='step', color='black',\
#           lw = 0.7, label= 'Data')
# plt.errorbar(centers_e2, hist_e2[0], yerr=err_e2, fmt='none', \
#               color='black')
# plt.plot(centers_e2, model_e2, label='Fit')
# plt.xlabel('Energy (keV)')
# plt.ylabel(r'Events/1.4 keV')
# plt.legend()
# plt.figtext(.2, .8, r'$\mu$ = 10.0 0keV')
# plt.figtext(.2, .76, r'$\sigma =  4.26$ keV ')
# plt.figtext(.2, .72, r'$\chi^2/DOF$ = 36.0/18')
# plt.figtext(.2, .68, r'$\chi^2$ prob. = 0.007')
# plt.savefig('2_e.png', bbox_inches='tight')

# E5
# plt.hist(e_3, bins=100, density=False, histtype='step', color='black',\
#           lw = 0.7, label= 'Data')
# plt.errorbar(centers_e3, hist_e3[0], yerr=err_e3, fmt='none', \
#               color='black')
# plt.plot(centers_e3, model_e3, label='Fit')
# plt.xlabel('Energy (keV)')
# plt.ylabel(r'Events/0.20 keV')
# plt.legend()
# plt.figtext(.2, .8, r'$\mu$ = 10.00 0keV')
# plt.figtext(.2, .76, r'$\sigma =  0.25$ keV ')
# plt.figtext(.2, .72, r'$\chi^2/DOF$ = 115.8/96')
# plt.figtext(.2, .68, r'$\chi^2$ prob. = 0.08')
# plt.savefig('3_e.png', bbox_inches='tight')


# E6
# plt.hist(e_op, bins=62, density=False, histtype='step', color='black',\
#           lw = 0.7, label= 'Data')
# plt.errorbar(centers_eop, hist_eop[0], yerr=err_eop, fmt='none', \
#               color='black')
# plt.plot(centers_eop, model_eop, label='Chi-squared Fit')
# plt.xlabel('Energy (keV)')
# plt.ylabel(r'Events/0.3 keV')
# plt.legend()
# plt.figtext(.2, .8, r'$\mu$ = 10.00  keV')
# plt.figtext(.2, .76, r'$\sigma =  0.85$ keV ')
# plt.figtext(.2, .72, r'$\chi^2/DOF$ = 90.5/58')
# plt.figtext(.2, .68, r'$\chi^2$ prob. = 0.004')
# plt.savefig('3_op.png', bbox_inches='tight')


plt.hist(e_signal, bins=100, density=False, histtype='step', color='black',\
          lw = 0.7, label= 'Data')
plt.title('Energy Spectrum of the Signal Source and Background')
plt.errorbar(centers_sig, hist_sig[0], yerr=err_sig, fmt='none', \
              color='black', lw=0.5)
plt.plot(centers_sig, model_sig, label='Fit')
plt.xlabel('Energy (keV)')
plt.ylabel(r'Events/0.2 keV')
plt.legend()
# plt.figtext(.2, .8, r'$\mu$ = 10.00  keV')
# plt.figtext(.2, .76, r'$\sigma =  0.85$ keV ')
plt.figtext(.6, .70, r'$\chi^2/DOF$ = 106.13/92')
plt.figtext(.6, .66, r'$\chi^2$ prob. = 0.15')
plt.savefig('energy.png', bbox_inches='tight')





    
    
    
    








    





