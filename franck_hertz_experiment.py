#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 22:14:28 2022

@author: souren
"""

import numpy as np 
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy as sc


e = 1.602e-19
name = ['fh1.txt','fh2.txt','fh3.txt','fh4.txt','fh5.txt']
i=3
V = np.loadtxt(name[i], skiprows=2, usecols= 0)
I = np.loadtxt(name[i], skiprows=2, usecols= 1)*10

V_err = np.array([4.982, 4.9499, 4.8715, 5.4062, 5.4309])/2
V_orders = np.array([2.0, 3.0, 4.0, 5.0, 6.0])

V_minima = np.array([8.231, 12.853, 18.053, 23.397, 28.719])
I_minima = np.array([0.4587, 0.8886, 1.4358, 2.0999, 2.6912])

def f(x, a, b):
    return a*x + b

popt, pcov = curve_fit(f, V_orders, V_minima, sigma=V_err, absolute_sigma=True)
err = np.sqrt(np.diag(pcov))

def chi(data, fit, sigma):
    chisquared = np.sum(((data-fit)/sigma)**2)
    return chisquared



# plt.plot(V_orders, f(V_orders, popt[0], popt[1]), '--', label='Chi-squared Fit', color='black')
# plt.errorbar(V_orders, V_minima, yerr=V_err, fmt='none')
# plt.scatter(V_orders, V_minima, label='Minima')
# # plt.scatter(V, I, s=9, label='Current Data')
# # plt.scatter(V_minima, I_minima, label='Current Minima')

# plt.xlabel('Dip Number')
# plt.ylabel('Accelerating Voltage (V)')
# plt.title(r'Accelerating Voltages Corresponding to the Current Minima With Respect to Dip Numbers')
# plt.grid()
# plt.legend()

# plt.savefig('raw_data8.png', bbox_inches='tight')


# dI = np.ones(len(I))*10**(-9)


# def b(x, w):
#     return np.convolve(x, np.ones(w), 'valid') / w

# I = b(I, 12)
# V = b(V, 12)



# def f(x, s, mu, a, b):
#     return a*np.exp(-np.power(x-mu, 2.)/(2*np.power(s, 2.))) + b


# n=(0,1,2,3,4,5,6,7,8,9,10,11)
# maxima =(np.where((I[2:-2] > I[0:-4]) * (I[2:-2] > I[4:]))[0] +1)
# minima =(np.where((I[1:-1] < I[0:-2]) * (I[1:-1] < I[2:]))[0] +1)

# Vmax = V[maxima]
# Vmin = V[minima]
# Imax = I[maxima]
# Imin = I[minima]

# def q(x,a,b):
#     return a*x + b

# m=0
# n=1716

# #Vmin = np.delete(Vmin[12:23],3)
# #Imin = np.delete(Imin[12:23],3)
# Vmin = np.delete(Vmin[11:17],1)
# Imin = np.delete(Imin[11:17],1)

# #for i in (0,1,2,3):
# #    delta[i] = Vmin[i+1]-Vmin[i]
    
# delta = np.diff(Vmin)

# a = 25.6
# b = 4.9
# guess = (0.02, 7, 0.01, 0.08)
# #popt, pcov = curve_fit(f, V[m:n], I[m:n], sigma=dI[m:n], absolute_sigma=True)
# #model = f(V, popt[0],popt[1], popt[2], popt[3])
# plt.plot(V[m:n],I[m:n], linestyle='--', label='Convoluted data')
# plt.xlabel("Accelerating voltage [V]")
# plt.ylabel('Anode Current [nA]')
# plt.suptitle('Variation of current with accelerating voltage at 170' r'$^o$''C', size=14)
# plt.scatter(Vmin ,Imin, label='minima', color='#781616')
# plt.legend(loc="upper left", prop={"size":12})
# plt.show()
# #plt.plot(V, model, linestyle ='--', color='r')

# r = (2,3,4,5,6)
# r = np.array(r)

# err = np.array([2.1, 3.1, 3.9, 4.2, 4.4])/2

# def chisquared(obs, exp, sig, N, m):
#     return (1/(N-m))*np.sum(((obs-exp)/sig)**2)


# popt, pcov = curve_fit(q, r, Vmin, sigma = err, absolute_sigma=True)
# std = np.sqrt(np.diag(pcov))
# model = q(r,popt[0],popt[1])

# chi = chisquared(Vmin, model, err, 5, 2)

# plt.errorbar(r, Vmin, yerr = err, color='#781616', linestyle='none')
# plt.scatter(r, Vmin, label = 'Voltage corresponding to to current minima', color ='#002b23')
# plt.plot(r, model, label = 'Model of energy transfer with order of minima', linestyle = '--')
# plt.ylabel('Voltages at current dips[V]')
# plt.xlabel("Order of minima")
# plt.suptitle('Voltage at current minima vs order of minima', size = 14)
# plt.legend(loc="upper left", prop={"size":12})
# plt.show()



