# Idil Yaktubay and Maheen Mufti - The Milikan Experiment 

import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 


#=================CALCULATING VALUES OF Q AND UNCERTAINTIES====================

# Loading the file that tells us which camera was used for each droplet
camera = pd.read_excel('organized_data.xlsx', usecols=[1])
# Loading the data that tells us the stopping voltage for each droplet
stopping_voltages = pd.read_excel('organized_data.xlsx', usecols=[2])


# Relevant data points will be accumulated in lists below
q_values = []
q_uncertainties = []
terminal_velocities = []
vt_uncertainties = []
i = 1

# Assigning the uncertainty in stopping voltage measurements
delta_vstop = 20

while i < 53:
    
    # Opening the file with data corresponding to a given trial
    file_name = str(i) + '.xlsx'
    file = pd.read_excel(file_name)
    
    # Loading frame numbers and positions from the file
    frames = np.array(file['Numbers'].values.tolist())
    pixel_positions = np.array(file['Untitled'].values.tolist())
    
    # Finding which camera the droplet was recorded with and stopping voltage
    which_camera = camera.values.tolist()[i-1][0]
    stopping_voltage = stopping_voltages.values.tolist()[i-1][0]
    
    # Assigning the camera calibration and the uncertainty in the calibration
    if which_camera == 'R':
        calibration = 520
    else: 
        calibration = 540
    calibration_uncertainty = 1
    
    # Converting frames to time in seconds and pixels to meters
    times = frames/10 
    meter_positions = (pixel_positions/calibration)/1000
    
    # Calculating the distance travelled by the droplet
    distance = meter_positions[meter_positions.size - 1] - meter_positions[0]
    
    # Calculating the time it took the droplet to travel this distance
    time = times[times.size - 1]
    
    # Calculating terminal velocity
    terminal_velocity = distance/time
    
    # Calculating the value of Q
    Q = (2e-10)*(terminal_velocity**(3/2))/stopping_voltage
    
    # Calculating the uncertainty in terminal velocity
    delta_vt = (calibration_uncertainty/calibration)*terminal_velocity
    vt_uncertainties.append(delta_vt)
    
    # Calculating the uncertainty in Q 
    delta_q = \
        np.sqrt((((3e-10)*(terminal_velocity**(1/2))*delta_vt)/ \
                 stopping_voltage)**2 + \
    (((2e-10)*(terminal_velocity**(3/2))*delta_vstop)/ \
     (stopping_voltage**2))**2)
    
    q_values.append(Q)
    q_uncertainties.append(delta_q)
    terminal_velocities.append(terminal_velocity)
    
    # Updating i
    i = i + 1

# Turning lists into arrays because it's more convenient to work with arrays
q_values = np.array(q_values)
q_uncertainties = np.array(q_uncertainties)
terminal_velocities = np.array(terminal_velocities)
vt_uncertainties = np.array(vt_uncertainties)


#==========================FINDING THE VALUE OF e==============================

# A list that contains the range of each data point within its uncertainty
uncertainty_ranges = []
for i in range(q_values.size):
    uncertainty_range = []
    left = q_values[i] - q_uncertainties[i]
    right = q_values[i] + q_uncertainties[i]
    uncertainty_range.append(left)
    uncertainty_range.append(right)
    uncertainty_ranges.append(uncertainty_range)

# In bins, we accumulate lists that contain data points that correspond to the
# same multiple of e within their uncertainties.
bins = []
for i in range(14):
    bins.append([])
    
ne = np.arange(1.602e-19, 2.403e-18, step=1.602e-19)
for uncertainty_range in uncertainty_ranges:
    for i in range(ne.size):
        if uncertainty_range[0] <= ne[i] <= uncertainty_range[1]:
            bins[i].append(q_values[\
                                uncertainty_ranges.index(uncertainty_range)])

# Some data points fall into more than two lists in bins. We eyeballed the 
# histogram to see where this happens. Then, we remove the duplicate data as 
# follows:
bins[3] = bins[3][0: 4] + [bins[3][6]] + bins[3][8: 12]
bins[4] = [bins[4][0]] + bins[4][2:5] +bins[4][6:]
bins[9] = bins[9][0:2] + [bins[9][3]]
bins[11] = bins[11][2:]


# Some data points do not contain any multiple of e within the range of their
# uncertainties. These data points are outliers and we won't use them to 
# calculate the final value of e. The following lines of code identifies
# the outliers so that we can present them in our lab report. 

# Prepare a list that contains empty strings
is_outlier = []
for i in range(q_values.size):
    is_outlier.append('')

# Accumulate 'yes' if the data point contains ne for at least one integer n 
# within its uncertainty. At the end, empty strings in is_oulier will 
# represent outlier data points. 
for r in uncertainty_ranges:
    for e in ne:
        if r[0] <= e <= r[1]:
            is_outlier[uncertainty_ranges.index(r)] = 'yes'
outlier_data = []
for i in range(len(is_outlier)):
    if is_outlier[i] == '':
        outlier_data.append(q_values[i])


# Calculating q_i for each group of data in bins and the uncertainties
qi = []
qi_uncertainties = []
for i in range(1, 12):
    mean_q = np.mean(bins[i])
    qi.append(mean_q)
    if len(bins[i]) == 1:
        qi_uncertainties.append(0.0)
    else:
        uncertainty = np.std(bins[i], ddof=1)/np.sqrt(len(bins[i]))
        qi_uncertainties.append(uncertainty)

# Extracting the value of e from each group of data in bins 
e_values = []
index = 0
for i in range(2, 11):
    e_value = qi[index]/i
    e_values.append(e_value)
    index = index + 1
e_values.append(qi[-1]/12)

# Finding the uncertainty in each value of e
e_uncertainties = []
for i in range(len(e_values)-1):
    e_uncertainty = (qi_uncertainties[i]/qi[i])*e_values[i]
    e_uncertainties.append(e_uncertainty)
e_uncertainty_12 = (qi_uncertainties[-1]/qi[-1])*e_values[-1]
e_uncertainties.append(e_uncertainty_12)

# Final value of e and its uncertainty
final_value_e = np.mean(e_values)
square_uncertainty_sum = 0
for uncertainty in e_uncertainties:
    square_uncertainty_sum = square_uncertainty_sum + uncertainty**2
sum_of_e_values = sum(np.array(e_values))
final_e_uncertainty = (np.sqrt(square_uncertainty_sum)/sum_of_e_values) \
    *final_value_e   

#===================Q: Radius of Typical Droplets==============================

# Defining the relevant constants in SI units to calculate radii
eta = 1.827e-5
g = 9.80
rho_oil = 875.3
rho_air = 1.204

radii = []
for vt in terminal_velocities:
    radius = np.sqrt((9*eta*vt)/(2*g*(rho_oil - rho_air)))
    radii.append(radius)
average_radius = np.mean(radii)
#=========================PRINTING THE IMPORTANT RESULTS ======================

print('This array contains the charge of each droplet:', q_values, '\n')
print('This array contains the uncertainty in the charge of each droplet:', \
      q_uncertainties, '\n')
print('This array contains the values of q_i:', qi, '\n')
print('This array contains the uncertainties in the values of q_i', \
      qi_uncertainties, '\n')
print('This array contains the values of e_i extracted from each group q_i:',\
      e_values, '\n')
print('This array contains the uncertainties in e_i values', e_uncertainties, \
      '\n')
print('THE FINAL VALUE OF THE ELEMENTARY CHARGE:', final_value_e, \
      'C +-', final_e_uncertainty, 'C', '\n' )

print('The typical droplet in our experiment had a radius of approximately', \
      str(round(average_radius*1e6, 1)), 'microns.')
    

#================================PLOTTING THE DATA============================


# droplet_numbers = np.arange(1, 53)
# 1) Plotting the calculated values of terminal velocities with uncertainties
# plt.scatter(droplet_numbers, terminal_velocities)
# plt.errorbar(droplet_numbers, terminal_velocities, yerr=vt_uncertainties, \
#              fmt='none', ecolor='red', capsize=3, \
#                  label='Uncertainty in Terminal Velocity')
# plt.title(r'Calculated Field-Free Terminal Velocity $v_t$ of Each Droplet')
# plt.xlabel('Droplet Number')
# plt.ylabel(r'Terminal Velocity (m/s)')
# plt.grid(which='major', color='#DDDDDD', linewidth=0.8)
# plt.grid(which='minor', color='grey', linestyle=':', linewidth=0.5)
# plt.minorticks_on()
# plt.legend()


# 2) Plotting the measured values of stopping voltages with uncertainties
# stopping_voltages_list = stopping_voltages.values.tolist()
# stopping_voltages_list.pop()
# stopping_voltages = np.array(stopping_voltages_list)
# stopping_voltages_uncertainties = np.zeros(52) + 20
# plt.scatter(droplet_numbers, stopping_voltages)
# plt.errorbar(droplet_numbers, stopping_voltages, \
#              yerr= stopping_voltages_uncertainties, 
#               fmt='none', ecolor='red', capsize=3, \
#                   label='Uncertainty in Stopping Voltage')
# plt.title(r'Measured Stopping Voltage $V_{stop}$ of Each Droplet')
# plt.xlabel('Droplet Number')
# plt.ylabel(r'Stopping Voltage $V_{stop}$ (V)')
# plt.grid(which='major', color='#DDDDDD', linewidth=0.8)
# plt.grid(which='minor', color='grey', linestyle=':', linewidth=0.5)
# plt.minorticks_on()
# plt.legend()


# 3) Plotting the calculated values of Q with uncertainties
# plt.scatter(droplet_numbers, q_values)
# plt.errorbar(droplet_numbers, q_values, yerr= q_uncertainties, fmt='none', \
#              ecolor='red', capsize = 3, \
#                  label='Uncertainty in Charge')
# plt.title('Calculated Charge Q of Each Oil Droplet')
# plt.xlabel('Droplet Number')
# plt.ylabel('Charge Q (C)')
# plt.grid(which='major', color='#DDDDDD', linewidth=0.8)
# plt.grid(which='minor', color='grey', linestyle=':', linewidth=0.5)
# plt.minorticks_on()
# plt.legend()
#==============================================================================

    
#========================PLOTTING THE OVERLAPPING HISTOGRAMS===================


for e in ne:
    plt.axvline(x=e, linestyle='--', color='grey')

plt.bar(np.array(bins[1]), np.array([1]), width=0.015e-18, color='skyblue', \
    edgecolor='grey', linewidth=1.0)
plt.hist(np.array(bins[2]), bins=5, rwidth=0.7, edgecolor='grey', \
    color='pink', linewidth=1.0)
plt.hist(np.array(bins[3]), bins=3, rwidth=0.4, color='skyblue', \
    edgecolor='grey', linewidth=1.0)
plt.hist(np.array(bins[4]),  bins=4, rwidth=0.5, color='pink', \
    edgecolor='grey', linewidth=1.0)
plt.hist(np.array(bins[5]), bins=6, rwidth=0.5, color='skyblue', \
    edgecolor='grey', linewidth=1.0)
plt.hist(np.array(bins[6]), bins=2, rwidth=0.3, color='pink', \
    edgecolor='grey', linewidth=1.0)
plt.bar(np.array(bins[7]), np.array([1]), width = 0.015e-18, \
    color='skyblue', edgecolor='grey', linewidth=1.0)
plt.hist(np.array(bins[8]), bins=2, rwidth=0.13, color='pink', \
    edgecolor='grey', linewidth=1.0)
plt.hist(np.array(bins[9]), bins=2, rwidth=0.1, color='skyblue', \
    edgecolor='grey', linewidth=1.0)
# got rid of 10 because it overlaps with 9
plt.hist(np.array(bins[11]), bins=2, rwidth=0.13, color='pink', \
    edgecolor='grey', linewidth=1.0)
# got rid of 12 because it overlaps with 11
# Got rid of 13 because it overlaps with 12
plt.grid()
plt.minorticks_on()
plt.ylabel('Number of Measurements')
plt.xlabel('Drop Charge (C)')











            
    


                                                       



        
    




    
    
    
    
    
    



