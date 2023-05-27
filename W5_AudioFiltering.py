""

PHY407F 
LAB #5
QUESTION #2

Author: Idil Yaktubay
October 2022

"""


"""

Question 2(b) PSEUDOCODE

CODE THAT PRODUCES A PLOT OF THE DATA IN THE GraviteaTime.wav FILE
AS A FUNCTION OF TIME FOR CHANNELS 0 AND 1

Author: Idil Yaktubay

"""


# Import read, write from scipy.io.wavfile
# Import rfft, rfftfrew, irfft from scipy.fft
# Import numpy
# Import matplotlib.pyplot

# Read the GraviteaTime.wav data using read() from scipy.io.wavfile
    # Specify what we are reading as a comment
    # Specify the dimensions of the data as a comment
    # Actually read the data

# Separate GraviteaTime.wav data into channels 0 and 1

# Set the number of data points in each channel to use when creating time axis

# Using the sample frequency, create the time axis

# Using matplotlib.pyplot, produce separate plots of channels 0 and 1 with 
# respect to time
    # Produce channel 0 plot
    # Produce channel 1 plot


"""

Question 2(b) REAL PYTHON CODE

CODE THAT PRODUCES A PLOT OF THE DATA IN THE GraviteaTime.wav FILE
AS A FUNCTION OF TIME FOR CHANNELS 0 AND 1

Author: Idil Yaktubay

"""


# Import necessary modules
from scipy.io.wavfile import read, write
from scipy.fft import rfft, rfftfreq, irfft
import numpy as np
import matplotlib.pyplot as plt

# Read the Data:
    # Sample is the sampling frequency, data is the data in each channel
    # Dimensions: nsamples rows and 2 columns [nsamples, 2]
sample, data = read('GraviteaTime.wav')

# Separate data into channels
channel_0 = data[:, 0]
channel_1 = data[:, 1]

# Set the number of data points in each channel
N_Points = len(channel_0)

# Using the sample frequency, set the time axis
t_ax = np.arange(0, N_Points/sample, 1/sample)

# UNCOMMENT TO GENERATE PLOTS
# Plot the data for channel 0 and and channel 1
# fig, axs = plt.subplots(2)
# fig.suptitle('Channel 0 and Channel 1 Data from GraviteaTime.wav as a Function of Time')
    
# Channel 0:
# axs[0].set_title('Channel 0 Data as a Function of Time')
# axs[0].set_xlabel('Time (s)')
# axs[0].set_ylabel('Channel 0 Data')
# axs[0].grid(linestyle='dotted')
# axs[0].plot(t_ax, channel_0)
    
# # Channel 1:
# axs[1].set_title('Channel 1 Data as a Function of Time')
# axs[1].set_xlabel('Time (s)')
# axs[1].set_ylabel('Channel 1 Data')
# axs[1].grid(linestyle='dotted')
# axs[1].plot(t_ax, channel_1, color='green')

# plt.subplot_tool()


"""

Question 2(c) REAL PYTHON CODE

CODE THAT CREATES A SHORTER TIME AXIS TO USE WHEN PLOTTING A SHORTER TIME 
SERIES TO SEE THE IMPACTS OF FILTERING

Author: Idil Yaktubay

"""

# Set new time axis for a time series 30 ms long
t_ax_2 = np.arange(0, float(30)/1000, 1.0/float(sample))

# Set number of data points in new channels 0 and 1 arrays
N_Points_2 = len(t_ax_2)


"""

Question 2(d) PSEUDOCODE

CODE THAT FILTERS ALL FREQUENCIES GREATER THAN 880 Hz FROM CHANNELS 0 AND 1  
AND PLOTS ORIGINAL AND FILTERED COEFFICIENT AMPLITUDES AND ORIGINAL AND 
FILTERED TIME SERIES OF CHANNELS 0 AND 1 OVER THE SMALL TIME SEGMENT FROM ABOVE

"""


# Create an array of the frequency domain

# Perform fast fourier transform to find coefficients for channels 0 and 1
	# Channel 0 transform
	# Channel 1 transform

# Copy the two previously set transforms into two arrays to record the original 
# coefficient amplitudes
	# Copy of channel 0 transform
	# Copy of channel 1 transform

# Filter out frequencies higher than 880 Hz by setting the corresponding 
# coefficients to zero for channels 0 and 1
	# Filter channel 0 coefficients
	# Filter channel 1 coefficients

# Transform the filtered coefficients back to the time domain using inverse 
# fast fourier transform

# Plot the amplitudes of original and filtered Fourier coefficients

# Plot the original and filtered time series over the 30 ms interval

# Output the filtered time series into a new .wav file
	# Set an empty array the same shape and type as original data
	# Fill out the empty array with the filtered data for channels 0 and 1
	# Write a .wav file with the same sampling frequency that contains 
	# filter data


"""

Question 2(d) REAL PYTHON CODE 

CODE THAT FILTERS ALL FREQUENCIES GREATER THAN 880 Hz FROM CHANNELS 0 AND 1  
AND PLOTS ORIGINAL AND FILTERED COEFFICIENT AMPLITUDES AND ORIGINAL AND 
FILTERED TIME SERIES OF CHANNELS 0 AND 1 OVER THE SMALL TIME SEGMENT FROM ABOVE

Author: Idil Yaktubay

"""


# Set frequency domain
freq_domain = rfftfreq(N_Points, 1.0/float(sample))

# Perform fast fourier transform to find coefficients for channels 0 and 1
c_0 = rfft(channel_0) # Channel 0 transform
c_1 = rfft(channel_1) # Channel 1 transform

# Copy the transforms to record original coefficients
c_0_og = np.copy(c_0) # Channel 0 copy
c_1_og = np.copy(c_1) # Channel 1 copy

# Set the coefficients corresponding to > 880 Hz to zero
for i in range(freq_domain.size):
    if freq_domain[i] > 880:
        c_0[i] = 0.0
        c_1[i] = 0.0

# Transform the filtered coefficients back to the time domain
filtered_data_0 = irfft(c_0)
filtered_data_1 = irfft(c_1)

# Output the filtered time series into a new .wav file
data_out = np.empty(data.shape, dtype = data.dtype)
data_out[:, 0] = filtered_data_0
data_out[:, 1] = filtered_data_1
write('GraviteaTime_lpf.wav', sample, data_out)


# UNCOMMENT BELOW TO GENERATE PLOTS
# Plot the amplitudes of original and filtered Fourier coefficients 
# fig, axs = plt.subplots(2, 2)
# fig.suptitle('Original and Filtered Amplitudes of Fourier Coefficients for Channels 0 and 1 of GraviteaTime.wav')

# axs[0, 0].set_title('Original Coefficient Amplitudes for Channel 0', y=1.08)
# axs[0, 0].set_xlabel('k')
# axs[0, 0].set_ylabel('Coefficient Amplitude')
# axs[0, 0].grid(linestyle='dotted')

# axs[0, 1].set_title('Filtered Coefficient Amplitudes for Channel 0', y=1.08)
# axs[0, 1].set_xlabel('k')
# axs[0, 1].set_ylabel('Coefficient Amplitude')
# axs[0, 1].grid(linestyle='dotted')

# axs[0, 0].plot(abs(c_0_og))
# axs[0, 1].plot(abs(c_0))

# axs[1, 0].set_title('Original Coefficient Amplitudes for Channel 1', y=1.08)
# axs[1, 0].set_xlabel('k')
# axs[1, 0].set_ylabel('Coefficient Amplitude')
# axs[1, 0].grid(linestyle='dotted')

# axs[1, 1].set_title('Filtered Coefficient Amplitudes for Channel 1', y=1.08)
# axs[1, 1].set_xlabel('k')
# axs[1, 1].set_ylabel('Coefficient Amplitude')
# axs[1, 1].grid(linestyle='dotted')

# axs[1, 0].plot(abs(c_1_og), color='green')
# axs[1, 1].plot(abs(c_1), color='green')

# plt.subplot_tool()

# UNCOMMENT BELOW TO GENERATE PLOTS
# Plot the original and filtered time series over the 30 ms interval
# fig, axs = plt.subplots(2, 2)
# fig.suptitle('Original and Filtered Time Series of Channels 0 and 1 of GraviteaTime.wav over 30 ms')

# axs[0, 0].set_title(\
#             'Original Time Series for Channel 0 with respect to Time', y=1.08)
# axs[0, 0].set_xlabel('Time (s)')
# axs[0, 0].set_ylabel('Original Channel 0 Data')
# axs[0, 0].grid(linestyle='dotted')

# axs[0, 1].set_title(\
#             'Filtered Time Series for Channel 0 with respect to Time', y=1.08)
# axs[0, 1].set_xlabel('Time (s)')
# axs[0, 1].set_ylabel('Filtered Channel 0 Data')
# axs[0, 1].grid(linestyle='dotted')

# axs[1, 0].set_title(
#     'Original Time Series for Channel 1 with respect to Time', y=1.08)
# axs[1, 0].set_xlabel('Time (s)')
# axs[1, 0].set_ylabel('Original Channel 1 Data')
# axs[1, 0].grid(linestyle='dotted')

# axs[1, 1].set_title(\
#         'Filtered Time Series for Channel 1 with respect to Time', y=1.08)
# axs[1, 1].set_xlabel('Time (s)')
# axs[1, 1].set_ylabel('Filtered Channel 1 Data')
# axs[1, 1].grid(linestyle='dotted')

# axs[0, 0].plot(t_ax_2, channel_0[: N_Points_2])
# axs[0, 1].plot(t_ax_2, filtered_data_0[: N_Points_2])
# axs[1, 0].plot(t_ax_2, channel_1[: N_Points_2], color='green')
# axs[1, 1].plot(t_ax_2, filtered_data_1[: N_Points_2], color='green')

# plt.subplot_tool()
