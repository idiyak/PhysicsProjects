
'''

PHY407F 
LAB #2
QUESTION #4

Authors: Idil Yaktubay and Souren Salehi
September 2022


'''

'''

Question 4(a) - DO NOT SUBMIT

PLOTTING P AND Q VERY CLOSE TO 1

'''

import numpy as np
import matplotlib.pyplot as plt

n = 500
C = 10**(-16)
step = (1.02 - 0.98)/n
u_values = np.arange(0.98, 1.02, step)

def f(u):
    f = u**8/((u**4)*(u**4))
    return f

f_std = np.std(f(u_values))
print('The numpy.std standard deviation is ', f_std)

f_std_tb = np.sqrt(2)*C*f(u_values[250])
print('The equation standard deviation is ', f_std_tb/np.sqrt(2))

# plt.scatter(u_values, f(u_values), s=5)
# plt.title('The Quantity f for Values Close to u = 1')
# plt.xlabel('u')
# plt.ylabel('f(u)')
# plt.grid(linestyle='dotted')

# plt.savefig('f.png', bbox_inches='tight')

plt.scatter(u_values, f(u_values)-1, s=5)
plt.title('The Quantity f-1 for Values Close to u = 1')
plt.xlabel('u')
plt.ylabel('f(u)-1')
plt.grid(linestyle='dotted')

plt.savefig('f1.png', bbox_inches='tight')
