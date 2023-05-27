"""
PHY407H F
LAB #11
QUESTION #2

Authors: Idil Yaktubay, Nicolas Grisuard, Paul Kushner

NOTE: THIS PROGRAM IS A MODIFIED VERSION OF l11-protein-start.py TO ANSWER
ONLY PARTS (d) and (e) OF QUESTION #2. I RECOMMEND COMMENTING OUT THE SECTION
FOR PART (d) BEFORE RUNNING THE CODE FOR PART (e) FOR FASTER EXECUTION AND 
VICE VERSA.

December 2022
"""

from random import random, randrange
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

def calc_energy(monomer_coords, monomer_array):
    """ Compute energy of tertiary structure of protein """
    energy = 0.0

    # compute energy due to all adjacencies (incl. directly bonded monomers)
    for i in range(N):
        for nghbr in [[-1, 0], [1, 0], [0, -1], [0, 1]]:  # 4 neighbours
            nghbr_monomer = monomer_array[monomer_coords[i, 0] + nghbr[0],
                                          monomer_coords[i, 1]+nghbr[1]]

            if nghbr_monomer == 1:  # check neighbour is not empty
                energy += eps

    # divide by 2 to correct for double-counting
    energy = .5*energy

    # correct energy to not count directly bonded monomer neighbours
    energy -= (N-1)*eps

    return energy


def dist(position1, position2):
    """ Compute distance """
    return ((position1[0]-position2[0])**2+(position1[1]-position2[1])**2)**.5


font = {'family': 'DejaVu Sans', 'size': 14}  # adjust fonts
rc('font', **font)
dpi = 150


eps = -5.0  # interaction energy
N = 30  # length of protein
T = 0.5  # temperature for Monte Carlo for part (d)
n = int(2e6)  # number of Monte Carlo steps for part (d)

energy_array = np.zeros(n)  # initialize array to hold energy

# initialize arrays to store protein information
# 1st column is x coordinates, 2nd column is y coordinates, of all N monomers
monomer_coords = np.zeros((N, 2), dtype='int')

# initialize position of polymer as horizontal line in middle of domain
monomer_coords[:, 0] = range(N//2, 3*N//2)
monomer_coords[:, 1] = N

# 2D array representing lattice,
# equal to 0 when a lattice point is empty,
# and equal to 1 when there is a monomer at the lattice point
monomer_array = np.zeros((2*N+1, 2*N+1), dtype='int')

# fill lattice array
for i in range(N):
    monomer_array[monomer_coords[i, 0], monomer_coords[i, 1]] = 1

# calculate energy of initial protein structure
energy = calc_energy(monomer_coords, monomer_array)

# These constants will be used for part (e)
MONOMER_COORDS = np.copy(monomer_coords) # UNCHANGABLE initial coordinates
MONOMER_ARRAY = np.copy(monomer_array) # UNCHANGABLE lattice 
ENERGY = energy # UNCHANGABLE initial energy

"""
QUESTION 2(d) 

CODE THAT USES COOL DOWN TECHNIQUE TO CALCULATE THE FINAL ENERGY OF 
PROTEIN FOLDING FOR A TEMPERATURE OF T=0.5

Authors: Idil Yaktubay and Nicolas Grisuard

"""

# Part (d) modification:
T_f = 0.5
T_steps = 4
T_i = T_f + T_steps - 1
T_array = np.zeros(n)
for step in range(T_steps):
    T_array[step*n//T_steps:(step+1)*n//T_steps] = \
    (T_i-T_f)*(1-step/(T_steps-1)) + T_f


# do Monte Carlo procedure to find optimal protein structure
for j in range(n):
    energy_array[j] = energy
    
    T = T_array[j] # Part (d) modification
    
    # move protein back to centre of array
    shift_x = int(np.mean(monomer_coords[:, 0])-N)
    shift_y = int(np.mean(monomer_coords[:, 1])-N)
    monomer_coords[:, 0] -= shift_x
    monomer_coords[:, 1] -= shift_y
    monomer_array = np.roll(monomer_array, -shift_x, axis=0)
    monomer_array = np.roll(monomer_array, -shift_y, axis=1)

    # pick random monomer
    i = randrange(N)
    cur_monomer_pos = monomer_coords[i, :]

    # pick random diagonal neighbour for monomer
    direction = randrange(4)

    if direction == 0:
        neighbour = np.array([-1, -1])  # left/down
    elif direction == 1:
        neighbour = np.array([-1, 1])  # left/up
    elif direction == 2:
        neighbour = np.array([1, 1])  # right/up
    elif direction == 3:
        neighbour = np.array([1, -1])  # right/down

    new_monomer_pos = cur_monomer_pos + neighbour

    # check if neighbour lattice point is empty
    if monomer_array[new_monomer_pos[0], new_monomer_pos[1]] == 0:
        # check if it is possible to move monomer to new position without
        # stretching chain
        distance_okay = False
        if i == 0:
            if dist(new_monomer_pos, monomer_coords[i+1, :]) < 1.1:
                distance_okay = True
        elif i == N-1:
            if dist(new_monomer_pos, monomer_coords[i-1, :]) < 1.1:
                distance_okay = True
        else:
            if dist(new_monomer_pos, monomer_coords[i-1, :]) < 1.1 \
                and dist(new_monomer_pos, monomer_coords[i+1, :]) < 1.1:
                distance_okay = True

        if distance_okay:
            # calculate new energy
            new_monomer_coords = np.copy(monomer_coords)
            new_monomer_coords[i, :] = new_monomer_pos

            new_monomer_array = np.copy(monomer_array)
            new_monomer_array[cur_monomer_pos[0], cur_monomer_pos[1]] = 0
            new_monomer_array[new_monomer_pos[0], new_monomer_pos[1]] = 1

            new_energy = calc_energy(new_monomer_coords, new_monomer_array)

            if random() < np.exp(-(new_energy-energy)/T):
                # make switch
                energy = new_energy
                monomer_coords = np.copy(new_monomer_coords)
                monomer_array = np.copy(new_monomer_array)

plt.figure()
plt.title('$T$ = {0:.1f}, $N$ = {1:d}'.format(T, N))
plt.plot(energy_array)
plt.xlabel('MC step')
plt.ylabel('Energy')
plt.grid()
plt.tight_layout()
plt.savefig('energy_vs_step_T{0:d}_N{1:d}_n{2:d}.pdf'.format(int(10*T), N, n),
            dpi=dpi)

plt.figure()
plt.plot(monomer_coords[:, 0], monomer_coords[:, 1], '-k')  # plot bonds
plt.title('$T$ = {0:.1f}, Energy = {1:.1f}'.format(T, energy))
# plot monomers
for i in range(N):
    plt.plot(monomer_coords[i, 0], monomer_coords[i, 1], '.r', markersize=15)
plt.xlim([N/3.0, 5.0*N/3.0])
plt.ylim([N/3.0, 5.0*N/3.0])
plt.axis('equal')
# plt.xticks([])  # we just want to see the shape
# plt.yticks([])
plt.tight_layout()
plt.savefig('final_protein_T{0:d}_N{1:d}_n{2:d}.pdf'.format(int(10*T), N, n),
            dpi=dpi)

print('Energy averaged over last QUARTER of simulations using a cooling down technique is: {0:.2f}'
      .format(np.mean(energy_array[n//4:])))

plt.show()

"""
QUESTION 2 (e)

CODE THAT FINDS A QUANTITATIVE RELATIONSHIP BETWEEN AVERAGE PROTEIN ENERGY AND 
TEMPERATURE USING DIFFERENT TEMPERATURE VALUES

Authors: Idil Yaktubay and Nicolas Grisuard

NOTE: The code for part (d) is embedded into our code for part (e).
"""

T_diff = np.arange(10.0, 0.0, step=-0.5) # Different temperatures for part (e)
n_e = 500000 # Number of Monte Carlo steps for part (e)
E_mean = np.zeros(T_diff.size) # Mean energies at last half of simulations
stds = np.zeros(T_diff.size) # Standard deviations at last half of simulations

# Do Monte Carlo for various temperatures 
for num in range(T_diff.size): 
    
    T = T_diff[num] 
    T_check = T_diff[num]
    
    # Cooling down technique from part (d) for lower temperatures
    if T_check <= 1.5: # Check if temperature is low
        # Create a "cool down" array to use for when temperature is low...
        T_f = T_check
        T_steps = 10
        T_i = T_f + T_steps - 1
        T_array = np.zeros(n_e) # "Cool down" array
        for step in range(T_steps):
            T_array[step*n_e//T_steps:(step+1)*n_e//T_steps] = \
            (T_i-T_f)*(1-step/(T_steps-1)) + T_f
    
    # Set the parameters and arrays to initial conditions
    energy_array = np.zeros(n_e)
    monomer_coords = MONOMER_COORDS
    monomer_array = MONOMER_ARRAY
    energy = ENERGY
   
    # Do Monte Carlo simulation for the current temperature
    for j in range(n_e):
        energy_array[j] = energy
        
        if T_check <= 1.5: # Check if the "cool down" array is needed...
            T = T_array[j] 
        
        # move protein back to centre of array
        shift_x = int(np.mean(monomer_coords[:, 0])-N)
        shift_y = int(np.mean(monomer_coords[:, 1])-N)
        monomer_coords[:, 0] -= shift_x
        monomer_coords[:, 1] -= shift_y
        monomer_array = np.roll(monomer_array, -shift_x, axis=0)
        monomer_array = np.roll(monomer_array, -shift_y, axis=1)
    
        # pick random monomer
        i = randrange(N)
        cur_monomer_pos = monomer_coords[i, :] # current monomer position
    
        # pick random diagonal neighbour for monomer
        direction = randrange(4)
    
        if direction == 0:
            neighbour = np.array([-1, -1])  # left/down
        elif direction == 1:
            neighbour = np.array([-1, 1])  # left/up
        elif direction == 2:
            neighbour = np.array([1, 1])  # right/up
        elif direction == 3:
            neighbour = np.array([1, -1])  # right/down
    
        new_monomer_pos = cur_monomer_pos + neighbour
    
        # check if neighbour lattice point is empty
        if monomer_array[new_monomer_pos[0], new_monomer_pos[1]] == 0:
            # check if it is possible to move monomer to new position without
            # stretching chain
            distance_okay = False
            if i == 0:
                if dist(new_monomer_pos, monomer_coords[i+1, :]) < 1.1:
                    distance_okay = True
            elif i == N-1:
                if dist(new_monomer_pos, monomer_coords[i-1, :]) < 1.1:
                    distance_okay = True
            else:
                if dist(new_monomer_pos, monomer_coords[i-1, :]) < 1.1 \
                    and dist(new_monomer_pos, monomer_coords[i+1, :]) < 1.1:
                    distance_okay = True
    
            if distance_okay:
                # calculate new energy
                new_monomer_coords = np.copy(monomer_coords)
                new_monomer_coords[i, :] = new_monomer_pos
    
                new_monomer_array = np.copy(monomer_array)
                new_monomer_array[cur_monomer_pos[0], cur_monomer_pos[1]] = 0
                new_monomer_array[new_monomer_pos[0], new_monomer_pos[1]] = 1
    
                new_energy = calc_energy(new_monomer_coords, new_monomer_array)
    
                if random() < np.exp(-(new_energy-energy)/T):
                    # make switch
                    energy = new_energy
                    monomer_coords = np.copy(new_monomer_coords)
                    monomer_array = np.copy(new_monomer_array)
       
    E_mean[num] = np.mean(energy_array[n_e//2:]) # Mean energy over last half
    stds[num] = np.std(energy_array[n_e//2:], ddof=1) # Std over last half


# PLOT MEAN ENERGY VERSUS TEMPERATURE
plt.figure()
plt.errorbar(T_diff, E_mean, stds, fmt='none', color='red',\
              label='Standard Deviation Over\nLast Half of Simulation')
plt.plot(T_diff, E_mean, 'o', color='blue', label='Mean Energy Over Last Half\n of Simulation')
plt.title('Average Protein Energy versus Temperature', size=15)
plt.grid(linestyle='dotted')
plt.xlabel('Temperature', size=15)
plt.ylabel('Mean Energy', size=15)
plt.legend(bbox_to_anchor=(0.5, -0.16) , \
            loc='upper center', prop={'size': 15})


