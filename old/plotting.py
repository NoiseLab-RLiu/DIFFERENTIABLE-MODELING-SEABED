# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 23:10:13 2025

@author: 13391
"""

import numpy as np
import matplotlib.pyplot as plt

# Load the .npz file
data = np.load('losses_archiveImp250219.npz', allow_pickle=True)

#np.savez('losses_archiveImp.npz', losses_total=loss_arr_total, losses_U = loss_arr_U, losses_F=loss_arr_F, A0=A0_arr, A1 = A1_arr, C0=C0_arr, C1=C1_arr, D=D_arr)
# List all the arrays stored in the .npz file
print("Arrays in the .npz file:", data.files)

# Access individual arrays using their keys
A0 = data['A0']  # Replace 'array1' with the actual key
A1 = data['A1']  # Replace 'array2' with the actual key
C0 = data['C0']  # Replace 'array1' with the actual key
C1 = data['C1']  # Replace 'array2' with the actual key
A0 = data['A0']  # Replace 'array1' with the actual key
D = data['D']  # Replace 'array2' with the actual key

# Print the arrays
# print("Array 1:", array1)
#%% print("Array 2:", array2)
A0_true = -2.3*np.ones(len(A0))
A1_true = -0.9*np.ones(len(A1))
C0_true = 1.92*np.ones(len(C0))
C1_true = -1.26*np.ones(len(C1))
D_true = 1.2*np.ones(len(D))
# Close the .npz file (optional, but good practice)
#data.close()
plt.figure()
plt.plot(A0, label='A0')
plt.plot(A1, label='A1')
plt.plot(C0, label='C0')
plt.plot(C1, label='C1')
plt.plot(D, label='D')
# Adding title and labels
plt.title('Plot of 5 Arrays')
plt.xlabel('Index')
plt.ylabel('Value')
# Adding a legend
plt.legend()
plt.show()
#%%
plt.figure(figsize=(8, 5))  # optional, to make the figure larger

# Plot lines with a certain linewidth
lw = 1
mk = 8
me = 4000
# Plot each main array
pA0, = plt.plot(A0, label='-a0', linewidth=lw)
pA1, = plt.plot(A1, label='-a1', linewidth=lw)
pC0, = plt.plot(C0, label='c0', linewidth=lw)
pC1, = plt.plot(C1, label='c1', linewidth=lw)
pD,  = plt.plot(D,  label='d',  linewidth=lw)

# Plot dashed "true" lines using the same colors
plt.plot(A0_true, label='-a0_true', color=pA0.get_color(), linestyle='None', marker='o', markerfacecolor='none', markevery=me, linewidth=lw, markersize=mk)
plt.plot(A1_true, label='-a1_true', color=pA1.get_color(), linestyle='None', marker='o', markerfacecolor='none', markevery=me, linewidth=lw,markersize=mk)
plt.plot(C0_true, label='c0_true', color=pC0.get_color(), linestyle='None', marker='o', markerfacecolor='none', markevery=me, linewidth=lw,markersize=mk)
plt.plot(C1_true, label='c1_true', color=pC1.get_color(), linestyle='None', marker='o', markerfacecolor='none', markevery=me, linewidth=lw,markersize=mk)
plt.plot(D_true,  label='d_true', color=pD.get_color(),  linestyle='None', marker='o', markerfacecolor='none', markevery=me, linewidth=lw,markersize=mk)

# Title, labels, and axis range
#plt.title('Plot of 5 Arrays and Their "True" Lines')
plt.xlabel('Epoch', fontsize=14)
#plt.ylabel('Value', fontsize=14)
plt.xlim([0, 30000])  # or plt.xlim([0, len(A0)]) if len(A0)=80000

# Place legend outside the figure to the right
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.,fontsize=14)

# Adjust the layout so that the legend does not overlap or get cut off
plt.tight_layout()

# Show the figure
plt.show()