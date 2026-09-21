# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 17:11:48 2026

@author: 13391
"""

import os
import numpy as np
import scipy.io as sio
import arlpy.uwapm as pm

# 1. Point Python to your Acoustic Toolbox binary folder
AT_BIN = r'D:\atWin10_2020_11_4\windows-bin-20201102'
os.environ['PATH'] += os.pathsep + AT_BIN

FREQ_LIST = [30.0, 35.0, 50.0, 100.0, 150.0, 200.0, 250.0]

# 2. Define the layered environment natively in Python


for freq in FREQ_LIST:
    print(f"\n>>> PROCESSING FREQUENCY: {freq} Hz <<<")
    
    env = pm.create_env2d(
        frequency=freq,       # <--- Frequency is defined HERE
        depth=100.0,
        soundspeed=[[0.0, 1520.0], [33.0, 1510.0], [67.0, 1490.0], [100.0, 1480.0]],
        bottom_soundspeed=2026.0,
        bottom_density=1300.0, 
        bottom_absorption=0.8,
        tx_depth=25.0,
        rx_depth=np.linspace(0, 100, 101),
        rx_range=np.linspace(1, 5000, 1001)
    )
    
    # 3. Generate the acoustic field natively using KRAKEN
    # arlpy handles all the binary extraction and coherent summation automatically
    tloss_complex = pm.compute_transmission_loss(env)
    
    # Calculate Transmission Loss (handling the proper absolute math)
    TL_2D = -20 * np.log10(np.abs(tloss_complex) + 1e-20)
    
    # 4. Save to .mat file exactly as your MATLAB script did
    out_name = f'TL_kraken_layered_{int(freq)}.mat'
    sio.savemat(out_name, {
        'TL_2D': TL_2D,
        'P_2D': tloss_complex,
        'r_vec': env['rx_range'],
        'z_vec': env['rx_depth'],
        'freq': freq,
        'cb': 2026.0,
        'rhob': 1.3,
        'alphab': 0.8
    })
    print(f"    -> Saved to {out_name}")

print("\n[DONE] All frequencies processed.")

#%% Plotting
import os
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np

# List of frequencies corresponding to your saved files
FREQ_LIST = [30, 35, 50, 100, 150, 200, 250]

for freq in FREQ_LIST:
    filename = f'TL_kraken_layered_{freq}.mat'
    
    if not os.path.exists(filename):
        print(f"Skipping {filename}: File not found.")
        continue
    
    # 1. Load the MATLAB file
    data = sio.loadmat(filename)
    
    # 2. Extract variables
    # We use .squeeze() because MATLAB vectors often load as (1, N) or (N, 1)
    TL_2D = data['TL_2D']
    r_vec = data['r_vec'].squeeze()
    z_vec = data['z_vec'].squeeze()
    
    print(f"Plotting {filename}...")

    # 3. Create the plot
    plt.figure(figsize=(12, 5))
    
    # 'jet_r' is often used for TL (Red/Yellow = high intensity/low loss)
    # Realistic TL values for this range usually sit between 40 and 90 dB
    mesh = plt.pcolormesh(r_vec, z_vec, TL_2D, cmap='jet_r', shading='auto', vmin=40, vmax=90)
    
    plt.colorbar(mesh, label='Transmission Loss (dB)')
    
    # Standard acoustics convention: Depth 0 (surface) at the top
    plt.gca().invert_yaxis()
    
    plt.xlabel('Range (m)', fontsize=12, fontweight='bold')
    plt.ylabel('Depth (m)', fontsize=12, fontweight='bold')
    plt.title(f'Layered Waveguide TL: Frequency = {freq} Hz', fontsize=14)
    
    plt.tight_layout()
    plt.show()
