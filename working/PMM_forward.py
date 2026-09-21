# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 22:59:59 2026

@author: 13391
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import math
import torch
import numpy as np
import scipy.io as sio

# =========================================================================
# 1. GLOBALS & SETUP
# =========================================================================
torch.set_default_dtype(torch.float64)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

FREQ_LIST = [30.0, 35.0, 50.0, 100.0, 150.0, 200.0, 250.0]
D_GLOBAL = 100.0
RHO_W_GLOBAL = 1.0
ZS_GLOBAL = 25.0

# True Seabed Parameters
cb = torch.tensor(2026.0, device=device)
rb = torch.tensor(1.3, device=device)
ab = torch.tensor(0.8, device=device)

# Receiver Grid (Matching your original arlpy setup)
r_vec = np.linspace(1.0, 5000.0, 1001)
r_tensor = torch.tensor(r_vec, device=device)

# =========================================================================
# 2. PYTORCH PMM FORWARD GENERATOR
# =========================================================================

def compute_prop_matrix(k, freq, return_modes=False):
    omega = 2 * math.pi * freq
    k = k.to(torch.complex128)

    # Use 1000 layers for high-fidelity mode shapes
    N_layers = 1000
    dz = D_GLOBAL / N_layers
    z_mid = torch.linspace(dz/2, D_GLOBAL - dz/2, N_layers, device=device, dtype=torch.float64)

    # Layered profile
    c = torch.zeros_like(z_mid)
    c = torch.where(z_mid <= 33.0, 1520.0 - (10.0/33.0)*z_mid, c)
    c = torch.where((z_mid > 33.0) & (z_mid <= 67.0), 1510.0 - (20.0/34.0)*(z_mid-33.0), c)
    c = torch.where(z_mid > 67.0, 1490.0 - (10.0/33.0)*(z_mid-67.0), c)

    V0 = torch.zeros_like(k, dtype=torch.complex128) 
    V1 = torch.ones_like(k, dtype=torch.complex128)  
    
    Psi_history = [V0.clone()] if return_modes else None

    # Propagate
    for j in range(N_layers):
        kz_j = torch.sqrt((omega/c[j])**2 - k**2 + 1e-12j)
        S = torch.sin(kz_j * dz)
        C = torch.cos(kz_j * dz)

        V0_new = C * V0 + (RHO_W_GLOBAL / kz_j) * S * V1
        V1_new = - (kz_j / RHO_W_GLOBAL) * S * V0 + C * V1
        V0, V1 = V0_new, V1_new

        if return_modes:
            Psi_history.append(V0.clone())

    # Half-Space Impedance
    kb_complex = (omega / cb) + 1j * ((ab * (freq/cb))/8.686)
    kz_b = torch.sqrt(kb_complex**2 - k**2 + 1e-12j)
    F = 1j * rb * V1 + kz_b * V0

    if return_modes:
        Psi_all = torch.stack(Psi_history)
        
        # Exact Normalization (Water + infinite bottom tail)
        Psi_sq = torch.abs(Psi_all)**2 / RHO_W_GLOBAL
        integral_water = (torch.sum(Psi_sq, dim=0) - 0.5 * Psi_sq[0, :] - 0.5 * Psi_sq[-1, :]) * dz
        integral_bottom = (torch.abs(V0)**2) / (2 * rb * torch.abs(kz_b.imag) + 1e-20)
        
        Psi_all = Psi_all / torch.sqrt(integral_water + integral_bottom)
        
        # We need the mode shapes at the source, and at ALL depths for the 2D plot
        z_idx_s = int(round((ZS_GLOBAL / D_GLOBAL) * N_layers))
        return F, Psi_all[z_idx_s, :], Psi_all, z_mid
    return F

for freq in FREQ_LIST:
    print(f"\n>>> GENERATING PMM FIELD: {freq} Hz <<<")
    omega = 2 * math.pi * freq
    
    # 1. Find the Roots (k)
    k0 = omega / 1480.0  
    k_min = omega / 3000.0
    
    # Real scan to find zero-crossings
    k_scan = torch.linspace(k0 * 0.999, k_min, 2000, device=device, dtype=torch.float64)
    F_scan = (compute_prop_matrix(k_scan, freq) / 1j).real
    sign_change = (F_scan[:-1] * F_scan[1:]) < 0
    valid_indices = torch.nonzero(sign_change).flatten()
    
    if len(valid_indices) == 0:
        print(f"No modes found for {freq} Hz.")
        continue
        
    k = k_scan[valid_indices].to(dtype=torch.complex128)
    
    # Secant Method Refinement
    eps = 1e-6
    for _ in range(20):
        F = compute_prop_matrix(k, freq)
        F_eps = compute_prop_matrix(k + eps, freq)
        k = k - F / ((F_eps - F) / eps + 1e-20)
        
    # Filter valid trapped modes
    valid = (k.real > k_min) & (k.real < k0*1.01) & (k.imag > -1e-8)
    k = k[valid]
    print(f"    -> Found {len(k)} trapped modes.")

    # 2. Extract Mode Shapes and Synthesize 2D Field
    _, Zs, Psi_all, z_vec_tensor = compute_prop_matrix(k, freq, return_modes=True)
    
    # Psi_all is [N_layers, N_modes]. r_tensor is [N_ranges].
    # We want a 2D Pressure matrix: [N_layers, N_ranges]
    phase = k[:, None] * r_tensor[None, :]
    H0 = torch.sqrt(2 / (math.pi * phase)) * torch.exp(1j * (phase - math.pi/4))
    
    # Sum over the modes (dim=1 of Psi_all, dim=0 of H0)
    # P(z, r) = i / (4 * rho_w) * sum_m [ Zs_m * Psi_m(z) * H0_m(r) ]
    p_2D = 1j/(4*RHO_W_GLOBAL) * torch.matmul(Psi_all * Zs, H0)
    
    # Calculate TL
    TL_2D = -20 * torch.log10(p_2D.abs() * 4.0 * math.pi + 1e-12)
    
    # 3. Save to .mat
    out_name = f'TL_Pytorch_PMM_layered_{int(freq)}.mat'
    sio.savemat(out_name, {
        'TL_2D': TL_2D.cpu().numpy(),
        'r_vec': r_vec,
        'z_vec': z_vec_tensor.cpu().numpy()
    })
    print(f"    -> Saved {out_name}")
    
#%% Plotting
import os
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np

FREQ_LIST = [30.0, 35.0, 50.0, 100.0, 150.0, 200.0, 250.0]

for freq in FREQ_LIST:
    filename = f'TL_Pytorch_PMM_layered_{int(freq)}.mat'
    
    if not os.path.exists(filename):
        print(f"Skipping {filename}: File not found.")
        continue
    
    print(f"Plotting {filename}...")
    
    # 1. Load the MATLAB file
    try:
        data = sio.loadmat(filename)
    except Exception as e:
        print(f"Could not read {filename}: {e}")
        continue
    
    # 2. Extract variables
    TL_2D = data['TL_2D']
    r_vec = data['r_vec'].squeeze()
    z_vec = data['z_vec'].squeeze()
    
    # 3. Visualization
    plt.figure(figsize=(12, 5))
    
    # imshow is much faster for dense Cartesian grids and ignores shape mismatches.
    # extent=[left, right, bottom, top] maps the axes.
    # By setting bottom=z_vec.max() and top=z_vec.min(), it automatically puts 0m at the top!
    mesh = plt.imshow(-TL_2D, aspect='auto', cmap='jet', vmin=-70, vmax=-30,
                      extent=[r_vec.min(), r_vec.max(), z_vec.max(), z_vec.min()])
    
    plt.colorbar(mesh, label='Transmission Loss (dB)')
    
    plt.xlabel('Range (m)', fontsize=12, fontweight='bold')
    plt.ylabel('Depth (m)', fontsize=12, fontweight='bold')
    plt.title(f'Acoustic Pressure Field (Stratified Waveguide): {freq} Hz', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
#%% Plotting profile
D_GLOBAL = 100.0
z = np.linspace(0, D_GLOBAL, 1000)

# Initialize and compute the piecewise sound speed profile
c = np.zeros_like(z)
c = np.where(z <= 33.0, 1520.0 - (10.0/33.0)*z, c)
c = np.where((z > 33.0) & (z <= 67.0), 1510.0 - (20.0/34.0)*(z-33.0), c)
c = np.where(z > 67.0, 1490.0 - (10.0/33.0)*(z-67.0), c)

# Create the plot
plt.figure(figsize=(5, 4))
plt.plot(c, z, color='b', linewidth=2.5)

# Invert the y-axis so the surface is at the top
plt.gca().invert_yaxis()

# Formatting
plt.xlabel('Sound Speed (m/s)', fontsize=12, fontweight='bold')
plt.ylabel('Depth (m)', fontsize=12, fontweight='bold')
plt.title('Stratified Sound Speed Profile', fontsize=14, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()
#%%
import os
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np

# --- Global Font Settings ---
plt.rc('xtick', labelsize=12)    
plt.rc('ytick', labelsize=12)    
plt.rc('axes', labelsize=14, titlesize=14) 

# --- Parameters ---
FREQ_LIST = [30.0, 35.0, 50.0, 100.0, 150.0, 200.0, 250.0]
ZS_GLOBAL = 25.0
D_GLOBAL = 100.0

# --- Pre-compute Sound Speed Profile ---
z_prof = np.linspace(0, D_GLOBAL, 1000)
c_prof = np.zeros_like(z_prof)
c_prof = np.where(z_prof <= 33.0, 1520.0 - (10.0/33.0)*z_prof, c_prof)
c_prof = np.where((z_prof > 33.0) & (z_prof <= 67.0), 1510.0 - (20.0/34.0)*(z_prof-33.0), c_prof)
c_prof = np.where(z_prof > 67.0, 1490.0 - (10.0/33.0)*(z_prof-67.0), c_prof)

# --- Plotting Loop ---
for freq in FREQ_LIST:
    filename = f'TL_Pytorch_PMM_layered_{int(freq)}.mat'
    
    if not os.path.exists(filename):
        print(f"Skipping {filename}: File not found.")
        continue
    
    print(f"Plotting {filename}...")
    
    # 1. Load the MATLAB file
    try:
        data = sio.loadmat(filename)
    except Exception as e:
        print(f"Could not read {filename}: {e}")
        continue
    
    # 2. Extract variables
    TL_2D = data['TL_2D']
    r_vec_m = data['r_vec'].squeeze()
    r_vec_km = r_vec_m / 1000.0 
    z_vec = data['z_vec'].squeeze()
    
    max_range_km = r_vec_km.max()
    
    # 3. Figure Setup
    fig = plt.figure(figsize=(12, 5))
    
    # FIX 2: Increased wspace from 0.15 to 0.30 to push the plots apart
    gs = fig.add_gridspec(2, 2, width_ratios=[2.5, 1], height_ratios=[1, 0.05], 
                          wspace=0.20, hspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, 0])      # Top-left: Pressure field
    ax2 = fig.add_subplot(gs[0, 1])      # Top-right: Sound speed profile
    cbar_ax = fig.add_subplot(gs[1, 0])  # Bottom-left: Colorbar
    
    # --- Subplot (a): Acoustic Pressure Field ---
    mesh = ax1.imshow(-TL_2D, aspect='auto', cmap='jet', vmin=-70, vmax=-30,
                      extent=[r_vec_km.min(), r_vec_km.max(), z_vec.max(), z_vec.min()])
    
    ax1.plot(0, ZS_GLOBAL, marker='*', markersize=16, markerfacecolor='m', 
             markeredgecolor='k', linestyle='None', label='Source', 
             clip_on=False, zorder=10)
    
    ax1.set_xlim(0, max_range_km)
    ax1.set_ylim(D_GLOBAL, 0)
    
    # FIX 1: Replaced the suptitle with this specific subplot title
    ax1.set_title(f'(a) Acoustic Pressure Field ({int(freq)} Hz)', fontweight='bold')
    ax1.set_xlabel('Range (km)')
    ax1.set_ylabel('Depth (m)')
    ax1.legend(loc='upper right', framealpha=1.0, fontsize=12)
    
    # Plot the colorbar
    cbar = plt.colorbar(mesh, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Pressure Level (dB)')
    
    # --- Subplot (b): Sound Speed Profile ---
    ax2.plot(c_prof, z_prof, color='b', linewidth=2.5)
    
    ax2.set_ylim(D_GLOBAL, 0)
    ax2.set_ylabel('Depth (m)')
    ax2.set_title('(b) Sound Speed Profile', fontweight='bold')
    ax2.set_xlabel('Sound Speed (m/s)')
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Clean up layout
    plt.subplots_adjust(top=0.88, bottom=0.1)
    
    plt.show()