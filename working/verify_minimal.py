# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 23:52:44 2026

@author: 13391
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 01:12:17 2026

@author: 13391
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import math
import torch
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

# =========================================================================
# 0. CONFIGURATION
# =========================================================================
torch.set_default_dtype(torch.float64)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Globals
C_W_GLOBAL   = 1500.0
RHO_W_GLOBAL = 1.0
D_GLOBAL     = 100.0
FREQ_GLOBAL  = 150.0
OMEGA_GLOBAL = 2 * math.pi * FREQ_GLOBAL
ZS_GLOBAL    = 25.0
ZR_GLOBAL    = 50.0 

# =========================================================================
# 1. LOAD DATA
# =========================================================================
mat_file = 'TL_kraken.mat'

try:
    mat_data = sio.loadmat(mat_file)
    TL_tensor = torch.tensor(mat_data['TL_2D'], device=device)
    r_full = torch.linspace(1.0, 5000.0, TL_tensor.shape[1], device=device)
    
    # Select Depth 50m and Range 1-5km
    depth_idx = 50 
    mask = (r_full >= 1000.0) & (r_full <= 5000.0)
    r_meas = r_full[mask]
    TL_meas = TL_tensor[depth_idx, mask]
    
    print(f"[OK] Data Loaded. Points: {len(r_meas)}")

except Exception as e:
    print(f"[ERROR] {e}")
    exit()

# =========================================================================
# 2. FORWARD MODEL (Exact Copy of Pattern-Matching Version)
# =========================================================================
def unified_complex_solver(c_b, rho_b, alpha_b):
    """
    Robust Solver using Mesh Scanning instead of WKB.
    Guarantees catching Mode 1.
    """
    c_w, rho_w, D, omega = C_W_GLOBAL, RHO_W_GLOBAL, D_GLOBAL, OMEGA_GLOBAL
    
    k0 = omega / c_w
    kb_complex = (omega / c_b) + 1j * (alpha_b * (FREQ_GLOBAL / c_b) / 8.686)
    
    # 1. DENSE MESH SCAN (The Fix)
    # We scan from k0 (water) down to kb (bottom)
    # 1000 points is enough to catch all peaks
    k_min = (omega / 2500.0) # Lower bound (faster speed)
    k_scan = torch.linspace(k0 * 0.999, k_min, 1000, device=device, dtype=torch.float64)
    
    # Check Characteristic Function Sign Changes
    kz_w = torch.sqrt(torch.clamp(k0**2 - k_scan**2, min=1e-12))
    kz_b = torch.sqrt(torch.clamp(k_scan**2 - (omega/c_b)**2, min=1e-12))
    
    # F = tan(kz_w * D) + (rho_b/rho_w)*(kz_w/kz_b)
    # We look for zero crossings of F
    tan_g = torch.tan(kz_w * D)
    F = tan_g + (rho_b/rho_w) * (kz_w / kz_b)
    
    # Find indices where sign changes
    sign_change = (F[:-1] * F[1:]) < 0
    # Also filter out asymptotes (where tan jumps from +inf to -inf)
    # Asymptotes happen when F jumps by a huge amount
    jump_filter = torch.abs(F[:-1] - F[1:]) < 50.0 
    
    valid_indices = torch.nonzero(sign_change & jump_filter).flatten()
    
    if len(valid_indices) == 0:
        return torch.tensor([]).to(device)
        
    # Initial Guesses from the scan
    k_guess = k_scan[valid_indices].to(dtype=torch.complex128)
    
    # 2. NEWTON REFINEMENT (Same as before)
    k = k_guess.clone()
    for _ in range(30): #30
        kz_w = torch.sqrt(k0**2 - k**2)
        kz_b = torch.sqrt(kb_complex**2 - k**2)
        
        s_w, c_w = torch.sin(kz_w*D), torch.cos(kz_w*D)
        F = rho_w * kz_b * s_w + 1j * rho_b * kz_w * c_w
        
        dkzw_dk = -k / (kz_w + 1e-20)
        dkzb_dk = -k / (kz_b + 1e-20)
        
        dT1 = rho_w * (dkzb_dk * s_w + kz_b * c_w * D * dkzw_dk)
        dT2 = 1j * rho_b * (dkzw_dk * c_w - kz_w * s_w * D * dkzw_dk)
        
        k = k - F / (dT1 + dT2 + 1e-20)
        
    # 3. FILTER DUPLICATES
    # Newton might converge 2 guesses to the same root
    k_real = k.real
    if len(k) > 1:
        # Sort
        idx = torch.argsort(k_real, descending=True)
        k = k[idx]
        # Unique check
        diff = torch.abs(k[1:] - k[:-1])
        unique = torch.cat([torch.tensor([True], device=device), diff > 1e-5])
        k = k[unique]

    return k

def forward_model(c_b, rho_b, alpha_b, r_range):
    k_complex = unified_complex_solver(c_b, rho_b, alpha_b)
    
    if len(k_complex) == 0: 
        return torch.ones_like(r_range)*100.0 + (c_b*0+rho_b*0+alpha_b*0)

    k0 = OMEGA_GLOBAL / C_W_GLOBAL
    kz_w = torch.sqrt(k0**2 - k_complex**2)
    
    alpha_neper = (alpha_b * (FREQ_GLOBAL / c_b)) / 8.686
    kb_complex = (OMEGA_GLOBAL / c_b) + 1j * alpha_neper
    kz_b = torch.sqrt(kb_complex**2 - k_complex**2)
    
    # Amplitudes
    term1 = D_GLOBAL/2
    term2 = torch.sin(2*kz_w*D_GLOBAL)/(4*kz_w)
    term3 = (RHO_W_GLOBAL/rho_b) * (torch.sin(kz_w*D_GLOBAL)**2) / (2j * kz_b)
    A_m = 1.0 / torch.sqrt(term1 - term2 + term3)
    
    # Field
    Zs = A_m * torch.sin(kz_w * ZS_GLOBAL)
    Zr = A_m * torch.sin(kz_w * ZR_GLOBAL)
    phase = k_complex[:, None] * r_range[None, :]
    H0 = torch.sqrt(2 / (math.pi * phase)) * torch.exp(1j * (phase - math.pi/4))
    
    p_sum = 1j/(4*RHO_W_GLOBAL) * (Zs[:, None] * Zr[:, None] * H0).sum(dim=0)
    
    # 4*pi Correction
    return -20 * torch.log10(p_sum.abs() * 4.0 * math.pi + 1e-12)

# =========================================================================
# 3. VERIFICATION
# =========================================================================
print("\n=== GROUND TRUTH VERIFICATION ===")

# True Parameters
cb_true   = torch.tensor(2000.0, device=device)
rho_true  = torch.tensor(1.800, device=device)
alpha_true = torch.tensor(0.500, device=device)

cb_true   = torch.tensor(1909.6, device=device)
rho_true  = torch.tensor(1.648, device=device)
alpha_true = torch.tensor(0.489, device=device)

cb_true = cb_opt
rho_true = rb_opt
alpha_true = ab_opt

# Compute Prediction
with torch.no_grad():
    #pred_tl = forward_model(cb_true, rho_true, alpha_true, r_meas)
    pred_tl = forward_model(cb_true, rho_true, alpha_true, r_meas, stage=2)
    # Compute Loss
    mse_loss = (pred_tl - TL_meas).pow(2).mean().item()

    # Compute Correlation
    p_cent = pred_tl - pred_tl.mean()
    t_cent = TL_meas - TL_meas.mean()
    corr = torch.sum((p_cent * t_cent) / (torch.norm(p_cent) * torch.norm(t_cent))).item()

print(f"Ground Truth Parameters: c_b=2000, rho=1.8, alpha=0.5")
print(f"MSE Loss        : {mse_loss:.6f}")
print(f"Correlation     : {corr:.6f} (Target: 1.0)")

# Plot
r_plot = r_meas.cpu().numpy()
p_plot = pred_tl.cpu().numpy()
t_plot = TL_meas.cpu().numpy()

plt.figure(figsize=(10, 5))
plt.plot(r_plot, t_plot, 'k', label='Kraken (MATLAB)', linewidth=2, alpha=0.7)
plt.plot(r_plot, p_plot, 'r--', label='Python Forward Model (True Params)', linewidth=2)
plt.title(f'Ground Truth Verification (MSE={mse_loss:.4f})')
plt.xlabel('Range (m)')
plt.ylabel('TL (dB)')
plt.legend()
plt.gca().invert_yaxis()
plt.grid(True)
plt.show()

#%%

def unified_complex_solver(c_b, rho_b, alpha_b):
    """
    Complex Grid Solver: Scans a 2D strip in the complex plane to catch 
    both Trapped (Real) and Leaky (Complex) modes.
    """
    c_w, rho_w, D, omega = C_W_GLOBAL, RHO_W_GLOBAL, D_GLOBAL, OMEGA_GLOBAL
    
    # 1. GENERATE COMPLEX GRID GUESSES
    # Real part: from k0 (water) down to well below cutoff
    k_min = omega / 3000.0
    k0    = omega / c_w
    
    # We use 400 points along Real axis and 3 points along Imaginary axis
    # Leaky modes have positive imaginary parts (decay). 
    # We guess 0.0 (trapped), 1e-4, and 5e-4 (leaky).
    real_vals = torch.linspace(k0*1.01, k_min, 400, device=device, dtype=torch.float64)
    imag_vals = torch.tensor([0.0, 1e-4, 1e-3], device=device, dtype=torch.float64)
    
    # Create the mesh (Total 1200 guesses)
    # This brute-force ensures we are inside the "basin of attraction" for every mode
    grid_real, grid_imag = torch.meshgrid(real_vals, imag_vals, indexing='ij')
    k = (grid_real + 1j * grid_imag).flatten()
    
    # Physics Constants for the Loop
    kb_real = omega / c_b
    alpha_neper = (alpha_b * (FREQ_GLOBAL / c_b)) / 8.686
    kb_complex = (omega / c_b) + 1j * alpha_neper
    
    # 2. VECTORIZED NEWTON SOLVER (Run on all 1200 points at once)
    for _ in range(30): #30
        # Wavenumbers
        kz_w = torch.sqrt(k0**2 - k**2)
        kz_b = torch.sqrt(kb_complex**2 - k**2)
        
        s_w, c_w = torch.sin(kz_w*D), torch.cos(kz_w*D)
        
        # Characteristic Equation: F(k)
        # F = rho_w * kz_b * s_w + 1j * rho_b * kz_w * c_w
        F = rho_w * kz_b * s_w + 1j * rho_b * kz_w * c_w
        
        # Analytic Derivative: dF/dk
        dkzw_dk = -k / (kz_w + 1e-20)
        dkzb_dk = -k / (kz_b + 1e-20)
        
        dT1 = rho_w * (dkzb_dk * s_w + kz_b * c_w * D * dkzw_dk)
        dT2 = 1j * rho_b * (dkzw_dk * c_w - kz_w * s_w * D * dkzw_dk)
        dF_dk = dT1 + dT2
        
        # Update
        # We add a small regularization to dF_dk to prevent div/0 explosions
        k = k - F / (dF_dk + 1e-20)
    
    # 3. FILTER AND SORT
    k_real = k.real
    
    # Filter 1: Physical Bounds
    # Real part must be positive and below max wavenumber
    # Imag part must be positive (decaying) or extremely close to 0
    valid_mask = (k_real > k_min) & (k_real < k0 - 1e-4) & (k.imag > -1e-8)
    k = k[valid_mask]
    
    
    # Filter 2: Remove Duplicates
    if len(k) > 0:
        # Sort by Real part descending (Phase Speed convention)
        idx = torch.argsort(k.real, descending=True)
        k = k[idx]
        
        # Calculate distance between consecutive roots
        diff = torch.abs(k[1:] - k[:-1])
        
        # Keep only roots separated by at least 1e-4
        # We assume the grid converged multiple guesses to the same root
        unique_mask = torch.cat([torch.tensor([True], device=device), diff > 1e-4])
        k = k[unique_mask]

    return k

cb_true   = torch.tensor(2000.0, device=device)
rho_true  = torch.tensor(1.800, device=device)
alpha_true = torch.tensor(0.500, device=device)

cb_true   = torch.tensor(2026, device=device)
rho_true  = torch.tensor(1.55, device=device)
alpha_true = torch.tensor(0.3, device=device)
with torch.no_grad():
    # Call your solver
    #k_complex = unified_complex_solver(cb_true, rho_true, alpha_true)
    k_complex = unified_complex_solver(cb_true, rho_true, alpha_true, coarse_mode=False)
    # Sort descending by Real part (standard acoustic convention)
    # KRAKEN sorts from highest k (lowest speed) to lowest k (highest speed)
    sorted_indices = torch.argsort(k_complex.real, descending=True)
    k_sorted = k_complex[sorted_indices]

# =========================================================================
# PRINT COMPARISON TABLE
# =========================================================================
print("\n=== PYTHON SOLVER OUTPUT ===")
print(f"{'Mode':<5} | {'k_real (1/m)':<15} | {'k_imag (1/m)':<18} | {'Phase Speed (m/s)':<18}")
print("-" * 65)

for i, k in enumerate(k_sorted):
    k_r = k.real.item()
    k_i = k.imag.item()
    phase_speed = OMEGA_GLOBAL / k_r
    
    # Note: KRAKEN output uses 'alpha' for imaginary part. 
    # Check if sign matches your convention (usually k_i is positive for decay)
    print(f"{i+1:<5} | {k_r:<15.10f} | {k_i:<18.10e} | {phase_speed:<18.6f}")

print("-" * 65)
print(f"Total Modes Found: {len(k_sorted)}")