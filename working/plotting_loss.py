# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 21:16:28 2026

@author: l00905881
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import math

# =========================================================================
# 0. CONFIGURATION
# =========================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float64)

# Fixed Geometry
C_W_GLOBAL   = 1500.0
RHO_W_GLOBAL = 1.0
D_GLOBAL     = 100.0
ZS_GLOBAL    = 25.0
ZR_GLOBAL    = 50.0 

# Frequencies to use for the loss calculation
FREQ_LIST = [30.0, 35.0, 50.0, 100.0, 150.0, 200, 250]

# Grid Search Configuration
cb_start, cb_end, cb_step = 1510, 2100, 10
rho_start, rho_end, rho_step = 1.0, 2.0, 0.02 # Using 0.02 for slightly better resolution than 0.1

# True Parameters (Ground Truth)
TRUE_PARAMS = {
    'cb': 2026.0,
    'rho': 1.3,
    'alpha': 0.8
}

# =========================================================================
# 1. SOLVER ENGINE (Directly from your code)
# =========================================================================
def unified_complex_solver(c_b, rho_b, alpha_b, omega, freq, coarse_mode=False, initial_seeds=None):
    c_w, rho_w, D = C_W_GLOBAL, RHO_W_GLOBAL, D_GLOBAL
    k0 = omega / c_w
    k_min = omega / 3000.0
    
    alpha_safe = torch.clamp(alpha_b, min=1e-4)
    kb_complex = (omega / c_b) + 1j * ((alpha_safe * (freq/c_b))/8.686)

    # Simplified Solver for Grid Search (Always Cold Start or Coarse)
    # Using the complex grid search strategy from your code for robustness
    real_vals = torch.linspace(k0*1.01, k_min, 400, device=device)
    imag_vals = torch.tensor([0.0, 1e-4, 5e-4], device=device) 
    g_real, g_imag = torch.meshgrid(real_vals, imag_vals, indexing='ij')
    k = (g_real + 1j * g_imag).flatten()
    iters = 20

    # Newton Refinement
    for _ in range(iters):
        kz_w = torch.sqrt(k0**2 - k**2)
        kz_b = torch.sqrt(kb_complex**2 - k**2)
        s_w, c_w_ = torch.sin(kz_w*D), torch.cos(kz_w*D)
        
        F = rho_w * kz_b * s_w + 1j * rho_b * kz_w * c_w_
        
        dkzw_dk = -k/(kz_w + 1e-20)
        dkzb_dk = -k/(kz_b + 1e-20)
        dF = rho_w*(dkzb_dk*s_w + kz_b*c_w_*D*dkzw_dk) + 1j*rho_b*(dkzw_dk*c_w_ - kz_w*s_w*D*dkzw_dk)
        
        k = k - F / (dF + 1e-20)

    # Filtering
    valid_mask = (k.real > k_min) & (k.real < k0*1.01) & (k.imag > -1e-8)
    k = k[valid_mask]
    if len(k) > 0:
        k = k[torch.argsort(k.real, descending=True)]
        diff = torch.abs(k[1:] - k[:-1])
        k = k[torch.cat([torch.tensor([True], device=device), diff > 1e-4])]
            
    return k

def forward_model(c_b, rho_b, alpha_b, r_range, omega, freq):
    # Running in "Complex Mode" (Stage 2 style) for accuracy
    k_complex = unified_complex_solver(c_b, rho_b, alpha_b, omega, freq)
    
    if len(k_complex) == 0: 
        return torch.ones_like(r_range)*100.0

    k0 = omega / C_W_GLOBAL
    
    # Remove Ghost Artifacts
    is_ghost = torch.abs(k_complex.real - k0) < 2e-4
    k_complex = k_complex[~is_ghost]

    if len(k_complex) == 0:
        return torch.ones_like(r_range)*100.0

    # Field Calculation
    kz_w = torch.sqrt(k0**2 - k_complex**2)
    kb_complex = (omega/c_b) + 1j*((alpha_b * (freq/c_b))/8.686)
    kz_b = torch.sqrt(kb_complex**2 - k_complex**2)

    term1 = D_GLOBAL/2
    term2 = torch.sin(2*kz_w*D_GLOBAL)/(4*kz_w)
    term3 = (RHO_W_GLOBAL/rho_b) * (torch.sin(kz_w*D_GLOBAL)**2) / (2j * kz_b)
    A_m = 1.0 / torch.sqrt(term1 - term2 + term3)
    
    phase = k_complex[:, None] * r_range[None, :]
    H0 = torch.sqrt(2 / (math.pi * phase)) * torch.exp(1j * (phase - math.pi/4))
    
    Zs = A_m * torch.sin(kz_w * ZS_GLOBAL)
    Zr = A_m * torch.sin(kz_w * ZR_GLOBAL)
    p_sum = 1j/(4*RHO_W_GLOBAL) * (Zs[:, None] * Zr[:, None] * H0).sum(dim=0)
    
    tl_pred = -20 * torch.log10(p_sum.abs() * 4.0 * math.pi + 1e-12)
    return tl_pred

# =========================================================================
# 2. GENERATE GROUND TRUTH
# =========================================================================
print("--- GENERATING GROUND TRUTH FIELDS ---")
# Simulate range: 1km to 5km with 5m spacing (approx 800 points)
r_meas = torch.linspace(1000.0, 5000.0, 801, device=device)

TL_true_dict = {}
for f in FREQ_LIST:
    omega = 2 * math.pi * f
    cb_t = torch.tensor(TRUE_PARAMS['cb'], device=device)
    rho_t = torch.tensor(TRUE_PARAMS['rho'], device=device)
    alpha_t = torch.tensor(TRUE_PARAMS['alpha'], device=device)
    
    with torch.no_grad():
        tl = forward_model(cb_t, rho_t, alpha_t, r_meas, omega, f)
    TL_true_dict[f] = tl

# =========================================================================
# 3. COMPUTE LOSS LANDSCAPE
# =========================================================================
print("--- STARTING GRID SEARCH ---")

# Create Grid Axes
cb_vals = np.arange(cb_start, cb_end + cb_step, cb_step)
rho_vals = np.arange(rho_start, rho_end + rho_step, rho_step) # smaller step for smoother plot

loss_grid = np.zeros((len(rho_vals), len(cb_vals)))

# Fixed alpha for the search
alpha_fixed = torch.tensor(TRUE_PARAMS['alpha'], device=device)

with torch.no_grad():
    for i, rho_val in enumerate(rho_vals):
        print(f"Scanning Rho row {i+1}/{len(rho_vals)} ({rho_val:.2f})")
        for j, cb_val in enumerate(cb_vals):
            
            # Tensor-ize current grid point
            cb_curr = torch.tensor(float(cb_val), device=device)
            rho_curr = torch.tensor(float(rho_val), device=device)
            
            total_loss = 0.0
            
            # Sum MSE over all frequencies
            for f in FREQ_LIST:
                omega = 2 * math.pi * f
                pred = forward_model(cb_curr, rho_curr, alpha_fixed, r_meas, omega, f)
                target = TL_true_dict[f]
                
                # MSE Loss
                total_loss += torch.nn.functional.mse_loss(pred, target).item()
            
            loss_grid[i, j] = total_loss

# =========================================================================
# 4. VISUALIZATION (IMSHOW)
# =========================================================================
print("--- PLOTTING ---")

plt.figure(figsize=(10, 8))

# Extent defines the physical boundaries [left, right, bottom, top]
# We typically want cb on X and rho on Y
extent = [cb_vals.min(), cb_vals.max(), rho_vals.min(), rho_vals.max()]

# Use log scale for loss if the dynamic range is huge, otherwise standard is fine.
# Here we use standard linear color mapping for MSE.
im = plt.imshow(loss_grid, 
                origin='lower',       # (0,0) is bottom-left
                aspect='auto',        # Fit the square figure
                extent=extent,        # Map pixels to physical units
                cmap='viridis',       # 'viridis', 'turbo', or 'jet'
                interpolation='bilinear') # Smooths the pixels slightly

cbar = plt.colorbar(im)
cbar.set_label('Total MSE Loss (dB)')

# Mark the Ground Truth
plt.plot(TRUE_PARAMS['cb'], TRUE_PARAMS['rho'], 'r*', markersize=15, label='Ground Truth')

plt.title(f"Loss Landscape ($\alpha_b$ fixed at {TRUE_PARAMS['alpha']})")
plt.xlabel(r'Seabed Sound Speed $c_b$ (m/s)')
plt.ylabel(r'Seabed Density $\rho_b$ (g/cm$^3$)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.3)

plt.tight_layout()
plt.show()