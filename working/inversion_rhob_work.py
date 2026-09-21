# -*- coding: utf-8 -*-
"""
Created on Sun Jan 18 21:40:16 2026

@author: 13391
"""

import math
import torch
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import gc
import copy

# 1. MEMORY SAFETY
gc.collect()
torch.cuda.empty_cache()
torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ─────────────────────────── 1. SETUP ───────────────────────────────────
TRUE_CB    = 2000.0
TRUE_RHOB  = 1.8
TRUE_ALPHA = 0.5 

c_w, rho_w, D = 1500.0, 1.0, 100.0
z_s, z_recv = 25.0, 50.0
freq = 150.0
omega = 2 * math.pi * freq

# Load Data
mat_file = 'TL.mat' 
try:
    mat_data = sio.loadmat(mat_file)
    TL_matrix_np = mat_data[[k for k in mat_data.keys() if not k.startswith('__')][0]]
    TL_tensor = torch.tensor(TL_matrix_np, device=device)
    r_full = torch.linspace(0, 5000.0, TL_tensor.shape[1], device=device)
    mask = r_full > 1000.0 
    r_meas = r_full[mask]
    TL_meas = TL_tensor[50, mask]
    print(f"Data Loaded: {len(r_meas)} points")
except:
    print("Using Synthetic Data")
    r_meas = torch.linspace(1000, 5000, 200, device=device)
    TL_meas = torch.zeros_like(r_meas)

# ─────────────────────────── 2. SOLVER ──────────────────────────────────
def unified_complex_solver(c_b, rho_b, alpha_b):
    k0 = omega / c_w
    kb_real = omega / c_b
    alpha_neper = (alpha_b * (freq / c_b)) / 8.686
    kb_complex = (omega / c_b) + 1j * alpha_neper
    
    num_modes = 40 
    m = torch.arange(1, num_modes+1, device=device, dtype=torch.float64)
    kz_guess = (m - 0.5) * math.pi / D
    k_sq_guess = k0**2 - kz_guess**2
    
    is_propagating = k_sq_guess.real > 0
    k_guess = torch.zeros_like(k_sq_guess, dtype=torch.complex128)
    k_guess[is_propagating] = torch.sqrt(k_sq_guess[is_propagating]).to(dtype=torch.complex128)
    
    cutoff_indices = torch.nonzero(~is_propagating).reshape(-1)
    if len(cutoff_indices) > 0:
        factors = torch.linspace(0.99, 0.5, len(cutoff_indices), device=device)
        k_guess[cutoff_indices] = torch.complex(kb_real * factors, torch.tensor(0.005, device=device))

    k = k_guess.clone()
    for _ in range(30):
        kz_w = torch.sqrt(k0**2 - k**2)
        kz_b = torch.sqrt(kb_complex**2 - k**2)
        
        sin_w = torch.sin(kz_w * D)
        cos_w = torch.cos(kz_w * D)
        F = rho_w * kz_b * sin_w + 1j * rho_b * kz_w * cos_w
        
        dkzw_dk = -k / (kz_w + 1e-30)
        dkzb_dk = -k / (kz_b + 1e-30)
        dT1 = rho_w * (dkzb_dk * sin_w + kz_b * cos_w * D * dkzw_dk)
        dT2 = 1j * rho_b * (dkzw_dk * cos_w - kz_w * sin_w * D * dkzw_dk)
        dF_dk = dT1 + dT2
        
        k = k - F / (dF_dk + 1e-20)

    valid_mask = (k.real > 0.2) & (k.imag > -1e-8) & (k.real < k0*1.01)
    k_valid = k[valid_mask]
    
    if len(k_valid) > 0:
        sorted_idx = torch.argsort(k_valid.real, descending=True)
        k_sorted = k_valid[sorted_idx]
        unique_mask = torch.ones_like(k_sorted, dtype=torch.bool)
        if len(k_sorted) > 1:
            diff = torch.abs(k_sorted[1:] - k_sorted[:-1])
            unique_mask[1:] = diff > 1e-4
        return k_sorted[unique_mask]
    return k_valid

def forward_model(c_b, rho_b, alpha_b, r_range):
    k_complex = unified_complex_solver(c_b, rho_b, alpha_b)
    
    # Stability penalty if solver fails
    if len(k_complex) == 0: 
        return torch.ones_like(r_range) * 100.0, k_complex

    k0 = omega / c_w
    kz_w = torch.sqrt(k0**2 - k_complex**2)
    alpha_neper = (alpha_b * (freq / c_b)) / 8.686
    kb_complex = (omega / c_b) + 1j * alpha_neper
    kz_b = torch.sqrt(kb_complex**2 - k_complex**2)
    
    denom = (D/2 - torch.sin(2*kz_w*D)/(4*kz_w) + 
             (rho_w/rho_b) * (torch.sin(kz_w*D)**2) / (2*kz_b * 1j))
    A_m = 1.0 / torch.sqrt(denom)
    
    Zs = A_m * torch.sin(kz_w * z_s)
    Zr = A_m * torch.sin(kz_w * z_recv)
    
    phase_term = k_complex[:, None] * r_range[None, :]
    H0 = torch.sqrt(2 / (math.pi * phase_term)) * torch.exp(1j * (phase_term - math.pi/4))
    
    p_sum = 1j/(4*rho_w) * (Zs[:, None] * Zr[:, None] * H0).sum(dim=0)
    tl = -20 * torch.log10(p_sum.abs() * 4.0 * math.pi + 1e-12)
    
    return tl, k_complex

# ─────────────────────────── 3. INVERSION WITH SAFEGUARDS ────────────────
print("\n=== STARTING INVERSION ===")

# Start where we know it works
cb_opt = torch.tensor(1995.0, device=device, requires_grad=True) 
rb_opt = torch.tensor(1.2, device=device, requires_grad=True)
ab_fixed = torch.tensor(0.5, device=device)

optimizer = torch.optim.Adam([cb_opt, rb_opt], lr=0.01)

# Variables to track "Best Model"
best_loss = float('inf')
best_params = {'c_b': 1995.0, 'rho_b': 1.5}
patience_counter = 0

loss_history = []
rho_history = []

for i in range(1501):
    optimizer.zero_grad()
    
    pred, _ = forward_model(cb_opt, rb_opt, ab_fixed, r_meas)
    loss = (pred - TL_meas).pow(2).mean()
    
    loss.backward()
    
    # 1. GRADIENT CLIPPING (Prevents the explosion!)
    torch.nn.utils.clip_grad_norm_([cb_opt, rb_opt], max_norm=1.0)
    
    optimizer.step()
    
    with torch.no_grad():
        cb_opt.clamp_(1900.0, 2100.0)
        rb_opt.clamp_(1.1, 3.0) 
    
    # 2. SAVE AS FLOATS (Prevents Memory Leak / Kernel Death)
    current_loss = loss.item()
    current_cb = cb_opt.item()
    current_rb = rb_opt.item()
    
    loss_history.append(current_loss)
    rho_history.append(current_rb)
    
    # 3. SAVE BEST MODEL
    if current_loss < best_loss:
        best_loss = current_loss
        best_params = {'c_b': current_cb, 'rho_b': current_rb}
        patience_counter = 0
    else:
        patience_counter += 1
        
    # Logging
    if i % 100 == 0:
        print(f"Iter {i:04d}: Loss={current_loss:.4f} | c_b={current_cb:.1f} | rho_b={current_rb:.3f}")
        
    # 4. EXPLOSION CHECK (Early Stopping)
    if current_loss > best_loss * 5.0 and i > 200:
        print(f"\n[!] Instability detected at Iter {i}. Stopping early.")
        break

print("\n=== FINAL RESULTS ===")
print(f"Best Loss: {best_loss:.4f}")
print(f"Recovered c_b  : {best_params['c_b']:.2f}")
print(f"Recovered rho_b: {best_params['rho_b']:.3f}")

# ─────────────────────────── 4. PLOT (SAFE MODE) ────────────────────────
r_plot = r_meas.cpu().numpy()
tl_meas_plot = TL_meas.cpu().numpy()

# Run prediction with BEST params
with torch.no_grad():
    best_cb_t = torch.tensor(best_params['c_b'], device=device)
    best_rb_t = torch.tensor(best_params['rho_b'], device=device)
    final_pred, _ = forward_model(best_cb_t, best_rb_t, ab_fixed, r_meas)
    tl_pred_plot = final_pred.cpu().numpy()

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(r_plot, tl_meas_plot, 'k', label='Ground Truth')
plt.plot(r_plot, tl_pred_plot, 'r--', label='Recovered')
plt.title(f"Best Fit (Loss={best_loss:.2f})")
plt.legend()
plt.gca().invert_yaxis()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(rho_history)
plt.title("Rho_b Convergence")
plt.xlabel("Iteration")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()