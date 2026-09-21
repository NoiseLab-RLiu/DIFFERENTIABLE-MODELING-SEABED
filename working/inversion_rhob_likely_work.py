# -*- coding: utf-8 -*-
"""
Created on Sun Jan 18 21:26:39 2026

@author: 13391
"""

import math
import torch
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

# Safety: Clear GPU cache to prevent Spyder crashes
torch.cuda.empty_cache()

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ─────────────────────────── 1. SETUP ───────────────────────────────────
# Ground Truth (Target)
TRUE_CB    = 2000.0
TRUE_RHOB  = 1.8
TRUE_ALPHA = 0.5 

# Geometry
c_w    = 1500.0
rho_w  = 1.0
D      = 100.0
z_s    = 25.0
z_recv = 50.0
freq   = 150.0
omega  = 2 * math.pi * freq

# Load Data
mat_file = 'TL.mat' 
try:
    mat_data = sio.loadmat(mat_file)
    TL_matrix_np = mat_data[[k for k in mat_data.keys() if not k.startswith('__')][0]]
    TL_tensor = torch.tensor(TL_matrix_np, device=device)
    r_full = torch.linspace(0, 5000.0, TL_tensor.shape[1], device=device)
    
    # Mask 1km - 5km
    mask = r_full > 1000.0 
    r_meas = r_full[mask]
    TL_meas = TL_tensor[50, mask]
    print(f"Data Loaded: {len(r_meas)} points")
except:
    print("Using Synthetic Data for Demo")
    r_meas = torch.linspace(1000, 5000, 200, device=device)
    # Create dummy target if file missing (will be overwritten if file exists)
    TL_meas = torch.zeros_like(r_meas)

# ─────────────────────────── 2. ROBUST SOLVER ───────────────────────────
def unified_complex_solver(c_b, rho_b, alpha_b):
    k0 = omega / c_w
    kb_real = omega / c_b
    
    # Material Attenuation
    alpha_neper = (alpha_b * (freq / c_b)) / 8.686
    kb_complex = (omega / c_b) + 1j * alpha_neper
    
    # --- Initialization ---
    num_modes = 40 
    m = torch.arange(1, num_modes+1, device=device, dtype=torch.float64)
    kz_guess = (m - 0.5) * math.pi / D
    k_sq_guess = k0**2 - kz_guess**2
    
    is_propagating = k_sq_guess.real > 0
    k_guess = torch.zeros_like(k_sq_guess, dtype=torch.complex128)
    
    # Trapped
    k_guess[is_propagating] = torch.sqrt(k_sq_guess[is_propagating]).to(dtype=torch.complex128)
    
    # Leaky: Staggered Guesses
    cutoff_indices = torch.nonzero(~is_propagating).squeeze()
    if len(cutoff_indices) > 0:
        num_leaky = len(cutoff_indices)
        # Stagger from 0.99 down to 0.5 to catch distinct roots
        factors = torch.linspace(0.99, 0.5, num_leaky, device=device)
        k_guess[cutoff_indices] = torch.complex(
            kb_real * factors, 
            torch.tensor(0.005, device=device)
        )

    # --- Newton-Raphson ---
    k = k_guess.clone()
    for _ in range(30):
        kz_w = torch.sqrt(k0**2 - k**2)
        kz_b = torch.sqrt(kb_complex**2 - k**2)
        
        sin_w = torch.sin(kz_w * D)
        cos_w = torch.cos(kz_w * D)
        F = rho_w * kz_b * sin_w + 1j * rho_b * kz_w * cos_w
        
        # Analytic Derivative
        dkzw_dk = -k / kz_w
        dkzb_dk = -k / kz_b
        dT1 = rho_w * (dkzb_dk * sin_w + kz_b * cos_w * D * dkzw_dk)
        dT2 = 1j * rho_b * (dkzw_dk * cos_w - kz_w * sin_w * D * dkzw_dk)
        dF_dk = dT1 + dT2
        
        k = k - F / (dF_dk + 1e-20)

    # --- Filtering ---
    # Real > 0.2 (Physical), Imag > -1e-8 (Stable)
    valid_mask = (k.real > 0.2) & (k.imag > -1e-8) & (k.real < k0*1.01)
    k_valid = k[valid_mask]
    
    # Sort & Unique
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
    
    # If solver fails (empty), return high loss penalty
    if len(k_complex) == 0:
        return torch.ones_like(r_range) * 1000.0, k_complex

    k0 = omega / c_w
    kz_w = torch.sqrt(k0**2 - k_complex**2)
    alpha_neper = (alpha_b * (freq / c_b)) / 8.686
    kb_complex = (omega / c_b) + 1j * alpha_neper
    kz_b = torch.sqrt(kb_complex**2 - k_complex**2)
    
    # Mode Amplitudes
    denom = (D/2 - torch.sin(2*kz_w*D)/(4*kz_w) + 
             (rho_w/rho_b) * (torch.sin(kz_w*D)**2) / (2*kz_b * 1j))
    A_m = 1.0 / torch.sqrt(denom)
    
    # Pressure Field
    Zs = A_m * torch.sin(kz_w * z_s)
    Zr = A_m * torch.sin(kz_w * z_recv)
    
    phase_term = k_complex[:, None] * r_range[None, :]
    H0 = torch.sqrt(2 / (math.pi * phase_term)) * torch.exp(1j * (phase_term - math.pi/4))
    
    p_sum = 1j/(4*rho_w) * (Zs[:, None] * Zr[:, None] * H0).sum(dim=0)
    p_abs = torch.abs(p_sum * 4.0 * math.pi)
    tl = -20 * torch.log10(p_abs + 1e-12)
    
    return tl, k_complex

# ─────────────────────────── 3. INVERSION LOOP ──────────────────────────
print("\n=== STARTING INVERSION ===")

# INITIAL GUESSES
# We start c_b near the known solution (from Step 1)
# We start rho_b intentionally WRONG (1.2) to see if it climbs to 1.8
cb_opt = torch.tensor(1995.0, device=device, requires_grad=True) 
rb_opt = torch.tensor(1.5, device=device, requires_grad=True)
# We assume alpha is roughly known or small (can also optimize)
ab_fixed = torch.tensor(0.5, device=device)

# Optimizer
opt = torch.optim.Adam([cb_opt, rb_opt], lr=0.01)

loss_history = []
rho_history = []

for i in range(15100):
    opt.zero_grad()
    
    # Predict
    pred, _ = forward_model(cb_opt, rb_opt, ab_fixed, r_meas)
    
    # Loss: MSE (Matches Amplitude + Decay)
    loss = (pred - TL_meas).pow(2).mean()
    
    loss.backward()
    opt.step()
    
    # Constraints (Clamping)
    with torch.no_grad():
        cb_opt.clamp_(1900.0, 2100.0)
        rb_opt.clamp_(1.1, 3.0) # Physical density limits
    
    # Logging
    loss_val = loss.item()
    loss_history.append(loss_val)
    rho_history.append(rb_opt.item())
    
    if i % 10 == 0:
        print(f"Iter {i:03d}: Loss={loss_val:.4f} | c_b={cb_opt.item():.1f} | rho_b={rb_opt.item():.3f}")

print("\n=== FINAL RESULTS ===")
print(f"Recovered c_b  : {cb_opt.item():.2f} (True: {TRUE_CB})")
print(f"Recovered rho_b: {rb_opt.item():.3f} (True: {TRUE_RHOB})")

# ─────────────────────────── 4. SAFE PLOTTING ───────────────────────────
# Move everything to CPU first to avoid Kernel Death
r_plot = r_meas.detach().cpu().numpy()
tl_meas_plot = TL_meas.detach().cpu().numpy()
rho_hist_plot = np.array(rho_history)

# Final Prediction
with torch.no_grad():
    final_pred, _ = forward_model(cb_opt, rb_opt, ab_fixed, r_meas)
    tl_pred_plot = final_pred.detach().cpu().numpy()

plt.figure(figsize=(12, 5))

# Subplot 1: Field Match
plt.subplot(1, 2, 1)
plt.plot(r_plot, tl_meas_plot, 'k', label='Ground Truth', alpha=0.6)
plt.plot(r_plot, tl_pred_plot, 'r--', label='Recovered Model')
plt.title(f"Field Match (MSE={loss_history[-1]:.2f})")
plt.xlabel("Range (m)")
plt.ylabel("TL (dB)")
plt.legend()
plt.gca().invert_yaxis()
plt.grid(True, alpha=0.3)

# Subplot 2: Convergence
plt.subplot(1, 2, 2)
plt.plot(rho_hist_plot, 'b-')
plt.axhline(TRUE_RHOB, color='r', linestyle='--', label='True rho_b (1.8)')
plt.title("Density (rho_b) Convergence")
plt.xlabel("Iteration")
plt.ylabel("rho_b")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()