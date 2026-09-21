# -*- coding: utf-8 -*-
"""
Created on Mon Dec 22 18:06:31 2025

@author: l00905881
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Dec 22 11:31:01 2025

@author: l00905881
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import math, torch, os, scipy.io as sio, numpy as np
import matplotlib.pyplot as plt
import time

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ─────────────────────────── CONSTANTS ──────────────────────────────────
c_w    = 1500.0
D      = 100.0
rho_w  = 1.0
z_s    = 25.0
r_meas = torch.linspace(20., 180., 10, device=device)

# SCALES
CB_SCALE = 2000.0
RHOB_SCALE = 2.0
ALPHA_SCALE = 1.0
# ─────────────────────────── PHYSICS KERNEL ─────────────────────────────
class DispersionRoot(torch.autograd.Function):
    @staticmethod
    def _f(k, cb, rhob, omega):
        g   = torch.sqrt(torch.clamp((omega/c_w)**2 - k**2, min=1e-12))
        gb  = torch.sqrt(torch.clamp(k**2 - (omega/cb)**2, min=1e-12))
        f   = torch.tan(g*D) + rhob * (g/gb)
        return f, g, gb

    @staticmethod
    def forward(ctx, c_b, rho_b, k_guess, omega):
        # 1. Solve for Real Root k_r
        cb, rb, k = c_b.detach(), rho_b.detach(), k_guess.detach().clone()
        for _ in range(30): 
            f, g, gb = DispersionRoot._f(k, cb, rb, omega)
            sec2  = 1.0 / torch.cos(g*D)**2
            term1 = sec2 * (-k/g) * D
            term2 = rb * (-k/(g*gb) - g*k/(gb**3))
            df_dk = term1 + term2
            k = k - f / df_dk

        # 2. Compute Derivative dk/dcb for Sensitivity (Perturbation Physics)
        # This is the "Modal Filling Factor" equivalent to COA Eq 5.109
        f, g, gb = DispersionRoot._f(k, cb, rb, omega)
        sec2  = 1.0 / torch.cos(g*D)**2
        df_dk = (sec2 * (-k/g) * D) + rb * (-k/(g*gb) - g*k/(gb**3))
        
        # dF/dcb
        df_dcb = - (rb * g * omega**2) / (gb**3 * cb**3)
        
        # Sensitivity: dk/dcb
        dk_dcb = - df_dcb / df_dk

        ctx.save_for_backward(k, cb, rb, dk_dcb, df_dk)
        ctx.omega = omega
        
        # Return both k_real and sensitivity
        return k, dk_dcb

    @staticmethod
    def backward(ctx, grad_k, grad_dk_dcb):
        # Standard implicit gradients for backprop
        k, cb, rb, dk_dcb, df_dk = ctx.saved_tensors
        omega = ctx.omega
        f, g, gb = DispersionRoot._f(k, cb, rb, omega)
        
        df_drhob = g / gb
        dk_drhob = - df_drhob / df_dk
        
        return grad_k * dk_dcb, grad_k * dk_drhob, None, None

def get_modes_complex(c_b, rho_b, alpha_b, freq):
    omega = 2 * math.pi * freq
    # ... (Guess logic same as before) ...
    if freq < 60: num_modes = 4
    elif freq < 120: num_modes = 8
    else: num_modes = 12
        
    mode_ids = torch.arange(1, num_modes+1, device=c_b.device, dtype=torch.float64)
    gamma_guess = (mode_ids - 0.5) * math.pi / D
    k_sq = (omega/c_w)**2 - gamma_guess**2
    valid = k_sq > 0
    k_guess = torch.sqrt(k_sq[valid])
    
    # 1. Get Real Parts AND Sensitivity
    k_real, dk_dcb = DispersionRoot.apply(c_b, rho_b, k_guess, omega)
    
    # 2. Convert Alpha (dB/lambda) to Material Imaginary Speed (Delta_cb)
    # COA Eq 2.204: alpha_lambda approx 54.58 * (c_i/c_r)
    # So: c_i = c_r * alpha / 54.58
    c_i = (c_b * alpha_b) / 54.575
    
    # 3. Apply Perturbation Theory (COA Sec 5.7)
    # Modal Attenuation k_i = Sensitivity * Material_Loss
    k_imag = torch.abs(dk_dcb) * c_i
    
    # 4. Complex Wavenumber
    k_complex = torch.complex(k_real, k_imag)
    
    # ... (Amplitude calc same as before) ...
    k0 = omega/c_w
    gamma_m  = torch.sqrt(torch.clamp(k0**2 - k_real**2, min=1e-12))
    kb       = omega/c_b
    gamma_bm = torch.sqrt(torch.clamp(k_real**2 - kb**2, min=1e-12))
    denom = (D/2 - torch.sin(2*gamma_m*D)/(4*gamma_m) + (rho_w/rho_b) * (torch.sin(gamma_m*D)**2) / (2*gamma_bm))
    A_m = 1.0 / torch.sqrt(denom)
    
    return k_complex, A_m, gamma_m

def get_modes_complex2(c_b, rho_b, alpha_b, freq):
    omega = 2 * math.pi * freq
    # ... (Guess logic same as before) ...
    if freq < 60: num_modes = 2
    elif freq < 120: num_modes = 2
    else: num_modes = 2
        
    mode_ids = torch.arange(1, num_modes+1, device=c_b.device, dtype=torch.float64)
    gamma_guess = (mode_ids - 0.5) * math.pi / D
    k_sq = (omega/c_w)**2 - gamma_guess**2
    valid = k_sq > 0
    k_guess = torch.sqrt(k_sq[valid])
    
    # 1. Get Real Parts AND Sensitivity
    k_real, dk_dcb = DispersionRoot.apply(c_b, rho_b, k_guess, omega)
    
    # 2. Convert Alpha (dB/lambda) to Material Imaginary Speed (Delta_cb)
    # COA Eq 2.204: alpha_lambda approx 54.58 * (c_i/c_r)
    # So: c_i = c_r * alpha / 54.58
    c_i = (c_b * alpha_b) / 54.575
    
    # 3. Apply Perturbation Theory (COA Sec 5.7)
    # Modal Attenuation k_i = Sensitivity * Material_Loss
    k_imag = torch.abs(dk_dcb) * c_i
    
    # 4. Complex Wavenumber
    k_complex = torch.complex(k_real, k_imag)
    
    # ... (Amplitude calc same as before) ...
    k0 = omega/c_w
    gamma_m  = torch.sqrt(torch.clamp(k0**2 - k_real**2, min=1e-12))
    kb       = omega/c_b
    gamma_bm = torch.sqrt(torch.clamp(k_real**2 - kb**2, min=1e-12))
    denom = (D/2 - torch.sin(2*gamma_m*D)/(4*gamma_m) + (rho_w/rho_b) * (torch.sin(gamma_m*D)**2) / (2*gamma_bm))
    A_m = 1.0 / torch.sqrt(denom)
    
    return k_complex, A_m, gamma_m

def pressure_field(c_b, rho_b, alpha_b, r, freq):
    # This function remains mostly the same, but acts as the bridge
    km_complex, A_m, gamma_m = get_modes_complex(c_b, rho_b, alpha_b, freq)
    
    k_real = km_complex.real
    k_imag = km_complex.imag
    
    Zs = A_m * torch.sin(gamma_m * z_s)
    Z  = A_m * torch.sin(gamma_m * D)
    
    # Compute Propagation
    arg_real = k_real[:, None] * r  
    H0_real_arg = torch.special.bessel_j0(arg_real) + 1j * torch.special.bessel_y0(arg_real)
    
    # Apply Decay (Gradient flows from alpha -> k_imag -> decay -> p)
    decay = torch.exp(-k_imag[:, None] * r)
    H0_complex = H0_real_arg * decay
    
    p = 1j/(4*rho_w) * (Zs[:, None] * Z[:, None] * H0_complex).sum(dim=0)
    return p

# ─────────────────────────── SETUP ──────────────────────────────────
# True Params
c_b_true   = torch.tensor(2000.0, device=device)
rho_b_true = torch.tensor(1.8, device=device)
alpha_true = torch.tensor(0.5, device=device) # 0.5 dB/lambda

# Generate Data
all_freqs_data = [30.0, 35.0, 50.0, 100.0, 150.0]
p_data_map = {}
for f in all_freqs_data:
    p_data_map[f] = pressure_field(c_b_true, rho_b_true, alpha_true, r_meas, f).detach()

# ─────────────────────────── TRAINING ───────────────────────────────
cb_norm    = torch.tensor(1950.0/CB_SCALE, device=device, requires_grad=True)
rb_norm    = torch.tensor(1.5/RHOB_SCALE, device=device, requires_grad=True)
# Start alpha at 0.0 or a small guess
al_norm    = torch.tensor(0.1/ALPHA_SCALE, device=device, requires_grad=True)

opt = torch.optim.Adam([
    {'params': cb_norm, 'lr': 1e-3}, 
    {'params': rb_norm, 'lr': 1e-3},
    {'params': al_norm, 'lr': 0.0}   # FROZEN INITIALLY!
], eps=1e-16)

print(f"Start: c_b={cb_norm.item()*CB_SCALE:.1f}, "
      f"rho_b={rb_norm.item()*RHOB_SCALE:.3f}, "
      f"alpha={al_norm.item()*ALPHA_SCALE:.3f}")


cb_hist = []
rhob_hist = []
alpha_hist = []
start_time = time.time()
# STAGE 1: Recover c_b and rho_b (Alpha Frozen)
print("=== STAGE 1: Geometry Recovery (Alpha Frozen) ===")
stage1_freqs = [50.0, 100.0, 150.0]
for epoch in range(2001):
    opt.zero_grad()
    c_b = cb_norm * CB_SCALE
    rho_b = rb_norm * RHOB_SCALE
    alpha_b = al_norm * ALPHA_SCALE # Will stay constant
    
    loss = 0
    for f in stage1_freqs: # Use Low Freq first
        p_pred = pressure_field(c_b, rho_b, alpha_b, r_meas, f)
        loss += (p_pred - p_data_map[f]).abs().pow(2).mean()
    loss.backward()
    opt.step()
    
    with torch.no_grad():
        cb_norm.clamp_(1500.1/CB_SCALE, 3000./CB_SCALE)
        rb_norm.clamp_(1.1/RHOB_SCALE, 3.0/RHOB_SCALE)
        al_norm.clamp_(0.0, 5.0)

    if epoch % 2 == 0:
        cb_hist.append(c_b.item())
        rhob_hist.append(rho_b.item())
        alpha_hist.append(alpha_b.item())
        if epoch % 500 == 0:
            print(f"Ep {epoch}: L={loss.item():.2e} | "
                  f"c_b={c_b.item():.1f} | rho_b={rho_b.item():.3f} | alpha={alpha_b.item():.3f}")

# STAGE 2: Unfreeze Alpha + Multi-Freq
print("\n=== STAGE 2: Full Recovery (Alpha Unfrozen) ===")
# Set learning rate for alpha
opt.param_groups[2]['lr'] = 1e-1 
stage2_freqs = [30.0, 35.0, 100.0, 150.0]
for epoch in range(330001):
    if epoch==100:
        stop_time1 = time.time()
        print('stop_time1')
        print(stop_time1-start_time)
    if epoch==150:
        stop_time2 = time.time()
        print('stop_time2')
        print(stop_time2-start_time)
        break
    opt.zero_grad()
    c_b = cb_norm * CB_SCALE
    rho_b = rb_norm * RHOB_SCALE
    alpha_b = al_norm * ALPHA_SCALE
    
    loss = 0
    for f in stage2_freqs:
        p_pred = pressure_field(c_b, rho_b, alpha_b, r_meas, f)
        loss += (p_pred - p_data_map[f]).abs().pow(2).mean()
    loss.backward()
    opt.step()
    
    with torch.no_grad():
        cb_norm.clamp_(1500.1/CB_SCALE, 3000./CB_SCALE)
        rb_norm.clamp_(1.1/RHOB_SCALE, 3.0/RHOB_SCALE)
        al_norm.clamp_(0.0, 5.0)

    if epoch % 2 == 0:
        cb_hist.append(c_b.item())
        rhob_hist.append(rho_b.item())
        alpha_hist.append(alpha_b.item())
        if epoch % 50 == 0:
            print(f"DEBUG GRAD: alpha={al_norm.grad.item():.2e}, cb={cb_norm.grad.item():.2e}")
            print(f"Ep {epoch}: L={loss.item():.2e} | "
                  f"c_b={c_b.item():.1f} | rho_b={rho_b.item():.3f} | alpha={alpha_b.item():.3f}")

#%%
import torch
# ─────────────────────────── VISUALIZATION ──────────────────────────────
def pressure_field_highreso(c_b, rho_b, alpha_b, r, freq, z_r=50.0): 
    km_complex, A_m, gamma_m = get_modes_complex(c_b, rho_b, alpha_b, freq)
    
    k_real = km_complex.real
    k_imag = km_complex.imag
    
    # Source is at z_s (25m), Receiver is now at z_r (e.g. 50m)
    Zs = A_m * torch.sin(gamma_m * z_s)
    Z  = A_m * torch.sin(gamma_m * z_r) # <--- CHANGED FROM D TO z_r
    
    # Compute Propagation
    arg_real = k_real[:, None] * r  
    H0_real_arg = torch.special.bessel_j0(arg_real) + 1j * torch.special.bessel_y0(arg_real)
    
    decay = torch.exp(-k_imag[:, None] * r)
    H0_complex = H0_real_arg * decay
    
    p = 1j/(4*rho_w) * (Zs[:, None] * Z[:, None] * H0_complex).sum(dim=0)
    return p

def pressure_field_highreso2(c_b, rho_b, alpha_b, r, freq, z_r=50.0): 
    km_complex, A_m, gamma_m = get_modes_complex2(c_b, rho_b, alpha_b, freq)
    
    k_real = km_complex.real
    k_imag = km_complex.imag
    
    # Source is at z_s (25m), Receiver is now at z_r (e.g. 50m)
    Zs = A_m * torch.sin(gamma_m * z_s)
    Z  = A_m * torch.sin(gamma_m * z_r) # <--- CHANGED FROM D TO z_r
    
    # Compute Propagation
    arg_real = k_real[:, None] * r  
    H0_real_arg = torch.special.bessel_j0(arg_real) + 1j * torch.special.bessel_y0(arg_real)
    
    decay = torch.exp(-k_imag[:, None] * r)
    H0_complex = H0_real_arg * decay
    
    p = 1j/(4*rho_w) * (Zs[:, None] * Z[:, None] * H0_complex).sum(dim=0)
    return p

print("\n=== PLOTTING RESULTS ===")

# 1. Recover Final Estimated Parameters
# 1. Use DENSE range for plotting (Physics needs ~5 points per wavelength)
# 100 Hz -> lambda = 15m. We want step < 3m. 
# 160m span / 200 points = 0.8m step. Perfect.
r_dense = torch.linspace(20., 15800., 15000, device=device) 

# 2. Define Receiver Depth (e.g., middle of water column)
z_receiver = 50.0 

plt.figure(figsize=(10, 12))

freqs_to_plot = [30, 35, 50, 100, 150]
plt.subplot(2,1,1)
for f in freqs_to_plot:
    # Pass z_r explicitly
    p_complex = pressure_field_highreso(c_b_true, rho_b_true, alpha_true, r_dense, f, z_r=z_receiver).detach().cpu().numpy()
    
    tl_db = 20 * np.log10(np.abs(p_complex) + 1e-12)
    
    r_numpy = r_dense.cpu().numpy()
    plt.plot(r_numpy, tl_db, '-', linewidth=1.5, label=f'{f} Hz')

plt.title(f"Transmission loss (receiver depth = {z_receiver}m)", fontsize=14) #2-Mode Approximation of Transmission Loss
plt.xlabel("Range (m)", fontsize=14)
plt.ylabel("Transmission Loss (dB)", fontsize=14)
plt.grid(True, which='both', alpha=0.3)
plt.ylim([-100, -50])
plt.legend()
plt.tight_layout()

plt.subplot(2,1,2)
for f in freqs_to_plot:
    # Pass z_r explicitly
    p_complex = pressure_field_highreso2(c_b_true, rho_b_true, alpha_true, r_dense, f, z_r=z_receiver).detach().cpu().numpy()
    
    tl_db = 20 * np.log10(np.abs(p_complex) + 1e-12)
    
    r_numpy = r_dense.cpu().numpy()
    plt.plot(r_numpy, tl_db, '-', linewidth=1.5, label=f'{f} Hz')

plt.title(f"2-Mode Approximation of Transmission Loss", fontsize=14) #2-Mode Approximation of Transmission Loss
plt.xlabel("Range (m)", fontsize=14)
plt.ylabel("Transmission Loss (dB)", fontsize=14)
plt.grid(True, which='both', alpha=0.3)
plt.ylim([-100, -50])
plt.legend()
plt.tight_layout()
plt.show()

#%%
# 1. Define the 2D Grid
# We need high resolution in depth to see mode shapes, and range for interference
n_r = 2500#2500#500  # Number of range steps
n_z = 200  # Number of depth steps
max_range=5000#1000
max_depth=D
freq=150.0
c_b=c_b_true
rho_b=rho_b_true
alpha_b=alpha_true

# 2. GRID: High resolution
# Start range at 1.0 meter (not 0) to avoid singularity, but close enough to see source
r_vec = torch.linspace(1.0, max_range, n_r, device=device) 
z_vec = torch.linspace(0.0, max_depth, 400, device=device)

# 3. COMPUTE MODES (Standard)
# Ensure you use the 'get_modes_complex' with the DYNAMIC mode count
k_complex, A_m, gamma_m = get_modes_complex(c_b, rho_b, alpha_b, freq)

k_real = k_complex.real
k_imag = k_complex.imag

# 4. FIELD CALCULATION
# Mode shapes in Depth: [num_modes, n_z]
phi_z = A_m[:, None] * torch.sin(gamma_m[:, None] * z_vec[None, :])

# Source Excitation at z_s: [num_modes]
phi_source = A_m * torch.sin(gamma_m * z_s)

# Range Propagation (Hankel): [num_modes, n_r]
# Use standard Hankel approx for speed: sqrt(2/pi*k*r) * exp(i(kr - pi/4))
# Note: We keep the complex phase! The "Petals" come from Phase Interference.
arg_real = k_real[:, None] * r_vec[None, :]
decay    = torch.exp(-k_imag[:, None] * r_vec[None, :])

# Full Bessel/Hankel function
H0 = (torch.special.bessel_j0(arg_real) + 1j * torch.special.bessel_y0(arg_real)) * decay

# Matrix Multiplication: Sum_m [ phi(z) * phi(z_s) * H0(r) ]
# (n_z, n_modes) @ (n_modes, n_r) -> (n_z, n_r)
mode_term = phi_source[:, None] * H0
p_field = torch.matmul(phi_z.T.to(torch.complex128), mode_term)

p_field *=  math.pi
# 5. CONVERT TO TL (dB)
p_abs = p_field.abs().cpu().numpy()
TL = -20 * np.log10(p_abs + 1e-12)

# 6. VISUALIZATION SCALING
# Center the colorscale on the "average" decay to highlight peaks/nulls
# Source is ~ 0 to -10 dB. Far field is -60 dB.
# We clip the "Infinite" source peak to -30 so it doesn't wash out the petals
vmin = 40#-70
vmax = 80#-30 

# 7. PLOT
plt.figure(figsize=(12, 6))

R_np = r_vec.cpu().numpy()
Z_np = z_vec.cpu().numpy()

# 'jet' is the classic "Rainbow" map used in textbooks
# 'gouraud' shading smooths the pixels
plt.pcolormesh(R_np, Z_np, TL, shading='gouraud', cmap='jet', vmin=vmin, vmax=vmax)

# Mark Source
plt.plot(0, z_s, 'w*', markersize=15, markeredgecolor='k', label='Source')
plt.legend(loc='upper right', fontsize=14)
cbar = plt.colorbar()
# Set the label with specific font size
cbar.set_label('TL (dB)', fontsize=14) 
# Set the size of the tick numbers (the numbers along the bar)
cbar.ax.tick_params(labelsize=12)

plt.gca().invert_yaxis() # Depth 0 at top

plt.title(f"TL contours for {freq} Hz", fontsize=14)
plt.xlabel("Range (m)", fontsize=14)
plt.ylabel("Depth (m)", fontsize=14)

# Force Aspect Ratio to be somewhat realistic (not 1:1, but not 1:1000)
# This helps you see the "Beams" instead of vertical stripes
plt.gca().set_aspect('auto') 

plt.tight_layout()
plt.show()

#%%
# --- Dummy Data Generation (Replace this with your actual data) ---
n_points = 1076
# Create x-axis values: each index corresponds to 2 epochs
x_epochs = np.arange(n_points) * 2 

# Simulating data
cb_true = 2000
rhob_true = 1.8
alpha_true = 0.5

# Replace these with your actual lists
# cb_hist = cb_true + (np.random.rand(n_points) - 0.5) * 100 + 500 * np.exp(-np.linspace(0, 10, n_points))
# rhob_hist = rhob_true + (np.random.rand(n_points) - 0.5) * 0.1 + 0.5 * np.exp(-np.linspace(0, 10, n_points))
# alpha_hist = alpha_true + (np.random.rand(n_points) - 0.5) * 0.05 + 0.2 * np.exp(-np.linspace(0, 10, n_points))
# ---------------------------------------------------------------

# --- Plotting Configuration ---
# Manual range for x-axis (in epochs)
x_min = 0
x_max = n_points * 2
# -----------------------------

fig, axs = plt.subplots(3, 1)#, figsize=(14, 10))
#fig.suptitle('Optimization Trajectory', fontsize=16)

# Plot 1: cb_hist
axs[0].plot(x_epochs, cb_hist, label='Estimated $c_b$')
axs[0].axhline(y=cb_true, color='k', linestyle='--', linewidth=1.5, label=f'True $c_b$={cb_true}')
axs[0].set_title('$c_b$ History',fontsize=14)
axs[0].set_xlabel('Epochs', fontsize=14)
axs[0].set_ylabel('Value', fontsize=14)
axs[0].set_xlim(x_min, x_max)
axs[0].set_ylim(1940, 2070)
axs[0].legend(fontsize=14)
axs[0].grid(True, alpha=0.3)

# Plot 2: rhob_hist
axs[1].plot(x_epochs, rhob_hist, label=r'Estimated $\rho_b$', color='orange')
axs[1].axhline(y=rhob_true, color='k', linestyle='--', linewidth=1.5, label=f'True $\\rho_b$={rhob_true}')
axs[1].set_title(r'$\rho_b$ History',fontsize=14)
axs[1].set_xlabel('Epochs', fontsize=14)
axs[1].set_ylabel('Value', fontsize=14)
axs[1].set_xlim(x_min, x_max)
axs[1].legend(fontsize=14)
axs[1].grid(True, alpha=0.3)

# Plot 3: alpha_hist
axs[2].plot(x_epochs, alpha_hist, label=r'Estimated $\alpha$', color='green')
axs[2].axhline(y=alpha_true, color='k', linestyle='--', linewidth=1.5, label=f'True $\\alpha$={alpha_true}')
axs[2].set_title(r'$\alpha$ History',fontsize=14)
axs[2].set_xlabel('Epochs', fontsize=14)
axs[2].set_ylabel('Value', fontsize=14)
axs[2].set_xlim(1990, 2150)
axs[2].legend(fontsize=14)
axs[2].grid(True, alpha=0.3)

# Remove the empty 4th subplot
#fig.delaxes(axs[1, 1])
#plt.tight_layout()
plt.tight_layout()
plt.show()
#%%
import matplotlib.pyplot as plt

n_points = 1076
# Create x-axis values: each index corresponds to 2 epochs
x_epochs = np.arange(n_points) * 2 

# Simulating data
cb_true = 2000
rhob_true = 1.8
alpha_true = 0.5

# Replace these with your actual lists
# cb_hist = cb_true + (np.random.rand(n_points) - 0.5) * 100 + 500 * np.exp(-np.linspace(0, 10, n_points))
# rhob_hist = rhob_true + (np.random.rand(n_points) - 0.5) * 0.1 + 0.5 * np.exp(-np.linspace(0, 10, n_points))
# alpha_hist = alpha_true + (np.random.rand(n_points) - 0.5) * 0.05 + 0.2 * np.exp(-np.linspace(0, 10, n_points))
# ---------------------------------------------------------------

# --- Plotting Configuration ---
# Manual range for x-axis (in epochs)
x_min = 0
x_max = n_points * 2
# -----------------------------

# Increase figsize height to fix the "flat" (扁) look
fig, axs = plt.subplots(3, 1, figsize=(10, 12)) 

# Global font size settings for consistency
label_fs = 14
title_fs = 16
legend_fs = 14
tick_fs = 12

# -----------------------------
# Plot 1: cb_hist
# Note: \mathrm{b} makes the subscript non-italic
axs[0].plot(x_epochs, cb_hist, label=r'Estimated $\hat{c}_{\mathrm{b}}$')
axs[0].axhline(y=cb_true, color='k', linestyle='--', linewidth=1.5, 
               label=fr'True $c_{{\mathrm{{b}}}}$={cb_true} m/s') # Added Unit

axs[0].set_title(r'$\hat{c}_{\mathrm{b}}$ History', fontsize=title_fs)
axs[0].set_xlabel('Epochs', fontsize=label_fs)
axs[0].set_ylabel('Value', fontsize=label_fs)
axs[0].set_xlim(x_min, x_max)
axs[0].set_ylim(1940, 2070)

# Custom Ticks: 0, 1000, 2000 only
axs[0].set_xticks([0, 1000, 2000])
axs[0].tick_params(axis='both', which='major', labelsize=tick_fs)

axs[0].legend(fontsize=legend_fs)
axs[0].grid(True, alpha=0.3)

# -----------------------------
# Plot 2: rhob_hist
axs[1].plot(x_epochs, rhob_hist, label=r'Estimated $\hat{\rho}_{\mathrm{b}}$', color='orange')
axs[1].axhline(y=rhob_true, color='k', linestyle='--', linewidth=1.5, 
               label=fr'True $\rho_{{\mathrm{{b}}}}$={rhob_true} g/cm$^3$') # Added Unit

axs[1].set_title(r'$\hat{\rho}_{\mathrm{b}}$ History', fontsize=title_fs)
axs[1].set_xlabel('Epochs', fontsize=label_fs)
axs[1].set_ylabel('Value', fontsize=label_fs)
axs[1].set_xlim(x_min, x_max)

# Custom Ticks: 0, 1000, 2000 only
axs[1].set_xticks([0, 1000, 2000])
axs[1].tick_params(axis='both', which='major', labelsize=tick_fs)

axs[1].legend(fontsize=legend_fs)
axs[1].grid(True, alpha=0.3)

# -----------------------------
# Plot 3: alpha_hist
axs[2].plot(x_epochs, alpha_hist, label=r'Estimated $\hat{\alpha}$', color='green')
axs[2].axhline(y=alpha_true, color='k', linestyle='--', linewidth=1.5, 
               label=fr'True $\alpha$={alpha_true} dB/$\lambda$') # Added Unit

axs[2].set_title(r'$\hat{\alpha}$ History', fontsize=title_fs)
axs[2].set_xlabel('Epochs', fontsize=label_fs)
axs[2].set_ylabel('Value', fontsize=label_fs)
# Your original code had this limit, ensuring the ticks below are visible
axs[2].set_xlim(1990, 2150) 

# Custom Ticks: 2000, 2050, 2100, 2150
axs[2].set_xticks([2000, 2050, 2100, 2150])
axs[2].tick_params(axis='both', which='major', labelsize=tick_fs)

axs[2].legend(fontsize=legend_fs)
axs[2].grid(True, alpha=0.3)

# -----------------------------
plt.tight_layout()
plt.show()
#%%
# --- Dummy Data Generation (Replace this with your actual data) ---
n_points = 1105
# Create x-axis values: each index corresponds to 2 epochs
x_epochs = np.arange(n_points) * 2 

# Simulating data
cb_true = 2000
rhob_true = 1.8
alpha_true = 0.5

# Replace these with your actual lists
# (Dummy data for visualization purposes)
# ---------------------------------------------------------------

# --- Plotting Configuration ---
# Manual range for x-axis (in epochs)
x_min = 0
x_max = n_points * 2
# -----------------------------

fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# Settings for font sizes
label_fs = 16   # Axis labels
title_fs = 16   # Titles
tick_fs = 14    # Tick numbers (Requested to be larger)
legend_fs = 14  # Legend text

# Plot 1: cb_hist
# Note: \mathrm{b} makes the subscript non-italic
axs[0, 0].plot(x_epochs, cb_hist, label=r'Estimated $\hat{c}_{\mathrm{b}}$')
axs[0, 0].axhline(y=cb_true, color='k', linestyle='--', linewidth=1.5, label=r'True $c_{\mathrm{b}}$=' + f'{cb_true}')
axs[0, 0].set_title(r'$\hat{c}_{\mathrm{b}}$ History', fontsize=title_fs)
axs[0, 0].set_xlabel('Epochs', fontsize=label_fs)
axs[0, 0].set_ylabel('Value', fontsize=label_fs)
axs[0, 0].set_xlim(x_min, x_max)
axs[0, 0].set_ylim(1940, 2070)
axs[0, 0].legend(fontsize=legend_fs)
axs[0, 0].grid(True, alpha=0.3)
# Increase tick label size
axs[0, 0].tick_params(axis='both', which='major', labelsize=tick_fs)

# Plot 2: rhob_hist
axs[0, 1].plot(x_epochs, rhob_hist, label=r'Estimated $\hat{\rho}_{\mathrm{b}}$', color='orange')
axs[0, 1].axhline(y=rhob_true, color='k', linestyle='--', linewidth=1.5, label=r'True $\rho_{\mathrm{b}}$=' + f'{rhob_true}')
axs[0, 1].set_title(r'$\hat{\rho}_{\mathrm{b}}$ History', fontsize=title_fs)
axs[0, 1].set_xlabel('Epochs', fontsize=label_fs)
axs[0, 1].set_ylabel('Value', fontsize=label_fs)
axs[0, 1].set_xlim(x_min, x_max)
axs[0, 1].legend(fontsize=legend_fs)
axs[0, 1].grid(True, alpha=0.3)
# Increase tick label size
axs[0, 1].tick_params(axis='both', which='major', labelsize=tick_fs)

# Plot 3: alpha_hist
axs[1, 0].plot(x_epochs, alpha_hist, label=r'Estimated $\hat{\alpha}_{\mathrm{b}}$', color='green')
axs[1, 0].axhline(y=alpha_true, color='k', linestyle='--', linewidth=1.5, label=r'True $\alpha_{\mathrm{b}}$=' + f'{alpha_true}')
axs[1, 0].set_title(r'$\hat{\alpha}_{\mathrm{b}}$ History', fontsize=title_fs)
axs[1, 0].set_xlabel('Epochs', fontsize=label_fs)
axs[1, 0].set_ylabel('Value', fontsize=label_fs)
axs[1, 0].set_xlim(1990, 2150) # Preserving your specific zoom for alpha
axs[1, 0].legend(fontsize=legend_fs)
axs[1, 0].grid(True, alpha=0.3)
# Increase tick label size
axs[1, 0].tick_params(axis='both', which='major', labelsize=tick_fs)

# Remove the empty 4th subplot
fig.delaxes(axs[1, 1])

plt.tight_layout()
plt.show()

#%%
import math
import torch
import scipy.io as sio
import numpy as np

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ─────────────────────────── CONSTANTS ──────────────────────────────────
c_w    = 1500.0
D      = 100.0
rho_w  = 1.0
z_s    = 25.0

# ─────────────────────────── DATA LOADING ───────────────────────────────
mat_file = 'TL.mat' 
try:
    mat_data = sio.loadmat(mat_file)
    keys = [k for k in mat_data.keys() if not k.startswith('__')]
    TL_matrix = mat_data[keys[0]] 
except FileNotFoundError:
    print(f"Error: {mat_file} not found.")
    exit()

max_range_bellhop = 5000.0
r_full = torch.linspace(0, max_range_bellhop, TL_matrix.shape[1], device=device)
valid_range_mask = r_full > 2000.0
r_meas = r_full[valid_range_mask]
meas_indices = torch.where(valid_range_mask)[0]
TL_measured_db = torch.tensor(TL_matrix[50, meas_indices], device=device)

# ─────────────────────────── RAY-MODE KERNEL ──────────────────────────
class DispersionRoot(torch.autograd.Function):
    @staticmethod
    def _f(k, cb, rhob, omega):
        # Standard Real Dispersion
        g   = torch.sqrt(torch.clamp((omega/c_w)**2 - k**2, min=1e-12))
        gb  = torch.sqrt(torch.clamp(k**2 - (omega/cb)**2, min=1e-12))
        f   = torch.tan(g*D) + rhob * (g/gb)
        return f, g, gb
    
    @staticmethod
    def forward(ctx, c_b, rho_b, k_guess, omega):
        cb, rb, k = c_b.detach(), rho_b.detach(), k_guess.detach().clone()
        for _ in range(30): 
            f, g, gb = DispersionRoot._f(k, cb, rb, omega)
            sec2  = 1.0 / torch.cos(g*D)**2
            term1 = sec2 * (-k/g) * D
            term2 = rb * (-k/(g*gb) - g*k/(gb**3))
            df_dk = term1 + term2
            k = k - f / df_dk
            
        f, g, gb = DispersionRoot._f(k, cb, rb, omega)
        sec2  = 1.0 / torch.cos(g*D)**2
        df_dk = (sec2 * (-k/g) * D) + rb * (-k/(g*gb) - g*k/(gb**3))
        df_dcb = - (rb * g * omega**2) / (gb**3 * cb**3)
        dk_dcb = - df_dcb / df_dk
        ctx.save_for_backward(k, cb, rb, dk_dcb, df_dk)
        return k, dk_dcb
    
    @staticmethod
    def backward(ctx, grad_k, grad_dk_dcb):
        k, cb, rb, dk_dcb, df_dk = ctx.saved_tensors
        grad_cb = grad_k * dk_dcb
        return grad_cb, None, None, None # Simply return gradients

def get_modes_ray_analogy(c_b, rho_b, alpha_b, freq):
    omega = 2 * math.pi * freq
    
    # 1. Real Roots (Standard)
    num_modes = 20
    mode_ids = torch.arange(1, num_modes+1, device=c_b.device, dtype=torch.float64)
    gamma_guess = (mode_ids - 0.5) * math.pi / D
    k_sq = (omega/c_w)**2 - gamma_guess**2
    valid = k_sq > 0
    k_guess = torch.sqrt(k_sq[valid])
    
    k_real, _ = DispersionRoot.apply(c_b, rho_b, k_guess, omega)
    
    # 2. Perturbation Attenuation (Friction)
    # We keep this for alpha (material loss)
    c_i = (c_b * alpha_b) / 54.575
    # Simplistic sensitivity approx
    k_imag_material = alpha_b * 1e-4 # Placeholder, keeping it small
    
    # 3. LEAKAGE ATTENUATION (The Fix)
    # Calculate grazing angle theta
    k0 = omega / c_w
    theta = torch.acos(k_real / k0) # Grazing angle in radians
    
    # Plane Wave Reflection Coefficient R(theta)
    # R = (rho_b*k_zw - rho_w*k_zb) / (rho_b*k_zw + rho_w*k_zb)
    k_zw = k0 * torch.sin(theta)
    
    # Complex bottom vertical wavenumber
    # k_b = omega/c_b + i*alpha...
    kb_real = omega / c_b
    # Add alpha to bottom wavenumber to allow absorption there
    kb = kb_real + 1j * (alpha_b / 8.686) 
    
    # k_zb = sqrt(kb^2 - k_real^2)
    k_zb = torch.sqrt(kb**2 - k_real**2 + 0j)
    
    R_top = rho_b * k_zw - rho_w * k_zb
    R_bot = rho_b * k_zw + rho_w * k_zb
    R_coef = R_top / R_bot
    
    # Reflection Loss (Neper/m) = -ln(|R|) / Cycle_Distance
    # Cycle Distance L = 2 * D / tan(theta)
    L_cycle = 2 * D / torch.tan(theta)
    
    # Leakage decay rate
    k_imag_leakage = -torch.log(torch.abs(R_coef) + 1e-12) / L_cycle
    
    # Total imaginary part
    k_imag_total = k_imag_material + k_imag_leakage
    
    k_complex = torch.complex(k_real, k_imag_total)
    
    # Amplitudes
    gamma_m = k_zw
    denom = (D/2 - torch.sin(2*gamma_m*D)/(4*gamma_m)) # Simple norm
    A_m = 1.0 / torch.sqrt(denom)
    
    return k_complex, A_m, gamma_m

def pressure_field(c_b, rho_b, alpha_b, gain_db, r, freq):
    k_complex, A_m, gamma_m = get_modes_ray_analogy(c_b, rho_b, alpha_b, freq)
    
    k_real = k_complex.real
    k_imag = k_complex.imag
    
    Zs = A_m * torch.sin(gamma_m * z_s)
    Zr = A_m * torch.sin(gamma_m * 50.0)
    
    arg_real = k_real[:, None] * r[None, :]
    decay = torch.exp(-k_imag[:, None] * r[None, :])
    
    H0 = torch.sqrt(2 / (math.pi * arg_real)) * torch.exp(1j * (arg_real - math.pi/4)) * decay
    
    p_sum = 1j/(4*rho_w) * (Zs[:, None] * Zr[:, None] * H0).sum(dim=0)
    p_sum = p_sum * 4.0 * math.pi
    
    return -20 * torch.log10(p_sum.abs() + 1e-12) + gain_db

# ─────────────────────────── RUN OPTIMIZATION ───────────────────────────
# Use Best guess from before
cb_norm = torch.tensor(1900.0/CB_SCALE, device=device, requires_grad=True)
rb_norm = torch.tensor(1.5/2.0, device=device, requires_grad=True)
ab_norm = torch.tensor(0.5, device=device, requires_grad=True)
gain    = torch.tensor(0.0, device=device, requires_grad=True)

opt = torch.optim.Adam([cb_norm, rb_norm, ab_norm, gain], lr=0.01)

print("\n=== STARTING RAY-MODE INVERSION ===")
for epoch in range(1001):
    opt.zero_grad()
    c_b = cb_norm * 2000.0
    rho_b = rb_norm * 2.0
    alpha_b = ab_norm * 1.0
    
    pred = pressure_field(c_b, rho_b, alpha_b, gain, r_meas, 150.0)
    loss = (pred - TL_measured_db).pow(2).mean()
    loss.backward()
    opt.step()
    
    if epoch % 100 == 0:
        print(f"Ep {epoch}: Loss={loss.item():.2f} | c_b={c_b.item():.1f} | rho_b={rho_b.item():.3f} | alpha={alpha_b.item():.3f}")

print("------------------------------------------------")
print(f"Recovered: c_b={c_b.item():.1f}, rho_b={rho_b.item():.3f}")
print("True Ref : c_b=2000.0, rho_b=1.800") 
print("------------------------------------------------")