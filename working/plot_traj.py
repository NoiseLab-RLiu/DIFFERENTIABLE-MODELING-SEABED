# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 22:11:42 2026

@author: 13391
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
import matplotlib.pyplot as plt
import math

import matplotlib.lines as mlines
# =========================================================================
# 0. CONFIGURATION & GLOBALS
# =========================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float64)

# Environment Constants
C_W_GLOBAL   = 1500.0
RHO_W_GLOBAL = 1.0
D_GLOBAL     = 100.0
ZS_GLOBAL    = 25.0
ZR_GLOBAL    = 50.0 

# Frequencies
FREQ_LIST = [30.0, 35.0, 50.0, 100.0, 150.0, 200, 250]

# Ground Truth for "Measured" Data
TRUE_PARAMS = {'cb': 2026.0, 'rho': 1.3, 'alpha': 0.8}

# Search Config (Starting Points)
START_CBS = [1510, 1600, 1650, 1700, 1750, 1850, 1950, 2050, 2090]
FIXED_START_RHO = 1.2
FIXED_START_ALPHA = 0.1

# =========================================================================
# 1. SOLVER ENGINE 
# =========================================================================
def unified_complex_solver(c_b, rho_b, alpha_b, omega, freq, coarse_mode=False, initial_seeds=None):
    c_w, rho_w, D = C_W_GLOBAL, RHO_W_GLOBAL, D_GLOBAL
    k0 = omega / c_w
    k_min = omega / 3000.0
    
    alpha_safe = torch.clamp(alpha_b, min=1e-4)
    kb_complex = (omega / c_b) + 1j * ((alpha_safe * (freq/c_b))/8.686)

    if initial_seeds is not None:
        k = initial_seeds
        iters = 20
    elif coarse_mode:
        k_scan = torch.linspace(k0 * 0.999, k_min, 1000, device=device)
        kz_w = torch.sqrt(torch.clamp(k0**2 - k_scan**2, min=1e-12))
        kz_b = torch.sqrt(torch.clamp(k_scan**2 - (omega/c_b)**2, min=1e-12))
        s_w, c_w_val = torch.sin(kz_w * D), torch.cos(kz_w * D)
        F_scan = rho_w * kz_b * s_w + rho_b * kz_w * c_w_val
        
        sign_change = (F_scan[:-1] * F_scan[1:]) < 0
        jump_filter = torch.abs(F_scan[:-1] - F_scan[1:]) < 500.0 
        valid_indices = torch.nonzero(sign_change & jump_filter).flatten()
        if len(valid_indices) == 0: return torch.tensor([]).to(device)
        k = k_scan[valid_indices].to(dtype=torch.complex128)
        iters = 20
    else:
        real_vals = torch.linspace(k0*1.01, k_min, 400, device=device)
        imag_vals = torch.tensor([0.0, 1e-4, 5e-4], device=device) 
        g_real, g_imag = torch.meshgrid(real_vals, imag_vals, indexing='ij')
        k = (g_real + 1j * g_imag).flatten()
        iters = 30

    for _ in range(iters):
        kz_w = torch.sqrt(k0**2 - k**2)
        kz_b = torch.sqrt(kb_complex**2 - k**2)
        s_w, c_w_ = torch.sin(kz_w*D), torch.cos(kz_w*D)
        F = rho_w * kz_b * s_w + 1j * rho_b * kz_w * c_w_
        dkzw_dk = -k/(kz_w + 1e-20)
        dkzb_dk = -k/(kz_b + 1e-20)
        dF = rho_w*(dkzb_dk*s_w + kz_b*c_w_*D*dkzw_dk) + 1j*rho_b*(dkzw_dk*c_w_ - kz_w*s_w*D*dkzw_dk)
        k = k - F / (dF + 1e-20)

    if coarse_mode and initial_seeds is None:
        k_real = k.real
        if len(k) > 1:
            idx = torch.argsort(k_real, descending=True)
            k = k[idx]
            diff = torch.abs(k[1:] - k[:-1])
            unique = torch.cat([torch.tensor([True], device=device), diff > 1e-5])
            k = k[unique]
    else:
        valid_mask = (k.real > k_min) & (k.real < k0*1.01) & (k.imag > -1e-8)
        k = k[valid_mask]
        if len(k) > 0:
            k = k[torch.argsort(k.real, descending=True)]
            diff = torch.abs(k[1:] - k[:-1])
            k = k[torch.cat([torch.tensor([True], device=device), diff > 1e-4])]
            
    return k

class IFTSolver(torch.autograd.Function):
    @staticmethod
    def forward(ctx, cb, rho_b, alpha_b, omega, freq, coarse_mode, initial_seeds):
        with torch.no_grad():
            k_star = unified_complex_solver(cb, rho_b, alpha_b, omega, freq, coarse_mode, initial_seeds)
        ctx.save_for_backward(k_star, cb, rho_b, alpha_b)
        ctx.consts = (omega, freq)
        return k_star

    @staticmethod
    def backward(ctx, grad_output):
        k, cb, rb, ab = ctx.saved_tensors
        omega, freq = ctx.consts
        cw, rw, D = C_W_GLOBAL, RHO_W_GLOBAL, D_GLOBAL
        k_g, cb_v, rb_v, ab_v = k, cb.expand_as(k), rb.expand_as(k), ab.expand_as(k)
        
        k0 = omega / cw
        kb = (omega / cb_v) + 1j * ((ab_v * (freq/cb_v))/8.686)
        kz_w = torch.sqrt(k0**2 - k_g**2)
        kz_b = torch.sqrt(kb**2 - k_g**2)
        s_w, c_w_v = torch.sin(kz_w * D), torch.cos(kz_w * D)

        dkzw_dk = -k_g / (kz_w + 1e-20)
        dkzb_dk = -k_g / (kz_b + 1e-20)
        term1_k = dkzb_dk * s_w + kz_b * c_w_v * D * dkzw_dk
        term2_k = dkzw_dk * c_w_v - kz_w * s_w * D * dkzw_dk
        dF_dk = rw * term1_k + 1j * rb_v * term2_k

        dF_dkzb = rw * s_w
        dkzb_dkb = kb / (kz_b + 1e-20)
        dF_drb = 1j * kz_w * c_w_v
        dkb_dcb = -kb / (cb_v + 1e-20)
        dF_dcb = dF_dkzb * dkzb_dkb * dkb_dcb
        dkb_dab = 1j * freq / (8.686 * cb_v + 1e-20)
        dF_dab = dF_dkzb * dkzb_dkb * dkb_dab

        denom = dF_dk
        if (denom.abs() < 1e-4).any(): denom = torch.where(denom.abs() < 1e-4, denom + 1e-4, denom)

        dk_dcb = - dF_dcb / (denom + 1e-20)
        dk_drb = - dF_drb / (denom + 1e-20)
        dk_dab = - dF_dab / (denom + 1e-20)

        grad_conj = grad_output.conj()
        return (grad_conj * dk_dcb).sum().real, (grad_conj * dk_drb).sum().real, (grad_conj * dk_dab).sum().real, None, None, None, None

def forward_model(c_b, rho_b, alpha_b, r_range, omega, freq, stage=1, initial_seeds=None):
    if stage == 1:
        k_complex = IFTSolver.apply(c_b, rho_b, alpha_b, omega, freq, True, None)
    else:
        k_complex = IFTSolver.apply(c_b, rho_b, alpha_b, omega, freq, False, initial_seeds)
    
    if len(k_complex) == 0: return torch.ones_like(r_range)*100.0, k_complex
    k0 = omega / C_W_GLOBAL
    if stage == 2: k_complex = k_complex[~ (torch.abs(k_complex.real - k0) < 2e-4)]
    if len(k_complex) == 0: return torch.ones_like(r_range)*100.0, k_complex

    kz_w = torch.sqrt(k0**2 - k_complex**2)
    kb_complex = (omega/c_b) + 1j*((alpha_b * (freq/c_b))/8.686)
    kz_b = torch.sqrt(kb_complex**2 - k_complex**2)
    
    term1, term2 = D_GLOBAL/2, torch.sin(2*kz_w*D_GLOBAL)/(4*kz_w)
    term3 = (RHO_W_GLOBAL/rho_b) * (torch.sin(kz_w*D_GLOBAL)**2) / (2j * kz_b)
    A_m = 1.0 / torch.sqrt(term1 - term2 + term3)
    
    phase = k_complex[:, None] * r_range[None, :]
    H0 = torch.sqrt(2 / (math.pi * phase)) * torch.exp(1j * (phase - math.pi/4))
    Zs, Zr = A_m * torch.sin(kz_w * ZS_GLOBAL), A_m * torch.sin(kz_w * ZR_GLOBAL)
    p_sum = 1j/(4*RHO_W_GLOBAL) * (Zs[:, None] * Zr[:, None] * H0).sum(dim=0)
    
    return -20 * torch.log10(p_sum.abs() * 4.0 * math.pi + 1e-12), k_complex

# =========================================================================
# 2. DATA GENERATION (Target)
# =========================================================================
print("--- Generating Synthetic Target Data ---")
r_meas = torch.linspace(1000.0, 5000.0, 801, device=device)
TL_meas_dict = {}
cb_t = torch.tensor(TRUE_PARAMS['cb'], device=device)
rho_t = torch.tensor(TRUE_PARAMS['rho'], device=device)
alpha_t = torch.tensor(TRUE_PARAMS['alpha'], device=device)
for f in FREQ_LIST:
    omega = 2 * math.pi * f
    tl, _ = forward_model(cb_t, rho_t, alpha_t, r_meas, omega, f, stage=1)
    TL_meas_dict[f] = tl.detach()
active_omegas = [2 * math.pi * f for f in FREQ_LIST]

# =========================================================================
# 3. LOAD PRE-COMPUTED LOSS GRID
# =========================================================================
print("--- Loading Loss Grid ---")
if not os.path.exists('loss_grid.npy'):
    print("[ERROR] loss_grid.npy not found. Please run the generation script first.")
    exit()

loss_map = np.load('loss_grid.npy')
loss_db = 10 * np.log10(loss_map + 1e-12)
print(f"Loaded loss_grid with shape: {loss_map.shape}")

# === CORRECTION: EXACT GRID ALIGNMENT ===
# np.arange excludes the stop value, so we add the step to include it
# cb: 1510 to 2100 with step 10
cb_grid = np.arange(1510, 2100 + 10, 10)

# rho: 1.0 to 2.0 with step 0.02
rho_grid = np.arange(1.0, 2.0 + 0.02, 0.02) 

# Verify dimensions
if loss_map.shape != (len(rho_grid), len(cb_grid)):
    print(f"[WARNING] Dimension Mismatch!")
    print(f"  Loaded map: {loss_map.shape}")
    print(f"  Constructed grid: {(len(rho_grid), len(cb_grid))}")
    print("  Check your .npy generation code parameters.")

# =========================================================================
# 4. RUN TRAJECTORIES
# =========================================================================
trajectories = []

def get_max_eta(optimizer):
    max_step = 0.0
    for group in optimizer.param_groups:
        for p in group['params']:
            state = optimizer.state[p]
            if 'step_size' in state:
                step_val = state['step_size']
                curr = step_val.max().item() if torch.is_tensor(step_val) else step_val
                max_step = max(max_step, curr)
    return max_step

print("\n--- Running Multi-Start Optimization ---")
for start_cb in START_CBS:
    print(f"Running Inversion starting at cb={start_cb}...")
    
    cb_opt = torch.tensor(float(start_cb), device=device, requires_grad=True)
    rb_opt = torch.tensor(float(FIXED_START_RHO), device=device, requires_grad=True)
    ab_opt = torch.tensor(float(FIXED_START_ALPHA), device=device, requires_grad=True)
    
    path_cb = [start_cb]
    path_rho = [FIXED_START_RHO]
    
    # --- STAGE 1 ---
    optimizer = torch.optim.Rprop([cb_opt, rb_opt, ab_opt], lr=0.005, etas=(0.5, 1.2))
    max_eta = 1.0; it = 0
    while max_eta > 5e-4 and it < 150:
        optimizer.zero_grad()
        loss = 0.0
        for f, omega in zip(FREQ_LIST, active_omegas):
            pred, _ = forward_model(cb_opt, rb_opt, ab_opt, r_meas, omega, f, stage=1)
            loss += torch.nn.functional.mse_loss(pred, TL_meas_dict[f])
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            cb_opt.clamp_(1500, 2500); rb_opt.clamp_(1.01, 1.99); ab_opt.clamp_(0.01, 5.0)
            #cb_opt.clamp_(1500, 2500); rb_opt.clamp_(1.0, 3.0); ab_opt.clamp_(0.01, 5.0)
        path_cb.append(cb_opt.item())
        path_rho.append(rb_opt.item())
        max_eta = get_max_eta(optimizer)
        it += 1

    # --- STAGE 2 ---
    optimizer = torch.optim.Rprop([cb_opt, rb_opt, ab_opt], lr=0.005, etas=(0.5, 1.2))
    max_eta = 1.0; it = 0; prev_modes = {}
    while max_eta > 5e-4 and it < 150:
        optimizer.zero_grad()
        loss = 0.0
        for f, omega in zip(FREQ_LIST, active_omegas):
            seeds = prev_modes.get(f, None)
            pred, k_f = forward_model(cb_opt, rb_opt, ab_opt, r_meas, omega, f, stage=2, initial_seeds=seeds)
            prev_modes[f] = k_f.detach()
            loss += torch.nn.functional.mse_loss(pred, TL_meas_dict[f])
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            cb_opt.clamp_(1500, 2500); rb_opt.clamp_(1.01, 1.99)#rb_opt.clamp_(1.0, 3.0)
        path_cb.append(cb_opt.item())
        path_rho.append(rb_opt.item())
        max_eta = get_max_eta(optimizer)
        it += 1
        
    trajectories.append((path_cb, path_rho))

# =========================================================================
# 2. PLOTTING
# =========================================================================
# Increase figure width to accommodate Plot + Colorbar + Legend side-by-side
fig, ax = plt.subplots(figsize=(14, 7))

# A. Plot Heatmap
extent = [cb_grid.min(), cb_grid.max(), rho_grid.min(), rho_grid.max()]
im = ax.imshow(loss_map, 
               origin='lower', 
               aspect='auto', 
               extent=extent, 
               cmap='viridis', 
               interpolation='nearest',
               vmin = 0,
               vmax = 600)

# B. Add Colorbar
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label(r'MSE Loss in linear scale', fontsize=15)

# C. Plot Trajectories
legend_handles = []
colors = plt.cm.jet(np.linspace(0, 1, len(trajectories)))

for i, (pcb, prho) in enumerate(trajectories[1:]):
    # 1. Trajectory Line (PINK - as requested)
    # Using 'violet' or 'orchid' can sometimes be clearer on dark backgrounds, 
    # but 'pink' is used here specifically.
    ax.plot(pcb, prho, color='pink', linewidth=1.5, alpha=0.9)
    
    # 2. Start Point (Colorful Dot)
    l_handle = ax.scatter(pcb[0], prho[0], color=colors[i], edgecolors='white', 
                          s=60, zorder=10, label=f'Start $c_b$={int(pcb[0])}')
    legend_handles.append(l_handle)
    
    # 3. End Point (RED Cross)
    ax.scatter(pcb[-1], prho[-1], marker='x', color='red', s=80, zorder=15, linewidth=2.5)

# D. Ground Truth (RED Hollow Circle)
# Distinguishable from the cross because it is much larger (s=500)
ax.scatter(TRUE_PARAMS['cb'], TRUE_PARAMS['rho'], s=500, marker='o', 
           facecolors='none', edgecolors='red', linewidth=3, zorder=20)

# -------------------------------------------------------------------------
# LEGEND PROXIES
# -------------------------------------------------------------------------

# 1. Converged Point Proxy (Red Cross)
end_proxy = mlines.Line2D([], [], color='red', marker='x', linestyle='None',
                          markersize=10, markeredgewidth=2.5, label='Converged Point')

# 2. Ground Truth Proxy (Red Circle)
gt_proxy = mlines.Line2D([], [], color='white', marker='o', linestyle='None',
                         markerfacecolor='none', markeredgecolor='red', markeredgewidth=2,
                         markersize=15, label='Ground Truth')

legend_handles.append(end_proxy)
legend_handles.append(gt_proxy)

# F. Formatting
ax.set_title(f'Inversion Trajectories on Log-Loss Landscape (True $c_b$={int(TRUE_PARAMS["cb"])})', fontsize=14)
ax.set_xlabel('Seabed Sound Speed (m/s)', fontsize=12)
ax.set_ylabel('Seabed Density (g/cm³)', fontsize=12)
ax.grid(True, linestyle=':', alpha=0.4)
ax.set_ylim(1.0, 2.0)

# G. Legend Positioning (No Frame)
# frameon=False removes the surrounding square
ax.legend(handles=legend_handles, bbox_to_anchor=(1.25, 1), loc='upper left', 
          borderaxespad=0., frameon=False)

plt.subplots_adjust(right=0.75) 

plt.show()