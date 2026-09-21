# -*- coding: utf-8 -*-
"""
Created on Thu Feb  5 23:07:13 2026

@author: 13391
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Feb  1 21:04:19 2026

@author: l00905881
"""

import torch
import numpy as np
import scipy.io as sio
import os
import math
import time

# =========================================================================
# 0. CONFIGURATION & GLOBALS
# =========================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float64)

# Environment Constants (Fixed Geometry)
C_W_GLOBAL   = 1500.0
RHO_W_GLOBAL = 1.0
D_GLOBAL     = 100.0
ZS_GLOBAL    = 25.0
ZR_GLOBAL    = 50.0 

# Frequency List
FREQ_LIST = [30.0, 35.0, 50.0, 100.0, 150.0, 200, 250]

# =========================================================================
# 1. DATA LOADING (External KRAKEN Files)
# =========================================================================
print("--- LOADING KRAKEN DATA ---")
TL_meas_dict = {}
r_meas = None

for f in FREQ_LIST:
    mat_file = f'TL_kraken_2000_260122_{int(f)}.mat'
    #mat_file = f'TL_kraken_1910_{int(f)}.mat'
    
    if not os.path.exists(mat_file):
        print(f"[WARN] File {mat_file} not found. Skipping {f} Hz.")
        continue

    try:
        mat_data = sio.loadmat(mat_file)
        # Handle potential variable name differences if needed
        key_name = 'TL_2D' if 'TL_2D' in mat_data else 'TL'
        TL_tensor = torch.tensor(mat_data[key_name], device=device)
        
        # Define Range Axis (Once)
        if r_meas is None:
            r_full = torch.linspace(1.0, 5000.0, TL_tensor.shape[1], device=device)
            mask = (r_full >= 1000.0) & (r_full <= 5000.0)
            r_meas = r_full[mask]
        
        # Extract Depth 50
        TL_meas_dict[f] = TL_tensor[50, mask]
        print(f"[OK] Loaded {f} Hz from {mat_file}")
        
    except Exception as e:
        print(f"[ERROR] Failed to load {mat_file}: {e}")

if not TL_meas_dict:
    print("[FATAL] No data loaded. Please run MATLAB script first.")
    exit()

# =========================================================================
# 2. SOLVER ENGINE
# =========================================================================
def unified_complex_solver(c_b, rho_b, alpha_b, omega, freq, coarse_mode=False, initial_seeds=None):
    c_w, rho_w, D = C_W_GLOBAL, RHO_W_GLOBAL, D_GLOBAL
    k0 = omega / c_w
    k_min = omega / 3000.0
    
    # Complex Bottom Wavenumber
    alpha_safe = torch.clamp(alpha_b, min=1e-4)
    kb_complex = (omega / c_b) + 1j * ((alpha_safe * (freq/c_b))/8.686)

    # --- 1. INITIALIZATION ---
    if initial_seeds is not None:
        # STRATEGY: ROBUST WARM START
        # Use previous roots. High precision requirement.
        k = initial_seeds
        iters = 20  # Increased from 5 -> 20 to restore 0.466 accuracy
    elif coarse_mode:
        # STRATEGY: REAL SCAN (Stage 1)
        k_scan = torch.linspace(k0 * 0.999, k_min, 1000, device=device, dtype=torch.float64)
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
        # STRATEGY: COMPLEX GRID (Cold Start)
        real_vals = torch.linspace(k0*1.01, k_min, 400, device=device)
        imag_vals = torch.tensor([0.0, 1e-4, 5e-4], device=device) 
        g_real, g_imag = torch.meshgrid(real_vals, imag_vals, indexing='ij')
        k = (g_real + 1j * g_imag).flatten()
        iters = 30

    # --- 2. NEWTON REFINEMENT ---
    for _ in range(iters):
        kz_w = torch.sqrt(k0**2 - k**2)
        kz_b = torch.sqrt(kb_complex**2 - k**2)
        s_w, c_w_ = torch.sin(kz_w*D), torch.cos(kz_w*D)
        
        F = rho_w * kz_b * s_w + 1j * rho_b * kz_w * c_w_
        
        dkzw_dk = -k/(kz_w + 1e-20)
        dkzb_dk = -k/(kz_b + 1e-20)
        dF = rho_w*(dkzb_dk*s_w + kz_b*c_w_*D*dkzw_dk) + 1j*rho_b*(dkzw_dk*c_w_ - kz_w*s_w*D*dkzw_dk)
        
        k = k - F / (dF + 1e-20)

    # --- 3. FILTERING ---
    if coarse_mode and initial_seeds is None:
        k_real = k.real
        if len(k) > 1:
            idx = torch.argsort(k_real, descending=True)
            k = k[idx]
            diff = torch.abs(k[1:] - k[:-1])
            unique = torch.cat([torch.tensor([True], device=device), diff > 1e-5])
            k = k[unique]
    else:
        # Complex filtering (Warm or Cold)
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
        # 1. Run the Solver (Forward Pass)
        with torch.no_grad():
            k_star = unified_complex_solver(cb, rho_b, alpha_b, omega, freq, coarse_mode, initial_seeds)
        
        # 2. Save Context
        ctx.save_for_backward(k_star, cb, rho_b, alpha_b)
        ctx.consts = (omega, freq)
        return k_star

    @staticmethod
    def backward(ctx, grad_output):
        k, cb, rb, ab = ctx.saved_tensors
        omega, freq = ctx.consts
        cw, rw, D = C_W_GLOBAL, RHO_W_GLOBAL, D_GLOBAL
        
        # --- PURE MANUAL ANALYTICAL DERIVATION ---
        k_g = k
        cb_v = cb.expand_as(k)
        rb_v = rb.expand_as(k)
        ab_v = ab.expand_as(k)
        
        k0 = omega / cw
        kb = (omega / cb_v) + 1j * ((ab_v * (freq/cb_v))/8.686)
        
        kz_w = torch.sqrt(k0**2 - k_g**2)
        kz_b = torch.sqrt(kb**2 - k_g**2)
        s_w, c_w_v = torch.sin(kz_w * D), torch.cos(kz_w * D)

        # Denominator: dF/dk 
        dkzw_dk = -k_g / (kz_w + 1e-20)
        dkzb_dk = -k_g / (kz_b + 1e-20)
        
        term1_k = dkzb_dk * s_w + kz_b * c_w_v * D * dkzw_dk
        term2_k = dkzw_dk * c_w_v - kz_w * s_w * D * dkzw_dk
        dF_dk = rw * term1_k + 1j * rb_v * term2_k

        # Numerators: dF/dTheta
        dF_dkzb = rw * s_w
        dkzb_dkb = kb / (kz_b + 1e-20)
        
        # A. dF/d(rho_b)
        dF_drb = 1j * kz_w * c_w_v

        # B. dF/d(c_b)
        dkb_dcb = -kb / (cb_v + 1e-20)
        dF_dcb = dF_dkzb * dkzb_dkb * dkb_dcb

        # C. dF/d(alpha_b)
        dkb_dab = 1j * freq / (8.686 * cb_v + 1e-20)
        dF_dab = dF_dkzb * dkzb_dkb * dkb_dab

        # Clamping for stability
        denom = dF_dk
        is_stiff = denom.abs() < 1e-4
        if is_stiff.any():
            denom = torch.where(is_stiff, denom + 1e-4, denom)

        dk_dcb = - dF_dcb / (denom + 1e-20)
        dk_drb = - dF_drb / (denom + 1e-20)
        dk_dab = - dF_dab / (denom + 1e-20)

        # Gradient accumulation
        grad_conj = grad_output.conj()
        grad_cb = (grad_conj * dk_dcb).sum().real
        grad_rb = (grad_conj * dk_drb).sum().real
        grad_ab = (grad_conj * dk_dab).sum().real
        
        # Return gradients matching input signature (added None for initial_seeds)
        return grad_cb, grad_rb, grad_ab, None, None, None, None

def forward_model(c_b, rho_b, alpha_b, r_range, omega, freq, stage=1, initial_seeds=None):
    if stage == 1:
        # Stage 1: Coarse Mode (Real Scan)
        k_complex = IFTSolver.apply(c_b, rho_b, alpha_b, omega, freq, True, None)
    else:
        # Stage 2: Complex Mode (Trapped + Leaky) - Clamped IFT + Warm Start Support
        k_complex = IFTSolver.apply(c_b, rho_b, alpha_b, omega, freq, False, initial_seeds)
    
    # --- Handling Empty Modes ---
    if len(k_complex) == 0: 
        dummy_pred = torch.ones_like(r_range)*100.0 + (c_b*0+rho_b*0+alpha_b*0)
        return dummy_pred, k_complex

    k0 = omega / C_W_GLOBAL

    # --- Filters ---
    if stage == 2:
        # Remove Ghost Artifacts (Singularity)
        is_ghost = torch.abs(k_complex.real - k0) < 2e-4
        k_complex = k_complex[~is_ghost]

    if len(k_complex) == 0:
        dummy_pred = torch.ones_like(r_range)*100.0 + (c_b*0+rho_b*0+alpha_b*0)
        return dummy_pred, k_complex

    # --- Field Calculation ---
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
    
    # Return BOTH TL and Modes (for tracking)
    return tl_pred, k_complex

# =========================================================================
# 3. OPTIMIZATION LOOP
# =========================================================================
print("\n=== STARTING 2-STAGE MULTI-FREQ INVERSION ===")

cb_opt = torch.tensor(1950.0, device=device, requires_grad=True)
rb_opt = torch.tensor(1.2, device=device, requires_grad=True)
ab_opt = torch.tensor(0.1, device=device, requires_grad=True)

# Precompute available freqs/omegas
active_freqs = [f for f in FREQ_LIST if f in TL_meas_dict]
active_omegas = [2 * math.pi * f for f in active_freqs]

# --- CONVERGENCE HELPER ---
def get_max_eta(optimizer):
    """Returns the maximum step size currently stored in the Rprop optimizer."""
    max_step = 0.0
    # Iterate over all parameter groups and all parameters
    for group in optimizer.param_groups:
        for p in group['params']:
            state = optimizer.state[p]
            # 'step_size' is only initialized after the first .step()
            if 'step_size' in state:
                # It can be a tensor or a float, so we use .max() and .item()
                step_val = state['step_size']
                if torch.is_tensor(step_val):
                    current_max = step_val.max().item()
                else:
                    current_max = step_val
                if current_max > max_step:
                    max_step = current_max
    return max_step

EPSILON = 5e-4  # Convergence tolerance 1e-6
MAX_ITERS = 500 # Safety Break

# -------------------------------------------------------------------------
# STAGE 1: BASIN FINDING (All Freqs, Equal Weight)
# -------------------------------------------------------------------------
print(f">>> STAGE 1: Basin Finding (Freqs: {active_freqs})")
optimizer = torch.optim.Rprop([cb_opt, rb_opt, ab_opt], lr=0.005)

time_start = time.time()
iteration = 0
max_eta = 1.0 # Force entry into loop

# Dynamic Loop: Run until step sizes decay below EPSILON or hit MAX_ITERS
while max_eta > EPSILON and iteration < MAX_ITERS:
    optimizer.zero_grad()
    total_loss = 0.0
    
    for freq, omega in zip(active_freqs, active_omegas):
        pred, _ = forward_model(cb_opt, rb_opt, ab_opt, r_meas, omega, freq, stage=1)
        total_loss += torch.nn.functional.mse_loss(pred, TL_meas_dict[freq])
    
    total_loss.backward()
    optimizer.step()
    
    with torch.no_grad():
        cb_opt.clamp_(1550, 2500); rb_opt.clamp_(1.0, 3.0); ab_opt.clamp_(0.01, 5.0)
    
    # Update convergence metric
    max_eta = get_max_eta(optimizer)
    iteration += 1

    if iteration % 10 == 0:
        print(f"Iter {iteration:04d} | Loss: {total_loss.item():.4f} | MaxEta: {max_eta:.2e} | cb: {cb_opt.item():.1f} | rho: {rb_opt.item():.3f} | alpha: {ab_opt.item():.3f}")

# -------------------------------------------------------------------------
# STAGE 2: PHYSICS-BASED DECOUPLING (Range Split)
# -------------------------------------------------------------------------
print(f"\n>>> STAGE 2: Range-Split Refinement")

def get_loss(pred, target, range_mode='full'):
    if range_mode == 'far':
        n_points = pred.shape[0]
        half_idx = n_points // 4*3 
        return torch.nn.functional.mse_loss(pred[half_idx:], target[half_idx:])
    else:
        return torch.nn.functional.mse_loss(pred, target)

prev_modes_dict = {}

# --- STEP 1: FIX PHASE (cb, rho) using FULL DATA ---
print("   [Step 1] Optimizing Phase (cb, rho)...")
ab_opt.requires_grad = True
cb_opt.requires_grad = True
rb_opt.requires_grad = True

# Re-initialize optimizer for Stage 2 to reset step sizes
optimizer_phase = torch.optim.Rprop([ab_opt, cb_opt, rb_opt], lr=0.005)

iteration = 0
max_eta = 1.0
MAX_ITERS_S2 = 500

while max_eta > EPSILON and iteration < MAX_ITERS_S2:
    optimizer_phase.zero_grad()
    total_loss = 0.0
    
    for freq, omega in zip(active_freqs, active_omegas):
        seeds = prev_modes_dict.get(freq, None)
        pred, k_final = forward_model(
            cb_opt, rb_opt, ab_opt, r_meas, omega, freq, 
            stage=2, initial_seeds=seeds
        )
        prev_modes_dict[freq] = k_final.detach()
        total_loss += get_loss(pred, TL_meas_dict[freq], 'far')
    
    total_loss.backward()
    optimizer_phase.step()
    
    with torch.no_grad():
        cb_opt.clamp_(1550, 2500); rb_opt.clamp_(1.0, 3.0)
    
    # Update convergence metric
    max_eta = get_max_eta(optimizer_phase)
    iteration += 1

    if iteration % 10 == 0:
        print(f"   Iter {iteration:03d} | Loss: {total_loss.item():.4f} | MaxEta: {max_eta:.2e} | cb: {cb_opt.item():.1f} | rho: {rb_opt.item():.3f} | alpha: {ab_opt.item():.3f}")

time_end = time.time()
print('time')
print(time_end-time_start)