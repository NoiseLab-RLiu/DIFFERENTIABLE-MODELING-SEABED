# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 23:29:59 2026

@author: 13391
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import math
import torch
import scipy.io as sio
import numpy as np
import gc

# =========================================================================
# 0. CONFIGURATION
# =========================================================================
gc.collect()
torch.cuda.empty_cache()
torch.set_num_threads(1)
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
mat_file = 'TL_kraken2.mat'
try:
    mat_data = sio.loadmat(mat_file)
    TL_tensor = torch.tensor(mat_data['TL_2D'], device=device)
    r_full = torch.linspace(1.0, 5000.0, TL_tensor.shape[1], device=device)
    mask = (r_full >= 1000.0) & (r_full <= 5000.0)
    r_meas, TL_meas = r_full[mask], TL_tensor[50, mask]
    print(f"[OK] Data Loaded. Points: {len(r_meas)}")
except Exception as e:
    print(f"[ERROR] {e}"); exit()

# =========================================================================
# 2. HYBRID PHYSICS ENGINE
# =========================================================================

def unified_complex_solver(c_b, rho_b, alpha_b, coarse_mode=False):
    """
    Hybrid Solver:
    - coarse_mode=True:  Real Scan (Mimics Stage 1, finds Evanescent Ghost)
    - coarse_mode=False: Complex Grid (Mimics Stage 3, finds Leaky)
    """
    c_w, rho_w, D, omega = C_W_GLOBAL, RHO_W_GLOBAL, D_GLOBAL, OMEGA_GLOBAL
    k0 = omega / c_w
    k_min = omega / 3000.0
    kb_complex = (omega / c_b) + 1j * ((alpha_b * (FREQ_GLOBAL/c_b))/8.686)

    if coarse_mode:
        # --- MODE A: REAL SCAN (Your Stage 1 Logic) ---
        # This guarantees finding the "Good Ghost" naturally
        k_scan = torch.linspace(k0 * 0.999, k_min, 1000, device=device, dtype=torch.float64)
        
        # Characteristic Function (Tan form)
        kz_w = torch.sqrt(torch.clamp(k0**2 - k_scan**2, min=1e-12))
        kz_b = torch.sqrt(torch.clamp(k_scan**2 - (omega/c_b)**2, min=1e-12))
        F_scan = torch.tan(kz_w * D) + (rho_b/rho_w) * (kz_w / kz_b)
        
        # Sign Change Detection
        sign_change = (F_scan[:-1] * F_scan[1:]) < 0
        jump_filter = torch.abs(F_scan[:-1] - F_scan[1:]) < 50.0
        valid_indices = torch.nonzero(sign_change & jump_filter).flatten()
        
        if len(valid_indices) == 0: return torch.tensor([]).to(device)
        k = k_scan[valid_indices].to(dtype=torch.complex128)
        
        # Newton (20 iters)
        iters = 20
    else:
        # --- MODE B: COMPLEX GRID (Your Stage 3 Logic) ---
        # This finds Leaky modes for high precision
        real_vals = torch.linspace(k0*1.01, k_min, 400, device=device)
        imag_vals = torch.tensor([0.0, 1e-4, 1e-3], device=device)
        g_real, g_imag = torch.meshgrid(real_vals, imag_vals, indexing='ij')
        k = (g_real + 1j * g_imag).flatten()
        
        iters = 30

    # Common Newton Refinement
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
    if coarse_mode:
        # Stage 1: Just remove duplicates
        k_real = k.real
        if len(k) > 1:
            idx = torch.argsort(k_real, descending=True)
            k = k[idx]
            diff = torch.abs(k[1:] - k[:-1])
            unique = torch.cat([torch.tensor([True], device=device), diff > 1e-5])
            k = k[unique]
    else:
        # Stage 3: Strict physical bounds
        valid_mask = (k.real > k_min) & (k.real < k0*1.01) & (k.imag > -1e-8)
        k = k[valid_mask]
        if len(k) > 0:
            idx = torch.argsort(k.real, descending=True)
            k = k[idx]
            diff = torch.abs(k[1:] - k[:-1])
            unique = torch.cat([torch.tensor([True], device=device), diff > 1e-4])
            k = k[unique]
            
    return k

class IFTSolver(torch.autograd.Function):
    @staticmethod
    def forward(ctx, cb, rho_b, alpha_b, coarse_mode):
        # 1. Run the solver in No-Grad mode
        with torch.no_grad():
            k_star = unified_complex_solver(cb, rho_b, alpha_b, coarse_mode=coarse_mode)
        
        # 2. Save context
        ctx.save_for_backward(k_star, cb, rho_b, alpha_b)
        return k_star

    @staticmethod
    def backward(ctx, grad_output):
        k_star, cb, rb, ab = ctx.saved_tensors
        c_w, rho_w, D, omega = C_W_GLOBAL, RHO_W_GLOBAL, D_GLOBAL, OMEGA_GLOBAL
        
        # 1. Re-enable Grads locally
        with torch.enable_grad():
            k = k_star.detach().requires_grad_(True)
            
            # Expand parameters to vectors
            cb_vec = cb.expand_as(k).clone().requires_grad_(True)
            rb_vec = rb.expand_as(k).clone().requires_grad_(True)
            ab_vec = ab.expand_as(k).clone().requires_grad_(True)
            
            # Reconstruct F
            k0 = omega / c_w
            kb_complex = (omega / cb_vec) + 1j * ((ab_vec * (FREQ_GLOBAL/cb_vec))/8.686)
            kz_w = torch.sqrt(k0**2 - k**2)
            kz_b = torch.sqrt(kb_complex**2 - k**2)
            s_w, c_w_ = torch.sin(kz_w*D), torch.cos(kz_w*D)
            
            F = rho_w * kz_b * s_w + 1j * rb_vec * kz_w * c_w_
            
            # 2. Compute Jacobian (Partials)
            # FIX: Add retain_graph=True for the first 3 calls!
            # Otherwise the graph is freed after dF_dk is computed.
            # 1. dF/dk
            dF_dk = torch.autograd.grad(F, k, torch.ones_like(F), create_graph=False, retain_graph=True)[0]
            # 2. dF/d_cb
            dF_dcb = torch.autograd.grad(F, cb_vec, torch.ones_like(F), create_graph=False, retain_graph=True)[0]
            # 3. dF/d_rb
            dF_drb = torch.autograd.grad(F, rb_vec, torch.ones_like(F), create_graph=False, retain_graph=True)[0]
            # 4. dF/d_ab (Last one doesn't strictly need retain_graph, but safe to add or leave False)
            dF_dab = torch.autograd.grad(F, ab_vec, torch.ones_like(F), create_graph=False, retain_graph=False)[0]

        # 3. Apply IFT Formula
        # dk/dp = - (dF/dp) / (dF/dk)
        # grad_input = grad_output * dk/dp
        
        term = -grad_output / (dF_dk + 1e-20)
        
        # Sum and take Real part (Fixes Data Type Error)
        grad_cb = (term * dF_dcb).sum().real
        grad_rb = (term * dF_drb).sum().real
        grad_ab = (term * dF_dab).sum().real
        
        # Return 4 gradients (one for each input to forward)
        # The last input 'coarse_mode' is not differentiable, so return None
        return grad_cb, grad_rb, grad_ab, None

def forward_model(c_b, rho_b, alpha_b, r_range, stage=1):
    # 1. SELECT SOLVER STRATEGY
    # CRITICAL FIX: Stage 2 must use the SAME solver as Stage 1 (coarse_mode=True)
    # to avoid "dataset shift". We only switch to the Complex Solver in Stage 3.
    # if stage == 1 or stage == 2:
    #     # Use Real Scan (Evanescent Ghost + Trapped)
    #     k_complex = unified_complex_solver(c_b, rho_b, alpha_b, coarse_mode=True)
    # else:
    #     # Stage 3: Use Complex Grid (Trapped + Leaky)
    #     #k_complex = unified_complex_solver(c_b, rho_b, alpha_b, coarse_mode=False)
    #     k_complex = IFTSolver.apply(c_b, rho_b, alpha_b, False)
        
    if stage == 1:
        # Use Real Scan (Evanescent Ghost + Trapped)
        k_complex = unified_complex_solver(c_b, rho_b, alpha_b, coarse_mode=True)
    elif stage ==2 :
        k_complex = unified_complex_solver(c_b, rho_b, alpha_b, coarse_mode=False)
        #k_complex = IFTSolver.apply(c_b, rho_b, alpha_b, False)
    else:
        # Stage 3: Use Complex Grid (Trapped + Leaky)
        k_complex = unified_complex_solver(c_b, rho_b, alpha_b, coarse_mode=False)
        #k_complex = IFTSolver.apply(c_b, rho_b, alpha_b, False)
        
    
    # use_coarse = (stage == 1 or stage == 2)
    # k_complex = IFTSolver.apply(c_b, rho_b, alpha_b, use_coarse)
    
    # Gradient safety
    if len(k_complex) == 0: 
        dummy_grad = c_b*0 + rho_b*0 + alpha_b*0
        return torch.ones_like(r_range)*100.0 + dummy_grad

    k0 = OMEGA_GLOBAL / C_W_GLOBAL

    # 2. PHYSICS FILTERS
    if stage == 2:
        # BRIDGE: Clean Trapped Only.
        # We are using the Coarse Solver, so we just need to kill the Ghost.
        # The Ghost in coarse mode is "Evanescent" (k > k0).
        # We also filter extreme artifacts near k0 just in case.
        
        # Filter: Keep only k < k0 (Propagating Trapped Modes)
        # This removes the "Good Ghost" found in Stage 1.
        mask_clean = k_complex.real < (k0 - 1e-5)
        k_complex = k_complex[mask_clean]
        
    elif stage == 3:
        # FULL: Trapped + Leaky.
        # We are using the Complex Solver. 
        # We must remove the "Singularity Artifact" (Ghost) if it appears.
        is_ghost = torch.abs(k_complex.real - k0) < 2e-4
        k_complex = k_complex[~is_ghost]

    # Safety check after filtering
    if len(k_complex) == 0:
        dummy_grad = c_b*0 + rho_b*0 + alpha_b*0
        return torch.ones_like(r_range)*100.0 + dummy_grad

    # 3. FIELD CALC
    kz_w = torch.sqrt(k0**2 - k_complex**2)
    kb_complex = (OMEGA_GLOBAL/c_b) + 1j*((alpha_b * (FREQ_GLOBAL/c_b))/8.686)
    kz_b = torch.sqrt(kb_complex**2 - k_complex**2)

    term1 = D_GLOBAL/2
    term2 = torch.sin(2*kz_w*D_GLOBAL)/(4*kz_w)
    term3 = (RHO_W_GLOBAL/rho_b) * (torch.sin(kz_w*D_GLOBAL)**2) / (2j * kz_b)
    A_m = 1.0 / torch.sqrt(term1 - term2 + term3)

    Zs = A_m * torch.sin(kz_w * ZS_GLOBAL)
    Zr = A_m * torch.sin(kz_w * ZR_GLOBAL)
    
    phase = k_complex[:, None] * r_range[None, :]
    H0 = torch.sqrt(2 / (math.pi * phase)) * torch.exp(1j * (phase - math.pi/4))
    p_sum = 1j/(4*RHO_W_GLOBAL) * (Zs[:, None] * Zr[:, None] * H0).sum(dim=0)
    
    return -20 * torch.log10(p_sum.abs() * 4.0 * math.pi + 1e-12)

# =========================================================================
# 3. ROBUST 3-STAGE OPTIMIZATION
# =========================================================================
print("\n=== STARTING 3-STAGE INVERSION ===")

cb_opt = torch.tensor(1850.0, device=device, requires_grad=True)
rb_opt = torch.tensor(1.2, device=device, requires_grad=True)
ab_opt = torch.tensor(0.1, device=device, requires_grad=True)

# STAGE 1: Ghost Assisted (Coarse Mode)
print(">>> STAGE 1: Ghost Assisted (Basin Finding)")
optimizer = torch.optim.Rprop([cb_opt, rb_opt, ab_opt], lr=0.01)

for i in range(300):
    optimizer.zero_grad()
    pred = forward_model(cb_opt, rb_opt, ab_opt, r_meas, stage=1)
    loss = torch.nn.functional.mse_loss(pred, TL_meas)
    loss.backward()
    optimizer.step()
    
    with torch.no_grad():
        cb_opt.clamp_(1550.0, 2500.0); rb_opt.clamp_(1.0, 3.0); ab_opt.clamp_(0.0, 5.0)
    
    if i % 100 == 0:
        print(f"Iter {i:04d} | Loss: {loss.item():.4f} | cb: {cb_opt.item():.1f} | rho: {rb_opt.item():.3f} | alpha: {ab_opt.item():.3f}")

# STAGE 2: Bridge (Clean Trapped)
print("\n>>> STAGE 2: Bridge (Clean Trapped)")
optimizer = torch.optim.Rprop([cb_opt, rb_opt, ab_opt], lr=0.01)

for i in range(1000):
    optimizer.zero_grad()
    pred = forward_model(cb_opt, rb_opt, ab_opt, r_meas, stage=2)
    loss = torch.nn.functional.mse_loss(pred, TL_meas)
    loss.backward()
    optimizer.step()
    
    with torch.no_grad():
        cb_opt.clamp_(1550.0, 2500.0); rb_opt.clamp_(1.0, 3.0); ab_opt.clamp_(0.0, 5.0)
        
    if i % 100 == 0:
        print(f"Iter {i:04d} | Loss: {loss.item():.4f} | cb: {cb_opt.item():.1f} | rho: {rb_opt.item():.3f} | alpha: {ab_opt.item():.3f}")

# # STAGE 3: Full Physics (Leaky)
# print("\n>>> STAGE 3: Full Physics (Final Polish)")
# optimizer = torch.optim.Rprop([cb_opt, rb_opt, ab_opt], lr=0.005)

# for i in range(600):
#     optimizer.zero_grad()
#     pred = forward_model(cb_opt, rb_opt, ab_opt, r_meas, stage=3)
#     loss = torch.nn.functional.mse_loss(pred, TL_meas)
#     loss.backward()
#     optimizer.step()
    
#     with torch.no_grad():
#         cb_opt.clamp_(1550.0, 2500.0); rb_opt.clamp_(1.0, 3.0); ab_opt.clamp_(0.0, 5.0)
        
#     if i % 100 == 0:
#         print(f"S3 Iter {i:04d} | Loss: {loss.item():.4f} | cb: {cb_opt.item():.2f} | rho: {rb_opt.item():.3f} | alpha: {ab_opt.item():.3f}")

# print(f"\nFINAL: cb={cb_opt.item():.2f}, rho={rb_opt.item():.3f}, alpha={ab_opt.item():.3f}")