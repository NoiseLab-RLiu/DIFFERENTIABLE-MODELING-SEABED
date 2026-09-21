# -*- coding: utf-8 -*-
"""
Created on Sun May 31 23:30:44 2026

@author: 13391
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 15:05:01 2026

@author: l00905881
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 11:58:19 2026

@author: l00905881
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import math
import torch
import scipy.io as sio
import gc

# =========================================================================
# 0. CONFIGURATION & GLOBALS
# =========================================================================
gc.collect()
torch.cuda.empty_cache()
torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

C_W_GLOBAL   = 1500.0  
RHO_W_GLOBAL = 1.0     
D_GLOBAL     = 100.0   
ZS_GLOBAL    = 25.0    
ZR_GLOBAL    = 50.0    

# Define your list of frequencies here
FREQ_LIST = [30.0, 35.0, 50.0, 100.0, 150.0, 200.0, 250.0]

# =========================================================================
# 1. LOAD PMM GROUND TRUTH (ALL FREQUENCIES)
# =========================================================================
meas_data = {}

for freq in FREQ_LIST:
    mat_file = f'TL_Pytorch_PMM_layered_{int(freq)}.mat'
    try:
        mat_data = sio.loadmat(mat_file)
        TL_tensor = torch.tensor(mat_data['TL_2D'], device=device)
        r_full = torch.tensor(mat_data['r_vec'].squeeze(), device=device)
        z_full = torch.tensor(mat_data['z_vec'].squeeze(), device=device)
        
        z_idx = int(round((ZR_GLOBAL / D_GLOBAL) * (len(z_full) - 1)))
        mask = (r_full >= 1000.0) & (r_full <= 5000.0)
        
        meas_data[freq] = {
            'r_meas': r_full[mask],
            'TL_meas': TL_tensor[z_idx, mask]
        }
        print(f"[OK] Ground Truth Loaded for {freq} Hz.")
    except Exception as e:
        print(f"[ERROR] Loading {freq} Hz: {e}")
        exit()

# =========================================================================
# 2. HYBRID PHYSICS ENGINE (FULLY DIFFERENTIABLE)
# =========================================================================
def get_layered_profile(freq):
    # Calculate the approximate wavelength (assuming an average c ~ 1500 m/s)
    lambda_approx = 1500.0 / freq
    
    # Rule of thumb: dz should be at most lambda / 10
    dz_target = lambda_approx / 10.0
    
    # Calculate required layers, enforcing a minimum of 100 layers for gradient stability
    N_layers = max(100, math.ceil(D_GLOBAL / dz_target))
    dz = D_GLOBAL / N_layers
    
    z_mid = torch.linspace(dz/2, D_GLOBAL - dz/2, N_layers, device=device, dtype=torch.float64)
    c = torch.zeros_like(z_mid)
    
    # 3-part downward refracting profile
    c = torch.where(z_mid <= 33.0, 1520.0 - (10.0/33.0)*z_mid, c)
    c = torch.where((z_mid > 33.0) & (z_mid <= 67.0), 1510.0 - (20.0/34.0)*(z_mid-33.0), c)
    c = torch.where(z_mid > 67.0, 1490.0 - (10.0/33.0)*(z_mid-67.0), c)
    
    return c, dz, N_layers

def compute_F(k, cb, rb, ab, freq):
    omega = 2 * math.pi * freq
    
    # Dynamically fetch the profile and exactly how many layers we need
    c, dz, N_layers = get_layered_profile(freq)

    V0 = torch.zeros_like(k, dtype=torch.complex128) 
    V1 = torch.ones_like(k, dtype=torch.complex128)  

    for j in range(N_layers):
        kz_j = torch.sqrt((omega/c[j])**2 - k**2 + 1e-12j)
        S = torch.sin(kz_j * dz)
        C = torch.cos(kz_j * dz)

        V0_new = C * V0 + (RHO_W_GLOBAL / kz_j) * S * V1
        V1_new = - (kz_j / RHO_W_GLOBAL) * S * V0 + C * V1
        V0, V1 = V0_new, V1_new

    kb_complex = (omega / cb) + 1j * ((ab * (freq/cb))/8.686)
    kz_b = torch.sqrt(kb_complex**2 - k**2 + 1e-12j)

    return 1j * rb * V1 + kz_b * V0

def extract_mode_shapes_diff(k, cb, rb, ab, freq):
    omega = 2 * math.pi * freq
    c, dz, N_layers = get_layered_profile(freq)

    V0 = torch.zeros_like(k, dtype=torch.complex128) 
    V1 = torch.ones_like(k, dtype=torch.complex128)  
    Psi_history = [V0]

    for j in range(N_layers):
        kz_j = torch.sqrt((omega/c[j])**2 - k**2 + 1e-12j)
        S = torch.sin(kz_j * dz)
        C = torch.cos(kz_j * dz)
        V0_new = C * V0 + (RHO_W_GLOBAL / kz_j) * S * V1
        V1_new = - (kz_j / RHO_W_GLOBAL) * S * V0 + C * V1
        V0, V1 = V0_new, V1_new
        Psi_history.append(V0)

    Psi_all = torch.stack(Psi_history)
    
    kb_complex = (omega / cb) + 1j * ((ab * (freq/cb))/8.686)
    kz_b = torch.sqrt(kb_complex**2 - k**2 + 1e-12j)
    
    Psi_sq = (Psi_all.real**2 + Psi_all.imag**2) / RHO_W_GLOBAL
    integral_water = (torch.sum(Psi_sq, dim=0) - 0.5 * Psi_sq[0, :] - 0.5 * Psi_sq[-1, :]) * dz
    
    V0_sq = V0.real**2 + V0.imag**2
    kz_b_imag_abs = torch.sqrt(kz_b.imag**2 + 1e-20)
    integral_bottom = V0_sq / (2 * rb * kz_b_imag_abs + 1e-20)
    
    norm_factor = torch.sqrt(integral_water + integral_bottom + 1e-20)
    Psi_all = Psi_all / norm_factor
    
    z_idx_s = int(round((ZS_GLOBAL / D_GLOBAL) * N_layers))
    z_idx_r = int(round((ZR_GLOBAL / D_GLOBAL) * N_layers))
    return Psi_all[z_idx_s, :], Psi_all[z_idx_r, :]

def unified_complex_solver(cb, rb, ab, freq, coarse_mode=False):
    omega = 2 * math.pi * freq
    k0 = omega / 1480.0  
    k_min = omega / 3000.0

    if coarse_mode:
        k_scan = torch.linspace(k0 * 0.999, k_min, 2000, device=device, dtype=torch.float64)
        with torch.no_grad():
            F_scan = (compute_F(k_scan, cb, rb, ab, freq) / 1j).real
        
        sign_change = (F_scan[:-1] * F_scan[1:]) < 0
        valid_indices = torch.nonzero(sign_change).flatten()
        if len(valid_indices) == 0: return torch.tensor([], device=device, dtype=torch.complex128)
        k = k_scan[valid_indices].to(dtype=torch.complex128)
        iters = 20
    else:
        real_vals = torch.linspace(k0*1.01, k_min, 400, device=device)
        imag_vals = torch.tensor([0.0, 1e-4, 1e-3], device=device)
        g_real, g_imag = torch.meshgrid(real_vals, imag_vals, indexing='ij')
        k = (g_real + 1j * g_imag).flatten()
        iters = 30

    for _ in range(iters):
        k = k.detach().requires_grad_(True)
        with torch.enable_grad():
            F = compute_F(k, cb, rb, ab, freq)
            dF_dk = torch.autograd.grad(F.real.sum(), k)[0].conj()
        k = k - F.detach() / (dF_dk.detach() + 1e-20)

    if coarse_mode:
        if len(k) > 1:
            idx = torch.argsort(k.real, descending=True)
            k = k[idx]
            unique = torch.cat([torch.tensor([True], device=device), torch.abs(k[1:] - k[:-1]) > 1e-5])
            k = k[unique]
    else:
        valid_mask = (k.real > k_min) & (k.real < k0*1.01) & (k.imag > -1e-8)
        k = k[valid_mask]
        if len(k) > 0:
            idx = torch.argsort(k.real, descending=True)
            k = k[idx]
            unique = torch.cat([torch.tensor([True], device=device), torch.abs(k[1:] - k[:-1]) > 1e-4])
            k = k[unique]
            
    return k

def get_full_complex_grad(F, x):
    dF_real = torch.autograd.grad(F.real.sum(), x, retain_graph=True)[0]
    dF_imag = torch.autograd.grad(F.imag.sum(), x, retain_graph=True)[0]
    return dF_real + 1j * dF_imag

class IFTSolver(torch.autograd.Function):
    @staticmethod
    def forward(ctx, cb, rb, ab, coarse_mode, freq):
        with torch.no_grad():
            k_star = unified_complex_solver(cb, rb, ab, freq, coarse_mode=coarse_mode)
        ctx.save_for_backward(k_star, cb, rb, ab)
        ctx.freq = freq
        return k_star

    @staticmethod
    def backward(ctx, grad_output):
        k_star, cb, rb, ab = ctx.saved_tensors
        freq = ctx.freq
        if len(k_star) == 0:
            return torch.zeros_like(cb), torch.zeros_like(rb), torch.zeros_like(ab), None, None
            
        with torch.enable_grad():
            k = k_star.detach().requires_grad_(True)
            cb_vec = cb.expand_as(k).clone().requires_grad_(True)
            rb_vec = rb.expand_as(k).clone().requires_grad_(True)
            ab_vec = ab.expand_as(k).clone().requires_grad_(True)

            F = compute_F(k, cb_vec, rb_vec, ab_vec, freq)

            dF_dk = torch.autograd.grad(F.real.sum(), k, retain_graph=True)[0].conj()
            dF_dcb = get_full_complex_grad(F, cb_vec)
            dF_drb = get_full_complex_grad(F, rb_vec)
            dF_dab = get_full_complex_grad(F, ab_vec)

        denom = dF_dk
        denom_mag = denom.abs()
        CLAMP_THRESHOLD = 1e4 
        is_stiff = denom_mag > CLAMP_THRESHOLD
        
        scale_factor = torch.ones_like(denom_mag)
        scale_factor[is_stiff] = CLAMP_THRESHOLD / denom_mag[is_stiff]
        denom_clamped = denom * scale_factor

        term = -grad_output.conj() / (denom_clamped + 1e-20)
        
        grad_cb = (term * dF_dcb).sum().real
        grad_rb = (term * dF_drb).sum().real
        grad_ab = (term * dF_dab).sum().real
        
        return grad_cb, grad_rb, grad_ab, None, None

def forward_model(cb, rb, ab, r_range, freq, stage=1):
    coarse = (stage == 1)
    k_complex = IFTSolver.apply(cb, rb, ab, coarse, freq)
    
    if len(k_complex) == 0: 
        return torch.ones_like(r_range)*100.0 + (cb*0+rb*0+ab*0)

    Zs, Zr = extract_mode_shapes_diff(k_complex, cb, rb, ab, freq)
    
    phase = k_complex[:, None] * r_range[None, :]
    H0 = torch.sqrt(2 / (math.pi * phase)) * torch.exp(1j * (phase - math.pi/4))
    
    p_sum = 1j/(4*RHO_W_GLOBAL) * (Zs[:, None] * Zr[:, None] * H0).sum(dim=0)
    
    p_sq = p_sum.real**2 + p_sum.imag**2
    return -10 * torch.log10(p_sq * (4.0 * math.pi)**2 + 1e-12)


#=============================================
# Safe root-finding version
#=============================================
def compute_F_terms(k, cb, rb, ab, freq):
    """
    Same as compute_F, but also returns the two terms of the bottom boundary
    equation, so we can compute a relative residual.

    F = term1 + term2
      = 1j * rb * V1 + kz_b * V0
    """
    omega = 2 * math.pi * freq
    c, dz, N_layers = get_layered_profile(freq)

    V0 = torch.zeros_like(k, dtype=torch.complex128)
    V1 = torch.ones_like(k, dtype=torch.complex128)

    for j in range(N_layers):
        kz_j = torch.sqrt((omega / c[j])**2 - k**2 + 1e-12j)
        S = torch.sin(kz_j * dz)
        C = torch.cos(kz_j * dz)

        V0_new = C * V0 + (RHO_W_GLOBAL / kz_j) * S * V1
        V1_new = -(kz_j / RHO_W_GLOBAL) * S * V0 + C * V1

        V0, V1 = V0_new, V1_new

    kb_complex = (omega / cb) + 1j * ((ab * (freq / cb)) / 8.686)
    kz_b = torch.sqrt(kb_complex**2 - k**2 + 1e-12j)

    term1 = 1j * rb * V1
    term2 = kz_b * V0
    F = term1 + term2

    return F, term1, term2


def compute_dF_dk(k, cb, rb, ab, freq):
    """
    Compute complex Newton derivative dF/dk.
    This follows your current convention.
    """
    k_req = k.detach().requires_grad_(True)

    with torch.enable_grad():
        F = compute_F(k_req, cb, rb, ab, freq)
        dF_dk = torch.autograd.grad(F.real.sum(), k_req)[0].conj()

    return F.detach(), dF_dk.detach()


def filter_roots(k, cb, rb, ab, freq,
                 relF_thr=1e-6,
                 delta_rel_thr=1e-7,
                 delta_abs_thr=1e-10,
                 imag_max=None):
    """
    Keep only candidates that look like true roots.
    """
    if len(k) == 0:
        return k

    F, dF_dk = compute_dF_dk(k, cb, rb, ab, freq)
    delta = F / (dF_dk + 1e-30)

    F2, term1, term2 = compute_F_terms(k, cb, rb, ab, freq)
    relF = torch.abs(F2) / (torch.abs(term1) + torch.abs(term2) + 1e-30)

    delta_ok = torch.abs(delta) < (delta_abs_thr + delta_rel_thr * torch.abs(k))
    relF_ok = relF < relF_thr

    keep = delta_ok & relF_ok

    if imag_max is not None:
        keep = keep & (k.imag < imag_max)

    return k[keep]

def unified_complex_solver_safe(cb, rb, ab, freq, coarse_mode=False):
    omega = 2 * math.pi * freq
    k0 = omega / 1480.0
    k_min = omega / 3000.0

    if coarse_mode:
        k_scan = torch.linspace(k0 * 0.999, k_min, 2000, device=device, dtype=torch.float64)

        with torch.no_grad():
            F_scan = (compute_F(k_scan, cb, rb, ab, freq) / 1j).real

        sign_change = (F_scan[:-1] * F_scan[1:]) < 0
        valid_indices = torch.nonzero(sign_change).flatten()

        if len(valid_indices) == 0:
            return torch.tensor([], device=device, dtype=torch.complex128)

        k = k_scan[valid_indices].to(dtype=torch.complex128)
        iters = 20

    else:
        real_vals = torch.linspace(k0 * 1.01, k_min, 400, device=device)
        imag_vals = torch.tensor([0.0, 1e-4, 1e-3], device=device)

        g_real, g_imag = torch.meshgrid(real_vals, imag_vals, indexing="ij")
        k = (g_real + 1j * g_imag).flatten()
        iters = 30

    # Newton refinement
    for _ in range(iters):
        F, dF_dk = compute_dF_dk(k, cb, rb, ab, freq)
        k = k - F / (dF_dk + 1e-30)

    # Basic physical region filter
    if coarse_mode:
        valid_mask = (k.real > k_min) & (k.real < k0 * 1.01) & (k.imag > -1e-8)
    else:
        valid_mask = (k.real > k_min) & (k.real < k0 * 1.01) & (k.imag > -1e-8)

    k = k[valid_mask]

    if len(k) == 0:
        return k

    # Sort by real part
    idx = torch.argsort(k.real, descending=True)
    k = k[idx]

    # Remove near-duplicates
    unique = torch.cat([
        torch.tensor([True], device=device),
        torch.abs(k[1:] - k[:-1]) > 1e-4
    ])
    k = k[unique]

    # Final polishing from unique candidates
    for _ in range(5):
        F, dF_dk = compute_dF_dk(k, cb, rb, ab, freq)
        k = k - F / (dF_dk + 1e-30)

    # Re-sort and de-duplicate after polishing
    idx = torch.argsort(k.real, descending=True)
    k = k[idx]

    unique = torch.cat([
        torch.tensor([True], device=device),
        torch.abs(k[1:] - k[:-1]) > 1e-5
    ])
    k = k[unique]

    # Strong root-quality filter
    k = filter_roots(
        k,
        cb,
        rb,
        ab,
        freq,
        relF_thr=1e-6,
        delta_rel_thr=1e-7,
        delta_abs_thr=1e-10,
        imag_max=0.1
    )

    # Final sort
    if len(k) > 0:
        idx = torch.argsort(k.real, descending=True)
        k = k[idx]

    return k

def root_quality_table(k, cb, rb, ab, freq, name="roots"):
    """
    Print root quality:
        relF  = |F| / (|term1| + |term2|)
        delta = |F / dFdk|
    """
    if len(k) == 0:
        print(f"{name}: no roots")
        return

    # compute F and dF/dk
    k_req = k.detach().clone().requires_grad_(True)

    with torch.enable_grad():
        F = compute_F(k_req, cb, rb, ab, freq)
        dF_dk = torch.autograd.grad(F.real.sum(), k_req)[0].conj()

    F_det = F.detach()
    dF_dk_det = dF_dk.detach()
    delta = torch.abs(F_det / (dF_dk_det + 1e-30))

    # compute relative F using term scale
    F2, term1, term2 = compute_F_terms(k.detach(), cb, rb, ab, freq)
    relF = torch.abs(F2) / (torch.abs(term1) + torch.abs(term2) + 1e-30)

    print(f"\n========== {name} ==========")
    print(f"{'m':>3s} | {'Re(k)':>14s} | {'Im(k)':>14s} | {'|F|':>12s} | {'relF':>12s} | {'|F/Fk|':>12s}")
    print("-" * 86)

    for m in range(len(k)):
        print(
            f"{m:3d} | "
            f"{k[m].real.item():14.8e} | "
            f"{k[m].imag.item():14.8e} | "
            f"{torch.abs(F2[m]).item():12.3e} | "
            f"{relF[m].item():12.3e} | "
            f"{delta[m].item():12.3e}"
        )

    print("-" * 86)
    print("max relF   =", relF.max().item())
    print("max |F/Fk| =", delta.max().item())
    
    
def forward_model_given_roots(k_complex, cb, rb, ab, r_range, freq):
    if len(k_complex) == 0:
        return torch.ones_like(r_range) * 100.0

    Zs, Zr = extract_mode_shapes_diff(k_complex, cb, rb, ab, freq)

    phase = k_complex[:, None] * r_range[None, :]
    H0 = torch.sqrt(2 / (math.pi * phase)) * torch.exp(1j * (phase - math.pi / 4))

    p_sum = 1j / (4 * RHO_W_GLOBAL) * (Zs[:, None] * Zr[:, None] * H0).sum(dim=0)

    p_sq = p_sum.real**2 + p_sum.imag**2
    TL = -10 * torch.log10(p_sq * (4.0 * math.pi)**2 + 1e-12)

    return TL

# =========================================================================
# 3. GROUND TRUTH CHECK & INVERSION
# =========================================================================
if __name__ == "__main__":
    print("\n=== GROUND TRUTH VALIDATION (MUST BE ~0.0) ===")
    with torch.no_grad():
        true_cb = torch.tensor(2026.0, device=device)
        true_rb = torch.tensor(1.3, device=device)
        true_ab = torch.tensor(0.8, device=device)
        
        gt_total_loss = 0.0
        for freq in FREQ_LIST:
            print(f'Processing {freq}')
            gt_pred = forward_model(true_cb, true_rb, true_ab, meas_data[freq]['r_meas'], freq, stage=3)
            gt_loss = torch.nn.functional.mse_loss(gt_pred, meas_data[freq]['TL_meas'])
            gt_total_loss += gt_loss
            print(f'Processed {freq}')
            
        print(f"Average Loss at Ground Truth across all freqs: {gt_total_loss.item() / len(FREQ_LIST):.8f}\n")
    
    print("=== STARTING 3-STAGE MULTI-FREQUENCY INVERSION ===")
    cb_opt = torch.tensor(1950.0, device=device, requires_grad=True)
    rb_opt = torch.tensor(1.2, device=device, requires_grad=True)
    ab_opt = torch.tensor(0.1, device=device, requires_grad=True)
    
    optimizer = torch.optim.Rprop([
        {'params': cb_opt, 'lr': 1.0},
        {'params': rb_opt, 'lr': 0.01},
        {'params': ab_opt, 'lr': 0.01}
    ])
    
    print(">>> STAGE 1: Ghost Assisted (Basin Finding)")
    for i in range(100):
        optimizer.zero_grad()
        
        total_loss = 0.0
        for freq in FREQ_LIST:
            pred = forward_model(cb_opt, rb_opt, ab_opt, meas_data[freq]['r_meas'], freq, stage=1)
            total_loss += torch.nn.functional.mse_loss(pred, meas_data[freq]['TL_meas'])
        
        # Average the loss so the gradient scales remain consistent with the single-frequency logic
        avg_loss = total_loss / len(FREQ_LIST)
        avg_loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            cb_opt.clamp_(1550.0, 2500.0)
            rb_opt.clamp_(1.0, 3.0)
            ab_opt.clamp_(0.01, 5.0) 
        
        if i % 1 == 0:
            print(f"Iter {i:04d} | Avg Loss: {avg_loss.item():.4f} | cb: {cb_opt.item():.1f} | rho: {rb_opt.item():.3f} | alpha: {ab_opt.item():.3f}")
    
    print("\n>>> STAGE 2: Bridge (Clean Trapped)")
    for param_group in optimizer.param_groups: param_group['lr'] *= 2.0  
    
    for i in range(300):
        optimizer.zero_grad()
        
        total_loss = 0.0
        for freq in FREQ_LIST:
            pred = forward_model(cb_opt, rb_opt, ab_opt, meas_data[freq]['r_meas'], freq, stage=2)
            total_loss += torch.nn.functional.mse_loss(pred, meas_data[freq]['TL_meas'])
            
        avg_loss = total_loss / len(FREQ_LIST)
        avg_loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            cb_opt.clamp_(1550.0, 2500.0); rb_opt.clamp_(1.0, 3.0); ab_opt.clamp_(0.01, 5.0)
            
        if i % 1 == 0:
            print(f"Iter {i:04d} | Avg Loss: {avg_loss.item():.4f} | cb: {cb_opt.item():.1f} | rho: {rb_opt.item():.3f} | alpha: {ab_opt.item():.3f}")
    
    print("\n>>> STAGE 3: Full Physics (Final Polish)")
    for param_group in optimizer.param_groups: param_group['lr'] = 0.05  
    
    for i in range(300):
        optimizer.zero_grad()
        
        total_loss = 0.0
        for freq in FREQ_LIST:
            pred = forward_model(cb_opt, rb_opt, ab_opt, meas_data[freq]['r_meas'], freq, stage=3)
            total_loss += torch.nn.functional.mse_loss(pred, meas_data[freq]['TL_meas'])
            
        avg_loss = total_loss / len(FREQ_LIST)
        avg_loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            cb_opt.clamp_(1550.0, 2500.0); rb_opt.clamp_(1.0, 3.0); ab_opt.clamp_(0.01, 5.0)
            
        if i % 50 == 0:
            print(f"S3 Iter {i:04d} | Avg Loss: {avg_loss.item():.4f} | cb: {cb_opt.item():.2f} | rho: {rb_opt.item():.3f} | alpha: {ab_opt.item():.3f}")
    
    print(f"\nFINAL: cb={cb_opt.item():.2f}, rho={rb_opt.item():.3f}, alpha={ab_opt.item():.3f}")


