import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" # Prevent OMP crash

import math
import torch
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
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
# 2. COMPLEX SOLVER
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
    for _ in range(30): 
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
# 3. HELPER: CORRELATION LOSS
# =========================================================================
def correlation_loss(pred, target):
    """
    Computes 1 - Pearson Correlation Coefficient.
    Invariant to constant offsets (attenuation errors).
    Ideal for finding phase/frequency matches (c_b).
    """
    # Remove mean (center the signals)
    p_centered = pred - pred.mean()
    t_centered = target - target.mean()
    
    # Normalize
    p_norm = p_centered / (torch.norm(p_centered) + 1e-12)
    t_norm = t_centered / (torch.norm(t_centered) + 1e-12)
    
    # Dot product is the correlation
    correlation = torch.sum(p_norm * t_norm)
    
    # We want to maximize correlation, so minimize 1 - corr
    return 1.0 - correlation

# =========================================================================
# 4. INVERSION
# =========================================================================
print("\n=== STARTING PATTERN-MATCHING INVERSION ===")

# Start far away
cb_opt = torch.tensor(1850.0, device=device, requires_grad=True) # Truth: 2000
rb_opt = torch.tensor(1.2, device=device, requires_grad=True)    # Truth: 1.8
ab_opt = torch.tensor(0.1, device=device, requires_grad=True)    # Truth: 0.5

loss_history = []
param_history = {'cb': [], 'rb': [], 'ab': []}

# --- STAGE 1: C_B using CORRELATION ---
print(">>> STAGE 1: Recovering c_b (Pattern Matching)")
# We use Rprop with a generous step size to jump out of local minima
optimizer = torch.optim.Rprop([cb_opt], lr=0.05, step_sizes=(1e-6, 20.0))

for i in range(300):
    optimizer.zero_grad()
    pred = forward_model(cb_opt, rb_opt.detach(), ab_opt.detach(), r_meas)
    
    # CRITICAL CHANGE: Use Correlation Loss
    loss = correlation_loss(pred, TL_meas)
    
    loss.backward()
    optimizer.step()
    
    with torch.no_grad(): cb_opt.clamp_(1550.0, 2500.0)
    
    loss_history.append(loss.item())
    param_history['cb'].append(cb_opt.item())
    param_history['rb'].append(rb_opt.item())
    param_history['ab'].append(ab_opt.item())
    
    if i%50==0: print(f"Iter {i:03d}: Corr Loss={loss.item():.4f} | cb={cb_opt.item():.1f}")

# --- STAGE 2: RHO_B using MIXED LOSS ---
# Now that c_b is locked, the wiggles align. 
# We introduce Density, which affects null depth.
print(">>> STAGE 2: Recovering rho_b (Mixed MSE/Corr)")
optimizer = torch.optim.Rprop([cb_opt, rb_opt], lr=0.01, step_sizes=(1e-6, 5.0))

for i in range(300, 600):
    optimizer.zero_grad()
    pred = forward_model(cb_opt, rb_opt, ab_opt.detach(), r_meas)
    
    # Mix: 50% Shape (Corr), 50% Magnitude (MSE)
    # Note: Normalized MSE to keep scale similar to Corr
    mse = (pred - TL_meas).pow(2).mean() / 100.0
    corr = correlation_loss(pred, TL_meas)
    loss = corr + mse
    
    loss.backward()
    optimizer.step()
    
    with torch.no_grad(): 
        cb_opt.clamp_(1550.0, 2500.0)
        rb_opt.clamp_(1.05, 3.0)
        
    loss_history.append(loss.item())
    param_history['cb'].append(cb_opt.item())
    param_history['rb'].append(rb_opt.item())
    param_history['ab'].append(ab_opt.item())
    
    if i%50==0: print(f"Iter {i:03d}: Mix Loss={loss.item():.4f} | cb={cb_opt.item():.1f} | rho={rb_opt.item():.3f}")

# --- STAGE 3: ALPHA_B using MSE ---
# Attenuation controls the global slope/offset. MSE is the perfect metric here.
print(">>> STAGE 3: Recovering alpha_b (MSE)")
optimizer = torch.optim.Rprop([cb_opt, rb_opt, ab_opt], lr=0.005, step_sizes=(1e-6, 1.0))

for i in range(600, 150000):
    optimizer.zero_grad()
    pred = forward_model(cb_opt, rb_opt, ab_opt, r_meas)
    
    # Pure MSE to fix the offset/slope
    loss = (pred - TL_meas).pow(2).mean()
    
    loss.backward()
    optimizer.step()
    
    with torch.no_grad(): 
        cb_opt.clamp_(1550.0, 2500.0)
        rb_opt.clamp_(1.05, 3.0)
        ab_opt.clamp_(0.0, 5.0)
        
    loss_history.append(loss.item())
    param_history['cb'].append(cb_opt.item())
    param_history['rb'].append(rb_opt.item())
    param_history['ab'].append(ab_opt.item())
    
    if i%100==0: print(f"Iter {i:04d}: MSE Loss={loss.item():.4f} | cb={cb_opt.item():.1f} | rho={rb_opt.item():.3f} | alpha={ab_opt.item():.3f}")

# =========================================================================
# 5. RESULTS
# =========================================================================
print("\n=== FINAL RECOVERY ===")
print(f"c_b   : {cb_opt.item():.2f}")
print(f"rho_b : {rb_opt.item():.3f}")
print(f"alpha : {ab_opt.item():.3f}")

# Plotting
with torch.no_grad():
    pred_tl = forward_model(cb_opt, rb_opt, ab_opt, r_meas).cpu().numpy()
    meas_tl = TL_meas.cpu().numpy()
    r_plot = r_meas.cpu().numpy()

plt.figure(figsize=(10, 8))

# Subplot 1: Convergence
plt.subplot(2, 1, 1)
plt.plot(param_history['cb'], 'b', label='c_b')
plt.axhline(2000, color='b', linestyle='--', alpha=0.3)
plt.ylabel('Speed')
plt.legend(loc='upper left')
plt.grid(True)

ax2 = plt.gca().twinx()
ax2.plot(param_history['rb'], 'r', label='rho')
ax2.plot(param_history['ab'], 'g', label='alpha')
ax2.axhline(1.8, color='r', linestyle='--', alpha=0.3)
ax2.axhline(0.5, color='g', linestyle='--', alpha=0.3)
ax2.set_ylabel('Rho / Alpha')
ax2.legend(loc='lower left')

# Subplot 2: Fit
plt.subplot(2, 1, 2)
plt.plot(r_plot, meas_tl, 'k', label='Truth', alpha=0.6)
plt.plot(r_plot, pred_tl, 'r--', label='Model')
plt.title(f'Final Match (Alpha={ab_opt.item():.2f})')
plt.xlabel('Range (m)')
plt.ylabel('TL (dB)')
plt.gca().invert_yaxis()
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()