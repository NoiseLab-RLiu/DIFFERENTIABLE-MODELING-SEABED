import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

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

# Globals (Constants for the Forward Model)
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
    
    depth_idx = 50 
    mask = (r_full >= 1000.0) & (r_full <= 5000.0)
    r_meas = r_full[mask]
    TL_meas = TL_tensor[depth_idx, mask]
    
    print(f"[OK] Data Loaded. Points: {len(r_meas)}")

except Exception as e:
    print(f"[ERROR] {e}")
    exit()

# =========================================================================
# 2. PHYSICS ENGINE (Mesh Scan Solver)
# =========================================================================
def unified_complex_solver(c_b, rho_b, alpha_b):
    c_w, rho_w, D, omega = C_W_GLOBAL, RHO_W_GLOBAL, D_GLOBAL, OMEGA_GLOBAL
    
    k0 = omega / c_w
    kb_complex = (omega / c_b) + 1j * (alpha_b * (FREQ_GLOBAL / c_b) / 8.686)
    
    # 1. DENSE MESH SCAN
    k_min = (omega / 3000.0) 
    k_scan = torch.linspace(k0 * 0.999, k_min, 1000, device=device, dtype=torch.float64)
    
    kz_w = torch.sqrt(torch.clamp(k0**2 - k_scan**2, min=1e-12))
    kz_b = torch.sqrt(torch.clamp(k_scan**2 - (omega/c_b)**2, min=1e-12))
    
    tan_g = torch.tan(kz_w * D)
    F = tan_g + (rho_b/rho_w) * (kz_w / kz_b)
    
    sign_change = (F[:-1] * F[1:]) < 0
    jump_filter = torch.abs(F[:-1] - F[1:]) < 50.0 
    valid_indices = torch.nonzero(sign_change & jump_filter).flatten()
    
    if len(valid_indices) == 0:
        return torch.tensor([]).to(device)
        
    k_guess = k_scan[valid_indices].to(dtype=torch.complex128)
    
    # 2. NEWTON REFINEMENT
    k = k_guess.clone()
    for _ in range(20): 
        kz_w = torch.sqrt(k0**2 - k**2)
        kz_b = torch.sqrt(kb_complex**2 - k**2)
        
        s_w, c_w = torch.sin(kz_w*D), torch.cos(kz_w*D)
        F = rho_w * kz_b * s_w + 1j * rho_b * kz_w * c_w
        
        dkzw_dk = -k / (kz_w + 1e-20)
        dkzb_dk = -k / (kz_b + 1e-20)
        dT1 = rho_w * (dkzb_dk * s_w + kz_b * c_w * D * dkzw_dk)
        dT2 = 1j * rho_b * (dkzw_dk * c_w - kz_w * s_w * D * dkzw_dk)
        
        k = k - F / (dT1 + dT2 + 1e-20)
        
    k_real = k.real
    if len(k) > 1:
        idx = torch.argsort(k_real, descending=True)
        k = k[idx]
        diff = torch.abs(k[1:] - k[:-1])
        unique = torch.cat([torch.tensor([True], device=device), diff > 1e-5])
        k = k[unique]

    return k

def forward_model(c_b, rho_b, alpha_b, r_range):
    k_complex = unified_complex_solver(c_b, rho_b, alpha_b)
    
    if len(k_complex) == 0: 
        dummy_grad = c_b*0 + rho_b*0 + alpha_b*0
        return torch.ones_like(r_range)*100.0 + dummy_grad

    k0 = OMEGA_GLOBAL / C_W_GLOBAL
    kz_w = torch.sqrt(k0**2 - k_complex**2)
    
    alpha_neper = (alpha_b * (FREQ_GLOBAL / c_b)) / 8.686
    kb_complex = (OMEGA_GLOBAL / c_b) + 1j * alpha_neper
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
# 3. SIMULTANEOUS MSE INVERSION
# =========================================================================
print("\n=== STARTING SIMULTANEOUS MSE INVERSION ===")

# Starting points
cb_opt = torch.tensor(1850.0, device=device, requires_grad=True)
rb_opt = torch.tensor(1.2, device=device, requires_grad=True)
ab_opt = torch.tensor(0.1, device=device, requires_grad=True)

# Using Rprop for all 3 parameters simultaneously
optimizer = torch.optim.Rprop([cb_opt, rb_opt, ab_opt], lr=0.01)
# optimizer = torch.optim.Adam([
#     {'params': cb_opt, 'lr': 1e-2}, 
#     {'params': rb_opt, 'lr': 1e-2},
#     {'params': ab_opt, 'lr': 1e-2}   # FROZEN INITIALLY!
# ], eps=1e-16)

loss_history = []
param_history = {'cb': [], 'rb': [], 'ab': []}

for i in range(200000):
    optimizer.zero_grad()
    
    pred = forward_model(cb_opt, rb_opt, ab_opt, r_meas)
    loss = torch.nn.functional.mse_loss(pred, TL_meas)
    
    loss.backward()
    optimizer.step()
    
    # Constraints
    with torch.no_grad():
        cb_opt.clamp_(1550.0, 2500.0)
        rb_opt.clamp_(1.0, 3.0)
        ab_opt.clamp_(0.0, 5.0)
        
    loss_history.append(loss.item())
    param_history['cb'].append(cb_opt.item())
    param_history['rb'].append(rb_opt.item())
    param_history['ab'].append(ab_opt.item())
    
    if i % 100 == 0:
        print(f"Iter {i:04d} | Loss: {loss.item():.4f} | cb: {cb_opt.item():.1f} | rho: {rb_opt.item():.3f} | alpha: {ab_opt.item():.3f}")

# =========================================================================
# 4. RESULTS
# =========================================================================
print("\n=== FINAL RECOVERY ===")
print(f"c_b   : {cb_opt.item():.2f} (True: 2000.0)")
print(f"rho_b : {rb_opt.item():.3f} (True: 1.8)")
print(f"alpha : {ab_opt.item():.3f} (True: 0.5)")

with torch.no_grad():
    final_pred = forward_model(cb_opt, rb_opt, ab_opt, r_meas).cpu().numpy()
    meas_tl = TL_meas.cpu().numpy()
    r_plot = r_meas.cpu().numpy()

plt.figure(figsize=(10, 8))
plt.subplot(2, 1, 1)
plt.plot(param_history['cb'], label='Recovered cb', color='blue')
plt.axhline(2000, color='blue', linestyle='--', label='True cb')
plt.title('Simultaneous Parameter Convergence')
plt.ylabel('Sound Speed (m/s)')
plt.legend()
plt.grid()

ax2 = plt.gca().twinx()
ax2.plot(param_history['rb'], label='Recovered rho', color='red')
ax2.plot(param_history['ab'], label='Recovered alpha', color='green')
ax2.axhline(1.8, color='red', linestyle='--', alpha=0.5)
ax2.axhline(0.5, color='green', linestyle='--', alpha=0.5)
ax2.set_ylabel('Density / Attenuation')
ax2.legend(loc='lower right')

plt.subplot(2, 1, 2)
plt.plot(r_plot, meas_tl, 'k', label='Ground Truth (Kraken)', alpha=0.6)
plt.plot(r_plot, final_pred, 'r--', label='Model Prediction')
plt.title(f'Final TL Match (MSE: {loss_history[-1]:.4f})')
plt.xlabel('Range (m)')
plt.ylabel('TL (dB)')
plt.gca().invert_yaxis()
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()