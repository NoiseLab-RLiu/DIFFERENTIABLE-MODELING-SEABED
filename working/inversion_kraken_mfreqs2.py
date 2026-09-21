import torch
import math
import scipy.io as sio
import os

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

# Frequency List (Must match your MATLAB generation)
# High frequencies (200, 250) are crucial for Alpha recovery
FREQ_LIST = [30.0, 35.0, 50.0, 100.0, 150.0, 200, 250]

# =========================================================================
# 1. DATA LOADING (External KRAKEN Files)
# =========================================================================
print("--- LOADING KRAKEN DATA ---")
TL_meas_dict = {}
r_meas = None

for f in FREQ_LIST:
    # File naming convention from your MATLAB script
    mat_file = f'TL_kraken_2026_260122_{int(f)}.mat'
    
    # Fallback for 150Hz if using old file name
    # if f == 150.0 and not os.path.exists(mat_file):
    #     if os.path.exists('TL_kraken2.mat'): mat_file = 'TL_kraken2.mat'

    if not os.path.exists(mat_file):
        print(f"[WARN] File {mat_file} not found. Skipping {f} Hz.")
        continue

    try:
        mat_data = sio.loadmat(mat_file)
        TL_tensor = torch.tensor(mat_data['TL_2D'], device=device)
        
        # Define Range Axis (Once)
        if r_meas is None:
            r_full = torch.linspace(1.0, 5000.0, TL_tensor.shape[1], device=device)
            mask = (r_full >= 1000.0) & (r_full <= 5000.0)
            r_meas = r_full[mask]
        
        # Extract Depth 50 (Index 50 usually, check MATLAB script indexing)
        # Assuming MATLAB 'z_vec' had 101 points (0..100), index 50 is z=50m
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
def unified_complex_solver(c_b, rho_b, alpha_b, omega, freq, coarse_mode=False):
    c_w, rho_w, D = C_W_GLOBAL, RHO_W_GLOBAL, D_GLOBAL
    k0 = omega / c_w
    k_min = omega / 3000.0
    
    # Complex Bottom Wavenumber
    # Clamp alpha to > 1e-4 to ensure complex solver stability
    alpha_safe = torch.clamp(alpha_b, min=1e-4)
    kb_complex = (omega / c_b) + 1j * ((alpha_safe * (freq/c_b))/8.686)

    if coarse_mode:
        # --- MODE A: REAL SCAN (Stage 1) ---
        k_scan = torch.linspace(k0 * 0.999, k_min, 1000, device=device, dtype=torch.float64)
        
        kz_w = torch.sqrt(torch.clamp(k0**2 - k_scan**2, min=1e-12))
        kz_b = torch.sqrt(torch.clamp(k_scan**2 - (omega/c_b)**2, min=1e-12))
        
        # Safe Characteristic Function (Sine/Cosine form)
        s_w, c_w_val = torch.sin(kz_w * D), torch.cos(kz_w * D)
        F_scan = rho_w * kz_b * s_w + rho_b * kz_w * c_w_val
        
        sign_change = (F_scan[:-1] * F_scan[1:]) < 0
        jump_filter = torch.abs(F_scan[:-1] - F_scan[1:]) < 500.0 
        valid_indices = torch.nonzero(sign_change & jump_filter).flatten()
        
        if len(valid_indices) == 0: return torch.tensor([]).to(device)
        k = k_scan[valid_indices].to(dtype=torch.complex128)
        iters = 20
    else:
        # --- MODE B: COMPLEX GRID (Stage 2) ---
        real_vals = torch.linspace(k0*1.01, k_min, 400, device=device)
        imag_vals = torch.tensor([0.0, 1e-4, 5e-4], device=device) 
        g_real, g_imag = torch.meshgrid(real_vals, imag_vals, indexing='ij')
        k = (g_real + 1j * g_imag).flatten()
        iters = 30

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
    if coarse_mode:
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
    def forward(ctx, cb, rho_b, alpha_b, omega, freq, coarse_mode):
        with torch.no_grad():
            k_star = unified_complex_solver(cb, rho_b, alpha_b, omega, freq, coarse_mode=coarse_mode)
        ctx.save_for_backward(k_star, cb, rho_b, alpha_b)
        ctx.consts = (omega, freq)
        return k_star

    @staticmethod
    def backward(ctx, grad_output):
        k_star, cb, rb, ab = ctx.saved_tensors
        omega, freq = ctx.consts
        c_w, rho_w, D = C_W_GLOBAL, RHO_W_GLOBAL, D_GLOBAL
        
        with torch.enable_grad():
            k = k_star.detach().requires_grad_(True)
            cb_vec = cb.expand_as(k).clone().requires_grad_(True)
            rb_vec = rb.expand_as(k).clone().requires_grad_(True)
            ab_vec = ab.expand_as(k).clone().requires_grad_(True)
            
            # Reconstruct F
            k0 = omega / c_w
            kb_complex = (omega / cb_vec) + 1j * ((ab_vec * (freq/cb_vec))/8.686)
            kz_w = torch.sqrt(k0**2 - k**2)
            kz_b = torch.sqrt(kb_complex**2 - k**2)
            s_w, c_w_ = torch.sin(kz_w*D), torch.cos(kz_w*D)
            F = rho_w * kz_b * s_w + 1j * rb_vec * kz_w * c_w_
            
            dF_dk = torch.autograd.grad(F, k, torch.ones_like(F), create_graph=False, retain_graph=True)[0]
            dF_dcb = torch.autograd.grad(F, cb_vec, torch.ones_like(F), create_graph=False, retain_graph=True)[0]
            dF_drb = torch.autograd.grad(F, rb_vec, torch.ones_like(F), create_graph=False, retain_graph=True)[0]
            dF_dab = torch.autograd.grad(F, ab_vec, torch.ones_like(F), create_graph=False, retain_graph=False)[0]

        # Clamping Fix for Leaky Modes
        denom = dF_dk
        denom_mag = denom.abs()
        CLAMP_THRESHOLD = 1e4 
        is_stiff = denom_mag > CLAMP_THRESHOLD
        scale_factor = torch.ones_like(denom_mag)
        scale_factor[is_stiff] = CLAMP_THRESHOLD / denom_mag[is_stiff]
        denom_clamped = denom * scale_factor

        term = -grad_output / (denom_clamped + 1e-20)
        
        grad_cb = (term * dF_dcb).sum().real
        grad_rb = (term * dF_drb).sum().real
        grad_ab = (term * dF_dab).sum().real
        
        return grad_cb, grad_rb, grad_ab, None, None, None

def forward_model(c_b, rho_b, alpha_b, r_range, omega, freq, stage=1):
    if stage == 1:
        # Stage 1: Coarse Mode (Real Scan)
        k_complex = unified_complex_solver(c_b, rho_b, alpha_b, omega, freq, coarse_mode=True)
    else:
        # Stage 2: Complex Mode (Trapped + Leaky) - Clamped IFT
        k_complex = IFTSolver.apply(c_b, rho_b, alpha_b, omega, freq, False)
    
    if len(k_complex) == 0: 
        return torch.ones_like(r_range)*100.0 + (c_b*0+rho_b*0+alpha_b*0)

    k0 = omega / C_W_GLOBAL

    # Filters
    if stage == 2:
        # Remove Ghost Artifacts (Singularity)
        is_ghost = torch.abs(k_complex.real - k0) < 2e-4
        k_complex = k_complex[~is_ghost]
    # Stage 1: Raw output

    if len(k_complex) == 0:
        return torch.ones_like(r_range)*100.0 + (c_b*0+rho_b*0+alpha_b*0)

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
    
    return -20 * torch.log10(p_sum.abs() * 4.0 * math.pi + 1e-12)

# =========================================================================
# 3. OPTIMIZATION LOOP
# =========================================================================
print("\n=== STARTING 2-STAGE MULTI-FREQ INVERSION ===")

# Start parameters (Within basin of 2000 +/- 50 is safe)
cb_opt = torch.tensor(1950.0, device=device, requires_grad=True)
rb_opt = torch.tensor(1.2, device=device, requires_grad=True)
ab_opt = torch.tensor(0.1, device=device, requires_grad=True)

# Precompute available freqs/omegas
active_freqs = [f for f in FREQ_LIST if f in TL_meas_dict]
active_omegas = [2 * math.pi * f for f in active_freqs]

cb_hist = []
rhob_hist = []
alpha_hist = []

# -------------------------------------------------------------------------
# STAGE 1: BASIN FINDING (All Freqs, Equal Weight)
# -------------------------------------------------------------------------
print(f">>> STAGE 1: Basin Finding (Freqs: {active_freqs})")
optimizer = torch.optim.Rprop([cb_opt, rb_opt, ab_opt], lr=0.005)

for i in range(100):
    optimizer.zero_grad()
    total_loss = 0.0
    
    for freq, omega in zip(active_freqs, active_omegas):
        # Stage 1: Coarse Mode
        pred = forward_model(cb_opt, rb_opt, ab_opt, r_meas, omega, freq, stage=1)
        total_loss += torch.nn.functional.mse_loss(pred, TL_meas_dict[freq])
    
    total_loss.backward()
    optimizer.step()
    
    with torch.no_grad():
        cb_opt.clamp_(1550, 2500); rb_opt.clamp_(1.0, 3.0); ab_opt.clamp_(0.01, 5.0)
        cb_hist.append(cb_opt.item())
        rhob_hist.append(rb_opt.item())
        alpha_hist.append(ab_opt.item())
    
    if i % 50 == 0:
        print(f"Iter {i:04d} | SumLoss: {total_loss.item():.4f} | cb: {cb_opt.item():.1f} | rho: {rb_opt.item():.3f} | alpha: {ab_opt.item():.3f}")

# -------------------------------------------------------------------------
# STAGE 2: ALPHA REFINEMENT (Differential LR)
# -------------------------------------------------------------------------
print(f"\n>>> STAGE 2: Alpha Refinement (High LR on Alpha)")
# optimizer = torch.optim.Rprop([
#     {'params': [cb_opt, rb_opt], 'lr': 0}, # Freeze/Slow Phase Params
#     {'params': [ab_opt],         'lr': 0.05}   # Aggressive Alpha Update
# ])
optimizer = torch.optim.Rprop([ab_opt], lr=0.05)

for i in range(400):
    optimizer.zero_grad()
    total_loss = 0.0
    
    for freq, omega in zip(active_freqs, active_omegas):
        # Stage 2: Complex IFT (Trapped + Leaky)
        pred = forward_model(cb_opt, rb_opt, ab_opt, r_meas, omega, freq, stage=2)
        total_loss += torch.nn.functional.mse_loss(pred, TL_meas_dict[freq])
    
    total_loss.backward()
    optimizer.step()
    
    with torch.no_grad():
        cb_opt.clamp_(1550, 2500); rb_opt.clamp_(1.0, 3.0); ab_opt.clamp_(0.01, 5.0)
        cb_hist.append(cb_opt.item())
        rhob_hist.append(rb_opt.item())
        alpha_hist.append(ab_opt.item())
        
    if i % 50 == 0:
        print(f"Iter {i:04d} | SumLoss: {total_loss.item():.4f} | cb: {cb_opt.item():.1f} | rho: {rb_opt.item():.3f} | alpha: {ab_opt.item():.3f}")

print(f"\nFINAL RESULT: cb={cb_opt.item():.2f}, rho={rb_opt.item():.3f}, alpha={ab_opt.item():.3f}")

#%%
import os
# --- CRITICAL FIX: PREVENT KERNEL CRASH (OMP ERROR) ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# CONFIGURATION
# ==========================================
# True values (Update these if your MATLAB generation used different ones)
TRUE_CB = 1910.0
TRUE_RHO = 1.65
TRUE_ALPHA = 0.45 

TRUE_CB = 2000
TRUE_RHO = 1.80
TRUE_ALPHA = 0.5 

# % Physics Parameters (Dataset 2 Target)
# D     = 100.0;  
# cw    = 1500.0; 
# cb    = 2026;%2000;%1910.0;   % Target CB
# rhob  = 1.3;%1.8;%1.65;     % Target Rho
# rhow  = 1.0;    
# alphab= 0.8;%0.5;%0.45;      % Target Alpha
# zs    = 25.0;   

# Set plot style globally
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'lines.linewidth': 2.5,
    'figure.figsize': (8, 6),
    'lines.markersize': 8
})

# ==========================================
# PLOT 1: SOUND SPEED
# ==========================================
plt.figure()
iterations = np.arange(len(cb_hist))
plt.plot(iterations, cb_hist, label='Estimate', color='#0072BD', linewidth=3)
plt.axhline(y=TRUE_CB, color='#D95319', linestyle='--', linewidth=3, label='Ground Truth')

plt.title(r'Sound Speed $c_b$ Optimization')
plt.xlabel('Iteration')
plt.ylabel('Sound Speed (m/s)')
plt.grid(True, which='both', linestyle=':', linewidth=0.5)
plt.legend(loc='best', frameon=True, fancybox=True, framealpha=0.9)
plt.tight_layout()
plt.show()

# ==========================================
# PLOT 2: DENSITY
# ==========================================
plt.figure()
iterations = np.arange(len(rhob_hist))
plt.plot(iterations, rhob_hist, label='Estimate', color='#0072BD', linewidth=3)
plt.axhline(y=TRUE_RHO, color='#D95319', linestyle='--', linewidth=3, label='Ground Truth')

plt.title(r'Density $\rho_b$ Optimization')
plt.xlabel('Iteration')
plt.ylabel(r'Density (g/cm$^3$)')
plt.grid(True, which='both', linestyle=':', linewidth=0.5)
plt.legend(loc='best', frameon=True, fancybox=True, framealpha=0.9)
plt.tight_layout()
plt.show()

# ==========================================
# PLOT 3: ATTENUATION
# ==========================================
plt.figure()
iterations = np.arange(len(alpha_hist))
plt.plot(iterations, alpha_hist, label='Estimate', color='#0072BD', linewidth=3)
plt.axhline(y=TRUE_ALPHA, color='#D95319', linestyle='--', linewidth=3, label='Ground Truth')

plt.title(r'Attenuation $\alpha_b$ Optimization')
plt.xlabel('Iteration')
# Using raw string r'' to fix the SyntaxWarning with backslash
plt.ylabel(r'Attenuation (dB/$\lambda$)') 
plt.grid(True, which='both', linestyle=':', linewidth=0.5)
plt.legend(loc='best', frameon=True, fancybox=True, framealpha=0.9)
plt.tight_layout()
plt.show()

#%%
import os
# --- CRITICAL FIX: PREVENT KERNEL CRASH (OMP ERROR) ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
# Ground Truths for Exp 1, 2, 3
TRUE_CB    = [1910.0, 2000.0, 2026.0]
TRUE_RHO   = [1.65, 1.8, 1.3]
TRUE_ALPHA = [0.45, 0.5, 0.8]

# Labels & Colors
EXP_LABELS = ['Exp 1', 'Exp 2', 'Exp 3']
COLORS = ['#0072BD', '#D95319', '#77AC30']

# Truncation Limit
LIMIT_EPOCH = 150

# Plot Styling
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 12,
    'lines.linewidth': 2.5,
    'figure.figsize': (10, 6), # Wider figure to accommodate outside legend
    'lines.markersize': 8
})

iterations = np.arange(LIMIT_EPOCH)

# ==========================================
# 2. PLOTTING SCRIPT
# ==========================================

# --- PLOT 1: SOUND SPEED (cb) ---
plt.figure()
for i in range(3):
    # Truncate to LIMIT_EPOCH
    traj = np.array(cb_H[i])[:LIMIT_EPOCH]
    
    # Plot Trajectory (Solid)
    plt.plot(iterations, traj, color=COLORS[i], linewidth=2.5, 
             label=f'{EXP_LABELS[i]} (Est)')
    
    # Plot Ground Truth (Dashed)
    plt.axhline(y=TRUE_CB[i], color=COLORS[i], linestyle='--', linewidth=2.0, alpha=0.8,
                label=f'{EXP_LABELS[i]} (True: {int(TRUE_CB[i])})')

plt.title(r'Sound Speed $c_{\mathrm{b}}$ Optimization')
plt.xlabel('Epoch')
plt.ylabel('Sound Speed (m/s)')
plt.ylim([1900, 2050]) # Fixed Range
plt.grid(True, which='both', linestyle=':', linewidth=0.5)

# Legend Outside
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., frameon=True)
plt.subplots_adjust(right=0.7) # Adjust layout so legend fits
plt.show()


# --- PLOT 2: DENSITY (rho) ---
plt.figure()
for i in range(3):
    traj = np.array(rb_H[i])[:LIMIT_EPOCH]
    
    plt.plot(iterations, traj, color=COLORS[i], linewidth=2.5, 
             label=f'{EXP_LABELS[i]} (Est)')
    
    plt.axhline(y=TRUE_RHO[i], color=COLORS[i], linestyle='--', linewidth=2.0, alpha=0.8,
                label=f'{EXP_LABELS[i]} (True: {TRUE_RHO[i]})')

plt.title(r'Density $\rho_{\mathrm{b}}$ Optimization')
plt.xlabel('Epoch')
plt.ylabel(r'Density (g/cm$^3$)')
plt.ylim([1, 2]) # Fixed Range
plt.grid(True, which='both', linestyle=':', linewidth=0.5)

# Legend Outside
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., frameon=True)
plt.subplots_adjust(right=0.7)
plt.show()


# --- PLOT 3: ATTENUATION (alpha) ---
plt.figure()
for i in range(3):
    traj = np.array(ab_H[i])[:LIMIT_EPOCH]
    
    plt.plot(iterations, traj, color=COLORS[i], linewidth=2.5, 
             label=f'{EXP_LABELS[i]} (Est)')
    
    plt.axhline(y=TRUE_ALPHA[i], color=COLORS[i], linestyle='--', linewidth=2.0, alpha=0.8,
                label=f'{EXP_LABELS[i]} (True: {TRUE_ALPHA[i]})')

plt.title(r'Attenuation $\alpha_{\mathrm{b}}$ Optimization')
plt.xlabel('Epoch')
plt.ylabel(r'Attenuation (dB/$\lambda$)') 
plt.ylim([0, 1.5]) # Fixed Range
plt.grid(True, which='both', linestyle=':', linewidth=0.5)

# Legend Outside
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., frameon=True)
plt.subplots_adjust(right=0.7)
plt.show()
#%%
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import os

# ==========================================
# CONFIGURATION
# ==========================================
FILENAME = 'TL_kraken_2000_260122_150.mat'
MAX_RANGE = 5000.0  # meters
MAX_DEPTH = 100.0   # meters

if not os.path.exists(FILENAME):
    print(f"Error: File '{FILENAME}' not found.")
else:
    # ==========================================
    # LOAD DATA
    # ==========================================
    data = sio.loadmat(FILENAME)
    
    # Handle variable names
    if 'TL_2D' in data:
        TL_matrix = data['TL_2D']
    elif 'TL' in data:
        TL_matrix = data['TL']
    else:
        raise ValueError("Could not find 'TL_2D' variable.")

    # ==========================================
    # CONVERT TO PRESSURE LEVEL (dB)
    # ==========================================
    # Pressure Level = -TL
    # High pressure (strong signal) = Small negative number (e.g., -40 dB)
    # Low pressure (weak signal) = Large negative number (e.g., -90 dB)
    pressure_field = -TL_matrix

    # ==========================================
    # PLOTTING
    # ==========================================
    plt.figure(figsize=(10, 6))
    
    # Extent: [x_min, x_max, y_max, y_min]
    # Note: We put y_max (100) first in the bottom slot if we don't invert axis manually,
    # but standard imshow extent order is [left, right, bottom, top].
    # To have Depth 0 at top and 100 at bottom:
    extent = [0, MAX_RANGE/1000.0, MAX_DEPTH, 0] 

    # Plot Pressure Field
    # vmin/vmax set to standard dynamic range (e.g., -90 to -40 dB)
    # You can remove vmin/vmax to let matplotlib auto-scale
    im = plt.imshow(pressure_field, aspect='auto', cmap='jet', extent=extent, vmin=-90, vmax=-40)
    
    cbar = plt.colorbar(im)
    cbar.set_label('Pressure Level (dB)')
    
    plt.title('Acoustic Pressure Field (150 Hz)')
    plt.xlabel('Range (km)')
    plt.ylabel('Depth (m)')
    
    plt.tight_layout()
    plt.show()