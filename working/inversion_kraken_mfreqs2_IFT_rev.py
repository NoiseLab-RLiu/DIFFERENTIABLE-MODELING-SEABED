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
    mat_file = f'TL_kraken_2026_260122_{int(f)}.mat'#TL_kraken_2026_260122_{int(f)}.mat
    
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
    raise ValueError(-1) 

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

# Start parameters
cb_opt = torch.tensor(1950.0, device=device, requires_grad=True)
rb_opt = torch.tensor(1.2, device=device, requires_grad=True)
ab_opt = torch.tensor(0.1, device=device, requires_grad=True)

# Precompute available freqs/omegas
active_freqs = [f for f in FREQ_LIST if f in TL_meas_dict]
active_omegas = [2 * math.pi * f for f in active_freqs]

cb_hist = []
rb_hist = []
ab_hist = []
# -------------------------------------------------------------------------
# STAGE 1: BASIN FINDING (All Freqs, Equal Weight)
# -------------------------------------------------------------------------
print(f">>> STAGE 1: Basin Finding (Freqs: {active_freqs})")
optimizer = torch.optim.Rprop([cb_opt, rb_opt, ab_opt], lr=0.005)

time_start = time.time()
for i in range(100):
    optimizer.zero_grad()
    total_loss = 0.0
    
    for freq, omega in zip(active_freqs, active_omegas):
        # Stage 1: Coarse Mode (Ignore returned modes, no tracking needed here)
        pred, _ = forward_model(cb_opt, rb_opt, ab_opt, r_meas, omega, freq, stage=1)
        total_loss += torch.nn.functional.mse_loss(pred, TL_meas_dict[freq])
    
    total_loss.backward()
    optimizer.step()
    
    with torch.no_grad():
        cb_opt.clamp_(1550, 2500); rb_opt.clamp_(1.0, 3.0); ab_opt.clamp_(0.01, 5.0)
    
        cb_hist.append(cb_opt.item())
        rb_hist.append(rb_opt.item())
        ab_hist.append(ab_opt.item())
    
    if i % 50 == 0:
        print(f"Iter {i:04d} | SumLoss: {total_loss.item():.4f} | cb: {cb_opt.item():.1f} | rho: {rb_opt.item():.3f} | alpha: {ab_opt.item():.3f}")

# -------------------------------------------------------------------------
# STAGE 2: PHYSICS-BASED DECOUPLING (Range Split)
# -------------------------------------------------------------------------
print(f"\n>>> STAGE 2: Range-Split Refinement")

# Helper to slice data for Far-Field (Alpha) vs Full Range (Phase)
def get_loss(pred, target, range_mode='full'):
    if range_mode == 'far':
        # Only look at the last 50% of the range (e.g., 3km - 5km)
        n_points = pred.shape[0]
        half_idx = n_points // 4*3 
        return torch.nn.functional.mse_loss(pred[half_idx:], target[half_idx:])
    else:
        # Full range
        return torch.nn.functional.mse_loss(pred, target)

prev_modes_dict = {}

# --- STEP 1: FIX PHASE (cb, rho) using FULL DATA ---
print("   [Step 1] Optimizing Phase (cb, rho)...")
# Lock Alpha to a reasonable start (e.g., 0.1 or current guess)
# We prevent it from moving so it doesn't compensate for cb
ab_opt.requires_grad = True#False 
cb_opt.requires_grad = True
rb_opt.requires_grad = True

optimizer_phase = torch.optim.Rprop([ab_opt, cb_opt, rb_opt], lr=0.005)

#optimizer_phase = torch.optim.Rprop([cb_opt, rb_opt], lr=0.005)

for i in range(60): 
    optimizer_phase.zero_grad()
    total_loss = 0.0
    
    for freq, omega in zip(active_freqs, active_omegas):
        seeds = prev_modes_dict.get(freq, None)
        pred, k_final = forward_model(
            cb_opt, rb_opt, ab_opt, r_meas, omega, freq, 
            stage=2, initial_seeds=seeds
        )
        prev_modes_dict[freq] = k_final.detach()
        # Phase needs full interference pattern
        total_loss += get_loss(pred, TL_meas_dict[freq], 'far')
    
    total_loss.backward()
    optimizer_phase.step()
    
    with torch.no_grad():
        cb_opt.clamp_(1550, 2500); rb_opt.clamp_(1.0, 3.0)
        
        cb_hist.append(cb_opt.item())
        rb_hist.append(rb_opt.item())
        ab_hist.append(ab_opt.item())
    
    if i % 10 == 0:
        print(f"   Iter {i:03d} | Loss: {total_loss.item():.4f} | cb: {cb_opt.item():.1f} | rho: {rb_opt.item():.3f} | alpha: {ab_opt.item():.3f}")

time_end = time.time()
print('time')
print(time_end-time_start)
# --- STEP 2: FIX ATTENUATION (alpha) using FAR FIELD ---
print("   [Step 2] Optimizing Attenuation (Alpha) using FAR FIELD only...")
ab_opt.requires_grad = True
cb_opt.requires_grad = False # LOCK Phase completely
rb_opt.requires_grad = False

optimizer_alpha = torch.optim.Rprop([ab_opt], lr=0.02) # Slower LR for precision

for i in range(60): 
    optimizer_alpha.zero_grad()
    total_loss = 0.0
    
    for freq, omega in zip(active_freqs, active_omegas):
        seeds = prev_modes_dict.get(freq, None)
        pred, k_final = forward_model(
            cb_opt, rb_opt, ab_opt, r_meas, omega, freq, 
            stage=2, initial_seeds=seeds
        )
        prev_modes_dict[freq] = k_final.detach()
        
        # KEY CHANGE: Only calculate loss on Far Field (r > 2.5km)
        # This forces alpha to fit the DECAY, not the interference wiggles
        total_loss += get_loss(pred, TL_meas_dict[freq], 'far')
    
    total_loss.backward()
    optimizer_alpha.step()
    
    with torch.no_grad():
        ab_opt.clamp_(0.01, 5.0)
        
    if i % 10 == 0:
        print(f"   Iter {i:03d} | FarLoss: {total_loss.item():.4f} | cb: {cb_opt.item():.1f} | rho: {rb_opt.item():.3f} | alpha: {ab_opt.item():.3f}")

print(f"\nFINAL RESULT: cb={cb_opt.item():.2f}, rho={rb_opt.item():.3f}, alpha={ab_opt.item():.3f}")
#%%
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# =============================================================================
# 1. SETUP DUMMY DATA (Replace these with your actual lists)
# =============================================================================
# Structure: [Exp1_traj, Exp2_traj, Exp3_traj]
# Example lengths are just for demo; your actual lists will likely vary or match in length.

# --- Dummy Data for c_b (Sound Speed) ---
# True values: 1910, 2000, 2026
# CB_RECORD = [
#     np.linspace(1500, 1910, 151) + np.random.normal(0, 5, 151),  # Exp 1
#     np.linspace(1500, 2000, 151) + np.random.normal(0, 5, 151),  # Exp 2
#     np.linspace(1500, 2026, 151) + np.random.normal(0, 5, 151)   # Exp 3
# ]
CB_TRUE = [1910, 2000, 2026]

# --- Dummy Data for rho_b (Density) ---
# True values: 1.65, 1.8, 1.3
# RB_RECORD = [
#     np.linspace(1.2, 1.65, 151) + np.random.normal(0, 0.02, 151), # Exp 1
#     np.linspace(1.2, 1.8, 151) + np.random.normal(0, 0.02, 151),  # Exp 2
#     np.linspace(1.2, 1.3, 151) + np.random.normal(0, 0.02, 151)   # Exp 3
# ]
RB_TRUE = [1.65, 1.8, 1.3]

# --- Dummy Data for alpha_b (Attenuation) ---
# True values: 0.45, 0.5, 0.8
# AB_RECORD = [
#     np.linspace(0.1, 0.45, 151) + np.random.normal(0, 0.01, 151), # Exp 1
#     np.linspace(0.1, 0.5, 151) + np.random.normal(0, 0.01, 151),  # Exp 2
#     np.linspace(0.1, 0.8, 151) + np.random.normal(0, 0.01, 151)   # Exp 3
# ]
AB_TRUE = [0.45, 0.5, 0.8]

# =============================================================================
# 2. PLOTTING CONFIGURATION
# =============================================================================
# Using explicit colors for consistency across subplots (Blue, Red, Green)
colors = ['#1f77b4', '#d62728', '#2ca02c'] 
exp_labels = ['Exp #1', 'Exp #2', 'Exp #3']

# Create the figure with 3 subplots in 1 row
fig, axes = plt.subplots(1, 3, figsize=(18, 3.5), constrained_layout=True)

# Font sizes
TITLE_SIZE = 18
LABEL_SIZE = 16
TICK_SIZE = 14
LEGEND_SIZE = 14

# -----------------------------------------------------------------------------
# SUBPLOT 1: Sound Speed (c_b)
# -----------------------------------------------------------------------------
ax = axes[0]
for i in range(3):
    # Plot Trajectory (Solid)
    ax.plot(CB_RECORD[i], color=colors[i], linewidth=2.5)
    # Plot Ground Truth (Dashed)
    ax.axhline(y=CB_TRUE[i], color=colors[i], linestyle='--', linewidth=2.0, alpha=0.8)

ax.set_title(r'(a) Sound Speed $c_{\mathrm{b}}$', fontsize=TITLE_SIZE)
ax.set_ylabel(r'Speed (m/s)', fontsize=LABEL_SIZE)
ax.set_xlabel('Epoch', fontsize=LABEL_SIZE)
ax.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
ax.grid(True, linestyle=':', alpha=0.6)

# -----------------------------------------------------------------------------
# SUBPLOT 2: Density (rho_b)
# -----------------------------------------------------------------------------
ax = axes[1]
for i in range(3):
    ax.plot(RB_RECORD[i], color=colors[i], linewidth=2.5)
    ax.axhline(y=RB_TRUE[i], color=colors[i], linestyle='--', linewidth=2.0, alpha=0.8)

ax.set_title(r'(b) Density $\rho_{\mathrm{b}}$', fontsize=TITLE_SIZE)
ax.set_ylabel(r'Density (g/cm$^3$)', fontsize=LABEL_SIZE)
ax.set_xlabel('Epoch', fontsize=LABEL_SIZE)
ax.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
ax.grid(True, linestyle=':', alpha=0.6)

# -----------------------------------------------------------------------------
# SUBPLOT 3: Attenuation (alpha_b)
# -----------------------------------------------------------------------------
ax = axes[2]
for i in range(3):
    ax.plot(AB_RECORD[i], color=colors[i], linewidth=2.5)
    ax.axhline(y=AB_TRUE[i], color=colors[i], linestyle='--', linewidth=2.0, alpha=0.8)

ax.set_title(r'(c) Attenuation $\alpha_{\mathrm{b}}$', fontsize=TITLE_SIZE)
ax.set_ylabel(r'Attenuation (dB/$\lambda$)', fontsize=LABEL_SIZE)
ax.set_xlabel('Epoch', fontsize=LABEL_SIZE)
ax.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
ax.grid(True, linestyle=':', alpha=0.6)

# =============================================================================
# 3. SINGLE CONSOLIDATED LEGEND
# =============================================================================
# We create a custom legend to the right of the last plot.
custom_lines = []
custom_labels = []

for i in range(3):
    # Solid line entry
    custom_lines.append(Line2D([0], [0], color=colors[i], lw=2.5))
    custom_labels.append(f'{exp_labels[i]} (Est)')
    
    # Dashed line entry
    custom_lines.append(Line2D([0], [0], color=colors[i], lw=2.0, linestyle='--'))
    custom_labels.append(f'{exp_labels[i]} (True)')

# Place legend outside the last subplot (to the right)
# bbox_to_anchor=(x, y) coordinates are relative to the subplot axes
axes[2].legend(custom_lines, custom_labels, 
               loc='center left', bbox_to_anchor=(1.05, 0.5), 
               fontsize=LEGEND_SIZE, frameon=True)

# =============================================================================
# 4. SAVE AND SHOW
# =============================================================================
# plt.savefig('optimization_history_combined.png', dpi=300, bbox_inches='tight')
plt.show()

#%%
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
import torch

# =============================================================================
# 0. HELPER: DATA CLEANING (Safety First)
# =============================================================================
def clean_data_list(data_list):
    """Converts a list of tensors/arrays into a clean list of floats."""
    clean = []
    for item in data_list:
        if torch.is_tensor(item):
            clean.append(item.detach().cpu().item())
        elif isinstance(item, (np.ndarray, np.generic)):
            clean.append(item.item())
        else:
            clean.append(item)
    return clean

# =============================================================================
# 1. DATA PREPARATION (Using your loaded variables)
# =============================================================================
# Assuming CB_RECORD, RB_RECORD, AB_RECORD are list-of-lists (3 experiments each)
# If they are not yet clean (contain tensors), we clean them here:
try:
    # Ensure data is clean (convert tensors to floats)
    cb_plot_data = [clean_data_list(exp) for exp in CB_RECORD]
    rb_plot_data = [clean_data_list(exp) for exp in RB_RECORD]
    ab_plot_data = [clean_data_list(exp) for exp in AB_RECORD]
except NameError:
    print("Warning: Data variables (CB_RECORD, etc.) not found. Using dummy data for demo.")
    # Fallback for testing the plot without data
    cb_plot_data = [np.random.rand(150) * 100 + 1500 for _ in range(3)]
    rb_plot_data = [np.random.rand(150) * 0.5 + 1.2 for _ in range(3)]
    ab_plot_data = [np.random.rand(150) * 0.5 + 0.1 for _ in range(3)]

# Ground Truths
CB_TRUE = [1910, 2000, 2026]
RB_TRUE = [1.65, 1.8, 1.3]
AB_TRUE = [0.45, 0.5, 0.8]

# =============================================================================
# 2. PLOTTING CONFIGURATION
# =============================================================================
# Force serif fonts for a professional look (optional but recommended)
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm" 
})

colors = ['#1f77b4', '#d62728', '#2ca02c'] 
exp_labels = ['Exp #1', 'Exp #2', 'Exp #3']

# Height reduced slightly to 4 to make it compact but readable
fig, axes = plt.subplots(1, 3, figsize=(18, 4), constrained_layout=True)

# Font sizes
TITLE_SIZE = 18
LABEL_SIZE = 16
TICK_SIZE = 14
LEGEND_SIZE = 14

# -----------------------------------------------------------------------------
# SUBPLOT 1: Sound Speed (c_b)
# -----------------------------------------------------------------------------
ax = axes[0]
for i in range(3):
    ax.plot(cb_plot_data[i], color=colors[i], linewidth=2.5)
    ax.axhline(y=CB_TRUE[i], color=colors[i], linestyle='--', linewidth=2.0, alpha=0.8)

# REVISED LABELS & TITLE
ax.set_title(r'(a) Sound Speed $c_{\mathrm{b}}$', fontsize=TITLE_SIZE)
ax.set_ylabel(r'Speed (m/s)', fontsize=LABEL_SIZE)
ax.set_xlabel('Epoch', fontsize=LABEL_SIZE)

ax.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
ax.grid(True, linestyle=':', alpha=0.6)

# -----------------------------------------------------------------------------
# SUBPLOT 2: Density (rho_b)
# -----------------------------------------------------------------------------
ax = axes[1]
for i in range(3):
    ax.plot(rb_plot_data[i], color=colors[i], linewidth=2.5)
    ax.axhline(y=RB_TRUE[i], color=colors[i], linestyle='--', linewidth=2.0, alpha=0.8)

# REVISED LABELS & TITLE
ax.set_title(r'(b) Density $\rho_{\mathrm{b}}$', fontsize=TITLE_SIZE)
ax.set_ylabel(r'Density (g/cm$^3$)', fontsize=LABEL_SIZE)
ax.set_xlabel('Epoch', fontsize=LABEL_SIZE)

ax.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
ax.grid(True, linestyle=':', alpha=0.6)

# -----------------------------------------------------------------------------
# SUBPLOT 3: Attenuation (alpha_b)
# -----------------------------------------------------------------------------
ax = axes[2]
for i in range(3):
    ax.plot(ab_plot_data[i], color=colors[i], linewidth=2.5)
    ax.axhline(y=AB_TRUE[i], color=colors[i], linestyle='--', linewidth=2.0, alpha=0.8)

# REVISED LABELS & TITLE
ax.set_title(r'(c) Attenuation $\alpha_{\mathrm{b}}$', fontsize=TITLE_SIZE)
ax.set_ylabel(r'Attenuation (dB/$\lambda$)', fontsize=LABEL_SIZE)
ax.set_xlabel('Epoch', fontsize=LABEL_SIZE)

ax.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
ax.grid(True, linestyle=':', alpha=0.6)

# =============================================================================
# 3. SINGLE CONSOLIDATED LEGEND
# =============================================================================
custom_lines = []
custom_labels = []

for i in range(3):
    # Solid line entry (Estimate)
    custom_lines.append(Line2D([0], [0], color=colors[i], lw=2.5))
    custom_labels.append(f'{exp_labels[i]} (Est)')
    
    # Dashed line entry (True)
    custom_lines.append(Line2D([0], [0], color=colors[i], lw=2.0, linestyle='--'))
    custom_labels.append(f'{exp_labels[i]} (True)')

axes[2].legend(custom_lines, custom_labels, 
               loc='center left', bbox_to_anchor=(1.05, 0.5), 
               fontsize=LEGEND_SIZE, frameon=True)

# Save
plt.savefig('optimization_history_final.png', dpi=300, bbox_inches='tight')
plt.show()