import math
import torch
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ─────────────────────────── 1. SETUP & DATA ────────────────────────────
# Ground Truth Parameters (From your Bellhop setup)
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

# Load Bellhop Data
mat_file = 'TL.mat' 
try:
    mat_data = sio.loadmat(mat_file)
    keys = [k for k in mat_data.keys() if not k.startswith('__')]
    
    # 1. Load as Numpy
    TL_matrix_np = mat_data[keys[0]] 
    # 2. Convert to Tensor on GPU immediately to avoid device mismatch
    TL_tensor = torch.tensor(TL_matrix_np, device=device)

    r_full = torch.linspace(0, 5000.0, TL_tensor.shape[1], device=device)
    
    # Masking
    mask = r_full > 1000.0 
    r_meas = r_full[mask]
    TL_meas = TL_tensor[50, mask]
    
    print(f"Data Loaded: {len(r_meas)} points (1km - 5km)")
except FileNotFoundError:
    print("Error: TL_data.mat not found. Using dummy data.")
    r_meas = torch.linspace(1000, 5000, 200, device=device)
    TL_meas = torch.zeros_like(r_meas)

# ─────────────────────────── 2. UNIFIED COMPLEX SOLVER ──────────────────
def unified_complex_solver(c_b, rho_b, alpha_b):
    """
    Finds Trapped and Leaky modes with Staggered Initialization and Unique Filtering.
    """
    k0 = omega / c_w
    kb_real = omega / c_b
    
    # Material Attenuation
    alpha_neper = (alpha_b * (freq / c_b)) / 8.686
    kb_complex = (omega / c_b) + 1j * alpha_neper
    
    # --- A. INITIALIZATION ---
    num_modes = 40 # Increased to ensure we catch enough leaky modes
    m = torch.arange(1, num_modes+1, device=device, dtype=torch.float64)
    
    # Analytic approx for Hard Bottom
    kz_guess = (m - 0.5) * math.pi / D
    k_sq_guess = k0**2 - kz_guess**2
    
    # Classify
    is_propagating = k_sq_guess.real > 0
    k_guess = torch.zeros_like(k_sq_guess, dtype=torch.complex128)
    
    # 1. Trapped Guesses: Standard real sqrt
    k_guess[is_propagating] = torch.sqrt(k_sq_guess[is_propagating]).to(dtype=torch.complex128)
    
    # 2. Leaky Guesses: STAGGERED INITIALIZATION
    # Instead of all starting at 0.995, we space them out linearly
    # from 0.99 down to 0.5 of kb_real
    cutoff_indices = torch.nonzero(~is_propagating).squeeze()
    if len(cutoff_indices) > 0:
        num_leaky = len(cutoff_indices)
        # Create staggered factors: [0.99, 0.95, 0.91, ...]
        factors = torch.linspace(0.99, 0.5, num_leaky, device=device)
        
        # Assign staggered guesses
        # We use a slightly larger imaginary seed (0.005) to ensure stability
        k_guess[cutoff_indices] = torch.complex(
            kb_real * factors, 
            torch.tensor(0.005, device=device)
        )

    # --- B. NEWTON-RAPHSON ---
    k = k_guess.clone()
    for _ in range(30): # Increased iterations for complex convergence
        kz_w = torch.sqrt(k0**2 - k**2)
        kz_b = torch.sqrt(kb_complex**2 - k**2)
        
        sin_w = torch.sin(kz_w * D)
        cos_w = torch.cos(kz_w * D)
        F = rho_w * kz_b * sin_w + 1j * rho_b * kz_w * cos_w
        
        dkzw_dk = -k / kz_w
        dkzb_dk = -k / kz_b
        
        dT1 = rho_w * (dkzb_dk * sin_w + kz_b * cos_w * D * dkzw_dk)
        dT2 = 1j * rho_b * (dkzw_dk * cos_w - kz_w * sin_w * D * dkzw_dk)
        dF_dk = dT1 + dT2
        
        k = k - F / (dF_dk + 1e-20)

    # --- C. FILTERING & UNIQUENESS ---
    # 1. Physical Validity
    valid_mask = (k.real > 0.2) & (k.imag > -1e-8) & (k.real < k0*1.01)
    k_valid = k[valid_mask]
    
    if len(k_valid) == 0: return k_valid
    
    # 2. Sort by Real part descending
    sorted_idx = torch.argsort(k_valid.real, descending=True)
    k_sorted = k_valid[sorted_idx]
    
    # 3. Remove Duplicates (Critical Step!)
    # We compare each root to the previous one
    unique_mask = torch.ones_like(k_sorted, dtype=torch.bool)
    # Tolerance for "sameness": 1e-4
    diff = torch.abs(k_sorted[1:] - k_sorted[:-1])
    unique_mask[1:] = diff > 1e-4
    
    return k_sorted[unique_mask]

def forward_model(c_b, rho_b, alpha_b, r_range):
    # 1. Find Unique Roots
    k_complex = unified_complex_solver(c_b, rho_b, alpha_b)
    
    # 2. Compute Amplitudes
    k0 = omega / c_w
    kz_w = torch.sqrt(k0**2 - k_complex**2)
    
    alpha_neper = (alpha_b * (freq / c_b)) / 8.686
    kb_complex = (omega / c_b) + 1j * alpha_neper
    kz_b = torch.sqrt(kb_complex**2 - k_complex**2)
    
    denom = (D/2 - torch.sin(2*kz_w*D)/(4*kz_w) + 
             (rho_w/rho_b) * (torch.sin(kz_w*D)**2) / (2*kz_b * 1j))
    A_m = 1.0 / torch.sqrt(denom)
    
    # 3. Compute Pressure
    Zs = A_m * torch.sin(kz_w * z_s)
    Zr = A_m * torch.sin(kz_w * z_recv)
    
    phase_term = k_complex[:, None] * r_range[None, :]
    H0 = torch.sqrt(2 / (math.pi * phase_term)) * torch.exp(1j * (phase_term - math.pi/4))
    
    p_sum = 1j/(4*rho_w) * (Zs[:, None] * Zr[:, None] * H0).sum(dim=0)
    
    p_abs = torch.abs(p_sum * 4.0 * math.pi)
    tl = -20 * torch.log10(p_abs + 1e-12)
    
    return tl, k_complex


# ─────────────────────────── 3. EXECUTE & PLOT ──────────────────────────
print("\n=== RUNNING FORWARD MODEL (GROUND TRUTH) ===")
print(f"Parameters: c_b={TRUE_CB}, rho_b={TRUE_RHOB}, alpha={TRUE_ALPHA}")

# Run Model
tl_pred, roots = forward_model(
    torch.tensor(TRUE_CB, device=device), 
    torch.tensor(TRUE_RHOB, device=device), 
    torch.tensor(TRUE_ALPHA, device=device), 
    r_meas
)

# Calculate Bias
bias = (TL_meas - tl_pred).mean()
tl_pred_adjusted = tl_pred + bias

# Error Metric
mse = (tl_pred_adjusted - TL_meas).pow(2).mean().item()
print(f"\nPrediction MSE: {mse:.4f}")
print(f"Optimal Gain Bias: {bias:.2f} dB")

# --- Analyze Roots ---
print("\n--- DETECTED MODES ---")
print(f"{'Mode':<5} | {'Real(k)':<10} | {'Imag(k) (Decay)':<15} | {'Type'}")
print("-" * 50)
k_btm_real = omega / TRUE_CB
# Sort by real part descending
sorted_indices = torch.argsort(roots.real, descending=True)
roots_sorted = roots[sorted_indices]

for i, r in enumerate(roots_sorted):
    r_val = r.real.item()
    i_val = r.imag.item()
    
    if r_val > k_btm_real:
        mtype = "TRAPPED"
    else:
        mtype = "LEAKY"
        if i_val < 0.005: mtype += " (Near-Crit)"
        
    print(f"{i+1:<5} | {r_val:.5f}    | {i_val:.6f}          | {mtype}")

# --- Plotting ---
tl_pred_np = tl_pred_adjusted.detach().cpu().numpy()
tl_meas_np = TL_meas.detach().cpu().numpy()
r_np = r_meas.detach().cpu().numpy()

plt.figure(figsize=(12, 6))
plt.plot(r_np, tl_meas_np, 'k', linewidth=1.5, alpha=0.6, label='Bellhop Data (Ground Truth)')
plt.plot(r_np, tl_pred_np, 'r--', linewidth=2.0, label='PyTorch Unified Solver')

plt.title(f"Forward Model Verification\nParams: c_b={TRUE_CB}, rho_b={TRUE_RHOB}, alpha={TRUE_ALPHA}")
plt.xlabel("Range (m)")
plt.ylabel("Transmission Loss (dB)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.gca().invert_yaxis() 
plt.show()