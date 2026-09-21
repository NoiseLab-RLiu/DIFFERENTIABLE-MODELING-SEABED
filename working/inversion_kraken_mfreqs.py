import torch
import math
import scipy.io as sio
import os

# Device Config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =========================================================================
# 1. CONFIGURATION
# =========================================================================
C_W_GLOBAL   = 1500.0
RHO_W_GLOBAL = 1.0
D_GLOBAL     = 100.0
ZS_GLOBAL    = 25.0
ZR_GLOBAL    = 50.0 

FREQ_LIST = [30.0, 35.0, 50.0, 100.0, 150.0] 

# =========================================================================
# 2. DATA LOADING
# =========================================================================
TL_meas_dict = {}
r_meas = None

print("--- LOADING MULTI-FREQ DATA ---")
for f in FREQ_LIST:
    mat_file = f'TL_kraken_1910_{int(f)}.mat'
    if f == 150.0 and not os.path.exists(mat_file):
        if os.path.exists('TL_kraken2.mat'): mat_file = 'TL_kraken2.mat'

    try:
        if not os.path.exists(mat_file): continue
        mat_data = sio.loadmat(mat_file)
        TL_tensor = torch.tensor(mat_data['TL_2D'], device=device)
        if r_meas is None:
            r_full = torch.linspace(1.0, 5000.0, TL_tensor.shape[1], device=device)
            mask = (r_full >= 1000.0) & (r_full <= 5000.0)
            r_meas = r_full[mask]
        TL_meas_dict[f] = TL_tensor[50, mask]
        print(f"[OK] Loaded {f} Hz")
    except: pass

if not TL_meas_dict: exit()

# =========================================================================
# 3. UNIFIED SOLVER (Robust)
# =========================================================================

def unified_complex_solver(c_b, rho_b, alpha_b, omega, freq, coarse_mode=False):
    c_w, rho_w, D = C_W_GLOBAL, RHO_W_GLOBAL, D_GLOBAL
    k0 = omega / c_w
    k_min = omega / 3000.0
    
    # 1. Complex Bottom Wavenumber
    # Prevent divide by zero if c_b is crazy, though optimizer clamps handle it
    kb_complex = (omega / c_b) + 1j * ((alpha_b * (freq/c_b))/8.686)

    if coarse_mode:
        # --- MODE A: REAL SCAN (Safe Form) ---
        k_scan = torch.linspace(k0 * 0.999, k_min, 1000, device=device, dtype=torch.float64)
        
        # Calculate Vertical Wavenumbers
        # Note: We assume Trapped regime (k > kb) for Real Scan validity
        # We clamp to ensure no NaNs in sqrt, but physically these terms
        # correspond to evanescent decay in bottom.
        kz_w = torch.sqrt(torch.clamp(k0**2 - k_scan**2, min=1e-12))
        kz_b = torch.sqrt(torch.clamp(k_scan**2 - (omega/c_b)**2, min=1e-12))
        
        # FIX: Use Singularity-Free Characteristic Equation
        # F = rho_w * kz_b * sin(kz_w*D) + rho_b * kz_w * cos(kz_w*D)
        # This removes the (1/kz_b) explosion at critical angle
        s_w, c_w_val = torch.sin(kz_w * D), torch.cos(kz_w * D)
        F_scan = rho_w * kz_b * s_w + rho_b * kz_w * c_w_val
        
        # Sign Change Detection
        sign_change = (F_scan[:-1] * F_scan[1:]) < 0
        
        # Jump filter is less critical with Safe F, but still good to keep
        # to avoid high-frequency aliasing artifacts
        jump_filter = torch.abs(F_scan[:-1] - F_scan[1:]) < 500.0 # Relaxed threshold
        
        valid_indices = torch.nonzero(sign_change & jump_filter).flatten()
        
        if len(valid_indices) == 0: return torch.tensor([]).to(device)
        k = k_scan[valid_indices].to(dtype=torch.complex128)
        iters = 20
    else:
        # --- MODE B: COMPLEX GRID ---
        real_vals = torch.linspace(k0*1.01, k_min, 400, device=device)
        imag_vals = torch.tensor([0.0, 1e-4, 1e-3], device=device)
        g_real, g_imag = torch.meshgrid(real_vals, imag_vals, indexing='ij')
        k = (g_real + 1j * g_imag).flatten()
        iters = 30

    # Newton Refinement (Safe Complex Form)
    for _ in range(iters):
        kz_w = torch.sqrt(k0**2 - k**2)
        kz_b = torch.sqrt(kb_complex**2 - k**2)
        s_w, c_w_ = torch.sin(kz_w*D), torch.cos(kz_w*D)
        
        F = rho_w * kz_b * s_w + 1j * rho_b * kz_w * c_w_
        
        # Derivatives
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

# =========================================================================
# 4. CLAMPED IFT SOLVER
# =========================================================================
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

        # Clamp Fix
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

# =========================================================================
# 5. FORWARD MODEL
# =========================================================================
def forward_model(c_b, rho_b, alpha_b, r_range, omega, freq, stage=1):
    
    if stage == 1:
        # Stage 1: Coarse Autograd (Normalized F)
        k_complex = unified_complex_solver(c_b, rho_b, alpha_b, omega, freq, coarse_mode=True)
    else:
        # Stage 2: Clamped IFT Complex
        k_complex = IFTSolver.apply(c_b, rho_b, alpha_b, omega, freq, False)
    
    if len(k_complex) == 0: 
        return torch.ones_like(r_range)*100.0 + (c_b*0+rho_b*0+alpha_b*0)

    k0 = omega / C_W_GLOBAL

    if stage == 1:
        pass # No filtering
    else:
        is_ghost = torch.abs(k_complex.real - k0) < 2e-4
        k_complex = k_complex[~is_ghost]

    # Field Calc
    kz_w = torch.sqrt(k0**2 - k_complex**2)
    kb_complex = (omega/c_b) + 1j*((alpha_b * (freq/c_b))/8.686)
    kz_b = torch.sqrt(kb_complex**2 - k_complex**2)

    term1 = D_GLOBAL/2
    term2 = torch.sin(2*kz_w*D_GLOBAL)/(4*kz_w)
    term3 = (RHO_W_GLOBAL/rho_b) * (torch.sin(kz_w*D_GLOBAL)**2) / (2j * kz_b)
    A_m = 1.0 / torch.sqrt(term1 - term2 + term3)
    
    phase = k_complex[:, None] * r_range[None, :]
    H0 = torch.sqrt(2 / (math.pi * phase)) * torch.exp(1j * (phase - math.pi/4))
    
    # Mode Sum
    Zs = A_m * torch.sin(kz_w * ZS_GLOBAL)
    Zr = A_m * torch.sin(kz_w * ZR_GLOBAL)
    p_sum = 1j/(4*RHO_W_GLOBAL) * (Zs[:, None] * Zr[:, None] * H0).sum(dim=0)
    
    return -20 * torch.log10(p_sum.abs() * 4.0 * math.pi + 1e-12)

# =========================================================================
# 6. OPTIMIZATION
# =========================================================================
print("\n=== STARTING MULTI-FREQ INVERSION ===")

cb_opt = torch.tensor(1950.0, device=device, requires_grad=True)
rb_opt = torch.tensor(1.2, device=device, requires_grad=True)
ab_opt = torch.tensor(0.1, device=device, requires_grad=True)

# Precompute
active_freqs = [f for f in FREQ_LIST if f in TL_meas_dict]
active_omegas = [2 * math.pi * f for f in active_freqs]

# Stage 1
print(f">>> STAGE 1: Multi-Freq Basin Finding (Freqs: {active_freqs})")
# Reduced LR to prevent explosion
optimizer = torch.optim.Rprop([cb_opt, rb_opt, ab_opt], lr=0.005)

for i in range(350):
    optimizer.zero_grad()
    total_loss = 0.0
    for freq, omega in zip(active_freqs, active_omegas):
        pred = forward_model(cb_opt, rb_opt, ab_opt, r_meas, omega, freq, stage=1)
        total_loss += torch.nn.functional.mse_loss(pred, TL_meas_dict[freq])
    
    total_loss.backward()
    optimizer.step()
    
    with torch.no_grad():
        # FIX: Clamp alpha > 0.0 to prevent real-wavenumber singularity
        cb_opt.clamp_(1550.0, 2500.0); rb_opt.clamp_(1.0, 3.0); ab_opt.clamp_(0.01, 5.0)
    
    if i % 50 == 0:
        print(f"Iter {i:04d} | SumLoss: {total_loss.item():.4f} | cb: {cb_opt.item():.1f} | rho: {rb_opt.item():.3f} | alpha: {ab_opt.item():.3f}")

# Stage 2
print("\n>>> STAGE 2: Multi-Freq Precision")
optimizer = torch.optim.Rprop([cb_opt, rb_opt, ab_opt], lr=0.005) # Start stronger, decay later

for i in range(1000):
    #if i == 150:
        # print("--- Decaying LR to 0.002 ---")
        # for pg in optimizer.param_groups: pg['lr'] = 0.002
        
    optimizer.zero_grad()
    total_loss = 0.0
    for freq, omega in zip(active_freqs, active_omegas):
        pred = forward_model(cb_opt, rb_opt, ab_opt, r_meas, omega, freq, stage=2)
        total_loss += torch.nn.functional.mse_loss(pred, TL_meas_dict[freq])
    
    total_loss.backward()
    optimizer.step()
    
    with torch.no_grad():
        cb_opt.clamp_(1550.0, 2500.0); rb_opt.clamp_(1.0, 3.0); ab_opt.clamp_(0.01, 5.0)
        
    if i % 10 == 0:
        print(f"Iter {i:04d} | SumLoss: {total_loss.item():.4f} | cb: {cb_opt.item():.1f} | rho: {rb_opt.item():.3f} | alpha: {ab_opt.item():.3f}")