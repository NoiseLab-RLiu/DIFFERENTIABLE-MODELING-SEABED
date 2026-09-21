# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 16:48:50 2025

@author: 13391
"""

#!/usr/bin/env python3
"""
Single-parameter PINN inversion of bottom sound speed c_b
in a two-layer Pekeris waveguide (100 Hz test case).

> python pinn_pekeris.py
"""

# ────────────────────────────────────────────────────────────── imports ──
import math, torch, os, scipy.io as sio, numpy as np
import matplotlib.pyplot as plt
torch.set_num_threads(1)
os.environ["OMP_NUM_THREADS"]   = "1"
os.environ["MKL_NUM_THREADS"]   = "1"
torch.autograd.set_detect_anomaly(True)
torch.set_default_dtype(torch.float64)

# ───────────────────────────────────── constants / geometry / data  ──
f      = 100.0                                 # Hz
omega  = 2*math.pi*f
c_w    = 1500.0                                # water sound speed (m/s)
D      = 100.0                                 # water depth  (m)
rho_w  = rho_b = 1.0
z_s    = 25.0
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# receivers
r_meas = torch.linspace(20., 180., 10, device=device)    # (N,)
z_meas = torch.tensor(D, device=device)

# asymptotic initial k guesses (m = 0…8)
m = torch.arange(9, dtype=torch.float64, device=device)
gamma_as = (m + 0.52)*math.pi/D
k_guess  = torch.sqrt((omega/c_w)**2 - gamma_as**2)      # (9,)

# ─────────────────────────── implicit, differentiable root op ──
class DispersionRoot(torch.autograd.Function):
    @staticmethod
    def _f(k, cb):
        g   = torch.sqrt(omega**2/c_w**2 - k**2)          # γ
        gb  = torch.sqrt(k**2 - omega**2/cb**2)           # γ_b
        f   = torch.tan(g*D) + g/gb
        return f, g, gb

    @staticmethod
    def forward(ctx, c_b, k0, max_it: int = 12):
        cb = c_b.detach()
        k  = k0.detach().clone()
        for _ in range(max_it):
            f, g, gb = DispersionRoot._f(k, cb)

            # ----- full three-term df/dk  ---------------------------------
            sec2 = 1.0 / torch.cos(g*D)**2                   # = sec²
            sec2 = torch.clamp(sec2, max=1e12)               # optional clamp
            df_dk = (sec2 * (-k/g) * D) - (k / (g*gb)) - (g * k / gb**3)
            k = k - f / df_dk                                # Newton step
        ctx.save_for_backward(k, cb)
        return k.to(c_b.device)

    @staticmethod
    def backward(ctx, grad_out):
        k, cb   = ctx.saved_tensors
        f, g, gb = DispersionRoot._f(k, cb)

        sec2 = 1.0 / torch.cos(g*D)**2
        sec2 = torch.clamp(sec2, max=1e12)
        df_dk = (sec2 * (-k/g) * D) - (k / (g*gb)) - (g * k / gb**3)

        df_dcb = -(g / gb**3) * (omega**2 / cb**3)
        dk_dcb = - df_dcb / df_dk

        return grad_out * dk_dcb, None, None                 # grad for c_b only

# helper: current mode set (differentiable w.r.t. c_b)
def km_current(c_b):
    roots = [DispersionRoot.apply(c_b, k_guess[i]) for i in range(9)]
    km    = torch.stack(roots)                    # (9,)
    return km[torch.isfinite(km)]                 # mask NaN (if any)

# ─────────────────────── forward model & physics residual ──
def pressure_field(c_b, r, z):
    km = km_current(c_b)                  # (M,)
    if km.numel() == 0:                   # safeguard
        return torch.full_like(r, torch.nan+0j)

    k0 = omega/c_w
    gamma_m  = torch.sqrt(k0**2 - km**2)
    kb       = omega/c_b
    gamma_bm = torch.sqrt(torch.clamp(km**2 - kb**2, min=0.0))

    A_m = 1/torch.sqrt(D/2
          - torch.sin(2*gamma_m*D)/(4*gamma_m)
          + torch.sin(gamma_m*D)**2/(2*rho_b*gamma_bm))

    Zs = A_m*torch.sin(gamma_m*z_s)               # (M)
    Z  = A_m*torch.sin(gamma_m*z)                 # (M)
    j0 = torch.special.bessel_j0
    y0 = torch.special.bessel_y0
    H0 = j0(km[:,None]*r) + 1j*y0(km[:,None]*r)   # (M,N)

    return 1j/(4*rho_w)*(Zs[:,None]*Z[:,None]*H0).sum(dim=0)

def boundary_residual(c_b, r):
    km = km_current(c_b)
    if km.numel() == 0:
        return torch.full_like(r, torch.nan+0j)

    k0 = omega/c_w
    gamma_m  = torch.sqrt(k0**2 - km**2)
    kb       = omega/c_b
    gamma_bm = torch.sqrt(torch.clamp(km**2 - kb**2, min=0.0))

    A_m = 1/torch.sqrt(D/2
          - torch.sin(2*gamma_m*D)/(4*gamma_m)
          + torch.sin(gamma_m*D)**2/(2*rho_b*gamma_bm))

    Zs   = A_m*torch.sin(gamma_m*z_s)
    ZD   = A_m*torch.sin(gamma_m*D)
    dZD  = A_m*gamma_m*torch.cos(gamma_m*D)
    j0,y0 = torch.special.bessel_j0, torch.special.bessel_y0
    H0   = j0(km[:,None]*r)+1j*y0(km[:,None]*r)

    pD   = 1j/(4*rho_w)*(Zs[:,None]*ZD [:,None]*H0).sum(dim=0)
    dpDz = 1j/(4*rho_w)*(Zs[:,None]*dZD[:,None]*H0).sum(dim=0)
    gamma_b = gamma_bm.max()                       # scalar scale
    return pD + (rho_b/gamma_b)*dpDz

# ─────────────────────────── synthetic measurements ──
c_b_true = torch.tensor(2000., device=device)
p_true   = pressure_field(c_b_true, r_meas, z_meas)
noise    = 0.00 * p_true.abs().max() * \
           (torch.randn_like(p_true)+1j*torch.randn_like(p_true))
p_meas   = p_true + noise

# ───────────────────────── optimization loop ────────
c_b_est = torch.tensor(2050., device=device, requires_grad=True)
opt     = torch.optim.Adam([c_b_est], lr=1.0)

lambda_phys = 0
cb_history  = []
save_every = 10_000
save_dir = "checkpoints"
os.makedirs(save_dir, exist_ok=True)  
for epoch in range(80000):
    opt.zero_grad()
    p_pred   = pressure_field(c_b_est, r_meas, z_meas)
    data_l   = (p_pred - p_meas).abs().pow(2).mean()

    #resid    = boundary_residual(c_b_est, r_meas)
    #phys_l   = resid.abs().pow(2).mean()

    loss = data_l# + lambda_phys*phys_l
    loss.backward()
    
    # # check gradient
    # if torch.isnan(c_b_est.grad):
    #     print(">>> GRADIENT IS NaN – aborting")
    #     break
    # for p_state in opt.state.values():
    #     if p_state:
    #         print("  Adam exp_avg[0] =", p_state['exp_avg'][0].item())
    opt.step()

    with torch.no_grad():
        c_b_est.clamp_(1000., 3000.)          # keep physical
    cb_history.append(c_b_est.item())

    if epoch % 50 == 0:
        print(f"epoch {epoch:6d}: loss={loss.item():.3e}  "
              f"c_b={c_b_est.item():.2f} m/s")
        
    if epoch % save_every == 0:
        tag = f"{epoch:06d}"            # "000010", "000020", …
        arr = np.array(cb_history, dtype=np.float64)   # to NumPy
        sio.savemat(f"{save_dir}/cb_history_{tag}.mat", {"cb_history": arr})
        print(f"checkpoint written at epoch {epoch}")
# ─────────────────────────── result plot (optional) ──
plt.figure()    
plt.plot(cb_history); plt.xlabel('epoch'); plt.ylabel('c_b (m/s)')
plt.title('Convergence'); plt.show()
