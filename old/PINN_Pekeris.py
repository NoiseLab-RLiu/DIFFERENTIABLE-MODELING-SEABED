# -*- coding: utf-8 -*-
"""
Created on Sun Jun  8 17:29:59 2025

@author: 13391
"""

import torch, math, scipy.io as sio
from torch import pi
torch.set_default_dtype(torch.float64)        # high precision like MATLAB

# ------------- constants -------------
f      = 100.0                       # Hz
omega  = 2 * pi * f
c_w    = 1500.0                      # water
D      = 100.0                       # depth
rho_w  = rho_b = 1.0
z_s    = 25.0
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# measurement grid
r_meas = torch.linspace(20., 180., 10, device=device)   # (10,)
z_meas = torch.tensor(D, device=device)

# fixed modes (load once)
km = torch.tensor(sio.loadmat('modes_100Hz.mat')['km'].ravel(), 
                  device=device)          # (M,)
k0 = omega / c_w
gamma_m = torch.sqrt(k0**2 - km**2)       # (M,)

# ---------------- forward model -----------------
def pressure_field(c_b, r, z):
    """
    c_b : tensor(*)      bottom sound speed (trainable scalar!)
    r   : tensor(N)      ranges
    z   : scalar         receiver depth
    returns tensor(N)    complex pressure
    """
    kb         = omega / c_b
    gamma_bm   = torch.sqrt(torch.clamp(km**2 - kb**2, min=0.0))   # real
    Am = 1.0 / torch.sqrt(                   # normalisation (vectorised)
        D/2 - torch.sin(2*gamma_m*D)/(4*gamma_m) +
        torch.sin(gamma_m*D)**2 /(2*rho_b*gamma_bm)
    )

    # mode sum – broadcast (M,1) * (1,N)
    Zs = Am * torch.sin(gamma_m * z_s)          # (M)
    Z  = Am * torch.sin(gamma_m * z)         # (M)
    H0 = torch.special.bessel_j0(km[:,None]*r) + 1j*torch.special.bessel_y0(km[:,None]*r)

    p  = 1j/(4*rho_w) * (Zs[:,None]*Z[:,None]*H0).sum(dim=0)        # (N)
    return p

# synthetic data (add 0 % noise)
c_b_true = torch.tensor(2000.0, device=device)
p_true = pressure_field(c_b_true, r_meas, z_meas)
noise  = 0.0 * p_true.abs().max() * (torch.randn_like(p_true)+1j*torch.randn_like(p_true))
p_meas = p_true + noise
#%%
def boundary_residual(c_b, r):
    kb         = omega / c_b
    gamma_bm   = torch.sqrt(torch.clamp(km**2 - kb**2, min=0.0))
    Am         = 1.0 / torch.sqrt(
        D/2 - torch.sin(2*gamma_m*D)/(4*gamma_m) +
        torch.sin(gamma_m*D)**2 /(2*rho_b*gamma_bm)
    )
    Zs   = Am * torch.sin(gamma_m * z_s)
    Z_D  = Am * torch.sin(gamma_m * D)
    dZ_D = Am * gamma_m * torch.cos(gamma_m * D)
    H0   = torch.special.bessel_j0(km[:,None]*r) + 1j*torch.special.bessel_y0(km[:,None]*r)

    p_D      = 1j/(4*rho_w) * (Zs[:,None]*Z_D[:,None]*H0).sum(dim=0)
    dpdz_D   = 1j/(4*rho_w) * (Zs[:,None]*dZ_D[:,None]*H0).sum(dim=0)

    # impedance BC  p + (rho_b/gamma_b) dp/dz = 0  at z = D
    gamma_b = gamma_bm.max()          # simple scalar scale
    resid   = p_D + (rho_b/gamma_b) * dpdz_D
    return resid
#%%
# ------------ coarse PINN loop ------------
c_b_est = torch.tensor(2010.0, device=device, requires_grad=True)
opt     = torch.optim.Adam([c_b_est], lr=1.5)   # lr can be 0.1–2

cb_history = []
lambda_phys = 0
for epoch in range(0, 150001):
    opt.zero_grad()

    p_pred = pressure_field(c_b_est, r_meas, z_meas)
    data_loss   = torch.mean(torch.abs(p_pred - p_meas)**2)

    resid  = boundary_residual(c_b_est, r_meas)
    phys_loss  = torch.mean(torch.abs(resid)**2)

    loss = data_loss + lambda_phys * phys_loss
    loss.backward()
    opt.step()

    cb_history.append(c_b_est.item())
    # optional clamp
    with torch.no_grad():
        c_b_est.clamp_(1000., 3000.)

    if epoch % 50 == 0:
        print(f"Epoch {epoch:3d}: loss={loss.item():.3e}, lossData={data_loss.item():.3e}, lossPhy={phys_loss.item():.3e},  c_b={c_b_est.item():.1f} m/s")

