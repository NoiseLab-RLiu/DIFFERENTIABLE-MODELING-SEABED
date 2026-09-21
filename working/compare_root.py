# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 01:17:02 2026

@author: 13391
"""

import math
import torch

# Globals
c_w, rho_w, D = 1500.0, 1.0, 100.0
c_b, rho_b, alpha_b = 2000.0, 1.8, 0.5
freq = 150.0
omega = 2 * math.pi * freq

# Complex Solver (Same as before)
k0 = omega / c_w
kb_complex = (omega / c_b) + 1j * (alpha_b * freq / c_b / 8.686)

# Solver Loop
m = torch.arange(1, 41, dtype=torch.float64)
kz_guess = (m - 0.5) * math.pi / D
k = torch.sqrt(k0**2 - kz_guess**2).to(dtype=torch.complex128)

for _ in range(50): 
    kz_w = torch.sqrt(k0**2 - k**2)
    kz_b = torch.sqrt(kb_complex**2 - k**2)
    s_w, c_w = torch.sin(kz_w*D), torch.cos(kz_w*D)
    
    # Pekeris Characteristic Eq
    F = rho_w * kz_b * s_w + 1j * rho_b * kz_w * c_w
    
    # Derivative
    dkzw_dk, dkzb_dk = -k/kz_w, -k/kz_b
    dT1 = rho_w * (dkzb_dk * s_w + kz_b * c_w * D * dkzw_dk)
    dT2 = 1j * rho_b * (dkzw_dk * c_w - kz_w * s_w * D * dkzw_dk)
    k = k - F / (dT1 + dT2)

# Filter and Print
valid = (k.real > 0.1) & (k.real < k0 * 1.05)
k_valid = k[valid]
k_sorted = k_valid[torch.argsort(k_valid.real, descending=True)]

print("--- PYTHON CALCULATED MODES ---")
for i in range(len(k_sorted)):
    ki = k_sorted[i]
    print(f"Mode {i+1}: {ki.real:.6f} + {ki.imag:.8f}j")