# -*- coding: utf-8 -*-
"""
Created on Sun Jan 18 20:39:48 2026

@author: 13391
"""

import numpy as np
import matplotlib.pyplot as plt
import cmath

# ─────────────────────────── PHYSICS CONSTANTS ──────────────────────────
freq = 150.0            # Hz
omega = 2 * np.pi * freq
c_w = 1500.0            # Water speed
c_b = 2000.0            # Bottom speed
rho_w = 1.0
rho_b = 1.8             # Density contrast (controls leakage amount)
D = 100.0               # Depth
alpha_b = 0.0           # Material attenuation (set to 0 to see pure geometric leakage)

# Wavenumber Limits
k0 = omega / c_w
kb_real = omega / c_b

print(f"k_water  = {k0:.4f} rad/m")
print(f"k_bottom = {kb_real:.4f} rad/m (Critical Angle Cutoff)")

# ─────────────────────────── GRID SETUP ────────────────────────────────
# Real axis: From slightly below k_bottom up to k_water
k_real = np.linspace(kb_real * 0.5, k0 * 1.05, 400)

# Imaginary axis: From 0 (Trapped) up to high attenuation
# Note: In some conventions, decay is Positive Imag, in others Negative.
# We scan both just to be sure, but usually for exp(i*k*r), decay is Positive k_i.
k_imag = np.linspace(-0.05, 0.15, 200) 

K_real, K_imag = np.meshgrid(k_real, k_imag)
K_complex = K_real + 1j * K_imag

# ─────────────────────────── CHARACTERISTIC FUNCTION ─────────────────────
# We use the robust form: rho_w * k_zb * sin(k_zw*D) + i * rho_b * k_zw * cos(k_zw*D)
def characteristic_eq(k, w, cw, cb, rw, rb, D):
    # Vertical wavenumber in water
    # k_zw = sqrt(k0^2 - k^2)
    # If k > k0, this becomes imaginary (evanescent)
    k0 = w / cw
    k_zw = np.sqrt(k0**2 - k**2 + 0j)
    
    # Vertical wavenumber in bottom
    # k_zb = sqrt(kb^2 - k^2)
    kb = w / cb
    k_zb = np.sqrt(kb**2 - k**2 + 0j)
    
    # The Equation
    # Standard Pekeris: tan(k_zw*D) + i*(rho_w/rho_b)*(k_zw/k_zb) = 0
    # Cross-multiplied to avoid singularities:
    term1 = rw * k_zb * np.sin(k_zw * D)
    term2 = 1j * rb * k_zw * np.cos(k_zw * D)
    
    return term1 + term2

# Compute Surface
F_val = characteristic_eq(K_complex, omega, c_w, c_b, rho_w, rho_b, D)

# Log Magnitude for visibility (like dB)
F_mag = 20 * np.log10(np.abs(F_val) + 1e-9)

# ─────────────────────────── PLOTTING ──────────────────────────────────
plt.figure(figsize=(12, 8))
plt.pcolormesh(K_real, K_imag, F_mag, shading='auto', cmap='jet_r', vmin=-40, vmax=60)
plt.colorbar(label="Log Magnitude |F(k)| (dB)")

# Mark Key Wavenumbers
plt.axvline(k0, color='w', linestyle='--', label='k_water (1500 m/s)')
plt.axvline(kb_real, color='r', linestyle='--', label='k_bottom (2000 m/s)')
plt.axhline(0, color='k', linewidth=0.5)

# Annotations
plt.text(k0, 0.02, " Trapped Modes Range", color='white', fontweight='bold', ha='right')
plt.text(kb_real, 0.05, " Leaky Modes Range \n (Complex)", color='red', fontweight='bold', ha='right')

plt.title(f"Visualizing Roots in the Complex Plane (Freq={freq}Hz, rho_b={rho_b})")
plt.xlabel("Real Part of k (rad/m)")
plt.ylabel("Imaginary Part of k (Attenuation)")
plt.legend()
plt.grid(True, alpha=0.3)

# ─── FINDING APPROX ROOTS FOR VISUALIZATION ───
# Simple local minima finder
from scipy.ndimage import minimum_filter
local_min = minimum_filter(np.abs(F_val), size=5) == np.abs(F_val)
# Filter noise
threshold = np.abs(F_val) < 0.5 # Close to zero
roots_mask = local_min & threshold
root_k = K_complex[roots_mask]

plt.scatter(root_k.real, root_k.imag, color='white', edgecolors='black', s=50, label='Roots', zorder=5)

plt.show()