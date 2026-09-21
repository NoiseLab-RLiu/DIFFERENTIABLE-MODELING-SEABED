# -*- coding: utf-8 -*-
"""
Created on Sun Jan 18 21:03:35 2026

@author: 13391
"""

import numpy as np
import scipy.optimize as optimize

# Parameters
freq = 150.0
omega = 2 * np.pi * freq
c_w = 1500.0
c_b = 2000.0
rho_w = 1.0
rho_b = 1.8
D = 100.0

k0 = omega / c_w
kb = omega / c_b

print(f"k_water  (White Line) = {k0:.6f}")
print(f"k_bottom (Red Line)   = {kb:.6f}")
print("-" * 60)

# Characteristic Equation (Analytic)
def func_to_solve(k_complex):
    # k is complex scalar
    k = k_complex[0] + 1j * k_complex[1]
    
    kz_w = np.lib.scimath.sqrt(k0**2 - k**2)
    kz_b = np.lib.scimath.sqrt(kb**2 - k**2)
    
    # Pekeris Characteristic Equation
    F = rho_w * kz_b * np.sin(kz_w * D) + 1j * rho_b * kz_w * np.cos(kz_w * D)
    
    # Return vector [Real, Imag] for solver
    return [F.real, F.imag]

# Scan the real axis to find rough brackets for the 13 modes
# Theoretical approximation: k_z * D = m * pi
real_guesses = np.linspace(kb+0.001, k0-0.001, 13)

print("   Mode Type    |      Real(k)      |      Imag(k)      |  Decay (dB/km)")
print("-" * 60)

found_roots = []

# 1. FIND TRAPPED MODES (Right of Red Line)
for guess in real_guesses:
    # Use Newton-Raphson
    sol = optimize.root(func_to_solve, [guess, 0.0], tol=1e-12)
    if sol.success:
        k_found = sol.x[0] + 1j * sol.x[1]
        
        # Filter duplicates
        if not any(np.abs(k_found - r) < 1e-4 for r in found_roots):
            found_roots.append(k_found)
            
            # Check if strictly real
            is_real = abs(k_found.imag) < 1e-9
            type_str = "TRAPPED (Real)" if is_real else "COMPLEX??"
            print(f"{type_str:^15} | {k_found.real:.9f} | {k_found.imag:.9f} | {0.0:.4f}")

# 2. FIND LEAKY MODES (Left of Red Line)
# We guess to the left of k_bottom, with some imaginary part
leaky_guesses = np.linspace(kb-0.2, kb-0.01, 5)
for guess in leaky_guesses:
    # Start with a small imaginary seed
    sol = optimize.root(func_to_solve, [guess, 0.05], tol=1e-12)
    if sol.success:
        k_found = sol.x[0] + 1j * sol.x[1]
        
        # We only care about physical leaky modes (Im > 0)
        if k_found.real < kb and k_found.imag > 1e-6:
             if not any(np.abs(k_found - r) < 1e-4 for r in found_roots):
                found_roots.append(k_found)
                decay = k_found.imag * 8686 # Convert Neper/m to dB/km
                print(f"{'LEAKY':^15} | {k_found.real:.9f} | {k_found.imag:.9f} | {decay:.4f}")

print("-" * 60)
print(f"Total Unique Modes Found: {len(found_roots)}")