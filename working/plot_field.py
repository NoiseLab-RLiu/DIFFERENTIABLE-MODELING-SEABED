import scipy.io as sio
import torch
import numpy as np
import matplotlib.pyplot as plt
import os

# =========================================================================
# 1. CONFIGURATION
# =========================================================================
device = 'cpu'
target_freq = 150
mat_file = f'TL_kraken_2000_260122_{target_freq}.mat'

# --- FONT SETTINGS (SAFE MODE) ---
# We remove the specific request for "Computer Modern Roman" to avoid the crash.
plt.rcParams.update({
    "text.usetex": False,        # Disable external LaTeX engine
    "mathtext.fontset": "cm",    # Use Matplotlib's internal Computer Modern for math
    "font.family": "serif",      # Use default system serif (safe)
    "font.size": 16              # Base font size
})

# =========================================================================
# 2. DATA LOADING
# =========================================================================
if not os.path.exists(mat_file):
    print(f"[WARN] {mat_file} not found. Generating dummy data...")
    z = np.linspace(0, 100, 200)
    r_full = np.linspace(0, 5000, 1000)
    R, Z = np.meshgrid(r_full, z)
    # Simulating Positive TL data
    TL_2D = 20 * np.log10(R + 1e-6) + (0.5 * R/1000) - 10 * np.sin(2 * np.pi * Z / 20)
    TL_2D = np.clip(TL_2D, 30, 80) 
    TL_tensor = torch.tensor(TL_2D, device=device)
else:
    try:
        mat_data = sio.loadmat(mat_file)
        key_name = 'TL_2D' if 'TL_2D' in mat_data else 'TL'
        TL_tensor = torch.tensor(mat_data[key_name], device=device)
        print(f"[OK] Loaded {target_freq} Hz")
    except Exception as e:
        print(f"[ERROR] Failed to load {mat_file}: {e}")
        exit()

field_data = TL_tensor.cpu().numpy()
plot_data = -field_data  # Flip to negative Pressure

# =========================================================================
# 3. PLOTTING
# =========================================================================
fig, ax = plt.subplots(figsize=(10, 6))

extent = [0, 5, 100, 0] 

# A. Heatmap
im = ax.imshow(plot_data, 
               aspect='auto', 
               cmap='jet', 
               extent=extent,
               interpolation='bilinear',
               vmin=-70, vmax=-0)

# B. Source Indicator
# - Magenta Star (Visible against blue/red)
# - Clip_on=False (Visible even if on the edge)
ax.scatter(0, 25, s=400, marker='*', facecolors='magenta', edgecolors='black', 
           linewidth=1.5, zorder=50, clip_on=False, label=r'Source ($z_s=25$ m)')

# C. Labels (MathText enabled)
ax.set_title(rf'Acoustic Pressure Field ($f={target_freq}$ Hz)', fontsize=20)
ax.set_xlabel(r'Range (km)', fontsize=18)
ax.set_ylabel(r'Depth (m)', fontsize=18)

# D. Axis Ticks (Enlarged)
ax.tick_params(axis='both', which='major', labelsize=16)
ax.set_xlim(0, 5)
ax.set_ylim(100, 0)

# E. Colorbar
cbar = plt.colorbar(im, ax=ax, pad=0.02)
cbar.set_label(r'Pressure Level (dB)', fontsize=18)
cbar.ax.tick_params(labelsize=16)

plt.tight_layout()
plt.show()

#%%
import scipy.io as sio
import torch
import numpy as np
import matplotlib.pyplot as plt
import os

# =========================================================================
# 1. CONFIGURATION
# =========================================================================
device = 'cpu'
# Compare a Low Freq (Modal) vs High Freq (Ray-like)
check_freqs = [50, 150] 
zoom_range_km = 0.5  # Zoom in to first 500m to see "radiation"

# Font Settings (Safe Mode)
plt.rcParams.update({
    "text.usetex": False,
    "mathtext.fontset": "cm",
    "font.family": "serif", 
    "font.size": 14
})

# =========================================================================
# 2. HELPER FUNCTION
# =========================================================================
def load_and_plot(ax, freq, zoom_lim):
    mat_file = f'TL_kraken_2000_260122_{freq}.mat'
    
    if not os.path.exists(mat_file):
        print(f"[WARN] {mat_file} not found. using dummy.")
        # Dummy: Spherical spreading + Lloyd's mirror pattern
        z = np.linspace(0, 100, 200)
        r = np.linspace(0, 5000, 1000)
        R, Z = np.meshgrid(r, z)
        # k = 2pi * f / c (approx c=1500)
        k = 2 * np.pi * freq / 1500
        # Source at 25, Image at -25
        R_direct = np.sqrt(R**2 + (Z - 25)**2)
        R_image  = np.sqrt(R**2 + (Z + 25)**2)
        # Coherent sum (approx)
        P = (np.exp(1j * k * R_direct)/R_direct) - (np.exp(1j * k * R_image)/R_image)
        TL_2D = -20 * np.log10(np.abs(P) + 1e-6)
        TL_2D = np.clip(TL_2D, 40, 90)
    else:
        try:
            mat_data = sio.loadmat(mat_file)
            key = 'TL_2D' if 'TL_2D' in mat_data else 'TL'
            TL_2D = mat_data[key]
            # If your data is Complex Pressure, convert to TL; if TL, use as is.
            # Assuming your file is already TL (positive dB).
            # If it's pure Pressure (complex), use: -20*np.log10(abs(TL_2D))
        except:
            return

    # Convert to Negative Pressure for visualization
    plot_data = -TL_2D 
    
    # Plot
    extent = [0, 5, 100, 0] # Full extent 5km
    im = ax.imshow(plot_data, aspect='auto', cmap='jet', extent=extent,
                   vmin=-80, vmax=-0, interpolation='bilinear')
    
    # --- CRITICAL: THE ZOOM ---
    ax.set_xlim(0, zoom_lim) # Cut off at 500m
    
    # Marker
    ax.scatter(0, 25, s=200, marker='*', facecolors='magenta', edgecolors='black', 
               linewidth=1.5, clip_on=False, zorder=10, label='Source')
    
    ax.set_title(rf'$f={freq}$ Hz (Near Field)', fontsize=16)
    ax.set_xlabel(r'Range (km)', fontsize=14)
    ax.set_ylabel(r'Depth (m)', fontsize=14)
    ax.set_ylim(100, 0)
    return im

# =========================================================================
# 3. MAIN PLOT
# =========================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for i, f in enumerate(check_freqs):
    load_and_plot(axes[i], f, zoom_range_km)

# Add Colorbar (Shared)
# Adjust layout to make room for colorbar
plt.tight_layout()
fig.subplots_adjust(right=0.9)
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
cbar = fig.colorbar(axes[0].images[0], cax=cbar_ax)
cbar.set_label(r'Pressure (dB)', fontsize=14)

plt.show()
#%%
import scipy.io as sio
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

# =========================================================================
# 1. CONFIGURATION
# =========================================================================
device = 'cpu'
target_freq = 150
mat_file = f'TL_kraken_2000_260122_{target_freq}.mat'

# Font Settings (Safe Mode - No external LaTeX required)
plt.rcParams.update({
    "text.usetex": False,
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "font.size": 14
})

# =========================================================================
# 2. DATA LOADING
# =========================================================================
if not os.path.exists(mat_file):
    print(f"[WARN] {mat_file} not found. Generating dummy data...")
    z = np.linspace(0, 100, 200)
    r_full = np.linspace(0, 5000, 1000)
    R, Z = np.meshgrid(r_full, z)
    # Simulating data with near-field spherical spreading
    k = 2 * np.pi * target_freq / 1500
    R_direct = np.sqrt(R**2 + (Z - 25)**2)
    R_image  = np.sqrt(R**2 + (Z + 25)**2)
    P = (np.exp(1j * k * R_direct)/(R_direct+1e-3)) - (np.exp(1j * k * R_image)/(R_image+1e-3))
    TL_2D = -20 * np.log10(np.abs(P) + 1e-12)
    TL_2D = np.clip(TL_2D, -10, 80) # Allow high values for near field
    TL_tensor = torch.tensor(TL_2D, device=device)
else:
    try:
        mat_data = sio.loadmat(mat_file)
        key_name = 'TL_2D' if 'TL_2D' in mat_data else 'TL'
        TL_tensor = torch.tensor(mat_data[key_name], device=device)
        print(f"[OK] Loaded {target_freq} Hz")
    except Exception as e:
        print(f"[ERROR] Failed to load {mat_file}: {e}")
        exit()

field_data = TL_tensor.cpu().numpy()
plot_data = -field_data  # Convert positive TL to negative Pressure

# =========================================================================
# 3. PLOTTING
# =========================================================================
fig = plt.figure(figsize=(14, 6))

# GridSpec: 1 Row, 2 Cols. 
# width_ratios=[2.5, 1] makes the full range plot 2.5x wider than the zoom
gs = gridspec.GridSpec(1, 2, width_ratios=[2.5, 1], wspace=0.05)

ax_full = plt.subplot(gs[0])
ax_near = plt.subplot(gs[1], sharey=ax_full) # Share Y axis with left plot

extent = [0, 5, 100, 0] 

# --- A. LEFT PLOT: Full Range (0-5 km) ---
im_full = ax_full.imshow(plot_data, 
                         aspect='auto', 
                         cmap='jet', 
                         extent=extent,
                         interpolation='bilinear',
                         vmin=-70, vmax=-30) # Standard dynamic range

# Source Marker (Magenta Star)
ax_full.scatter(0, 25, s=300, marker='*', facecolors='magenta', edgecolors='black', 
                linewidth=1.2, zorder=50, clip_on=False)

ax_full.set_title(f'(a) Full Range ($0$ \u2013 $5$ km)', fontsize=16)
ax_full.set_xlabel(r'Range (km)', fontsize=14)
ax_full.set_ylabel(r'Depth (m)', fontsize=14)
ax_full.set_xlim(0, 5)
ax_full.tick_params(labelsize=14)

# Colorbar for Left (Horizontal, Bottom)
cbar_full = plt.colorbar(im_full, ax=ax_full, orientation='horizontal', pad=0.12, fraction=0.05)
cbar_full.set_label(r'Pressure Level (dB)', fontsize=14)
cbar_full.ax.tick_params(labelsize=12)


# --- B. RIGHT PLOT: Near Field (0-0.5 km) ---
# Note: We use vmin=-70, vmax=0 here to show the high-intensity near source
im_near = ax_near.imshow(plot_data, 
                         aspect='auto', 
                         cmap='jet', 
                         extent=extent,
                         interpolation='bilinear',
                         vmin=-60, vmax=0) # Extended dynamic range for near field

# Source Marker
ax_near.scatter(0, 25, s=300, marker='*', facecolors='magenta', edgecolors='black', 
                linewidth=1.2, zorder=50, clip_on=False, label=r'Source')

ax_near.set_title(f'(b) Near Field ($0$ \u2013 $0.3$ km)', fontsize=16)
ax_near.set_xlabel(r'Range (km)', fontsize=14)
ax_near.set_xlim(0, 0.3) # ZOOM IN
ax_near.tick_params(labelsize=14)

# Hide Y labels on the right plot (since they are shared)
plt.setp(ax_near.get_yticklabels(), visible=False)
ax_near.set_ylabel('')

# Legend (Only on right plot)
ax_near.legend(loc='upper right', fontsize=12, framealpha=0.9)

# Colorbar for Right (Horizontal, Bottom)
cbar_near = plt.colorbar(im_near, ax=ax_near, orientation='horizontal', pad=0.12, fraction=0.05)
cbar_near.set_label(r'Pressure Level (dB)', fontsize=14)
cbar_near.ax.tick_params(labelsize=12)

# Global Title (Optional, remove if not needed)
plt.suptitle(rf'Acoustic Pressure Field (${target_freq}$ Hz)', fontsize=18, y=0.98)

plt.show()