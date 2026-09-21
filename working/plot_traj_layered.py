import numpy as np
import matplotlib.pyplot as plt

# --- 1. Global Settings for Internal LaTeX-style Rendering ---
plt.rcParams.update({
    "text.usetex": False,               
    "font.family": "serif",             
    "font.serif": ["STIXGeneral"],      
    "mathtext.fontset": "stix",         
    "axes.labelsize": 18,               # Amplified axis label size
    "xtick.labelsize": 14,              # Amplified x-axis tick size
    "ytick.labelsize": 14,              # Amplified y-axis tick size
    "legend.fontsize": 14               # Amplified legend font size
})

# --- 2. Load and Parse the Log Data ---
log_text = """Iter 0000 | Avg Loss: 18.5755 | cb: 1950.0 | rho: 1.200 | alpha: 0.100
Iter 0001 | Avg Loss: 18.5755 | cb: 1951.0 | rho: 1.190 | alpha: 0.110
Iter 0002 | Avg Loss: 17.6546 | cb: 1952.2 | rho: 1.178 | alpha: 0.122
Iter 0003 | Avg Loss: 16.4043 | cb: 1953.6 | rho: 1.164 | alpha: 0.136
Iter 0004 | Avg Loss: 15.2394 | cb: 1955.4 | rho: 1.146 | alpha: 0.154
Iter 0005 | Avg Loss: 14.2684 | cb: 1957.4 | rho: 1.126 | alpha: 0.174
Iter 0006 | Avg Loss: 13.3865 | cb: 1959.9 | rho: 1.126 | alpha: 0.199
Iter 0007 | Avg Loss: 12.7769 | cb: 1962.9 | rho: 1.136 | alpha: 0.229
Iter 0008 | Avg Loss: 12.0415 | cb: 1966.5 | rho: 1.148 | alpha: 0.265
Iter 0009 | Avg Loss: 11.4066 | cb: 1966.5 | rho: 1.148 | alpha: 0.308
Iter 0010 | Avg Loss: 10.6526 | cb: 1968.3 | rho: 1.155 | alpha: 0.360
Iter 0011 | Avg Loss: 9.8292 | cb: 1970.4 | rho: 1.162 | alpha: 0.422
Iter 0012 | Avg Loss: 8.8206 | cb: 1973.0 | rho: 1.171 | alpha: 0.496
Iter 0013 | Avg Loss: 7.9580 | cb: 1976.1 | rho: 1.182 | alpha: 0.585
Iter 0014 | Avg Loss: 7.0003 | cb: 1979.8 | rho: 1.195 | alpha: 0.692
Iter 0015 | Avg Loss: 5.8581 | cb: 1984.3 | rho: 1.195 | alpha: 0.820
Iter 0016 | Avg Loss: 5.2455 | cb: 1989.6 | rho: 1.201 | alpha: 0.820
Iter 0017 | Avg Loss: 4.5142 | cb: 1996.1 | rho: 1.209 | alpha: 0.756
Iter 0018 | Avg Loss: 3.8307 | cb: 2003.8 | rho: 1.218 | alpha: 0.756
Iter 0019 | Avg Loss: 2.7513 | cb: 2013.0 | rho: 1.229 | alpha: 0.724
Iter 0020 | Avg Loss: 1.9419 | cb: 2013.0 | rho: 1.243 | alpha: 0.686
Iter 0021 | Avg Loss: 1.2804 | cb: 2017.6 | rho: 1.259 | alpha: 0.686
Iter 0022 | Avg Loss: 0.7602 | cb: 2023.2 | rho: 1.278 | alpha: 0.705
Iter 0023 | Avg Loss: 0.3518 | cb: 2023.2 | rho: 1.301 | alpha: 0.728
Iter 0024 | Avg Loss: 0.1758 | cb: 2025.9 | rho: 1.301 | alpha: 0.756
Iter 0025 | Avg Loss: 0.0399 | cb: 2029.3 | rho: 1.290 | alpha: 0.789
Iter 0026 | Avg Loss: 0.3533 | cb: 2029.3 | rho: 1.290 | alpha: 0.789
Iter 0027 | Avg Loss: 0.3533 | cb: 2027.6 | rho: 1.295 | alpha: 0.772
Iter 0028 | Avg Loss: 0.0881 | cb: 2025.6 | rho: 1.302 | alpha: 0.772
Iter 0029 | Avg Loss: 0.0262 | cb: 2025.6 | rho: 1.302 | alpha: 0.781
Iter 0030 | Avg Loss: 0.0186 | cb: 2026.6 | rho: 1.299 | alpha: 0.791
Iter 0031 | Avg Loss: 0.0137 | cb: 2026.6 | rho: 1.299 | alpha: 0.803
Iter 0032 | Avg Loss: 0.0123 | cb: 2026.1 | rho: 1.301 | alpha: 0.803
Iter 0033 | Avg Loss: 0.0002 | cb: 2026.1 | rho: 1.301 | alpha: 0.797
Iter 0034 | Avg Loss: 0.0005 | cb: 2025.9 | rho: 1.300 | alpha: 0.797
Iter 0035 | Avg Loss: 0.0002 | cb: 2025.9 | rho: 1.299 | alpha: 0.800
Iter 0036 | Avg Loss: 0.0013 | cb: 2025.7 | rho: 1.299 | alpha: 0.800
Iter 0037 | Avg Loss: 0.0011 | cb: 2025.6 | rho: 1.299 | alpha: 0.798
Iter 0038 | Avg Loss: 0.0010 | cb: 2025.6 | rho: 1.299 | alpha: 0.796
Iter 0039 | Avg Loss: 0.0011 | cb: 2025.7 | rho: 1.299 | alpha: 0.796
Iter 0040 | Avg Loss: 0.0008 | cb: 2025.8 | rho: 1.299 | alpha: 0.797
Iter 0041 | Avg Loss: 0.0007 | cb: 2025.9 | rho: 1.299 | alpha: 0.798
Iter 0042 | Avg Loss: 0.0006 | cb: 2025.9 | rho: 1.299 | alpha: 0.800
Iter 0043 | Avg Loss: 0.0004 | cb: 2025.8 | rho: 1.299 | alpha: 0.800
Iter 0044 | Avg Loss: 0.0003 | cb: 2025.8 | rho: 1.300 | alpha: 0.799
Iter 0045 | Avg Loss: 0.0002 | cb: 2025.8 | rho: 1.300 | alpha: 0.799
Iter 0046 | Avg Loss: 0.0002 | cb: 2025.9 | rho: 1.300 | alpha: 0.799
Iter 0047 | Avg Loss: 0.0001 | cb: 2025.9 | rho: 1.300 | alpha: 0.800
Iter 0048 | Avg Loss: 0.0001 | cb: 2026.0 | rho: 1.300 | alpha: 0.800
Iter 0049 | Avg Loss: 0.0000 | cb: 2026.0 | rho: 1.300 | alpha: 0.799
Iter 0050 | Avg Loss: 0.0000 | cb: 2026.0 | rho: 1.300 | alpha: 0.799
Iter 0051 | Avg Loss: 0.0000 | cb: 2026.0 | rho: 1.300 | alpha: 0.800
Iter 0052 | Avg Loss: 0.0000 | cb: 2026.0 | rho: 1.300 | alpha: 0.800
Iter 0053 | Avg Loss: 0.0000 | cb: 2026.0 | rho: 1.300 | alpha: 0.800
Iter 0054 | Avg Loss: 0.0000 | cb: 2026.0 | rho: 1.300 | alpha: 0.800
Iter 0055 | Avg Loss: 0.0000 | cb: 2026.0 | rho: 1.300 | alpha: 0.800
Iter 0056 | Avg Loss: 0.0000 | cb: 2026.0 | rho: 1.300 | alpha: 0.800
Iter 0057 | Avg Loss: 0.0000 | cb: 2026.0 | rho: 1.300 | alpha: 0.800
Iter 0058 | Avg Loss: 0.0000 | cb: 2026.0 | rho: 1.300 | alpha: 0.800
Iter 0059 | Avg Loss: 0.0000 | cb: 2026.0 | rho: 1.300 | alpha: 0.800
Iter 0060 | Avg Loss: 0.0000 | cb: 2026.0 | rho: 1.300 | alpha: 0.800"""

epochs = []
cbs = []
rhos = []
alphas = []

for line in log_text.strip().split('\n'):
    parts = line.split('|')
    epochs.append(int(parts[0].replace('Iter', '').strip()))
    cbs.append(float(parts[2].split(':')[1].strip()))
    rhos.append(float(parts[3].split(':')[1].strip()))
    alphas.append(float(parts[4].split(':')[1].strip()))

epochs = np.array(epochs)
cbs = np.array(cbs)
rhos = np.array(rhos)
alphas = np.array(alphas)

# --- 3. Ground Truth Values ---
GT_CB = 2026.0
GT_RHO = 1.300
GT_ALPHA = 0.800

# --- 4. Figure Setup ---
fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
fig.subplots_adjust(hspace=0)

line_width = 2.5
grid_alpha = 0.5
colors = ['#1f77b4', '#d62728', '#2ca02c'] # Blue, Red, Green
legend_loc = 'lower right'

# --- 5. Plot cb (Top) ---
axes[0].plot(epochs, cbs, color=colors[0], linewidth=line_width, label='Estimated')
axes[0].axhline(GT_CB, color=colors[0], linestyle='--', linewidth=line_width, label='True')
axes[0].set_ylabel(r'$\hat{c}_{\mathrm{b}}$ (m/s)')
axes[0].grid(True, linestyle=':', alpha=grid_alpha)
axes[0].legend(loc=legend_loc, framealpha=1.0)

# --- 6. Plot rhob (Middle) ---
axes[1].plot(epochs, rhos, color=colors[1], linewidth=line_width, label='Estimated')
axes[1].axhline(GT_RHO, color=colors[1], linestyle='--', linewidth=line_width, label='True')
axes[1].set_ylabel(r'$\hat{\rho}_{\mathrm{b}}$ (g/cm$^3$)')
axes[1].grid(True, linestyle=':', alpha=grid_alpha)
axes[1].legend(loc=legend_loc, framealpha=1.0)

# --- 7. Plot alphab (Bottom) ---
axes[2].plot(epochs, alphas, color=colors[2], linewidth=line_width, label='Estimated')
axes[2].axhline(GT_ALPHA, color=colors[2], linestyle='--', linewidth=line_width, label='True')
axes[2].set_ylabel(r'$\hat{\alpha}_{\mathrm{b}}$ (dB/$\lambda$)')
axes[2].set_xlabel('Epoch')
axes[2].grid(True, linestyle=':', alpha=grid_alpha)
axes[2].legend(loc=legend_loc, framealpha=1.0)

# --- 8. Final Polish ---
for ax in axes:
    # We no longer need ax.tick_params(labelsize=...) since rcParams handles it!
    if ax != axes[0]:
        yticks = ax.yaxis.get_major_ticks()
        if yticks:
            yticks[-1].label1.set_visible(False)

plt.show()