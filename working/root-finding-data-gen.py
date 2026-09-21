# -*- coding: utf-8 -*-
"""
Created on Sun May 31 23:31:35 2026

@author: 13391
"""

# -*- coding: utf-8 -*-
import os, json, math, gc
import numpy as np
import torch
from tqdm import tqdm
from torch.quasirandom import SobolEngine
import PMM_inverse_mfreqs_rev as pmm

gc.collect()
torch.set_default_dtype(torch.float64)
raw_device = getattr(pmm, "device", "cuda" if torch.cuda.is_available() else "cpu")
device = torch.device(str(raw_device))
print("Using device:", device)

RHO_W = getattr(pmm, "RHO_W_GLOBAL", 1.0)
D = getattr(pmm, "D_GLOBAL", 100.0)
ZS = getattr(pmm, "ZS_GLOBAL", 25.0)
ZR = getattr(pmm, "ZR_GLOBAL", 50.0)

FREQ_LIST = [30.0, 35.0, 50.0, 100.0, 150.0, 200.0, 250.0]
N_PARAM = 3000
CB_MIN, CB_MAX = 1800.0, 2200.0
RB_MIN, RB_MAX = 1.1, 2.0
AB_MIN, AB_MAX = 0.05, 1.2
SEED = 1234

M_MAX = 256
LAYER_MODE = "aligned_adaptive"   # use "fixed100" if you want to reproduce the old .mat exactly
ROOT_OUT_DIR = r"D:\PMM_root_dataset_aligned_x64"
OVERWRITE = True
VALIDATE_FIRST_N_SEQ = 15
ROOT_DIFF_THR = 1e-12
TL_MAE_THR, TL_MAX_THR = 1e-8, 1e-6

VALIDATE_WITH_PMM_FREQ_MAX = 100.0   # or 150.0 if you also want to validate 150 Hz
VALIDATE_FIRST_N_PARAM = 15

os.makedirs(ROOT_OUT_DIR, exist_ok=True)
X_PATH = os.path.join(ROOT_OUT_DIR, "X_root.dat")
KR_PATH = os.path.join(ROOT_OUT_DIR, "K_real.dat")
KI_PATH = os.path.join(ROOT_OUT_DIR, "K_imag.dat")
KM_PATH = os.path.join(ROOT_OUT_DIR, "K_mask.dat")
MC_PATH = os.path.join(ROOT_OUT_DIR, "mode_count.dat")
PID_PATH = os.path.join(ROOT_OUT_DIR, "param_id.dat")
FID_PATH = os.path.join(ROOT_OUT_DIR, "freq_id.dat")
NL_PATH = os.path.join(ROOT_OUT_DIR, "N_layers.dat")
META_PATH = os.path.join(ROOT_OUT_DIR, "root_meta.json")

def choose_N_layers(freq):
    if LAYER_MODE == "fixed100":
        return 100
    dz_target = (1500.0 / freq) / 10.0
    N0 = max(100, math.ceil(D / dz_target))
    return int(100 * math.ceil(N0 / 100))

def profile_N(freq, N):
    dz = D / N
    z = torch.linspace(dz / 2, D - dz / 2, N, device=device, dtype=torch.float64)
    c = torch.zeros_like(z)
    c = torch.where(z <= 33.0, 1520.0 - (10.0 / 33.0) * z, c)
    c = torch.where((z > 33.0) & (z <= 67.0), 1510.0 - (20.0 / 34.0) * (z - 33.0), c)
    c = torch.where(z > 67.0, 1490.0 - (10.0 / 33.0) * (z - 67.0), c)
    return c, dz

def compute_F_N(k, cb, rb, ab, freq, N):
    omega = 2 * math.pi * freq
    c, dz = profile_N(freq, N)
    V0 = torch.zeros_like(k, dtype=torch.complex128)
    V1 = torch.ones_like(k, dtype=torch.complex128)
    for j in range(N):
        kz = torch.sqrt((omega / c[j])**2 - k**2 + 1e-12j)
        S, C = torch.sin(kz * dz), torch.cos(kz * dz)
        V0, V1 = C * V0 + (RHO_W / kz) * S * V1, -(kz / RHO_W) * S * V0 + C * V1
    kb = (omega / cb) + 1j * ((ab * (freq / cb)) / 8.686)
    kz_b = torch.sqrt(kb**2 - k**2 + 1e-12j)
    return 1j * rb * V1 + kz_b * V0

def roots_N(cb, rb, ab, freq, N):
    omega = 2 * math.pi * freq
    k0, k_min = omega / 1480.0, omega / 3000.0
    real_vals = torch.linspace(k0 * 1.01, k_min, 400, device=device)
    imag_vals = torch.tensor([0.0, 1e-4, 1e-3], device=device)
    gr, gi = torch.meshgrid(real_vals, imag_vals, indexing="ij")
    k = (gr + 1j * gi).flatten()
    for _ in range(30):
        k = k.detach().requires_grad_(True)
        with torch.enable_grad():
            F = compute_F_N(k, cb, rb, ab, freq, N)
            dFdk = torch.autograd.grad(F.real.sum(), k)[0].conj()
        k = k - F.detach() / (dFdk.detach() + 1e-20)
    k = k[(k.real > k_min) & (k.real < k0 * 1.01) & (k.imag > -1e-8)]
    if len(k) > 0:
        k = k[torch.argsort(k.real, descending=True)]
        keep = torch.cat([torch.tensor([True], device=device), torch.abs(k[1:] - k[:-1]) > 1e-4])
        k = k[keep]
    return k

def mode_values_N(k, cb, rb, ab, freq, N):
    omega = 2 * math.pi * freq
    c, dz = profile_N(freq, N)
    V0 = torch.zeros_like(k, dtype=torch.complex128)
    V1 = torch.ones_like(k, dtype=torch.complex128)
    hist = [V0]
    for j in range(N):
        kz = torch.sqrt((omega / c[j])**2 - k**2 + 1e-12j)
        S, C = torch.sin(kz * dz), torch.cos(kz * dz)
        V0, V1 = C * V0 + (RHO_W / kz) * S * V1, -(kz / RHO_W) * S * V0 + C * V1
        hist.append(V0)
    Psi = torch.stack(hist)
    kb = (omega / cb) + 1j * ((ab * (freq / cb)) / 8.686)
    kz_b = torch.sqrt(kb**2 - k**2 + 1e-12j)
    Psi_sq = (Psi.real**2 + Psi.imag**2) / RHO_W
    water = (torch.sum(Psi_sq, dim=0) - 0.5 * Psi_sq[0] - 0.5 * Psi_sq[-1]) * dz
    bottom = (V0.real**2 + V0.imag**2) / (2 * rb * torch.sqrt(kz_b.imag**2 + 1e-20) + 1e-20)
    Psi = Psi / torch.sqrt(water + bottom + 1e-20)
    izs = int(round((ZS / D) * N))
    izr = int(round((ZR / D) * N))
    return Psi[izs], Psi[izr]

def TL_from_roots(k, cb, rb, ab, freq, N, r_range):
    if len(k) == 0:
        return torch.ones_like(r_range) * 100.0
    Zs, Zr = mode_values_N(k, cb, rb, ab, freq, N)
    phase = k[:, None] * r_range[None, :]
    H0 = torch.sqrt(2.0 / (math.pi * phase)) * torch.exp(1j * (phase - math.pi / 4.0))
    p = 1j / (4.0 * RHO_W) * (Zs[:, None] * Zr[:, None] * H0).sum(dim=0)
    return -10.0 * torch.log10((p.real**2 + p.imag**2) * (4.0 * math.pi)**2 + 1e-12)

def make_params():
    sobol = SobolEngine(dimension=3, scramble=True, seed=SEED)
    u = sobol.draw(N_PARAM).to(device=device, dtype=torch.float64)
    cb = CB_MIN + (CB_MAX - CB_MIN) * u[:, 0]
    rb = RB_MIN + (RB_MAX - RB_MIN) * u[:, 1]
    ab = AB_MIN + (AB_MAX - AB_MIN) * u[:, 2]
    return cb, rb, ab

def save_roots(idx, k, Kr, Ki, Km, Mc):
    n = len(k)
    if n > M_MAX:
        raise RuntimeError(f"idx={idx}: mode count {n} exceeds M_MAX={M_MAX}")
    Kr[idx, :] = 0.0
    Ki[idx, :] = 0.0
    Km[idx, :] = False
    if n > 0:
        knp = k.detach().cpu().numpy().astype(np.complex128)
        Kr[idx, :n] = knp.real
        Ki[idx, :n] = knp.imag
        Km[idx, :n] = True
    Mc[idx] = n

def load_roots(idx, Kr, Ki, Km):
    mask = np.array(Km[idx, :], dtype=bool)
    k_np = np.array(Kr[idx, mask], dtype=np.float64) + 1j * np.array(Ki[idx, mask], dtype=np.float64)
    return torch.tensor(k_np, dtype=torch.complex128, device=device)

VALIDATE_WITH_PMM_FREQ_MAX = 150.0

def root_diagnostics(k, cb, rb, ab, freq, N):
    if len(k) == 0:
        return 0.0, 0.0, 0.0
    k_req = k.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        F = compute_F_N(k_req, cb, rb, ab, freq, N)
        dFdk = torch.autograd.grad(F.real.sum(), k_req)[0].conj()
    step = torch.abs(F.detach() / (dFdk.detach() + 1e-30))
    omega = 2 * math.pi * freq
    c, dz = profile_N(freq, N)
    V0 = torch.zeros_like(k, dtype=torch.complex128)
    V1 = torch.ones_like(k, dtype=torch.complex128)
    for j in range(N):
        kz = torch.sqrt((omega / c[j])**2 - k**2 + 1e-12j)
        S, C = torch.sin(kz * dz), torch.cos(kz * dz)
        V0, V1 = C * V0 + (RHO_W / kz) * S * V1, -(kz / RHO_W) * S * V0 + C * V1
    kb = (omega / cb) + 1j * ((ab * (freq / cb)) / 8.686)
    kz_b = torch.sqrt(kb**2 - k**2 + 1e-12j)
    term1 = 1j * rb * V1
    term2 = kz_b * V0
    F2 = term1 + term2
    relF = torch.abs(F2) / (torch.abs(term1) + torch.abs(term2) + 1e-30)
    return float(torch.max(torch.abs(F2)).item()), float(torch.max(relF).item()), float(torch.max(step).item())

def compare_roots(k1, k2):
    if len(k1) != len(k2):
        return False, float("nan"), len(k1), len(k2)
    if len(k1) == 0:
        return True, 0.0, 0, 0
    return True, float(torch.max(torch.abs(k1 - k2)).item()), len(k1), len(k2)

def validate_one(idx, X, Kr, Ki, Km, r_range):
    cb0, rb0, ab0, freq0 = X[idx, :]
    cb = torch.tensor(float(cb0), dtype=torch.float64, device=device)
    rb = torch.tensor(float(rb0), dtype=torch.float64, device=device)
    ab = torch.tensor(float(ab0), dtype=torch.float64, device=device)
    freq0 = float(freq0)
    N = choose_N_layers(freq0)

    k_saved = load_roots(idx, Kr, Ki, Km)
    k_aligned_ref = roots_N(cb, rb, ab, freq0, N)
    same_count, dk_aligned, ns, na = compare_roots(k_saved, k_aligned_ref)

    TL_saved = TL_from_roots(k_saved, cb, rb, ab, freq0, N, r_range)
    TL_aligned = TL_from_roots(k_aligned_ref, cb, rb, ab, freq0, N, r_range)
    d = (TL_saved - TL_aligned).detach().cpu().numpy()
    tl_mae = float(np.mean(np.abs(d)))
    tl_max = float(np.max(np.abs(d)))

    maxF, maxRelF, maxStep = root_diagnostics(k_saved, cb, rb, ab, freq0, N)

    msg = f"[VALIDATE idx={idx:05d}] f={freq0:7.1f} Hz, N={N:3d}, modes={len(k_saved):3d}, self_dk={dk_aligned:.3e}, self_TL_MAE={tl_mae:.3e}, maxRelF={maxRelF:.3e}, max|F/Fk|={maxStep:.3e}"

    if freq0 <= VALIDATE_WITH_PMM_FREQ_MAX:
        with torch.no_grad():
            k_pmm = pmm.unified_complex_solver(cb, rb, ab, freq0, coarse_mode=False)
        pmm_same_count, dk_pmm, _, _ = compare_roots(k_saved, k_pmm)
        msg += f", pmm_dk={dk_pmm:.3e}"
        if (not pmm_same_count) or dk_pmm > ROOT_DIFF_THR:
            raise RuntimeError(f"low-frequency pmm validation failed at idx={idx}, freq={freq0}, saved_modes={len(k_saved)}, pmm_modes={len(k_pmm)}, dk={dk_pmm:.3e}")
    else:
        with torch.no_grad():
            k_pmm = pmm.unified_complex_solver(cb, rb, ab, freq0, coarse_mode=False)
        pmm_same_count, dk_pmm, _, _ = compare_roots(k_saved, k_pmm)
        msg += f", old_pmm_modes={len(k_pmm):3d}, old_pmm_dk={dk_pmm:.3e} diagnostic_only"

    print(msg)

    if (not same_count) or dk_aligned > ROOT_DIFF_THR or tl_mae > TL_MAE_THR or tl_max > TL_MAX_THR:
        raise RuntimeError(f"aligned-model save/load validation failed at idx={idx}, freq={freq0}")

def remove_old():
    paths = [X_PATH, KR_PATH, KI_PATH, KM_PATH, MC_PATH, PID_PATH, FID_PATH, NL_PATH, META_PATH]
    if OVERWRITE:
        for p in paths:
            if os.path.exists(p):
                os.remove(p)
    else:
        for p in paths:
            if os.path.exists(p):
                raise RuntimeError(f"existing file found: {p}")

def write_meta():
    meta = {
        "dataset_type": "self_contained_modal_root_dataset",
        "layer_mode": LAYER_MODE,
        "N_PARAM": N_PARAM,
        "N_FREQ": len(FREQ_LIST),
        "N_TOTAL": N_PARAM * len(FREQ_LIST),
        "M_MAX": M_MAX,
        "X_root_path": X_PATH,
        "X_root_shape": [N_PARAM * len(FREQ_LIST), 4],
        "X_root_dtype": "float64",
        "X_root_columns": ["cb", "rho_b", "alpha_b", "freq"],
        "K_real_path": KR_PATH,
        "K_imag_path": KI_PATH,
        "K_mask_path": KM_PATH,
        "mode_count_path": MC_PATH,
        "param_id_path": PID_PATH,
        "freq_id_path": FID_PATH,
        "N_layers_path": NL_PATH,
        "K_shape": [N_PARAM * len(FREQ_LIST), M_MAX],
        "K_dtype": "float64",
        "freq_list": FREQ_LIST,
        "parameter_ranges": {"cb": [CB_MIN, CB_MAX], "rho_b": [RB_MIN, RB_MAX], "alpha_b": [AB_MIN, AB_MAX]},
        "sampling": {"method": "SobolEngine", "seed": SEED, "scramble": True},
        "validation": {"VALIDATE_FIRST_N_SEQ": VALIDATE_FIRST_N_SEQ, "ROOT_DIFF_THR": ROOT_DIFF_THR, "TL_MAE_THR": TL_MAE_THR, "TL_MAX_THR": TL_MAX_THR}
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

def generate():
    remove_old()
    n_freq = len(FREQ_LIST)
    n_total = N_PARAM * n_freq
    X = np.memmap(X_PATH, dtype=np.float64, mode="w+", shape=(n_total, 4))
    Kr = np.memmap(KR_PATH, dtype=np.float64, mode="w+", shape=(n_total, M_MAX))
    Ki = np.memmap(KI_PATH, dtype=np.float64, mode="w+", shape=(n_total, M_MAX))
    Km = np.memmap(KM_PATH, dtype=np.bool_, mode="w+", shape=(n_total, M_MAX))
    Mc = np.memmap(MC_PATH, dtype=np.int32, mode="w+", shape=(n_total,))
    Pid = np.memmap(PID_PATH, dtype=np.int32, mode="w+", shape=(n_total,))
    Fid = np.memmap(FID_PATH, dtype=np.int32, mode="w+", shape=(n_total,))
    Nlay = np.memmap(NL_PATH, dtype=np.int32, mode="w+", shape=(n_total,))
    X[:] = 0.0
    Kr[:] = 0.0
    Ki[:] = 0.0
    Km[:] = False
    Mc[:] = 0
    Pid[:] = -1
    Fid[:] = -1
    Nlay[:] = 0
    write_meta()
    cb_all, rb_all, ab_all = make_params()
    r_range = torch.arange(1000.0, 5000.0 + 2.5, 5.0, device=device, dtype=torch.float64)
    counts = []
    print(f"Output folder: {ROOT_OUT_DIR}")
    print(f"N_total={n_total}, M_MAX={M_MAX}, layer_mode={LAYER_MODE}")
    pbar = tqdm(total=n_total, desc="Generating modal roots")
    try:
        for ip in range(N_PARAM):
            cbv, rbv, abv = cb_all[ip].item(), rb_all[ip].item(), ab_all[ip].item()
            for jf, freq in enumerate(FREQ_LIST):
                idx = ip * n_freq + jf
                N = choose_N_layers(float(freq))
                cb = torch.tensor(cbv, dtype=torch.float64, device=device)
                rb = torch.tensor(rbv, dtype=torch.float64, device=device)
                ab = torch.tensor(abv, dtype=torch.float64, device=device)
                k = roots_N(cb, rb, ab, float(freq), N)
                X[idx, :] = np.array([cbv, rbv, abv, float(freq)], dtype=np.float64)
                Pid[idx], Fid[idx], Nlay[idx] = ip, jf, N
                save_roots(idx, k, Kr, Ki, Km, Mc)
                counts.append(len(k))
                if ip < VALIDATE_FIRST_N_PARAM:
                    X.flush(); Kr.flush(); Ki.flush(); Km.flush(); Mc.flush(); Pid.flush(); Fid.flush(); Nlay.flush()
                    validate_one(idx, X, Kr, Ki, Km, r_range)
                elif idx % 20 == 0:
                    X.flush(); Kr.flush(); Ki.flush(); Km.flush(); Mc.flush(); Pid.flush(); Fid.flush(); Nlay.flush()
                pbar.update(1)
    finally:
        X.flush(); Kr.flush(); Ki.flush(); Km.flush(); Mc.flush(); Pid.flush(); Fid.flush(); Nlay.flush()
        pbar.close()
    counts = np.array(counts, dtype=np.int32)
    print(f"Finished. mode count min={counts.min()}, max={counts.max()}, mean={counts.mean():.2f}")
    print(f"Saved meta: {META_PATH}")

if __name__ == "__main__":
    generate()