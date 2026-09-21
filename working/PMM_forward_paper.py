# -*- coding: utf-8 -*-
"""PMM_forward.py 的论文公式修订副本（CPU / float64 / complex128）。

Spyder: 直接运行本文件。批处理: python PMM_forward_paper.py --no-show
PAPER CHANGE 标出相对原程序的重要改动。反演脚本复用本文件的正演，
导入本文件不会生成数据或弹出窗口。

模态范围：从无损束缚模态连续延拓得到的含损耗模态，要求海底向下衰减。
这是论文有限模态和式的一个明确选择；不包含空间增长的泄漏模态/连续谱。
"""

from pathlib import Path
from functools import lru_cache
import argparse
import json
import math
import os

# Windows/Anaconda: 避免 numpy MKL 与 PyTorch 同时加载不同的 OpenMP runtime。
# 使用顺序 MKL，而不是原脚本的 KMP_DUPLICATE_LIB_OK 绕过检查。
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

import torch
import numpy as np
from scipy.special import hankel1
from numpy.polynomial.legendre import leggauss

# =========================================================================
# 1. GLOBALS & SETUP (保留原脚本的物理设置)
# =========================================================================
torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)
device = "cpu"  # SciPy 复 Hankel 与 numpy 求根均在 CPU 上。
FREQ_LIST = [30.0, 35.0, 50.0, 100.0, 150.0, 200.0, 250.0]
D_GLOBAL = 100.0
RHO_W_GLOBAL = 1.0  # 与论文实验一致：g/cm^3；全程序使用同一密度尺度。
ZS_GLOBAL = 25.0
ZR_GLOBAL = 50.0
TRUE_PARAMS = np.array([2026.0, 1.3, 0.8])  # m/s, g/cm^3, dB/lambda
N_LAYERS = 200  # PAPER CHANGE: 固定层数；正演/反演一致，可改 1000 做网格检查。
SSP_Z = np.array([0.0, 33.0, 67.0, 100.0])
SSP_C = np.array([1520.0, 1510.0, 1490.0, 1480.0])
OUTPUT_DIR = Path(__file__).resolve().parent / "PMM_paper_results"
FORMULA_VERSION = "paper_bilinear_complex_pressure_v1"


def validate_params(theta):
    cb, rb, ab = np.asarray(theta, dtype=float)
    if not np.all(np.isfinite(theta)) or cb <= max(SSP_C) or rb <= 0 or ab < 0:
        raise ValueError("此束缚模态基线要求 cb > max(SSP), rho_b > 0, alpha_b >= 0。")


@lru_cache(maxsize=16)
def _profile(n_layers):
    if n_layers < 1:
        raise ValueError("n_layers 必须为正整数")
    dz = D_GLOBAL / n_layers
    z_nodes = np.linspace(0.0, D_GLOBAL, n_layers + 1)
    z_mid = 0.5 * (z_nodes[1:] + z_nodes[:-1])
    return np.interp(z_mid, SSP_Z, SSP_C), dz, z_nodes


def get_layered_profile(freq, n_layers=N_LAYERS):
    """原反演函数名保留；本版固定离散层数，不再随频率切换网格。"""
    c, dz, _ = _profile(n_layers)
    return torch.as_tensor(c.copy(), device=device), dz, n_layers


def _coefficients_np(k, c, length):
    """P 的 C=cos(q*l), B=sin(q*l)/q 及其解析 k 导数。

    小 q*l 用同一解析函数的 Taylor 极限，消除 0/0（不改变特征方程）。
    """
    q2 = c - k**2  # c 此处已经是 (omega/c_j)^2，可广播。
    y = q2 * length**2
    small = np.abs(y) < 1e-6
    q_safe = np.sqrt(np.where(small, 1.0 + 0j, q2))
    C = np.where(small, 1-y/2+y**2/24-y**3/720,
                 np.cos(q_safe * length))
    B = np.where(small, length*(1-y/6+y**2/120-y**3/5040),
                 np.sin(q_safe * length)/q_safe)
    Ck = k * length * B
    Bk = np.where(small, k*length**3*(1/3-y/30+y**2/840-y**3/45360),
                  k*(B-length*C)/np.where(small, 1.0+0j, q2))
    return q2, C, B, Ck, Bk


def compute_F_and_partials(k, cb, rb, ab, freq, n_layers=N_LAYERS,
                           derivatives=True):
    """PAPER CHANGE: 论文的 P_j 与 dP_j/dk 联合递推；numpy 仅用于加速。

    v=[Psi, Psi'/rho_w], F=Bv=rho_w*q_b*v0+i*rho_w*rho_b*v1。
    返回 F, dF/dk, [dF/dcb,dF/drb,dF/dab], 相对边界残差。
    所有偏导均保持其他独立变量固定；无 IFT 分母截断。
    """
    k = np.atleast_1d(np.asarray(k, dtype=np.complex128))
    c, dz, _ = _profile(n_layers)
    omega = 2*math.pi*freq
    q2, C, B, Ck, Bk = _coefficients_np(k[None, :],
                                        (omega/c[:, None])**2, dz)
    V0, V1 = np.zeros_like(k), np.ones_like(k)
    W0, W1 = np.zeros_like(k), np.zeros_like(k)  # dv/dk
    rw = RHO_W_GLOBAL
    for j in range(n_layers):
        if derivatives:
            W0_new = Ck[j]*V0 + rw*Bk[j]*V1 + C[j]*W0 + rw*B[j]*W1
            W1_new = ((2*k*B[j]-q2[j]*Bk[j])/rw)*V0 + Ck[j]*V1 \
                - (q2[j]*B[j]/rw)*W0 + C[j]*W1
            W0, W1 = W0_new, W1_new
        V0_new = C[j]*V0 + rw*B[j]*V1
        V1_new = -(q2[j]*B[j]/rw)*V0 + C[j]*V1
        V0, V1 = V0_new, V1_new
    # PAPER CHANGE: 完全采用论文的 alpha/54.58。
    kb = omega/cb*(1+1j*ab/54.58)
    qb = np.sqrt(kb**2-k**2)
    term1, term2 = rw*qb*V0, 1j*rw*rb*V1
    F = term1 + term2
    relative = np.abs(F)/(np.abs(term1)+np.abs(term2)+np.finfo(float).tiny)
    if not derivatives:
        return F, None, None, relative
    Fk = rw*((-k/qb)*V0+qb*W0) + 1j*rw*rb*W1
    Fcb = rw*V0*(kb/qb)*(-kb/cb)
    Frb = 1j*rw*V1
    Fab = rw*V0*(kb/qb)*(1j*omega/(54.58*cb))
    return F, Fk, np.column_stack((Fcb, Frb, Fab)), relative


def _lossless_brackets(theta, freq, n_layers, scan_points=2049):
    cb, rb, _ = theta
    c, _, _ = _profile(n_layers)
    omega = 2*math.pi*freq
    lower, upper = omega/cb, omega/min(c)
    grid = np.linspace(lower+1e-10*(upper-lower),
                       upper-1e-10*(upper-lower), scan_points)
    values = compute_F_and_partials(grid, cb, rb, 0., freq, n_layers, False)[0].imag
    if not np.all(np.isfinite(values)):
        raise RuntimeError("实轴扫描溢出；请检查波导设置和传播矩阵稳定性。")
    ind = np.flatnonzero(np.signbit(values[:-1]) != np.signbit(values[1:]))
    if not len(ind):
        raise RuntimeError(f"{freq:g} Hz 在当前束缚模态区间没有找到根。")
    return grid[ind], grid[ind+1], values[ind]


def _newton_refine(k, theta, freq, n_layers):
    k = np.asarray(k, dtype=np.complex128).copy()
    for _ in range(40):
        F, Fk, _, residual = compute_F_and_partials(k, *theta, freq, n_layers)
        if not np.all(np.isfinite(Fk)) or np.any(np.abs(Fk) < 1e-14):
            raise RuntimeError("Newton 遇到病态根；停止而不是裁剪根梯度。")
        step = F/Fk
        if np.max(residual) < 1e-10 and np.max(np.abs(step)) < 1e-11:
            return k
        k -= step  # 论文的 complex Newton-Raphson 更新。
        if not np.all(np.isfinite(k)):
            break
    raise RuntimeError(f"{freq:g} Hz 复 Newton 未通过残差/步长检查。")


def solve_modes(theta, freq, n_layers=N_LAYERS, initial_seeds=None):
    """保留实扫描+复 Newton 思路，明确筛选衰减分支并检查根数。

    每次扫描核对无损束缚模态根数；首次从 alpha=0 连续延拓。
    热启动只来自上一轮预测参数，不读取合成数据的真值模态。
    """
    theta = np.asarray(theta, dtype=float)
    validate_params(theta)
    lo, hi, flo = _lossless_brackets(theta, freq, n_layers)
    if initial_seeds is not None and len(initial_seeds) == len(lo):
        try:
            candidate = _newton_refine(initial_seeds, theta, freq, n_layers)
            _check_modes(candidate, theta, freq, n_layers, len(lo))
            return candidate
        except RuntimeError:
            pass  # 重建该参数下的种子，再连续延拓。
    # 向量化二分仅用于可靠初始化；含损耗的最终根由复 Newton 求得。
    for _ in range(43):
        mid = (lo+hi)/2
        fm = compute_F_and_partials(mid, theta[0], theta[1], 0., freq,
                                    n_layers, False)[0].imag
        left = np.signbit(flo) != np.signbit(fm)
        hi = np.where(left, mid, hi)
        lo, flo = np.where(left, lo, mid), np.where(left, flo, fm)
    k = ((lo+hi)/2).astype(np.complex128)
    for alpha in np.linspace(0., theta[2], max(2, math.ceil(theta[2]/0.15)+1))[1:]:
        k = _newton_refine(k, [theta[0], theta[1], alpha], freq, n_layers)
    k = k[np.argsort(k.real)[::-1]]
    _check_modes(k, theta, freq, n_layers, len(lo))
    return k


def _check_modes(k, theta, freq, n_layers, expected):
    kb = 2*math.pi*freq/theta[0]*(1+1j*theta[2]/54.58)
    qb = np.sqrt(kb**2-k**2)
    rel = compute_F_and_partials(k, *theta, freq, n_layers, False)[3]
    if len(k) != expected or not np.all(np.isfinite(k)) or np.max(rel) > 1e-9:
        raise RuntimeError("模态数或特征方程残差检查失败。")
    if np.any(k.imag < -1e-10) or np.any(qb.imag <= 0):
        raise RuntimeError("出现非衰减模态；不满足本版海底无穷积分的分支约定。")
    if len(k)>1 and np.min(np.abs(np.diff(k[np.argsort(k.real)]))) < 1e-8:
        raise RuntimeError("发现重复模态根。")


def _coefficients_torch(k, omega_over_c_squared, length):
    q2 = omega_over_c_squared-k**2
    y = q2*length**2
    small = y.abs() < 1e-6
    q_safe = torch.sqrt(torch.where(small, torch.ones_like(q2), q2))
    C = torch.where(small, 1-y/2+y**2/24-y**3/720, torch.cos(q_safe*length))
    B = torch.where(small, length*(1-y/6+y**2/120-y**3/5040),
                    torch.sin(q_safe*length)/q_safe)
    return q2, C, B


def compute_prop_matrix(k, freq, cb, rb, ab, return_modes=False,
                        n_layers=N_LAYERS, quadrature_order=8):
    """保留原正演函数名和 v0/v1 递推。

    PAPER CHANGE: u^2/rho 的水层积分使用每层 Gauss-Legendre 积分；
    海底尾项为 +i*u(D)^2/(2*rho_b*q_b)，不使用 abs(u)^2。
    返回模式时输出 F, 归一化模态节点, 正确的 N+1 深度节点,
    未归一化历史, 复归一化因子，便于精确求源/接收深度处的值。
    """
    k = k.to(torch.complex128)
    omega = 2*math.pi*freq
    c, dz, _ = get_layered_profile(freq, n_layers)
    q2, C, B = _coefficients_torch(k[None, :], (omega/c[:, None])**2, dz)
    V0, V1 = torch.zeros_like(k), torch.ones_like(k)
    Psi_history, Deriv_history = [V0], [V1]
    for j in range(n_layers):
        V0_new = C[j]*V0 + RHO_W_GLOBAL*B[j]*V1
        V1_new = -(q2[j]*B[j]/RHO_W_GLOBAL)*V0 + C[j]*V1
        V0, V1 = V0_new, V1_new
        if return_modes:
            Psi_history.append(V0)
            Deriv_history.append(V1)
    kb = omega/cb*(1+1j*ab/54.58)
    qb = torch.sqrt(kb**2-k**2)
    F = RHO_W_GLOBAL*qb*V0 + 1j*RHO_W_GLOBAL*rb*V1
    if not return_modes:
        return F
    U, V = torch.stack(Psi_history), torch.stack(Deriv_history)
    nodes, weights = leggauss(quadrature_order)
    offsets = torch.as_tensor((nodes+1)*dz/2)[None, :, None]
    weights = torch.as_tensor(weights*dz/2)[None, :, None]
    _, CQ, BQ = _coefficients_torch(k[None, None, :],
                                     (omega/c[:, None, None])**2, offsets)
    values = CQ*U[:-1, None, :] + RHO_W_GLOBAL*BQ*V[:-1, None, :]
    integral_water = (weights*values.square()/RHO_W_GLOBAL).sum(dim=(0, 1))
    integral_bottom = 1j*U[-1].square()/(2*rb*qb)
    norm = torch.sqrt(integral_water+integral_bottom)
    if not torch.isfinite(norm).all() or (norm.abs() < 1e-12).any():
        raise RuntimeError("复模态归一化接近零或非有限，请检查模态分支。")
    z_nodes = torch.linspace(0., D_GLOBAL, n_layers+1)
    return F, U/norm, z_nodes, (U, V), norm


def mode_at_depth(k, freq, depth, history, norm, n_layers):
    """PAPER CHANGE: 层内精确传播到指定深度，避免 round(depth/dz) 偏移。"""
    if not 0 <= depth <= D_GLOBAL:
        raise ValueError("深度必须在水层内。")
    U, V = history
    if depth == D_GLOBAL:
        return U[-1]/norm
    c, dz, _ = get_layered_profile(freq, n_layers)
    j = min(int(depth/dz), n_layers-1)
    _, C, B = _coefficients_torch(k, (2*math.pi*freq/c[j])**2, depth-j*dz)
    return (C*U[j]+RHO_W_GLOBAL*B*V[j])/norm


class Hankel0(torch.autograd.Function):
    """PAPER CHANGE: 精确 H0^(1)，解析反向 dH0/dz=-H1，保留复梯度。"""
    @staticmethod
    def forward(ctx, z):
        z_np = z.detach().cpu().numpy()
        h0 = torch.as_tensor(hankel1(0, z_np), dtype=z.dtype, device=z.device)
        hp = torch.as_tensor(-hankel1(1, z_np), dtype=z.dtype, device=z.device)
        ctx.save_for_backward(hp)
        return h0

    @staticmethod
    def backward(ctx, grad_output):
        (hp,) = ctx.saved_tensors
        return grad_output*hp.conj()


def pressure_from_roots(k, theta, freq, ranges, n_layers=N_LAYERS,
                        receiver_depth=ZR_GLOBAL, full_field=False):
    cb, rb, ab = theta.unbind()
    _, psi, z_nodes, history, norm = compute_prop_matrix(
        k, freq, cb, rb, ab, True, n_layers)
    Zs = mode_at_depth(k, freq, ZS_GLOBAL, history, norm, n_layers)
    Zr = psi if full_field else mode_at_depth(
        k, freq, receiver_depth, history, norm, n_layers)[None, :]
    H0 = Hankel0.apply(k[:, None]*ranges[None, :])
    # 论文 pressure_sum：不再把 TL 或 4*pi 修正混入观测。
    pressure = 1j/(4*RHO_W_GLOBAL)*((Zr*Zs) @ H0)
    return (pressure if full_field else pressure[0]), z_nodes


def forward_model(theta, freq, ranges, n_layers=N_LAYERS, full_field=False):
    roots = solve_modes(theta, freq, n_layers)
    with torch.no_grad():
        p, z = pressure_from_roots(torch.as_tensor(roots), torch.as_tensor(theta),
                                   freq, torch.as_tensor(ranges), n_layers,
                                   full_field=full_field)
    return p.numpy(), z.numpy(), roots


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--layers", type=int, default=N_LAYERS)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    if args.no_show:
        import matplotlib
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # PAPER CHANGE: 正确的 HLA 观测坐标：1000:5:5000，共 801 点。
    r_meas = np.arange(1000., 5000.1, 5.)
    r_vec = np.linspace(1., 5000., 1001)
    report = {"formula_version": FORMULA_VERSION, "n_layers": args.layers,
              "true_params": TRUE_PARAMS.tolist(), "frequencies": {}}
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for freq in FREQ_LIST:
        p_meas, _, roots = forward_model(TRUE_PARAMS, freq, r_meas, args.layers)
        rel = compute_F_and_partials(roots, *TRUE_PARAMS, freq, args.layers)[3]
        np.savez(args.output_dir/f"paper_layered_{int(freq)}.npz",
                 P_meas=p_meas, r_meas=r_meas, frequency=freq,
                 true_params=TRUE_PARAMS, n_layers=args.layers,
                 D=D_GLOBAL, rho_w=RHO_W_GLOBAL, zs=ZS_GLOBAL, zr=ZR_GLOBAL,
                 ssp_z=SSP_Z, ssp_c=SSP_C, roots=roots,
                 formula_version=FORMULA_VERSION)
        report["frequencies"][str(int(freq))] = {
            "mode_count": len(roots), "max_relative_boundary_residual": float(max(rel))}
        axes[0].plot(r_meas/1000, p_meas.real, label=f"{freq:g} Hz")
        axes[1].plot(r_meas/1000, p_meas.imag)
        print(f"{freq:5g} Hz: {len(roots):2d} modes, boundary residual {max(rel):.2e}")
        if freq == 150.:
            with torch.no_grad():
                field, z = pressure_from_roots(torch.as_tensor(roots),
                    torch.as_tensor(TRUE_PARAMS), freq, torch.as_tensor(r_vec),
                    args.layers, full_field=True)
            field, z = field.numpy(), z.numpy()
            np.savez(args.output_dir/"paper_layered_field_150.npz",
                     P_2D=field, r_vec=r_vec, z_vec=z, roots=roots)
            fig_field, ax = plt.subplots(1, 2, figsize=(12, 5),
                                         gridspec_kw={"width_ratios": [3, 1]})
            level = 20*np.log10(np.maximum(np.abs(field), 1e-30))
            vmax = float(np.percentile(level[1:], 98))
            im = ax[0].pcolormesh(r_vec/1000, z, level, shading="auto",
                                  cmap="viridis", vmin=vmax-60, vmax=vmax)
            ax[0].plot(0, ZS_GLOBAL, "r*", markersize=12, clip_on=False)
            ax[0].plot([1, 5], [ZR_GLOBAL]*2, "w--", label="HLA")
            ax[0].set(xlabel="Range (km)", ylabel="Depth (m)", title="150 Hz: paper modal field")
            ax[0].invert_yaxis()
            ax[0].legend()
            fig_field.colorbar(im, ax=ax[0], label="20 log10 |P| (formula reference)")
            ax[1].plot(SSP_C, SSP_Z, "o-")
            ax[1].set(xlabel="Sound speed (m/s)", ylabel="Depth (m)", title="Known SSP")
            ax[1].invert_yaxis()
            fig_field.tight_layout()
            fig_field.savefig(args.output_dir/"forward_field_150.png", dpi=160)
    axes[0].set(ylabel="Re(P)", title="Synthetic complex-pressure observations at z=50 m")
    axes[0].legend(ncol=4)
    axes[1].set(xlabel="Range (km)", ylabel="Im(P)")
    fig.tight_layout()
    fig.savefig(args.output_dir/"forward_observations.png", dpi=160)
    (args.output_dir/"forward_diagnostics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved to {args.output_dir}")
    if not args.no_show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
