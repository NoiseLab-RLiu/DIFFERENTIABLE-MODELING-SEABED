# -*- coding: utf-8 -*-
"""PMM_inverse_mfreqs_rev.py 的论文公式修订副本。

先运行 PMM_forward_paper.py，再在 Spyder 运行本文件。
python PMM_inverse_mfreqs_paper.py --check-only --no-show  仅做公式/梯度检查
python PMM_inverse_mfreqs_paper.py --no-show              检查并反演

主要函数名沿用原脚本。PAPER CHANGE 标明复声压 SSE、解析 IFT、
无共轭归一化、精确 Hankel 及单阶段论文 Rprop 的修改。
"""

from pathlib import Path
import argparse
import json
import time
import math

# 先导入正演模块，让 Windows MKL 设置在 numpy/torch 导入前生效。
import PMM_forward_paper as physics
import torch
import numpy as np
from scipy.special import hankel1

device = physics.device
C_W_GLOBAL = 1500.0  # 仅保留旧命名；实际逐层声速由 get_layered_profile 给出。
RHO_W_GLOBAL = physics.RHO_W_GLOBAL
D_GLOBAL = physics.D_GLOBAL
ZS_GLOBAL = physics.ZS_GLOBAL
ZR_GLOBAL = physics.ZR_GLOBAL
FREQ_LIST = physics.FREQ_LIST
INITIAL_PARAMS = [1950.0, 1.2, 0.1]
INITIAL_STEPS = [1.0, 0.01, 0.01]  # 论文允许逐参数初始步长。
EPSILON = 1e-5
MAX_ITERS = 600  # 安全上限；达到上限与按论文判据收敛分别报告。


def get_layered_profile(freq, n_layers=physics.N_LAYERS):
    return physics.get_layered_profile(freq, n_layers)


def compute_F(k, cb, rb, ab, freq, n_layers=physics.N_LAYERS):
    """与原函数相同的 F(k,theta)，共享正演以防两份物理实现漂移。"""
    return physics.compute_prop_matrix(k, freq, cb, rb, ab, False, n_layers)


def extract_mode_shapes_diff(k, cb, rb, ab, freq, n_layers=physics.N_LAYERS):
    """PAPER CHANGE: 无共轭归一化；精确求源和接收深度的模态值。"""
    _, _, _, history, norm = physics.compute_prop_matrix(
        k, freq, cb, rb, ab, True, n_layers)
    Zs = physics.mode_at_depth(k, freq, ZS_GLOBAL, history, norm, n_layers)
    Zr = physics.mode_at_depth(k, freq, ZR_GLOBAL, history, norm, n_layers)
    return Zs, Zr


def unified_complex_solver(cb, rb, ab, freq, n_layers=physics.N_LAYERS,
                            initial_seeds=None):
    theta = [cb.detach().item(), rb.detach().item(), ab.detach().item()]
    roots = physics.solve_modes(theta, freq, n_layers, initial_seeds)
    return torch.as_tensor(roots, dtype=torch.complex128, device=device)


class IFTSolver(torch.autograd.Function):
    @staticmethod
    def forward(ctx, cb, rb, ab, freq, n_layers, initial_seeds):
        with torch.no_grad():
            k_star = unified_complex_solver(cb, rb, ab, freq, n_layers, initial_seeds)
        ctx.save_for_backward(k_star, cb, rb, ab)
        ctx.freq, ctx.n_layers = freq, n_layers
        return k_star

    @staticmethod
    def backward(ctx, grad_output):
        k, cb, rb, ab = ctx.saved_tensors
        # PAPER CHANGE: 论文的 dP/dk 递推 + 解析边界偏导，不通过 Newton 回传。
        _, Fk, Ftheta, _ = physics.compute_F_and_partials(
            k.detach().numpy(), cb.item(), rb.item(), ab.item(), ctx.freq, ctx.n_layers)
        if not np.all(np.isfinite(Fk)) or np.any(np.abs(Fk) < 1e-14):
            raise RuntimeError("IFT 分母病态；保留错误诊断而不截断梯度。")
        dk_dtheta = torch.as_tensor(-Ftheta/Fk[:, None], dtype=torch.complex128)
        gradient = (grad_output.conj()[:, None]*dk_dtheta).sum(dim=0).real
        return gradient[0], gradient[1], gradient[2], None, None, None


def forward_model(cb, rb, ab, r_range, freq, n_layers=physics.N_LAYERS,
                  initial_seeds=None):
    k_complex = IFTSolver.apply(cb, rb, ab, freq, n_layers, initial_seeds)
    # 保留普通 AD 路径：归一化对 cb/rb/ab 的直接依赖在求根算子之外。
    Zs, Zr = extract_mode_shapes_diff(k_complex, cb, rb, ab, freq, n_layers)
    phase = k_complex[:, None]*r_range[None, :]
    H0 = physics.Hankel0.apply(phase)  # PAPER CHANGE: 精确 Hankel + 解析 backward。
    p_sum = 1j/(4*RHO_W_GLOBAL)*(Zs[:, None]*Zr[:, None]*H0).sum(dim=0)
    return p_sum, k_complex  # PAPER CHANGE: 返回复声压，不返回 TL。


def load_meas_data(data_dir):
    meas_data, n_layers, truth_for_reporting = {}, None, None
    for freq in FREQ_LIST:
        file = data_dir/f"paper_layered_{int(freq)}.npz"
        if not file.exists():
            raise FileNotFoundError(f"缺少 {file}；请先运行 PMM_forward_paper.py。")
        with np.load(file, allow_pickle=False) as data:
            if str(data["formula_version"]) != physics.FORMULA_VERSION:
                raise ValueError("观测公式版本不匹配。")
            nl = int(data["n_layers"])
            if n_layers is not None and nl != n_layers:
                raise ValueError("不同频率数据的层数不一致。")
            n_layers = nl
            for key, expected in [("D", D_GLOBAL), ("rho_w", RHO_W_GLOBAL),
                                   ("zs", ZS_GLOBAL), ("zr", ZR_GLOBAL), ("frequency", freq)]:
                if not np.isclose(float(data[key]), expected):
                    raise ValueError(f"{key} 与当前已知环境不一致。")
            if not np.array_equal(data["ssp_z"], physics.SSP_Z) or \
                    not np.array_equal(data["ssp_c"], physics.SSP_C):
                raise ValueError("观测与拟合所用已知 SSP 不一致。")
            p, r = data["P_meas"].copy(), data["r_meas"].copy()
            if not np.iscomplexobj(p) or p.shape != r.shape or not np.isfinite(p).all():
                raise ValueError("P_meas 必须是与 r_meas 等长的有限复声压数组。")
            meas_data[freq] = {"r_meas": torch.as_tensor(r),
                               "P_meas": torch.as_tensor(p)}
            # 只用于最终误差和参考虚线；不传给损失、求根或优化器。
            reported = data["true_params"].copy()
            if truth_for_reporting is not None and not np.array_equal(reported, truth_for_reporting):
                raise ValueError("各频率标签真值元数据不一致。")
            truth_for_reporting = reported
    return meas_data, n_layers, truth_for_reporting


def loss_function(params, meas_data, n_layers, initial_seeds=None, fixed_roots=None):
    """论文 loss_func：sum_f sum_receivers |P_pred-P_meas|^2。

    fixed_roots 只供诊断直接参数梯度；训练始终通过 IFTSolver。
    """
    cb, rb, ab = params.unbind()
    total_loss = torch.zeros((), dtype=torch.float64)
    predicted, roots = {}, {}
    for freq, data in meas_data.items():
        if fixed_roots is None:
            seeds = None if initial_seeds is None else initial_seeds.get(freq)
            p, k = forward_model(cb, rb, ab, data["r_meas"], freq, n_layers, seeds)
        else:
            k = torch.as_tensor(fixed_roots[freq], dtype=torch.complex128)
            Zs, Zr = extract_mode_shapes_diff(k, cb, rb, ab, freq, n_layers)
            H0 = physics.Hankel0.apply(k[:, None]*data["r_meas"][None, :])
            p = 1j/(4*RHO_W_GLOBAL)*(Zs[:, None]*Zr[:, None]*H0).sum(dim=0)
        total_loss = total_loss+(p-data["P_meas"]).abs().square().sum()
        predicted[freq] = p.detach().numpy()
        roots[freq] = k.detach().numpy().copy()
    return total_loss, predicted, roots


def _relative_error(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a-b)/np.maximum(np.maximum(np.abs(a), np.abs(b)), 1e-12)))


def verify_gradients(meas_data, n_layers, initial_params):
    """有物理意义的检查：解析局部偏导、全目标有限差分、归一化和单层极限。"""
    report = {}
    params = torch.tensor(initial_params, requires_grad=True)
    loss, _, roots = loss_function(params, meas_data, n_layers)
    loss.backward()
    analytic = params.grad.detach().numpy().copy()
    steps = np.array([1e-3, 1e-6, 1e-6])
    finite = np.empty(3)
    for j, step in enumerate(steps):
        plus, minus = np.array(initial_params, float), np.array(initial_params, float)
        plus[j] += step
        minus[j] -= step
        with torch.no_grad():
            lp = loss_function(torch.as_tensor(plus), meas_data, n_layers, roots)[0].item()
            lm = loss_function(torch.as_tensor(minus), meas_data, n_layers, roots)[0].item()
        finite[j] = (lp-lm)/(2*step)
    relative = np.abs(analytic-finite)/np.maximum(np.maximum(abs(analytic), abs(finite)), 1e-12)
    report["total_gradient_IFT_AD"] = analytic.tolist()
    report["total_gradient_finite_difference"] = finite.tolist()
    report["total_gradient_relative_error"] = relative.tolist()
    # 显式核对固定根时仍存在的归一化直接参数梯度。
    direct_params = torch.tensor(initial_params, requires_grad=True)
    direct_loss = loss_function(direct_params, meas_data, n_layers, fixed_roots=roots)[0]
    direct_loss.backward()
    report["direct_gradient_at_fixed_roots"] = direct_params.grad.tolist()

    freq = 150.0
    k = torch.tensor(roots[freq], requires_grad=True)
    vectors = [torch.full(k.shape, x, requires_grad=True) for x in initial_params]
    F = compute_F(k, *vectors, freq, n_layers)
    Fk_ad = torch.autograd.grad(F.real.sum(), k, retain_graph=True)[0].conj().numpy()
    Ftheta_ad = []
    for x in vectors:
        re = torch.autograd.grad(F.real.sum(), x, retain_graph=True)[0]
        im = torch.autograd.grad(F.imag.sum(), x, retain_graph=True)[0]
        Ftheta_ad.append((re+1j*im).numpy())
    _, Fk, Ftheta, residual = physics.compute_F_and_partials(
        roots[freq], *initial_params, freq, n_layers)
    report["Fk_manual_vs_AD_relative_error"] = _relative_error(Fk, Fk_ad)
    report["Ftheta_manual_vs_AD_relative_error"] = _relative_error(Ftheta, np.column_stack(Ftheta_ad))
    report["max_boundary_residual_150"] = float(max(residual))

    with torch.no_grad():
        scalars = torch.as_tensor(initial_params).unbind()
        result8 = physics.compute_prop_matrix(k.detach(), freq, *scalars, True, n_layers, 8)
        result12 = physics.compute_prop_matrix(k.detach(), freq, *scalars, True, n_layers, 12)
        norm_error = ((result12[-1]/result8[-1]).square()-1).abs().max().item()
        report["normalization_checked_with_12_point_quadrature"] = norm_error
        # 一层中点 SSP = 1500 m/s，此时与论文 Pekeris 闭式公式比较。
        k1 = torch.as_tensor(physics.solve_modes(initial_params, freq, 1))
        _, _, _, hist, norm = physics.compute_prop_matrix(k1, freq, *scalars, True, 1, 96)
        src = physics.mode_at_depth(k1, freq, ZS_GLOBAL, hist, norm, 1)
        rec = physics.mode_at_depth(k1, freq, ZR_GLOBAL, hist, norm, 1)
        omega = 2*math.pi*freq
        qw = torch.sqrt((omega/1500.)**2-k1**2)
        kb = omega/scalars[0]*(1+1j*scalars[2]/54.58)
        qb = torch.sqrt(kb**2-k1**2)
        Am = (D_GLOBAL/(2*RHO_W_GLOBAL)-torch.sin(2*qw*D_GLOBAL)/(4*RHO_W_GLOBAL*qw)
              +1j*torch.sin(qw*D_GLOBAL).square()/(2*scalars[1]*qb)).rsqrt()
        closed_product = Am.square()*torch.sin(qw*ZS_GLOBAL)*torch.sin(qw*ZR_GLOBAL)
        report["single_layer_Pekeris_modal_product_relative_error"] = _relative_error(
            (src*rec).numpy(), closed_product.numpy())
    checks = [float(max(relative)), report["Fk_manual_vs_AD_relative_error"],
              report["Ftheta_manual_vs_AD_relative_error"], norm_error,
              report["single_layer_Pekeris_modal_product_relative_error"]]
    report["passed"] = bool(np.all(np.isfinite(checks)) and max(checks) < 2e-4)
    print("Gradient / formula checks:\n"+json.dumps(report, indent=2))
    if not report["passed"]:
        raise RuntimeError("公式或梯度检查未通过；停止反演。")
    return report


def invert(meas_data, n_layers, initial_params, initial_steps, epsilon, max_iters):
    """PAPER CHANGE: 直接实现 Algorithm 1 的 Rprop，便于逐行对应。

    一个复声压 SSE 目标、一个优化阶段；无归一化损失、无距离截取。
    物性离开正值域时明确报错，不通过静默裁剪改变论文更新规则。
    """
    theta = np.array(initial_params, dtype=float)
    eta = np.array(initial_steps, dtype=float)
    if epsilon <= 0 or np.min(eta) <= epsilon or max_iters < 1:
        raise ValueError("要求 epsilon>0, min(initial_steps)>epsilon, max_iters>=1。")
    previous_grad = np.zeros(3)
    history, roots = [], None
    initial_prediction = None
    started = time.perf_counter()
    status = "max_iterations"
    for epoch in range(max_iters+1):
        physics.validate_params(theta)
        params = torch.tensor(theta, requires_grad=True)
        loss, prediction, roots = loss_function(params, meas_data, n_layers, roots)
        loss.backward()
        grad = params.grad.detach().numpy().copy()
        if not np.isfinite(loss.item()) or not np.isfinite(grad).all():
            raise RuntimeError("出现非有限损失或梯度。")
        if initial_prediction is None:
            initial_prediction = {f: p.copy() for f, p in prediction.items()}
        history.append([epoch, loss.item(), *theta, *eta, *grad])
        if epoch % 10 == 0:
            print(f"Iter {epoch:04d} | SSE {loss.item():.7e} | "
                  f"cb {theta[0]:.6f} rho {theta[1]:.7f} alpha {theta[2]:.7f} | "
                  f"max_eta {max(eta):.2e}", flush=True)
        if max(eta) <= epsilon:
            status = "step_tolerance"
            break
        if epoch == max_iters:
            break
        products = grad*previous_grad
        eta[products > 0] *= 1.2
        eta[products < 0] *= 0.5
        update_grad = grad.copy()
        update_grad[products < 0] = 0.0
        theta -= eta*np.sign(update_grad)
        previous_grad = update_grad
    return theta, np.array(history), initial_prediction, prediction, roots, {
        "status": status, "iterations": len(history)-1,
        "seconds_excluding_gradient_checks": time.perf_counter()-started,
        "complex_pressure_SSE": float(history[-1][1]), "max_eta": float(max(eta)),
        "mode_counts": {str(int(f)): len(k) for f, k in roots.items()}}


def plot_results(output_dir, data_dir, meas_data, history, truth, initial, final,
                 final_params, roots, n_layers):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes[0, 0].semilogy(history[:, 0], np.maximum(history[:, 1], 1e-30))
    axes[0, 0].set(title="Complex-pressure SSE (training objective)", xlabel="Iteration")
    for ax, column, target, label in zip(axes.flat[1:], [2, 3, 4], truth,
                                        ["c_b (m/s)", "rho_b (g/cm^3)", "alpha_b (dB/lambda)"]):
        ax.plot(history[:, 0], history[:, column], label="Estimated")
        ax.axhline(target, color="r", linestyle="--", label="Truth (report only)")
        ax.set(xlabel="Iteration", ylabel=label)
        ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_dir/"inversion_history.png", dpi=160)

    freq = 150.0
    r = meas_data[freq]["r_meas"].numpy()/1000
    observed = meas_data[freq]["P_meas"].numpy()
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    for p, label, style in [(observed, "Observed", "k-"),
                             (initial[freq], "Initial", "b:"), (final[freq], "Recovered", "r--")]:
        axes[0].plot(r, p.real, style, label=label)
        axes[1].plot(r, p.imag, style)
        axes[2].plot(r, -20*np.log10(np.maximum(abs(p), 1e-30)), style)
    axes[0].set(ylabel="Re(P)", title="150 Hz: observation and recovery")
    axes[0].legend(loc="upper right")
    axes[1].set(ylabel="Im(P)")
    axes[2].set(ylabel="-20 log10 |P| (display only)", xlabel="Range (km)")
    fig.tight_layout()
    fig.savefig(output_dir/"inversion_observation_fit.png", dpi=160)

    field_file = data_dir/"paper_layered_field_150.npz"
    if field_file.exists():
        with np.load(field_file) as saved:
            r_field, z_field, field_true = saved["r_vec"], saved["z_vec"], saved["P_2D"]
        with torch.no_grad():
            field, _ = physics.pressure_from_roots(torch.as_tensor(roots[freq]),
                torch.as_tensor(final_params), freq, torch.as_tensor(r_field), n_layers,
                full_field=True)
        field = field.numpy()
        np.savez(output_dir/"recovered_field_150.npz", P_2D=field, r_vec=r_field, z_vec=z_field)
        true_level = 20*np.log10(np.maximum(abs(field_true), 1e-30))
        vmax = float(np.percentile(true_level[1:], 98))
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
        for ax, p, title in zip(axes, [field_true, field], ["Synthetic reference", "Recovered"]):
            im = ax.pcolormesh(r_field/1000, z_field,
                              20*np.log10(np.maximum(abs(p), 1e-30)),
                              shading="auto", cmap="viridis", vmin=vmax-60, vmax=vmax)
            ax.set(xlabel="Range (km)", title=title, ylim=(D_GLOBAL, 0))
            fig.colorbar(im, ax=ax, label="20 log10 |P|")
        axes[0].set_ylabel("Depth (m)")
        fig.tight_layout()
        fig.savefig(output_dir/"inversion_field_comparison_150.png", dpi=160)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=physics.OUTPUT_DIR)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--skip-check", action="store_true")
    parser.add_argument("--max-iters", type=int, default=MAX_ITERS)
    parser.add_argument("--epsilon", type=float, default=EPSILON)
    parser.add_argument("--initial", type=float, nargs=3, default=INITIAL_PARAMS)
    args = parser.parse_args()
    if args.no_show:
        import matplotlib
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output_dir = args.output_dir or args.data_dir/"inversion"
    output_dir.mkdir(parents=True, exist_ok=True)
    meas_data, n_layers, truth = load_meas_data(args.data_dir)
    if not args.skip_check:
        checks = verify_gradients(meas_data, n_layers, args.initial)
        (output_dir/"gradient_checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    if args.check_only:
        return
    result = invert(meas_data, n_layers, args.initial, INITIAL_STEPS, args.epsilon, args.max_iters)
    theta, history, initial, final, roots, report = result
    report.update({"formula_version": physics.FORMULA_VERSION, "n_layers": n_layers,
                   "initial_params": args.initial, "estimated_params": theta.tolist(),
                   "truth_for_reporting_only": truth.tolist(),
                   "parameter_error": (theta-truth).tolist()})
    signal_energy = sum(float(d["P_meas"].abs().square().sum()) for d in meas_data.values())
    report["relative_complex_pressure_error"] = math.sqrt(report["complex_pressure_SSE"]/signal_energy)
    (output_dir/"inversion_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savetxt(output_dir/"inversion_history.csv", history, delimiter=",", comments="",
               header="iteration,SSE,cb,rho_b,alpha_b,eta_cb,eta_rho,eta_alpha,grad_cb,grad_rho,grad_alpha")
    np.savez(output_dir/"inversion_history.npz", history=history, final_params=theta,
             **{f"P_pred_{int(f)}": p for f, p in final.items()})
    plot_results(output_dir, args.data_dir, meas_data, history, truth, initial, final, theta, roots, n_layers)
    print("Final report:\n"+json.dumps(report, indent=2))
    print(f"Saved to {output_dir}")
    if not args.no_show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
