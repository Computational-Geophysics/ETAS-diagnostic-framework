#!/usr/bin/env python3

r"""
NONSTATIONARY ETAS (UNITS IN DAYS) — log-spline for q_mu(t), q_K0(t)

- Same I/O and same outputs as your original script
- Vectorized log-likelihood O(n^2) using dt-matrix
- Splines built once per evaluation (not per event)
- Fast τ(t_i): analytical trigger (dt) + cumulative background (trapezoidal on event times)
- (Optional) parallel ABIC on Mac

Dependencies: numpy, scipy, matplotlib
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.interpolate import CubicSpline
from scipy.integrate import trapezoid

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# =============================================================================
# 0) SETTINGS
# =============================================================================

def safe_exp(x, clip=30.0):
    """Avoid overflow in exp(). clip=50 => exp(50) ~ 3e21."""
    return np.exp(np.clip(x, -clip, clip))


DAYS_PER_YEAR = 365.2425

Mth = 2.5
M0  = 2.5

mu_ref = 0.01   # events/day
K0_ref = 0.02   # events/day

N_knots_1 = 6
N_knots_2 = 6

change_point_t0 = None

W_GRID = [0.0, 0.05, 0.1]
INIT_CAP = (0.001, 1.0, 1.1)   # (c [days], alpha, p)

LABEL_FS = 15
TICK_FS  = 13
PANEL_FS = 16

OUTPUT_DIR = "output_nonstat_etas"

# ---- optional parallel ABIC ----
USE_PARALLEL_ABIC = False
N_WORKERS = None  # None => os.cpu_count()

# =============================================================================
# 1) I/O
# =============================================================================

def load_catalog_decimal_years(filename: str, Mth: float = 2.5):
    data = np.loadtxt(filename)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError("The file must have at least 2 columns: t_decimal_years, magnitude")

    data = data[data[:, 1] >= Mth]
    if len(data) < 2:
        raise ValueError("Too few events after magnitude filtering")

    data = data[np.argsort(data[:, 0])]
    return data[:, 0].astype(float), data[:, 1].astype(float)

def convert_decimal_years_to_days(t_decimal, days_per_year: float = DAYS_PER_YEAR):
    t_decimal = np.asarray(t_decimal, float)
    return (t_decimal - t_decimal[0]) * days_per_year

# =============================================================================
# 2) SPLINE HELPERS
# =============================================================================

def _build_splines(params_spline, knots_mu, knots_k0, n_knots_mu, n_knots_k0,
                  bc_mu="natural", bc_k0="natural"):
    
    c_smu = params_spline[:n_knots_mu]
    c_sk0 = params_spline[n_knots_mu:n_knots_mu+n_knots_k0]
    sp_smu = CubicSpline(knots_mu, c_smu, bc_type=bc_mu, extrapolate=True)
    sp_sk0 = CubicSpline(knots_k0, c_sk0, bc_type=bc_k0, extrapolate=True)
    return sp_smu, sp_sk0

def build_knots_and_layout(t_start_days, t_end_days, change_point_days=None):
    """
    For simplicity/robustness: in this vectorized version we handle:
    - no change point: a single set of knots for mu and K0 (identical)
    - with change point: two separate splines concatenated (two knot sets),
    but here (to keep things compact) we implement only the NO change-point case.
    """
    if change_point_days is not None:
        raise NotImplementedError(
            "For speed: only no-change-point is implemented here. "
        )
    knots = np.linspace(t_start_days, t_end_days, N_knots_1)
    # params_spline = [c_smu (N_knots_1), c_sk0 (N_knots_1)]
    n_params_spline = 2 * N_knots_1
    return knots, knots, N_knots_1, N_knots_1, n_params_spline

# =============================================================================
# 3) CORE: loglik
# =============================================================================

def _trigger_kernel_integral(dt_pos, c, p):
    """
    G(dt) = ∫_0^{dt} (u + c)^(-p) du
    dt_pos >= 0
    """
    if abs(p - 1.0) < 1e-12:
        return np.log((dt_pos + c) / c)
    else:
        return ((dt_pos + c)**(1.0 - p) - (c)**(1.0 - p)) / (1.0 - p)

def nonstat_loglik_vectorized(params_full,
                              t_days, m_events,
                              dt_mat, tril_mask,
                              knots_mu, knots_k0, n_knots_mu, n_knots_k0, n_params_spline,
                              t_start_day, t_end_day,
                              mu_ref, K0_ref, M0=M0,
                              n_bg_int_points=120):
    """
    LogL = Σ log λ(t_i) - ∫ λ(t) dt
    - Σ log λ(t_i): computed vectorized using dt_mat
    - ∫ λ(t) dt: background via numerical grid; trigger analytically (as in your code)
    """
    params_spline = params_full[:n_params_spline]
    c_par  = float(params_full[n_params_spline + 0])
    alpha_ = float(params_full[n_params_spline + 1])
    p_     = float(params_full[n_params_spline + 2])

    if (c_par <= 0) or (alpha_ <= 0) or (p_ <= 0):
        return -1e30

    # build spline objects ONCE
    sp_smu, sp_sk0 = _build_splines(
        params_spline, knots_mu, knots_k0, n_knots_mu, n_knots_k0,
        bc_mu="natural", bc_k0="natural"
    )

    # q_mu(t_i), q_k0(t_i)
    s_mu_i = sp_smu(t_days)
    s_k0_i = sp_sk0(t_days)
    q_mu_i = safe_exp(s_mu_i)
    q_k0_i = safe_exp(s_k0_i)

    mu_i = mu_ref * q_mu_i  # (n,)

    # parents weight: K0(t_j)*exp(alpha(Mj-M0))
    e_j = safe_exp(alpha_ * (m_events - M0))
    k0e = (K0_ref * q_k0_i) * e_j  # (n,)

    # triggered intensity at each i:
    # trig_i = Σ_{j<i} k0e[j] / (dt_ij + c)^p
    inv = np.zeros_like(dt_mat, dtype=float)

    dt_pos = dt_mat[tril_mask] + c_par  # are > 0
    

    inv_vals = dt_pos ** (-p_)          # = 1 / (dt_pos**p)
    inv[tril_mask] = inv_vals

    trig_i = inv @ k0e

    lam_i = mu_i + trig_i
    if np.any(lam_i <= 0) or (not np.all(np.isfinite(lam_i))):
        return -1e30

    ll_sum = np.sum(np.log(lam_i))

    # ---- integral part ----
    # bg: numeric integral on grid
    t_grid = np.linspace(t_start_day, t_end_day, n_bg_int_points)
    mu_grid = mu_ref * safe_exp(sp_smu(t_grid))
    bg_part = trapezoid(mu_grid, t_grid)

 
    dt_end = (t_end_day - t_days)
    if np.any(dt_end < 0):
        return -1e30

    if abs(p_ - 1.0) < 1e-12:
        Ij = k0e * np.log((dt_end + c_par) / c_par)
    else:
        Ij = k0e * (((dt_end + c_par)**(1.0 - p_) - (c_par**(1.0 - p_))) / (1.0 - p_))

    if (not np.all(np.isfinite(Ij))):
        return -1e30

    int_val = bg_part + np.sum(Ij)

    return ll_sum - int_val

# =============================================================================
# 4) Penalty roughness
# =============================================================================

def penalty_roughness_KO_fast(params_full, w_smooth,
                              knots_mu, knots_k0, n_knots_mu, n_knots_k0, n_params_spline,
                              t_start_day, t_end_day,
                              n_pen_grid=250):
    if w_smooth <= 0:
        return 0.0

    params_spline = params_full[:n_params_spline]
    sp_smu, sp_sk0 = _build_splines(
        params_spline, knots_mu, knots_k0, n_knots_mu, n_knots_k0,
        bc_mu="natural", bc_k0="natural"
    )
    t_grid = np.linspace(t_start_day, t_end_day, n_pen_grid)
    smu_dd = sp_smu(t_grid, 2)
    sk0_dd = sp_sk0(t_grid, 2)
    pen = trapezoid(smu_dd**2 + sk0_dd**2, t_grid)
    return w_smooth * pen

def neg_loglik_penalized_fast(params_full,
                              t_days, m_events,
                              dt_mat, tril_mask,
                              knots_mu, knots_k0, n_knots_mu, n_knots_k0, n_params_spline,
                              t_start_day, t_end_day,
                              mu_ref, K0_ref,
                              w_smooth):
    logL = nonstat_loglik_vectorized(
        params_full,
        t_days, m_events,
        dt_mat, tril_mask,
        knots_mu, knots_k0, n_knots_mu, n_knots_k0, n_params_spline,
        t_start_day, t_end_day,
        mu_ref, K0_ref, M0=M0
    )
    pen = penalty_roughness_KO_fast(
        params_full, w_smooth,
        knots_mu, knots_k0, n_knots_mu, n_knots_k0, n_params_spline,
        t_start_day, t_end_day
    )
    return -logL + pen

# =============================================================================
# 5) τ(t_i)  (for residual plot)
# =============================================================================

def tau_at_events_fast(params_opt,
                       t_days, m_events,
                       dt_mat, tril_mask,
                       knots_mu, knots_k0, n_knots_mu, n_knots_k0, n_params_spline,
                       mu_ref, K0_ref, M0=M0):
    """
    τ_i = ∫_{t0}^{t_i} λ(u) du = τ_bg_i + τ_trig_i
    background: trapezoidal integration on event times (fast)
    trigger: Σ_{j<i} k0e_j * G(dt_ij)
    """
    params_spline = params_opt[:n_params_spline]
    c_par  = float(params_opt[n_params_spline + 0])
    alpha_ = float(params_opt[n_params_spline + 1])
    p_     = float(params_opt[n_params_spline + 2])

    sp_smu, sp_sk0 = _build_splines(
        params_spline, knots_mu, knots_k0, n_knots_mu, n_knots_k0,
        bc_mu="natural", bc_k0="natural"
    )

    # bg at event times
    mu_i = mu_ref * safe_exp(sp_smu(t_days))
    dt_evt = np.diff(t_days)
    tau_bg = np.zeros_like(t_days)
    tau_bg[1:] = np.cumsum(0.5*(mu_i[1:]+mu_i[:-1]) * dt_evt)

    # trig part
    q_k0_i = safe_exp(sp_sk0(t_days))
    k0e = (K0_ref * q_k0_i) * safe_exp(alpha_ * (m_events - M0))

    # G(dt) matrix
    G = np.zeros_like(dt_mat)
    dt_pos = dt_mat[tril_mask]
    G[tril_mask] = _trigger_kernel_integral(dt_pos, c_par, p_)

    tau_trig = G @ k0e
    tau = tau_bg + tau_trig
    return tau

# =============================================================================
# 6) ABIC fit
# =============================================================================

def _fit_for_one_w(w,
                   t_days, m_events,
                   dt_mat, tril_mask,
                   knots_mu, knots_k0, n_knots_mu, n_knots_k0, n_params_spline,
                   t_start_day, t_end_day,
                   mu_ref, K0_ref,
                   init_cap):
    def fobj(par):
        return neg_loglik_penalized_fast(
            par,
            t_days, m_events,
            dt_mat, tril_mask,
            knots_mu, knots_k0, n_knots_mu, n_knots_k0, n_params_spline,
            t_start_day, t_end_day,
            mu_ref, K0_ref,
            w_smooth=w
        )

    init_spline = np.zeros(n_params_spline)
    c0, a0, p0 = init_cap
    init_full  = np.concatenate([init_spline, [c0, a0, p0]])

    bnds_spline = [(-np.inf, np.inf)] * n_params_spline
    bnds_others = [(1e-10, None), (1e-10, None), (0.5, 5.0)]
    all_bounds  = bnds_spline + bnds_others

    res = minimize(fobj, init_full, method="L-BFGS-B", bounds=all_bounds, options=dict(maxfun=200000, maxiter=5000, ftol=1e-9, disp=True))
    min_obj = float(res.fun)
    abic_val = 2.0 * min_obj + 2.0 * 1  # n_hyper=1
    return w, res, abic_val

def fit_via_abic_fast(w_candidates,
                      t_days, m_events,
                      dt_mat, tril_mask,
                      knots_mu, knots_k0, n_knots_mu, n_knots_k0, n_params_spline,
                      t_start_day, t_end_day,
                      mu_ref, K0_ref,
                      init_cap=INIT_CAP,
                      use_parallel=False,
                      n_workers=None):
    results = {}

    if use_parallel and len(w_candidates) > 1:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=n_workers) as pool:
            jobs = []
            for w in w_candidates:
                jobs.append(pool.apply_async(
                    _fit_for_one_w,
                    (w,
                     t_days, m_events,
                     dt_mat, tril_mask,
                     knots_mu, knots_k0, n_knots_mu, n_knots_k0, n_params_spline,
                     t_start_day, t_end_day,
                     mu_ref, K0_ref,
                     init_cap)
                ))
            for j in jobs:
                w, res, abic = j.get()
                results[w] = (res, abic)
    else:
        for w in w_candidates:
            w, res, abic = _fit_for_one_w(
                w,
                t_days, m_events,
                dt_mat, tril_mask,
                knots_mu, knots_k0, n_knots_mu, n_knots_k0, n_params_spline,
                t_start_day, t_end_day,
                mu_ref, K0_ref,
                init_cap
            )
            results[w] = (res, abic)

    best_w, best_res, best_abic = None, None, np.inf
    for w, (r, ab) in results.items():
        if ab < best_abic:
            best_abic = ab
            best_w = w
            best_res = r
    return best_w, best_res, best_abic, results

# =============================================================================
# 7) Plot
# =============================================================================

def make_summary_figure_fast(t_days, m_events,
                            params_opt,
                            dt_mat, tril_mask,
                            knots_mu, knots_k0, n_knots_mu, n_knots_k0, n_params_spline,
                            t_start_day, t_end_day,
                            n_grid=600):

    params_spline = params_opt[:n_params_spline]
    c_par  = float(params_opt[n_params_spline + 0])
    alpha_ = float(params_opt[n_params_spline + 1])
    p_     = float(params_opt[n_params_spline + 2])

    sp_smu, sp_sk0 = _build_splines(
        params_spline, knots_mu, knots_k0, n_knots_mu, n_knots_k0,
        bc_mu="natural", bc_k0="natural"
    )

    # intensity series for plot
    T_plot = np.linspace(t_start_day, t_end_day, n_grid)
    mu_plot = mu_ref * np.exp(sp_smu(T_plot))

    # for triggered in plot, do it on plot grid with O(n_grid*n) vectorization
    qk0_evt = safe_exp(sp_sk0(t_days))
    k0e_evt = (K0_ref * qk0_evt) * safe_exp(alpha_ * (m_events - M0))

    DT = (T_plot[:, None] - t_days[None, :])   # (ng, n)
    mask = DT > 0

    DTp = np.where(mask, DT + c_par, 1.0)      # positivo dove serve
    Kmasked = np.where(mask, k0e_evt[None, :], 0.0)

    trig_plot = np.sum(Kmasked / (DTp ** p_), axis=1)

    lam_plot = mu_plot + trig_plot

    # residual τ_i fast
    tau_vals = tau_at_events_fast(
        params_opt, t_days, m_events,
        dt_mat, tril_mask,
        knots_mu, knots_k0, n_knots_mu, n_knots_k0, n_params_spline,
        mu_ref, K0_ref, M0=M0
    )
    Ncum = np.arange(1, len(tau_vals) + 1)

    # scatter sizes by magnitude
    mmin, mmax = float(np.min(m_events)), float(np.max(m_events))
    u = (m_events - mmin) / (mmax - mmin + 1e-12)
    u = u**2
    sizes = 6.0 + (80.0 - 6.0) * u

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(9.8, 7.6),
        gridspec_kw={"height_ratios": [1.25, 1.0]}
    )

    ax1.plot(T_plot, lam_plot, lw=1.8)
    ax1.set_yscale("log")
    ax1.set_ylabel(r"Conditional intensity $\lambda(t)$ [events/day]", fontsize=LABEL_FS)
    ax1.tick_params(axis="both", which="major", labelsize=TICK_FS)
    ax1.text(0.01, 0.95, "(a)", transform=ax1.transAxes,
             fontsize=PANEL_FS, fontweight="bold", va="top", ha="left")

    ax1r = ax1.twinx()
    ax1r.scatter(t_days, m_events, s=sizes, c="0.4", alpha=0.18, linewidths=0, zorder=0)
    ax1r.set_ylabel("Magnitude", fontsize=LABEL_FS)
    ax1r.tick_params(axis="y", which="major", labelsize=TICK_FS)

    tau_max = float(tau_vals[-1])
    tau_grid = np.linspace(0, tau_max, 400)
    y_line = tau_grid
    band = 1.96 * np.sqrt(np.maximum(tau_grid, 1e-12))
    y_low = np.maximum(y_line - band, 0.0)
    y_high = y_line + band

    ax2.fill_between(tau_grid, y_low, y_high, alpha=0.15, linewidth=0)
    ax2.plot(tau_grid, y_line, "k--", lw=1.2, label=r"Expected $N(\tau)=\tau$")
    ax2.plot(tau_vals, Ncum, "ko", ms=3.2, alpha=0.9, label="Empirical")
    ax2.set_xlabel("Transformed time $\\tau$", fontsize=LABEL_FS)
    ax2.set_ylabel("Cumulative # events", fontsize=LABEL_FS)
    ax2.tick_params(axis="both", which="major", labelsize=TICK_FS)
    ax2.text(0.01, 0.95, "(b)", transform=ax2.transAxes,
             fontsize=PANEL_FS, fontweight="bold", va="top", ha="left")
    ax2.legend(frameon=True, fontsize=12, loc="lower right")

    plt.tight_layout()
    return fig, tau_vals

# =============================================================================
# 8) OUTPUT SAVING (same filenames/structure)
# =============================================================================

def save_outputs(output_dir,
                 t_decimal, t_days, m_events,
                 best_w, best_abic, best_res,
                 all_res,
                 params_opt, n_params_spline,
                 knots_mu, knots_k0, n_knots_mu, n_knots_k0,
                 dt_mat, tril_mask,
                 t_start_day, t_end_day):

    os.makedirs(output_dir, exist_ok=True)

    np.savetxt(
        os.path.join(output_dir, "catalog_used.dat"),
        np.column_stack([t_decimal, t_days, m_events]),
        header="t_decimal_year  t_day  magnitude"
    )

    rows = []
    for w, (res, abic) in all_res.items():
        rows.append([w, abic, res.fun, int(res.success)])
    rows = np.array(rows, float)
    np.savetxt(
        os.path.join(output_dir, "abic_grid_results.dat"),
        rows,
        header="w_smooth  ABIC  obj_min  success(1/0)"
    )

    np.savetxt(
        os.path.join(output_dir, "params_optimal_vector.dat"),
        params_opt,
        header="Full parameter vector: [spline_coeffs..., c, alpha, p]"
    )

    c_est = params_opt[n_params_spline + 0]
    a_est = params_opt[n_params_spline + 1]
    p_est = params_opt[n_params_spline + 2]

    with open(os.path.join(output_dir, "fit_summary.txt"), "w") as f:
        f.write("NONSTATIONARY ETAS (log-spline) — SUMMARY\n\n")
        f.write(f"Input Mc (Mth) = {Mth}\n")
        f.write(f"M0 = {M0}\n")
        f.write(f"mu_ref = {mu_ref}  [events/day]\n")
        f.write(f"K0_ref = {K0_ref}  [events/day]\n")
        f.write(f"N events used = {len(t_days)}\n")
        f.write(f"Catalog duration (days) = {t_end_day - t_start_day:.6g}\n")
        f.write("\n--- ABIC selection ---\n")
        f.write(f"best w_smooth = {best_w}\n")
        f.write(f"best ABIC = {best_abic}\n")
        f.write(f"Optimizer success = {best_res.success}\n")
        f.write(f"Optimizer message = {best_res.message}\n")
        f.write(f"obj_min = {best_res.fun}\n")
        f.write("\n--- ETAS parameters ---\n")
        f.write(f"c = {c_est:.12g}\n")
        f.write(f"alpha = {a_est:.12g}\n")
        f.write(f"p = {p_est:.12g}\n")

    # fitted_time_series.dat
    params_spline = params_opt[:n_params_spline]
    sp_smu, sp_sk0 = _build_splines(
        params_spline, knots_mu, knots_k0, n_knots_mu, n_knots_k0,
        bc_mu="natural", bc_k0="natural"
    )

    T_plot = np.linspace(t_start_day, t_end_day, 600)
    qmu = safe_exp(sp_smu(T_plot))
    qk0 = safe_exp(sp_sk0(T_plot))
    mu_series = mu_ref * qmu
    K0_series = K0_ref * qk0

    # lambda(t) for plot series: reuse figure routine idea (vectorized on plot grid)
    c_par  = float(params_opt[n_params_spline + 0])
    alpha_ = float(params_opt[n_params_spline + 1])
    p_     = float(params_opt[n_params_spline + 2])

    qk0_evt = safe_exp(sp_sk0(t_days))
    k0e_evt = (K0_ref * qk0_evt) * safe_exp(alpha_ * (m_events - M0))
    DT = (T_plot[:, None] - t_days[None, :])
    mask = DT > 0
    DTp = np.where(mask, DT + c_par, 1.0)
    Kmasked = np.where(mask, k0e_evt[None, :], 0.0)
    trig_series = np.sum((Kmasked / (DTp ** p_)) * mask, axis=1)
    lam_series = mu_series + trig_series
    lam_series = np.where(np.isfinite(lam_series), lam_series, np.nan)

    np.savetxt(
        os.path.join(output_dir, "fitted_time_series.dat"),
        np.column_stack([T_plot, lam_series, mu_series, qmu, qk0, K0_series]),
        header="t_day  lambda(t)  mu(t)=mu_ref*q_mu  q_mu(t)  q_K0(t)  K0(t)=K0_ref*q_K0"
    )

    # residuals_time_transform.dat
    tau_vals = tau_at_events_fast(
        params_opt, t_days, m_events,
        dt_mat, tril_mask,
        knots_mu, knots_k0, n_knots_mu, n_knots_k0, n_params_spline,
        mu_ref, K0_ref, M0=M0
    )
    Ncum = np.arange(1, len(tau_vals) + 1)

    np.savetxt(
        os.path.join(output_dir, "residuals_time_transform.dat"),
        np.column_stack([np.arange(1, len(t_days)+1), t_days, tau_vals, Ncum]),
        header="index  t_day  tau(t)  Ncum"
    )

# =============================================================================
# 9) MAIN
# =============================================================================

if __name__ == "__main__":

    filename = "../catalogs/modified_decimal_days_mL.txt"
    t_decimal, m_events = load_catalog_decimal_years(filename, Mth=Mth)
    t_days = convert_decimal_years_to_days(t_decimal, days_per_year=DAYS_PER_YEAR)

    t_start_day = float(t_days[0])
    t_end_day   = float(t_days[-1])

    
    change_point_days = None
    if change_point_t0 is not None:
        cp_d = (change_point_t0 - t_decimal[0]) * DAYS_PER_YEAR
        if (cp_d > t_start_day) and (cp_d < t_end_day):
            change_point_days = float(cp_d)

    if change_point_days is not None:
        raise SystemExit("This version is no-change-point.")

    # knots/layout
    knots_mu, knots_k0, n_knots_mu, n_knots_k0, n_params_spline = build_knots_and_layout(
        t_start_day, t_end_day, change_point_days=None
    )

    # precompute dt matrix ONCE
    t_col = t_days[:, None]
    dt_mat = t_col - t_col.T  # dt[i,j] = t_i - t_j
    tril_mask = np.tril(np.ones_like(dt_mat, dtype=bool), k=-1)  # i>j

    # fit via ABIC (fast + optional parallel)
    best_w, best_res, best_abic, all_res = fit_via_abic_fast(
        W_GRID,
        t_days, m_events,
        dt_mat, tril_mask,
        knots_mu, knots_k0, n_knots_mu, n_knots_k0, n_params_spline,
        t_start_day, t_end_day,
        mu_ref, K0_ref,
        init_cap=INIT_CAP,
        use_parallel=USE_PARALLEL_ABIC,
        n_workers=N_WORKERS
    )

    params_opt = best_res.x
    c_est  = params_opt[n_params_spline + 0]
    a_est  = params_opt[n_params_spline + 1]
    p_est  = params_opt[n_params_spline + 2]

    print("\n=== NONSTATIONARY ETAS (log-spline, giorni) — FAST ===")
    print("Mth = M0 =", Mth)
    print("best w_smooth =", best_w)
    print("ABIC =", best_abic)
    print("success:", best_res.success, "| message:", best_res.message)
    print("obj_min =", best_res.fun)
    print(f"Parametri (c, alpha, p) = ({c_est:.6g}, {a_est:.6g}, {p_est:.6g})")

    # save outputs
    save_outputs(
        OUTPUT_DIR,
        t_decimal, t_days, m_events,
        best_w, best_abic, best_res,
        all_res,
        params_opt, n_params_spline,
        knots_mu, knots_k0, n_knots_mu, n_knots_k0,
        dt_mat, tril_mask,
        t_start_day, t_end_day
    )
    print(f"\n[OK] file saved in: {OUTPUT_DIR}")

    # figure
    fig, _tau = make_summary_figure_fast(
        t_days, m_events,
        params_opt,
        dt_mat, tril_mask,
        knots_mu, knots_k0, n_knots_mu, n_knots_k0, n_params_spline,
        t_start_day, t_end_day
    )
    plt.show()
