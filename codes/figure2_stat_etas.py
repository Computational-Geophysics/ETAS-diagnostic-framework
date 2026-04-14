#!/usr/bin/env python3

"""
STATIONARY ETAS (mu, K0, c, alpha, p) with input times in DECIMAL YEARS.

Input file:
  col 0: time in decimal years
  col 1: magnitude

Workflow:
  1) load catalog in decimal years (filter M>=Mth, sort by time)
  2) convert times to DAYS (t=0 at the first event)
  3) estimate stationary ETAS parameters (mu, K0 in events/day; c in days)
  4) produce ONE figure with two panels:
     (a) conditional intensity λ(t) (log-y) + catalog (transparent gray dots)
     (b) time-transform residual (N(τ) vs τ) + expected line y=x + band ~±1.96 sqrt(τ)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import matplotlib as mpl

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial"],
    "font.size": 13,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "axes.linewidth": 1.0,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# =============================================================================
# 1) I/O: catalogo in anni decimali -> giorni
# =============================================================================

def load_catalog_decimal_years(filename: str, Mth: float = 2.5):
    """
    Read data (t_decimal, M) in decimal years from file,
    filter by magnitude >= Mth,
    sort by time, and return (t_decimal, m_events).
    """
    data = np.loadtxt(filename)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError("The file must have at least 2 columns: t_decimal_years, magnitude")

    data = data[data[:, 1] >= Mth]
    if len(data) < 2:
        raise ValueError("Too few events after magnitude filtering.")

    data = data[np.argsort(data[:, 0])]
    return data[:, 0].astype(float), data[:, 1].astype(float)


def convert_decimal_years_to_days(t_decimal, days_per_year: float = 365.2425):
    """
    Convert an array of times in decimal years to DAYS, setting t=0 at the first event.
    """
    t_decimal = np.asarray(t_decimal, float)
    return (t_decimal - t_decimal[0]) * days_per_year


# =============================================================================
# 2) Stationary ETAS: log-likelihood (time in days)
# =============================================================================

def etas_loglik_staz(params, t_days, m_events, t_start, t_end, M0: float = 2.5):
    """
    params = (mu, K0, c, alpha, p)
      mu, K0 : events/day
      c      : days
      alpha, p : dimensionless
    """
    mu, K0, c_par, alpha, p_ = params
    N = len(t_days)

    # parametri fisicamente validi
    if (mu <= 0) or (K0 <= 0) or (c_par <= 0) or (alpha <= 0) or (p_ <= 0):
        return -1e30

    # (A) Σ log λ(t_i)
    ll_sum = 0.0
    for i in range(N):
        ti = t_days[i]
        lam_i = mu

        idx_par = np.where(t_days < ti)[0]
        if idx_par.size > 0:
            t_par = t_days[idx_par]
            m_par = m_events[idx_par]
            e_fac = np.exp(alpha * (m_par - M0))
            denom = (ti - t_par + c_par) ** p_
            lam_i += np.sum(K0 * e_fac / denom)

        if (lam_i <= 0) or (not np.isfinite(lam_i)):
            return -1e30

        ll_sum += np.log(lam_i)

    # (B) ∫ λ(t) dt = mu*(T) + Σ ∫ triggered
    bg_part = mu * (t_end - t_start)

    trig_sum = 0.0
    for j in range(N):
        tj = t_days[j]
        if tj >= t_end:
            continue

        mj = m_events[j]
        e_j = np.exp(alpha * (mj - M0))

        # ∫_{u=tj}^{t_end} K0 e_j (u - tj + c)^(-p) du
        if abs(p_ - 1.0) < 1e-10:
            I_j = (K0 * e_j) * np.log((t_end - tj + c_par) / c_par)
        else:
            A = (t_end - tj + c_par) ** (1.0 - p_)
            B = (c_par) ** (1.0 - p_)
            I_j = (K0 * e_j / (1.0 - p_)) * (A - B)

        if not np.isfinite(I_j):
            return -1e30

        trig_sum += I_j

    return ll_sum - (bg_part + trig_sum)


def neg_loglik_staz(params, t_days, m_events, t_start, t_end, M0: float = 2.5):
    return -etas_loglik_staz(params, t_days, m_events, t_start, t_end, M0)


# =============================================================================
# 3) Fit + AIC
# =============================================================================

def fit_etas_stationary_and_aic(t_days, m_events, t_start, t_end, M0: float = 2.5):
    """
    Minimize -logL and also return the correct AIC:
        AIC = 2k - 2logL
    with k = 5 parameters for the stationary ETAS model.
    """
    init_guess = [0.01, 0.02, 0.001, 1.0, 1.1]  # mu, K0, c, alpha, p
    bounds = [
        (1e-12, None),  # mu
        (1e-12, None),  # K0
        (1e-12, None),  # c
        (1e-12, None),  # alpha
        (0.01, 5.0)     # p
    ]

    def fobj(par):
        return neg_loglik_staz(par, t_days, m_events, t_start, t_end, M0)

    res = minimize(fobj, init_guess, method="L-BFGS-B", bounds=bounds)

    k = 5
    aic_val = 2 * k + 2 * res.fun   # perché res.fun = -logL

    return res, aic_val


# =============================================================================
# 4) λ(t) and cumulative for residual (time in days)
# =============================================================================

def staz_lambda_of_t(t, params, t_days, m_events, M0: float = 2.5):
    mu, K0, c_par, alpha, p_ = params
    lam = mu

    idx = np.where(t_days < t)[0]
    if idx.size > 0:
        t_par = t_days[idx]
        m_par = m_events[idx]
        e_fac = np.exp(alpha * (m_par - M0))
        denom = (t - t_par + c_par) ** p_
        lam += np.sum(K0 * e_fac / denom)

    return lam


def staz_cumulative_Lambda(t, par, t_days, m_events, t_start, M0: float = 2.5, n_int_points: int = 400):
    if t <= t_start:
        return 0.0

    grid = np.linspace(t_start, t, n_int_points)
    lam = np.array([staz_lambda_of_t(u, par, t_days, m_events, M0) for u in grid])
    return np.trapz(lam, grid)


# =============================================================================
# 5) Plot: (a) intensity + catalog, (b) residuals
# =============================================================================

def make_summary_figure(t_days, m_events, params, t_start, t_end, M0: float = 2.5,
                        n_grid: int = 600, n_int_points: int = 400):
    """
    Create a figure with 2 panels:
      (a) λ(t) (log y) + catalog (transparent gray dots, size~M) on right axis
      (b) residuals: N(τ) vs τ + y=x + band ~±1.96 sqrt(τ)
    """
    panel_fs = 13

    # --- λ(t) on a grid ---
    T_plot = np.linspace(t_start, t_end, n_grid)
    lam_vals = np.array([staz_lambda_of_t(tg, params, t_days, m_events, M0=M0) for tg in T_plot])

    # --- point sizes ~ magnitude (smooth mapping) ---
    mmin, mmax = float(np.min(m_events)), float(np.max(m_events))
    u = (m_events - mmin) / (mmax - mmin + 1e-12)
    u = u**2
    s_min, s_max = 6.0, 80.0
    sizes = s_min + (s_max - s_min) * u

    # --- τ_i for residuals ---
    tau_vals = np.array([
        staz_cumulative_Lambda(ti, params, t_days, m_events, t_start, M0=M0, n_int_points=n_int_points)
        for ti in t_days
    ])
    Ncum = np.arange(1, len(tau_vals) + 1)

   
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8, 5.6),
        gridspec_kw={"height_ratios": [1.25, 1.0]}
    )

    for ax in [ax1, ax2]:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.0)

    # ===================== (a) intensity + catalog =====================
    ax1.plot(T_plot, lam_vals, linewidth=1.5, color="black")
    ax1.set_yscale("log")
    ax1.set_ylabel(r"$\lambda(t)$ [events/day]")
    ax1.tick_params(axis="both", which="both", top=True, right=True)


    ax1r = ax1.twinx()
    ax1r.scatter(
        t_days, m_events,
        s=sizes * 0.6,
        c="0.2",
        alpha=0.25,
        edgecolors="none",
        zorder=0
    )
    ax1r.set_ylabel("Magnitude")
    ax1r.tick_params(axis="y", which="both", right=True)

    # panel label
    ax1.text(0.01, 0.95, "(a)", transform=ax1.transAxes,
             fontsize=panel_fs, fontweight="bold", va="top", ha="left")

    # ===================== (b) residuals =====================
    tau_max = float(tau_vals[-1])
    tau_grid = np.linspace(0, tau_max, 400)
    y_line = tau_grid

    
    band = 1.96 * np.sqrt(np.maximum(tau_grid, 1e-12))
    y_low = np.maximum(y_line - band, 0.0)
    y_high = y_line + band

    ax2.fill_between(tau_grid, y_low, y_high, alpha=0.15, linewidth=0.0)
    ax2.plot(tau_grid, y_line, linewidth=1.2, linestyle="--", color="black", label=r"Expected $N(\tau)=\tau$")
    ax2.plot(tau_vals, Ncum, "ko", markersize=3.2, alpha=0.9, label="Empirical")

    ax2.set_xlabel("Transformed time $\\tau$")
    ax2.set_ylabel("Cumulative # events")
    ax2.tick_params(axis="both", which="both", top=True, right=True)

    ax2.text(0.01, 0.95, "(b)", transform=ax2.transAxes,
             fontsize=panel_fs, fontweight="bold", va="top", ha="left")

    ax2.legend(frameon=True, loc="lower right")

    plt.tight_layout()
    return fig


# =============================================================================
# 6) MAIN
# =============================================================================

if __name__ == "__main__":

    # --- input: decimal years in the file ---
    filename = "../catalogs/modified_decimal_days_mL.txt" 
    Mth = 2.5
    M0  = 2.5

    # A) load in DECIMAL YEARS
    t_decimal, m_events = load_catalog_decimal_years(filename, Mth=Mth)

    # B) convert to DAYS (t=0 at the first event)
    t_days = convert_decimal_years_to_days(t_decimal, days_per_year=365.2425)

    t_start = float(t_days[0])
    t_end   = float(t_days[-1])

    # C) fit ETAS
    res, aic = fit_etas_stationary_and_aic(t_days, m_events, t_start, t_end, M0=M0)

    print("\n=== STATIONARY ETAS (times converted to days) ===")
    print("Success:", res.success, "| message:", res.message)
    print("min_obj = -logL =", res.fun)
    print("AIC =", aic)

    mu_est, K0_est, c_est, alpha_est, p_est = res.x
    print("\nOptimal parameters:")
    print(f"  mu    = {mu_est:.6g}  [events/day]")
    print(f"  K0    = {K0_est:.6g}  [events/day]")
    print(f"  c     = {c_est:.6g}  [day]")
    print(f"  alpha = {alpha_est:.6g}")
    print(f"  p     = {p_est:.6g}")

    fig = make_summary_figure(
        t_days=t_days,
        m_events=m_events,
        params=res.x,
        t_start=t_start,
        t_end=t_end,
        M0=M0,
        n_grid=600,
        n_int_points=400
    )
    plt.show()

    #fig.savefig("etas_intensity_and_residuals.png", dpi=300)
    fig.savefig("etas_intensity_and_residuals.svg", bbox_inches="tight")
    fig.savefig("etas_intensity_and_residuals.eps", format="eps", bbox_inches="tight")
