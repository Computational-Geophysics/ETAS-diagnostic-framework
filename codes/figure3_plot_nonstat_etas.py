#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import matplotlib as mpl

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial"],
    "font.size": 11,
    "axes.labelsize": 12,
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

# ----------------------------
# Paths
# ----------------------------
OUTDIR = "output_nonstat_etas"
CATFILE = os.path.join(OUTDIR, "catalog_used.dat")
TSFILE  = os.path.join(OUTDIR, "fitted_time_series.dat")
RESFILE = os.path.join(OUTDIR, "residuals_time_transform.dat")

FIG_PNG = os.path.join(OUTDIR, "figure_multipanel_nonstat_etas.png")
FIG_PDF = os.path.join(OUTDIR, "figure_multipanel_nonstat_etas.pdf")

# ----------------------------
# Load helpers
# ----------------------------
def _load_with_header(path):
    
    return np.loadtxt(path)

def magnitude_sizes(m, smin=6.0, smax=80.0, gamma=2.0):
    m = np.asarray(m, float)
    mmin, mmax = float(np.min(m)), float(np.max(m))
    u = (m - mmin) / (mmax - mmin + 1e-12)
    u = u**gamma
    return smin + (smax - smin) * u

# ----------------------------
# Main plotting
# ----------------------------
def make_multipanel():
    # --- catalog_used.dat: t_decimal, t_day, magnitude
    cat = _load_with_header(CATFILE)
    t_evt = cat[:, 1].astype(float)
    m_evt = cat[:, 2].astype(float)

    # --- fitted_time_series.dat:
    # t_day, lambda(t), mu(t), q_mu(t), q_K0(t), K0(t)
    ts = _load_with_header(TSFILE)
    t = ts[:, 0].astype(float)
    lam = ts[:, 1].astype(float)
    mu  = ts[:, 2].astype(float)
    K0  = ts[:, 5].astype(float)

    # --- residuals_time_transform.dat: index, t_day, tau(t), Ncum
    res = _load_with_header(RESFILE)
    tau_vals = res[:, 2].astype(float)
    Ncum = res[:, 3].astype(float)

    # sanity
    ok_lam = np.isfinite(lam) & (lam > 0)
    ok_mu  = np.isfinite(mu)  & (mu > 0)
    ok_K0  = np.isfinite(K0)  & (K0 > 0)

    # sizes for event scatter
    sizes = magnitude_sizes(m_evt)

    # figure layout: 2 cols; left has 2 rows; right spans rows
    fig = plt.figure(figsize=(7.8, 3.9))
    gs = GridSpec(nrows=2, ncols=2, figure=fig, width_ratios=[1.05, 1.05], height_ratios=[1.0, 0.85],
                  wspace=0.22, hspace=0.18)

    ax_a = fig.add_subplot(gs[0, 0])     # (a) lambda
    ax_b = fig.add_subplot(gs[1, 0])     # (b) residuals
    ax_c = fig.add_subplot(gs[:, 1])     # right column spanning both rows

    for ax in [ax_a, ax_b, ax_c]:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.0)

    # ----------------------------
    # (a) Conditional intensity + catalog
    # ----------------------------
    ax_a.plot(t[ok_lam], lam[ok_lam], lw=1.8)
    ax_a.set_yscale("log")
    ax_a.set_ylabel(r"$\lambda(t)$ [events/day]")
    ax_a.tick_params(axis="both", which="both", top=True, right=True)
    ax_a.text(0.01, 0.95, "(a)", transform=ax_a.transAxes,
              fontsize=12, fontweight="bold", va="top", ha="left")

    ax_a_r = ax_a.twinx()
    ax_a_r.scatter(t_evt, m_evt, s=sizes*0.6, c="0.2", alpha=0.25, edgecolors="none", zorder=0)
    ax_a_r.set_ylabel("Magnitude")
    ax_a_r.tick_params(axis="y", which="both", right=True)

    # ----------------------------
    # (b) Residuals in transformed time
    # ----------------------------
    # expected N(tau)=tau with 95% band
    tau_max = float(np.nanmax(tau_vals))
    tau_grid = np.linspace(0.0, max(tau_max, 1e-12), 500)

    y_line = tau_grid
    band = 1.96 * np.sqrt(np.maximum(tau_grid, 1e-12))
    y_low = np.maximum(y_line - band, 0.0)
    y_high = y_line + band

    ax_b.fill_between(tau_grid, y_low, y_high, alpha=0.15, linewidth=0.0)
    ax_b.plot(tau_grid, y_line, linestyle="--", linewidth=1.2, color="black", label=r"Expected $N(\tau)=\tau$")
    ax_b.plot(tau_vals, Ncum, "ko", markersize=3.2, alpha=0.9, label="Empirical")

    ax_b.set_xlabel(r"Transformed time $\tau$")
    ax_b.set_ylabel("Cumulative # events")
    ax_b.tick_params(axis="both", which="both", top=True, right=True)
    ax_b.text(0.01, 0.95, "(b)", transform=ax_b.transAxes,
              fontsize=12, fontweight="bold", va="top", ha="left")
    ax_b.legend(frameon=True, fontsize=12, loc="lower right")

    # ----------------------------
    # (c) mu(t) and K0(t) + catalog background
    # ----------------------------
    # plot mu and K0 on same axis (log), with legend
    ax_c.plot(t[ok_mu], mu[ok_mu], lw=1.8, label=r"$\mu(t)$")
    ax_c.plot(t[ok_K0], K0[ok_K0], lw=1.8, ls="--", label=r"$K_0(t)$")
    ax_c.set_yscale("log")
    ax_c.set_ylabel(r"$\mu(t)$ and $K_0(t)$")
    ax_c.tick_params(axis="both", which="both", top=True, right=True)
    ax_c.text(0.01, 0.98, "(c)", transform=ax_c.transAxes,
              fontsize=12, fontweight="bold", va="top", ha="left")
    ax_c.legend(frameon=True, fontsize=12, loc="upper right")

    ax_c_r = ax_c.twinx()
    ax_c_r.scatter(t_evt, m_evt, s=sizes*0.6, c="0.2", alpha=0.25, edgecolors="none", zorder=0)
    ax_c_r.set_ylabel("Magnitude")
    ax_c_r.tick_params(axis="y", which="both", right=True)

    # common x-label only on bottom-left + right (optional)
    ax_a.set_xticklabels([])  # cleaner like many multi-panels
    ax_c.set_xlabel("Time [days since catalog start]")

    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.10, top=0.95, wspace=0.25, hspace=0.25)

    fig.savefig(FIG_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(OUTDIR, "figure_multipanel_nonstat_etas.svg"), bbox_inches="tight")
    fig.savefig(os.path.join(OUTDIR, "figure_multipanel_nonstat_etas.eps"), format="eps", bbox_inches="tight")
    fig.savefig(FIG_PDF, bbox_inches="tight")
    print(f"[OK] Saved: {FIG_PNG}")
    print(f"[OK] Saved: {FIG_PDF}")

    plt.show()

if __name__ == "__main__":
    make_multipanel()
