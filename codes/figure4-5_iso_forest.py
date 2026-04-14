#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
isolation_bump.py

Unsupervised pattern recognition on an earthquake catalog:
- temporal binning
- feature engineering (rate, magnitude statistics, depth statistics, spatial dispersion)
- anomaly detection using Isolation Forest (if sklearn is available),
  otherwise a robust fallback based on aggregated MAD z-scores

Input catalog format (whitespace-separated):
yyyy mm dd hh mm ss.ss lat lon depth Ml seqID
(lines starting with # are ignored)

Example:
python isolation_bump.py --file catalog.txt --dt 0.5 --tmax 12 --mc 2.5
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

import matplotlib as mpl

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial"],
    "font.size": 13,
    "axes.labelsize": 14,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
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


# -----------------------------
# Utilities
# -----------------------------

def robust_mad(x):
    """Median Absolute Deviation (MAD), robust scale estimator."""
    x = np.asarray(x, float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    return mad

def robust_zscore(x, eps=1e-12):
    """Robust z-score using MAD: z = 0.6745*(x-med)/MAD"""
    x = np.asarray(x, float)
    med = np.nanmedian(x)
    mad = robust_mad(x)
    if mad < eps:
        return np.zeros_like(x, dtype=float)
    return 0.6745 * (x - med) / mad

def parse_catalog(filename, mc=2.5):
    """
    Read Dingri catalog:
    yyyy mm dd hh mm ss.ss lat lon depth Ml seqID
    Ignore comments (#) and empty lines

    Keep only events with Ml >= mc
    """
    times, lat, lon, dep, mag = [], [], [], [], []

    with open(filename, "r") as f:
        for line in f:
            s = line.strip()
            if (not s) or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 11:
                continue

            yyyy = int(parts[0]); mm = int(parts[1]); dd = int(parts[2])
            hh   = int(parts[3]); mi = int(parts[4]); ss = float(parts[5])

            la = float(parts[6])
            lo = float(parts[7])
            de = float(parts[8])
            ml = float(parts[9])

            # completeness
            if ml < mc:
                continue

            sec = int(ss)
            micro = int((ss - sec) * 1e6)
            t = datetime(yyyy, mm, dd, hh, mi, sec, micro)

            times.append(t)
            lat.append(la)
            lon.append(lo)
            dep.append(de)
            mag.append(ml)

    if len(times) == 0:
        raise ValueError(f"No events with magnitude >= Mc = {mc}")

    times = np.array(times)
    lat   = np.array(lat, float)
    lon   = np.array(lon, float)
    dep   = np.array(dep, float)
    mag   = np.array(mag, float)

    # sort by time
    idx = np.argsort(times)
    return times[idx], lat[idx], lon[idx], dep[idx], mag[idx]

def times_to_days(times):
    t0 = times[0]
    return np.array([(t - t0).total_seconds()/86400.0 for t in times], float)

# -----------------------------
# Feature engineering by bins
# -----------------------------

def compute_bin_features(t_days, lat, lon, dep, mag, dt=0.5, tmin=None, tmax=None):
    """
    Bin the catalog in time windows of width dt (days).
    For each bin compute a feature vector.

    Returns:
      t_centers: (nbins,)
      X: (nbins, nfeat)
      feat_names: list[str]
      counts: (nbins,) number of events/bin
    """
    if tmin is None:
        tmin = float(np.min(t_days))
    if tmax is None:
        tmax = float(np.max(t_days))

    # bin edges
    edges = np.arange(tmin, tmax + dt + 1e-12, dt)
    nb = len(edges) - 1
    t_cent = 0.5 * (edges[:-1] + edges[1:])

    feat_names = [
        "count",
        "rate_per_day",
        "M_mean",
        "M_max",
        "frac_Mge4.5",
        "depth_mean",
        "depth_median",
        "depth_std",
        "lon_mean",
        "lon_std",
        "lat_mean",
        "lat_std",
        "spatial_area_proxy"
    ]

    X = np.full((nb, len(feat_names)), np.nan, float)
    counts = np.zeros(nb, int)

    for k in range(nb):
        a, b = edges[k], edges[k+1]
        idx = np.where((t_days >= a) & (t_days < b))[0]
        counts[k] = idx.size

        if idx.size == 0:
            X[k, 0] = 0.0
            X[k, 1] = 0.0
            continue

        mk = mag[idx]
        dk = dep[idx]
        lok = lon[idx]
        lak = lat[idx]

        std_lon = float(np.std(lok)) if idx.size >= 2 else 0.0
        std_lat = float(np.std(lak)) if idx.size >= 2 else 0.0

        X[k, 0]  = float(idx.size)
        X[k, 1]  = float(idx.size) / dt
        X[k, 2]  = float(np.mean(mk))
        X[k, 3]  = float(np.max(mk))
        X[k, 4]  = float(np.mean(mk >= 4.5))
        X[k, 5]  = float(np.mean(dk))
        X[k, 6]  = float(np.median(dk))
        X[k, 7]  = float(np.std(dk)) if idx.size >= 2 else 0.0
        X[k, 8]  = float(np.mean(lok))
        X[k, 9]  = std_lon
        X[k, 10] = float(np.mean(lak))
        X[k, 11] = std_lat
        X[k, 12] = std_lon * std_lat

    return t_cent, X, feat_names, counts

def impute_and_standardize(X):
    """
    Robust standardization:
    - replace NaNs with column median
    - robust z-score via MAD
    Returns:
      X_imp, Z (robust standardized)
    """
    X = np.asarray(X, float).copy()
    for j in range(X.shape[1]):
        col = X[:, j]
        med = np.nanmedian(col)
        col[np.isnan(col)] = med
        X[:, j] = col

    Z = np.zeros_like(X)
    for j in range(X.shape[1]):
        Z[:, j] = robust_zscore(X[:, j])

    return X, Z

# -----------------------------
# Anomaly detection
# -----------------------------

def run_isolation_forest(Z, seed=0, contamination="auto"):
    """
    Try sklearn IsolationForest.
    Returns:
      scores: higher => more anomalous
      method_name
    """
    try:
        from sklearn.ensemble import IsolationForest
    except Exception:
        return None, "fallback"

    clf = IsolationForest(
        n_estimators=500,
        random_state=seed,
        contamination=contamination
    )
    clf.fit(Z)
    normal_score = clf.decision_function(Z)
    anomaly_score = -normal_score
    return anomaly_score, "isolation_forest"

def fallback_anomaly_score(Z):
    """
    Fallback robust score always available:
    anomaly_score = sqrt(sum_j z_j^2)
    """
    return np.sqrt(np.sum(Z**2, axis=1))

# -----------------------------
# Plotting
# -----------------------------

def plot_anomaly_time(t_cent, score, method_name, highlight=(8.0, 10.0), outpng=None):
    fig, ax = plt.subplots(figsize=(9.5*0.7, 4.8*0.7))
    # Boxed axes style
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
    ax.plot(t_cent, score, 'o-', linewidth=1.2, markersize=3, color=plt.cm.plasma(0.7))
    ax.set_xlabel("Time since start (days)")
    ax.set_ylabel("Anomaly score")
    ax.grid(True, alpha=0.25)
    ax.tick_params(axis="both", which="both", top=True, right=True)

    if highlight is not None:
        a, b = highlight
        ax.axvspan(a, b, alpha=0.15)
        ax.text(0.98, 0.92, f"highlight: {a:g}-{b:g} d",
                transform=ax.transAxes, ha="right", va="top", fontsize=10)

    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.12, top=0.95)
    if outpng:
        fig.savefig(outpng, dpi=300)
        fig.savefig(outpng.replace(".png", ".svg"), bbox_inches="tight")
        fig.savefig(outpng.replace(".png", ".eps"), format="eps", bbox_inches="tight")
    return fig

def plot_feature_heatmap(t_cent, Z, feat_names, highlight=(8.0, 10.0), outpng=None):
    fig, ax = plt.subplots(figsize=(10.5*0.7, 5.5*0.7))
    im = ax.imshow(Z.T, aspect="auto", origin="lower",
                   extent=[t_cent[0], t_cent[-1], 0, Z.shape[1]], cmap="plasma")
    ax.grid(False)
    ax.plot(t_cent, np.full_like(t_cent, 0.5), 'o', markersize=2, color='black', alpha=0.6)
    ax.tick_params(axis="both", which="both", top=True, right=True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)

    ax.set_yticks(np.arange(len(feat_names)) + 0.5)
    ax.set_yticklabels(feat_names)
    ax.set_xlabel("Time since start (days)")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("z-score")

    if highlight is not None:
        a, b = highlight
        ax.axvspan(a, b, alpha=0.15)

    fig.subplots_adjust(left=0.12, right=0.95, bottom=0.12, top=0.95)
    if outpng:
        fig.savefig(outpng, dpi=300)
        fig.savefig(outpng.replace(".png", ".svg"), bbox_inches="tight")
        fig.savefig(outpng.replace(".png", ".eps"), format="eps", bbox_inches="tight")
    return fig

# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=None, help="Path to Dingri catalog txt")
    ap.add_argument("--dt", type=float, default=0.5, help="Bin width in days (default 0.5)")
    ap.add_argument("--tmin", type=float, default=None, help="Min time in days (default start)")
    ap.add_argument("--tmax", type=float, default=None, help="Max time in days (default end)")
    ap.add_argument("--mc", type=float, default=2.5, help="Magnitude of completeness Mc (default 2.5)")
    ap.add_argument("--highlight", type=float, nargs=2, default=[8.0, 10.0],
                    help="Highlight interval [t1 t2] days (default 8 10)")
    ap.add_argument("--seed", type=int, default=0, help="Random seed")
    ap.add_argument("--no_heatmap", action="store_true", help="Disable heatmap plot")
    ap.add_argument("--out_prefix", default="anomaly", help="Prefix for output figures")
    args = ap.parse_args()

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    if args.file is None:
        args.file = os.path.join(BASE_DIR, "..", "catalogs", "catalog.txt")

    # make it absolute (robust)
    args.file = os.path.abspath(args.file)

    print("Using catalog file:", args.file)

    # 1) load
    times, lat, lon, dep, mag = parse_catalog(args.file, mc=args.mc)
    t_days = times_to_days(times)

    # 2) bin features
    t_cent, X, feat_names, counts = compute_bin_features(
        t_days, lat, lon, dep, mag,
        dt=args.dt, tmin=args.tmin, tmax=args.tmax
    )

    # 3) impute & robust standardize
    X_imp, Z = impute_and_standardize(X)

    # 4) anomaly detection
    score, method = run_isolation_forest(Z, seed=args.seed)
    if score is None:
        score = fallback_anomaly_score(Z)
        method = "fallback_MAD_norm (sklearn not found)"

    # 5) plots
    plot_anomaly_time(
        t_cent, score, method,
        highlight=tuple(args.highlight),
        outpng=f"{args.out_prefix}_score.png"
    )

    if not args.no_heatmap:
        plot_feature_heatmap(
            t_cent, Z, feat_names,
            highlight=tuple(args.highlight),
            outpng=f"{args.out_prefix}_heatmap.png"
        )

    plt.show()

if __name__ == "__main__":
    main()
