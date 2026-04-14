import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ----------------------------
# helpers
# ----------------------------
def gaussian_smooth(y, sigma_pts=5):
    """
    Smooth 1D array with a Gaussian kernel (sigma in number of points).
    """
    y = np.asarray(y, float)
    if sigma_pts <= 0:
        return y.copy()

    radius = int(np.ceil(4 * sigma_pts))
    x = np.arange(-radius, radius + 1)
    k = np.exp(-0.5 * (x / sigma_pts) ** 2)
    k /= k.sum()

    # pad with edge values to avoid boundary artifacts
    ypad = np.pad(y, (radius, radius), mode="edge")
    ys = np.convolve(ypad, k, mode="same")[radius:-radius]
    return ys

def bin_time_series(t, y, dt=0.1, agg="median"):
    """
    Bin (t,y) into uniform time bins of width dt.
    agg: "median" or "mean".
    Returns: t_center, y_agg (NaN where empty).
    """
    t = np.asarray(t)
    y = np.asarray(y)
    tmin, tmax = np.nanmin(t), np.nanmax(t)

    edges = np.arange(tmin, tmax + dt, dt)
    centers = 0.5 * (edges[:-1] + edges[1:])
    yb = np.full_like(centers, np.nan, dtype=float)

    which = np.digitize(t, edges) - 1
    for i in range(len(centers)):
        mask = which == i
        if np.any(mask):
            if agg == "mean":
                yb[i] = np.nanmean(y[mask])
            else:
                yb[i] = np.nanmedian(y[mask])

    return centers, yb

# ----------------------------
# input files
# ----------------------------
cat_file = "../catalogs/decimal_days_mL.txt"   # 
mc_file  = "../catalogs/fort.10"               #

# ----------------------------
# load data
# ----------------------------
cat = np.loadtxt(cat_file)
t_cat, m_cat = cat[:, 0], cat[:, 1]

mc = np.loadtxt(mc_file)
t_mc, mc_t = mc[:, 0], mc[:, -1]

# sort Mc by time (safe)
idx = np.argsort(t_mc)
t_mc, mc_t = t_mc[idx], mc_t[idx]

# ----------------------------
# scatter style: size ~ magnitude
# ----------------------------
mmin, mmax = np.nanmin(m_cat), np.nanmax(m_cat)

# mapping size: tune these two if you want bigger/smaller points
s_min, s_max = 5.0, 80.0
# normalize magnitude into [0,1]
u = (m_cat - mmin) / (mmax - mmin + 1e-12)
# emphasize big magnitudes a bit (power law)
u = u**2
sizes = s_min + (s_max - s_min) * u

# ----------------------------
# Mc smoothing: bin + gaussian smooth
# ----------------------------
# 1) binning (stabilizza spike)
dt_bin = 0.1      # giorni (es. 0.1 = 2.4h). Prova anche 0.05 o 0.2
t_b, mc_b = bin_time_series(t_mc, mc_t, dt=dt_bin, agg="median")

# rimuovi bin vuoti (NaN) prima di smooth
mask = ~np.isnan(mc_b)
t_b2, mc_b2 = t_b[mask], mc_b[mask]

# 2) gaussian smoothing on binned series
sigma_pts = 4     # in numero di punti (dipende da dt_bin). Prova 2-8.
mc_s = gaussian_smooth(mc_b2, sigma_pts=sigma_pts)

# ----------------------------
# plot (clean GMT-style)
# ----------------------------
import matplotlib as mpl

# --- GLOBAL FONT CONTROL (do it here, once) ---
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial"],
    "font.size": 12,          # base font
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "axes.linewidth": 1.0,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})

fig, ax = plt.subplots(figsize=(8.6, 2.4))  # wide + compressed

# --- color by magnitude ---
norm = Normalize(vmin=mmin, vmax=mmax)
cmap = cm.viridis
#cmap = cm.cividis #viridis

sc = ax.scatter(
    t_cat, m_cat,
    s=sizes * 0.6,
    c=m_cat,
    cmap=cmap,
    norm=norm,
    alpha=0.8,
    edgecolors="black",    # colore contorno
    linewidths=0.05        # spessore contorno
)

# Mc curve (clean line)
ax.plot(t_b2, mc_s, color="red", lw=1.2)
cbar = fig.colorbar(sc, ax=ax, pad=0.02)
cbar.set_label("Magnitude (Mw)")

# labels (NO manual fontsize here)
ax.set_xlabel("Time (days)")
ax.set_ylabel("Magnitude")

# GMT-style axes
ax.spines["top"].set_visible(True)
ax.spines["right"].set_visible(True)

# no grid
ax.grid(False)

# legend (minimal)
ax.legend(["$M_c(t)$"], frameon=False)

# layout (controlled, not tight_layout)
fig.subplots_adjust(left=0.08, right=0.98, bottom=0.18, top=0.95)

# save vector formats for Illustrator
fig.savefig("figure1.svg", bbox_inches="tight")
fig.savefig("figure1.eps", format="eps", bbox_inches="tight")

plt.show()
