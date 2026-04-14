#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generates a semi-synthetic catalog with a strong nonstationary perturbation
between t1 and t2 days by adding a compact and spatially shifted cluster.

Input catalog format:
yyyy mm dd hh mm ss.ss lat lon depth Ml seqID

Example usage:
python simu_bump.py \
    --input catalog.txt \
    --output catalog_cluster_test.txt \
    --t1 8 --t2 10 \
    --amp 1.5 \
    --dt 0.01 \
    --new_lat_shift 0.08 \
    --new_lon_shift 0.08 \
    --cluster_sigma 0.003 \
    --depth_shift -2.0 \
    --seed 0

Interpretation:
- amp = 1.5 means adding a rate equal to 1.5 times the observed rate
  in the window [t1, t2], so the total rate increases significantly.
- new_lat_shift / new_lon_shift shift the new cluster relative to
  the real centroid in the window.
- cluster_sigma controls how compact the new cluster is.
- depth_shift shifts the depth of the new events.
"""

import argparse
import numpy as np
from datetime import datetime, timedelta

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ------------------------------------------------
# Reading catalog
# ------------------------------------------------

def read_catalog(fname):
    events = []

    with open(fname, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue

            p = s.split()
            if len(p) < 11:
                continue

            yyyy = int(p[0])
            mm   = int(p[1])
            dd   = int(p[2])
            hh   = int(p[3])
            mi   = int(p[4])
            ss   = float(p[5])

            lat  = float(p[6])
            lon  = float(p[7])
            dep  = float(p[8])
            mag  = float(p[9])
            seq  = p[10]

            sec = int(ss)
            micro = int(round((ss - sec) * 1e6))
            if micro >= 1000000:
                sec += 1
                micro -= 1000000

            t = datetime(yyyy, mm, dd, hh, mi, sec, micro)
            events.append([t, lat, lon, dep, mag, seq])

    if len(events) == 0:
        raise ValueError("Empty or invalid catalog")

    events.sort(key=lambda x: x[0])
    return events


def time_to_days(events):
    t0 = events[0][0]
    t_days = np.array([(ev[0] - t0).total_seconds() / 86400.0 for ev in events], dtype=float)
    return t0, t_days


# ------------------------------------------------
# Generating strong non stationary cluster
# ------------------------------------------------

def generate_strong_cluster(events, t_days,
                            t1, t2,
                            amp=1.5,
                            dt=0.01,
                            new_lat_shift=0.08,
                            new_lon_shift=0.08,
                            cluster_sigma=0.003,
                            depth_shift=-2.0,
                            mag_jitter=0.05,
                            depth_jitter=0.1,
                            seed=0):

    rng = np.random.default_rng(seed)

    idx = np.where((t_days >= t1) & (t_days < t2))[0]
    n0 = len(idx)

    if n0 == 0:
        raise ValueError(f"No event found in the window [{t1}, {t2}] giorni.")

    duration = t2 - t1
    if duration <= 0:
        raise ValueError("Need t2 > t1.")

   
    lambda_obs = n0 / duration
    lambda_add = amp * lambda_obs

    
    lat0 = np.array([events[i][1] for i in idx], dtype=float)
    lon0 = np.array([events[i][2] for i in idx], dtype=float)
    dep0 = np.array([events[i][3] for i in idx], dtype=float)
    mag0 = np.array([events[i][4] for i in idx], dtype=float)

    
    lat_mean = np.mean(lat0)
    lon_mean = np.mean(lon0)

    
    lat_new_center = lat_mean + new_lat_shift
    lon_new_center = lon_mean + new_lon_shift

    t0 = events[0][0]
    new_events = events.copy()

    edges = np.arange(t1, t2 + 1e-12, dt)
    if edges[-1] < t2:
        edges = np.append(edges, t2)

    n_generated = 0

    for a, b in zip(edges[:-1], edges[1:]):
        mu = lambda_add * (b - a)
        nk = rng.poisson(mu)

        for _ in range(nk):
            
            tau = rng.uniform(a, b)
            t_new = t0 + timedelta(days=float(tau))

            
            lat_new = rng.normal(lat_new_center, cluster_sigma)
            lon_new = rng.normal(lon_new_center, cluster_sigma)

            
            j = rng.integers(0, n0)
            dep_new = dep0[j] + depth_shift + rng.normal(0.0, depth_jitter)
            mag_new = mag0[j] + rng.normal(0.0, mag_jitter)

           
            dep_new = max(dep_new, 0.0)

           

            seq_new = f"{n_generated+1:06d}"
            new_events.append([t_new, lat_new, lon_new, dep_new, mag_new, seq_new])
            n_generated += 1

    new_events.sort(key=lambda x: x[0])

    info = {
        "n_window_original": n0,
        "lambda_obs": lambda_obs,
        "lambda_add": lambda_add,
        "n_generated": n_generated,
        "lat_mean_window": lat_mean,
        "lon_mean_window": lon_mean,
        "lat_new_center": lat_new_center,
        "lon_new_center": lon_new_center
    }

    return new_events, info


# ------------------------------------------------
# Writing catalog
# ------------------------------------------------

def write_catalog(events, fname):
    with open(fname, "w") as f:
        for ev in events:
            t, lat, lon, dep, mag, seq = ev
            ss = t.second + t.microsecond / 1e6
            line = (
                f"{t.year:4d} {t.month:02d} {t.day:02d} "
                f"{t.hour:02d} {t.minute:02d} {ss:06.3f} "
                f"{lat:.5f} {lon:.5f} {dep:.3f} {mag:.2f} {seq}\n"
            )
            f.write(line)


# ------------------------------------------------
# Main
# ------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=None, help="Catalogo input")
    ap.add_argument("--output", default=None, help="Catalogo output")
    ap.add_argument("--t1", type=float, default=8.0, help="Inizio finestra (giorni)")
    ap.add_argument("--t2", type=float, default=10.0, help="Fine finestra (giorni)")
    ap.add_argument("--amp", type=float, default=1.5,
                    help="Rate additivo relativo al rate osservato nella finestra")
    ap.add_argument("--dt", type=float, default=0.01,
                    help="Passo temporale per Poisson (giorni)")
    ap.add_argument("--new_lat_shift", type=float, default=0.08,
                    help="Shift del nuovo cluster in latitudine")
    ap.add_argument("--new_lon_shift", type=float, default=0.08,
                    help="Shift del nuovo cluster in longitudine")
    ap.add_argument("--cluster_sigma", type=float, default=0.003,
                    help="Deviazione standard spaziale del nuovo cluster")
    ap.add_argument("--depth_shift", type=float, default=-2.0,
                    help="Shift di profondità dei nuovi eventi")
    ap.add_argument("--mag_jitter", type=float, default=0.05,
                    help="Piccolo rumore su magnitudo")
    ap.add_argument("--depth_jitter", type=float, default=0.1,
                    help="Piccolo rumore su profondità")
    ap.add_argument("--seed", type=int, default=0, help="Seed random")
    args = ap.parse_args()

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # default input: ../catalogs/catalog.txt
    if args.input is None:
        args.input = os.path.join(BASE_DIR, "..", "catalogs", "catalog.txt")

    # default output: ../output_nonstat_etas/catalog_cluster_test.txt
    if args.output is None:
        OUTDIR = os.path.join(BASE_DIR, "..", "output_nonstat_etas")
        os.makedirs(OUTDIR, exist_ok=True)
        args.output = os.path.join(OUTDIR, "catalog_cluster_test.txt")

    # make paths absolute (robust)
    args.input = os.path.abspath(args.input)
    args.output = os.path.abspath(args.output)

    print("Input file:", args.input)
    print("Output file:", args.output)

    if args.t2 <= args.t1:
        raise ValueError("Serve t2 > t1.")
    if args.dt <= 0:
        raise ValueError("dt deve essere positivo.")
    if args.cluster_sigma <= 0:
        raise ValueError("cluster_sigma deve essere positivo.")

    events = read_catalog(args.input)
    _, t_days = time_to_days(events)

    new_events, info = generate_strong_cluster(
        events, t_days,
        t1=args.t1,
        t2=args.t2,
        amp=args.amp,
        dt=args.dt,
        new_lat_shift=args.new_lat_shift,
        new_lon_shift=args.new_lon_shift,
        cluster_sigma=args.cluster_sigma,
        depth_shift=args.depth_shift,
        mag_jitter=args.mag_jitter,
        depth_jitter=args.depth_jitter,
        seed=args.seed
    )

    print(f"Original events:               {len(events)}")
    print(f"Events in [{args.t1}, {args.t2}] d:       {info['n_window_original']}")
    print(f"Observed rate in window:       {info['lambda_obs']:.3f} ev/day")
    print(f"Added rate:                    {info['lambda_add']:.3f} ev/day")
    print(f"Generated extra events:        {info['n_generated']}")
    print(f"Final events:                  {len(new_events)}")
    print(f"Original window centroid:      ({info['lat_mean_window']:.5f}, {info['lon_mean_window']:.5f})")
    print(f"New artificial cluster center: ({info['lat_new_center']:.5f}, {info['lon_new_center']:.5f})")

    write_catalog(new_events, args.output)


if __name__ == "__main__":
    main()
