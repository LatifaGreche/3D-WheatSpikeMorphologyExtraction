import os
import re
import glob
import numpy as np
import pandas as pd


def detect_ear_triplets(df):

    cols = list(df.columns)
    trips = []
    i = 0
    while i < len(cols):
        name = str(cols[i]).strip()
        if re.search(r"^Ear ID", name, flags=re.I):
            z_col, t_col = None, None
            for j in range(i+1, min(i+6, len(cols))):
                cand = str(cols[j]).strip()
                if z_col is None and re.search(r"^Z\s*coor", cand, flags=re.I):
                    z_col = cand
                elif t_col is None and re.search(r"^thickness", cand, flags=re.I):
                    t_col = cand
                if z_col and t_col:
                    break
            if z_col and t_col:
                trips.append((name, z_col, t_col))
                i = cols.index(t_col, i) + 1
                continue
        i += 1
    return trips

def normalize_ear_label(val, default_label):
    try:
        s = str(val)
    except Exception:
        return default_label
    m = re.search(r"(ear\d+)", s, flags=re.I)
    return m.group(1).lower() if m else s.strip("[]' ").strip()

def find_extrema(vals):

    maxima_idx, minima_idx = [], []
    for i in range(1, len(vals)-1):
        if vals[i] > vals[i-1] and vals[i] > vals[i+1]:
            maxima_idx.append(i)
        if vals[i] < vals[i-1] and vals[i] < vals[i+1]:
            minima_idx.append(i)
    return np.array(maxima_idx, dtype=int), np.array(minima_idx, dtype=int)

def local_prominence(vals, idx, kind):
    i = idx
    if kind == "max":
        left  = vals[i] - vals[i-1]
        right = vals[i] - vals[i+1]
        return float(max(0.0, min(left, right)))
    else:  # "min"
        left  = vals[i-1] - vals[i]
        right = vals[i+1] - vals[i]
        return float(max(0.0, min(left, right)))

def build_extrema_records(vals, max_idx, min_idx):

    recs = []
    for i in max_idx:
        recs.append((int(i), float(vals[i]), "max", local_prominence(vals, i, "max")))
    for i in min_idx:
        recs.append((int(i), float(vals[i]), "min", local_prominence(vals, i, "min")))
    recs.sort(key=lambda t: t[0])
    return recs

def very_close(idx1, idx2, x_sorted, y_sorted):

    if CLEAN_DISTANCE_MODE == "index":
        return (idx2 - idx1) <= INDEX_DISTANCE_THRESHOLD
    elif CLEAN_DISTANCE_MODE == "z":
        dz = abs(float(x_sorted[idx2]) - float(x_sorted[idx1]))
        return dz <= Z_DISTANCE_THRESHOLD
    elif CLEAN_DISTANCE_MODE == "thickness":
        dy = abs(float(y_sorted[idx2]) - float(y_sorted[idx1]))
        return dy <= THICKNESS_DISTANCE_THRESHOLD
    elif CLEAN_DISTANCE_MODE == "both":
        dz = abs(float(x_sorted[idx2]) - float(x_sorted[idx1]))
        dy = abs(float(y_sorted[idx2]) - float(y_sorted[idx1]))
        return (dz <= Z_DISTANCE_THRESHOLD) and (dy <= THICKNESS_DISTANCE_THRESHOLD)
    else:
        return (idx2 - idx1) <= INDEX_DISTANCE_THRESHOLD

def clean_close_pairs(extrema, x_sorted, y_sorted):

    if not extrema:
        return extrema
    extrema = extrema.copy()
    changed = True
    while changed:
        changed = False
        keep = []
        i = 0
        while i < len(extrema):
            if i < len(extrema) - 1:
                idx1, val1, k1, prom1 = extrema[i]
                idx2, val2, k2, prom2 = extrema[i+1]
                if k1 != k2 and very_close(idx1, idx2, x_sorted, y_sorted):

                    if prom1 > prom2:
                        keep.append(extrema[i])
                    elif prom2 > prom1:
                        keep.append(extrema[i+1])
                    else:
                        keep.append(extrema[i+1])
                    i += 2
                    changed = True
                    continue
            keep.append(extrema[i])
            i += 1
        extrema = keep
    return extrema

def summarize_cleaned(x_sorted, y_sorted, cleaned_records):

    if not cleaned_records:
        return 0, 0, "", "", "", ""
    c_max = [(ix, val) for (ix, val, k, p) in cleaned_records if k == "max"]
    c_min = [(ix, val) for (ix, val, k, p) in cleaned_records if k == "min"]
    max_zs = ";".join(f"{x_sorted[ix]:.6f}" for ix, _ in c_max)
    max_vs = ";".join(f"{v:.6f}" for _, v in c_max)
    min_zs = ";".join(f"{x_sorted[ix]:.6f}" for ix, _ in c_min)
    min_vs = ";".join(f"{v:.6f}" for _, v in c_min)
    return len(c_max), len(c_min), max_zs, max_vs, min_zs, min_vs