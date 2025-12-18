import os
import re
import numpy as np
import pandas as pd


def clean_cols(cols):
    """Lowercase, strip spaces for robust matching."""
    return [str(c).strip() for c in cols]

def detect_ear_triplets(df):
    cols = list(df.columns)
    triplets = []
    i = 0
    while i < len(cols):
        name = str(cols[i]).strip()
        if re.search(r'^Ear ID', name, flags=re.I):
            # Find the next two columns that look like Z and thickness
            z_col = None
            t_col = None
            # look ahead a few columns (sheet sometimes has extra cols like Variety ID)
            for j in range(i+1, min(i+6, len(cols))):
                cand = str(cols[j]).strip()
                if z_col is None and re.search(r'^Z\s*coor', cand, flags=re.I):
                    z_col = cand
                elif t_col is None and re.search(r'^thickness', cand, flags=re.I):
                    t_col = cand
                if z_col is not None and t_col is not None:
                    break
            if z_col and t_col:
                triplets.append((name, z_col, t_col))
                # move index past the pair we just consumed; continue scanning
                i = cols.index(t_col, i) + 1
                continue
        i += 1
    return triplets

def segment_profile(x, y, base_eps=BASE_EPS):
    # Drop NaNs and ensure increasing x
    mask = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x[mask], dtype=float)
    y = np.asarray(y[mask], dtype=float)
    if len(x) < 3:
        return []

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    peak_idx = int(np.argmax(y))
    y_peak = y[peak_idx]
    y_base0 = y[0]
    threshold = y_base0 + base_eps * max(0.0, (y_peak - y_base0))
    base_end_idx = int(np.argmax(y > threshold)) if np.any(y > threshold) else 0

    segments = {
        "Base": (0, base_end_idx),
        "Rising": (base_end_idx, peak_idx),
        "Declining": (peak_idx, len(y) - 1)
    }

    out = []
    for seg, (i1, i2) in segments.items():
        if i2 > i1:
            xs = x[i1:i2+1]
            ys = y[i1:i2+1]
            length = float(xs[-1] - xs[0])
            mean_thick = float(np.mean(ys))
            slope = float((ys[-1] - ys[0]) / length) if length != 0 else np.nan
            auc = float(np.trapz(ys, xs))
            out.append((seg, float(xs[0]), float(xs[-1]), length, mean_thick, slope, auc))
    return out

def safe_get(series, default=""):
    try:
        v = series.dropna().iloc[0]
        return v if pd.notna(v) else default
    except Exception:
        return default
