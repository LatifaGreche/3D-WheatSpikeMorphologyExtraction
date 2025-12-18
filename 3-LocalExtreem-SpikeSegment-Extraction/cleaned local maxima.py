import os
import re
import glob
import numpy as np
import pandas as pd
from utils3 import *

# =========================
# Paths (edit if needed)
# =========================
INPUT_DIR = r"path to data foler\data"
FILE_GLOB = r"WATDE*.xlsx"   
OUTPUT_CSV = r"path to save outputs\ear_local_extrema_counts_cleaned.csv"


CLEAN_DISTANCE_MODE = "both"  

INDEX_DISTANCE_THRESHOLD = 1      
Z_DISTANCE_THRESHOLD = 1        
THICKNESS_DISTANCE_THRESHOLD = 3 




def main():
    files = sorted(glob.glob(os.path.join(INPUT_DIR, FILE_GLOB)))
    if not files:
        print(f"No files matching {FILE_GLOB} found in {INPUT_DIR}")
        return

    rows = []
    for f in files:
        base = os.path.splitext(os.path.basename(f))[0]  # e.g., WAT-0_131-L
        variety_id = base.split("_")[0]                  # e.g., WAT-0
        try:
            df = pd.read_excel(f, sheet_name="curves")
        except Exception as e:
            print(f"Skipping {f}: cannot read 'curves' sheet ({e})")
            continue

        trips = detect_ear_triplets(df)
        if not trips:
            print(f"Warning: no ear triplets detected in {f}")
            continue

        for ear_col, z_col, t_col in trips:
            ear_series = df[ear_col].dropna()
            ear_label = normalize_ear_label(ear_series.iloc[0] if len(ear_series) else ear_col, ear_col)
            x = df[z_col].values
            y = df[t_col].values
            mask = np.isfinite(x) & np.isfinite(y)
            x = np.asarray(x[mask], dtype=float)
            y = np.asarray(y[mask], dtype=float)
            if len(x) < 3:
                rows.append({
                    "Variety_ID": variety_id,
                    "File": base,
                    "Ear_ID": ear_label,
                    "NumLocalMaxima_raw": 0,
                    "NumLocalMinima_raw": 0,
                    "NumLocalMaxima_clean": 0,
                    "NumLocalMinima_clean": 0,
                    "Cleaned_Maxima_Z": "",
                    "Cleaned_Maxima_Thickness": "",
                    "Cleaned_Minima_Z": "",
                    "Cleaned_Minima_Thickness": "",
                    "Clean_DistanceMode": CLEAN_DISTANCE_MODE,
                    "IndexThreshold": INDEX_DISTANCE_THRESHOLD,
                    "ZThreshold": Z_DISTANCE_THRESHOLD,
                    "ThicknessThreshold": THICKNESS_DISTANCE_THRESHOLD,
                })
                continue

            order = np.argsort(x)
            x_sorted = x[order]
            y_sorted = y[order]
            max_idx, min_idx = find_extrema(y_sorted)
            nmax_raw, nmin_raw = int(len(max_idx)), int(len(min_idx))

            records = build_extrema_records(y_sorted, max_idx, min_idx)
            cleaned = clean_close_pairs(records, x_sorted, y_sorted)
            nmax_c, nmin_c, max_zs, max_vs, min_zs, min_vs = summarize_cleaned(x_sorted, y_sorted, cleaned)

            rows.append({
                "Variety_ID": variety_id,
                "File": base,
                "Ear_ID": ear_label,
                "NumLocalMaxima_raw": nmax_raw,
                "NumLocalMinima_raw": nmin_raw,
                "NumLocalMaxima_clean": nmax_c,
                "NumLocalMinima_clean": nmin_c,
                "Cleaned_Maxima_Z": max_zs,
                "Cleaned_Maxima_Thickness": max_vs,
                "Cleaned_Minima_Z": min_zs,
                "Cleaned_Minima_Thickness": min_vs,
                "Clean_DistanceMode": CLEAN_DISTANCE_MODE,
                "IndexThreshold": INDEX_DISTANCE_THRESHOLD,
                "ZThreshold": Z_DISTANCE_THRESHOLD,
                "ThicknessThreshold": THICKNESS_DISTANCE_THRESHOLD,
            })

    out_df = pd.DataFrame(rows).sort_values(["Variety_ID","File","Ear_ID"]).reset_index(drop=True)
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved cleaned extrema counts to:\n{OUTPUT_CSV}")
    print(f"Ears processed: {out_df.shape[0]}")

if __name__ == "__main__":
    main()
