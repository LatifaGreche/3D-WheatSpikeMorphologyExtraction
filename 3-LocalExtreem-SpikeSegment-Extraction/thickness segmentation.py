import os
import re
import numpy as np
import pandas as pd
from utils4 import *
# -------------------------------
# User inputs 
# -------------------------------
INPUT_FILES = [
    r"...\data\WATDE0020.xlsx",
    r"...\data\WATDE0045.xlsx",
    r"...\data\WATDE0227.xlsx",
    r"...\data\WATDE0228.xlsx",
    r"...\data\WATDE0253.xlsx",
    r"...\data\WATDE0296.xlsx",
    r"...\data\WATDE0323.xlsx",
    r"...\data\WATDE0347.xlsx",
    r"...\data\WATDE0345.xlsx",
    r"...\data\WATDE0784.xlsx",
    r"...\data\WATDE0930.xlsx",
    r"...\data\Paragon.xlsx",
]
OUTPUT_XLSX = r"path to output\Spike_Thickness_Segmentation.xlsx"


BASE_EPS = 0.10  



all_rows = []

for f in INPUT_FILES:
    if not os.path.isfile(f):
        print(f"WARNING: file not found: {f}")
        continue

    base = os.path.splitext(os.path.basename(f))[0]
    df = pd.read_excel(f, sheet_name="curves")
    df.columns = clean_cols(df.columns)
    variety_from_col = df.get("Variety ID") if "Variety ID" in df.columns else None
    variety_fallback = base.split("_")[0]  # e.g., WAT-0 from "WAT-0_131-L"
    variety_name = safe_get(variety_from_col, variety_fallback)


    triplets = detect_ear_triplets(df)
    if not triplets:
        print(f"WARNING: no ear triplets detected in {f}")
        continue
        ear_label = safe_get(df[ear_id_col], ear_id_col)
        x = df[z_col].values
        y = df[t_col].values
        segments = segment_profile(x, y, base_eps=BASE_EPS)
        for seg_name, x_start, x_end, length, mean_thick, slope, auc in segments:
            all_rows.append({
                "Variety_ID": variety_name,
                "Ear_ID": ear_label,
                "File": base,
                "Ear_ID_Column": ear_id_col,
                "Z_Column": z_col,
                "Thickness_Column": t_col,
                "Segment": seg_name,
                "X_start": x_start,
                "X_end": x_end,
                "Length": length,
                "MeanThickness": mean_thick,
                "Slope": slope,
                "AUC": auc
            })

# Save results
out_df = pd.DataFrame(all_rows)
os.makedirs(os.path.dirname(OUTPUT_XLSX), exist_ok=True)
out_df.to_excel(OUTPUT_XLSX, index=False)
print(f"Saved segmentation for {len(out_df)} segments -> {OUTPUT_XLSX}")
