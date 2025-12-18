import os
import re
import numpy as np
import pandas as pd

# ----------------------------
# Parsers
# ----------------------------
_FLOAT_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

def parse_area(area_str):
    """'1.2_3.4_...' -> np.array([1.2, 3.4, ...])"""
    if pd.isna(area_str):
        return np.array([], dtype=float)
    out = []
    for t in str(area_str).split("_"):
        try:
            out.append(float(t))
        except ValueError:
            pass
    return np.array(out, dtype=float)

def parse_centroids(centroid_str):
    """
    Expected formats seen in your files:
      "[x y z]_[x y z]_..."  (underscore separated)
    Returns:
      xyz: (N,3) float array
      z_corr: list of z's
      coor: list of np.array([x,y,z])
      skeleton_df: DataFrame columns x,y,z
    """
    if pd.isna(centroid_str):
        empty_df = pd.DataFrame(columns=["x", "y", "z"])
        return np.empty((0, 3), dtype=float), [], [], empty_df

    rows = []
    z_corr = []
    coor = []

    for chunk in str(centroid_str).split("_"):
        nums = _FLOAT_RE.findall(chunk)
        if len(nums) < 3:
            continue
        x, y, z = map(float, nums[:3])
        z_corr.append(z)
        coor.append(np.array([x, y, z], dtype=float))
        rows.append({"x": x, "y": y, "z": z})

    skeleton_df = pd.DataFrame(rows)
    xyz = skeleton_df[["x", "y", "z"]].to_numpy(dtype=float) if len(skeleton_df) else np.empty((0,3), dtype=float)
    return xyz, z_corr, coor, skeleton_df

def safe_sheet_name(name: str) -> str:
    """Excel sheet name rules: <=31 chars and cannot contain : \ / ? * [ ]"""
    bad = [":", "\\", "/", "?", "*", "[", "]"]
    for b in bad:
        name = name.replace(b, "_")
    return name[:31] if len(name) > 31 else name

# ----------------------------
# Column detection
# ----------------------------
def pick_first_existing(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def normalize_input_dataframe(path_file, sheet_name=0):
    df = pd.read_excel(path_file, sheet_name=sheet_name)

    # Try to support both file schemas
    col_genotype = pick_first_existing(df, ["Genotype ID", "variety ID", "Variety", "Genotype", "variety"])
    col_spike    = pick_first_existing(df, ["Spike ID", "ear ID", "Ear ID", "spike ID", "Spike"])
    col_area     = pick_first_existing(df, ["Area", "thickness", "Thickness", "area"])
    col_centroid = pick_first_existing(df, ["Centroid", "centroid", "Centroids"])
    col_volume   = pick_first_existing(df, ["Volume (mm³)", "volume", "Volume", "volume (mm3)"])

    missing = [("Area/thickness", col_area), ("Centroid", col_centroid)]
    missing = [name for name, col in missing if col is None]
    if missing:
        raise ValueError(
            f"Missing required column(s): {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    # Build a normalized table with consistent names
    out = pd.DataFrame({
        "genotype_id": df[col_genotype].astype(str) if col_genotype else "NA",
        "spike_id": df[col_spike].astype(str) if col_spike else df.index.astype(str),
        "area_str": df[col_area],
        "centroid_str": df[col_centroid],
    })
    if col_volume:
        out["volume"] = df[col_volume]
    else:
        out["volume"] = np.nan

    return out

# ----------------------------
# Main processor
# ----------------------------
def process_spike_excel(
    path_file: str,
    output_dir: str,
    sheet_name=0,
    save_each_spike=True,
    combined_output_name="all_spikes_thickness.xlsx"
):
    os.makedirs(output_dir, exist_ok=True)

    norm = normalize_input_dataframe(path_file, sheet_name=sheet_name)

    combined_rows = []  # for optional combined output

    for area_str, cent_str, geno, spike in zip(norm["area_str"], norm["centroid_str"], norm["genotype_id"], norm["spike_id"]):
        xyz, z_corr, coor, skeleton_df = parse_centroids(cent_str)
        Y_t = parse_area(area_str)

        if len(z_corr) == 0 or len(Y_t) == 0:
            print(f"Skipping {geno} / {spike}: empty centroids or area after parsing")
            continue

        X_t = np.array(z_corr, dtype=float)
        X_t = X_t - X_t[0]  # translate so first slice is zero

        # Align lengths safely (some rows can mismatch)
        n = min(len(X_t), len(Y_t))
        X_t = X_t[:n]
        Y_t = Y_t[:n]

        slicess = np.linspace(1, n, num=n)

        df_out = pd.DataFrame(
            {"Z coor": X_t.tolist(), "thickness": Y_t.tolist()},
            index=slicess
        )

        # Save per spike (like your original code)
        if save_each_spike:
            fname = f"{geno}_{spike}_thickness.xlsx"
            save_path = os.path.join(output_dir, fname)
            df_out.to_excel(save_path, sheet_name=safe_sheet_name(f"{geno}_{spike}"))

        # For combined output (one workbook with many sheets, or one long table)
        df_long = df_out.reset_index(names="slice")
        df_long.insert(0, "spike_id", spike)
        df_long.insert(0, "genotype_id", geno)
        combined_rows.append(df_long)

    # One combined excel with a single sheet (long format)
    if combined_rows:
        combined = pd.concat(combined_rows, ignore_index=True)
        combined_path = os.path.join(output_dir, combined_output_name)
        combined.to_excel(combined_path, index=False)
        return combined_path

    return None

# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    # 1) One-sample file
    process_spike_excel(
        path_file=r"C:\Users\grechel\OneDrive - Rothamsted Research\paper\github code\2-slicing\result.xlsx",
        output_dir=r"C:\Users\grechel\OneDrive - Rothamsted Research\paper\github code\2-slicing\out_result",
        sheet_name=0,
        save_each_spike=True,
        combined_output_name="result_all_spikes_long.xlsx"
    )

    # 2) Many-samples file
    #process_spike_excel(
      #  path_file=r"W:\spike shape analysis\results\shape_analysis_of_all_spikes_ply_Va.xlsx",
      #  output_dir=r"W:\spike shape analysis\results\out_all_spikes",
      #  sheet_name=0,
      #  save_each_spike=False,  # often better for large files
     #   combined_output_name="all_spikes_long.xlsx"
    #)
