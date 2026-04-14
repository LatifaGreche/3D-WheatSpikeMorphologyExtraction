#!/usr/bin/env python
import os
import argparse
import numpy as np
import pandas as pd
import pyvista as pv
from tqdm import tqdm
import tempfile
from utils2 import * 


def process_single_file(ply_path: str) -> pd.DataFrame:
    if not os.path.isfile(ply_path):
        raise FileNotFoundError(f"PLY file not found: {ply_path}")
    name = os.path.splitext(os.path.basename(ply_path))[0]
    genotype = '-'.join(name.split('_')[:-1])
    spike_id = name.split('_')[-1]
    areas, means, centroids, volume = shape_info(ply_path)

    if len(areas) > 50:
        try:
            idx_cut = np.where(areas[50:] < 4)[0][0] + 50
            print(f"[INFO] Removing awns for {name} at index {idx_cut}")
            clipped = RemoveApicalAwns(ply_path, centroids[idx_cut])
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_path = os.path.join(tmpdir, "clipped_no_awns.ply")
                clipped.save(tmp_path)
                areas, means, centroids, volume = shape_info(tmp_path)
        except Exception as e:
            print(f"[WARN] Awn removal failed for {name}: {e}")
    else:
        print(f"[INFO] {name}: too few slices for awn removal.")
    data = {
        "Genotype ID": [genotype],
        "Spike ID": [spike_id],
        "Volume (mm³)": [volume],
        "Area": ['_'.join(map(str, areas))],
        "Centroid": ['_'.join(map(str, centroids))],
    }
    return pd.DataFrame(data)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract spike shape info after removing apical awns."
    )
    parser.add_argument(
        "--input-file",
        "-i",
        required=True,
        help="Path to the input .ply mesh file."
    )
    parser.add_argument(
        "--output-excel",
        "-o",
        required=True,
        help="Path to the output Excel file (.xlsx)."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    df = process_single_file(args.input_file)
    os.makedirs(os.path.dirname(args.output_excel) or ".", exist_ok=True)
    df.to_excel(args.output_excel, index=False)
    print(f"✅ Results saved to: {args.output_excel}")


