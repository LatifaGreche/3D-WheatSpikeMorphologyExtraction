from utils6 import *
import os
import csv
import math
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict, deque


def try_import(name):
    try:
        return __import__(name)
    except Exception:
        return None
trimesh = try_import("trimesh")
open3d  = try_import("open3d") or try_import("open3d.cpu.pybind")
try:
    from skimage.morphology import skeletonize_3d
except Exception as e:
    raise RuntimeError("scikit-image is required (pip install scikit-image).") from e
def main():
    parser = argparse.ArgumentParser(description="Batch skeletonize ear meshes and aggregate branching stats.")
    parser.add_argument("--input_dir", type=str,
        default=r"path to ...\data",
        help="Folder containing .ply files (default: given Windows path)")
    parser.add_argument("--out_dir", type=str, default=None,
        help="Output folder (default: <input_dir>/branching_break_down_results)")
    parser.add_argument("--target_res", type=int, default=280,
        help="Target voxels along the longest bbox axis")
    parser.add_argument("--prune_ratio", type=float, default=0,#default=0.03,
        help="Leaf-prune threshold as fraction of bbox diagonal (0 disables)")
    parser.add_argument("--recurse", action="store_true",
        help="If set, search for .ply files recursively")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir) if args.out_dir else input_dir / "branching_break_down_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Gather .ply files
    files = sorted(input_dir.rglob("*.ply") if args.recurse else input_dir.glob("*.ply"))
    if not files:
        print(f"No .ply files found in: {input_dir}")
        return

    ear_csv  = out_dir / "ear_skel_components.csv"
    comp_csv = out_dir / "ear_components_appended.csv"

    ear_rows, comp_rows, errors = [], [], []

    print(f"Found {len(files)} .ply files. Processing...")
    for i, f in enumerate(files, 1):
        ear_id = f.stem
        print(f"[{i}/{len(files)}] {ear_id}")
        try:
            summary = skeletonize_mesh(
                mesh_path=str(f),
                target_res=args.target_res,
                prune_ratio=args.prune_ratio,
                out_dir=str(out_dir),
                prefix=ear_id
            )
            # ear-level row
            ear_rows.append({
                "ear_id": ear_id,
                "total_branching_points": summary["branching"]["total_branching_points"],
                "total_endpoints": summary["branching"]["total_endpoints"],
                "prune_ratio": args.prune_ratio,
            })
            # appended per-component rows
            comp_rows.extend(summary["per_component_rows"])

        except Exception as e:
            errors.append((ear_id, str(e)))
            print(f"  !! Error: {e}")

    # Write the two aggregate CSVs
    with open(str(ear_csv), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["ear_id", "total_branching_points", "total_endpoints", "prune_ratio"])
        w.writeheader()
        w.writerows(ear_rows)

    with open(str(comp_csv), "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["ear_id","component_index","size_voxels","branching_deg>=3","endpoints_deg==1","isolated_deg==0","is_lcc"]
        )
        w.writeheader()
        w.writerows(comp_rows)

    print(f"\nSaved ear-level CSV:        {ear_csv}")
    print(f"Saved appended components:  {comp_csv}")
    print(f"Skeleton PLYs written to:   {out_dir}")
    if errors:
        print("\nSome files failed:")
        for ear_id, msg in errors:
            print(f" - {ear_id}: {msg}")


if __name__ == "__main__":
    main()
