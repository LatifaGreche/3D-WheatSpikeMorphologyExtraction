import argparse
import os
import numpy as np
import open3d as o3d
from utils1 import *



def parse_args():
    parser = argparse.ArgumentParser(description="Align a single STL/OBJ mesh.")
    parser.add_argument(
        "--input-file",
        "-i",
        required=True,
        help="Path to the input STL or OBJ mesh to align.",
    )
    parser.add_argument(
        "--output-file",
        "-o",
        required=True,
        help="Path where the aligned mesh will be written.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Visualize both meshes together with XYZ axes in one Open3D window.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    align_single_mesh(
        input_file=args.input_file,
        output_file=args.output_file,
        show=args.show,
    )



