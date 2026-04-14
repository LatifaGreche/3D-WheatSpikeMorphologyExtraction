

import os
import csv
import pyvista as pv
import numpy as np
from scipy.interpolate import UnivariateSpline


# paths to spike files 
SPIKE_PATHS = ["list of paths .ply or change the code to work in loop"]


# parameters

NUMBER_OF_SLICES   = 100      
MAD_THRESHOLD      = 3.5    
SMOOTH_FACTOR      = 2.0    
N_SPLINE_SAMPLES   = 200     
###

OUTPUT_CSV = os.path.join(r"..", "skeleton_lengths.csv")


def clean_main_mesh(mesh):
    return mesh.connectivity(largest=True)


def clean_slice_largest(slc):

    if slc.n_points == 0:
        return slc
    try:
        return slc.connectivity(largest=True)
    except Exception:
        return slc


def get_clean_slices(mesh, number_of_slices):
    raw_slices = mesh.slice_along_axis(
        n=number_of_slices,
        axis='z',
        tolerance=None,
        generate_triangles=False,
        contour=False,
        bounds=None,
        center=None,
        progress_bar=False,
    )
    out = pv.MultiBlock()
    for i, slc in enumerate(raw_slices):
        out[f"slice{i}"] = clean_slice_largest(slc)
    return out


def centroid_np(arr):
    if arr.shape[0] == 0:
        return np.array([np.nan, np.nan, np.nan])
    return np.array([np.mean(arr[:, 0]),
                     np.mean(arr[:, 1]),
                     np.mean(arr[:, 2])])


def find_centroids_from_slices(slices, number_of_slices):

    points = np.full((number_of_slices, 3), np.nan, dtype=float)
    for i in range(number_of_slices):
        slc = slices[f"slice{i}"]
        if slc.n_points > 0:
            points[i] = centroid_np(slc.points)
    return points


def remove_centroid_outliers(points, threshold=3.5):
 
    valid = ~np.isnan(points).any(axis=1)
    pts = points[valid]

    if len(pts) < 5:
        return pts.copy()

    x = pts[:, 0]
    y = pts[:, 1]

    med_x, med_y = np.median(x), np.median(y)

    mad_x = np.median(np.abs(x - med_x))
    mad_y = np.median(np.abs(y - med_y))

  
    if mad_x == 0:
        mad_x = 1e-12
    if mad_y == 0:
        mad_y = 1e-12

    mz_x = 0.6745 * (x - med_x) / mad_x
    mz_y = 0.6745 * (y - med_y) / mad_y

    keep = (np.abs(mz_x) < threshold) & (np.abs(mz_y) < threshold)
    return pts[keep]


def smooth_centroid_curve(points, smooth_factor=1.0, n_samples=200, outlier_threshold=3.5):

    pts = remove_centroid_outliers(points, threshold=outlier_threshold)

    if len(pts) < 4:
        return pts.copy()

    # Sort by z before spline fitting
    order = np.argsort(pts[:, 2])
    pts = pts[order]

    z, x, y = pts[:, 2], pts[:, 0], pts[:, 1]

    sx = UnivariateSpline(z, x, s=smooth_factor * len(z))
    sy = UnivariateSpline(z, y, s=smooth_factor * len(z))

    z_new = np.linspace(z.min(), z.max(), n_samples)
    x_new = sx(z_new)
    y_new = sy(z_new)

    return np.column_stack((x_new, y_new, z_new))


def polyline_length(points):

    if len(points) < 2:
        return 0.0
    diffs = np.diff(points, axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=1)))


def compute_skeleton_length(path,
                            number_of_slices=NUMBER_OF_SLICES,
                            mad_threshold=MAD_THRESHOLD,
                            smooth_factor=SMOOTH_FACTOR,
                            n_spline_samples=N_SPLINE_SAMPLES):

    filename = os.path.basename(path)

    try:
        mesh = pv.read(path)
    except Exception as e:
        print(f"  [ERROR] Could not read {filename}: {e}")
        return None

    mesh = clean_main_mesh(mesh)

    if mesh.n_points == 0:
        print(f"  [WARNING] Empty mesh after cleaning: {filename}")
        return None

 
    slices_clean = get_clean_slices(mesh, number_of_slices)

    centroids = find_centroids_from_slices(slices_clean, number_of_slices)

    n_valid = int(np.sum(~np.isnan(centroids).any(axis=1)))
    if n_valid < 4:
        print(f"  [WARNING] Too few valid centroids ({n_valid}) for {filename}")
        return None

 
    smooth_pts = smooth_centroid_curve(
        centroids,
        smooth_factor=smooth_factor,
        n_samples=n_spline_samples,
        outlier_threshold=mad_threshold,
    )


    length_mm = polyline_length(smooth_pts)

    n_outliers = n_valid - len(remove_centroid_outliers(centroids, threshold=mad_threshold))

    return {
        "filename":         filename,
        "skeleton_length_mm": round(length_mm, 4),
        "valid_centroids":  n_valid,
        "removed_outliers": n_outliers,
        "spline_samples":   n_spline_samples,
    }



def main():
    print("=" * 60)
    print("Batch skeleton length computation")
    print(f"Number of slices   : {NUMBER_OF_SLICES}")
    print(f"MAD threshold      : {MAD_THRESHOLD}")
    print(f"Smooth factor      : {SMOOTH_FACTOR}")
    print(f"Spline samples     : {N_SPLINE_SAMPLES}")
    print("=" * 60)

    results = []

    for path in SPIKE_PATHS:
        filename = os.path.basename(path)
        print(f"\nProcessing: {filename}")

        if not os.path.isfile(path):
            print(f"  [SKIP] File not found: {path}")
            results.append({
                "filename":           filename,
                "skeleton_length_mm": "FILE_NOT_FOUND",
                "valid_centroids":    "",
                "removed_outliers":   "",
                "spline_samples":     "",
            })
            continue

        result = compute_skeleton_length(path)

        if result is None:
            results.append({
                "filename":           filename,
                "skeleton_length_mm": "ERROR",
                "valid_centroids":    "",
                "removed_outliers":   "",
                "spline_samples":     "",
            })
        else:
            results.append(result)
            print(f"  Skeleton length    : {result['skeleton_length_mm']} mm")
            print(f"  Valid centroids    : {result['valid_centroids']}")
            print(f"  Removed outliers   : {result['removed_outliers']}")

   
    fieldnames = ["filename", "skeleton_length_mm",
                  "valid_centroids", "removed_outliers", "spline_samples"]

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print("\n" + "=" * 60)
    print(f"Results saved to: {OUTPUT_CSV}")
    print("=" * 60)

    # Summary table
    print(f"\n{'Filename':<35} {'Skeleton length (mm)':>20}")
    print("-" * 57)
    for r in results:
        print(f"{r['filename']:<35} {str(r['skeleton_length_mm']):>20}")


if __name__ == "__main__":
    main()
