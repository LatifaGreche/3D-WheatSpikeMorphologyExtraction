import numpy as np
import pyvista as pv
from tqdm import tqdm
from filesNumericalSort import numericalSort
import os
import triangle as tr
from vedo import *
from findSlice import *
from shapely.geometry import MultiPolygon, JOIN_STYLE
import matplotlib.pyplot as plt
import xlsxwriter
import pandas as pd
from vedo import Mesh
import meshlib.mrmeshpy as mr
from scipy.optimize import curve_fit 
from scipy.ndimage import gaussian_filter1d
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
import random
from scipy.spatial import distance
from mpl_toolkits import mplot3d
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import splrep, BSpline
from sklearn.metrics import auc
from scipy.optimize import curve_fit 
from io import StringIO
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from scipy import interpolate

def triangulate_slice(slc):
    seg = slc.lines.reshape(-1, 3)[:, 1:]
    A = dict(vertices=np.array(slc.points[:, :2], dtype=np.float64), segments=seg)
    B = tr.triangulate(A, 'qp')
    n_faces = B['triangles'].shape[0]
    triangles = np.hstack((np.ones((n_faces, 1), np.int32)*3, B['triangles']))
    # back to 3D
    pts = np.empty((B['vertices'].shape[0], 3))
    pts[:, :2] = B['vertices']
    pts[:, -1] = slc.points[0, 2]
    pd = pv.PolyData(pts, triangles, n_faces)
    return pd
def triangulates(slc):
    seg = slc.lines.reshape(-1, 3)[:, 1:]
    A = dict(vertices=np.array(slc.points[:, :2], dtype=np.float64), segments=seg)
    B = tr.triangulate(A, 'qp')
    return B
def centeroidnp(arr):
    centroid_z=arr[0,2]
    length = arr.shape[0]
    sum_x = np.sum(arr[:, 0])
    sum_y = np.sum(arr[:, 1])
    return sum_x/length, sum_y/length,centroid_z

def mean_arr(arr):
    centroid_z=arr[0,2]
    sum_x = np.max(arr[:, 0])+np.min(arr[:, 0])
    sum_y = np.max(arr[:, 1])+np.min(arr[:, 1])
    return sum_x/2, sum_y/2,centroid_z

def find_meam(mesh, NumberOfSlices):
    slicess=mesh.slice_along_axis(n=NumberOfSlices, axis='z', tolerance=None, generate_triangles=False, contour=False, bounds=None, center=None, progress_bar=False)
    tslices = pv.MultiBlock([triangulate_slice(slc) for slc in slicess])
    areas = np.array([tslice.area for tslice in tslices])
    Points=np.zeros((NumberOfSlices,3))
    cmpp=0
    for SLICe in slicess:
        centroid_x, centroid_y, centroid_z = mean_arr(SLICe.points)
        Points[cmpp]=[centroid_x,centroid_y,centroid_z]
        cmpp+=1
    return Points
def find_centroid(mesh, NumberOfSlices):
    slicess=mesh.slice_along_axis(n=NumberOfSlices, axis='z', tolerance=None, generate_triangles=False, contour=False, bounds=None, center=None, progress_bar=False)
    tslices = pv.MultiBlock([triangulate_slice(slc) for slc in slicess])
    areas = np.array([tslice.area for tslice in tslices])
    Points=np.zeros((NumberOfSlices,3))
    cmpp=0
    for SLICe in slicess:
        centroid_x, centroid_y, centroid_z = centeroidnp(SLICe.points)
        Points[cmpp]=[centroid_x,centroid_y,centroid_z]
        cmpp+=1
    return Points

def shape_info(path):
    mesh = pv.read(path)
    z_lenth=int(mesh.bounds[5]-mesh.bounds[4])
    slicess=mesh.slice_along_axis(n=100, axis='z', tolerance=None, generate_triangles=False, contour=False, bounds=None, center=None, progress_bar=False)
    tslices = pv.MultiBlock([triangulate_slice(slc) for slc in slicess])
    areas = np.array([tslice.area for tslice in tslices])
    centroids=find_centroid(mesh, 100)
    means=find_meam(mesh, 100)
    msh = Mesh(path)
    vol=msh.volume()
    return areas,means,centroids,vol

def is_float(element: any) -> bool:
    if element is None: 
        return False
    try:
        float(element)
        return True
    except ValueError:
        return False
def func(x, a, x0, sigma): 
    return a*np.exp(-(x-x0)**2/(2*sigma**2)) 

def inflection_points(X_t,fit_y):
    smooth = gaussian_filter1d(fit_y, 1000)
    smooth_d2 = np.gradient(np.gradient(smooth))
    infls = np.where(np.diff(np.sign(smooth_d2)))[0]
    p_inflection=[]
    for i, infl in enumerate(infls, 1):
        p_inflection.append([X_t[infl], fit_y[infl]])
        ax.scatter(X_t[infl], fit_y[infl], color='b' )
    return p_inflection

def RemoveApicalAwns(spikepath, top_awns):
    mesh = pv.read(spikepath)
    clipped = mesh.clip(normal=[0, 0, 1], origin=top_awns, invert=True)
    # pv.save_meshio("some_path.ply", clipped)

    return clipped

_FLOAT_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

def parse_area_str(area_str) -> np.ndarray:
    """underscore-separated floats -> np.array(float)"""
    if pd.isna(area_str):
        return np.array([], dtype=float)
    vals = []
    for t in str(area_str).split("_"):
        try:
            vals.append(float(t))
        except ValueError:
            pass
    return np.array(vals, dtype=float)

def parse_centroid_str(centroid_str):
    if pd.isna(centroid_str):
        return np.empty((0, 3), dtype=float), []

    rows = []
    z_corr = []
    for chunk in str(centroid_str).split("_"):
        nums = _FLOAT_RE.findall(chunk)
        if len(nums) < 3:
            continue
        x, y, z = map(float, nums[:3])
        rows.append((x, y, z))
        z_corr.append(z)

    if not rows:
        return np.empty((0, 3), dtype=float), []

    return np.array(rows, dtype=float), z_corr

def pick_first_existing(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def load_spike_excel(pathFile, sheet_name=0):

    df = pd.read_excel(pathFile, sheet_name=sheet_name)

    col_geno = pick_first_existing(df, ["Genotype ID", "spikelet ID", "variety ID", "Variety", "Genotype"])
    col_ear  = pick_first_existing(df, ["Spike ID", "ear ID", "Ear ID", "Spike"])
    col_area = pick_first_existing(df, ["Area", "area", "thickness", "Thickness"])
    col_cent = pick_first_existing(df, ["Centroid", "centroid", "Centroids"])
    col_vol  = pick_first_existing(df, ["Volume (mm³)", "volume", "Volume", "volume (mm3)"])

    missing = [name for name, col in [
        ("Genotype/spikelet ID", col_geno),
        ("Spike/ear ID", col_ear),
        ("Area/area/thickness", col_area),
        ("Centroid/centroid", col_cent),
    ] if col is None]

    if missing:
        raise ValueError(f"Missing columns {missing}. Found: {list(df.columns)}")

    spike_ID  = df[col_geno].astype(str).values
    earID     = df[col_ear].astype(str).values
    area      = df[col_area].astype(str).values
    centroids = df[col_cent].astype(str).values
    vol       = df[col_vol].values if col_vol else np.full(len(df), np.nan)

    return spike_ID, earID, area, vol, centroids



def spike_length_from_skeleton(x, y, z) -> float:
    # spline through skeleton points, then sum segment lengths
    tck, _ = interpolate.splprep([x, y, z], s=5)
    u_fine = np.linspace(0, 1, 15)
    x_fine, y_fine, z_fine = interpolate.splev(u_fine, tck)

    diffs = np.sqrt(np.diff(x_fine)**2 + np.diff(y_fine)**2 + np.diff(z_fine)**2)
    return float(diffs.sum())



def process_file(pathFile: str, out_xlsx: str, sheet_name=0, verbose=True):
    spike_ID, earID, area, vol, centroids = load_spike_excel(pathFile, sheet_name=sheet_name)

    len_spike = []
    llll = []
    AUC = []

    for area_str, cent_str, geno, ear in zip(area, centroids, spike_ID, earID):
        xyz, z_corr = parse_centroid_str(cent_str)
        Y_t = parse_area_str(area_str)

        if len(z_corr) == 0 or len(Y_t) == 0:
            if verbose:
                print(f"Skipping {geno} / {ear}: empty centroid or area after parsing")
            len_spike.append(np.nan)
            llll.append(np.nan)
            AUC.append(np.nan)
            continue

        X_t = np.array(z_corr, dtype=float)
        X_t = X_t - X_t[0]  # translate

        # Align lengths safely
        n = min(len(X_t), len(Y_t))
        X_t = X_t[:n]
        Y_t = Y_t[:n]


        if len(Y_t) > 50:
            idx_candidates = np.where(Y_t[50:] < 8)[0]
            if len(idx_candidates) == 0:
                idx_cut = len(Y_t)
            else:
                idx_cut = int(idx_candidates[0] + 50)
        else:
            idx_cut = len(Y_t)

        # skeleton points up to idx_cut
        skel_cut = xyz[:idx_cut, :]
        if skel_cut.shape[0] < 4:
            # splprep needs enough points; fall back to polyline length
            diffs = np.sqrt(np.sum(np.diff(skel_cut, axis=0)**2, axis=1)) if skel_cut.shape[0] >= 2 else np.array([0.0])
            skel_len = float(diffs.sum())
        else:
            skel_len = spike_length_from_skeleton(skel_cut[:, 0], skel_cut[:, 1], skel_cut[:, 2])

        # spike z length (from 0 to last cut slice)
        z_len = float(np.linalg.norm(X_t[idx_cut - 1] - X_t[0])) if idx_cut >= 2 else 0.0

        # AUC under spline curve
        try:
            tck = splrep(X_t, Y_t, s=0)
            spline_y = BSpline(*tck)(X_t)
            auc_val = float(auc(X_t, spline_y))
        except Exception:
            auc_val = np.nan

        len_spike.append(z_len)
        llll.append(skel_len)
        AUC.append(auc_val)

    # Save ONLY what you asked for
    data = {
        "Genotype ID": spike_ID,
        "spike z length": len_spike,
        "skeleton length": llll,
        "volume": vol,
        "area under spline curve": AUC,
    }
    df_out = pd.DataFrame(data)
    os.makedirs(os.path.dirname(out_xlsx), exist_ok=True)
    df_out.to_excel(out_xlsx, index=False)

    if verbose:
        print("Saved:", out_xlsx)
        print("Rows:", len(df_out))

