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