import numpy as np
import open3d as o3d
from numpy.linalg import eig
import math
import os
from numpy.linalg import matrix_power

def pcaEign(pcd, magnitude):
    covariance = np.cov([pcd[:, 0], pcd[:, 1], pcd[:, 2]])
    D, V  = eig(covariance)
    index=np.argsort(D)
    V = V[:, index]
    if magnitude=='max':
        return  V[:, 2]
    elif magnitude=='middle':
        return  V[:, 1]
    elif magnitude=='min':
        return  V[:, 0]

def unitVectorToAngle(u):
    alpha = math.atan2(u[1], u[0])
    beta = math.atan2(np.sqrt(u[0]**2+ u[1]**2),u[2])
    return alpha, beta

def rotationalMatrix(alpha, beta):
    Rx = np.array([[1, 0 ,0],[0 ,np.cos(beta), -np.sin(beta)],[ 0 ,np.sin(beta), np.cos(beta)]])
    Ry = np.array([[np.cos(beta), 0, np.sin(beta)], [0, 1, 0], [-np.sin(beta), 0 ,np.cos(beta)]])
    Rz = np.array([[np.cos(alpha), -np.sin(alpha), 0], [np.sin(alpha), np.cos(alpha), 0], [0, 0 ,1]])
    return Rx,Ry,Rz

def rotatePC(pcd, Ry, Rz):
    matrix = pcd.transpose()
    matrix2 = np.matmul(Rz,matrix)
    matrix2 = np.matmul(Ry,matrix2)
    return matrix2.transpose()

def rotatePointCloudAlongZ(pcd, direction):
    pcd = pcd - pcd.mean(0)
    u = pcaEign(pcd, 'max')
    alpha, beta = unitVectorToAngle(u)
    Rx, Ry, Rz = rotationalMatrix(-alpha, math.pi-beta)
    pcd2 = rotatePC(pcd, Ry, Rz)
    if direction== 'x':
        offset = 0
    elif direction== 'y':
        offset = math.pi/2
    v = pcaEign(pcd2, 'middle')
    alpha, beta= unitVectorToAngle(v)
    Rx, Ry, Rz = rotationalMatrix(offset - alpha, 0)
    pcd2 = rotatePC(pcd2, Ry, Rz)
    return  pcd2,Rx, Ry, Rz,alpha, beta

def align_single_mesh(  input_file: str, output_file: str, direction: str = "y", translate_to=(0.0, 0.0, 0.0), show: bool = False,) -> None:

    if not os.path.isfile(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    print(f"Reading mesh: {input_file}")
    mesh_in = o3d.io.read_triangle_mesh(input_file)
    if mesh_in.is_empty():
        raise ValueError(f"Mesh is empty: {input_file}")
    vertices_in = np.asarray(mesh_in.vertices)
    triangles = np.asarray(mesh_in.triangles)
    uvs = np.asarray(mesh_in.triangle_uvs)
    pcx, Rx, Ry, Rz, alpha, beta = rotatePointCloudAlongZ(vertices_in, direction)
    vertices_out = np.asarray(pcx, dtype=float) + np.array(translate_to, dtype=float)
    mesh_out = o3d.geometry.TriangleMesh()
    mesh_out.vertices = o3d.utility.Vector3dVector(vertices_out)
    mesh_out.triangles = o3d.utility.Vector3iVector(triangles)
    if uvs is not None and uvs.size > 0:
        mesh_out.triangle_uvs = o3d.utility.Vector2dVector(uvs)
    if show:
        print("Showing original (yellow) and aligned (blue) together...")
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=50.0)
        mesh_in_vis = o3d.geometry.TriangleMesh()
        mesh_in_vis.vertices = o3d.utility.Vector3dVector(vertices_in)
        mesh_in_vis.triangles = o3d.utility.Vector3iVector(triangles)
        mesh_in_vis.paint_uniform_color([1.0, 0.706, 0.0])
        mesh_out_vis = o3d.geometry.TriangleMesh()
        mesh_out_vis.vertices = o3d.utility.Vector3dVector(vertices_out)
        mesh_out_vis.triangles = o3d.utility.Vector3iVector(triangles)
        mesh_out_vis.paint_uniform_color([0.0, 0.651, 0.929])
        o3d.visualization.draw_geometries(
            [mesh_in_vis, mesh_out_vis, coord_frame],
            window_name="Before (yellow) & After (blue) — Blue=PCA1, Green=PCA2, Red=PCA3",
            width=1200,
            height=800,
        )  
    o3d.io.write_triangle_mesh(output_file, mesh_out, write_triangle_uvs=True)
    print(f"Aligned mesh saved to: {output_file}")