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


def save_ply_points(path, points_xyz: np.ndarray):
    path = str(path)
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {points_xyz.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("end_header\n")
        for p in points_xyz:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")



def load_mesh_vertices_faces(mesh_path):
    if trimesh is not None:
        try:
            tm = trimesh.load(str(mesh_path), force="mesh")
            if isinstance(tm, trimesh.Scene):
                tm = trimesh.util.concatenate(tuple(m for m in tm.dump()))
            if isinstance(tm, trimesh.Trimesh):
                return tm.vertices.copy(), tm.faces.copy(), "trimesh", tm
        except Exception:
            pass

    if open3d is not None:
        o3d = open3d
        m = o3d.io.read_triangle_mesh(str(mesh_path))
        m.remove_duplicated_vertices()
        m.remove_degenerate_triangles()
        m.remove_duplicated_triangles()
        m.remove_non_manifold_edges()
        return np.asarray(m.vertices), np.asarray(m.triangles), "open3d", None

    raise RuntimeError("Could not load mesh. Please install trimesh and/or open3d.")


def voxelize(vertices, faces, prefer_tm, tm_obj, target_res=200):
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    bbox_size = bbox_max - bbox_min
    bbox_diag = float(np.linalg.norm(bbox_size))
    longest = float(np.max(bbox_size))
    pitch = longest / float(target_res) if longest > 0 else 1.0
    if trimesh is not None and prefer_tm and tm_obj is not None:
        try:
            vg = tm_obj.voxelized(pitch=pitch).fill()
            volume = vg.matrix.astype(bool)  # (Z,Y,X)
            dense_to_world = vg.transform # 4x4
            return volume, dense_to_world, bbox_diag
        except Exception:
            pass
    if trimesh is None:
        raise RuntimeError("Fallback voxelizer requires trimesh. Please install trimesh.")
    tm2 = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    nx = ny = nz = max(32, int(round(target_res)))
    xs = np.linspace(bbox_min[0], bbox_max[0], nx, endpoint=False) + (bbox_size[0] / nx) * 0.5
    ys = np.linspace(bbox_min[1], bbox_max[1], ny, endpoint=False) + (bbox_size[1] / ny) * 0.5
    zs = np.linspace(bbox_min[2], bbox_max[2], nz, endpoint=False) + (bbox_size[2] / nz) * 0.5
    grid = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).reshape(-1, 3)
    contains = tm2.contains(grid).reshape(nx, ny, nz)
    volume = contains.transpose(2, 1, 0)  # (Z,Y,X)
    dense_to_world = np.eye(4)
    voxel = np.array([xs[1]-xs[0], ys[1]-ys[0], zs[1]-zs[0]])
    dense_to_world[:3, :3] = np.diag([voxel[0], voxel[1], voxel[2]])
    dense_to_world[:3, 3]  = np.array([xs[0], ys[0], zs[0]])
    return volume, dense_to_world, bbox_diag

NEIGHBORS_26 = [(dz,dy,dx) for dz in (-1,0,1) for dy in (-1,0,1) for dx in (-1,0,1) if not (dz==0 and dy==0 and dx==0)]

def skeletonize_voxels(volume_bool):
    return skeletonize_3d(volume_bool.astype(np.uint8)).astype(bool)

def build_graph_from_skeleton(skel_bool):
    Z, Y, X = skel_bool.shape
    idxs = np.argwhere(skel_bool)      # (N,3) [z,y,x]
    idx_to_id = {tuple(p): i for i, p in enumerate(map(tuple, idxs))}
    adj = [[] for _ in range(len(idxs))]
    for i, (z,y,x) in enumerate(idxs):
        for dz,dy,dx in NEIGHBORS_26:
            nz, ny, nx = z+dz, y+dy, x+dx
            if 0 <= nz < Z and 0 <= ny < Y and 0 <= nx < X and skel_bool[nz,ny,nx]:
                j = idx_to_id.get((nz,ny,nx))
                if j is not None:
                    adj[i].append(j)
    deg = np.array([len(set(neis)) for neis in adj], dtype=int)
    return idxs, adj, deg

def collapse_segments(idxs, adj, deg):
    is_endpoint = (deg == 1)
    is_junction = (deg >= 3)
    key_nodes = np.where(is_endpoint | is_junction)[0]

    visited = np.zeros(len(idxs), dtype=bool)
    segments = []

    for start in key_nodes:
        for nb in set(adj[start]):
            if visited[start] and visited[nb]:
                continue
            if not (deg[nb] == 1 or deg[nb] >= 3):
                path = [start]
                prev = start
                cur  = nb
                while True:
                    path.append(cur)
                    visited[cur] = True
                    nexts = [k for k in set(adj[cur]) if k != prev]
                    if len(nexts) == 0:
                        end = cur; break
                    if len(nexts) > 1:
                        end = cur; break
                    nxt = nexts[0]
                    if deg[nxt] == 1 or deg[nxt] >= 3:
                        end = nxt; path.append(end); break
                    prev, cur = cur, nxt
                segments.append((start, end, path))
            else:
                segments.append((start, nb, [start, nb]))

    def seg_key(a,b): return (a,b) if a<=b else (b,a)
    unique = {}
    for a,b,path in segments:
        k = seg_key(a,b)
        if k not in unique or len(path) < len(unique[k][2]):
            unique[k] = (a,b,path)
    return list(unique.values())

def indices_to_world_points(idxs_zyx, dense_to_world, ids):
    ijk = idxs_zyx[np.array(ids)]  # (z,y,x)
    xyz1 = np.hstack([ijk[:, ::-1].astype(np.float64), np.ones((len(ids),1))])  # (x,y,z,1)
    return (xyz1 @ dense_to_world.T)[:, :3]

def segment_lengths_world(segments, idxs, dense_to_world):
    L = np.zeros(len(segments), dtype=float)
    for i, (_, _, path) in enumerate(segments):
        pts = indices_to_world_points(idxs, dense_to_world, path)
        L[i] = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))) if len(pts) >= 2 else 0.0
    return L

def prune_short_leaf_segments(segments, seg_lengths, prune_ratio, bbox_diag):
    if prune_ratio <= 0:
        return np.ones(len(segments), dtype=bool)
    min_len = prune_ratio * bbox_diag
    key_adj = defaultdict(list)
    for si, (a, b, _) in enumerate(segments):
        key_adj[a].append(si)
        key_adj[b].append(si)
    alive = np.ones(len(segments), dtype=bool)
    def degree_after(node):
        return sum(1 for si in key_adj[node] if alive[si])
    changed = True
    while changed:
        changed = False
        for si, (a, b, _) in enumerate(segments):
            if not alive[si]:
                continue
            if (degree_after(a) == 1 or degree_after(b) == 1) and seg_lengths[si] < min_len:
                alive[si] = False
                changed = True
    return alive

def build_pruned_graph_from_alive(idxs, adj_full, segments, alive_mask):
    keep = set()
    for si, (_, _, path) in enumerate(segments):
        if alive_mask[si]:
            keep.update(path)
    kept = sorted(list(keep))

    old_to_new = {old_i: new_i for new_i, old_i in enumerate(kept)}
    adj = [[] for _ in range(len(kept))]
    for old_i in kept:
        i = old_to_new[old_i]
        for old_j in set(adj_full[old_i]):
            if old_j in old_to_new:
                j = old_to_new[old_j]
                adj[i].append(j)
    deg = np.array([len(set(neis)) for neis in adj], dtype=int)
    return kept, adj, deg

def connected_components(adj):
    N = len(adj)
    visited = np.zeros(N, dtype=bool)
    comps = []
    for i in range(N):
        if not visited[i]:
            q = deque([i])
            visited[i] = True
            comp = []
            while q:
                u = q.popleft()
                comp.append(u)
                for v in set(adj[u]):
                    if not visited[v]:
                        visited[v] = True
                        q.append(v)
            comps.append(comp)
    return comps

def skeletonize_mesh(mesh_path, target_res=220, prune_ratio=0.03, out_dir=".", prefix="skeleton"):
    verts, faces, loader, tm_obj = load_mesh_vertices_faces(mesh_path)
    volume, dense_to_world, bbox_diag = voxelize(verts, faces, loader == "trimesh", tm_obj, target_res=target_res)

    skel = skeletonize_voxels(volume)
    idxs, adj_full, deg_full = build_graph_from_skeleton(skel)

    segments = collapse_segments(idxs, adj_full, deg_full)
    seg_lengths = segment_lengths_world(segments, idxs, dense_to_world)

    alive = prune_short_leaf_segments(segments, seg_lengths, prune_ratio, bbox_diag)
    kept_voxels, adj_pruned, deg_pruned = build_pruned_graph_from_alive(idxs, adj_full, segments, alive)

    # Components & branching breakdown
    components = connected_components(adj_pruned)
    sizes = [len(c) for c in components]
    lcc_idx = int(np.argmax(sizes)) if sizes else -1

    branching_counts = [int(np.sum(deg_pruned[np.array(comp, dtype=int)] >= 3)) for comp in components]
    endpoint_counts  = [int(np.sum(deg_pruned[np.array(comp, dtype=int)] == 1)) for comp in components]
    isolated_counts  = [int(np.sum(deg_pruned[np.array(comp, dtype=int)] == 0)) for comp in components]

    connected_branching   = branching_counts[lcc_idx] if lcc_idx >= 0 else 0
    unconnected_branching = int(sum(branching_counts) - connected_branching)

    # Export skeleton points (.ply only)
    pts_world = indices_to_world_points(idxs, dense_to_world, kept_voxels)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    ply_path = out_dir / f"{prefix}_points.ply"
    save_ply_points(ply_path, pts_world)

    # Per-component rows (for the appended CSV)
    per_comp_rows = []
    for comp_idx, comp in enumerate(components):
        per_comp_rows.append({
            "ear_id": Path(mesh_path).stem,
            "component_index": comp_idx,
            "size_voxels": len(comp),
            "branching_deg>=3": branching_counts[comp_idx],
            "endpoints_deg==1": endpoint_counts[comp_idx],
            "isolated_deg==0": isolated_counts[comp_idx],
            "is_lcc": int(comp_idx == lcc_idx)
        })
    return {
        "ear_id": Path(mesh_path).stem,
        "paths": {"points_ply": str(ply_path)},
        "branching": {
            "total_branching_points": int(sum(branching_counts)),
            "total_endpoints": int(sum(endpoint_counts)),
            "total_isolated_points": int(sum(isolated_counts)),
            "num_components": int(len(components)),
            "connected_branching_points": int(connected_branching),
            "unconnected_branching_points": int(unconnected_branching),
        },
        "per_component_rows": per_comp_rows
    }

