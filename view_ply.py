import numpy as np
import open3d as o3d
from pathlib import Path
from plyfile import PlyData

PLY_PATH = Path(r"C:\Users\Wallfacer\boxer\from_kiri\output\masked_gaussian\obj.ply")

SH_C0 = 0.28209479177387814


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x.astype(np.float64)))


def load_and_view(path):
    v = PlyData.read(str(path))["vertex"]
    data = {p.name: np.array(v[p.name], dtype=np.float32) for p in v.properties}

    xyz = np.stack([data["x"], data["y"], data["z"]], axis=-1)
    r = np.clip(data["f_dc_0"] * SH_C0 + 0.5, 0.0, 1.0)
    g = np.clip(data["f_dc_1"] * SH_C0 + 0.5, 0.0, 1.0)
    b = np.clip(data["f_dc_2"] * SH_C0 + 0.5, 0.0, 1.0)
    rgb = np.stack([r, g, b], axis=-1)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)

    print(f"Loaded: {path.name} ({len(xyz):,})")
    o3d.visualization.draw_geometries([pcd], window_name=path.name)


if __name__ == "__main__":
    load_and_view(PLY_PATH)
