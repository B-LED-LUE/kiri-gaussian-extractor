"""Multi-view voting: project Gaussians into SAM2 masks from multiple views.
Gaussians that appear inside the mask in >= min_votes views are classified as object.
"""
import argparse
import numpy as np
import json
import cv2
from pathlib import Path
from plyfile import PlyData, PlyElement

parser = argparse.ArgumentParser(description="Multi-view voting to extract object Gaussians")
parser.add_argument("--ply",       type=Path, required=True, help="Input 3DGS PLY file")
parser.add_argument("--mask_dir",  type=Path, required=True, help="Directory with mask_*.png files")
parser.add_argument("--cam_dir",   type=Path, required=True, help="Directory with view_*_cam.json files")
parser.add_argument("--out_dir",   type=Path, required=True, help="Output directory")
parser.add_argument("--min_votes", type=int,  default=3,     help="Min views to classify as object")
args = parser.parse_args()


def load_ply(path):
    v = PlyData.read(str(path))["vertex"]
    props = v.properties
    data  = {p.name: np.array(v[p.name]) for p in props}
    return data, props


def save_ply(data, props, mask, path):
    dtypes = [(p.name, p.val_dtype) for p in props]
    arr = np.zeros(int(mask.sum()), dtype=dtypes)
    for p in props:
        arr[p.name] = data[p.name][mask]
    PlyData([PlyElement.describe(arr, "vertex")], text=False).write(str(path))


def project_to_image(xyz_world, extrinsic, intrinsic):
    N = len(xyz_world)
    ones = np.ones((N, 1), dtype=np.float64)
    xyz_h = np.concatenate([xyz_world, ones], axis=1)

    E = np.array(extrinsic, dtype=np.float64)
    cam = (E @ xyz_h.T).T
    xc, yc, zc = cam[:, 0], cam[:, 1], cam[:, 2]

    valid = zc > 0

    fx = intrinsic["fx"]; fy = intrinsic["fy"]
    cx = intrinsic["cx"]; cy = intrinsic["cy"]
    W  = intrinsic["width"]; H = intrinsic["height"]

    px = np.where(valid, fx * xc / (zc + 1e-8) + cx, -1).astype(np.int32)
    py = np.where(valid, fy * yc / (zc + 1e-8) + cy, -1).astype(np.int32)

    in_frame = valid & (px >= 0) & (px < W) & (py >= 0) & (py < H)
    return px, py, in_frame


def main():
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading PLY: {args.ply}")
    data, props = load_ply(args.ply)
    N = len(data["x"])
    print(f"  Gaussians: {N:,}")

    xyz = np.stack([data["x"], data["y"], data["z"]], axis=-1).astype(np.float64)

    mask_paths = sorted(args.mask_dir.glob("mask_*.png"))
    cam_paths  = sorted(args.cam_dir.glob("view_*_cam.json"))

    if not mask_paths:
        print(f"No masks found in {args.mask_dir}"); return
    if not cam_paths:
        print(f"No camera JSONs found in {args.cam_dir}"); return

    pairs = list(zip(mask_paths, cam_paths))
    print(f"  {len(pairs)} mask-camera pairs")

    votes = np.zeros(N, dtype=np.int32)

    for mask_path, cam_path in pairs:
        mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask_img is None:
            print(f"  Failed to read: {mask_path.name}"); continue

        with open(cam_path, "r", encoding="utf-8") as f:
            cam = json.load(f)

        W_cam = int(cam["intrinsic"]["width"])
        H_cam = int(cam["intrinsic"]["height"])
        if mask_img.shape != (H_cam, W_cam):
            mask_img = cv2.resize(mask_img, (W_cam, H_cam), interpolation=cv2.INTER_NEAREST)

        mask_bool = mask_img > 127

        px, py, valid = project_to_image(xyz, cam["extrinsic"], cam["intrinsic"])

        hit = valid & mask_bool[py.clip(0, H_cam - 1), px.clip(0, W_cam - 1)]
        hit[~valid] = False
        votes += hit.astype(np.int32)

        print(f"  {mask_path.name}  valid={valid.sum():,}  hits={hit.sum():,}")

    obj_mask    = votes >= args.min_votes
    no_obj_mask = ~obj_mask

    print(f"\nVoting results (threshold: {args.min_votes} views):")
    print(f"  Object    : {obj_mask.sum():,}")
    print(f"  Background: {no_obj_mask.sum():,}")

    obj_ply    = args.out_dir / "obj.ply"
    no_obj_ply = args.out_dir / "without_obj.ply"
    save_ply(data, props, obj_mask,    obj_ply)
    save_ply(data, props, no_obj_mask, no_obj_ply)

    print(f"\nSaved:")
    print(f"  {obj_ply}")
    print(f"  {no_obj_ply}")


if __name__ == "__main__":
    main()
