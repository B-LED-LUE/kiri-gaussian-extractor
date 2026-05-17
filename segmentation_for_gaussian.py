# -*- coding: utf-8 -*-
import os
os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"   # RTX 30xx -> 8.6 / RTX 40xx -> 8.9

import sys
import io
import builtins
import argparse

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

_orig_open = builtins.open
def _open_utf8(file, mode="r", buffering=-1, encoding=None, **kw):
    if "b" not in str(mode) and encoding is None:
        encoding = "utf-8"
    return _orig_open(file, mode, buffering, encoding=encoding, **kw)
builtins.open = _open_utf8

import numpy as np
import json
import torch
import cv2
from pathlib import Path
from plyfile import PlyData

SH_C0        = 0.28209479177387814
IMG_W        = 1280
IMG_H        = 720
FOV_H_DEG    = 60.0
RADIUS_SCALE = 1.5


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x.astype(np.float64)))


def load_ply(path):
    v = PlyData.read(str(path))["vertex"]
    return {p.name: np.array(v[p.name], dtype=np.float32) for p in v.properties}


def normalize(v):
    return v / (np.linalg.norm(v) + 1e-8)


def lookat_opencv(eye, center, world_up=np.array([0., 1., 0.])):
    """world-to-camera 4x4 matrix, OpenCV convention (X right, Y down, Z forward)"""
    fwd = normalize(center - eye)
    if abs(np.dot(fwd, world_up)) > 0.99:
        world_up = np.array([1., 0., 0.])
    right = normalize(np.cross(fwd, world_up))
    down  = -np.cross(right, fwd)

    R = np.stack([right, down, fwd], axis=0).astype(np.float32)
    t = (-R @ eye).astype(np.float32)

    mat = np.eye(4, dtype=np.float32)
    mat[:3, :3] = R
    mat[:3, 3]  = t
    return mat


def make_cameras(center, radius):
    """6 cameras: 1 top-down + 5 pentagon ring"""
    yaws    = [0,    0,  72, 144, 216, 288]
    pitches = [-85, -30, -30, -30, -30, -30]

    fov_rad = np.radians(FOV_H_DEG)
    fx = IMG_W / (2.0 * np.tan(fov_rad / 2.0))
    fy = fx

    viewmats, cam_params = [], []

    for i, (yaw_deg, pitch_deg) in enumerate(zip(yaws, pitches)):
        yaw   = np.radians(yaw_deg)
        pitch = np.radians(pitch_deg)

        eye = center + radius * np.array([
            np.cos(pitch) * np.sin(yaw),
            np.sin(pitch),
            np.cos(pitch) * np.cos(yaw),
        ])

        viewmat = lookat_opencv(eye.astype(np.float64), center.astype(np.float64))
        viewmats.append(viewmat)

        cam_params.append({
            "view_id":   i + 1,
            "yaw_deg":   float(yaw_deg),
            "pitch_deg": float(pitch_deg),
            "eye":       eye.tolist(),
            "center":    center.tolist(),
            "extrinsic": viewmat.tolist(),
            "intrinsic": {
                "fx": fx, "fy": fy,
                "cx": IMG_W / 2.0, "cy": IMG_H / 2.0,
                "width": IMG_W,    "height": IMG_H,
            },
        })

    return np.stack(viewmats, axis=0), cam_params


def render_views(data, output_dir):
    from gsplat import rasterization

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    means     = torch.tensor(np.stack([data["x"], data["y"], data["z"]], -1),  device=device)
    scales    = torch.tensor(np.exp(np.stack([data["scale_0"], data["scale_1"], data["scale_2"]], -1)), device=device)
    quats     = torch.tensor(np.stack([data["rot_0"], data["rot_1"], data["rot_2"], data["rot_3"]], -1), device=device)
    quats     = quats / (quats.norm(dim=-1, keepdim=True) + 1e-8)
    opacities = torch.tensor(sigmoid(data["opacity"]).astype(np.float32), device=device)

    r = np.clip(data["f_dc_0"] * SH_C0 + 0.5, 0.0, 1.0)
    g = np.clip(data["f_dc_1"] * SH_C0 + 0.5, 0.0, 1.0)
    b = np.clip(data["f_dc_2"] * SH_C0 + 0.5, 0.0, 1.0)
    colors = torch.tensor(np.stack([r, g, b], -1), device=device)

    center = means.mean(dim=0).cpu().numpy()
    dists  = np.linalg.norm(means.cpu().numpy() - center, axis=1)
    radius = float(np.percentile(dists, 90)) * RADIUS_SCALE
    print(f"Center: {center.round(3)}  Radius: {radius:.3f}")

    viewmats_np, cam_params = make_cameras(center, radius)
    C = len(cam_params)
    viewmats = torch.tensor(viewmats_np, dtype=torch.float32, device=device)

    fov_rad = np.radians(FOV_H_DEG)
    fx = IMG_W / (2.0 * np.tan(fov_rad / 2.0))
    K_single = torch.tensor([
        [fx,  0.0, IMG_W / 2.0],
        [0.0, fx,  IMG_H / 2.0],
        [0.0, 0.0, 1.0],
    ], dtype=torch.float32, device=device)
    Ks = K_single.unsqueeze(0).expand(C, -1, -1)

    backgrounds = torch.ones(C, 3, device=device)

    print("Rendering...")
    renders, alphas, _ = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmats,
        Ks=Ks,
        width=IMG_W,
        height=IMG_H,
        backgrounds=backgrounds,
    )

    imgs = (renders.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)

    output_dir.mkdir(parents=True, exist_ok=True)
    for i, (img, param) in enumerate(zip(imgs, cam_params)):
        img_path  = output_dir / f"view_{i+1:03d}.png"
        json_path = output_dir / f"view_{i+1:03d}_cam.json"

        cv2.imwrite(str(img_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(param, f, indent=2, ensure_ascii=False)

        print(f"  [{i+1}/6] {img_path.name}  pitch={param['pitch_deg']}deg  yaw={param['yaw_deg']}deg")

    print(f"\nDone -> {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Render 6 views from a 3DGS PLY for SAM segmentation")
    parser.add_argument("--ply",    type=Path, required=True, help="Input 3DGS PLY file")
    parser.add_argument("--output", type=Path, required=True, help="Directory to save rendered views and camera JSONs")
    args = parser.parse_args()

    print(f"Loading: {args.ply}")
    data = load_ply(args.ply)
    print(f"Gaussians: {len(data['x']):,}")
    render_views(data, args.output)


if __name__ == "__main__":
    main()
