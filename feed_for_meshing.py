# -*- coding: utf-8 -*-
import os
os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"   # RTX 30xx → 8.6 / RTX 40xx → 8.9

import sys
import io
import builtins

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Force PyTorch JIT to use utf-8 instead of cp949 when calling built-in open()
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
from PIL import Image

PLY_PATH   = Path(r"C:\Users\Wallfacer\boxer\from_kiri\output\masked_gaussian\obj_clean.ply")
OUTPUT_DIR = Path(r"C:\Users\Wallfacer\boxer\from_kiri\output\feed_for_mesh")

SH_C0        = 0.28209479177387814
IMG_W        = 320
IMG_H        = 320
FOV_H_DEG    = 30.0       # FOV based on Zero123++ training
RADIUS_SCALE = 4.5        # Multiplier for bounding radius → camera distance


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x.astype(np.float64)))


def load_ply(path):
    v = PlyData.read(str(path))["vertex"]
    return {p.name: np.array(v[p.name], dtype=np.float32) for p in v.properties}


def normalize(v):
    return v / (np.linalg.norm(v) + 1e-8)


def lookat_opencv(eye, center, world_up=np.array([0., 1., 0.])):
    """world-to-camera 4x4, OpenCV convention (X-right, Y-down, Z-forward)"""
    fwd = normalize(center - eye)
    # Use alternative up vector when fwd is parallel to world_up (e.g., top view)
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
    """
    Zero123++ canonical camera positions → Transform to Kiri Y-down coordinate system
    Z-up canonical: (cx, cy, cz) → Kiri: (cx, -cz, cy)
    """
    azimuths   = np.array([30,  90, 150, 210, 270, 330], dtype=float)
    elevations = np.array([20, -10,  20, -10,  20, -10], dtype=float)

    az = np.radians(azimuths)
    el = np.radians(elevations)

    # Unit sphere positions (Z-up)
    cx = np.cos(el) * np.cos(az)
    cy = np.cos(el) * np.sin(az)
    cz = np.sin(el)

    # Kiri Y-down transformation: Z_canonical(up) → -Y_kiri
    kx, ky, kz = cx, -cz, cy

    fov_rad = np.radians(FOV_H_DEG)
    fx = IMG_W / (2.0 * np.tan(fov_rad / 2.0))
    fy = fx

    viewmats, cam_params = [], []

    for i in range(6):
        eye = center + radius * np.array([kx[i], ky[i], kz[i]])
        viewmat = lookat_opencv(eye.astype(np.float64), center.astype(np.float64))
        viewmats.append(viewmat)

        cam_params.append({
            "view_id":      i + 1,
            "azimuth_deg":  float(azimuths[i]),
            "elevation_deg": float(elevations[i]),
            "eye":          eye.tolist(),
            "center":       center.tolist(),
            "extrinsic":    viewmat.tolist(),
            "intrinsic": {
                "fx": fx, "fy": fy,
                "cx": IMG_W / 2.0, "cy": IMG_H / 2.0,
                "width": IMG_W,    "height": IMG_H,
            },
        })

    return np.stack(viewmats, axis=0), cam_params   # (6,4,4), list[dict]


def render_views(data):
    from gsplat import rasterization

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"장치: {device}")

    # ── Prepare Gaussian Tensors ──────────────────────────────────────────
    means     = torch.tensor(np.stack([data["x"], data["y"], data["z"]], -1),  device=device)
    scales    = torch.tensor(np.exp(np.stack([data["scale_0"], data["scale_1"], data["scale_2"]], -1)), device=device)
    quats     = torch.tensor(np.stack([data["rot_0"], data["rot_1"], data["rot_2"], data["rot_3"]], -1), device=device)
    quats     = quats / (quats.norm(dim=-1, keepdim=True) + 1e-8)
    opacities = torch.tensor(sigmoid(data["opacity"]).astype(np.float32), device=device)

    r = np.clip(data["f_dc_0"] * SH_C0 + 0.5, 0.0, 1.0)
    g = np.clip(data["f_dc_1"] * SH_C0 + 0.5, 0.0, 1.0)
    b = np.clip(data["f_dc_2"] * SH_C0 + 0.5, 0.0, 1.0)
    colors = torch.tensor(np.stack([r, g, b], -1), device=device)

    # ── Calculate Scene Center and Radius ─────────────────────────────────
    center = means.mean(dim=0).cpu().numpy()
    dists  = np.linalg.norm(means.cpu().numpy() - center, axis=1)
    radius = float(np.percentile(dists, 90)) * RADIUS_SCALE
    print(f"Center: {center.round(3)}  Radius: {radius:.3f}")

    # ── Generate Cameras ──────────────────────────────────────────────────
    viewmats_np, cam_params = make_cameras(center, radius)
    C = len(cam_params)
    viewmats = torch.tensor(viewmats_np, dtype=torch.float32, device=device)   # (6,4,4)

    fov_rad = np.radians(FOV_H_DEG)
    fx = IMG_W / (2.0 * np.tan(fov_rad / 2.0))
    K_single = torch.tensor([
        [fx,  0.0, IMG_W / 2.0],
        [0.0, fx,  IMG_H / 2.0],
        [0.0, 0.0, 1.0],
    ], dtype=torch.float32, device=device)
    Ks = K_single.unsqueeze(0).expand(C, -1, -1)                               # (9,3,3)

    backgrounds = torch.ones(C, 3, device=device)                              # White background

    # ── Rendering ──────────────────────────────────────────────────────────
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

    imgs = (renders.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)   # (6,H,W,3)

    # ── Initialize Real-ESRGAN Upscaler ───────────────────────────────────
    print("Loading Real-ESRGAN...")
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer
    esrgan_model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                           num_block=23, num_grow_ch=32, scale=4)
    upsampler = RealESRGANer(
        scale=4,
        model_path="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        model=esrgan_model,
        tile=0, tile_pad=10, pre_pad=0, half=True,
    )

    # ── Save Results ──────────────────────────────────────────────────────
    for i, (img, param) in enumerate(zip(imgs, cam_params)):
        img_path  = OUTPUT_DIR / f"view_{i+1:03d}.png"
        json_path = OUTPUT_DIR / f"view_{i+1:03d}_cam.json"

        # Convert RGB to BGR → Upscale → Convert back to RGB
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        enhanced_bgr, _ = upsampler.enhance(img_bgr, outscale=4)
        enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
        Image.fromarray(enhanced_rgb).save(str(img_path))
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(param, f, indent=2, ensure_ascii=False)

        print(f"  [{i+1}/6] {img_path.name}  elev={param['elevation_deg']}° azim={param['azimuth_deg']}°")

    print(f"\nCompleted → {OUTPUT_DIR}")


def main():
    print(f"Loading: {PLY_PATH}")
    data = load_ply(PLY_PATH)
    print(f"Gaussians: {len(data['x']):,}개")
    render_views(data)


if __name__ == "__main__":
    main()
