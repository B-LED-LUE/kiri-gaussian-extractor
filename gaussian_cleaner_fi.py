"""Remove floor/desk Gaussians by scanning Y-slices.
Slices with high Gaussian density and low color variance = uniform floor surface -> remove.
"""
import argparse
import numpy as np
from pathlib import Path
from plyfile import PlyData, PlyElement

parser = argparse.ArgumentParser(description="Remove floor Gaussians by Y-slice color variance")
parser.add_argument("--ply",       type=Path, required=True, help="Input 3DGS PLY file")
parser.add_argument("--output",    type=Path, default=None,  help="Output PLY (default: input_clean.ply)")
parser.add_argument("--n_slices",  type=int,   default=120,  help="Number of Y-axis slices")
parser.add_argument("--min_density", type=int, default=20,   help="Min Gaussians per slice to consider")
parser.add_argument("--max_color_var", type=float, default=0.05, help="Max color variance for floor detection")
args = parser.parse_args()

SH_C0 = 0.28209479177387814


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x.astype(np.float64)))


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


def main():
    out_path = args.output if args.output else args.ply.parent / (args.ply.stem + "_clean.ply")

    print(f"Loading: {args.ply}")
    data, props = load_ply(args.ply)
    N = len(data["x"])
    print(f"  Gaussians: {N:,}\n")

    # Y-down coordinate system: higher Y value = physically lower (floor)
    y   = data["y"].astype(np.float64)
    r   = np.clip(data["f_dc_0"] * SH_C0 + 0.5, 0.0, 1.0)
    g   = np.clip(data["f_dc_1"] * SH_C0 + 0.5, 0.0, 1.0)
    b   = np.clip(data["f_dc_2"] * SH_C0 + 0.5, 0.0, 1.0)
    rgb = np.stack([r, g, b], axis=-1)

    y_min, y_max = y.min(), y.max()
    edges = np.linspace(y_min, y_max, args.n_slices + 1)

    print("Scanning Y-slices from floor upward:")
    floor_y = None

    for i in range(args.n_slices - 1, -1, -1):
        lo, hi = edges[i], edges[i + 1]
        in_slice = (y >= lo) & (y < hi)
        cnt = in_slice.sum()
        if cnt < args.min_density:
            continue

        color_var = rgb[in_slice].var(axis=0).mean()
        print(f"  slice {i:02d}  Y[{lo:.3f}~{hi:.3f}]  cnt={cnt:4d}  color_var={color_var:.5f}")

        if color_var <= args.max_color_var:
            floor_y = lo
            print(f"\n  Floor detected at Y >= {floor_y:.4f} -> removing")
            break

    if floor_y is None:
        print("\nFloor not detected. Try lowering --min_density or increasing --max_color_var")
        return

    keep = y < floor_y
    print(f"  Removed: {(~keep).sum():,}  Kept: {keep.sum():,}")

    save_ply(data, props, keep, out_path)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
