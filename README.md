# 3DGS Object Gaussian Extractor

Extract per-object Gaussians from a 3D Gaussian Splatting scene using SAM2 multi-view voting.

Designed for PLY files exported from **Kiri Engine** (or any standard 3DGS trainer).

Render synthetic views from the trained 3DGS → label with SAM2 → project masks back to 3D → vote across views → clean output PLY per object.

## Pipeline

```
3DGS PLY
   │
   ▼
segmentation_for_gaussian.py   — render 6 views + save camera JSONs
   │
   ▼
feed_for_sam.py                — interactive SAM2 masking on rendered views
   │
   ▼
now_rebuilt.py                 — multi-view voting → obj.ply
   │
   ▼
gaussian_cleaner_fi.py         — remove floor/desk Gaussians
   │
   ▼
obj_clean.ply                  — clean per-object Gaussians
```

Optional second pass (for refinement):
```
feed_for_meshing.py            — render Zero123++ layout views + Real-ESRGAN upscale
feed_for_sam_sec.py            — re-mask on new views
now_rebuilt_sec.py             — second-pass voting
```

## Requirements

```bash
pip install -r requirements.txt
```

Install from source:
- **gsplat**: https://github.com/nerfstudio-project/gsplat
- **SAM2**: https://github.com/facebookresearch/sam2

SAM2 checkpoint: download `sam2.1_hiera_base_plus.pt` from [SAM2 releases](https://github.com/facebookresearch/sam2/releases) and place in `models/`.

## Usage

### Step 1 — Render views
```bash
python segmentation_for_gaussian.py \
    --ply <scene.ply> \
    --output output/views
```
Outputs 6 rendered images (`view_001.png` ... `view_006.png`) and camera JSONs.

### Step 2 — Label with SAM2
```bash
python feed_for_sam.py \
    --views_dir output/views \
    --mask_dir  output/masks \
    --ckpt      models/sam2.1_hiera_base_plus.pt
```
For each view: drag to draw a box, click for points, **S** to save, **Q** to quit.

### Step 3 — Multi-view voting
```bash
python now_rebuilt.py \
    --ply       <scene.ply> \
    --mask_dir  output/masks \
    --cam_dir   output/views \
    --out_dir   output/segmented \
    --min_votes 3
```
Outputs `obj.ply` (object) and `without_obj.ply` (background).

### Step 4 — Remove floor Gaussians
```bash
python gaussian_cleaner_fi.py \
    --ply    output/segmented/obj.ply \
    --output output/segmented/obj_clean.ply
```

---

## PROBLEMS (Personal Notes)

Things tried, things that failed, and why. Eventually led to moving from 3DGS to 2DGS.

1. **Floor always comes with object** — no matter how carefully the mask is drawn, floor Gaussians always leak into the object selection. The boundary between object base and floor is physically ambiguous in 3D.

2. **Geometry-based filtering before SAM** — tried filtering Gaussians by geometric features (normals, height, density) before masking. Didn't work — valid object Gaussians share the same geometric properties as floor Gaussians at the boundary.

3. **No camera parameters from Kiri Engine (Android)** — Kiri Engine Android export only provides the PLY file. No camera poses, no intrinsics, no training views. All camera positions had to be synthetically generated, which means masks don't correspond to real training views.

4. **Mesh quality was bad** — 3DGS mesh extraction (Poisson, alpha shape, etc.) produced poor quality meshes with holes, noise, and stretched artifacts. Not usable for Unity import without heavy manual cleanup.

5. **SuGaR** — tried as an alternative mesh extraction method on top of 3DGS. Too slow to run, result not worth the wait.

6. **Gaussian Grouping** — tried for automatic object segmentation using SAM2 video tracking. Too slow to train, not practical for this use case.

**Conclusion:** All approaches hit a ceiling with 3DGS + Kiri Engine due to missing camera data and poor mesh quality. Moved to 2DGS which provides camera parameters, better normals, and TSDF mesh extraction.

---

## How It Works

**Problem:** Extracting individual objects from a 3DGS scene requires knowing which Gaussians belong to each object. There is no label information in a standard trained 3DGS model.

**Approach — Multi-view voting:**
1. Render synthetic views of the full scene from known camera positions using gsplat
2. User labels the target object in each view using SAM2 (box/point prompts)
3. Each Gaussian is projected into each camera view
4. If a Gaussian lands inside the SAM2 mask in ≥ N views, it is classified as object
5. Voting across multiple views reduces false positives from single-view ambiguity

**Floor contamination:** Gaussians at the floor/desk surface are often included in object masks when viewed from above. `gaussian_cleaner_fi.py` detects the floor by scanning horizontal Y-slices — slices with high density and low color variance (uniform color = flat surface) are removed.
