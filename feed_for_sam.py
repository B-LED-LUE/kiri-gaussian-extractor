import argparse
import numpy as np
import cv2
import torch
from pathlib import Path

parser = argparse.ArgumentParser(description="Interactive SAM2 masking on rendered views")
parser.add_argument("--views_dir", type=Path, required=True, help="Directory with view_*.png images")
parser.add_argument("--mask_dir",  type=Path, required=True, help="Directory to save mask_*.png")
parser.add_argument("--ckpt",      type=Path, default=Path("models/sam2.1_hiera_base_plus.pt"))
parser.add_argument("--cfg",       type=str,  default="configs/sam2.1/sam2.1_hiera_b+.yaml")
args = parser.parse_args()

WIN = "SAM2 Segmentation"

# Controls:
#   Mouse Drag    : Draw bounding box
#   Left Click    : Add positive point
#   Right Click   : Add negative point
#   R             : Reset current image
#   S / Enter     : Save mask and move to next image
#   Q             : Quit


def load_sam2():
    print("Loading SAM2...", flush=True)
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = build_sam2(args.cfg, str(args.ckpt), device=device, apply_postprocessing=False)
    pred   = SAM2ImagePredictor(model)
    print(f"  SAM2 ready ({device})", flush=True)
    return pred


def overlay_mask(image, mask, color=(0, 255, 0), alpha=0.45):
    out = image.copy()
    out[mask] = (out[mask] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, color, 2)
    return out


class AnnotationState:
    def __init__(self):
        self.reset()

    def reset(self):
        self.box_start  = None
        self.box_end    = None
        self.drawing    = False
        self.pos_pts    = []
        self.neg_pts    = []
        self.mask       = None


def run_sam(pred, image_rgb, state):
    box = None
    if state.box_start and state.box_end:
        x0, y0 = state.box_start
        x1, y1 = state.box_end
        box = np.array([min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)], dtype=np.float32)

    points, labels = [], []
    for p in state.pos_pts:
        points.append(p); labels.append(1)
    for p in state.neg_pts:
        points.append(p); labels.append(0)

    pts_arr = np.array(points, dtype=np.float32) if points else None
    lbl_arr = np.array(labels, dtype=np.int32)   if labels else None

    if box is None and pts_arr is None:
        return

    masks, scores, _ = pred.predict(
        box=box, point_coords=pts_arr, point_labels=lbl_arr, multimask_output=False)
    state.mask = masks[0].astype(bool)


def draw_ui(image_rgb, state):
    vis = image_rgb.copy()
    if state.mask is not None:
        vis = overlay_mask(vis, state.mask)

    vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)

    if state.box_start and state.box_end:
        cv2.rectangle(vis_bgr, state.box_start, state.box_end, (0, 165, 255), 2)
    for p in state.pos_pts:
        cv2.circle(vis_bgr, p, 6, (0, 255, 0), -1)
        cv2.circle(vis_bgr, p, 6, (255, 255, 255), 1)
    for p in state.neg_pts:
        cv2.circle(vis_bgr, p, 6, (0, 0, 255), -1)
        cv2.circle(vis_bgr, p, 6, (255, 255, 255), 1)

    guide = "[Drag]Box  [L-Click]Pos  [R-Click]Neg  [R]Reset  [S/Enter]Save  [Q]Quit"
    cv2.putText(vis_bgr, guide, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(vis_bgr, guide, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    return vis_bgr


def process_image(img_path, pred, img_idx, total):
    image_bgr = cv2.imread(str(img_path))
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pred.set_image(image_rgb)
    state = AnnotationState()
    needs_predict = [False]
    drag_start = [None]

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            drag_start[0] = (x, y)
            state.drawing   = True
            state.box_start = (x, y)
            state.box_end   = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and state.drawing:
            state.box_end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and state.drawing:
            state.drawing = False
            sx, sy = drag_start[0]
            if abs(x - sx) < 10 and abs(y - sy) < 10:
                state.box_start = None
                state.box_end   = None
                state.pos_pts.append((x, y))
            else:
                state.box_end = (x, y)
            needs_predict[0] = True
            drag_start[0] = None
        elif event == cv2.EVENT_RBUTTONDOWN:
            state.neg_pts.append((x, y))
            needs_predict[0] = True

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.setWindowTitle(WIN, f"[{img_idx}/{total}] {img_path.name}  —  SAM2")
    cv2.setMouseCallback(WIN, on_mouse)
    print(f"\n[{img_idx}/{total}] {img_path.name}")

    while True:
        if needs_predict[0]:
            needs_predict[0] = False
            run_sam(pred, image_rgb, state)

        cv2.imshow(WIN, draw_ui(image_rgb, state))
        key = cv2.waitKey(30) & 0xFF

        if key in (ord('q'), ord('Q')):
            return None
        elif key in (ord('r'), ord('R')):
            state.reset()
            pred.set_image(image_rgb)
            print("  Reset")
        elif key in (ord('s'), ord('S'), 13):
            if state.mask is None:
                print("  No mask — draw a box or click a point first")
                continue
            return state.mask

    return state.mask


def main():
    img_paths = sorted(args.views_dir.glob("view_*.png"))
    if not img_paths:
        print(f"No images found in {args.views_dir}")
        return

    print(f"Found {len(img_paths)} images")
    pred = load_sam2()
    args.mask_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for idx, img_path in enumerate(img_paths, 1):
        mask = process_image(img_path, pred, idx, len(img_paths))
        if mask is None:
            print("Terminated")
            break
        mask_path = args.mask_dir / img_path.name.replace("view_", "mask_")
        cv2.imwrite(str(mask_path), (mask * 255).astype(np.uint8))
        print(f"  saved: {mask_path.name}")
        saved += 1

    cv2.destroyAllWindows()
    print(f"\nDone: {saved} masks saved -> {args.mask_dir}")


if __name__ == "__main__":
    main()
