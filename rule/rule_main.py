"""
Rule-based crystal counter for microscope images.

=== Image characteristics ===
  Crystal : horizontal bar (~50x20 px in 100x80 image), slightly tilted, bright
  Rim     : bright ring around the cavity — same brightness as crystals but curved
  Interior: slightly darker than crystals
  Cavity  : 1 per image, roughly centered

=== Processing pipeline ===
  1. CLAHE          — normalize uneven illumination inside cavity
  2. Border mask    — exclude outer N px to ignore the bright rim
  3. Detection
       tophat : Top-hat transform with horizontal kernel → shape filter
       contour: Otsu threshold + morphological open → shape filter
  4. Shape filter   — keep only elongated bars (area + aspect ratio)
  5. Output         — count + draw rotated bounding boxes

=== Top-hat kernel rule ===
  Top-hat = original - Opening(kernel)
  - Regions SMALLER than kernel → removed by Opening → appear in Top-hat
  - Regions LARGER  than kernel → survive Opening  → subtracted away

  Crystal is ~50x20 px, so kernel must be LARGER than crystal to detect it:
      NG: kernel (40x10) — kernel fits inside crystal → crystal disappears
      OK: kernel (60x25) — kernel larger than crystal → crystal appears

=== Why rim is excluded ===
  | Feature     | Crystal          | Rim                        |
  |-------------|------------------|----------------------------|
  | Border mask | centered → kept  | near edge → excluded       |
  | Top-hat     | smaller → kept   | curved/large → suppressed  |
  | Aspect ratio| high (50/20=2.5) | arc → low or unstable      |

=== Commands ===
  Infer:    python rule_main.py infer --image path/to/image.png [--save] [--debug]
  Tune:     python rule_main.py tune  --image path/to/image.png
  Validate: python rule_main.py validate --images ./data/images --annotations ./data/annotations.json

=== Key parameters ===
  --kernel_w   Top-hat kernel width  (must be > crystal width,  default: 40)
  --kernel_h   Top-hat kernel height (must be > crystal height, default: 10)
  --min_area   Minimum contour area in px² (default: 300)
  --max_area   Maximum contour area in px² (default: 2000)
  --min_aspect Minimum long/short ratio to classify as bar (default: 2.0)
  --margin     Border pixels to exclude for rim removal (default: 8)
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess(img_bgr: np.ndarray) -> np.ndarray:
    """Grayscale + CLAHE to normalize uneven illumination inside cavity."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    return clahe.apply(gray)


def cavity_mask(img_bgr: np.ndarray, margin: int = 8) -> np.ndarray:
    """
    Binary mask that covers only the cavity interior.
    Since the cavity is centered, exclude the outer margin pixels
    to avoid picking up the bright rim as crystals.
    """
    h, w = img_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[margin:h - margin, margin:w - margin] = 255
    return mask


# ---------------------------------------------------------------------------
# Detection methods
# ---------------------------------------------------------------------------

def _filter_by_shape(contours, min_area: int, max_area: int, min_aspect: float) -> list:
    """
    Keep contours that match crystal shape:
      - area within [min_area, max_area]
      - aspect ratio (long / short side) >= min_aspect  →  elongated bar
    Uses minAreaRect so slightly tilted bars are handled correctly.
    """
    results = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (min_area <= area <= max_area):
            continue
        _, (w, h), _ = cv2.minAreaRect(cnt)
        if w == 0 or h == 0:
            continue
        aspect = max(w, h) / min(w, h)
        if aspect >= min_aspect:
            results.append(cnt)
    return results


def _save_debug(debug_dir: Path, name: str, img: np.ndarray):
    cv2.imwrite(str(debug_dir / f"{name}.png"), img)


def detect_tophat(gray: np.ndarray, mask: np.ndarray,
                  min_area: int, max_area: int, min_aspect: float,
                  kernel_w: int = 40, kernel_h: int = 10,
                  debug_dir: Path = None) -> tuple[list, dict]:
    """
    Top-hat transform with a horizontal rectangular kernel.
    Enhances bar-shaped bright regions relative to local background,
    which suppresses the curved cavity rim (which varies in local brightness).
    kernel_w x kernel_h should be sized close to the crystal dimensions.
    Returns (contours, intermediates) where intermediates is a dict of labeled BGR images.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, kernel_h))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    tophat = cv2.bitwise_and(tophat, tophat, mask=mask)
    _, binary = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    all_vis = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(all_vis, contours, -1, (0, 0, 255), 1)

    filtered = _filter_by_shape(contours, min_area, max_area, min_aspect)

    filtered_vis = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    for cnt in filtered:
        box = cv2.boxPoints(cv2.minAreaRect(cnt)).astype(int)
        cv2.drawContours(filtered_vis, [box], 0, (0, 255, 0), 1)

    intermediates = {
        "1 CLAHE":          cv2.cvtColor(gray,   cv2.COLOR_GRAY2BGR),
        "2 Mask":           cv2.cvtColor(mask,   cv2.COLOR_GRAY2BGR),
        "3 Top-hat":        cv2.cvtColor(tophat, cv2.COLOR_GRAY2BGR),
        "4 Binary":         cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR),
        "5 All contours":   all_vis,
        "6 Filtered":       filtered_vis,
    }

    if debug_dir:
        for name, img in intermediates.items():
            _save_debug(debug_dir, name.replace(" ", "_"), img)

    return filtered, intermediates


def detect_contour(gray: np.ndarray, mask: np.ndarray,
                   min_area: int, max_area: int, min_aspect: float,
                   open_w: int = 5, open_h: int = 3,
                   debug_dir: Path = None) -> tuple[list, dict]:
    """
    Otsu thresholding + morphological cleanup + shape filtering.
    More general approach; relies on aspect ratio to exclude rim arcs.
    open_w x open_h: morphological Open kernel to remove noise smaller than this size.
    Returns (contours, intermediates) where intermediates is a dict of labeled BGR images.
    """
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = cv2.bitwise_and(binary, binary, mask=mask)
    open_k = cv2.getStructuringElement(cv2.MORPH_RECT, (open_w, open_h))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_k)
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    all_vis = cv2.cvtColor(opened, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(all_vis, contours, -1, (0, 0, 255), 1)

    filtered = _filter_by_shape(contours, min_area, max_area, min_aspect)

    filtered_vis = cv2.cvtColor(opened, cv2.COLOR_GRAY2BGR)
    for cnt in filtered:
        box = cv2.boxPoints(cv2.minAreaRect(cnt)).astype(int)
        cv2.drawContours(filtered_vis, [box], 0, (0, 255, 0), 1)

    intermediates = {
        "1 CLAHE":          cv2.cvtColor(gray,   cv2.COLOR_GRAY2BGR),
        "2 Mask":           cv2.cvtColor(mask,   cv2.COLOR_GRAY2BGR),
        "3 Binary":         cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR),
        "4 Opened":         cv2.cvtColor(opened, cv2.COLOR_GRAY2BGR),
        "5 All contours":   all_vis,
        "6 Filtered":       filtered_vis,
    }

    if debug_dir:
        for name, img in intermediates.items():
            _save_debug(debug_dir, name.replace(" ", "_"), img)

    return filtered, intermediates


METHODS = {
    "tophat":  detect_tophat,
    "contour": detect_contour,
}


# ---------------------------------------------------------------------------
# Unified detect + visualize
# ---------------------------------------------------------------------------

def detect(img_bgr: np.ndarray, method: str,
           min_area: int, max_area: int, min_aspect: float,
           margin: int, kernel_w: int = 40, kernel_h: int = 10,
           open_w: int = 5, open_h: int = 3,
           debug_dir: Path = None,
           return_intermediates: bool = False) -> tuple:
    """
    Run detection and return (count, annotated_image) or
    (count, annotated_image, intermediates) when return_intermediates=True.
    """
    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)

    gray = preprocess(img_bgr)
    mask = cavity_mask(img_bgr, margin=margin)
    fn = METHODS[method]
    kwargs = {"debug_dir": debug_dir}
    if method == "tophat":
        kwargs["kernel_w"] = kernel_w
        kwargs["kernel_h"] = kernel_h
    elif method == "contour":
        kwargs["open_w"] = open_w
        kwargs["open_h"] = open_h
    contours, intermediates = fn(gray, mask, min_area, max_area, min_aspect, **kwargs)

    vis = img_bgr.copy()
    for cnt in contours:
        box = cv2.boxPoints(cv2.minAreaRect(cnt)).astype(int)
        cv2.drawContours(vis, [box], 0, (0, 255, 0), 1)

    count = len(contours)
    cv2.putText(vis, f"count={count}", (4, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    if return_intermediates:
        return count, vis, intermediates
    return count, vis


# ---------------------------------------------------------------------------
# CLI handlers
# ---------------------------------------------------------------------------

def infer(args):
    img_bgr = cv2.imread(args.image)
    if img_bgr is None:
        print(f"Cannot open image: {args.image}")
        return

    debug_dir = Path(Path(args.image).stem + "_debug") if args.debug else None
    count, vis = detect(img_bgr, args.method,
                        args.min_area, args.max_area,
                        args.min_aspect, args.margin,
                        args.kernel_w, args.kernel_h,
                        args.open_w, args.open_h,
                        debug_dir=debug_dir)
    print(f"Detected crystal count: {count}")

    if args.save or args.debug:
        out_path = Path(args.image).stem + f"_{args.method}_result.png"
        cv2.imwrite(out_path, vis)
        print(f"Result image saved to: {out_path}")

    if args.debug:
        print(f"Debug images saved to: {debug_dir}/")


def tune(args):
    """
    Interactive parameter tuning with OpenCV trackbars.
    Sliders update the detection result in real time.
    Press 's' to save the current result, 'q' or ESC to quit.
    """
    img_bgr = cv2.imread(args.image)
    if img_bgr is None:
        print(f"Cannot open image: {args.image}")
        return

    # Scale up for visibility (100x80 is tiny on screen)
    scale = 4
    display_h = img_bgr.shape[0] * scale
    display_w = img_bgr.shape[1] * scale

    win = "Crystal Tuner  [s=save  q=quit]"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, display_w, display_h)

    # Trackbar values are integers; min_aspect is stored as x10
    cv2.createTrackbar("min_area",    win, args.min_area,   5000, lambda _: None)
    cv2.createTrackbar("max_area",    win, args.max_area,   5000, lambda _: None)
    cv2.createTrackbar("min_aspect*10", win, int(args.min_aspect * 10), 100, lambda _: None)
    cv2.createTrackbar("margin",      win, args.margin,     30,   lambda _: None)
    cv2.createTrackbar("kernel_w",    win, args.kernel_w,   100,  lambda _: None)
    cv2.createTrackbar("kernel_h",    win, args.kernel_h,   50,   lambda _: None)
    cv2.createTrackbar("open_w",      win, args.open_w,     30,   lambda _: None)
    cv2.createTrackbar("open_h",      win, args.open_h,     30,   lambda _: None)

    method_names = list(METHODS)
    method_idx = method_names.index(args.method) if args.method in method_names else 0
    cv2.createTrackbar("method(0=tophat 1=contour)", win, method_idx, len(method_names) - 1, lambda _: None)

    while True:
        min_area   = cv2.getTrackbarPos("min_area",    win)
        max_area   = cv2.getTrackbarPos("max_area",    win)
        min_aspect = cv2.getTrackbarPos("min_aspect*10", win) / 10.0
        margin     = cv2.getTrackbarPos("margin",      win)
        kernel_w   = cv2.getTrackbarPos("kernel_w",    win)
        kernel_h   = cv2.getTrackbarPos("kernel_h",    win)
        open_w     = cv2.getTrackbarPos("open_w",      win)
        open_h     = cv2.getTrackbarPos("open_h",      win)
        method     = method_names[cv2.getTrackbarPos("method(0=tophat 1=contour)", win)]

        # Prevent invalid ranges
        min_area   = max(1, min_area)
        max_area   = max(min_area + 1, max_area)
        min_aspect = max(1.0, min_aspect)
        kernel_w   = max(1, kernel_w)
        kernel_h   = max(1, kernel_h)
        open_w     = max(1, open_w)
        open_h     = max(1, open_h)

        count, vis, intermediates = detect(
            img_bgr, method, min_area, max_area, min_aspect,
            margin, kernel_w, kernel_h, open_w, open_h,
            return_intermediates=True)

        # Build tiled view: Original + 6 intermediate panels (3 cols x 2 rows + result)
        panels = [("Original", img_bgr)] + list(intermediates.items()) + [("Result", vis)]
        # Pad to fill a 4x2 grid
        while len(panels) < 8:
            panels.append(("", np.zeros_like(img_bgr)))
        cols, rows = 4, 2
        cell_h, cell_w = img_bgr.shape[0], img_bgr.shape[1]
        label_h = 12
        grid = np.zeros((rows * (cell_h + label_h), cols * cell_w, 3), dtype=np.uint8)
        for i, (label, panel) in enumerate(panels[:cols * rows]):
            r, c = divmod(i, cols)
            y = r * (cell_h + label_h)
            x = c * cell_w
            grid[y:y + label_h, x:x + cell_w] = 30  # dark label bar
            cv2.putText(grid, label, (x + 2, y + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1, cv2.LINE_AA)
            grid[y + label_h:y + label_h + cell_h, x:x + cell_w] = panel

        params_text = (f"count={count}  min_area={min_area}  max_area={max_area}"
                       f"  min_aspect={min_aspect:.1f}  margin={margin}"
                       f"  kernel={kernel_w}x{kernel_h}")
        cv2.putText(grid, params_text, (4, grid.shape[0] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1, cv2.LINE_AA)

        grid_large = cv2.resize(grid, (grid.shape[1] * scale, grid.shape[0] * scale),
                                interpolation=cv2.INTER_NEAREST)
        cv2.resizeWindow(win, grid_large.shape[1], grid_large.shape[0])
        cv2.imshow(win, grid_large)
        key = cv2.waitKey(30) & 0xFF

        if key in (ord("q"), 27):  # q or ESC
            break
        if key == ord("s"):
            out_path = Path(args.image).stem + f"_{method}_tuned.png"
            cv2.imwrite(out_path, vis)
            print(f"Saved: {out_path}")
            print(f"  method={method}  min_area={min_area}  max_area={max_area}"
                  f"  min_aspect={min_aspect:.1f}  margin={margin}"
                  f"  kernel_w={kernel_w}  kernel_h={kernel_h}"
                  f"  open_w={open_w}  open_h={open_h}")

    cv2.destroyAllWindows()


def validate(args):
    with open(args.annotations) as f:
        coco = json.load(f)

    count_map = {}
    for ann in coco["annotations"]:
        iid = ann["image_id"]
        count_map[iid] = count_map.get(iid, 0) + 1

    images_dir = Path(args.images)
    filenames, truths, preds = [], [], []

    for img_info in coco["images"]:
        img_path = images_dir / img_info["file_name"]
        if not img_path.exists():
            continue
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue

        truth = count_map.get(img_info["id"], 0)
        pred, _ = detect(img_bgr, args.method,
                         args.min_area, args.max_area,
                         args.min_aspect, args.margin,
                         args.kernel_w, args.kernel_h,
                         args.open_w, args.open_h)

        filenames.append(img_info["file_name"])
        truths.append(truth)
        preds.append(pred)

    truths = np.array(truths, dtype=float)
    preds  = np.array(preds,  dtype=float)
    errors = preds - truths
    mae    = np.mean(np.abs(errors))
    rmse   = np.sqrt(np.mean(errors ** 2))
    ss_tot = np.sum((truths - truths.mean()) ** 2)
    r2     = 1 - np.sum(errors ** 2) / ss_tot if ss_tot > 0 else float("nan")

    print(f"\nValidation Results  [method={args.method}]")
    print("==================")
    print(f"MAE  : {mae:.3f}")
    print(f"RMSE : {rmse:.3f}")
    print(f"R²   : {r2:.3f}")
    print()
    print(f"{'image':<30} {'truth':>6} {'pred':>6} {'error':>7}")
    print("-" * 54)
    for fname, t, p, e in zip(filenames, truths, preds, errors):
        print(f"{fname:<30} {t:>6.0f} {p:>6.0f} {e:>+7.0f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Rule-based crystal counter")
    sub = parser.add_subparsers(dest="mode", required=True)

    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--method",     default="tophat", choices=list(METHODS),
                        help="Detection method: tophat (default) | contour")
    shared.add_argument("--min_area",   type=int,   default=300,
                        help="Minimum contour area in pixels (default: 300)")
    shared.add_argument("--max_area",   type=int,   default=2000,
                        help="Maximum contour area in pixels (default: 2000)")
    shared.add_argument("--min_aspect", type=float, default=2.0,
                        help="Minimum aspect ratio (long/short) to classify as bar-shaped crystal (default: 2.0)")
    shared.add_argument("--margin",     type=int,   default=8,
                        help="Pixels to exclude from image border to avoid cavity rim (default: 8)")
    shared.add_argument("--kernel_w",   type=int,   default=40,
                        help="Top-hat kernel width in pixels, should be close to crystal width (default: 40)")
    shared.add_argument("--kernel_h",   type=int,   default=10,
                        help="Top-hat kernel height in pixels, should be close to crystal height (default: 10)")
    shared.add_argument("--open_w",     type=int,   default=5,
                        help="Contour method: morphological Open kernel width to remove noise (default: 5)")
    shared.add_argument("--open_h",     type=int,   default=3,
                        help="Contour method: morphological Open kernel height to remove noise (default: 3)")

    p_infer = sub.add_parser("infer", parents=[shared],
                             help="Detect crystals in a single image")
    p_infer.add_argument("--image", required=True,
                         help="Path to the input image file")
    p_infer.add_argument("--save", action="store_true",
                         help="Save annotated result image")
    p_infer.add_argument("--debug", action="store_true",
                         help="Save intermediate images for each processing step to <image_stem>_debug/")

    p_tune = sub.add_parser("tune", parents=[shared],
                            help="Interactive parameter tuning with trackbars (press s=save, q=quit)")
    p_tune.add_argument("--image", required=True,
                        help="Path to the input image file")

    p_val = sub.add_parser("validate", parents=[shared],
                           help="Batch validation against COCO annotations")
    p_val.add_argument("--images",      required=True,
                       help="Directory containing image files")
    p_val.add_argument("--annotations", required=True,
                       help="COCO annotations JSON path")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "infer":
        infer(args)
    elif args.mode == "tune":
        tune(args)
    elif args.mode == "validate":
        validate(args)
