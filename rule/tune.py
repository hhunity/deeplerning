"""
Parameter tuner for rule_main2 functions using OpenCV trackbars.
Usage:
  python tune.py <image_path> --mode 0   # make_cavity_mask
  python tune.py <image_path> --mode 1   # mask_frame_by_scan_v2
  python tune.py <image_path> --mode 2   # mask_frame_by_contour
Press q to quit.

Preview layout (top row: original + masked, bottom row: processing steps):
  mode 0: Canny | morphologyEx | findContours+approx | final mask
  mode 1: Canny | scan lines   | final mask
  mode 2: Canny | morphologyEx | findContours        | final mask
"""

import sys
import argparse
import cv2
import numpy as np
from rule_main2 import make_cavity_mask, mask_frame_by_scan_v2, mask_frame_by_contour

WIN_PARAMS = "Params"
WIN_VIEW   = "Preview"

MODE_NAMES = ["make_cavity_mask", "mask_frame_by_scan_v2", "mask_frame_by_contour"]

STEP_LABELS = {
    0: ["Canny", "morphologyEx", "findContours", "final mask"],
    1: ["Canny", "scan lines",   "final mask"],
    2: ["Canny", "morphologyEx", "findContours", "final mask"],
}


def nothing(_):
    pass


def build_params(mode: int):
    cv2.namedWindow(WIN_PARAMS)
    specs = {
        0: [("canny_low", 50, 255), ("canny_high", 150, 255), ("ksize", 21, 101), ("margin", 8, 50)],
        1: [("canny_low", 50, 255), ("canny_high", 150, 255), ("pad", 8, 50)],
        2: [("canny_low", 50, 255), ("canny_high", 150, 255),
            ("close_ksize", 21, 101), ("min_area_pct", 30, 100), ("pad", 8, 50)],
    }
    for name, default, maxval in specs[mode]:
        cv2.createTrackbar(name, WIN_PARAMS, default, maxval, nothing)


def tb(name: str) -> int:
    return cv2.getTrackbarPos(name, WIN_PARAMS)


def odd(v: int) -> int:
    v = max(v, 1)
    return v if v % 2 == 1 else v + 1


def apply_function(img_gray: np.ndarray, mode: int):
    """Return (mask, step_images)."""
    low, high = tb("canny_low"), tb("canny_high")
    if mode == 0:
        return make_cavity_mask(
            img_gray,
            ksize=odd(tb("ksize")), canny_low=low, canny_high=high,
            margin=tb("margin"), debug=True,
        )
    elif mode == 1:
        return mask_frame_by_scan_v2(
            img_gray, canny_low=low, canny_high=high,
            pad=tb("pad"), debug=True,
        )
    else:
        return mask_frame_by_contour(
            img_gray,
            canny_low=low, canny_high=high,
            close_ksize=odd(tb("close_ksize")),
            min_area_ratio=tb("min_area_pct") / 100.0,
            pad=tb("pad"), debug=True,
        )


def to_bgr(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img.copy()


def add_label(img: np.ndarray, text: str) -> np.ndarray:
    cv2.putText(img, text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    return img


def hstack_panels(panels: list[np.ndarray]) -> np.ndarray:
    h = panels[0].shape[0]
    sep = np.zeros((h, 2, 3), dtype=np.uint8)
    out = panels[0]
    for p in panels[1:]:
        out = np.hstack([out, sep, p])
    return out


def build_preview(img_gray: np.ndarray, mask: np.ndarray,
                  steps: list, mode: int) -> np.ndarray:
    # Top row: original | masked
    masked   = cv2.bitwise_and(img_gray, img_gray, mask=mask)
    top_row  = hstack_panels([
        add_label(to_bgr(img_gray), "original"),
        add_label(to_bgr(masked),   "masked"),
    ])

    # Bottom row: processing steps
    labels     = STEP_LABELS[mode]
    step_panels = [add_label(to_bgr(s), labels[i]) for i, s in enumerate(steps)]

    # Resize step panels to match top_row height / num_steps
    target_w = top_row.shape[1] // len(step_panels)
    target_h = top_row.shape[0]
    resized = [cv2.resize(p, (target_w, target_h)) for p in step_panels]
    bot_row = hstack_panels(resized)

    # Pad widths to match if needed
    tw = top_row.shape[1]
    bw = bot_row.shape[1]
    if tw > bw:
        bot_row = np.hstack([bot_row, np.zeros((target_h, tw - bw, 3), dtype=np.uint8)])
    elif bw > tw:
        top_row = np.hstack([top_row, np.zeros((top_row.shape[0], bw - tw, 3), dtype=np.uint8)])

    divider = np.zeros((2, top_row.shape[1], 3), dtype=np.uint8)
    return np.vstack([top_row, divider, bot_row])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="path to input image")
    parser.add_argument("--mode", type=int, choices=[0, 1, 2], default=0,
                        help="0=make_cavity_mask  1=mask_frame_by_scan_v2  2=mask_frame_by_contour")
    args = parser.parse_args()

    img_gray = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        print(f"Cannot load image: {args.image}")
        sys.exit(1)

    print(f"Mode {args.mode}: {MODE_NAMES[args.mode]}")
    cv2.namedWindow(WIN_VIEW)
    build_params(args.mode)

    while True:
        mask, steps = apply_function(img_gray, args.mode)
        cv2.imshow(WIN_VIEW, build_preview(img_gray, mask, steps, args.mode))
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
