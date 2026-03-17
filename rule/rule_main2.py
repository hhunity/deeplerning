import cv2
import numpy as np


def make_cavity_mask(img_gray, ksize=21, canny_low=50, canny_high=150, margin=5, debug=False):
    """
    フチ（キャビティ枠）を検出し、内側だけ白いマスクを返す。
    margin: フチ自体を何px分食い込ませてマスクするか（フチのエッジを消すため）
    debug: if True, return (mask, [step_images]) where step_images are labeled BGR images.
    """
    # 1. エッジ検出
    edges = cv2.Canny(img_gray, canny_low, canny_high)

    # 2. 大きなカーネルでクロージング → 結晶の小エッジを潰し、フチだけ残す
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # 3. 輪郭検出 → 最大の矩形を探す
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        mask = np.ones_like(img_gray, dtype=np.uint8) * 255
        if debug:
            contour_img = cv2.cvtColor(closed, cv2.COLOR_GRAY2BGR)
            return mask, [edges, closed, contour_img, mask]
        return mask

    # 面積最大の輪郭を選ぶ
    largest = max(contours, key=cv2.contourArea)

    # 4. 凸包 or 近似多角形でフチ形状を安定化
    epsilon = 0.02 * cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, epsilon, True)

    # Draw approx contour on black background so it's visually distinct from closed
    contour_img = np.zeros((*img_gray.shape, 3), dtype=np.uint8)
    cv2.drawContours(contour_img, [approx], -1, (0, 255, 0), 2)

    # 5. マスク生成（フチ内側 = 白）
    mask = np.zeros_like(img_gray, dtype=np.uint8)
    cv2.fillPoly(mask, [approx], 255)

    # 6. marginぶん縮小してフチ自体のエッジもマスクに含めない
    if margin > 0:
        shrink_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (margin*2+1, margin*2+1))
        mask = cv2.erode(mask, shrink_kernel)

    if debug:
        return mask, [edges, closed, contour_img, mask]
    return mask


def mask_frame_by_scan_v2(img_gray, canny_low=50, canny_high=150, pad=8, debug=False):
    """
    四辺からスキャンし、各辺のエッジ位置の中央値でフチ境界を決める。
    結晶がフチに掛かっている列があっても、中央値で吸収される。
    debug: if True, return (mask, [step_images]).
    """
    h, w = img_gray.shape
    edges = cv2.Canny(img_gray, canny_low, canny_high)

    # --- 上辺：各列で最初のエッジ ---
    top_positions = []
    for x in range(w):
        hits = np.where(edges[:, x] > 0)[0]
        if len(hits) > 0:
            top_positions.append(hits[0])
    top_line = int(np.median(top_positions)) + pad if top_positions else pad

    # --- 下辺：各列で最後のエッジ ---
    bot_positions = []
    for x in range(w):
        hits = np.where(edges[:, x] > 0)[0]
        if len(hits) > 0:
            bot_positions.append(hits[-1])
    bot_line = int(np.median(bot_positions)) - pad if bot_positions else h - pad

    # --- 左辺：各行で最初のエッジ ---
    left_positions = []
    for y in range(h):
        hits = np.where(edges[y, :] > 0)[0]
        if len(hits) > 0:
            left_positions.append(hits[0])
    left_line = int(np.median(left_positions)) + pad if left_positions else pad

    # --- 右辺：各行で最後のエッジ ---
    right_positions = []
    for y in range(h):
        hits = np.where(edges[y, :] > 0)[0]
        if len(hits) > 0:
            right_positions.append(hits[-1])
    right_line = int(np.median(right_positions)) - pad if right_positions else w - pad

    # 矩形マスク
    mask = np.zeros_like(img_gray, dtype=np.uint8)
    mask[top_line:bot_line, left_line:right_line] = 255

    if debug:
        # Draw detected scan lines on edge image
        scan_img = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        cv2.line(scan_img, (0, top_line),  (w, top_line),  (0, 255, 0), 1)
        cv2.line(scan_img, (0, bot_line),  (w, bot_line),  (0, 255, 0), 1)
        cv2.line(scan_img, (left_line, 0), (left_line, h), (0, 255, 0), 1)
        cv2.line(scan_img, (right_line, 0),(right_line, h),(0, 255, 0), 1)
        return mask, [edges, scan_img, mask]
    return mask


def mask_frame_by_contour(img_gray, canny_low=50, canny_high=150,
                           close_ksize=21, min_area_ratio=0.3, pad=8, debug=False):
    """
    外側から最初の大きな輪郭＝フチとみなし、その内側マスクを返す。
    Canny edges are closed with morphological closing before findContours,
    otherwise open edge fragments are detected as separate contours.
    debug: if True, return (mask, [step_images]).
    """
    h, w = img_gray.shape
    img_area = h * w
    edges = cv2.Canny(img_gray, canny_low, canny_high)

    # Close gaps in edges so the frame forms a single closed contour
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_ksize, close_ksize))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 面積でフィルタ：フチは画像面積の30%以上あるはず
    frame_contour = None
    for c in sorted(contours, key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(c)
        if area > img_area * min_area_ratio:
            frame_contour = c
            break

    if frame_contour is None:
        mask = np.ones((h, w), dtype=np.uint8) * 255
        if debug:
            contour_img = np.zeros((h, w, 3), dtype=np.uint8)
            return mask, [edges, closed, contour_img, mask]
        return mask

    # Draw selected contour on black background so it's visually distinct from closed
    contour_img = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.drawContours(contour_img, [frame_contour], -1, (0, 255, 0), 2)

    # フチ内側をマスク
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [frame_contour], 255)

    # pad分縮小してフチ自体のエッジも消す
    if pad > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (pad*2+1, pad*2+1))
        mask = cv2.erode(mask, kernel)

    if debug:
        return mask, [edges, closed, contour_img, mask]
    return mask


if __name__ == "__main__":
    img = cv2.imread("cavity.png", cv2.IMREAD_GRAYSCALE)
    mask = make_cavity_mask(img, ksize=21, margin=8)
    masked = cv2.bitwise_and(img, img, mask=mask)
