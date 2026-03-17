import cv2
import numpy as np

def make_cavity_mask(img_gray, ksize=21, canny_low=50, canny_high=150, margin=5):
    """
    フチ（キャビティ枠）を検出し、内側だけ白いマスクを返す。
    margin: フチ自体を何px分食い込ませてマスクするか（フチのエッジを消すため）
    """
    # 1. エッジ検出
    edges = cv2.Canny(img_gray, canny_low, canny_high)

    # 2. 大きなカーネルでクロージング → 結晶の小エッジを潰し、フチだけ残す
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # 3. 輪郭検出 → 最大の矩形を探す
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        # フチが見つからなければ全域マスク
        return np.ones_like(img_gray, dtype=np.uint8) * 255

    # 面積最大の輪郭を選ぶ
    largest = max(contours, key=cv2.contourArea)

    # 4. 凸包 or 近似多角形でフチ形状を安定化
    #    フチが厳密な矩形なら approxPolyDP が効く
    epsilon = 0.02 * cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, epsilon, True)

    # 5. マスク生成（フチ内側 = 白）
    mask = np.zeros_like(img_gray, dtype=np.uint8)
    cv2.fillPoly(mask, [approx], 255)

    # 6. marginぶん縮小してフチ自体のエッジもマスクに含めない
    if margin > 0:
        shrink_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (margin*2+1, margin*2+1))
        mask = cv2.erode(mask, shrink_kernel)

    return mask


# --- 使い方 ---
img = cv2.imread("cavity.png", cv2.IMREAD_GRAYSCALE)
mask = make_cavity_mask(img, ksize=21, margin=8)

# マスク適用：フチ外を黒（or 任意の値）に
masked = cv2.bitwise_and(img, img, mask=mask)

def mask_frame_by_scan_v2(img_gray, canny_low=50, canny_high=150, pad=8):
    """
    四辺からスキャンし、各辺のエッジ位置の中央値でフチ境界を決める。
    結晶がフチに掛かっている列があっても、中央値で吸収される。
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

    return mask

import cv2
import numpy as np

def mask_frame_by_contour(img_gray, canny_low=50, canny_high=150, 
                          min_area_ratio=0.3, pad=8):
    """
    外側から最初の大きな輪郭＝フチとみなし、その内側マスクを返す。
    """
    h, w = img_gray.shape
    img_area = h * w
    edges = cv2.Canny(img_gray, canny_low, canny_high)

    # RETR_EXTERNAL → 最も外側の輪郭だけ取得
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 面積でフィルタ：フチは画像面積の30%以上あるはず
    frame_contour = None
    for c in sorted(contours, key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(c)
        if area > img_area * min_area_ratio:
            frame_contour = c
            break

    if frame_contour is None:
        return np.ones((h, w), dtype=np.uint8) * 255

    # フチ内側をマスク
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [frame_contour], 255)

    # pad分縮小してフチ自体のエッジも消す
    if pad > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (pad*2+1, pad*2+1))
        mask = cv2.erode(mask, kernel)

    return mask



