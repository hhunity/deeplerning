import cv2
import numpy as np
import random

def add_white_dirt(input_path, output_path):
    # 1. 画像をグレースケールで読み込み
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("エラー: 画像を読み込めませんでした。")
        return

    h, w = img.shape

    # 2. 汚れを描画するための黒いキャンバス（マスク）を用意
    dirt_mask = np.zeros((h, w), dtype=np.uint8)

    # --- アプローチA：ペンキや液体の飛び散りのような「白い塊」 ---
    num_splatters = 8  # 塊の数
    for _ in range(num_splatters):
        # 塊の中心座標
        cx = random.randint(0, w)
        cy = random.randint(0, h)
        
        # 中心付近に複数の小さな円を重ねて、不規則な形のシミを作る
        for _ in range(random.randint(4, 10)):
            ox = cx + random.randint(-20, 20)
            oy = cy + random.randint(-20, 20)
            radius = random.randint(2, 9)
            # 白(255)で塗りつぶす
            cv2.circle(dirt_mask, (ox, oy), radius, 255, -1)

    # --- アプローチB：こすれたような「フワッとした白いカスレ」 ---
    smudge_mask = np.zeros((h, w), dtype=np.uint8)
    num_smudges = 5 # カスレの数
    for _ in range(num_smudges):
        # ランダムな太い線を引く
        x1, y1 = random.randint(0, w), random.randint(0, h)
        x2, y2 = x1 + random.randint(-60, 60), y1 + random.randint(-60, 60)
        thickness = random.randint(10, 40)
        # 少し暗めの白(100〜180)で線を引く
        color = random.randint(100, 180)
        cv2.line(smudge_mask, (x1, y1), (x2, y2), color, thickness)
    
    # 引いた線を大きくぼかして、自然な粉っぽいシミにする
    smudge_mask = cv2.GaussianBlur(smudge_mask, (41, 41), 0)

    # 3. 塊(A)とカスレ(B)の汚れを合体させる
    combined_dirt = cv2.add(dirt_mask, smudge_mask)

    # 4. 元画像に白い汚れを加算する
    # cv2.addを使うと、値が255（真っ白）を超えても綺麗に255で止めてくれます
    result = cv2.add(img, combined_dirt)

    # 5. TIFF画像として保存
    cv2.imwrite(output_path, result)
    print(f"白い汚れを付加した画像を保存しました: {output_path}")

# 実行例
if __name__ == "__main__":
    input_file = "input.tif"   # 対象のグレースケールTIFF
    output_file = "output_white_dirty.tif"
    
    add_white_dirt(input_file, output_file)
