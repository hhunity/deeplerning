import cv2
import numpy as np
import random

def add_dirt_to_tiff(input_path, output_path):
    # 1. 画像の読み込み
    img = cv2.imread(input_path)
    if img is None:
        print("エラー: 画像を読み込めませんでした。パスを確認してください。")
        return

    # 2. 細かい汚れ（砂ぼこり・ノイズ）の追加
    # 画像全体にランダムな暗いノイズを生成（0〜50の値を引いて暗くする）
    noise = np.random.randint(0, 50, img.shape, dtype=np.uint8)
    img_dirty = cv2.subtract(img, noise) # cv2.subtractを使うとマイナス値が0に丸められます

    # 3. シミや泥はね（斑点）の追加
    h, w = img.shape[:2]
    num_spots = 150  # 付加するシミの数（お好みで調整してください）
    
    for _ in range(num_spots):
        # ランダムな座標とサイズを決定
        x = random.randint(0, w - 1)
        y = random.randint(0, h - 1)
        radius = random.randint(1, 6) # 半径1〜6pxのシミ
        
        # 黒〜茶色っぽい色をランダムに生成 (B, G, R)
        color = (
            random.randint(10, 50),  # Blue
            random.randint(20, 60),  # Green
            random.randint(30, 80)   # Red (茶色っぽくするためにRを少し高めに)
        )
        
        # 画像に円（シミ）を描画（-1は塗りつぶし）
        cv2.circle(img_dirty, (x, y), radius, color, -1)

    # 4. ガウシアンフィルタで少しぼかして馴染ませる（オプション）
    # img_dirty = cv2.GaussianBlur(img_dirty, (3, 3), 0)

    # 5. TIFF画像として保存
    cv2.imwrite(output_path, img_dirty)
    print(f"汚れを付加した画像を保存しました: {output_path}")

# 実行例
if __name__ == "__main__":
    # 入力ファイル名と出力ファイル名を指定して関数を呼び出します
    input_file = "input.tif"   # 用意したTIFF画像のパス
    output_file = "output_dirty.tif"
    
    add_dirt_to_tiff(input_file, output_file)
