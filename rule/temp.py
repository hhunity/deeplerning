from PIL import Image, ImageFilter, ImageEnhance
import numpy as np

# --------------------------------------------------
# 1. 平行移動
# --------------------------------------------------
def shift_image(img, shift_x=10, shift_y=0):
    """
    画像を平行移動させる関数。はみ出た部分は消え、隙間は黒(0)になる。
    - shift_x: 横方向の移動量（正の数で右、負の数で左）
    - shift_y: 縦方向の移動量（正の数で下、負の数で上）
    """
    # Pillowのtransform(AFFINE)は「移動先の座標から元の座標をどう計算するか」
    # を指定するため、移動量にはマイナスをつけるのが正解です
    matrix = (1, 0, -shift_x, 0, 1, -shift_y)
    return img.transform(img.size, Image.AFFINE, matrix, fillcolor=0)

# --------------------------------------------------
# 2. ノイズ付加
# --------------------------------------------------
def add_noise(img, amount=20):
    """
    ランダムな砂嵐ノイズを付加する関数。
    - amount: ノイズの強さ（例: 20なら -20〜+20 の変動を加える）
    """
    # 計算中に255を超えたり0未満になったりしないよう、一度余裕のあるint16型にする
    img_array = np.array(img, dtype=np.int16)
    
    # ランダムなノイズを生成して足し合わせる
    noise = np.random.randint(-amount, amount + 1, img_array.shape)
    noisy_array = img_array + noise
    
    # 0〜255の範囲に収めて（クリッピング）、8ビット(uint8)に戻す
    noisy_array = np.clip(noisy_array, 0, 255).astype(np.uint8)
    
    return Image.fromarray(noisy_array, mode='L')

# --------------------------------------------------
# 3. ぼかす
# --------------------------------------------------
def blur_image(img, radius=2.0):
    """
    ガウスぼかしをかける関数。
    - radius: ぼかしの強さ（半径）。大きいほどボケる。
    """
    return img.filter(ImageFilter.GaussianBlur(radius=radius))

# --------------------------------------------------
# 4. 輝度UP (明るくする)
# --------------------------------------------------
def increase_brightness(img, factor=1.5):
    """
    画像を明るくする関数。
    - factor: 1.0で変化なし、1.0より大きいと明るくなる（例: 1.5で1.5倍）
    """
    enhancer = ImageEnhance.Brightness(img)
    return enhancer.enhance(factor)

# --------------------------------------------------
# 5. 輝度Down (暗くする)
# --------------------------------------------------
def decrease_brightness(img, factor=0.7):
    """
    画像を暗くする関数。仕組みはUPと同じ。
    - factor: 1.0で変化なし、1.0より小さいと暗くなる（例: 0.7で70%の明るさ）
    """
    enhancer = ImageEnhance.Brightness(img)
    return enhancer.enhance(factor)

# --------------------------------------------------
# 6. コントラスト変更
# --------------------------------------------------
def change_contrast(img, factor=1.5):
    """
    コントラスト（明暗の差）を変更する関数。
    - factor: 1.0で変化なし。1.0より大きいと明暗がくっきりし、小さいと眠い画像になる。
    """
    enhancer = ImageEnhance.Contrast(img)
    return enhancer.enhance(factor)

# --------------------------------------------------
# 7. トーンカーブ変更
# --------------------------------------------------
def apply_tone_curve(img, control_points):
    """
    折れ線グラフ（トーンカーブ）を指定してピクセル値を変換する関数。
    - control_points: (入力値, 出力値)のリスト。必ず入力は昇順にすること。
      例: [(0, 0), (128, 200), (255, 255)] -> 中間調(128)を200に持ち上げるS字カーブ
    """
    # 入力のX座標とY座標を分ける
    x_coords = [p[0] for p in control_points]
    y_coords = [p[1] for p in control_points]
    
    # NumPyの線形補間(interp)を使って、0〜255すべての入力値に対する出力値(LUT)を作る
    lut = np.interp(range(256), x_coords, y_coords)
    
    # 0〜255の整数に丸める
    lut = np.clip(lut, 0, 255).astype(np.uint8)
    
    # Pillowの point() 関数を使って一気に変換（高速）
    return img.point(lut)



from PIL import Image
import math
import os

def rotate_tiff_radians_no_opencv(input_path, output_path, center_x=None, center_y=None):
    """
    OpenCVを使わずに、PillowでTIFF画像をラジアン単位で回転させる。
    画像サイズは不変（クロップ）、余白は黒。
    回転中心を指定可能。
    """
    # 1. 角度の設定 (0.1ラジアン)
    # Pillowのrotateは「度 (degree)」数法を使うため、変換する
    radians = 0.1
    degrees = math.degrees(radians)
    
    try:
        # 2. 画像の読み込み (Pillow)
        # ※ 非常に巨大な画像を扱う場合、Pillowの安全制限(Decompression Bomb)を
        #    無効にする必要があります（メモリに注意！）
        # Image.MAX_IMAGE_PIXELS = None 
        img = Image.open(input_path)
    except FileNotFoundError:
        print(f"エラー: ファイルが見つかりません。 '{input_path}'")
        return
    except Exception as e:
        print(f"エラー: 画像の読み込みに失敗しました。 {e}")
        return

    # 3. 画像のモード(L, RGB, RGBAなど)とサイズを取得
    width, height = img.size
    mode = img.mode

    # 4. 回転中心の決定
    # 指定がなければ、デフォルトで画像中央にする
    if center_x is None:
        center_x = width / 2
    if center_y is None:
        center_y = height / 2

    # 5. 余白の「黒」の色をモードに合わせて自動的に決定
    # 'L' (グレースケール) -> 0 (黒)
    # 'RGB' -> (0, 0, 0)
    # 'RGBA' -> (0, 0, 0, 255) (不透明な黒)
    if mode == 'L':
        fill = 0
    elif mode == 'RGB':
        fill = (0, 0, 0)
    elif mode == 'RGBA':
        # アルファチャンネルも含めて不透明な黒にする
        fill = (0, 0, 0, 255)
    else:
        # 他のモードの場合、チャンネル数に合わせて0のタプルを作成
        fill = tuple([0] * len(img.getbands()))

    print(f"処理開始: {width}x{height}ピクセル, モード: {mode}")
    print(f"回転角: {radians}ラジアン ({degrees:.4f}度, 反時計回り)")
    print(f"回転中心: ({center_x}, {center_y})")

    # 6. 回転を実行
    # angle: 回転角（度）。反時計回りが正。
    # resample: 補間方法。NEAREST（ジャギーが残る）より
    #           BILINEAR（バイリニア）やBICUBIC（バイキュービック）の方が綺麗です。
    # expand: False（デフォルト）なら画像サイズを維持。Trueなら広げる。
    # center: 回転中心の(x, y)座標のタプル。
    # fillcolor: 余白を埋める色。
    
    rotated_img = img.rotate(
        angle=degrees,
        resample=Image.Resampling.BILINEAR, # 画像をきれいに保つためバイリニアを使用
        expand=False, 
        center=(center_x, center_y),
        fillcolor=fill
    )

    # 7. 保存
    # TIFFとして保存（ファイルサイズを抑えるためにLZW圧縮を指定）
    print(f"保存中...")
    rotated_img.save(output_path, compression="tiff_lzw")
    print(f"回転した画像を保存しました: {output_path}")

# 実行例
if __name__ == "__main__":
    # 用意した大きなTIFF画像のパス
    input_file = "input.tif" 
    output_file = "output_rotated_01rad.tif"
    
    # 1. 画像中央を中心に回転する場合 (Noneにするか指定しない)
    rotate_tiff_radians_no_opencv(input_file, output_file)
    
    # 2. 特定の座標 (例えば (1000, 500)) を中心に回転する場合
    # output_file_specified = "output_rotated_specified.tif"
    # rotate_tiff_radians_no_opencv(input_file, output_file_specified, center_x=1000, center_y=500)
