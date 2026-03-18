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
