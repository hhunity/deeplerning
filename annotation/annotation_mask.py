t# annotate_mask.py
# UNet用マスクアノテーションツール
# 物体をブラシで塗りつぶしてマスク画像（白=物体、黒=背景）を作成する
#
# 使い方:
#   python annotate_mask.py --images dataset/images --output dataset/masks
#
# 操作:
#   左クリック・ドラッグ : 白で塗る（物体）
#   右クリック・ドラッグ : 黒で塗る（消しゴム）
#   S キー              : 保存して次の画像へ
#   Z キー              : 1つ戻す（アンドゥ）
#   C キー              : 全クリア
#   Q キー              : 終了
#   マウスホイール       : ブラシサイズ変更

import os
import sys
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image


def annotate_images(images_dir: str, output_dir: str, brush_size: int = 10):
    os.makedirs(output_dir, exist_ok=True)

    exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
    image_paths = sorted(
        p for ext in exts for p in glob.glob(os.path.join(images_dir, ext))
    )
    if not image_paths:
        print(f"画像が見つかりません: {images_dir}")
        sys.exit(1)

    print(f"画像数: {len(image_paths)}")
    print("操作: 左ドラッグ=塗る / 右ドラッグ=消す / S=保存 / Z=アンドゥ / C=クリア / Q=終了")

    idx = 0
    while idx < len(image_paths):
        img_path = image_paths[idx]
        stem     = os.path.splitext(os.path.basename(img_path))[0]
        out_path = os.path.join(output_dir, stem + '.png')

        img_pil = Image.open(img_path).convert('RGB')
        w, h    = img_pil.size
        img_np  = np.array(img_pil)

        # 既存マスクがあれば読み込む
        if os.path.isfile(out_path):
            mask = np.array(Image.open(out_path).convert('L'))
            print(f"[{idx+1}/{len(image_paths)}] {stem} — 既存マスクを読み込み")
        else:
            mask = np.zeros((h, w), dtype=np.uint8)
            print(f"[{idx+1}/{len(image_paths)}] {stem} — 新規")

        state = {
            'brush_size': brush_size,
            'drawing'   : False,
            'erasing'   : False,
            'history'   : [mask.copy()],
        }
        saved = {'done': False, 'quit': False}

        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        fig.canvas.manager.set_window_title(f"{stem} ({idx+1}/{len(image_paths)})")

        axes[0].imshow(img_np)
        axes[0].set_title('元画像', fontsize=11)
        axes[0].axis('off')

        overlay = img_np.copy().astype(float)
        overlay[mask > 128] = overlay[mask > 128] * 0.5 + np.array([255, 0, 0]) * 0.5
        mask_disp = axes[1].imshow(overlay.astype(np.uint8))
        axes[1].set_title(
            f'マスク編集  ブラシサイズ: {state["brush_size"]}px\n'
            '左ドラッグ=塗る 右ドラッグ=消す S=保存 Z=アンドゥ C=クリア',
            fontsize=10
        )
        axes[1].axis('off')
        plt.tight_layout()

        def update_display():
            overlay = img_np.copy().astype(float)
            overlay[mask > 128] = overlay[mask > 128] * 0.5 + np.array([255, 0, 0]) * 0.5
            mask_disp.set_data(overlay.astype(np.uint8))
            axes[1].set_title(
                f'マスク編集  ブラシサイズ: {state["brush_size"]}px\n'
                '左ドラッグ=塗る 右ドラッグ=消す S=保存 Z=アンドゥ C=クリア',
                fontsize=10
            )
            fig.canvas.draw_idle()

        def paint(x, y, erase=False):
            xi, yi = int(round(x)), int(round(y))
            bs = state['brush_size']
            y1 = max(0, yi - bs); y2 = min(h, yi + bs + 1)
            x1 = max(0, xi - bs); x2 = min(w, xi + bs + 1)
            yy, xx = np.ogrid[y1:y2, x1:x2]
            circle = (yy - yi) ** 2 + (xx - xi) ** 2 <= bs ** 2
            mask[y1:y2, x1:x2][circle] = 0 if erase else 255

        def on_press(event):
            if event.inaxes != axes[1]:
                return
            state['history'].append(mask.copy())
            if len(state['history']) > 30:
                state['history'].pop(0)
            if event.button == 1:
                state['drawing'] = True
                state['erasing'] = False
                paint(event.xdata, event.ydata, erase=False)
            elif event.button == 3:
                state['erasing'] = True
                state['drawing'] = False
                paint(event.xdata, event.ydata, erase=True)
            update_display()

        def on_motion(event):
            if event.inaxes != axes[1]:
                return
            if state['drawing']:
                paint(event.xdata, event.ydata, erase=False)
                update_display()
            elif state['erasing']:
                paint(event.xdata, event.ydata, erase=True)
                update_display()

        def on_release(event):
            state['drawing'] = False
            state['erasing'] = False

        def on_scroll(event):
            if event.button == 'up':
                state['brush_size'] = min(50, state['brush_size'] + 2)
            elif event.button == 'down':
                state['brush_size'] = max(1, state['brush_size'] - 2)
            update_display()

        def on_key(event):
            nonlocal mask
            if event.key == 's':
                Image.fromarray(mask).save(out_path)
                print(f"  保存: {out_path}")
                saved['done'] = True
                plt.close(fig)
            elif event.key == 'q':
                saved['quit'] = True
                plt.close(fig)
            elif event.key == 'z':
                if len(state['history']) > 1:
                    state['history'].pop()
                    mask[:] = state['history'][-1]
                    update_display()
            elif event.key == 'c':
                state['history'].append(mask.copy())
                mask[:] = 0
                update_display()

        fig.canvas.mpl_connect('button_press_event',   on_press)
        fig.canvas.mpl_connect('motion_notify_event',  on_motion)
        fig.canvas.mpl_connect('button_release_event', on_release)
        fig.canvas.mpl_connect('scroll_event',         on_scroll)
        fig.canvas.mpl_connect('key_press_event',      on_key)

        plt.show()

        if saved['quit']:
            print("終了します。")
            break
        idx += 1

    print("アノテーション完了！")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='UNet用マスクアノテーションツール')
    parser.add_argument('--images',      required=True, help='画像ディレクトリ')
    parser.add_argument('--output',      required=True, help='マスク保存先')
    parser.add_argument('--brush-size',  type=int, default=10, help='初期ブラシサイズ（px）')
    args = parser.parse_args()
    annotate_images(args.images, args.output, args.brush_size)