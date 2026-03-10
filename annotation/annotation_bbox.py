# annotate_bbox.py
# バウンディングボックスアノテーションツール（Faster R-CNN用）
#
# 使い方:
#   python annotate_bbox.py --images dataset/images --output dataset/annotations
#
# 操作:
#   ドラッグ    : ボックスを描く
#   右クリック  : 最後のボックスを削除
#   S キー      : 保存して次の画像へ
#   Q キー      : 終了

import os
import sys
import glob
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Button


def annotate_images(images_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
    image_paths = sorted(
        p for ext in exts for p in glob.glob(os.path.join(images_dir, ext))
    )
    if not image_paths:
        print(f"画像が見つかりません: {images_dir}")
        sys.exit(1)

    print(f"画像数: {len(image_paths)}")
    print("操作: ドラッグ=ボックス描画 / 右クリック=最後を削除 / S=保存して次へ / Q=終了")

    idx = 0
    while idx < len(image_paths):
        img_path = image_paths[idx]
        stem = os.path.splitext(os.path.basename(img_path))[0]
        json_path = os.path.join(output_dir, stem + '.json')

        # 既存アノテーション読み込み
        if os.path.isfile(json_path):
            with open(json_path) as f:
                data = json.load(f)
            boxes = data.get('boxes', [])
            print(f"[{idx+1}/{len(image_paths)}] {stem} — 既存 {len(boxes)} ボックス")
        else:
            boxes = []
            print(f"[{idx+1}/{len(image_paths)}] {stem} — 新規")

        from PIL import Image as PILImage
        img = np.array(PILImage.open(img_path).convert('RGB'))

        fig, ax = plt.subplots(figsize=(14, 9))
        fig.canvas.manager.set_window_title(f"{stem}  ({idx+1}/{len(image_paths)})")
        ax.imshow(img)

        # 既存ボックスを描画
        rect_patches = []
        for b in boxes:
            r = mpatches.Rectangle(
                (b['x1'], b['y1']), b['x2'] - b['x1'], b['y2'] - b['y1'],
                linewidth=2, edgecolor='red', facecolor='none'
            )
            ax.add_patch(r)
            rect_patches.append(r)

        state = {'x0': None, 'y0': None, 'cur_rect': None}
        saved = {'done': False, 'quit': False}

        def update_title():
            ax.set_title(
                f"ボックス数: {len(boxes)}  |  ドラッグ=描画  右クリック=削除  S=保存  Q=終了",
                fontsize=11
            )
            fig.canvas.draw_idle()

        update_title()

        def on_press(event):
            if event.inaxes != ax:
                return
            if event.button == 1:
                state['x0'] = event.xdata
                state['y0'] = event.ydata
                r = mpatches.Rectangle(
                    (event.xdata, event.ydata), 0, 0,
                    linewidth=2, edgecolor='lime', facecolor='none', linestyle='--'
                )
                ax.add_patch(r)
                state['cur_rect'] = r
                fig.canvas.draw_idle()
            elif event.button == 3:
                if boxes:
                    boxes.pop()
                    if rect_patches:
                        rect_patches.pop().remove()
                    update_title()

        def on_motion(event):
            if event.inaxes != ax or state['x0'] is None:
                return
            r = state['cur_rect']
            if r is None:
                return
            x0, y0 = state['x0'], state['y0']
            x1, y1 = event.xdata, event.ydata
            r.set_xy((min(x0, x1), min(y0, y1)))
            r.set_width(abs(x1 - x0))
            r.set_height(abs(y1 - y0))
            fig.canvas.draw_idle()

        def on_release(event):
            if event.button != 1 or state['x0'] is None:
                return
            r = state['cur_rect']
            if r is None:
                return
            x0, y0 = state['x0'], state['y0']
            x1, y1 = event.xdata, event.ydata
            if abs(x1 - x0) > 3 and abs(y1 - y0) > 3:
                box = {
                    'x1': float(min(x0, x1)),
                    'y1': float(min(y0, y1)),
                    'x2': float(max(x0, x1)),
                    'y2': float(max(y0, y1)),
                }
                boxes.append(box)
                # 確定ボックスを赤で再描画
                r.set_edgecolor('red')
                r.set_linestyle('-')
                rect_patches.append(r)
            else:
                r.remove()
            state['x0'] = state['y0'] = state['cur_rect'] = None
            update_title()

        def on_key(event):
            if event.key == 's':
                with open(json_path, 'w') as f:
                    json.dump({
                        'image': os.path.basename(img_path),
                        'boxes': boxes
                    }, f, indent=2)
                print(f"  保存: {json_path}  ({len(boxes)} ボックス)")
                saved['done'] = True
                plt.close(fig)
            elif event.key == 'q':
                saved['quit'] = True
                plt.close(fig)

        fig.canvas.mpl_connect('button_press_event',   on_press)
        fig.canvas.mpl_connect('motion_notify_event',  on_motion)
        fig.canvas.mpl_connect('button_release_event', on_release)
        fig.canvas.mpl_connect('key_press_event',      on_key)

        plt.tight_layout()
        plt.show()

        if saved['quit']:
            print("終了します。")
            break
        idx += 1

    print("アノテーション完了！")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='バウンディングボックスアノテーションツール')
    parser.add_argument('--images', required=True, help='画像ディレクトリ')
    parser.add_argument('--output', required=True, help='アノテーション保存先')
    args = parser.parse_args()
    annotate_images(args.images, args.output)