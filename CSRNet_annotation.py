# annotate.py

# 点アノテーションツール（クリックで点を打ち、JSONに保存）

# 

# 使い方:

# python annotate.py –images dataset/images –output dataset/annotations

# 

# 操作:

# 左クリック  : 点を追加

# 右クリック  : 直前の点を削除

# S キー      : 保存して次の画像へ

# Q キー      : 終了

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

```
exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
image_paths = sorted(
    p for ext in exts for p in glob.glob(os.path.join(images_dir, ext))
)
if not image_paths:
    print(f"画像が見つかりません: {images_dir}")
    sys.exit(1)

print(f"画像数: {len(image_paths)}")
print("操作: 左クリック=点追加 / 右クリック=最後の点削除 / S=保存して次へ / Q=終了")

idx = 0

while idx < len(image_paths):
    img_path = image_paths[idx]
    stem = os.path.splitext(os.path.basename(img_path))[0]
    json_path = os.path.join(output_dir, stem + '.json')

    # 既存アノテーションがあれば読み込む
    if os.path.isfile(json_path):
        with open(json_path) as f:
            data = json.load(f)
        points = data.get('points', [])
        print(f"[{idx+1}/{len(image_paths)}] {stem} — 既存アノテーション {len(points)} 点")
    else:
        points = []
        print(f"[{idx+1}/{len(image_paths)}] {stem} — 新規")

    import matplotlib
    matplotlib.use('TkAgg')  # GUI バックエンド
    from PIL import Image
    img = np.array(Image.open(img_path).convert('RGB'))

    fig, ax = plt.subplots(figsize=(12, 8))
    fig.canvas.manager.set_window_title(f"{stem}  ({idx+1}/{len(image_paths)})")
    ax.imshow(img)
    ax.set_title(f"点数: {len(points)}  |  S=保存して次へ  Q=終了  右クリック=最後の点を削除")

    scatter = ax.scatter(
        [p[0] for p in points] if points else [],
        [p[1] for p in points] if points else [],
        c='red', s=20, marker='+'
    )

    saved = {'done': False, 'quit': False}

    def update_scatter():
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        scatter.set_offsets(np.c_[xs, ys] if points else np.empty((0, 2)))
        ax.set_title(f"点数: {len(points)}  |  S=保存して次へ  Q=終了  右クリック=最後を削除")
        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes != ax:
            return
        if event.button == 1:   # 左クリック: 追加
            points.append([event.xdata, event.ydata])
            update_scatter()
        elif event.button == 3:  # 右クリック: 削除
            if points:
                points.pop()
                update_scatter()

    def on_key(event):
        if event.key == 's':
            with open(json_path, 'w') as f:
                json.dump({'points': points, 'image': os.path.basename(img_path)}, f, indent=2)
            print(f"  保存: {json_path}  ({len(points)} 点)")
            saved['done'] = True
            plt.close(fig)
        elif event.key == 'q':
            saved['quit'] = True
            plt.close(fig)

    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.tight_layout()
    plt.show()

    if saved['quit']:
        print("終了します。")
        break
    if saved['done']:
        idx += 1
    else:
        # ウィンドウを閉じただけ（保存なし）
        idx += 1

print("アノテーション完了！")
```

if **name** == ‘**main**’:
parser = argparse.ArgumentParser(description=‘点アノテーションツール’)
parser.add_argument(’–images’, required=True, help=‘画像ディレクトリ’)
parser.add_argument(’–output’, required=True, help=‘アノテーション保存先’)
args = parser.parse_args()
annotate_images(args.images, args.output)