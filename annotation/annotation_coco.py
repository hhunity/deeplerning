# annotate_coco.py
# COCOフォーマット対応アノテーションツール
# バウンディングボックス・セグメンテーションマスク・点（CSRNet用）に対応
#
# 【使い方】
#   python annotate_coco.py --images dataset/images --output dataset/annotations.json
#                           --mode bbox
#   python annotate_coco.py --images dataset/images --output dataset/annotations.json
#                           --mode point
#
# 【操作（bboxモード）】
#   ドラッグ    : ボックスを描く
#   右クリック  : 最後のボックスを削除
#   S キー      : 保存して次の画像へ
#   Q キー      : 終了
#
# 【操作（pointモード・CSRNet用）】
#   左クリック  : 点を追加
#   右クリック  : 最後の点を削除
#   S キー      : 保存して次の画像へ
#   Q キー      : 終了
#
# 【出力COCOフォーマット】
# {
#   "info": {...},
#   "images": [{"id": 1, "file_name": "img.jpg", "width": 640, "height": 480}],
#   "annotations": [
#     {"id": 1, "image_id": 1, "category_id": 1,
#      "bbox": [x, y, w, h],           ← bboxモード
#      "point": [x, y],                ← pointモード（CSRNet用・独自拡張）
#      "area": 100, "iscrowd": 0}
#   ],
#   "categories": [{"id": 1, "name": "object"}]
# }

import os
import sys
import glob
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from datetime import datetime


def load_or_create_coco(output_path, category_name):
    """既存のCOCO JSONを読み込む、なければ新規作成"""
    if os.path.isfile(output_path):
        with open(output_path) as f:
            coco = json.load(f)
        print(f"[INFO] 既存アノテーションを読み込み: {len(coco['images'])} 画像")
    else:
        coco = {
            "info": {
                "description": "Custom Dataset",
                "version"    : "1.0",
                "year"       : datetime.now().year,
                "date_created": datetime.now().strftime("%Y/%m/%d"),
            },
            "images"     : [],
            "annotations": [],
            "categories" : [{"id": 1, "name": category_name, "supercategory": "object"}],
        }
    return coco


def get_image_entry(coco, file_name, w, h):
    """画像エントリを取得（なければ追加）"""
    for img in coco['images']:
        if img['file_name'] == file_name:
            return img
    new_id = max([img['id'] for img in coco['images']], default=0) + 1
    entry = {"id": new_id, "file_name": file_name, "width": w, "height": h}
    coco['images'].append(entry)
    return entry


def get_annotations_for_image(coco, image_id):
    """指定画像のアノテーションを取得"""
    return [a for a in coco['annotations'] if a['image_id'] == image_id]


def remove_annotations_for_image(coco, image_id):
    """指定画像のアノテーションを削除"""
    coco['annotations'] = [a for a in coco['annotations'] if a['image_id'] != image_id]


def next_ann_id(coco):
    """次のアノテーションIDを取得"""
    return max([a['id'] for a in coco['annotations']], default=0) + 1


def save_coco(coco, output_path):
    """COCO JSONを保存"""
    with open(output_path, 'w') as f:
        json.dump(coco, f, indent=2, ensure_ascii=False)


# ==================== bboxモード ====================

def annotate_bbox(image_paths, coco, output_path):
    """バウンディングボックスアノテーション"""

    for idx, img_path in enumerate(image_paths):
        file_name = os.path.basename(img_path)
        img_pil   = Image.open(img_path).convert('RGB')
        w, h      = img_pil.size
        img_np    = np.array(img_pil)

        img_entry = get_image_entry(coco, file_name, w, h)
        image_id  = img_entry['id']

        # 既存アノテーション読み込み
        existing = get_annotations_for_image(coco, image_id)
        boxes = [[a['bbox'][0], a['bbox'][1],
                  a['bbox'][0] + a['bbox'][2],
                  a['bbox'][1] + a['bbox'][3]] for a in existing]

        print(f"[{idx+1}/{len(image_paths)}] {file_name}  既存: {len(boxes)} ボックス")

        saved = {'done': False, 'quit': False}
        state = {'x0': None, 'y0': None, 'cur_rect': None}

        fig, ax = plt.subplots(figsize=(14, 9))
        fig.canvas.manager.set_window_title(f"{file_name} ({idx+1}/{len(image_paths)})")
        ax.imshow(img_np)

        rect_patches = []
        for b in boxes:
            x1, y1, x2, y2 = b
            r = mpatches.Rectangle((x1, y1), x2-x1, y2-y1,
                                    linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(r)
            rect_patches.append(r)

        def update_title():
            ax.set_title(f"ボックス数: {len(boxes)}  |  ドラッグ=描画  右クリック=削除  S=保存  Q=終了",
                         fontsize=11)
            fig.canvas.draw_idle()

        update_title()

        def on_press(event):
            if event.inaxes != ax:
                return
            if event.button == 1:
                state['x0'], state['y0'] = event.xdata, event.ydata
                r = mpatches.Rectangle((event.xdata, event.ydata), 0, 0,
                                        linewidth=2, edgecolor='lime',
                                        facecolor='none', linestyle='--')
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
            r.set_xy((min(x0, event.xdata), min(y0, event.ydata)))
            r.set_width(abs(event.xdata - x0))
            r.set_height(abs(event.ydata - y0))
            fig.canvas.draw_idle()

        def on_release(event):
            if event.button != 1 or state['x0'] is None:
                return
            r = state['cur_rect']
            if r is None:
                return
            x0, y0 = state['x0'], state['y0']
            x1, y1 = event.xdata, event.ydata
            if abs(x1-x0) > 3 and abs(y1-y0) > 3:
                boxes.append([min(x0,x1), min(y0,y1), max(x0,x1), max(y0,y1)])
                r.set_edgecolor('red')
                r.set_linestyle('-')
                rect_patches.append(r)
            else:
                r.remove()
            state['x0'] = state['y0'] = state['cur_rect'] = None
            update_title()

        def on_key(event):
            if event.key == 's':
                # 既存アノテーションを削除して書き直し
                remove_annotations_for_image(coco, image_id)
                ann_id = next_ann_id(coco)
                for b in boxes:
                    x1, y1, x2, y2 = b
                    bw, bh = x2 - x1, y2 - y1
                    coco['annotations'].append({
                        "id"         : ann_id,
                        "image_id"   : image_id,
                        "category_id": 1,
                        "bbox"       : [x1, y1, bw, bh],  # COCO形式: x,y,w,h
                        "area"       : bw * bh,
                        "iscrowd"    : 0,
                    })
                    ann_id += 1
                save_coco(coco, output_path)
                print(f"  保存: {len(boxes)} ボックス → {output_path}")
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


# ==================== pointモード（CSRNet用） ====================

def annotate_point(image_paths, coco, output_path):
    """点アノテーション（CSRNet用）"""

    for idx, img_path in enumerate(image_paths):
        file_name = os.path.basename(img_path)
        img_pil   = Image.open(img_path).convert('RGB')
        w, h      = img_pil.size
        img_np    = np.array(img_pil)

        img_entry = get_image_entry(coco, file_name, w, h)
        image_id  = img_entry['id']

        existing = get_annotations_for_image(coco, image_id)
        points = [[a['point'][0], a['point'][1]] for a in existing if 'point' in a]

        print(f"[{idx+1}/{len(image_paths)}] {file_name}  既存: {len(points)} 点")

        saved = {'done': False, 'quit': False}

        fig, ax = plt.subplots(figsize=(12, 8))
        fig.canvas.manager.set_window_title(f"{file_name} ({idx+1}/{len(image_paths)})")
        ax.imshow(img_np)

        scatter = ax.scatter(
            [p[0] for p in points] if points else [],
            [p[1] for p in points] if points else [],
            c='red', s=20, marker='+'
        )

        def update_scatter():
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            scatter.set_offsets(np.c_[xs, ys] if points else np.empty((0, 2)))
            ax.set_title(f"点数: {len(points)}  |  左クリック=追加  右クリック=削除  S=保存  Q=終了",
                         fontsize=11)
            fig.canvas.draw_idle()

        update_scatter()

        def on_click(event):
            if event.inaxes != ax:
                return
            if event.button == 1:
                points.append([event.xdata, event.ydata])
            elif event.button == 3:
                if points:
                    points.pop()
            update_scatter()

        def on_key(event):
            if event.key == 's':
                remove_annotations_for_image(coco, image_id)
                ann_id = next_ann_id(coco)
                for p in points:
                    coco['annotations'].append({
                        "id"         : ann_id,
                        "image_id"   : image_id,
                        "category_id": 1,
                        "point"      : [p[0], p[1]],  # CSRNet用独自拡張
                        "bbox"       : [p[0]-1, p[1]-1, 2, 2],  # ダミーbbox
                        "area"       : 4,
                        "iscrowd"    : 0,
                    })
                    ann_id += 1
                save_coco(coco, output_path)
                print(f"  保存: {len(points)} 点 → {output_path}")
                saved['done'] = True
                plt.close(fig)
            elif event.key == 'q':
                saved['quit'] = True
                plt.close(fig)

        fig.canvas.mpl_connect('button_press_event', on_click)
        fig.canvas.mpl_connect('key_press_event',    on_key)
        plt.tight_layout()
        plt.show()

        if saved['quit']:
            print("終了します。")
            break


# ==================== メイン ====================

def main():
    parser = argparse.ArgumentParser(description='COCOフォーマット対応アノテーションツール')
    parser.add_argument('--images',   required=True, help='画像ディレクトリ')
    parser.add_argument('--output',   required=True, help='COCO JSON保存先 (例: annotations.json)')
    parser.add_argument('--mode',     default='bbox', choices=['bbox', 'point'],
                        help='bbox: Faster R-CNN用  point: CSRNet用 (default: bbox)')
    parser.add_argument('--category', default='object', help='クラス名 (default: object)')
    args = parser.parse_args()

    exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
    image_paths = sorted(
        p for ext in exts for p in glob.glob(os.path.join(args.images, ext))
    )
    if not image_paths:
        print(f"画像が見つかりません: {args.images}")
        sys.exit(1)

    print(f"[INFO] 画像数: {len(image_paths)}  モード: {args.mode}")

    coco = load_or_create_coco(args.output, args.category)

    if args.mode == 'bbox':
        annotate_bbox(image_paths, coco, args.output)
    elif args.mode == 'point':
        annotate_point(image_paths, coco, args.output)

    print(f"\n完了！  総アノテーション数: {len(coco['annotations'])}")
    print(f"保存先: {args.output}")


if __name__ == '__main__':
    main()