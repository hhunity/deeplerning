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


def load_or_create_coco(output_path, categories):
    """
    既存のCOCO JSONを読み込む、なければ新規作成
    categories: [{"id":1,"name":"0個"}, ...] のリスト
    """
    if os.path.isfile(output_path):
        with open(output_path) as f:
            coco = json.load(f)
        print(f"[INFO] 既存アノテーションを読み込み: {len(coco['images'])} 画像")
        # categoriesが空の場合は引数で上書き
        if not coco.get('categories') and categories:
            coco['categories'] = categories
            print(f"[INFO] categoriesを設定: {[c['name'] for c in categories]}")
    else:
        coco = {
            "info": {
                "description" : "Custom Dataset",
                "version"     : "1.0",
                "year"        : datetime.now().year,
                "date_created": datetime.now().strftime("%Y/%m/%d"),
            },
            "images"     : [],
            "annotations": [],
            "categories" : categories,
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


# ==================== bbox-classifyモード ====================

def annotate_bbox_classify(image_paths, coco, output_path):
    """
    バウンディングボックス + クラス選択アノテーション
    ドラッグでbox描画 → 数字キーでクラス指定
    """
    categories = coco.get('categories', [])
    if not categories:
        print("[ERROR] JSONにcategoriesがありません。")
        print("        先に faster_rcnn_classify.py template を実行してください")
        return

    cat_map   = {c['id']: c['name'] for c in categories}
    cat_ids   = sorted(cat_map.keys())
    class_str = '  '.join(f"{i+1}={cat_map[cid]}" for i, cid in enumerate(cat_ids))
    print(f"[INFO] クラス: {class_str}")

    # クラスごとの色
    COLORS = ['#ff4444','#44cc44','#ff9900','#cc44ff','#44ccff','#ffff44']

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
                  a['bbox'][0]+a['bbox'][2],
                  a['bbox'][1]+a['bbox'][3],
                  a['category_id']] for a in existing]  # [x1,y1,x2,y2,cat_id]

        print(f"[{idx+1}/{len(image_paths)}] {file_name}  既存: {len(boxes)} ボックス")

        saved = {'done': False, 'quit': False}
        state = {
            'x0': None, 'y0': None,
            'cur_rect': None,
            'pending': None,   # クラス未選択のbox
            'pending_rect': None,
        }

        fig, ax = plt.subplots(figsize=(14, 9))
        fig.canvas.manager.set_window_title(
            f"{file_name} ({idx+1}/{len(image_paths)})"
        )
        ax.imshow(img_np)
        ax.axis('off')
        rect_patches = []

        def color_of(cat_id):
            try:
                return COLORS[cat_ids.index(cat_id) % len(COLORS)]
            except:
                return '#ffffff'

        def redraw_boxes():
            for r in rect_patches:
                r.remove()
            rect_patches.clear()
            for b in boxes:
                x1, y1, x2, y2, cat_id = b
                c = color_of(cat_id)
                r = mpatches.Rectangle(
                    (x1, y1), x2-x1, y2-y1,
                    linewidth=2, edgecolor=c, facecolor=c, alpha=0.15
                )
                ax.add_patch(r)
                ax.text(x1+2, y1+13, cat_map.get(cat_id, '?'),
                        color='white', fontsize=8, fontweight='bold',
                        bbox=dict(facecolor=c, alpha=0.8, pad=2, edgecolor='none'))
                rect_patches.append(r)

        def update_title():
            if state['pending'] is not None:
                key_hint = '  '.join(
                    f"[{i+1}]{cat_map[cid]}" for i, cid in enumerate(cat_ids)
                )
                ax.set_title(
                    f'クラスを選択してください → {key_hint}  [Del]=キャンセル',
                    fontsize=11, color='orange'
                )
            else:
                ax.set_title(
                    f'ボックス数: {len(boxes)}  |  ドラッグ=描画  右クリック=削除  '
                    f'S=保存  Q=終了\nクラス: {class_str}',
                    fontsize=10
                )
            fig.canvas.draw_idle()

        redraw_boxes()
        update_title()

        def on_press(event):
            if event.inaxes != ax or state['pending'] is not None:
                return
            if event.button == 1:
                state['x0'], state['y0'] = event.xdata, event.ydata
                r = mpatches.Rectangle(
                    (event.xdata, event.ydata), 0, 0,
                    linewidth=2, edgecolor='yellow',
                    facecolor='none', linestyle='--'
                )
                ax.add_patch(r)
                state['cur_rect'] = r
                fig.canvas.draw_idle()
            elif event.button == 3:
                if boxes:
                    boxes.pop()
                    redraw_boxes()
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
            x1 = min(x0, event.xdata); y1 = min(y0, event.ydata)
            x2 = max(x0, event.xdata); y2 = max(y0, event.ydata)
            if abs(x2-x1) > 3 and abs(y2-y1) > 3:
                # クラス選択待ち状態にする
                r.set_edgecolor('yellow')
                r.set_linestyle('-')
                state['pending']      = [x1, y1, x2, y2]
                state['pending_rect'] = r
            else:
                r.remove()
            state['x0'] = state['y0'] = state['cur_rect'] = None
            update_title()

        def on_key(event):
            # クラス選択中
            if state['pending'] is not None:
                if event.key == 'delete' or event.key == 'escape':
                    # キャンセル
                    state['pending_rect'].remove()
                    state['pending'] = state['pending_rect'] = None
                    update_title()
                    return
                # 数字キーでクラス選択
                try:
                    key_num = int(event.key)
                except:
                    return
                if 1 <= key_num <= len(cat_ids):
                    cat_id = cat_ids[key_num - 1]
                    x1, y1, x2, y2 = state['pending']
                    # pendingのrectを正式な色に更新
                    state['pending_rect'].remove()
                    boxes.append([x1, y1, x2, y2, cat_id])
                    state['pending'] = state['pending_rect'] = None
                    redraw_boxes()
                    update_title()
                return

            # 通常操作
            if event.key == 's':
                remove_annotations_for_image(coco, image_id)
                ann_id = next_ann_id(coco)
                for b in boxes:
                    x1, y1, x2, y2, cat_id = b
                    bw, bh = x2-x1, y2-y1
                    coco['annotations'].append({
                        "id"         : ann_id,
                        "image_id"   : image_id,
                        "category_id": cat_id,
                        "bbox"       : [x1, y1, bw, bh],
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
    parser.add_argument('--images',     required=True, help='画像ディレクトリ')
    parser.add_argument('--output',     required=True, help='COCO JSON保存先 (例: annotations.json)')
    parser.add_argument('--mode',       default='bbox',
                        choices=['bbox', 'point', 'bbox-classify'],
                        help=('bbox         : Faster R-CNN用（クラス固定）\n'
                              'bbox-classify: クラス選択あり（分類版Faster R-CNN用）\n'
                              'point        : CSRNet用 (default: bbox)'))
    parser.add_argument('--category',   default='object',
                        help='クラス名（bboxモード用 default: object）')
    parser.add_argument('--categories', default=None,
                        help='クラス名をカンマ区切りで指定（bbox-classifyモード用）\n'
                             '例: "0個,1個,2個,3個"')
    args = parser.parse_args()

    exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
    image_paths = sorted(
        p for ext in exts for p in glob.glob(os.path.join(args.images, ext))
    )
    if not image_paths:
        print(f"画像が見つかりません: {args.images}")
        sys.exit(1)

    print(f"[INFO] 画像数: {len(image_paths)}  モード: {args.mode}")

    # categoriesリストを構築
    if args.mode == 'bbox-classify':
        if args.categories:
            names = [n.strip() for n in args.categories.split(',') if n.strip()]
            categories = [
                {"id": i+1, "name": name, "supercategory": "object"}
                for i, name in enumerate(names)
            ]
            print(f"[INFO] カテゴリ: {names}")
        else:
            # --categoriesが省略された場合はJSONから読む（既存ファイルがある場合）
            categories = []
    else:
        categories = [{"id": 1, "name": args.category, "supercategory": "object"}]

    coco = load_or_create_coco(args.output, categories)

    # bbox-classifyでcategoriesが空ならエラー
    if args.mode == 'bbox-classify' and not coco.get('categories'):
        print("[ERROR] --categories を指定してください")
        print('  例: --categories "0個,1個,2個,3個"')
        sys.exit(1)

    if args.mode == 'bbox':
        annotate_bbox(image_paths, coco, args.output)
    elif args.mode == 'bbox-classify':
        annotate_bbox_classify(image_paths, coco, args.output)
    elif args.mode == 'point':
        annotate_point(image_paths, coco, args.output)

    print(f"\n完了！  総アノテーション数: {len(coco['annotations'])}")
    print(f"保存先: {args.output}")


if __name__ == '__main__':
    main()