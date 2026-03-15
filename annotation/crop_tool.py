# crop_tool.py
# 窪みの固定位置を登録して、画像から切り出すツール
#
# 【使い方】
#   ① 位置登録: python crop_tool.py register 画像.jpg --config cavities.json
#   ② 切り出し: python crop_tool.py crop 画像フォルダ --config cavities.json --output crops
#
# 【操作（registerモード）】
#   左クリック+ドラッグ : 窪みの範囲を指定
#   S キー              : 保存して終了
#   Z キー              : 最後の窪みを削除
#   Q キー              : 終了

import os
import sys
import glob
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image


# ==================== 位置登録 ====================

def register_cavities(img_path, config_path, margin, fixed_size=None):
    """
    画像上で窪みの位置を登録する
    fixed_size=(w,h): クリックした場所を中心に固定サイズで配置
    fixed_size=None : ドラッグで自由に範囲指定
    """
    img_pil = Image.open(img_path).convert('RGB')
    img_np  = np.array(img_pil)
    w, h    = img_pil.size

    # 既存config読み込み
    if os.path.isfile(config_path):
        with open(config_path) as f:
            config = json.load(f)
        cavities = config.get('cavities', [])
        print(f"[INFO] 既存の窪み: {len(cavities)} 個")
    else:
        cavities = []

    if fixed_size:
        print(f"[INFO] 固定サイズモード: {fixed_size[0]}×{fixed_size[1]}px  左クリック=配置")
    else:
        print(f"[INFO] 自由モード: ドラッグで範囲指定")

    state = {'x0': None, 'y0': None, 'cur_rect': None, 'preview': None}
    saved = {'done': False}

    fig, ax = plt.subplots(figsize=(14, 9))
    fig.canvas.manager.set_window_title('窪み位置登録  S=保存  Z=削除  Q=終了')
    ax.imshow(img_np)
    ax.axis('off')

    rect_patches = []

    # 既存の窪みを表示
    for i, c in enumerate(cavities):
        x1, y1, x2, y2 = c['x1'], c['y1'], c['x2'], c['y2']
        r = mpatches.Rectangle((x1, y1), x2-x1, y2-y1,
                                linewidth=2, edgecolor='lime', facecolor='none')
        ax.add_patch(r)
        ax.text(x1+2, y1+12, str(i+1), color='lime', fontsize=8, fontweight='bold')
        rect_patches.append(r)

    plt.tight_layout()

    def update_title():
        if fixed_size:
            hint = f'左クリック=配置（{fixed_size[0]}×{fixed_size[1]}px）  Z=最後を削除  S=保存  Q=終了'
        else:
            hint = 'ドラッグ=範囲指定  Z=最後を削除  S=保存  Q=終了'
        ax.set_title(f'窪み数: {len(cavities)}  |  {hint}', fontsize=11)
        fig.canvas.draw_idle()

    update_title()

    def add_cavity(x1, y1, x2, y2):
        cavities.append({
            'x1': int(x1), 'y1': int(y1),
            'x2': int(x2), 'y2': int(y2),
            'id': len(cavities) + 1
        })
        r = mpatches.Rectangle((x1, y1), x2-x1, y2-y1,
                                linewidth=2, edgecolor='lime', facecolor='none')
        ax.add_patch(r)
        ax.text(x1+2, y1+12, str(len(cavities)), color='lime', fontsize=8, fontweight='bold')
        rect_patches.append(r)
        update_title()

    # 固定サイズモード: マウス移動でプレビュー表示
    def on_motion(event):
        if event.inaxes != ax:
            return
        if fixed_size:
            fw, fh = fixed_size[0], fixed_size[1]
            px = event.xdata - fw / 2
            py = event.ydata - fh / 2
            if state['preview'] is None:
                r = mpatches.Rectangle((px, py), fw, fh,
                                       linewidth=1, edgecolor='yellow',
                                       facecolor='none', linestyle='--')
                ax.add_patch(r)
                state['preview'] = r
            else:
                state['preview'].set_xy((px, py))
                state['preview'].set_width(fw)
                state['preview'].set_height(fh)
            fig.canvas.draw_idle()
        else:
            # ドラッグ中の枠を更新
            if state['x0'] is None or state['cur_rect'] is None:
                return
            x0, y0 = state['x0'], state['y0']
            state['cur_rect'].set_xy((min(x0, event.xdata), min(y0, event.ydata)))
            state['cur_rect'].set_width(abs(event.xdata - x0))
            state['cur_rect'].set_height(abs(event.ydata - y0))
            fig.canvas.draw_idle()

    def on_press(event):
        if event.inaxes != ax or event.button != 1:
            return
        if fixed_size:
            fw, fh = fixed_size[0], fixed_size[1]
            x1 = max(0, event.xdata - fw / 2)
            y1 = max(0, event.ydata - fh / 2)
            x2 = min(w, x1 + fw)
            y2 = min(h, y1 + fh)
            add_cavity(x1, y1, x2, y2)
        else:
            # ドラッグ開始
            state['x0'], state['y0'] = event.xdata, event.ydata
            r = mpatches.Rectangle((event.xdata, event.ydata), 0, 0,
                                    linewidth=2, edgecolor='yellow',
                                    facecolor='none', linestyle='--')
            ax.add_patch(r)
            state['cur_rect'] = r
            fig.canvas.draw_idle()

    def on_release(event):
        if fixed_size or event.button != 1 or state['x0'] is None:
            return
        r = state['cur_rect']
        if r is None:
            return
        x0, y0 = state['x0'], state['y0']
        x1, y1 = min(x0, event.xdata), min(y0, event.ydata)
        x2, y2 = max(x0, event.xdata), max(y0, event.ydata)
        r.remove()
        if abs(x2-x1) > 5 and abs(y2-y1) > 5:
            add_cavity(x1, y1, x2, y2)
        state['x0'] = state['y0'] = state['cur_rect'] = None

    def on_key(event):
        if event.key == 's':
            config = {
                'image_size' : [w, h],
                'margin'     : margin,
                'fixed_size' : list(fixed_size) if fixed_size else None,
                'cavities'   : cavities,
            }
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            print(f"[INFO] 保存: {config_path}  ({len(cavities)} 個の窪み)")
            saved['done'] = True
            plt.close(fig)
        elif event.key == 'z':
            if cavities:
                cavities.pop()
                if rect_patches:
                    rect_patches.pop().remove()
                update_title()
        elif event.key == 'q':
            plt.close(fig)

    fig.canvas.mpl_connect('button_press_event',   on_press)
    fig.canvas.mpl_connect('motion_notify_event',  on_motion)
    fig.canvas.mpl_connect('button_release_event', on_release)
    fig.canvas.mpl_connect('key_press_event',      on_key)
    plt.show()


# ==================== 切り出し ====================

def crop_images(images_dir, config_path, output_dir, label_mode, crop_size=None):
    """
    登録済みの窪み位置を使って全画像から切り出す

    label_mode='manual' : ラベルを手動で入力（分類学習用）
    label_mode='auto'   : ラベルなし（推論用）
    crop_size=(w, h)    : 指定サイズにリサイズして保存（Noneなら切り出したまま）
    """
    if not os.path.isfile(config_path):
        print(f"[ERROR] configが見つかりません: {config_path}")
        print("        先に register モードで窪みを登録してください")
        sys.exit(1)

    with open(config_path) as f:
        config = json.load(f)

    cavities = config['cavities']
    margin   = config.get('margin', 10)
    size_str = f'{crop_size[0]}×{crop_size[1]}px' if crop_size else 'そのまま'
    print(f"[INFO] 窪み数: {len(cavities)}  マージン: {margin}px  出力サイズ: {size_str}")

    exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
    image_paths = sorted(
        p for ext in exts for p in glob.glob(os.path.join(images_dir, ext))
    )
    if not image_paths:
        print(f"[ERROR] 画像が見つかりません: {images_dir}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    total = 0
    for img_path in image_paths:
        stem    = os.path.splitext(os.path.basename(img_path))[0]
        img_pil = Image.open(img_path).convert('RGB')
        iw, ih  = img_pil.size

        for cavity in cavities:
            cid  = cavity['id']
            x1   = max(0,  cavity['x1'] - margin)
            y1   = max(0,  cavity['y1'] - margin)
            x2   = min(iw, cavity['x2'] + margin)
            y2   = min(ih, cavity['y2'] + margin)
            crop = img_pil.crop((x1, y1, x2, y2))

            # サイズ指定があればリサイズ
            if crop_size:
                crop = crop.resize(crop_size, Image.BILINEAR)

            save_path = os.path.join(output_dir, f'{stem}_c{cid:02d}.jpg')
            crop.save(save_path)
            total += 1

    print(f"[INFO] 切り出し完了: {total} 枚 → {output_dir}")

    if label_mode == 'manual':
        print("\n次のステップ:")
        print(f"  切り出した画像を以下のフォルダに振り分けてください:")
        print(f"  {output_dir}/0個/  → 結晶が0個の窪み")
        print(f"  {output_dir}/1個/  → 結晶が1個の窪み")
        print(f"  {output_dir}/2個/  → 結晶が2個の窪み")
        print(f"  ...")
        print(f"\n振り分け後:")
        print(f"  python resnet_classify.py train --data {output_dir}")


# ==================== 推論用切り出し ====================

def crop_for_predict(img_path, config_path, output_dir, margin_override=None, crop_size=None):
    """推論時に窪みを切り出す（ラベルなし）"""
    with open(config_path) as f:
        config = json.load(f)

    cavities  = config['cavities']
    margin    = margin_override if margin_override is not None else config.get('margin', 10)
    crop_size = crop_size or config.get('crop_size')  # configに保存されていれば使う

    img_pil = Image.open(img_path).convert('RGB')
    iw, ih  = img_pil.size
    os.makedirs(output_dir, exist_ok=True)

    crops = []
    for cavity in cavities:
        cid  = cavity['id']
        x1   = max(0,  cavity['x1'] - margin)
        y1   = max(0,  cavity['y1'] - margin)
        x2   = min(iw, cavity['x2'] + margin)
        y2   = min(ih, cavity['y2'] + margin)
        crop = img_pil.crop((x1, y1, x2, y2))
        if crop_size:
            crop = crop.resize(crop_size, Image.BILINEAR)
        path = os.path.join(output_dir, f'cavity_{cid:02d}.jpg')
        crop.save(path)
        crops.append({'id': cid, 'path': path, 'bbox': (x1, y1, x2, y2)})

    return crops


# ==================== メイン ====================

def main():
    parser = argparse.ArgumentParser(description='窪み切り出しツール')
    sub = parser.add_subparsers(dest='mode', required=True)

    # register
    r = sub.add_parser('register', help='窪みの位置を登録')
    r.add_argument('image',        help='基準画像パス')
    r.add_argument('--config',     default='cavities.json', help='設定ファイル保存先')
    r.add_argument('--margin',     type=int, default=10,
                   help='切り出し時のマージン（px default:10）')
    r.add_argument('--fixed-size', default=None,
                   help='固定サイズ指定 幅x高さ（例: 80x80）\n'
                        '指定時: クリックで配置\n'
                        '省略時: ドラッグで自由に指定')

    # crop
    c = sub.add_parser('crop', help='全画像から窪みを切り出す')
    c.add_argument('images',      help='画像ディレクトリ')
    c.add_argument('--config',    default='cavities.json')
    c.add_argument('--output',    default='crops', help='切り出し先ディレクトリ')
    c.add_argument('--mode',      default='manual', choices=['manual', 'auto'],
                   help='manual=ラベル振り分け用  auto=推論用')
    c.add_argument('--crop-size', default=None,
                   help='出力サイズ 幅x高さ（例: 128x128）省略時はそのまま')

    args = parser.parse_args()

    if args.mode == 'register':
        fixed_size = None
        if args.fixed_size:
            try:
                fw, fh     = args.fixed_size.lower().split('x')
                fixed_size = (int(fw), int(fh))
            except:
                print(f"[ERROR] --fixed-size の形式が正しくありません: {args.fixed_size}")
                print("        例: --fixed-size 80x80")
                sys.exit(1)
        register_cavities(args.image, args.config, args.margin, fixed_size)
    elif args.mode == 'crop':
        # --crop-size のパース
        crop_size = None
        if args.crop_size:
            try:
                cw, ch    = args.crop_size.lower().split('x')
                crop_size = (int(cw), int(ch))
            except:
                print(f"[ERROR] --crop-size の形式が正しくありません: {args.crop_size}")
                print("        例: --crop-size 128x128")
                sys.exit(1)
        crop_images(args.images, args.config, args.output, args.mode, crop_size)


if __name__ == '__main__':
    main()