# annotate_sam2.py
# SAM2を使った半自動マスクアノテーションツール
# SAM2で自動セグメント → クリックで選択/除外 → UNet用マスクとして保存
#
# 【インストール】
#   pip install torch torchvision
#   pip install git+https://github.com/facebookresearch/sam2.git
#   pip install matplotlib pillow numpy
#
# 【モデルダウンロード】
#   https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt
#
# 【使い方】
#   python annotate_sam2.py --images dataset/images --output dataset/masks
#                           --checkpoint sam2.1_hiera_small.pt --model-type small
#
# 【操作方法】
#   左クリック  : マスクを選択（緑）
#   右クリック  : マスクを除外（赤に戻る）
#   S キー      : 保存して次の画像へ
#   R キー      : SAM2を再実行（やり直し）
#   Q キー      : 終了

import os
import sys
import glob
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch


# SAM2のモデル設定ファイル名マッピング
MODEL_CONFIG_MAP = {
    'tiny'      : 'configs/sam2.1/sam2.1_hiera_t.yaml',
    'small'     : 'configs/sam2.1/sam2.1_hiera_s.yaml',
    'base_plus' : 'configs/sam2.1/sam2.1_hiera_b+.yaml',
    'large'     : 'configs/sam2.1/sam2.1_hiera_l.yaml',
}


def load_sam2(checkpoint, model_type, device, points_per_side=32):
    try:
        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    except ImportError:
        print('[ERROR] sam2がインストールされていません。')
        print('        pip install git+https://github.com/facebookresearch/sam2.git')
        raise

    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(
            f"チェックポイントが見つかりません: {checkpoint}\n"
            f"ダウンロード: https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt"
        )

    config = MODEL_CONFIG_MAP[model_type]
    sam2   = build_sam2(config, checkpoint, device=device)
    sam2.eval()

    print(f'[INFO] points_per_side: {points_per_side}')
    mask_generator = SAM2AutomaticMaskGenerator(
        model                  = sam2,
        points_per_side        = points_per_side,
        pred_iou_thresh        = 0.88,
        stability_score_thresh = 0.95,
        min_mask_region_area   = 50,
    )
    return mask_generator


def run_sam2(mask_generator, img_np):
    """SAM2で自動マスク生成"""
    print('[INFO] SAM2推論中...')
    masks = mask_generator.generate(img_np)
    print(f'[INFO] {len(masks)} 個のマスクを生成しました')
    return masks


def build_overlay(img_np, masks, selected):
    """
    マスクオーバーレイ画像を生成
    selected: set of mask indices that are selected (green)
    unselected: red
    """
    overlay = img_np.copy().astype(float)
    for i, m in enumerate(masks):
        if i in selected:
            color = np.array([0, 255, 0], dtype=float)    # 緑 = 選択済み
        else:
            color = np.array([255, 80, 80], dtype=float)  # 赤 = 未選択
        seg = m['segmentation']
        overlay[seg] = overlay[seg] * 0.4 + color * 0.6

    return overlay.astype(np.uint8)


def masks_to_binary(masks, selected, h, w):
    """選択済みマスクを合成して2値マスク画像を生成"""
    binary = np.zeros((h, w), dtype=np.uint8)
    for i in masks:
        if i in selected:
            binary[masks[i]['segmentation']] = 255
    return binary


def annotate_images(images_dir, output_dir, checkpoint, model_type, device,
                    min_area_ratio, max_area_ratio, min_ratio, max_ratio,
                    upscale, points_per_side):
    os.makedirs(output_dir, exist_ok=True)

    exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
    image_paths = sorted(
        p for ext in exts for p in glob.glob(os.path.join(images_dir, ext))
    )
    if not image_paths:
        print(f"画像が見つかりません: {images_dir}")
        sys.exit(1)

    print(f"[INFO] 画像数: {len(image_paths)}")
    print("操作: 左クリック=選択(緑) / 右クリック=除外 / S=保存 / R=再実行 / Q=終了")

    # SAM2読み込み
    mask_generator = load_sam2(checkpoint, model_type, device,
                                points_per_side=points_per_side)

    idx = 0
    while idx < len(image_paths):
        img_path = image_paths[idx]
        stem     = os.path.splitext(os.path.basename(img_path))[0]
        out_path = os.path.join(output_dir, stem + '.png')

        img_pil  = Image.open(img_path).convert('RGB')
        orig_w, orig_h = img_pil.size

        # アップスケール（SAM2に渡す前に拡大・縦横比を保って正方形パディング）
        if upscale > 0:
            max_wh   = max(orig_w, orig_h)
            pad_w    = max_wh - orig_w
            pad_h    = max_wh - orig_h
            pad_left = pad_w // 2
            pad_top  = pad_h // 2
            from torchvision.transforms import functional as TVF
            img_sq   = TVF.pad(img_pil,
                               (pad_left, pad_top, pad_w-pad_left, pad_h-pad_top),
                               fill=0)
            img_sam  = img_sq.resize((upscale, upscale), Image.BILINEAR)
            print(f'[INFO] {orig_w}×{orig_h} → パディング → {upscale}×{upscale} にアップスケール')
        else:
            img_sam  = img_pil
            pad_left = pad_top = 0
            max_wh   = max(orig_w, orig_h)

        img_np     = np.array(img_sam)
        sam_h, sam_w = img_np.shape[:2]
        image_area = sam_h * sam_w

        print(f"\n[{idx+1}/{len(image_paths)}] {stem}")

        # SAM2でマスク生成
        masks = run_sam2(mask_generator, img_np)

        # 面積・縦横比で事前フィルタして候補を絞る
        def passes_filter(m):
            area = m['area']
            x, y, bw, bh = m['bbox']
            if bw == 0 or bh == 0:
                return False
            ratio = max(bw, bh) / min(bw, bh)
            if area < image_area * min_area_ratio:
                return False
            if area > image_area * max_area_ratio:
                return False
            if ratio < min_ratio:
                return False
            if ratio > max_ratio:
                return False
            return True

        # フィルタ通過したマスクを最初から選択状態にする
        selected = set()
        for i, m in enumerate(masks):
            if passes_filter(m):
                selected.add(i)

        print(f'[INFO] 事前フィルタ後: {len(selected)}/{len(masks)} 個が選択状態')

        saved = {'done': False, 'quit': False, 'rerun': False}
        state = {
            'mode'      : 'select',   # 'select' or 'brush'
            'brush_size': 8,
            'drawing'   : False,
            'erasing'   : False,
        }
        # 手動ブラシ用マスク（SAM2マスクに追加で塗れる）
        manual_mask = np.zeros((sam_h, sam_w), dtype=np.uint8)

        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        fig.canvas.manager.set_window_title(f"{stem} ({idx+1}/{len(image_paths)})")

        # 左: 元画像
        axes[0].imshow(img_np)
        axes[0].set_title('元画像', fontsize=12)
        axes[0].axis('off')

        # 右: SAM2マスク（クリックで選択 or ブラシ塗り）
        overlay    = build_overlay(img_np, masks, selected)
        mask_im    = axes[1].imshow(overlay)
        title_text = axes[1].set_title('', fontsize=10)
        axes[1].axis('off')

        legend_elements = [
            mpatches.Patch(color='green',  alpha=0.6, label='選択済み（SAM2）'),
            mpatches.Patch(color='red',    alpha=0.6, label='未選択（SAM2）'),
            mpatches.Patch(color='blue',   alpha=0.6, label='手動ブラシ'),
        ]
        axes[1].legend(handles=legend_elements, loc='lower left', fontsize=9)
        plt.tight_layout()

        def build_overlay_with_manual(img_np, masks, selected, manual_mask):
            overlay = img_np.copy().astype(float)
            for i, m in enumerate(masks):
                color = np.array([0,255,0]) if i in selected else np.array([255,80,80])
                seg   = m['segmentation']
                overlay[seg] = overlay[seg] * 0.4 + color * 0.6
            # 手動ブラシ部分を青で表示
            blue_area = manual_mask > 0
            if blue_area.any():
                overlay[blue_area] = overlay[blue_area] * 0.4 + np.array([80,80,255]) * 0.6
            return overlay.astype(np.uint8)

        def update_display():
            overlay = build_overlay_with_manual(img_np, masks, selected, manual_mask)
            mask_im.set_data(overlay)
            mode_str = (f'【選択モード】左=選択 右=除外 B=ブラシモードへ' if state['mode'] == 'select'
                        else f'【ブラシモード】左=塗る 右=消す ホイール=サイズ({state["brush_size"]}px) B=選択モードへ')
            title_text.set_text(
                f'選択: {len(selected)}  手動: {manual_mask.sum()//255}px  |  {mode_str}\n'
                f'S=保存  R=再実行  Z=手動クリア  Q=終了'
            )
            fig.canvas.draw_idle()

        update_display()

        def paint_brush(x, y, erase=False):
            xi, yi = int(round(x)), int(round(y))
            bs = state['brush_size']
            y1 = max(0, yi-bs); y2 = min(sam_h, yi+bs+1)
            x1 = max(0, xi-bs); x2 = min(sam_w, xi+bs+1)
            yy, xx = np.ogrid[y1:y2, x1:x2]
            circle = (yy-yi)**2 + (xx-xi)**2 <= bs**2
            manual_mask[y1:y2, x1:x2][circle] = 0 if erase else 255

        def find_mask_at(x, y):
            """クリック座標に対応するマスクを探す（面積が小さい順に優先）"""
            xi, yi = int(round(x)), int(round(y))
            if xi < 0 or xi >= sam_w or yi < 0 or yi >= sam_h:
                return None
            candidates = [
                (m['area'], i) for i, m in enumerate(masks)
                if m['segmentation'][yi, xi]
            ]
            if not candidates:
                return None
            # 一番小さいマスクを優先（細かい物体を選びやすくする）
            return min(candidates)[1]

        def on_click(event):
            if event.inaxes != axes[1] or event.xdata is None:
                return
            if state['mode'] == 'select':
                mask_idx = find_mask_at(event.xdata, event.ydata)
                if mask_idx is None:
                    return
                if event.button == 1:
                    selected.add(mask_idx)
                elif event.button == 3:
                    selected.discard(mask_idx)
            elif state['mode'] == 'brush':
                state['drawing'] = (event.button == 1)
                state['erasing'] = (event.button == 3)
                paint_brush(event.xdata, event.ydata, erase=(event.button == 3))
            update_display()

        def on_motion(event):
            if event.inaxes != axes[1] or event.xdata is None:
                return
            if state['mode'] == 'brush':
                if state['drawing']:
                    paint_brush(event.xdata, event.ydata, erase=False)
                    update_display()
                elif state['erasing']:
                    paint_brush(event.xdata, event.ydata, erase=True)
                    update_display()

        def on_release(event):
            state['drawing'] = False
            state['erasing'] = False

        def on_scroll(event):
            if state['mode'] == 'brush':
                if event.button == 'up':
                    state['brush_size'] = min(50, state['brush_size'] + 2)
                elif event.button == 'down':
                    state['brush_size'] = max(1, state['brush_size'] - 2)
                update_display()

        def on_key(event):
            if event.key == 'b':
                # ブラシモード ↔ 選択モード切り替え
                state['mode'] = 'brush' if state['mode'] == 'select' else 'select'
                update_display()
            elif event.key == 'z':
                # 手動マスククリア
                manual_mask[:] = 0
                update_display()
            elif event.key == 's':
                # SAM2選択マスク + 手動ブラシマスクを合成して保存
                masks_dict = {i: m for i, m in enumerate(masks)}
                binary_sam = masks_to_binary(masks_dict, selected, sam_h, sam_w)
                binary_combined = np.clip(binary_sam.astype(int) + manual_mask.astype(int), 0, 255).astype(np.uint8)

                # アップスケールしていた場合は元のサイズに戻す
                if upscale > 0:
                    scale        = upscale / max_wh
                    crop_left    = int(pad_left * scale)
                    crop_top     = int(pad_top  * scale)
                    crop_right   = crop_left + int(orig_w * scale)
                    crop_bottom  = crop_top  + int(orig_h * scale)
                    binary_cropped = binary_combined[crop_top:crop_bottom, crop_left:crop_right]
                    binary = np.array(
                        Image.fromarray(binary_cropped).resize((orig_w, orig_h), Image.NEAREST)
                    )
                else:
                    binary = binary_combined

                Image.fromarray(binary).save(out_path)
                print(f'  保存: {out_path}  ({len(selected)} SAM2マスク + 手動ブラシ)  サイズ: {orig_w}×{orig_h}')
                saved['done'] = True
                plt.close(fig)
            elif event.key == 'q':
                saved['quit'] = True
                plt.close(fig)
            elif event.key == 'r':
                saved['rerun'] = True
                plt.close(fig)

        fig.canvas.mpl_connect('button_press_event',   on_click)
        fig.canvas.mpl_connect('motion_notify_event',  on_motion)
        fig.canvas.mpl_connect('button_release_event', on_release)
        fig.canvas.mpl_connect('scroll_event',         on_scroll)
        fig.canvas.mpl_connect('key_press_event',      on_key)
        plt.show()

        if saved['quit']:
            print("終了します。")
            break
        elif saved['rerun']:
            print("再実行します。")
            continue  # 同じ画像をもう一度
        else:
            idx += 1

    print("\nアノテーション完了！")
    print(f"保存先: {output_dir}")
    print("次のステップ: python unet_count.py train --images ... --masks " + output_dir)


def main():
    parser = argparse.ArgumentParser(
        description='SAM2を使った半自動マスクアノテーションツール（UNet学習用）'
    )
    parser.add_argument('--images',      required=True,  help='画像ディレクトリ')
    parser.add_argument('--output',      required=True,  help='マスク保存先')
    parser.add_argument('--checkpoint',  required=True,  help='SAM2チェックポイント (.pt)')
    parser.add_argument('--model-type',  default='small',
                        choices=['tiny', 'small', 'base_plus', 'large'])
    parser.add_argument('--min-area',       type=float, default=0.001,
                        help='最小面積フィルタ（画像面積の割合 default: 0.001）')
    parser.add_argument('--max-area',       type=float, default=0.3,
                        help='最大面積フィルタ（画像面積の割合 default: 0.3）')
    parser.add_argument('--min-ratio',      type=float, default=2.0,
                        help='縦横比の最小値（細長さ下限 default: 2.0）')
    parser.add_argument('--max-ratio',      type=float, default=20.0,
                        help='縦横比の最大値（細長さ上限 default: 20.0）')
    parser.add_argument('--upscale',        type=int,   default=1024,
                        help='SAM2に渡す前に拡大するサイズ 0で無効 (default: 1024)')
    parser.add_argument('--points-per-side',type=int,   default=64,
                        help='SAM2のグリッド密度 大きいほど細かく検出 (default: 64)')
    parser.add_argument('--cpu',            action='store_true')
    args = parser.parse_args()

    if args.cpu:
        device = 'cpu'
    elif torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f'[INFO] デバイス: {device}')

    annotate_images(
        images_dir      = args.images,
        output_dir      = args.output,
        checkpoint      = args.checkpoint,
        model_type      = args.model_type,
        device          = device,
        min_area_ratio  = args.min_area,
        max_area_ratio  = args.max_area,
        min_ratio       = args.min_ratio,
        max_ratio       = args.max_ratio,
        upscale         = args.upscale,
        points_per_side = args.points_per_side,
    )


if __name__ == '__main__':
    main()