# sam2_count.py
# SAM2（Segment Anything Model 2）で物体をセグメントしてカウントする
#
# 【インストール】
#   pip install torch torchvision
#   pip install git+https://github.com/facebookresearch/sam2.git
#
# 【モデルのダウンロード】
#   以下のいずれかを手動でダウンロードしてください（SAM2.1が最新推奨）
#
#   SAM2.1（推奨）:
#     ViT-T（超軽量）: https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt
#     ViT-S（軽量）  : https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt
#     ViT-B（バランス）: https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt
#     ViT-L（高精度）: https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
#
# 【使い方】
#   python sam2_count.py 画像.jpg --checkpoint sam2.1_hiera_small.pt --model-type small
#
# 【細長い物体向けフィルタの調整】
#   --min-area  : 小さすぎるマスクを除外（ノイズ対策）
#   --max-area  : 大きすぎるマスクを除外（背景対策）
#   --min-ratio : 縦横比の最小値（細長さフィルタ。1対5なら 3.0〜5.0 を推奨）
#   --max-ratio : 縦横比の最大値

import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import os


# SAM2のモデル設定ファイル名マッピング
MODEL_CONFIG_MAP = {
    'tiny'      : 'configs/sam2.1/sam2.1_hiera_t.yaml',
    'small'     : 'configs/sam2.1/sam2.1_hiera_s.yaml',
    'base_plus' : 'configs/sam2.1/sam2.1_hiera_b+.yaml',
    'large'     : 'configs/sam2.1/sam2.1_hiera_l.yaml',
}


def load_sam2(checkpoint, model_type, device):
    try:
        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    except ImportError:
        print('[ERROR] sam2がインストールされていません。')
        print('        以下を実行してください:')
        print('        pip install git+https://github.com/facebookresearch/sam2.git')
        raise

    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(
            f"チェックポイントが見つかりません: {checkpoint}\n"
            f"以下のURLからダウンロードしてください:\n"
            f"  tiny  : https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt\n"
            f"  small : https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt\n"
            f"  base+ : https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt\n"
            f"  large : https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
        )

    if model_type not in MODEL_CONFIG_MAP:
        raise ValueError(f"model-type は {list(MODEL_CONFIG_MAP.keys())} のいずれかを指定してください")

    config = MODEL_CONFIG_MAP[model_type]

    import torch
    sam2 = build_sam2(config, checkpoint, device=device)
    sam2.eval()

    # 自動マスク生成の設定
    # points_per_side     : 大きいほど細かく検出（重くなる）
    # pred_iou_thresh     : マスクの品質閾値
    # stability_score_thresh: マスクの安定性閾値
    mask_generator = SAM2AutomaticMaskGenerator(
        model                   = sam2,
        points_per_side         = 32,
        pred_iou_thresh         = 0.88,
        stability_score_thresh  = 0.95,
        crop_n_layers           = 0,
        min_mask_region_area    = 100,
    )
    return mask_generator


def get_bbox_ratio(mask):
    """マスクのバウンディングボックスの縦横比（長辺/短辺）を返す"""
    x, y, w, h = mask['bbox']  # xywh形式
    if w == 0 or h == 0:
        return 0
    return max(w, h) / min(w, h)


def filter_masks(masks, image_area, min_area_ratio, max_area_ratio, min_ratio, max_ratio):
    """
    面積・縦横比でマスクをフィルタする

    Args:
        masks          : SAM2が生成したマスクのリスト
        image_area     : 画像の総ピクセル数
        min_area_ratio : 画像面積に対する最小マスク面積の割合
        max_area_ratio : 画像面積に対する最大マスク面積の割合
        min_ratio      : バウンディングボックスの縦横比の最小値
        max_ratio      : バウンディングボックスの縦横比の最大値
    """
    filtered = []
    for m in masks:
        area  = m['area']
        ratio = get_bbox_ratio(m)

        if area < image_area * min_area_ratio:
            continue
        if area > image_area * max_area_ratio:
            continue
        if ratio < min_ratio:
            continue
        if ratio > max_ratio:
            continue

        filtered.append(m)

    return filtered


def visualize(img_pil, masks_all, masks_filtered, output_path):
    """元画像・全マスク・フィルタ後マスクを並べて表示"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    img_np = np.array(img_pil)

    # --- 元画像 ---
    axes[0].imshow(img_np)
    axes[0].set_title('元画像', fontsize=13)
    axes[0].axis('off')

    # --- 全マスク ---
    overlay_all = img_np.copy().astype(np.float32)
    rng = random.Random(42)
    for m in masks_all:
        color = np.array([rng.random() * 255 for _ in range(3)], dtype=np.float32)
        overlay_all[m['segmentation']] = (
            overlay_all[m['segmentation']] * 0.4 + color * 0.6
        )
    axes[1].imshow(overlay_all.astype(np.uint8))
    axes[1].set_title(f'全マスク: {len(masks_all)} 個', fontsize=13)
    axes[1].axis('off')

    # --- フィルタ後マスク ---
    overlay_filt = img_np.copy().astype(np.float32)
    rng2 = random.Random(0)
    for m in masks_filtered:
        color = np.array([rng2.random() * 255 for _ in range(3)], dtype=np.float32)
        overlay_filt[m['segmentation']] = (
            overlay_filt[m['segmentation']] * 0.4 + color * 0.6
        )
        x, y, w, h = m['bbox']
        rect = patches.Rectangle(
            (x, y), w, h,
            linewidth=1.5, edgecolor='yellow', facecolor='none'
        )
        axes[2].add_patch(rect)

    axes[2].imshow(overlay_filt.astype(np.uint8))
    axes[2].set_title(f'フィルタ後（カウント）: {len(masks_filtered)} 個', fontsize=13)
    axes[2].axis('off')

    plt.suptitle('SAM2 物体カウント', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'[INFO] 結果を保存: {output_path}')
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='SAM2で物体をセグメントしてカウント')
    parser.add_argument('image',
                        help='入力画像のパス')
    parser.add_argument('--checkpoint',  required=True,
                        help='SAM2モデルのチェックポイント (.pt)')
    parser.add_argument('--model-type',  default='small',
                        choices=['tiny', 'small', 'base_plus', 'large'],
                        help='モデルの種類 (default: small)\n'
                             '  tiny      : 最軽量・CPU向け\n'
                             '  small     : 軽量・バランス型（推奨）\n'
                             '  base_plus : 高精度\n'
                             '  large     : 最高精度・GPU推奨')
    parser.add_argument('--min-area',    type=float, default=0.001,
                        help='最小面積（画像面積の割合 default: 0.001 = 0.1%%）')
    parser.add_argument('--max-area',    type=float, default=0.3,
                        help='最大面積（画像面積の割合 default: 0.3 = 30%%）')
    parser.add_argument('--min-ratio',   type=float, default=1.0,
                        help='縦横比の最小値（細長さ下限 default: 1.0）\n'
                             '1対5の細長い物なら 3.0〜5.0 を推奨')
    parser.add_argument('--max-ratio',   type=float, default=20.0,
                        help='縦横比の最大値（細長さ上限 default: 20.0）')
    parser.add_argument('--output',      default='result_sam2.png',
                        help='結果画像の保存先 (default: result_sam2.png)')
    parser.add_argument('--cpu',         action='store_true',
                        help='CPUを強制使用（遅いので注意）')
    args = parser.parse_args()

    # デバイス設定
    import torch
    if args.cpu:
        device = 'cpu'
    elif torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f'[INFO] デバイス: {device}')
    print(f'[INFO] モデル  : SAM2.1 {args.model_type}')

    # モデル読み込み
    mask_generator = load_sam2(args.checkpoint, args.model_type, device)

    # 画像読み込み
    if not os.path.isfile(args.image):
        raise FileNotFoundError(f'画像が見つかりません: {args.image}')
    img_pil    = Image.open(args.image).convert('RGB')
    img_np     = np.array(img_pil)
    image_area = img_np.shape[0] * img_np.shape[1]
    print(f'[INFO] 画像サイズ: {img_pil.size[0]}x{img_pil.size[1]}')

    # SAM2推論
    print('[INFO] SAM2推論中...')
    masks_all = mask_generator.generate(img_np)
    print(f'[INFO] 生成マスク数: {len(masks_all)}')

    # フィルタ
    masks_filtered = filter_masks(
        masks_all, image_area,
        min_area_ratio = args.min_area,
        max_area_ratio = args.max_area,
        min_ratio      = args.min_ratio,
        max_ratio      = args.max_ratio,
    )

    # 結果表示
    print(f"\n{'='*40}")
    print(f'  全マスク数          : {len(masks_all)}')
    print(f'  フィルタ後（カウント）: {len(masks_filtered)}')
    print(f"{'='*40}\n")

    if len(masks_filtered) == 0:
        print('[HINT] カウントが0の場合は以下を試してください:')
        print('       --min-ratio を下げる（例: --min-ratio 1.0）')
        print('       --min-area  を下げる（例: --min-area 0.0005）')
        print('       --max-area  を上げる（例: --max-area 0.5）')

    # 可視化
    visualize(img_pil, masks_all, masks_filtered, args.output)


if __name__ == '__main__':
    main()