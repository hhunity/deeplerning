# faster_rcnn_coco.py
# COCO学習済みFaster R-CNNでお試し推論（学習不要）
#
# 使い方:
#   python faster_rcnn_coco.py 画像.jpg
#   python faster_rcnn_coco.py 画像.jpg --thresh 0.7
#   python faster_rcnn_coco.py 画像.jpg --classes 44 45 46  # 特定クラスのみ
#
# 初回実行時にCOCO学習済み重み（約160MB）を自動ダウンロードします。

import argparse
import torch
import torchvision.transforms.functional as TF
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
)
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as pe
import random

# COCO 80クラスのラベル一覧
COCO_LABELS = [
    '__background__',
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
    'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
    'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush',
]


def get_device(force_cpu=False):
    if force_cpu:
        return torch.device('cpu')
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def load_model(device):
    print('[INFO] COCO学習済みモデルを読み込み中...')
    print('       (初回は約160MBのダウンロードが発生します)')
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)
    model.to(device)
    model.eval()
    print('[INFO] モデル準備完了')
    return model


def predict(model, image_path, device, score_thresh=0.5, target_classes=None):
    img_pil = Image.open(image_path).convert('RGB')
    img_tensor = TF.to_tensor(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)

    output = outputs[0]
    boxes  = output['boxes'].cpu()
    labels = output['labels'].cpu()
    scores = output['scores'].cpu()

    # スコア閾値フィルタ
    keep = scores >= score_thresh
    boxes  = boxes[keep]
    labels = labels[keep]
    scores = scores[keep]

    # 特定クラスフィルタ（指定がある場合）
    if target_classes:
        mask = torch.tensor([l.item() in target_classes for l in labels])
        boxes  = boxes[mask]
        labels = labels[mask]
        scores = scores[mask]

    return img_pil, boxes.numpy(), labels.numpy(), scores.numpy()


def visualize(img_pil, boxes, labels, scores, output_path='result.png'):
    fig, ax = plt.subplots(1, 1, figsize=(14, 9))
    ax.imshow(img_pil)

    # クラスごとに色を固定
    color_map = {}
    rng = random.Random(42)

    # クラスごとのカウント集計
    count_per_class = {}

    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box
        class_name = COCO_LABELS[label] if label < len(COCO_LABELS) else str(label)

        if class_name not in color_map:
            color_map[class_name] = (
                rng.random(), rng.random(), rng.random()
            )
        color = color_map[class_name]

        count_per_class[class_name] = count_per_class.get(class_name, 0) + 1

        # バウンディングボックス
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor=color, facecolor='none'
        )
        ax.add_patch(rect)

        # ラベル
        text = ax.text(
            x1, y1 - 4,
            f'{class_name} {score:.2f}',
            color='white', fontsize=8, fontweight='bold',
        )
        text.set_path_effects([
            pe.Stroke(linewidth=2, foreground=color),
            pe.Normal()
        ])

    total = len(boxes)
    title = f'検出総数: {total}個'
    if count_per_class:
        detail = '  |  ' + '  '.join(
            f'{k}: {v}' for k, v in sorted(count_per_class.items())
        )
        title += detail

    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'[INFO] 結果を保存: {output_path}')
    plt.show()

    # コンソールにもサマリー表示
    print(f"{'='*40}")
    print(f'  検出総数: {total} 個')
    if count_per_class:
        print('  クラス別内訳:')
        for k, v in sorted(count_per_class.items(), key=lambda x: -x[1]):
            print(f'    {k:20s}: {v} 個')
    print(f"{'='*40}")


def main():
    parser = argparse.ArgumentParser(
        description='COCO学習済みFaster R-CNNで物体を検出・カウントします'
    )
    parser.add_argument('image',
                        help='入力画像のパス')
    parser.add_argument('--thresh', type=float, default=0.5,
                        help='検出スコア閾値 (default: 0.5)  低くすると検出が増える')
    parser.add_argument('--classes', type=int, nargs='+', default=None,
                        help='検出するクラスIDを絞り込む (例: --classes 1 で人のみ)')
    parser.add_argument('--output', default='result.png',
                        help='結果画像の保存先 (default: result.png)')
    parser.add_argument('--list-classes', action='store_true',
                        help='COCOクラス一覧を表示して終了')
    parser.add_argument('--cpu', action='store_true',
                        help='CPUを強制使用')
    args = parser.parse_args()

    # クラス一覧表示モード
    if args.list_classes:
        print('\
COCO クラス一覧:')
        for i, name in enumerate(COCO_LABELS):
            if i == 0:
                continue
            print(f'  {i:3d}: {name}')
        return

    device = get_device(args.cpu)
    print(f'[INFO] デバイス: {device}')

    model = load_model(device)

    img_pil, boxes, labels, scores = predict(
        model, args.image, device,
        score_thresh=args.thresh,
        target_classes=args.classes,
    )

    visualize(img_pil, boxes, labels, scores, output_path=args.output)


if __name__ == '__main__':
    main()