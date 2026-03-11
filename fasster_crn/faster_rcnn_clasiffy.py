# faster_rcnn_classify.py
# 窪みを検出しながら「中の結晶が何個か」を同時に分類するFaster R-CNN
#
# 通常のFaster R-CNNとの違い:
#   通常版: 物体か背景か（2クラス）
#   この版: 空/1個/2個/3個... （クラス数可変）
#
# 【アノテーション形式】
#   annotate_coco.py で bbox を描いて、category_id に結晶数を対応させる
#   categories: [
#     {"id": 1, "name": "0個"},
#     {"id": 2, "name": "1個"},
#     {"id": 3, "name": "2個"},
#     {"id": 4, "name": "3個"},
#   ]
#
# 【使い方】
#   学習:
#     python faster_rcnn_classify.py train \
#       --images dataset/images --json dataset/annotations.json \
#       --num-classes 4
#   推論:
#     python faster_rcnn_classify.py predict 画像.jpg \
#       --weights checkpoints_frcnn_cls/best.pth

import os
import sys
import json
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import torch
import torch.optim as optim
import torchvision.transforms.functional as TF
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import nms as torchvision_nms
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, os.path.dirname(__file__))
from coco_dataset import FasterRCNNCOCODataset


# ==================== モデル ====================

def build_model(num_classes, pretrained=True):
    """
    num_classes: 結晶数クラス数（背景を含まない）
    例: 0個/1個/2個/3個 → num_classes=4
    内部では背景+4 = 5クラスになる
    """
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    model   = fasterrcnn_resnet50_fpn(weights=weights)
    in_feat = model.roi_heads.box_predictor.cls_score.in_features
    # 背景(0) + num_classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feat, num_classes + 1)
    return model


def collate_fn(batch):
    return tuple(zip(*batch))


# ==================== 学習 ====================

class AverageMeter:
    def __init__(self): self.reset()
    def reset(self): self.sum = self.count = 0
    def update(self, v, n=1): self.sum += v*n; self.count += n
    @property
    def avg(self): return self.sum / max(self.count, 1)


def eval_epoch(model, loader, device, score_thresh, nms_thresh, class_names):
    """
    検証: 検出精度（Acc）と結晶総数のMAEを両方計測
    """
    model.eval()
    cls_correct = cls_total = 0
    count_errors = []

    with torch.no_grad():
        for images, targets in loader:
            images  = [img.to(device) for img in images]
            outputs = model(images)

            for output, target in zip(outputs, targets):
                # スコアフィルタ + NMS
                keep = output['scores'] >= score_thresh
                boxes_t  = output['boxes'][keep]
                scores_t = output['scores'][keep]
                labels_t = output['labels'][keep]
                if len(boxes_t) > 0:
                    nms_keep = torchvision_nms(boxes_t, scores_t, nms_thresh)
                    labels_t = labels_t[nms_keep]

                # 検出数と正解数の比較（簡易MAE）
                pred_count = sum(
                    int(''.join(filter(str.isdigit, class_names[l-1])) or '0')
                    for l in labels_t.cpu().tolist() if 1 <= l <= len(class_names)
                )
                gt_count   = sum(
                    int(''.join(filter(str.isdigit, class_names[l-1])) or '0')
                    for l in target['labels'].tolist() if 1 <= l <= len(class_names)
                )
                count_errors.append(abs(pred_count - gt_count))

    mae = np.mean(count_errors) if count_errors else 0.0
    return mae


def train(args):
    os.makedirs(args.output_dir, exist_ok=True)

    device = (torch.device('cpu') if args.cpu else
              torch.device('cuda') if torch.cuda.is_available() else
              torch.device('mps')  if torch.backends.mps.is_available() else
              torch.device('cpu'))
    print(f"[INFO] デバイス: {device}")

    # クラス名をJSONから読み込む
    with open(args.json) as f:
        coco = json.load(f)
    categories  = sorted(coco.get('categories', []), key=lambda x: x['id'])
    class_names = [c['name'] for c in categories]
    num_classes = len(class_names)

    if num_classes == 0:
        # categoriesがない場合はnum_classesオプションで指定
        num_classes = args.num_classes
        class_names = [f'{i}個' for i in range(num_classes)]
        print(f"[WARN] JSONにcategoriesがありません。{num_classes}クラスで学習します")

    print(f"[INFO] クラス数: {num_classes}  {class_names}")

    # クラス情報を保存
    with open(os.path.join(args.output_dir, 'classes.json'), 'w') as f:
        json.dump({'classes': class_names}, f, indent=2, ensure_ascii=False)

    full_ds = FasterRCNNCOCODataset(
        args.images, args.json, img_size=800, augment=True
    )
    n_val   = max(1, int(len(full_ds) * args.val_ratio))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val])
    val_ds.dataset.augment = False

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  collate_fn=collate_fn, num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=1,
                              shuffle=False, collate_fn=collate_fn)
    print(f"[INFO] 学習: {n_train} 枚  検証: {n_val} 枚")

    model     = build_model(num_classes, pretrained=not args.no_pretrained).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    best_mae   = float('inf')
    train_losses, val_maes = [], []

    for epoch in range(1, args.epochs + 1):
        model.train()
        m = AverageMeter()
        for images, targets in train_loader:
            images  = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            losses  = sum(model(images, targets).values())
            optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            m.update(losses.item(), len(images))

        scheduler.step()
        train_losses.append(m.avg)

        mae = eval_epoch(model, val_loader, device,
                         args.score_thresh, args.nms_thresh, class_names)
        val_maes.append(mae)
        print(f"Epoch {epoch:3d}/{args.epochs}  Loss: {m.avg:.4f}  Val MAE: {mae:.2f}")

        if mae < best_mae:
            best_mae = mae
            torch.save({
                'epoch'      : epoch,
                'state_dict' : model.state_dict(),
                'classes'    : class_names,
                'num_classes': num_classes,
            }, os.path.join(args.output_dir, 'best.pth'))
            print(f"  → ベストモデル保存 (MAE={best_mae:.2f})")

    # 学習曲線
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(train_losses); axes[0].set_title('Train Loss'); axes[0].set_xlabel('Epoch')
    axes[1].plot(val_maes);     axes[1].set_title('Val MAE（結晶数誤差）'); axes[1].set_xlabel('Epoch')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'training_curve.png'), dpi=120)
    plt.show()
    print(f"\n学習完了  最良MAE: {best_mae:.2f}")


# ==================== 推論 ====================

# カラーマップ（クラスごとに色を変える）
CLASS_COLORS = [
    '#888888',  # 0個: グレー
    '#44cc44',  # 1個: 緑
    '#ff9900',  # 2個: オレンジ
    '#ff4444',  # 3個: 赤
    '#cc44ff',  # 4個: 紫
    '#44ccff',  # 5個: 水色
]


def predict(args):
    device = (torch.device('cpu') if args.cpu else
              torch.device('cuda') if torch.cuda.is_available() else
              torch.device('mps')  if torch.backends.mps.is_available() else
              torch.device('cpu'))

    # モデル読み込み
    ckpt        = torch.load(args.weights, map_location=device)
    class_names = ckpt['classes']
    num_classes = ckpt['num_classes']
    model       = build_model(num_classes, pretrained=False).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    print(f"[INFO] クラス: {class_names}")

    img_pil    = Image.open(args.image).convert('RGB')
    img_tensor = TF.to_tensor(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)[0]

    # スコアフィルタ + NMS
    keep     = output['scores'] >= args.score_thresh
    boxes_t  = output['boxes'][keep]
    scores_t = output['scores'][keep]
    labels_t = output['labels'][keep]

    if len(boxes_t) > 0:
        nms_keep = torchvision_nms(boxes_t, scores_t, args.nms_thresh)
        boxes_t  = boxes_t[nms_keep]
        scores_t = scores_t[nms_keep]
        labels_t = labels_t[nms_keep]

    boxes  = boxes_t.cpu().numpy()
    scores = scores_t.cpu().numpy()
    labels = labels_t.cpu().numpy()

    # 総結晶数を集計
    total_crystals = 0
    class_counts   = {name: 0 for name in class_names}
    for label in labels:
        if 1 <= label <= len(class_names):
            name = class_names[label - 1]
            class_counts[name] += 1
            n = int(''.join(filter(str.isdigit, name)) or '0')
            total_crystals += n

    print(f"\n{'='*40}")
    print(f"  窪み検出数: {len(boxes)}")
    for name, cnt in class_counts.items():
        print(f"    {name}: {cnt} 個の窪み")
    print(f"  総結晶数: {total_crystals}")
    print(f"{'='*40}\n")

    # 可視化
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.patch.set_facecolor('#1a1a2e')
    for ax in axes:
        ax.set_facecolor('#1a1a2e')
        ax.axis('off')

    axes[0].imshow(img_pil)
    axes[0].set_title('元画像', color='white', fontsize=13)

    result_img = np.array(img_pil).copy()
    axes[1].imshow(result_img)

    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box
        if 1 <= label <= len(class_names):
            name  = class_names[label - 1]
            color = CLASS_COLORS[min(label - 1, len(CLASS_COLORS) - 1)]
        else:
            name, color = '?', '#ffffff'

        rect = mpatches.Rectangle(
            (x1, y1), x2-x1, y2-y1,
            linewidth=2, edgecolor=color, facecolor=color, alpha=0.2
        )
        axes[1].add_patch(rect)
        axes[1].text(
            x1 + 3, y1 + 14,
            f'{name}\n{score:.0%}',
            color='white', fontsize=8, fontweight='bold',
            bbox=dict(facecolor=color, alpha=0.75, pad=2, edgecolor='none')
        )

    # 凡例
    legend_handles = [
        mpatches.Patch(color=CLASS_COLORS[min(i, len(CLASS_COLORS)-1)],
                       label=f'{name}  ({class_counts[name]}窪み)')
        for i, name in enumerate(class_names)
    ]
    axes[1].legend(handles=legend_handles, loc='lower left',
                   fontsize=9, facecolor='#1a1a2e', labelcolor='white')
    axes[1].set_title(
        f'検出結果  窪み: {len(boxes)} 個  総結晶数: {total_crystals}',
        color='white', fontsize=13
    )

    plt.suptitle(
        f'Faster R-CNN 分類モード  |  総結晶数: {total_crystals}',
        color='white', fontsize=15, fontweight='bold', y=1.01
    )
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches='tight',
                facecolor='#1a1a2e', edgecolor='none')
    print(f"[INFO] 結果保存: {args.output}")
    plt.show()


# ==================== メイン ====================

def main():
    parser = argparse.ArgumentParser(
        description='Faster R-CNN 検出+分類（窪み内結晶数）'
    )
    sub = parser.add_subparsers(dest='mode', required=True)

    # train
    t = sub.add_parser('train', help='学習')
    t.add_argument('--images',        required=True)
    t.add_argument('--json',          required=True, help='COCO形式アノテーションJSON')
    t.add_argument('--num-classes',   type=int,   default=4,
                   help='結晶数クラス数（JSONにcategoriesがない場合に使用 default:4）')
    t.add_argument('--output-dir',    default='checkpoints_frcnn_cls')
    t.add_argument('--epochs',        type=int,   default=60)
    t.add_argument('--batch-size',    type=int,   default=2)
    t.add_argument('--lr',            type=float, default=5e-3)
    t.add_argument('--val-ratio',     type=float, default=0.1)
    t.add_argument('--score-thresh',  type=float, default=0.5)
    t.add_argument('--nms-thresh',    type=float, default=0.3)
    t.add_argument('--no-pretrained', action='store_true')
    t.add_argument('--cpu',           action='store_true')

    # predict
    p = sub.add_parser('predict', help='推論・可視化')
    p.add_argument('image')
    p.add_argument('--weights',      required=True)
    p.add_argument('--score-thresh', type=float, default=0.5)
    p.add_argument('--nms-thresh',   type=float, default=0.3)
    p.add_argument('--output',       default='result_frcnn_cls.png')
    p.add_argument('--cpu',          action='store_true')

    args = parser.parse_args()
    if args.mode == 'train':
        train(args)
    elif args.mode == 'predict':
        predict(args)


if __name__ == '__main__':
    main()