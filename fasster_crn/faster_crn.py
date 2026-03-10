# faster_rcnn_count.py
# Faster R-CNN による細長い物体のカウント
#
# 【CSRNetとの違い】
#   CSRNet  : 密度マップで「何個あるか」を推定（境界ボックスなし）
#   Faster R-CNN : 1本ずつ検出してカウント（位置・バウンディングボックスあり）
#
# 【データ構成例】
# dataset/
#   images/
#     img_001.jpg
#   annotations/
#     img_001.json  ← バウンディングボックスアノテーション（後述）
#
# 【アノテーションJSON フォーマット】
# {
#   "boxes": [
#     {"x1": 10, "y1": 20, "x2": 80, "y2": 35},  ← 左上(x1,y1) 右下(x2,y2)
#     ...
#   ]
# }

import os
import json
import glob
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


# COCO 80クラスのラベル一覧\r\nCOCO_LABELS = [\r\n    '__background__',\r\n    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',\r\n    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',\r\n    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',\r\n    'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',\r\n    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',\r\n    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',\r\n    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',\r\n    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',\r\n    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',\r\n    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',\r\n    'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',\r\n    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',\r\n    'scissors', 'teddy bear', 'hair drier', 'toothbrush',\r\n]\r\n\r\n\r\n# ==================== モデル構築 ====================

def build_model(num_classes=2, pretrained=True):
    """
    Faster R-CNN (ResNet-50 + FPN) を構築する。

    num_classes: 背景(0) + 物体クラス数
                 物体が1種類なら num_classes=2
    """
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    model = fasterrcnn_resnet50_fpn(weights=weights)

    # ヘッド部分を対象クラス数に差し替え
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


# ==================== データセット ====================

class RodDataset(Dataset):
    """
    バウンディングボックスアノテーション付きデータセット。
    細長い物体（棒・麺・製品など）向け。
    """

    def __init__(self, images_dir, annotations_dir, augment=True):
        self.images_dir      = images_dir
        self.annotations_dir = annotations_dir
        self.augment         = augment

        exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
        self.image_paths = sorted(
            p for ext in exts
            for p in glob.glob(os.path.join(images_dir, ext))
        )
        if not self.image_paths:
            raise RuntimeError(f"画像が見つかりません: {images_dir}")

    def __len__(self):
        return len(self.image_paths)

    def _load_annotation(self, img_path):
        stem = os.path.splitext(os.path.basename(img_path))[0]
        json_path = os.path.join(self.annotations_dir, stem + '.json')
        if not os.path.isfile(json_path):
            raise FileNotFoundError(f"アノテーションが見つかりません: {json_path}")
        with open(json_path) as f:
            data = json.load(f)
        return data.get('boxes', [])

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        w, h = img.size

        boxes_raw = self._load_annotation(img_path)

        # ボックス座標を float32 テンソルに変換
        boxes = []
        for b in boxes_raw:
            x1, y1 = float(b['x1']), float(b['y1'])
            x2, y2 = float(b['x2']), float(b['y2'])
            # 幅・高さが0以下のボックスをスキップ
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])

        # ---- データ拡張 ----
        if self.augment:
            # 水平反転
            if torch.rand(1) > 0.5:
                img = TF.hflip(img)
                boxes = [[w - b[2], b[1], w - b[0], b[3]] for b in boxes]
            # 垂直反転（俯瞰撮影では有効）
            if torch.rand(1) > 0.5:
                img = TF.vflip(img)
                boxes = [[b[0], h - b[3], b[2], h - b[1]] for b in boxes]
            # 輝度・コントラスト変動
            img = TF.adjust_brightness(img, 0.8 + torch.rand(1).item() * 0.4)
            img = TF.adjust_contrast(img,   0.8 + torch.rand(1).item() * 0.4)

        # ---- テンソル変換 ----
        img_tensor = TF.to_tensor(img)  # (C, H, W) in [0,1]

        if boxes:
            boxes_tensor  = torch.tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.ones(len(boxes), dtype=torch.int64)  # クラス1=物体
        else:
            boxes_tensor  = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros(0, dtype=torch.int64)

        target = {
            'boxes' : boxes_tensor,
            'labels': labels_tensor,
            'image_id': torch.tensor([idx]),
        }
        return img_tensor, target


def collate_fn(batch):
    """DataLoader 用コレート関数（サイズが違う画像に対応）"""
    return tuple(zip(*batch))


# ==================== 学習ループ ====================

class AverageMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = self.avg = self.sum = self.count = 0
    def update(self, val, n=1):
        self.val   = val
        self.sum  += val * n
        self.count += n
        self.avg   = self.sum / self.count


def train_one_epoch(model, loader, optimizer, device, print_freq=10):
    model.train()
    loss_meter = AverageMeter()

    for i, (images, targets) in enumerate(loader):
        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses    = sum(loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        # 勾配クリッピング（学習安定化）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        loss_meter.update(losses.item(), len(images))

        if (i + 1) % print_freq == 0:
            print(f"  step [{i+1}/{len(loader)}]  loss: {loss_meter.avg:.4f}")

    return loss_meter.avg


@torch.no_grad()
def validate(model, loader, device, score_thresh=0.5):
    """
    検証: 予測カウント数 vs GT カウント数 の MAE を返す
    """
    model.eval()
    mae_total = 0.0
    count = 0

    for images, targets in loader:
        images = [img.to(device) for img in images]
        preds  = model(images)

        for pred, target in zip(preds, targets):
            # スコア閾値でフィルタ
            keep = pred['scores'] >= score_thresh
            pred_count = keep.sum().item()
            gt_count   = len(target['boxes'])
            mae_total += abs(pred_count - gt_count)
            count += 1

    return mae_total / max(count, 1)


def plot_history(train_losses, val_maes, save_path='training_history.png'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(train_losses, label='Train Loss')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss'); ax1.legend(); ax1.grid(True)
    ax2.plot(val_maes, color='orange', label='Val MAE')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('MAE')
    ax2.set_title('Validation MAE'); ax2.legend(); ax2.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"[INFO] 学習曲線: {save_path}")


# ==================== 学習メイン ====================

def train(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # デバイス
    if args.cpu:
        device = torch.device('cpu')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"[INFO] デバイス: {device}")

    # データセット
    full_ds = RodDataset(args.images, args.annotations, augment=True)
    n_val   = max(1, int(len(full_ds) * args.val_ratio))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val])
    val_ds.dataset.augment = False

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  collate_fn=collate_fn, num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=1,
                              shuffle=False, collate_fn=collate_fn, num_workers=2)
    print(f"[INFO] 学習: {n_train} 枚  検証: {n_val} 枚")

    # モデル
    model = build_model(num_classes=2, pretrained=not args.no_pretrained).to(device)

    # オプティマイザ（バックボーンは小さいLR）
    params = [
        {'params': [p for n, p in model.named_parameters()
                    if 'backbone' in n and p.requires_grad], 'lr': args.lr * 0.1},
        {'params': [p for n, p in model.named_parameters()
                    if 'backbone' not in n and p.requires_grad], 'lr': args.lr},
    ]
    optimizer = optim.SGD(params, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    best_mae = float('inf')
    train_losses, val_maes = [], []
    start_epoch = 0

    # チェックポイント再開
    if args.pretrained_ckpt and os.path.isfile(args.pretrained_ckpt):
        ckpt = torch.load(args.pretrained_ckpt, map_location=device)
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt.get('epoch', 0)
        best_mae    = ckpt.get('best_mae', float('inf'))
        print(f"[INFO] 再開: epoch {start_epoch}, best MAE={best_mae:.2f}")

    print(f"\n{'='*50}")
    print(f"  Faster R-CNN 学習開始")
    print(f"  epochs={args.epochs}  lr={args.lr}  batch={args.batch_size}")
    print(f"{'='*50}\n")

    for epoch in range(start_epoch, args.epochs):
        loss = train_one_epoch(model, train_loader, optimizer, device)
        mae  = validate(model, val_loader, device, score_thresh=args.score_thresh)
        scheduler.step()

        train_losses.append(loss)
        val_maes.append(mae)
        print(f"Epoch [{epoch+1:>3}/{args.epochs}]  Loss: {loss:.4f}  Val MAE: {mae:.2f}")

        if mae < best_mae:
            best_mae = mae
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_mae': best_mae,
            }, os.path.join(args.output_dir, 'best_model.pth'))
            print(f"  → ベストモデル更新 (MAE={best_mae:.2f})")

        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_mae': best_mae,
            }, os.path.join(args.output_dir, f'epoch_{epoch+1:04d}.pth'))

    print(f"\n学習完了！  Best Val MAE: {best_mae:.2f}")
    plot_history(train_losses, val_maes,
                 os.path.join(args.output_dir, 'training_history.png'))


# ==================== 推論メイン ====================

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


def predict(args):
    # デバイス
    if args.cpu:
        device = torch.device('cpu')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    # --weights なし → COCOお試しモード
    coco_mode = (args.weights is None)

    if coco_mode:
        print('[INFO] --weights 未指定: COCO学習済みモデルで動作します（お試しモード）')
        print('       (初回は約160MBのダウンロードが発生します)')
        model = fasterrcnn_resnet50_fpn(
            weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        ).to(device)
    else:
        model = build_model(num_classes=2, pretrained=False).to(device)
        if not os.path.isfile(args.weights):
            raise FileNotFoundError(f"重みが見つかりません: {args.weights}")
        ckpt = torch.load(args.weights, map_location=device)
        state = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
        model.load_state_dict(state)
        print(f"[INFO] 重みを読み込みました: {args.weights}")

    model.eval()

    # 画像読み込み
    img_pil = Image.open(args.image).convert('RGB')
    img_tensor = TF.to_tensor(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)

    output = outputs[0]
    keep   = output['scores'] >= args.score_thresh
    boxes  = output['boxes'][keep].cpu().numpy()
    scores = output['scores'][keep].cpu().numpy()
    labels = output['labels'][keep].cpu().numpy() if coco_mode else None
    count  = len(boxes)

    print(f"\n{'='*40}")
    print(f"  検出数（カウント）: {count}")
    if coco_mode and count > 0:
        from collections import Counter
        class_counts = Counter(
            COCO_LABELS[l] if l < len(COCO_LABELS) else str(l)
            for l in labels
        )
        print('  クラス別内訳:')
        for name, cnt in class_counts.most_common():
            print(f'    {name:20s}: {cnt} 個')
    print(f"{'='*40}\n")

    # 可視化
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(img_pil)

    for i, (box, score) in enumerate(zip(boxes, scores)):
        x1, y1, x2, y2 = box
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor='red', facecolor='none'
        )
        ax.add_patch(rect)
        label_str = ''
        if coco_mode and labels is not None:
            l = labels[i]
            label_str = (COCO_LABELS[l] if l < len(COCO_LABELS) else str(l)) + ' '
        ax.text(x1, y1 - 4, f'{label_str}{score:.2f}',
                color='red', fontsize=7, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.5, pad=1, edgecolor='none'))

    mode_str = 'COCOお試しモード' if coco_mode else '自前学習モデル'
    ax.set_title(
        f'Faster R-CNN [{mode_str}]  検出数: {count}  (threshold={args.score_thresh})',
        fontsize=13, fontweight='bold'
    )
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"[INFO] 結果を保存: {args.output}")
    plt.show()


# ==================== エントリポイント ====================

def main():
    parser = argparse.ArgumentParser(description='Faster R-CNN 物体カウント')
    sub = parser.add_subparsers(dest='mode', required=True)

    # --- train ---
    t = sub.add_parser('train', help='学習')
    t.add_argument('--images',          required=True)
    t.add_argument('--annotations',     required=True)
    t.add_argument('--output-dir',      default='checkpoints')
    t.add_argument('--epochs',          type=int,   default=60)
    t.add_argument('--batch-size',      type=int,   default=2,
                   help='CPU では 1〜2 推奨')
    t.add_argument('--lr',              type=float, default=5e-3)
    t.add_argument('--val-ratio',       type=float, default=0.1)
    t.add_argument('--score-thresh',    type=float, default=0.5)
    t.add_argument('--no-pretrained',   action='store_true',
                   help='COCO 事前学習重みを使わない')
    t.add_argument('--pretrained-ckpt', type=str, default=None,
                   help='学習再開用チェックポイント')
    t.add_argument('--cpu',             action='store_true')

    # --- predict ---
    p = sub.add_parser('predict', help='推論')
    p.add_argument('image',             help='入力画像パス')
    p.add_argument('--weights',         default=None,
                   help='学習済み重みファイル (.pth)  省略するとCOCOお試しモードで動作')
    p.add_argument('--score-thresh',    type=float, default=0.5,
                   help='検出スコア閾値（上げると厳しく、下げると甘く）')
    p.add_argument('--output',          default='result.png')
    p.add_argument('--cpu',             action='store_true')

    args = parser.parse_args()

    if args.mode == 'train':
        train(args)
    elif args.mode == 'predict':
        predict(args)


if __name__ == '__main__':
    main()