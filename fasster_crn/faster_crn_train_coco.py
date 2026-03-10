# faster_rcnn_coco_train.py
# COCOフォーマット対応 Faster R-CNN 学習・推論スクリプト
#
# 【使い方】
#   学習: python faster_rcnn_coco_train.py train \
#           --images dataset/images --json dataset/annotations.json
#   推論: python faster_rcnn_coco_train.py predict 画像.jpg \
#           --weights checkpoints/best_model.pth

import os
import argparse
import torch
import torch.optim as optim
import torchvision.transforms.functional as TF
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader, random_split
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys
sys.path.insert(0, os.path.dirname(__file__))
from coco_dataset import FasterRCNNCOCODataset


def build_model(num_classes=2, pretrained=True):
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    model   = fasterrcnn_resnet50_fpn(weights=weights)
    in_feat = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feat, num_classes)
    return model


def collate_fn(batch):
    return tuple(zip(*batch))


class AverageMeter:
    def __init__(self): self.reset()
    def reset(self): self.sum = self.count = 0
    def update(self, v, n=1): self.sum += v*n; self.count += n
    @property
    def avg(self): return self.sum / max(self.count, 1)


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    m = AverageMeter()
    for images, targets in loader:
        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        losses  = sum(model(images, targets).values())
        optimizer.zero_grad(); losses.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        m.update(losses.item(), len(images))
    return m.avg


@torch.no_grad()
def validate(model, loader, device, thresh=0.5):
    model.eval()
    mae = AverageMeter()
    for images, targets in loader:
        images = [img.to(device) for img in images]
        preds  = model(images)
        for pred, target in zip(preds, targets):
            pred_count = (pred['scores'] >= thresh).sum().item()
            gt_count   = len(target['boxes'])
            mae.update(abs(pred_count - gt_count))
    return mae.avg


def train(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = (torch.device('cpu') if args.cpu else
              torch.device('cuda') if torch.cuda.is_available() else
              torch.device('mps')  if torch.backends.mps.is_available() else
              torch.device('cpu'))
    print(f"[INFO] デバイス: {device}")

    full_ds = FasterRCNNCOCODataset(args.images, args.json)
    n_val   = max(1, int(len(full_ds) * args.val_ratio))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val])
    val_ds.dataset.augment = False

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  collate_fn=collate_fn, num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=1,
                              shuffle=False, collate_fn=collate_fn)
    print(f"[INFO] 学習: {n_train} 枚  検証: {n_val} 枚")

    model = build_model(num_classes=2, pretrained=not args.no_pretrained).to(device)
    optimizer = torch.optim.SGD([
        {'params': [p for n,p in model.named_parameters()
                    if 'backbone' in n and p.requires_grad], 'lr': args.lr * 0.1},
        {'params': [p for n,p in model.named_parameters()
                    if 'backbone' not in n and p.requires_grad], 'lr': args.lr},
    ], momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    best_mae = float('inf')
    start_epoch = 0

    if args.pretrained_ckpt and os.path.isfile(args.pretrained_ckpt):
        ckpt = torch.load(args.pretrained_ckpt, map_location=device)
        model.load_state_dict(ckpt['state_dict'])
        start_epoch = ckpt.get('epoch', 0)
        best_mae    = ckpt.get('best_mae', float('inf'))

    print(f"\n{'='*50}\n  Faster R-CNN 学習開始（COCOフォーマット）\n{'='*50}\n")

    train_losses, val_maes = [], []
    for epoch in range(start_epoch, args.epochs):
        loss = train_one_epoch(model, train_loader, optimizer, device)
        mae  = validate(model, val_loader, device, args.score_thresh)
        scheduler.step()
        train_losses.append(loss); val_maes.append(mae)
        print(f"Epoch [{epoch+1:>3}/{args.epochs}]  Loss: {loss:.4f}  Val MAE: {mae:.2f}")

        if mae < best_mae:
            best_mae = mae
            torch.save({'epoch': epoch+1, 'state_dict': model.state_dict(),
                        'best_mae': best_mae},
                       os.path.join(args.output_dir, 'best_model.pth'))
            print(f"  → ベストモデル更新 (MAE={best_mae:.2f})")

        if (epoch+1) % 10 == 0:
            torch.save({'epoch': epoch+1, 'state_dict': model.state_dict(),
                        'best_mae': best_mae},
                       os.path.join(args.output_dir, f'epoch_{epoch+1:04d}.pth'))

    print(f"\n学習完了！  Best MAE: {best_mae:.2f}")


def predict(args):
    device = (torch.device('cpu') if args.cpu else
              torch.device('cuda') if torch.cuda.is_available() else
              torch.device('mps')  if torch.backends.mps.is_available() else
              torch.device('cpu'))

    # --weights なし → COCOお試しモード
    coco_mode = (args.weights is None)
    if coco_mode:
        print('[INFO] COCOお試しモード（学習なし）')
        model = fasterrcnn_resnet50_fpn(
            weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT).to(device)
    else:
        model = build_model(num_classes=2, pretrained=False).to(device)
        ckpt  = torch.load(args.weights, map_location=device)
        model.load_state_dict(ckpt['state_dict'] if 'state_dict' in ckpt else ckpt)
        print(f"[INFO] 重みを読み込み: {args.weights}")

    model.eval()
    img_pil    = Image.open(args.image).convert('RGB')
    img_tensor = TF.to_tensor(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)[0]

    keep   = output['scores'] >= args.score_thresh
    boxes  = output['boxes'][keep].cpu().numpy()
    scores = output['scores'][keep].cpu().numpy()
    count  = len(boxes)

    print(f"\n{'='*40}\n  検出数: {count}\n{'='*40}\n")

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img_pil)
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = box
        ax.add_patch(patches.Rectangle((x1,y1), x2-x1, y2-y1,
                     linewidth=2, edgecolor='red', facecolor='none'))
        ax.text(x1, y1-4, f'{score:.2f}', color='red', fontsize=7, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.5, pad=1, edgecolor='none'))
    ax.set_title(f'Faster R-CNN  検出数: {count}', fontsize=13, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"[INFO] 結果を保存: {args.output}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='COCOフォーマット対応 Faster R-CNN')
    sub = parser.add_subparsers(dest='mode', required=True)

    t = sub.add_parser('train')
    t.add_argument('--images',          required=True)
    t.add_argument('--json',            required=True, help='COCO形式アノテーションJSON')
    t.add_argument('--output-dir',      default='checkpoints_frcnn')
    t.add_argument('--epochs',          type=int,   default=60)
    t.add_argument('--batch-size',      type=int,   default=2)
    t.add_argument('--lr',              type=float, default=5e-3)
    t.add_argument('--val-ratio',       type=float, default=0.1)
    t.add_argument('--score-thresh',    type=float, default=0.5)
    t.add_argument('--no-pretrained',   action='store_true')
    t.add_argument('--pretrained-ckpt', type=str,   default=None)
    t.add_argument('--cpu',             action='store_true')

    p = sub.add_parser('predict')
    p.add_argument('image')
    p.add_argument('--weights',      default=None)
    p.add_argument('--score-thresh', type=float, default=0.5)
    p.add_argument('--output',       default='result.png')
    p.add_argument('--cpu',          action='store_true')

    args = parser.parse_args()
    if args.mode == 'train':
        train(args)
    else:
        predict(args)


if __name__ == '__main__':
    main()
