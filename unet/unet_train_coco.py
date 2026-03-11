# unet_coco_train.py
# COCOフォーマット対応 UNet 学習・推論スクリプト
#
# 【使い方】
#   学習: python unet_coco_train.py train \
#           --images dataset/images --masks dataset/masks \
#           --json dataset/annotations.json
#   推論: python unet_coco_train.py predict 画像.jpg \
#           --weights checkpoints/best_model.pth

import os
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import ndimage

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, random_split
import sys
sys.path.insert(0, os.path.dirname(__file__))
from coco_dataset import UNetCOCODataset


# ==================== モデル ====================

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64,128,256,512]):
        super().__init__()
        self.downs      = nn.ModuleList()
        self.ups        = nn.ModuleList()
        self.pool       = nn.MaxPool2d(2, 2)

        ch = in_channels
        for f in features:
            self.downs.append(DoubleConv(ch, f)); ch = f

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)

        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(f*2, f, 2, stride=2))
            self.ups.append(DoubleConv(f*2, f))

        self.final = nn.Conv2d(features[0], out_channels, 1)

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x); skips.append(x); x = self.pool(x)
        x = self.bottleneck(x)
        skips = skips[::-1]
        for i in range(0, len(self.ups), 2):
            x    = self.ups[i](x)
            skip = skips[i//2]
            if x.shape != skip.shape:
                x = TF.resize(x, skip.shape[2:], TF.InterpolationMode.BILINEAR)
            x = torch.cat([skip, x], dim=1)
            x = self.ups[i+1](x)
        return self.final(x)


# ==================== 損失・評価 ====================

class DiceBCELoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.bce    = nn.BCEWithLogitsLoss()
        self.smooth = smooth

    def forward(self, pred, target):
        bce  = self.bce(pred, target)
        p    = torch.sigmoid(pred)
        intr = (p * target).sum(dim=(1,2,3))
        dice = 1 - (2*intr + self.smooth) / (p.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3)) + self.smooth)
        return bce + dice.mean()


def iou_score(pred, target, thresh=0.5):
    p    = (torch.sigmoid(pred) > thresh).float()
    intr = (p * target).sum()
    union = p.sum() + target.sum() - intr
    return (intr / (union + 1e-6)).item()


class AverageMeter:
    def __init__(self): self.reset()
    def reset(self): self.sum = self.count = 0
    def update(self, v, n=1): self.sum += v*n; self.count += n
    @property
    def avg(self): return self.sum / max(self.count, 1)


# ==================== 学習 ====================

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    m = AverageMeter()
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        loss = criterion(model(imgs), masks)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        m.update(loss.item(), imgs.size(0))
    return m.avg


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    m = AverageMeter()
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        m.update(iou_score(model(imgs), masks), imgs.size(0))
    return m.avg


def train(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = (torch.device('cpu') if args.cpu else
              torch.device('cuda') if torch.cuda.is_available() else
              torch.device('mps')  if torch.backends.mps.is_available() else
              torch.device('cpu'))
    print(f"[INFO] デバイス: {device}")

    full_ds = UNetCOCODataset(args.images, args.masks, args.json,
                               img_size=args.img_size)
    n_val   = max(1, int(len(full_ds) * args.val_ratio))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val])
    val_ds.dataset.augment = False

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False)
    print(f"[INFO] 学習: {n_train} 枚  検証: {n_val} 枚")

    model     = UNet(in_channels=args.in_channels).to(device)
    criterion = DiceBCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                      factor=0.5, patience=10)

    best_iou    = 0.0
    start_epoch = 0
    train_losses, val_ious = [], []

    if args.pretrained and os.path.isfile(args.pretrained):
        ckpt = torch.load(args.pretrained, map_location=device)
        model.load_state_dict(ckpt['state_dict'])
        start_epoch = ckpt.get('epoch', 0)
        best_iou    = ckpt.get('best_iou', 0.0)

    print(f"\n{'='*50}\n  UNet 学習開始（COCOフォーマット）\n{'='*50}\n")

    for epoch in range(start_epoch, args.epochs):
        loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        iou  = validate(model, val_loader, device)
        scheduler.step(iou)
        train_losses.append(loss); val_ious.append(iou)
        print(f"Epoch [{epoch+1:>3}/{args.epochs}]  Loss: {loss:.4f}  Val IoU: {iou:.4f}")

        if iou > best_iou:
            best_iou = iou
            torch.save({'epoch': epoch+1, 'state_dict': model.state_dict(),
                        'best_iou': best_iou},
                       os.path.join(args.output_dir, 'best_model.pth'))
            print(f"  → ベストモデル更新 (IoU={best_iou:.4f})")

        if (epoch+1) % 10 == 0:
            torch.save({'epoch': epoch+1, 'state_dict': model.state_dict(),
                        'best_iou': best_iou},
                       os.path.join(args.output_dir, f'epoch_{epoch+1:04d}.pth'))

    print(f"\n学習完了！  Best IoU: {best_iou:.4f}")


# ==================== 推論 ====================

def count_objects(mask_np, min_size=50):
    labeled, n = ndimage.label(mask_np)
    boxes = []; valid = 0
    for i in range(1, n+1):
        comp = (labeled == i)
        if comp.sum() < min_size:
            labeled[comp] = 0; continue
        rows = np.any(comp, axis=1); cols = np.any(comp, axis=0)
        y1, y2 = np.where(rows)[0][[0,-1]]
        x1, x2 = np.where(cols)[0][[0,-1]]
        boxes.append((x1,y1,x2,y2)); valid += 1
    return valid, labeled, boxes


def predict(args):
    device = (torch.device('cpu') if args.cpu else
              torch.device('cuda') if torch.cuda.is_available() else
              torch.device('mps')  if torch.backends.mps.is_available() else
              torch.device('cpu'))

    model = UNet(in_channels=args.in_channels).to(device)
    ckpt  = torch.load(args.weights, map_location=device)
    model.load_state_dict(ckpt['state_dict'] if 'state_dict' in ckpt else ckpt)
    model.eval()

    img_pil        = Image.open(args.image).convert('RGB' if args.in_channels == 3 else 'L')
    orig_w, orig_h = img_pil.size

    # 学習時と同じ処理: パディング → 正方形 → リサイズ
    max_wh   = max(orig_w, orig_h)
    pad_w    = max_wh - orig_w
    pad_h    = max_wh - orig_h
    pad_left = pad_w // 2
    pad_top  = pad_h // 2
    img_padded = TF.pad(img_pil,
                        (pad_left, pad_top, pad_w - pad_left, pad_h - pad_top),
                        fill=0)
    img_r = img_padded.resize((args.img_size, args.img_size), Image.BILINEAR)
    t     = TF.to_tensor(img_r)
    if args.in_channels == 3:
        t = TF.normalize(t, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    t = t.unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(t)
    mask = (torch.sigmoid(pred) > args.threshold).squeeze().cpu().numpy().astype(np.uint8)

    count, labeled, boxes = count_objects(mask, args.min_size)
    print(f"\n{'='*40}\n  カウント数: {count}\n{'='*40}\n")

    # パディング込みのスケールで元画像座標に戻す
    scale      = max_wh / args.img_size
    boxes_orig = [
        (max(0, int(x1*scale) - pad_left),
         max(0, int(y1*scale) - pad_top),
         min(orig_w, int(x2*scale) - pad_left),
         min(orig_h, int(y2*scale) - pad_top))
        for x1, y1, x2, y2 in boxes
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(img_pil, cmap='gray' if args.in_channels == 1 else None)
    axes[0].set_title('元画像'); axes[0].axis('off')
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title(f'セグメンテーション (thresh={args.threshold})'); axes[1].axis('off')
    axes[2].imshow(img_pil, cmap='gray' if args.in_channels == 1 else None)
    for x1,y1,x2,y2 in boxes_orig:
        axes[2].add_patch(patches.Rectangle((x1,y1), x2-x1, y2-y1,
                          linewidth=2, edgecolor='red', facecolor='none'))
    axes[2].set_title(f'カウント結果: {count} 個'); axes[2].axis('off')
    plt.suptitle('UNet 物体カウント', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"[INFO] 結果を保存: {args.output}")
    plt.show()


# ==================== エントリポイント ====================

def main():
    parser = argparse.ArgumentParser(description='COCOフォーマット対応 UNet')
    sub = parser.add_subparsers(dest='mode', required=True)

    t = sub.add_parser('train')
    t.add_argument('--images',       required=True)
    t.add_argument('--masks',        required=True, help='マスクPNGディレクトリ')
    t.add_argument('--json',         default=None,  help='COCO形式JSON（省略可）')
    t.add_argument('--output-dir',   default='checkpoints_unet')
    t.add_argument('--epochs',       type=int,   default=100)
    t.add_argument('--batch-size',   type=int,   default=4)
    t.add_argument('--lr',           type=float, default=1e-4)
    t.add_argument('--img-size',     type=int,   default=512)
    t.add_argument('--in-channels',  type=int,   default=3,
                   help='入力チャンネル数 3=カラー 1=白黒 (default: 3)')
    t.add_argument('--val-ratio',    type=float, default=0.1)
    t.add_argument('--pretrained',   type=str,   default=None)
    t.add_argument('--cpu',          action='store_true')

    p = sub.add_parser('predict')
    p.add_argument('image')
    p.add_argument('--weights',      required=True)
    p.add_argument('--img-size',     type=int,   default=512)
    p.add_argument('--in-channels',  type=int,   default=3,
                   help='1=白黒画像（結晶など）  3=カラー画像')
    p.add_argument('--threshold',    type=float, default=0.5)
    p.add_argument('--min-size',     type=int,   default=50)
    p.add_argument('--output',       default='result_unet.png')
    p.add_argument('--cpu',          action='store_true')

    args = parser.parse_args()
    if args.mode == 'train':
        train(args)
    else:
        predict(args)


if __name__ == '__main__':
    main()