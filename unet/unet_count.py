# unet_count.py
# UNetによるセグメンテーション + 連結成分カウント
#
# 【データ構成例】
# dataset/
#   images/
#     img_001.jpg
#   masks/
#     img_001.png  ← 物体=白(255)、背景=黒(0) の2値マスク
#
# 【使い方】
#   学習: python unet_count.py train --images dataset/images --masks dataset/masks
#   推論: python unet_count.py predict 画像.jpg --weights checkpoints/best_model.pth

import os
import glob
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
from torch.utils.data import Dataset, DataLoader, random_split
import random


# ==================== モデル定義 ====================

class DoubleConv(nn.Module):
    """UNetの基本ブロック: Conv → BN → ReLU × 2"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    """
    UNet: Convolutional Networks for Biomedical Image Segmentation
    論文: https://arxiv.org/abs/1505.04597

    構造:
      エンコーダ: 画像の特徴を圧縮しながら抽出
      デコーダ  : スキップ接続で細部を復元しながら拡大
      出力      : 各ピクセルが物体か背景かの確率マップ
    """

    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups   = nn.ModuleList()
        self.pool  = nn.MaxPool2d(kernel_size=2, stride=2)

        # エンコーダ
        ch = in_channels
        for f in features:
            self.downs.append(DoubleConv(ch, f))
            ch = f

        # ボトルネック
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # デコーダ
        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(f * 2, f))

        # 最終出力（1チャンネル）
        self.final = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skips = []

        # エンコーダ
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skips = skips[::-1]

        # デコーダ
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip = skips[i // 2]

            # サイズが合わない場合はパッド
            if x.shape != skip.shape:
                x = TF.resize(x, size=skip.shape[2:],
                               interpolation=TF.InterpolationMode.BILINEAR)
            x = torch.cat([skip, x], dim=1)
            x = self.ups[i + 1](x)

        return self.final(x)


# ==================== データセット ====================

class SegmentationDataset(Dataset):
    """
    画像 + 2値マスクのデータセット。
    マスクは白(255)=物体、黒(0)=背景 のPNG画像。
    """

    def __init__(self, images_dir, masks_dir, img_size=512, augment=True):
        self.images_dir = images_dir
        self.masks_dir  = masks_dir
        self.img_size   = img_size
        self.augment    = augment

        exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
        self.image_paths = sorted(
            p for ext in exts
            for p in glob.glob(os.path.join(images_dir, ext))
        )
        if not self.image_paths:
            raise RuntimeError(f"画像が見つかりません: {images_dir}")

    def __len__(self):
        return len(self.image_paths)

    def _load_mask(self, img_path):
        stem = os.path.splitext(os.path.basename(img_path))[0]
        # PNG優先、なければJPG
        for ext in ['.png', '.jpg', '.jpeg']:
            mask_path = os.path.join(self.masks_dir, stem + ext)
            if os.path.isfile(mask_path):
                return Image.open(mask_path).convert('L')
        raise FileNotFoundError(f"マスクが見つかりません: {stem}")

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img  = Image.open(img_path).convert('RGB')
        mask = self._load_mask(img_path)

        # リサイズ
        img  = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)

        # データ拡張
        if self.augment:
            if random.random() > 0.5:
                img  = TF.hflip(img)
                mask = TF.hflip(mask)
            if random.random() > 0.5:
                img  = TF.vflip(img)
                mask = TF.vflip(mask)
            # 輝度・コントラスト変動
            img = TF.adjust_brightness(img, 0.8 + random.random() * 0.4)
            img = TF.adjust_contrast(img,   0.8 + random.random() * 0.4)

        # テンソル変換
        img_tensor  = TF.to_tensor(img)  # (3, H, W) in [0,1]
        img_tensor  = TF.normalize(img_tensor,
                                   mean=[0.485, 0.456, 0.406],
                                   std =[0.229, 0.224, 0.225])
        mask_np     = np.array(mask, dtype=np.float32) / 255.0  # [0,1]
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)    # (1, H, W)

        return img_tensor, mask_tensor


# ==================== 損失関数 ====================

class DiceBCELoss(nn.Module):
    """
    Dice Loss + BCE Loss の組み合わせ
    細長い物体のような不均衡なセグメンテーションに有効
    """
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
        self.bce    = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)

        pred_sig = torch.sigmoid(pred)
        intersection = (pred_sig * target).sum(dim=(1, 2, 3))
        dice_loss = 1 - (2 * intersection + self.smooth) / (
            pred_sig.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + self.smooth
        )
        return bce_loss + dice_loss.mean()


# ==================== 評価指標 ====================

def iou_score(pred, target, threshold=0.5):
    """IoU（Intersection over Union）スコアを計算"""
    pred_bin = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred_bin * target).sum()
    union        = pred_bin.sum() + target.sum() - intersection
    return (intersection / (union + 1e-6)).item()


# ==================== 学習ループ ====================

class AverageMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.sum = self.count = 0
    def update(self, val, n=1):
        self.sum   += val * n
        self.count += n
    @property
    def avg(self):
        return self.sum / max(self.count, 1)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_meter = AverageMeter()
    for imgs, masks in loader:
        imgs  = imgs.to(device)
        masks = masks.to(device)
        preds = model(imgs)
        loss  = criterion(preds, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_meter.update(loss.item(), imgs.size(0))
    return loss_meter.avg


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    iou_meter = AverageMeter()
    for imgs, masks in loader:
        imgs  = imgs.to(device)
        masks = masks.to(device)
        preds = model(imgs)
        iou   = iou_score(preds, masks)
        iou_meter.update(iou, imgs.size(0))
    return iou_meter.avg


def plot_history(train_losses, val_ious, save_path='training_history.png'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(train_losses, label='Train Loss')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss'); ax1.legend(); ax1.grid(True)
    ax2.plot(val_ious, color='orange', label='Val IoU')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('IoU')
    ax2.set_title('Validation IoU'); ax2.legend(); ax2.grid(True)
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
    full_ds = SegmentationDataset(
        args.images, args.masks,
        img_size=args.img_size, augment=True
    )
    n_val   = max(1, int(len(full_ds) * args.val_ratio))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val])
    val_ds.dataset.augment = False

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=1,
                              shuffle=False, num_workers=2)
    print(f"[INFO] 学習: {n_train} 枚  検証: {n_val} 枚")

    # モデル
    model = UNet().to(device)
    criterion = DiceBCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10
    )

    best_iou = 0.0
    train_losses, val_ious = [], []
    start_epoch = 0

    # チェックポイント再開
    if args.pretrained and os.path.isfile(args.pretrained):
        ckpt = torch.load(args.pretrained, map_location=device)
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt.get('epoch', 0)
        best_iou    = ckpt.get('best_iou', 0.0)
        print(f"[INFO] 再開: epoch {start_epoch}, best IoU={best_iou:.4f}")

    print(f"\n{'='*50}")
    print(f"  UNet 学習開始")
    print(f"  epochs={args.epochs}  lr={args.lr}  batch={args.batch_size}  img_size={args.img_size}")
    print(f"{'='*50}\n")

    for epoch in range(start_epoch, args.epochs):
        loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        iou  = validate(model, val_loader, device)
        scheduler.step(iou)

        train_losses.append(loss)
        val_ious.append(iou)
        print(f"Epoch [{epoch+1:>3}/{args.epochs}]  Loss: {loss:.4f}  Val IoU: {iou:.4f}")

        if iou > best_iou:
            best_iou = iou
            torch.save({
                'epoch'     : epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'best_iou'  : best_iou,
            }, os.path.join(args.output_dir, 'best_model.pth'))
            print(f"  → ベストモデル更新 (IoU={best_iou:.4f})")

        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch'     : epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'best_iou'  : best_iou,
            }, os.path.join(args.output_dir, f'epoch_{epoch+1:04d}.pth'))

    print(f"\n学習完了！  Best Val IoU: {best_iou:.4f}")
    plot_history(train_losses, val_ious,
                 os.path.join(args.output_dir, 'training_history.png'))


# ==================== 推論メイン ====================

def count_objects(mask_np, min_size=50):
    """
    2値マスクから連結成分を数えてカウントする

    Args:
        mask_np  : 2値numpy配列 (H, W) 0or1
        min_size : 小さすぎる連結成分を除外するピクセル数閾値

    Returns:
        count    : カウント数
        labeled  : ラベル画像（各物体に番号が振られている）
        boxes    : 各物体のバウンディングボックスリスト
    """
    labeled, num_features = ndimage.label(mask_np)

    boxes = []
    valid_count = 0
    for i in range(1, num_features + 1):
        component = (labeled == i)
        size = component.sum()
        if size < min_size:
            labeled[component] = 0  # 小さすぎる成分を除外
            continue
        # バウンディングボックスを計算
        rows = np.any(component, axis=1)
        cols = np.any(component, axis=0)
        y1, y2 = np.where(rows)[0][[0, -1]]
        x1, x2 = np.where(cols)[0][[0, -1]]
        boxes.append((x1, y1, x2, y2))
        valid_count += 1

    return valid_count, labeled, boxes


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

    # モデル読み込み
    model = UNet().to(device)
    if not os.path.isfile(args.weights):
        raise FileNotFoundError(f"重みが見つかりません: {args.weights}")
    ckpt  = torch.load(args.weights, map_location=device)
    state = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    print(f"[INFO] 重みを読み込みました: {args.weights}")

    # 画像読み込み・前処理
    img_pil    = Image.open(args.image).convert('RGB')
    orig_w, orig_h = img_pil.size
    img_resized = img_pil.resize((args.img_size, args.img_size), Image.BILINEAR)
    img_tensor  = TF.to_tensor(img_resized)
    img_tensor  = TF.normalize(img_tensor,
                               mean=[0.485, 0.456, 0.406],
                               std =[0.229, 0.224, 0.225])
    img_tensor  = img_tensor.unsqueeze(0).to(device)

    # 推論
    with torch.no_grad():
        pred = model(img_tensor)
    pred_mask = (torch.sigmoid(pred) > args.threshold).squeeze().cpu().numpy().astype(np.uint8)

    # 連結成分カウント
    count, labeled, boxes = count_objects(pred_mask, min_size=args.min_size)

    print(f"\n{'='*40}")
    print(f"  カウント数: {count}")
    print(f"{'='*40}\n")

    # 元のサイズにスケーリング（ボックス座標）
    sx = orig_w / args.img_size
    sy = orig_h / args.img_size
    boxes_orig = [(int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy))
                  for x1, y1, x2, y2 in boxes]

    # 可視化
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 元画像
    axes[0].imshow(img_pil)
    axes[0].set_title('元画像', fontsize=13)
    axes[0].axis('off')

    # セグメンテーション結果
    axes[1].imshow(pred_mask, cmap='gray')
    axes[1].set_title(f'セグメンテーション結果\n(threshold={args.threshold})', fontsize=13)
    axes[1].axis('off')

    # カウント結果（元画像+バウンディングボックス）
    axes[2].imshow(img_pil)
    for x1, y1, x2, y2 in boxes_orig:
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor='red', facecolor='none'
        )
        axes[2].add_patch(rect)
    axes[2].set_title(f'カウント結果: {count} 個', fontsize=13)
    axes[2].axis('off')

    plt.suptitle('UNet 物体カウント', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"[INFO] 結果を保存: {args.output}")
    plt.show()


# ==================== エントリポイント ====================

def main():
    parser = argparse.ArgumentParser(description='UNet 学習・推論')
    sub = parser.add_subparsers(dest='mode', required=True)

    # --- train ---
    t = sub.add_parser('train', help='学習')
    t.add_argument('--images',     required=True, help='画像ディレクトリ')
    t.add_argument('--masks',      required=True, help='マスクディレクトリ')
    t.add_argument('--output-dir', default='checkpoints')
    t.add_argument('--epochs',     type=int,   default=100)
    t.add_argument('--batch-size', type=int,   default=4,
                   help='CPU では 1〜2 推奨')
    t.add_argument('--lr',         type=float, default=1e-4)
    t.add_argument('--img-size',   type=int,   default=512,
                   help='学習時のリサイズサイズ（default: 512）')
    t.add_argument('--val-ratio',  type=float, default=0.1)
    t.add_argument('--pretrained', type=str,   default=None,
                   help='学習再開用チェックポイント')
    t.add_argument('--cpu',        action='store_true')

    # --- predict ---
    p = sub.add_parser('predict', help='推論')
    p.add_argument('image',          help='入力画像パス')
    p.add_argument('--weights',      required=True, help='学習済み重みファイル (.pth)')
    p.add_argument('--img-size',     type=int,   default=512)
    p.add_argument('--threshold',    type=float, default=0.5,
                   help='セグメンテーション閾値（default: 0.5）')
    p.add_argument('--min-size',     type=int,   default=50,
                   help='最小連結成分サイズ（小さいノイズを除外 default: 50px）')
    p.add_argument('--output',       default='result_unet.png')
    p.add_argument('--cpu',          action='store_true')

    args = parser.parse_args()

    if args.mode == 'train':
        train(args)
    elif args.mode == 'predict':
        predict(args)


if __name__ == '__main__':
    main()