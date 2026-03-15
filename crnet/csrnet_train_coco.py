# csrnet_train_coco.py
# COCOフォーマット対応 CSRNet 学習スクリプト
#
# 【使い方】
#   python csrnet_train_coco.py \
#     --images  dataset/images \
#     --json    dataset/annotations.json \
#     --output-dir checkpoints \
#     --epochs 80 --batch-size 4 --sigma 10 --cpu

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, os.path.dirname(__file__))
from coco_dataset import CSRNetCOCODataset


# ==================== モデル ====================

class CSRNet(nn.Module):
    def __init__(self, load_weights=True):
        super().__init__()
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat  = [512, 512, 512, 256, 128, 64]
        self.frontend      = self._make_layers(self.frontend_feat)
        self.backend       = self._make_layers(self.backend_feat, in_channels=512, dilation=True)
        self.output_layer  = nn.Conv2d(64, 1, kernel_size=1)
        if load_weights:
            self._load_vgg_weights()

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        return self.output_layer(x)

    def _make_layers(self, cfg, in_channels=3, dilation=False):
        d = 2 if dilation else 1
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(2, 2)]
            else:
                layers += [nn.Conv2d(in_channels, v, 3, padding=d, dilation=d), nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def _load_vgg_weights(self):
        vgg16    = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        vgg_feat = list(vgg16.features.children())
        vi = 0
        for layer in self.frontend.children():
            if isinstance(layer, nn.Conv2d):
                while vi < len(vgg_feat) and not isinstance(vgg_feat[vi], nn.Conv2d):
                    vi += 1
                if vi < len(vgg_feat):
                    layer.weight.data = vgg_feat[vi].weight.data
                    layer.bias.data   = vgg_feat[vi].bias.data
                    vi += 1
        print("[INFO] VGG-16 事前学習済み重みを読み込みました")


# ==================== 学習 ====================

class AverageMeter:
    def __init__(self): self.reset()
    def reset(self): self.sum = self.count = 0
    def update(self, v, n=1): self.sum += v*n; self.count += n
    @property
    def avg(self): return self.sum / max(self.count, 1)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    m = AverageMeter()
    for imgs, dmaps in loader:
        imgs, dmaps = imgs.to(device), dmaps.to(device)
        loss = criterion(model(imgs), dmaps)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        m.update(loss.item(), imgs.size(0))
    return m.avg


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    mae = AverageMeter()
    for imgs, dmaps in loader:
        imgs, dmaps = imgs.to(device), dmaps.to(device)
        preds = model(imgs)
        diff  = (preds.sum(dim=[1,2,3]) - dmaps.sum(dim=[1,2,3])).abs()
        mae.update(diff.mean().item(), imgs.size(0))
    return mae.avg


def main():
    parser = argparse.ArgumentParser(description='COCOフォーマット対応 CSRNet 学習')
    parser.add_argument('--images',      required=True)
    parser.add_argument('--json',        required=True, help='COCO形式アノテーションJSON')
    parser.add_argument('--output-dir',  default='checkpoints_csrnet')
    parser.add_argument('--epochs',      type=int,   default=80)
    parser.add_argument('--batch-size',  type=int,   default=4)
    parser.add_argument('--lr',          type=float, default=1e-5)
    parser.add_argument('--sigma',       type=float, default=10)
    parser.add_argument('--patch-size',  type=int,   default=256)
    parser.add_argument('--val-ratio',   type=float, default=0.1)
    parser.add_argument('--pretrained',  type=str,   default=None)
    parser.add_argument('--no-vgg-init', action='store_true')
    parser.add_argument('--cpu',         action='store_true')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = (torch.device('cpu') if args.cpu else
              torch.device('cuda') if torch.cuda.is_available() else
              torch.device('mps')  if torch.backends.mps.is_available() else
              torch.device('cpu'))
    print(f"[INFO] デバイス: {device}")

    full_ds = CSRNetCOCODataset(args.images, args.json,
                                 sigma=args.sigma, patch_size=args.patch_size)
    n_val   = max(1, int(len(full_ds) * args.val_ratio))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val])
    val_ds.dataset.augment = False

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False)
    print(f"[INFO] 学習: {n_train} 枚  検証: {n_val} 枚")

    model     = CSRNet(load_weights=not args.no_vgg_init).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam([
        {'params': model.frontend.parameters(),     'lr': args.lr * 0.1},
        {'params': model.backend.parameters(),      'lr': args.lr},
        {'params': model.output_layer.parameters(), 'lr': args.lr},
    ])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    best_mae    = float('inf')
    train_losses, val_maes = [], []
    start_epoch = 0

    if args.pretrained and os.path.isfile(args.pretrained):
        ckpt = torch.load(args.pretrained, map_location=device)
        model.load_state_dict(ckpt['state_dict'])
        start_epoch = ckpt.get('epoch', 0)
        best_mae    = ckpt.get('best_mae', float('inf'))
        print(f"[INFO] 再開: epoch {start_epoch}")

    print(f"\n{'='*50}\n  CSRNet 学習開始（COCOフォーマット）\n{'='*50}\n")

    for epoch in range(start_epoch, args.epochs):
        loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        mae  = validate(model, val_loader, device)
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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(train_losses); ax1.set_title('Train Loss'); ax1.grid(True)
    ax2.plot(val_maes, color='orange'); ax2.set_title('Val MAE'); ax2.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'history.png'), dpi=120)
    print(f"[INFO] 学習曲線: {args.output_dir}/history.png")


if __name__ == '__main__':
    main()