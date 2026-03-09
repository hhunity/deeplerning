# csrnet_train.py

# CSRNet 自前データ学習スクリプト

# 

# 【データ構成例】

# dataset/

# images/

# img_001.jpg

# img_002.jpg

# annotations/

# img_001.json   ← 点アノテーション（後述フォーマット）

# img_002.json

# 

# 【アノテーションJSON フォーマット】

# {

# “points”: [[x1,y1], [x2,y2], …]   ← 物体の中心座標リスト

# }

import os
import json
import glob
import argparse
import numpy as np
from PIL import Image
import scipy.ndimage

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random
import matplotlib.pyplot as plt

# ==================== モデル定義（推論スクリプトと共通） ====================

class CSRNet(nn.Module):
def **init**(self, load_weights=True):
super(CSRNet, self).**init**()
self.frontend_feat = [64, 64, ‘M’, 128, 128, ‘M’, 256, 256, 256, ‘M’, 512, 512, 512]
self.backend_feat  = [512, 512, 512, 256, 128, 64]
self.frontend = self._make_layers(self.frontend_feat)
self.backend  = self._make_layers(self.backend_feat, in_channels=512, dilation=True)
self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
if load_weights:
self._load_vgg_weights()

```
def forward(self, x):
    x = self.frontend(x)
    x = self.backend(x)
    x = self.output_layer(x)
    return x

def _make_layers(self, cfg, in_channels=3, dilation=False):
    d_rate = 2 if dilation else 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3,
                               padding=d_rate, dilation=d_rate)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

def _load_vgg_weights(self):
    vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    vgg_features = list(vgg16.features.children())
    vgg_idx = 0
    for layer in list(self.frontend.children()):
        if isinstance(layer, nn.Conv2d):
            while vgg_idx < len(vgg_features) and not isinstance(vgg_features[vgg_idx], nn.Conv2d):
                vgg_idx += 1
            if vgg_idx < len(vgg_features):
                layer.weight.data = vgg_features[vgg_idx].weight.data
                layer.bias.data   = vgg_features[vgg_idx].bias.data
                vgg_idx += 1
    print("[INFO] VGG-16 事前学習済み重みを読み込みました。")
```

# ==================== 密度マップ生成 ====================

def points_to_density_map(points, img_h, img_w, sigma=15):
“””
点座標リスト → ガウシアン密度マップ

```
Args:
    points : [[x,y], ...] 形式の点座標リスト
    img_h, img_w : 画像サイズ
    sigma  : ガウシアンの広がり（物体サイズに合わせて調整）

Returns:
    density : (img_h, img_w) の float32 numpy 配列
              全要素の合計 ≈ len(points)
"""
density = np.zeros((img_h, img_w), dtype=np.float32)
for (x, y) in points:
    xi, yi = int(round(x)), int(round(y))
    if 0 <= xi < img_w and 0 <= yi < img_h:
        density[yi, xi] += 1.0

# ガウシアンフィルタで滑らかな密度マップに変換
density = scipy.ndimage.gaussian_filter(density, sigma=sigma, mode='constant')
return density
```

# ==================== データセット ====================

class CrowdCountingDataset(Dataset):
“””
汎用カウンティングデータセット。
images_dir/  と  annotations_dir/  を対応付けて読む。
アノテーションは JSON {“points”: [[x,y], …]} 形式。
“””

```
def __init__(self, images_dir, annotations_dir,
             sigma=15, patch_size=None, augment=True):
    self.images_dir      = images_dir
    self.annotations_dir = annotations_dir
    self.sigma           = sigma
    self.patch_size      = patch_size   # None なら画像全体を使用
    self.augment         = augment

    # 画像ファイル一覧
    exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
    self.image_paths = sorted(
        p for ext in exts
        for p in glob.glob(os.path.join(images_dir, ext))
    )
    if len(self.image_paths) == 0:
        raise RuntimeError(f"画像が見つかりません: {images_dir}")

    self.img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])

def __len__(self):
    return len(self.image_paths)

def _load_annotation(self, img_path):
    """画像パスに対応するアノテーション JSON を読む"""
    stem = os.path.splitext(os.path.basename(img_path))[0]
    json_path = os.path.join(self.annotations_dir, stem + '.json')
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"アノテーションが見つかりません: {json_path}")
    with open(json_path) as f:
        data = json.load(f)
    return data.get('points', [])

def __getitem__(self, idx):
    img_path = self.image_paths[idx]

    # --- 画像読み込み ---
    img = Image.open(img_path).convert('RGB')
    w, h = img.size

    # --- アノテーション読み込み ---
    points = self._load_annotation(img_path)  # [[x,y], ...]

    # --- 密度マップ生成 ---
    density = points_to_density_map(points, h, w, sigma=self.sigma)

    # --- ランダムクロップ（オプション） ---
    if self.patch_size is not None:
        ph = pw = self.patch_size
        if h < ph or w < pw:
            # 画像がパッチより小さい場合はパッド
            img = TF.pad(img, (0, 0, max(0, pw - w), max(0, ph - h)))
            density = np.pad(density,
                             ((0, max(0, ph - h)), (0, max(0, pw - w))),
                             mode='constant')
            w, h = img.size

        top  = random.randint(0, h - ph)
        left = random.randint(0, w - pw)
        img     = TF.crop(img, top, left, ph, pw)
        density = density[top:top+ph, left:left+pw]

    # --- データ拡張 ---
    if self.augment:
        # 水平反転
        if random.random() > 0.5:
            img     = TF.hflip(img)
            density = density[:, ::-1].copy()

    # --- テンソル変換 ---
    img_tensor = self.img_transform(img)

    # 密度マップは CSRNet 出力が 1/8 サイズなので縮小
    dmap_h, dmap_w = density.shape[0] // 8, density.shape[1] // 8
    if dmap_h < 1: dmap_h = 1
    if dmap_w < 1: dmap_w = 1

    density_small = np.array(
        Image.fromarray(density).resize((dmap_w, dmap_h), Image.BILINEAR)
    ) * 64   # 面積補正: (1/8)^2 = 1/64 → 64 倍でカウント数を保存

    density_tensor = torch.from_numpy(density_small).float().unsqueeze(0)  # (1,H,W)

    return img_tensor, density_tensor
```

# ==================== 学習ユーティリティ ====================

class AverageMeter:
def **init**(self):
self.reset()

```
def reset(self):
    self.val = self.avg = self.sum = self.count = 0

def update(self, val, n=1):
    self.val   = val
    self.sum  += val * n
    self.count += n
    self.avg   = self.sum / self.count
```

def mae_mse(pred_maps, gt_maps):
“”“バッチ全体の MAE と MSE を返す”””
pred_counts = pred_maps.sum(dim=[1,2,3]).cpu().detach()
gt_counts   = gt_maps.sum(dim=[1,2,3]).cpu().detach()
mae = (pred_counts - gt_counts).abs().mean().item()
mse = ((pred_counts - gt_counts) ** 2).mean().sqrt().item()
return mae, mse

def save_checkpoint(state, path):
torch.save(state, path)
print(f”[CKPT] 保存: {path}”)

def plot_history(train_losses, val_maes, save_path=‘training_history.png’):
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(train_losses, label=‘Train Loss (MSE)’)
ax1.set_xlabel(‘Epoch’); ax1.set_ylabel(‘Loss’); ax1.set_title(‘Training Loss’)
ax1.legend(); ax1.grid(True)
ax2.plot(val_maes, label=‘Val MAE’, color=‘orange’)
ax2.set_xlabel(‘Epoch’); ax2.set_ylabel(‘MAE’); ax2.set_title(‘Validation MAE’)
ax2.legend(); ax2.grid(True)
plt.tight_layout()
plt.savefig(save_path, dpi=120)
plt.close()
print(f”[INFO] 学習曲線を保存: {save_path}”)

# ==================== 学習ループ ====================

def train_one_epoch(model, loader, criterion, optimizer, device):
model.train()
loss_meter = AverageMeter()

```
for imgs, dmaps in loader:
    imgs  = imgs.to(device)
    dmaps = dmaps.to(device)

    preds = model(imgs)
    loss  = criterion(preds, dmaps)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_meter.update(loss.item(), imgs.size(0))

return loss_meter.avg
```

@torch.no_grad()
def validate(model, loader, device):
model.eval()
mae_meter = AverageMeter()
mse_meter = AverageMeter()

```
for imgs, dmaps in loader:
    imgs  = imgs.to(device)
    dmaps = dmaps.to(device)
    preds = model(imgs)
    mae, mse = mae_mse(preds, dmaps)
    mae_meter.update(mae, imgs.size(0))
    mse_meter.update(mse, imgs.size(0))

return mae_meter.avg, mse_meter.avg
```

# ==================== メイン ====================

def main():
parser = argparse.ArgumentParser(description=‘CSRNet 学習スクリプト’)
parser.add_argument(’–images’,      required=True, help=‘画像ディレクトリ’)
parser.add_argument(’–annotations’, required=True, help=‘アノテーションディレクトリ’)
parser.add_argument(’–output-dir’,  default=‘checkpoints’, help=‘チェックポイント保存先’)
parser.add_argument(’–epochs’,      type=int,   default=100)
parser.add_argument(’–batch-size’,  type=int,   default=8)
parser.add_argument(’–lr’,          type=float, default=1e-5)
parser.add_argument(’–sigma’,       type=float, default=15,
help=‘密度マップのガウシアン幅（物体サイズに合わせて調整）’)
parser.add_argument(’–patch-size’,  type=int,   default=None,
help=‘ランダムクロップのパッチサイズ（省略=フル画像）’)
parser.add_argument(’–val-ratio’,   type=float, default=0.1,
help=‘検証データ割合（0〜1）’)
parser.add_argument(’–pretrained’,  type=str,   default=None,
help=‘学習再開用チェックポイント (.pth)’)
parser.add_argument(’–no-vgg-init’, action=‘store_true’)
parser.add_argument(’–cpu’,         action=‘store_true’)
args = parser.parse_args()

```
os.makedirs(args.output_dir, exist_ok=True)

# --- デバイス ---
if args.cpu:
    device = torch.device('cpu')
elif torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
print(f"[INFO] デバイス: {device}")

# --- データセット ---
full_dataset = CrowdCountingDataset(
    images_dir      = args.images,
    annotations_dir = args.annotations,
    sigma           = args.sigma,
    patch_size      = args.patch_size,
    augment         = True,
)
n_val   = max(1, int(len(full_dataset) * args.val_ratio))
n_train = len(full_dataset) - n_val
train_dataset, val_dataset = random_split(full_dataset, [n_train, n_val])
val_dataset.dataset.augment = False  # 検証はaugmentなし

train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                          shuffle=True,  num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=1,
                          shuffle=False, num_workers=2)
print(f"[INFO] 学習: {n_train} 枚  検証: {n_val} 枚")

# --- モデル ---
model = CSRNet(load_weights=not args.no_vgg_init).to(device)

# --- オプティマイザ ---
# フロントエンド（VGG-16部）は小さいLR、バックエンドは通常LR
optimizer = optim.Adam([
    {'params': model.frontend.parameters(),     'lr': args.lr * 0.1},
    {'params': model.backend.parameters(),      'lr': args.lr},
    {'params': model.output_layer.parameters(), 'lr': args.lr},
])
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
criterion = nn.MSELoss()

start_epoch = 0
best_mae = float('inf')
train_losses, val_maes = [], []

# --- チェックポイント再開 ---
if args.pretrained and os.path.isfile(args.pretrained):
    ckpt = torch.load(args.pretrained, map_location=device)
    model.load_state_dict(ckpt['state_dict'])
    optimizer.load_state_dict(ckpt['optimizer'])
    start_epoch = ckpt.get('epoch', 0)
    best_mae    = ckpt.get('best_mae', float('inf'))
    print(f"[INFO] チェックポイントから再開: epoch {start_epoch}, best MAE={best_mae:.2f}")

# --- 学習ループ ---
print(f"\n{'='*50}")
print(f"  学習開始  epochs={args.epochs}  lr={args.lr}  batch={args.batch_size}")
print(f"{'='*50}\n")

for epoch in range(start_epoch, args.epochs):
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_mae, val_mse = validate(model, val_loader, device)
    scheduler.step()

    train_losses.append(train_loss)
    val_maes.append(val_mae)

    print(f"Epoch [{epoch+1:>3}/{args.epochs}]  "
          f"Loss: {train_loss:.4f}  "
          f"Val MAE: {val_mae:.2f}  Val MSE: {val_mse:.2f}")

    # ベストモデル保存
    if val_mae < best_mae:
        best_mae = val_mae
        save_checkpoint({
            'epoch'      : epoch + 1,
            'state_dict' : model.state_dict(),
            'optimizer'  : optimizer.state_dict(),
            'best_mae'   : best_mae,
        }, os.path.join(args.output_dir, 'best_model.pth'))

    # 定期保存
    if (epoch + 1) % 10 == 0:
        save_checkpoint({
            'epoch'      : epoch + 1,
            'state_dict' : model.state_dict(),
            'optimizer'  : optimizer.state_dict(),
            'best_mae'   : best_mae,
        }, os.path.join(args.output_dir, f'epoch_{epoch+1:04d}.pth'))

print(f"\n学習完了！  Best Val MAE: {best_mae:.2f}")
plot_history(train_losses, val_maes,
             save_path=os.path.join(args.output_dir, 'training_history.png'))
```

if **name** == ‘**main**’:
main()