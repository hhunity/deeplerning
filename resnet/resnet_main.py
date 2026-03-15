# resnet_classify.py
# 窪みの切り出し画像から結晶数を分類するResNetベース分類器
#
# 【使い方】
#   学習:
#     python resnet_classify.py train --images crops/images --annotations crops/annotations.json
#   推論（1枚）:
#     python resnet_classify.py predict 画像.jpg --config cavities.json --weights best.pth
#   推論（フォルダ）:
#     python resnet_classify.py predict-dir crops/ --weights best.pth
#
# 【データ構造】
#   crops/
#     images/          image001.jpg ...
#     annotations.json  COCOフォーマット（annotation_coco2.py で作成）

import os
import sys
import glob
import json
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
import torchvision.transforms.functional as TF


# ==================== データセット ====================

class CavityDataset(Dataset):
    """
    COCO-format dataset for crystal count classification.
    Expected layout:
        images_dir/   *.png (or *.jpg)
        json_path     COCO annotations JSON (created by annotation_coco2.py)

    Crystal count per image = number of annotations for that image.
    Count values are mapped to class indices automatically.
    """

    def __init__(self, images_dir, json_path, img_size=128, augment=False):
        self.img_size = img_size
        self.augment  = augment
        self.samples  = []
        self.classes  = []

        with open(json_path) as f:
            coco = json.load(f)

        count_map = {}
        for ann in coco['annotations']:
            iid = ann['image_id']
            count_map[iid] = count_map.get(iid, 0) + 1

        raw = []
        for img_info in coco['images']:
            iid      = img_info['id']
            count    = count_map.get(iid, 0)
            img_path = os.path.join(images_dir, img_info['file_name'])
            if os.path.isfile(img_path):
                raw.append((img_path, count))

        if not raw:
            raise RuntimeError(f"画像が見つかりません: {images_dir}")

        unique_counts = sorted(set(c for _, c in raw))
        self.classes  = [str(c) for c in unique_counts]
        count_to_idx  = {c: i for i, c in enumerate(unique_counts)}

        self.samples = [{'path': p, 'label': count_to_idx[c]} for p, c in raw]
        print(f"[INFO] クラス: {self.classes}")
        print(f"[INFO] 合計: {len(self.samples)} 枚")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s     = self.samples[idx]
        img   = Image.open(s['path']).convert('RGB')
        label = s['label']

        # リサイズ
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)

        # データ拡張
        if self.augment:
            if torch.rand(1) > 0.5:
                img = TF.hflip(img)
            if torch.rand(1) > 0.5:
                img = TF.vflip(img)
            angle = (torch.rand(1).item() - 0.5) * 30  # ±15度
            img   = TF.rotate(img, angle)
            img   = TF.adjust_brightness(img, 0.8 + torch.rand(1).item() * 0.4)
            img   = TF.adjust_contrast(img,   0.8 + torch.rand(1).item() * 0.4)

        t = TF.to_tensor(img)
        t = TF.normalize(t, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        return t, label


# ==================== モデル ====================

def build_model(num_classes, pretrained=True):
    """ResNet18ベースの分類モデル（軽量・少ないデータでも学習しやすい）"""
    model = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


# ==================== 学習 ====================

def train(args):
    os.makedirs(args.output_dir, exist_ok=True)

    device = (torch.device('cpu') if args.cpu else
              torch.device('cuda') if torch.cuda.is_available() else
              torch.device('mps')  if torch.backends.mps.is_available() else
              torch.device('cpu'))
    print(f"[INFO] デバイス: {device}")

    full_ds = CavityDataset(args.images, args.annotations, img_size=args.img_size, augment=True)
    n_class = len(full_ds.classes)
    n_val   = max(1, int(len(full_ds) * args.val_ratio))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val])
    val_ds.dataset.augment = False

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=2)
    print(f"[INFO] 学習: {n_train} 枚  検証: {n_val} 枚  クラス数: {n_class}")

    model     = build_model(n_class, pretrained=not args.no_pretrain).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10
    )

    # クラス情報を保存
    meta = {'classes': full_ds.classes, 'img_size': args.img_size}
    with open(os.path.join(args.output_dir, 'classes.json'), 'w') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    best_acc   = 0.0
    train_losses, val_accs = [], []

    for epoch in range(1, args.epochs + 1):
        # 学習
        model.train()
        total_loss = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out  = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)

        # 検証
        model.eval()
        correct = total = accepted = uncertain = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                out      = model(imgs)
                probs    = torch.softmax(out, dim=1)
                sorted_p = probs.sort(dim=1, descending=True).values
                top1     = sorted_p[:, 0]
                margin   = sorted_p[:, 0] - sorted_p[:, 1]
                preds    = out.argmax(dim=1)

                for i in range(len(labels)):
                    total += 1
                    is_uncertain = (
                        top1[i].item()   < args.conf_thresh or
                        margin[i].item() < args.margin_thresh
                    )
                    if is_uncertain:
                        uncertain += 1
                    else:
                        accepted += 1
                        if preds[i] == labels[i]:
                            correct += 1

        acc          = correct / accepted  if accepted  > 0 else 0
        acc_all      = correct / total     if total     > 0 else 0
        uncertain_r  = uncertain / total   if total     > 0 else 0
        val_accs.append(acc_all)
        scheduler.step(acc_all)

        if args.conf_thresh > 0 or args.margin_thresh > 0:
            print(f"Epoch {epoch:3d}/{args.epochs}  Loss: {avg_loss:.4f}"
                  f"  Acc(採用): {acc:.3f}  Acc(全体): {acc_all:.3f}"
                  f"  要確認: {uncertain}/{total} ({uncertain_r:.0%})")
        else:
            print(f"Epoch {epoch:3d}/{args.epochs}  Loss: {avg_loss:.4f}  Val Acc: {acc_all:.3f}")

        if acc_all > best_acc:
            best_acc = acc_all
            torch.save({
                'epoch'        : epoch,
                'model'        : model.state_dict(),
                'classes'      : full_ds.classes,
                'img_size'     : args.img_size,
                'conf_thresh'  : args.conf_thresh,
                'margin_thresh': args.margin_thresh,
            }, os.path.join(args.output_dir, 'best.pth'))
            print(f"  → ベストモデル保存 (Acc={best_acc:.3f})")

    # 学習曲線
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(train_losses); axes[0].set_title('Train Loss'); axes[0].set_xlabel('Epoch')
    axes[1].plot(val_accs);     axes[1].set_title('Val Accuracy'); axes[1].set_xlabel('Epoch')
    plt.tight_layout()
    curve_path = os.path.join(args.output_dir, 'training_curve.png')
    plt.savefig(curve_path, dpi=120)
    print(f"[INFO] 学習曲線: {curve_path}")
    plt.show()

    print(f"\n学習完了  最高精度: {best_acc:.3f}")
    print(f"モデル: {os.path.join(args.output_dir, 'best.pth')}")


# ==================== 推論（フォルダ） ====================

def predict_dir(args):
    """切り出し済みフォルダの全画像を推論"""
    device, model, classes, img_size = load_model(args.weights, args.cpu)

    exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
    paths = sorted(
        p for ext in exts for p in glob.glob(os.path.join(args.crop_dir, ext))
    )
    if not paths:
        print(f"[ERROR] 画像が見つかりません: {args.crop_dir}")
        return

    results = []
    for path in paths:
        label, conf, all_probs, uncertain = infer_one(
            path, device, model, classes, img_size,
            args.conf_thresh, args.margin_thresh
        )
        results.append({'path': path, 'label': label, 'conf': conf,
                        'all_probs': all_probs, 'uncertain': uncertain})
        prob_str = '  '.join(f"{cls}:{p:.0%}" for cls, p in all_probs.items())
        flag     = '  ⚠️ 要確認' if uncertain else ''
        print(f"  {os.path.basename(path)}: {label}  [{prob_str}]{flag}")

    total = sum(
        int(''.join(filter(str.isdigit, r['label'])) or '0')
        for r in results
    )
    print(f"\n{'='*40}")
    print(f"  窪み数: {len(results)}")
    print(f"  総結晶数: {total}")
    print(f"{'='*40}")
    return results


# ==================== 推論（1枚の元画像） ====================

def predict_image(args):
    """元画像 + cavities.json から推論して結果を可視化"""
    from crop_tool import crop_for_predict

    device, model, classes, img_size = load_model(args.weights, args.cpu)

    # 窪みを切り出す
    tmp_dir = '_tmp_crops'
    crops   = crop_for_predict(args.image, args.config, tmp_dir)

    results = []
    for crop_info in crops:
        label, conf, all_probs, uncertain = infer_one(
            crop_info['path'], device, model, classes, img_size,
            args.conf_thresh, args.margin_thresh
        )
        results.append({**crop_info, 'label': label, 'conf': conf,
                        'all_probs': all_probs, 'uncertain': uncertain})
        prob_str = '  '.join(f"{cls}:{p:.0%}" for cls, p in all_probs.items())
        flag     = '  ⚠️ 要確認' if uncertain else ''
        print(f"  窪み{crop_info['id']:02d}: {label}  [{prob_str}]{flag}")

    # 結果表示
    img_pil = Image.open(args.image).convert('RGB')
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(np.array(img_pil))
    ax.axis('off')

    total        = 0
    n_uncertain  = 0
    for r in results:
        x1, y1, x2, y2 = r['bbox']
        n     = int(''.join(filter(str.isdigit, r['label'])) or '0')
        total += n
        if r['uncertain']:
            n_uncertain += 1
            color    = '#ff0000'   # 要確認は赤
            alpha    = 0.35
        else:
            color = ['#888888', '#44cc44', '#ff9900', '#ff4444', '#cc44ff'][min(n, 4)]
            alpha = 0.25

        rect = mpatches.Rectangle(
            (x1, y1), x2-x1, y2-y1,
            linewidth=2, edgecolor=color, facecolor=color, alpha=alpha
        )
        ax.add_patch(rect)

        # 全クラスの確率 + 要確認フラグを表示
        prob_lines = '\n'.join(f"{cls}:{p:.0%}" for cls, p in r['all_probs'].items())
        if r['uncertain']:
            prob_lines += '\n⚠️要確認'
        ax.text(x1+3, y1+14, prob_lines,
                color='white', fontsize=7, fontweight='bold',
                bbox=dict(facecolor=color, alpha=0.75, pad=2, edgecolor='none'))

    title = f'総結晶数: {total}  (窪み数: {len(results)}'
    if n_uncertain:
        title += f'  ⚠️要確認: {n_uncertain}個'
    title += ')'
    ax.set_title(title, fontsize=14)
    plt.tight_layout()

    save_path = args.output or f'result_classify_{os.path.splitext(os.path.basename(args.image))[0]}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n{'='*40}\n  総結晶数: {total}\n{'='*40}")
    print(f"[INFO] 結果保存: {save_path}")
    plt.show()

    # 一時ファイル削除
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)


# ==================== ユーティリティ ====================

def load_model(weights_path, cpu=False):
    device = (torch.device('cpu') if cpu else
              torch.device('cuda') if torch.cuda.is_available() else
              torch.device('mps')  if torch.backends.mps.is_available() else
              torch.device('cpu'))

    ckpt    = torch.load(weights_path, map_location=device)
    classes = ckpt['classes']
    img_size = ckpt.get('img_size', 128)
    model   = build_model(len(classes), pretrained=False).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f"[INFO] モデル読み込み: {weights_path}  クラス: {classes}")
    return device, model, classes, img_size


def infer_one(img_path, device, model, classes, img_size,
              conf_thresh=0.0, margin_thresh=0.0):
    """
    Returns:
        label      : 最高確率のクラス名
        conf       : 最高確率の値
        all_probs  : 全クラスの確率 {クラス名: 確率}
        uncertain  : 要確認フラグ（True=自信なし）
    """
    img = Image.open(img_path).convert('RGB')
    img = img.resize((img_size, img_size), Image.BILINEAR)
    t   = TF.to_tensor(img)
    t   = TF.normalize(t, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    t   = t.unsqueeze(0).to(device)
    with torch.no_grad():
        out       = model(t)
        probs     = torch.softmax(out, dim=1)[0]
        sorted_p  = probs.sort(descending=True).values
        idx       = probs.argmax().item()
        all_probs = {cls: probs[i].item() for i, cls in enumerate(classes)}

    conf   = sorted_p[0].item()
    margin = (sorted_p[0] - sorted_p[1]).item() if len(sorted_p) > 1 else 1.0

    # 要確認フラグ
    uncertain = (conf < conf_thresh) or (margin < margin_thresh)

    return classes[idx], conf, all_probs, uncertain


# ==================== メイン ====================

def main():
    parser = argparse.ArgumentParser(description='窪み内結晶数分類（ResNet）')
    sub    = parser.add_subparsers(dest='mode', required=True)

    # train
    t = sub.add_parser('train', help='分類モデルを学習')
    t.add_argument('--images',         required=True, help='画像ファイルのディレクトリ')
    t.add_argument('--annotations',    required=True, help='COCOアノテーションJSONのパス（annotation_coco2.py で作成）')
    t.add_argument('--output-dir',     default='checkpoints_resnet')
    t.add_argument('--epochs',         type=int,   default=50)
    t.add_argument('--batch-size',     type=int,   default=16)
    t.add_argument('--lr',             type=float, default=1e-4)
    t.add_argument('--img-size',       type=int,   default=128)
    t.add_argument('--val-ratio',      type=float, default=0.2)
    t.add_argument('--conf-thresh',    type=float, default=0.0,
                   help='最高確率の閾値（例:0.8 → 80%%未満は要確認扱い）')
    t.add_argument('--margin-thresh',  type=float, default=0.0,
                   help='1位2位の確率差の閾値（例:0.2 → 差20%%未満は要確認扱い）')
    t.add_argument('--no-pretrain',    action='store_true')
    t.add_argument('--cpu',            action='store_true')

    # predict（元画像 + config）
    p = sub.add_parser('predict', help='元画像から推論・可視化')
    p.add_argument('image',           help='入力画像')
    p.add_argument('--config',        default='cavities.json')
    p.add_argument('--weights',       required=True)
    p.add_argument('--output',        default=None)
    p.add_argument('--conf-thresh',   type=float, default=0.0,
                   help='最高確率の閾値（例:0.8 → 80%%未満は要確認）')
    p.add_argument('--margin-thresh', type=float, default=0.0,
                   help='1位2位の確率差の閾値（例:0.2 → 差20%%未満は要確認）')
    p.add_argument('--cpu',           action='store_true')

    # predict-dir（切り出し済みフォルダ）
    d = sub.add_parser('predict-dir', help='切り出し済みフォルダを推論')
    d.add_argument('crop_dir',          help='切り出し画像フォルダ')
    d.add_argument('--weights',         required=True)
    d.add_argument('--conf-thresh',     type=float, default=0.0,
                   help='最高確率の閾値（例:0.8 → 80%%未満は要確認）')
    d.add_argument('--margin-thresh',   type=float, default=0.0,
                   help='1位2位の確率差の閾値（例:0.2 → 差20%%未満は要確認）')
    d.add_argument('--cpu',             action='store_true')

    args = parser.parse_args()
    if args.mode == 'train':
        train(args)
    elif args.mode == 'predict':
        predict_image(args)
    elif args.mode == 'predict-dir':
        predict_dir(args)


if __name__ == '__main__':
    main()