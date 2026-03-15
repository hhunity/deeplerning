# coco_dataset.py
# COCOフォーマット対応データローダー
# CSRNet・Faster R-CNN・UNet で共通して使用する

import os
import glob
import json
import numpy as np
from PIL import Image
import scipy.ndimage
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import random


def pad_to_square(img, mask=None, fill=0):
    """
    画像（とマスク）を正方形にパディングする
    長辺に合わせて短辺に黒帯を追加する（中央寄せ）

    Args:
        img  : PIL Image
        mask : PIL Image or None
        fill : パディングの値（画像は0=黒、マスクは0=背景）

    Returns:
        img_padded, mask_padded（maskがNoneならNone）
    """
    w, h   = img.size
    max_wh = max(w, h)
    pad_w  = max_wh - w
    pad_h  = max_wh - h
    # 左右・上下に均等にパディング（余りは右・下に追加）
    left   = pad_w // 2
    right  = pad_w - left
    top    = pad_h // 2
    bottom = pad_h - top
    padding = (left, top, right, bottom)

    img_padded  = TF.pad(img,  padding, fill=fill)
    mask_padded = TF.pad(mask, padding, fill=0) if mask is not None else None
    return img_padded, mask_padded


def load_coco(json_path):
    """COCO JSONを読み込んでimage_id→annotationsのマップを返す"""
    with open(json_path) as f:
        coco = json.load(f)

    # image_id → file_name マップ
    id_to_image = {img['id']: img for img in coco['images']}

    # image_id → annotations リスト マップ
    id_to_anns = {}
    for ann in coco['annotations']:
        iid = ann['image_id']
        id_to_anns.setdefault(iid, []).append(ann)

    return coco, id_to_image, id_to_anns


# ==================== CSRNet用データセット ====================

def points_to_density_map(points, img_h, img_w, sigma=10):
    """点座標リスト → ガウシアン密度マップ"""
    density = np.zeros((img_h, img_w), dtype=np.float32)
    for (x, y) in points:
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < img_w and 0 <= yi < img_h:
            density[yi, xi] += 1.0
    density = scipy.ndimage.gaussian_filter(density, sigma=sigma, mode='constant')
    return density


class CSRNetCOCODataset(Dataset):
    """
    COCOフォーマット対応 CSRNet データセット
    アノテーションの 'point' フィールドを使用（pointモードで作成したJSON）
    """

    def __init__(self, images_dir, json_path, sigma=10, patch_size=None, augment=True):
        self.images_dir = images_dir
        self.sigma      = sigma
        self.patch_size = patch_size
        self.augment    = augment

        coco, id_to_image, id_to_anns = load_coco(json_path)
        self.samples = []
        for img_info in coco['images']:
            iid      = img_info['id']
            anns     = id_to_anns.get(iid, [])
            points   = [[a['point'][0], a['point'][1]] for a in anns if 'point' in a]
            img_path = os.path.join(images_dir, img_info['file_name'])
            if os.path.isfile(img_path):
                self.samples.append({'path': img_path, 'points': points})

        self.img_transform = __import__('torchvision').transforms.Compose([
            __import__('torchvision').transforms.ToTensor(),
            __import__('torchvision').transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s      = self.samples[idx]
        img    = Image.open(s['path']).convert('RGB')
        w, h   = img.size
        points = s['points']

        density = points_to_density_map(points, h, w, sigma=self.sigma)

        if self.patch_size is not None:
            ph = pw = self.patch_size
            if h < ph or w < pw:
                img     = TF.pad(img, (0, 0, max(0, pw-w), max(0, ph-h)))
                density = np.pad(density, ((0, max(0, ph-h)), (0, max(0, pw-w))))
                w, h    = img.size
            top  = random.randint(0, h - ph)
            left = random.randint(0, w - pw)
            img     = TF.crop(img, top, left, ph, pw)
            density = density[top:top+ph, left:left+pw]

        if self.augment and random.random() > 0.5:
            img     = TF.hflip(img)
            density = density[:, ::-1].copy()

        img_tensor = self.img_transform(img)

        dmap_h = max(1, density.shape[0] // 8)
        dmap_w = max(1, density.shape[1] // 8)
        density_small = np.array(
            Image.fromarray(density).resize((dmap_w, dmap_h), Image.BILINEAR)
        ) * 64

        return img_tensor, torch.from_numpy(density_small).float().unsqueeze(0)


# ==================== Faster R-CNN用データセット ====================

class FasterRCNNCOCODataset(Dataset):
    """
    COCOフォーマット対応 Faster R-CNN データセット
    アノテーションの 'bbox' フィールドを使用（bboxモードで作成したJSON）
    COCOのbboxはx,y,w,h形式 → x1,y1,x2,y2に変換
    """

    def __init__(self, images_dir, json_path, augment=True):
        self.images_dir = images_dir
        self.augment    = augment

        coco, id_to_image, id_to_anns = load_coco(json_path)
        self.samples = []
        for img_info in coco['images']:
            iid      = img_info['id']
            anns     = id_to_anns.get(iid, [])
            # COCOのbbox [x,y,w,h] → [x1,y1,x2,y2] に変換
            boxes = []
            for a in anns:
                if 'bbox' in a and a.get('iscrowd', 0) == 0:
                    x, y, bw, bh = a['bbox']
                    if bw > 0 and bh > 0:
                        boxes.append([x, y, x+bw, y+bh])
            img_path = os.path.join(images_dir, img_info['file_name'])
            if os.path.isfile(img_path):
                self.samples.append({'path': img_path, 'boxes': boxes})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s   = self.samples[idx]
        img = Image.open(s['path']).convert('RGB')
        w, h = img.size

        boxes = list(s['boxes'])

        if self.augment:
            if random.random() > 0.5:
                img   = TF.hflip(img)
                boxes = [[w-b[2], b[1], w-b[0], b[3]] for b in boxes]
            if random.random() > 0.5:
                img   = TF.vflip(img)
                boxes = [[b[0], h-b[3], b[2], h-b[1]] for b in boxes]
            img = TF.adjust_brightness(img, 0.8 + random.random() * 0.4)
            img = TF.adjust_contrast(img,   0.8 + random.random() * 0.4)

        img_tensor = TF.to_tensor(img)

        if boxes:
            boxes_tensor  = torch.tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.ones(len(boxes), dtype=torch.int64)
        else:
            boxes_tensor  = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros(0, dtype=torch.int64)

        target = {
            'boxes'   : boxes_tensor,
            'labels'  : labels_tensor,
            'image_id': torch.tensor([idx]),
        }
        return img_tensor, target


# ==================== UNet用データセット ====================

class UNetCOCODataset(Dataset):
    """
    UNet データセット
    imagesフォルダとmasksフォルダを直接スキャンして使う
    対応するマスクPNGがある画像だけ学習に使用
    json_pathは不要（省略可能・指定しても無視）
    """

    def __init__(self, images_dir, masks_dir, json_path=None, img_size=512, augment=True):
        self.images_dir = images_dir
        self.masks_dir  = masks_dir
        self.img_size   = img_size
        self.augment    = augment

        # imagesフォルダを直接スキャン
        exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
        self.samples = []
        for ext in exts:
            for img_path in sorted(glob.glob(os.path.join(images_dir, ext))):
                stem      = os.path.splitext(os.path.basename(img_path))[0]
                mask_path = os.path.join(masks_dir, stem + '.png')
                if os.path.isfile(mask_path):
                    self.samples.append({'img': img_path, 'mask': mask_path})

        if not self.samples:
            raise RuntimeError(
                f"学習データが見つかりません\n"
                f"  画像: {images_dir}\n"
                f"  マスク: {masks_dir}\n"
                f"  マスクと同名のPNGファイルが必要です"
            )
        print(f"[INFO] データセット: {len(self.samples)} 枚")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s    = self.samples[idx]
        img  = Image.open(s['img']).convert('RGB')
        mask = Image.open(s['mask']).convert('L')

        # 縦横比を保つためにパディングしてから正方形にリサイズ
        img, mask = pad_to_square(img, mask)
        img  = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)

        if self.augment:
            if random.random() > 0.5:
                img  = TF.hflip(img)
                mask = TF.hflip(mask)
            if random.random() > 0.5:
                img  = TF.vflip(img)
                mask = TF.vflip(mask)
            img = TF.adjust_brightness(img, 0.8 + random.random() * 0.4)
            img = TF.adjust_contrast(img,   0.8 + random.random() * 0.4)

        img_tensor  = TF.to_tensor(img)
        img_tensor  = TF.normalize(img_tensor,
                                   mean=[0.485, 0.456, 0.406],
                                   std =[0.229, 0.224, 0.225])
        mask_np     = np.array(mask, dtype=np.float32) / 255.0
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)

        return img_tensor, mask_tensor