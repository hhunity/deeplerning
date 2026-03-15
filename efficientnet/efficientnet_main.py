"""
EfficientNet-B0 for crystal count regression.
Usage:
  Train:   python efficientnet_main.py train --images ./data/images --annotations ./data/annotations.json --epochs 30
  Infer:   python efficientnet_main.py infer --image path/to/image.png --checkpoint best.pth
  GradCAM: python efficientnet_main.py gradcam --image path/to/image.png --checkpoint best.pth

Size:224x224

"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models, transforms
from torchvision.models import EfficientNet_B0_Weights
from PIL import Image
import numpy as np
from torch.utils.tensorboard import SummaryWriter


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CrystalDataset(Dataset):
    """
    COCO-format dataset for crystal count regression.
    Expected layout:
        images_dir/   *.png (or *.jpg)
        json_path     COCO annotations JSON (created by annotation_coco2.py)

    Crystal count per image = number of annotations for that image.
    """

    def __init__(self, images_dir: str, json_path: str, transform=None):
        import json
        self.transform = transform
        self.samples = []

        with open(json_path) as f:
            coco = json.load(f)

        count_map = {}
        for ann in coco['annotations']:
            iid = ann['image_id']
            count_map[iid] = count_map.get(iid, 0) + 1

        images_dir = Path(images_dir)
        for img_info in coco['images']:
            iid      = img_info['id']
            count    = float(count_map.get(iid, 0))
            img_path = images_dir / img_info['file_name']
            if img_path.exists():
                self.samples.append((img_path, count))

        print(f"[INFO] Dataset: {len(self.samples)} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, count = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(count, dtype=torch.float32)


def get_transforms(train: bool):
    if train:
        # Rotation padding is filled with black (default fill=0).
        # Rotation is applied after Resize, so the output is always 224x224.
        # Normalize scales each channel to match ImageNet statistics (mean/std),
        # which is required because EfficientNet-B0 was pretrained on ImageNet.
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(10),
            transforms.GaussianBlur(3),
            transforms.RandomErasing(p=0.2),   # Randomly masks a rectangular region with random values (20% chance). Makes the model robust to partial occlusion (e.g. dust or debris in microscope images).
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(dropout: float = 0.4) -> nn.Module:
    model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, 1),
    )
    return model


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Dataset
    dataset = CrystalDataset(args.images, args.annotations, transform=get_transforms(train=True))
    val_size = max(1, int(len(dataset) * args.val_split))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    val_ds.dataset = CrystalDataset(args.images, args.annotations, transform=get_transforms(train=False))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Model
    model = build_model(dropout=args.dropout).to(device)

    # Backbone frozen (head_only mode or Stage 1)
    if args.head_only:
        print("Mode: head only (backbone stays frozen throughout)")
    for param in model.features.parameters():
        param.requires_grad = False
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-3,
                                 weight_decay=args.weight_decay)

    # Loss
    criterion = nn.HuberLoss()

    # Scheduler (optional)
    scheduler = None
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    writer = SummaryWriter(log_dir=args.log_dir)
    best_val_loss = float("inf")
    no_improve = 0
    save_path = Path(args.checkpoint_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        # --- Stage 2: unfreeze backbone after stage1_epochs (head_only=Falseのときのみ) ---
        if not args.head_only and epoch == args.stage1_epochs + 1:
            print(f"[Epoch {epoch}] Unfreezing backbone for fine-tuning.")
            for param in model.features.parameters():
                param.requires_grad = True
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-5,
                                         weight_decay=args.weight_decay)
            if args.scheduler == "cosine":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=args.epochs - args.stage1_epochs)
            elif args.scheduler == "step":
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

        # Train
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            preds = model(images).squeeze(1)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(images)
        train_loss /= train_size

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                preds = model(images).squeeze(1)
                val_loss += criterion(preds, labels).item() * len(images)
        val_loss /= val_size

        if scheduler:
            scheduler.step()
            writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        writer.add_scalars("Loss", {"train": train_loss, "val": val_loss}, epoch)
        print(f"Epoch {epoch:3d}/{args.epochs}  train={train_loss:.4f}  val={val_loss:.4f}")

        # Early stopping & checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            torch.save(model.state_dict(), save_path / "best.pth")
            print(f"  -> Saved best model (val_loss={best_val_loss:.4f})")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    writer.close()
    print("Training complete. Run: tensorboard --logdir", args.log_dir)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def infer(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    transform = get_transforms(train=False)
    image = Image.open(args.image).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(tensor).squeeze().item()

    print(f"Predicted crystal count: {pred:.2f}  (rounded: {round(pred)})")
    return pred


# ---------------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------------

def gradcam(args):
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
    except ImportError:
        print("Install grad-cam: pip install grad-cam")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    transform = get_transforms(train=False)
    raw_image = Image.open(args.image).convert("RGB").resize((224, 224))
    input_tensor = transform(raw_image).unsqueeze(0).to(device)

    target_layers = [model.features[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=input_tensor)[0]

    rgb_img = np.array(raw_image, dtype=np.float32) / 255.0
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    out_path = Path(args.image).stem + "_gradcam.png"
    Image.fromarray(visualization).save(out_path)
    print(f"Grad-CAM saved to: {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="EfficientNet-B0 crystal counter")
    sub = parser.add_subparsers(dest="mode", required=True)

    # --- train ---
    p_train = sub.add_parser("train", help="Run training")
    p_train.add_argument("--images",         required=True,
                         help="Directory containing image files")
    p_train.add_argument("--annotations",    required=True,
                         help="COCO annotations JSON path (created by annotation_coco2.py)")
    p_train.add_argument("--epochs",         type=int,   default=30,
                         help="Maximum number of training epochs (default: 30)")
    p_train.add_argument("--batch_size",     type=int,   default=16,
                         help="Mini-batch size (default: 16)")
    p_train.add_argument("--val_split",      type=float, default=0.2,
                         help="Fraction of data reserved for validation, range 0-1 (default: 0.2)")
    p_train.add_argument("--dropout",        type=float, default=0.4,
                         help="Dropout rate applied in the classifier head, range 0-1 (default: 0.4)")
    p_train.add_argument("--weight_decay",   type=float, default=1e-4,
                         help="L2 weight decay (regularization) for Adam optimizer (default: 1e-4)")
    p_train.add_argument("--patience",       type=int,   default=5,
                         help="Early stopping patience: training stops if val_loss does not improve for this many consecutive epochs (default: 5)")
    p_train.add_argument("--stage1_epochs",  type=int,   default=10,
                         help="Number of epochs to train with backbone frozen before fine-tuning (Stage 1). Ignored when --head_only is set (default: 10)")
    p_train.add_argument("--head_only",      action="store_true",
                         help="Keep the backbone frozen for the entire training run; only the classifier head is updated. Useful for small datasets or quick experiments")
    p_train.add_argument("--scheduler",      default=None,
                         choices=["cosine", "step"],
                         help="Learning rate scheduler: 'cosine'=CosineAnnealingLR, 'step'=StepLR (lr x0.5 every 10 epochs). Omit to disable (default: None)")
    p_train.add_argument("--log_dir",        default="runs/efficientnet",
                         help="Output directory for TensorBoard logs (default: runs/efficientnet)")
    p_train.add_argument("--checkpoint_dir", default="checkpoints",
                         help="Directory where the best model weights (best.pth) are saved (default: checkpoints)")

    # --- infer ---
    p_infer = sub.add_parser("infer", help="Predict crystal count for a single image")
    p_infer.add_argument("--image",      required=True,
                         help="Path to the input image file")
    p_infer.add_argument("--checkpoint", default="checkpoints/best.pth",
                         help="Path to the trained model weights file (.pth) (default: checkpoints/best.pth)")

    # --- gradcam ---
    p_gc = sub.add_parser("gradcam", help="Visualize prediction basis with Grad-CAM (requires: pip install grad-cam)")
    p_gc.add_argument("--image",      required=True,
                      help="Path to the input image file")
    p_gc.add_argument("--checkpoint", default="checkpoints/best.pth",
                      help="Path to the trained model weights file (.pth) (default: checkpoints/best.pth)")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "train":
        train(args)
    elif args.mode == "infer":
        infer(args)
    elif args.mode == "gradcam":
        gradcam(args)
