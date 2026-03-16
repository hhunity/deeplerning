"""
EfficientNet-B0 for crystal count regression or classification.
Usage:
  Train (regression):      python efficientnet_main.py train --images ./data/images --annotations ./data/annotations.json
  Train (classification):  python efficientnet_main.py train --images ./data/images --annotations ./data/annotations.json --task classification --num_classes 11
  Infer:                   python efficientnet_main.py infer --image path/to/image.png --checkpoint best.pth
  Validate:                python efficientnet_main.py validate --images ./data/images --annotations ./data/annotations.json

Size: 224x224
"""

import argparse
import datetime
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
        # Return count as int; caller casts to float (regression) or long (classification)
        return image, int(count)


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

def build_model(dropout: float = 0.4, num_classes: int = 1) -> nn.Module:
    """num_classes=1 → regression output; num_classes>1 → classification output."""
    model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
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

    # Task setup
    is_classification = (args.task == "classification")
    num_classes = args.num_classes if is_classification else 1
    print(f"Task: {args.task}" + (f"  (num_classes={num_classes})" if is_classification else ""))

    # Model
    model = build_model(dropout=args.dropout, num_classes=num_classes).to(device)

    # Backbone frozen (head_only mode or Stage 1)
    if args.head_only:
        print("Mode: head only (backbone stays frozen throughout)")
    for param in model.features.parameters():
        param.requires_grad = False
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-3,
                                 weight_decay=args.weight_decay)

    # Loss
    criterion = nn.CrossEntropyLoss() if is_classification else nn.HuberLoss()

    # Scheduler (optional)
    scheduler = None
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(args.log_dir) / timestamp
    writer = SummaryWriter(log_dir=str(log_dir))
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
            images = images.to(device)
            labels = labels.long().to(device) if is_classification else labels.float().to(device)
            optimizer.zero_grad()
            preds = model(images) if is_classification else model(images).squeeze(1)
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
                images = images.to(device)
                labels = labels.long().to(device) if is_classification else labels.float().to(device)
                preds = model(images) if is_classification else model(images).squeeze(1)
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
            torch.save({
                "model"      : model.state_dict(),
                "task"       : args.task,
                "num_classes": num_classes,
            }, save_path / "best.pth")
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

def _load_checkpoint(checkpoint_path, device):
    """Load model from checkpoint. Supports both legacy (state_dict only) and new (dict) format."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        task        = ckpt.get("task", "regression")
        num_classes = ckpt.get("num_classes", 1)
        state_dict  = ckpt["model"]
    else:
        task        = "regression"
        num_classes = 1
        state_dict  = ckpt
    model = build_model(num_classes=num_classes).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, task, num_classes


def infer(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, task, num_classes = _load_checkpoint(args.checkpoint, device)

    transform = get_transforms(train=False)
    raw_image = Image.open(args.image).convert("RGB").resize((224, 224))
    tensor = transform(raw_image).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(tensor)
        if task == "classification":
            probs      = torch.softmax(out, dim=1)[0]
            pred_class = probs.argmax().item()
            confidence = probs[pred_class].item()
            pred_label = f"count={pred_class} ({confidence:.2%})"
            print(f"Predicted crystal count: {pred_class}  (confidence: {confidence:.2%})")
            result = pred_class
        else:
            pred = out.squeeze().item()
            pred_label = f"count={pred:.2f} (rounded={round(pred)})"
            print(f"Predicted crystal count: {pred:.2f}  (rounded: {round(pred)})")
            result = pred

    if args.gradcam:
        try:
            import matplotlib.pyplot as plt
            from pytorch_grad_cam import GradCAM
            from pytorch_grad_cam.utils.image import show_cam_on_image
        except ImportError:
            print("Install required packages: pip install grad-cam matplotlib")
            return result

        target_layers = [model.features[-1]]
        cam = GradCAM(model=model, target_layers=target_layers)
        grayscale_cam = cam(input_tensor=tensor)[0]

        rgb_img = np.array(raw_image, dtype=np.float32) / 255.0
        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        if task == "classification":
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(raw_image)
            axes[0].set_title("Original")
            axes[0].axis("off")
            axes[1].imshow(cam_image)
            axes[1].set_title(f"Grad-CAM  {pred_label}")
            axes[1].axis("off")
            num_cls = probs.shape[0]
            colors = ["tomato" if i == pred_class else "steelblue" for i in range(num_cls)]
            axes[2].bar(range(num_cls), probs.cpu().numpy(), color=colors)
            axes[2].set_xlabel("Crystal count (class)")
            axes[2].set_ylabel("Probability")
            axes[2].set_title("Class Probabilities")
            axes[2].set_xticks(range(num_cls))
        else:
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(raw_image)
            axes[0].set_title("Original")
            axes[0].axis("off")
            axes[1].imshow(cam_image)
            axes[1].set_title(f"Grad-CAM  {pred_label}")
            axes[1].axis("off")
        fig.tight_layout()

        out_path = Path(args.image).stem + "_gradcam.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Grad-CAM saved to: {out_path}")

    return result


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(args):
    import json
    import matplotlib.pyplot as plt

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, task, _ = _load_checkpoint(args.checkpoint, device)

    with open(args.annotations) as f:
        coco = json.load(f)

    count_map = {}
    for ann in coco["annotations"]:
        iid = ann["image_id"]
        count_map[iid] = count_map.get(iid, 0) + 1

    transform = get_transforms(train=False)
    images_dir = Path(args.images)

    filenames, truths, preds = [], [], []
    for img_info in coco["images"]:
        img_path = images_dir / img_info["file_name"]
        if not img_path.exists():
            continue
        truth = count_map.get(img_info["id"], 0)

        raw = Image.open(img_path).convert("RGB").resize((224, 224))
        tensor = transform(raw).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(tensor)
            if task == "classification":
                pred = torch.softmax(out, dim=1)[0].argmax().item()
            else:
                pred = out.squeeze().item()

        filenames.append(img_info["file_name"])
        truths.append(truth)
        preds.append(pred)

    truths = np.array(truths, dtype=float)
    preds  = np.array(preds,  dtype=float)
    errors = preds - truths
    mae    = np.mean(np.abs(errors))
    rmse   = np.sqrt(np.mean(errors ** 2))
    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((truths - truths.mean()) ** 2)
    r2     = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    print("\nValidation Results")
    print("==================")
    print(f"MAE  : {mae:.3f}")
    print(f"RMSE : {rmse:.3f}")
    print(f"R²   : {r2:.3f}")
    print()
    print(f"{'image':<30} {'truth':>6} {'pred':>7} {'error':>7}")
    print("-" * 55)
    for fname, t, p, e in zip(filenames, truths, preds, errors):
        print(f"{fname:<30} {t:>6.0f} {p:>7.2f} {e:>+7.2f}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Scatter plot: prediction vs truth
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(truths, preds, alpha=0.7, edgecolors="k", linewidths=0.5)
    lim = max(truths.max(), preds.max()) * 1.05
    ax.plot([0, lim], [0, lim], "r--", linewidth=1, label="ideal")
    ax.set_xlabel("Ground truth")
    ax.set_ylabel("Prediction")
    ax.set_title(f"Prediction vs Truth  (MAE={mae:.2f}, R²={r2:.3f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "scatter_plot.png", dpi=150)
    plt.close(fig)

    # Error histogram
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(errors, bins=20, edgecolor="k")
    ax.axvline(0, color="r", linestyle="--", linewidth=1)
    ax.set_xlabel("Error (pred - truth)")
    ax.set_ylabel("Count")
    ax.set_title(f"Error Distribution  (RMSE={rmse:.2f})")
    fig.tight_layout()
    fig.savefig(out_dir / "error_hist.png", dpi=150)
    plt.close(fig)

    # Per-count error breakdown
    unique_counts = sorted(set(truths.astype(int).tolist()))
    breakdown = []
    for c in unique_counts:
        mask = truths == c
        e = errors[mask]
        breakdown.append((c, mask.sum(), np.mean(np.abs(e)), np.sqrt(np.mean(e**2)), np.mean(e)))

    print("\nPer-count Error Breakdown")
    print("=========================")
    print(f"{'truth':>6} {'n':>4} {'MAE':>7} {'RMSE':>7} {'mean_err':>10}")
    print("-" * 40)
    for c, n, c_mae, c_rmse, c_mean in breakdown:
        print(f"{c:>6} {n:>4} {c_mae:>7.2f} {c_rmse:>7.2f} {c_mean:>+10.2f}")

    counts_arr  = [b[0] for b in breakdown]
    maes_arr    = [b[2] for b in breakdown]
    ns_arr      = [b[1] for b in breakdown]
    fig, ax = plt.subplots(figsize=(max(6, len(counts_arr) * 0.7), 4))
    bars = ax.bar(range(len(counts_arr)), maes_arr, edgecolor="k")
    for bar, n in zip(bars, ns_arr):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"n={n}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(counts_arr)))
    ax.set_xticklabels(counts_arr)
    ax.set_xlabel("Ground truth (crystal count)")
    ax.set_ylabel("MAE")
    ax.set_title("MAE per Crystal Count")
    fig.tight_layout()
    fig.savefig(out_dir / "per_count_mae.png", dpi=150)
    plt.close(fig)

    # Confusion matrix (classification only)
    if task == "classification":
        labels = sorted(set(truths.astype(int).tolist()) | set(np.round(preds).astype(int).tolist()))
        label_to_idx = {l: i for i, l in enumerate(labels)}
        n = len(labels)
        cm = np.zeros((n, n), dtype=int)
        for t, p in zip(truths.astype(int), np.round(preds).astype(int)):
            if t in label_to_idx and p in label_to_idx:
                cm[label_to_idx[t], label_to_idx[p]] += 1

        fig, ax = plt.subplots(figsize=(max(5, n * 0.7), max(5, n * 0.7)))
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        fig.colorbar(im, ax=ax)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Ground truth")
        ax.set_title("Confusion Matrix")
        thresh = cm.max() / 2
        for i in range(n):
            for j in range(n):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black", fontsize=9)
        fig.tight_layout()
        fig.savefig(out_dir / "confusion_matrix.png", dpi=150)
        plt.close(fig)
        print(f"Confusion matrix saved to: {out_dir}/confusion_matrix.png")

    # CSV export
    import csv
    csv_path = out_dir / "results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "truth", "pred", "error", "abs_error"])
        for fname, t, p, e in zip(filenames, truths, preds, errors):
            writer.writerow([fname, int(t), round(p, 4), round(e, 4), round(abs(e), 4)])

    print(f"\nPlots saved to: {out_dir}/")
    print(f"CSV  saved to: {csv_path}")

    # Grad-CAM for worst predictions
    if args.top_errors > 0:
        try:
            from pytorch_grad_cam import GradCAM
            from pytorch_grad_cam.utils.image import show_cam_on_image
        except ImportError:
            print("Install grad-cam to use --top_errors: pip install grad-cam")
            return

        worst_indices = np.argsort(np.abs(errors))[::-1][:args.top_errors]
        gradcam_dir = out_dir / "top_errors"
        gradcam_dir.mkdir(exist_ok=True)

        target_layers = [model.features[-1]]
        cam = GradCAM(model=model, target_layers=target_layers)

        print(f"\nTop-{args.top_errors} errors — Grad-CAM:")
        for rank, idx in enumerate(worst_indices, start=1):
            img_path = images_dir / filenames[idx]
            raw = Image.open(img_path).convert("RGB").resize((224, 224))
            tensor = transform(raw).unsqueeze(0).to(device)

            grayscale_cam = cam(input_tensor=tensor)[0]
            rgb_img = np.array(raw, dtype=np.float32) / 255.0
            cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

            title = f"truth={truths[idx]:.0f}  pred={preds[idx]:.2f}  error={errors[idx]:+.2f}"
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(raw)
            axes[0].set_title(f"#{rank}  {filenames[idx]}")
            axes[0].axis("off")
            axes[1].imshow(cam_image)
            axes[1].set_title(title)
            axes[1].axis("off")
            fig.tight_layout()

            out_path = gradcam_dir / f"{rank:02d}_{Path(filenames[idx]).stem}_gradcam.png"
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            print(f"  #{rank:2d}  {filenames[idx]:<30}  {title}  -> {out_path.name}")


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
    p_train.add_argument("--task",           default="regression",
                         choices=["regression", "classification"],
                         help="regression: HuberLoss scalar output / classification: CrossEntropyLoss (default: regression)")
    p_train.add_argument("--num_classes",    type=int, default=11,
                         help="Number of classes for classification task (default: 11, i.e. 0-10 crystals)")
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
    p_infer.add_argument("--gradcam", action="store_true",
                         help="Also output Grad-CAM visualization alongside prediction (requires: pip install grad-cam matplotlib)")

    # --- validate ---
    p_val = sub.add_parser("validate", help="Batch validation: compare predictions against ground truth labels")
    p_val.add_argument("--images",      required=True,
                       help="Directory containing image files")
    p_val.add_argument("--annotations", required=True,
                       help="COCO annotations JSON path")
    p_val.add_argument("--checkpoint",  default="checkpoints/best.pth",
                       help="Path to the trained model weights file (.pth) (default: checkpoints/best.pth)")
    p_val.add_argument("--out_dir",     default="validation_results",
                       help="Directory to save scatter_plot.png and error_hist.png (default: validation_results)")
    p_val.add_argument("--top_errors", type=int, default=0,
                       help="Output Grad-CAM for the N worst predictions (default: 0 = disabled, requires: pip install grad-cam)")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "train":
        train(args)
    elif args.mode == "infer":
        infer(args)
    elif args.mode == "validate":
        validate(args)
