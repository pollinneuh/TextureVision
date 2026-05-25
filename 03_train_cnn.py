#!/usr/bin/env python3
"""
04_train_cnn.py — Transfer learning with MobileNetV3-Small for texture recognition.

Architecture:
    MobileNetV3-Small (ImageNet pretrained, 2.5 MB)
    → custom head: AdaptiveAvgPool → Dropout(0.3) → Linear(N_classes)

Two-phase training strategy:
    Phase 1 (head only) : backbone frozen, only the head is trained (LR=1e-3)
    Phase 2 (fine-tune) : full network, low LR (1e-4) + weight decay

Outputs:
    cnn_output/model_cnn_best.pth   — best PyTorch weights
    cnn_output/model_cnn.onnx       — ONNX export for deployment
    cnn_output/classes.txt          — class list
    cnn_output/training_curves.png  — loss and accuracy curves

Usage:
    python 04_train_cnn.py --dataset my_dataset_preprocessed
    python 04_train_cnn.py --dataset my_dataset_preprocessed --phase1-epochs 5 --phase2-epochs 20
    python 04_train_cnn.py --dataset my_dataset_preprocessed --no-onnx
"""

import argparse
import time
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from torch.utils.data import DataLoader, Subset
    from torchvision import datasets, models, transforms
    HAS_TORCH = True
except Exception as _e:
    HAS_TORCH = False
    print(f"[WARN] PyTorch import failed: {_e}")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


IMG_SIZE   = 128   # reduced from 224 — avoids over-upsampling low-res endoscope frames
BATCH_SIZE = 32

# ImageNet normalization statistics (the backbone was trained with these)
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


# ── Transforms ────────────────────────────────────────────────────────────────

def get_transform(train: bool) -> "transforms.Compose":
    """
    Train : augmentation (rotation, flip, jitter) + ImageNet normalization.
    Val   : resize + normalization only.
    Grayscale images are converted to pseudo-RGB (3 identical channels)
    to be compatible with the pretrained backbone.
    """
    base = [
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
    ]
    aug = [
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=180),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    ] if train else []

    norm = [
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ]
    return transforms.Compose(base + aug + norm)


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model(num_classes: int) -> "nn.Module":
    """MobileNetV3-Small with a custom classification head."""
    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    model = models.mobilenet_v3_small(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes),
    )
    return model


def freeze_backbone(model: "nn.Module"):
    """Freeze all parameters except the classifier."""
    for name, param in model.named_parameters():
        param.requires_grad = ("classifier" in name)


def unfreeze_all(model: "nn.Module"):
    """Unfreeze all parameters for full fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True


# ── Train / eval loops ────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device) -> tuple:
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        out  = model(inputs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
        correct    += (out.detach().argmax(1) == labels).sum().item()
        n          += len(labels)
    return total_loss / n, correct / n


def eval_epoch(model, loader, criterion, device) -> tuple:
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            out = model(inputs)
            total_loss += criterion(out, labels).item() * len(labels)
            correct    += (out.argmax(1) == labels).sum().item()
            n          += len(labels)
    return total_loss / n, correct / n


# ── ONNX export ───────────────────────────────────────────────────────────────

def export_onnx(model: "nn.Module", save_path: Path, device):
    model.eval()
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=device)
    torch.onnx.export(
        model, dummy, str(save_path),
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    print(f"ONNX export → {save_path}")


# ── Training curves ───────────────────────────────────────────────────────────

def plot_curves(history: dict, save_path: Path):
    if not HAS_MPL:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ep = range(1, len(history["train_loss"]) + 1)
    phase_split = history.get("phase_split", None)

    for ax, key_tr, key_vl, title in [
        (ax1, "train_loss", "val_loss", "Loss"),
        (ax2, "train_acc",  "val_acc",  "Accuracy"),
    ]:
        ax.plot(ep, history[key_tr], label="Train", color="#1f77b4")
        ax.plot(ep, history[key_vl], label="Val",   color="#ff7f0e")
        if phase_split:
            ax.axvline(x=phase_split, color="grey", linestyle="--", label="Phase 2 start")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150)
    plt.close()
    print(f"Training curves → {save_path}")


# ── Training pipeline ─────────────────────────────────────────────────────────

def train(
    dataset_root: Path,
    save_dir: Path,
    epochs_phase1: int,
    epochs_phase2: int,
    no_onnx: bool,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch {torch.__version__} | Device: {device}")

    save_dir.mkdir(parents=True, exist_ok=True)

    # Dual dataset objects so train and val can have different transforms
    ds_train = datasets.ImageFolder(str(dataset_root), transform=get_transform(True))
    ds_val   = datasets.ImageFolder(str(dataset_root), transform=get_transform(False))

    classes     = ds_train.classes
    num_classes = len(classes)
    n_total     = len(ds_train)

    print(f"Dataset: {n_total} images | {num_classes} classes | device: {device}")

    # Manual stratified split (same indices for both transform variants)
    rng = np.random.default_rng(42)
    class_indices = [[] for _ in range(num_classes)]
    for idx, (_, label) in enumerate(ds_train.samples):
        class_indices[label].append(idx)

    train_idx, val_idx = [], []
    for ci in class_indices:
        rng.shuffle(ci)
        n_v = max(1, int(0.2 * len(ci)))
        val_idx   += ci[:n_v]
        train_idx += ci[n_v:]

    print(f"Train: {len(train_idx)} | Val: {len(val_idx)}")

    train_loader = DataLoader(
        Subset(ds_train, train_idx),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True,
    )
    val_loader = DataLoader(
        Subset(ds_val, val_idx),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True,
    )

    model     = build_model(num_classes).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    history = {k: [] for k in ("train_loss", "train_acc", "val_loss", "val_acc")}
    best_val_acc = 0.0
    best_ckpt    = save_dir / "model_cnn_best.pth"

    def _save_best(val_acc):
        nonlocal best_val_acc
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {"model_state": model.state_dict(), "classes": classes, "val_acc": val_acc},
                str(best_ckpt),
            )

    # ── Phase 1: head only ────────────────────────────────────────────────────
    if epochs_phase1 > 0:
        freeze_backbone(model)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n─── Phase 1: head only ({epochs_phase1} epochs, {n_params:,} params) ───")

        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3,
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs_phase1, eta_min=1e-5)

        for ep in range(1, epochs_phase1 + 1):
            t0 = time.time()
            tr_l, tr_a = train_epoch(model, train_loader, optimizer, criterion, device)
            vl_l, vl_a = eval_epoch(model, val_loader, criterion, device)
            scheduler.step()
            history["train_loss"].append(tr_l)
            history["train_acc"].append(tr_a)
            history["val_loss"].append(vl_l)
            history["val_acc"].append(vl_a)
            _save_best(vl_a)
            print(f"  Ep {ep:02d}/{epochs_phase1} | "
                  f"loss={tr_l:.4f} acc={tr_a:.3f} | "
                  f"val_loss={vl_l:.4f} val_acc={vl_a:.3f} | "
                  f"{time.time()-t0:.1f}s")

    history["phase_split"] = len(history["train_loss"])

    # ── Phase 2: full fine-tuning ─────────────────────────────────────────────
    if epochs_phase2 > 0:
        unfreeze_all(model)

        # Reload best checkpoint from phase 1
        if best_ckpt.exists():
            ckpt = torch.load(str(best_ckpt), map_location=device)
            model.load_state_dict(ckpt["model_state"])

        n_params = sum(p.numel() for p in model.parameters())
        print(f"\n─── Phase 2: fine-tuning ({epochs_phase2} epochs, {n_params:,} params) ───")

        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs_phase2, eta_min=1e-6)

        for ep in range(1, epochs_phase2 + 1):
            t0 = time.time()
            tr_l, tr_a = train_epoch(model, train_loader, optimizer, criterion, device)
            vl_l, vl_a = eval_epoch(model, val_loader, criterion, device)
            scheduler.step()
            history["train_loss"].append(tr_l)
            history["train_acc"].append(tr_a)
            history["val_loss"].append(vl_l)
            history["val_acc"].append(vl_a)
            _save_best(vl_a)
            print(f"  Ep {ep:02d}/{epochs_phase2} | "
                  f"loss={tr_l:.4f} acc={tr_a:.3f} | "
                  f"val_loss={vl_l:.4f} val_acc={vl_a:.3f} | "
                  f"{time.time()-t0:.1f}s")

    print(f"\nBest val accuracy: {best_val_acc:.4f}")

    # ── ONNX export ───────────────────────────────────────────────────────────
    if not no_onnx and best_ckpt.exists():
        ckpt = torch.load(str(best_ckpt), map_location=device)
        model.load_state_dict(ckpt["model_state"])
        export_onnx(model, save_dir / "model_cnn.onnx", device)

    # ── Curves ────────────────────────────────────────────────────────────────
    plot_curves(history, save_dir / "training_curves.png")

    # ── Classes ───────────────────────────────────────────────────────────────
    (save_dir / "classes.txt").write_text("\n".join(classes))

    print(f"\nOutput files in: {save_dir}/")
    print(f"  model_cnn_best.pth  — best weights")
    print(f"  model_cnn.onnx      — ONNX export")
    print(f"  classes.txt         — class list")
    print(f"  training_curves.png — curves")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    if not HAS_TORCH:
        print("PyTorch is not installed. Run: pip install torch torchvision")
        return

    parser = argparse.ArgumentParser(
        description="MobileNetV3-Small transfer learning for texture recognition"
    )
    parser.add_argument("--dataset", required=True,
                        help="Dataset directory (ImageFolder structure)")
    parser.add_argument("--output", default="cnn_output",
                        help="Output directory (default: cnn_output)")
    parser.add_argument("--phase1-epochs", type=int, default=5,
                        help="Phase 1 epochs — head only (default: 5)")
    parser.add_argument("--phase2-epochs", type=int, default=20,
                        help="Phase 2 epochs — full fine-tuning (default: 20)")
    parser.add_argument("--no-onnx", action="store_true",
                        help="Skip ONNX export")
    args = parser.parse_args()

    train(
        Path(args.dataset),
        Path(args.output),
        args.phase1_epochs,
        args.phase2_epochs,
        args.no_onnx,
    )


if __name__ == "__main__":
    main()
