#!/usr/bin/env python3
"""
05_evaluate.py — Comprehensive evaluation of a texture recognition model.

Generates in the output directory:
    confusion_matrix.png    — confusion matrix
    per_class_accuracy.png  — per-class accuracy (recall)
    tsne.png                — t-SNE visualization of the feature space
    report.txt              — full text report

Supports both model types:
    LBP+SVM  (.pkl file)
    CNN       (.pth file)

Usage:
    python 05_evaluate.py --model model_lbp.pkl --dataset my_dataset_preprocessed --type lbp
    python 05_evaluate.py --model cnn_output/model_cnn_best.pth --dataset my_dataset_preprocessed --type cnn
"""

import argparse
import pickle
from pathlib import Path

import cv2
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_VIZ = True
except ImportError:
    HAS_VIZ = False

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    top_k_accuracy_score,
)
from sklearn.manifold import TSNE

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw):
        return it


CROP_SIZE = 175
IMG_SIZE  = 128


# ── Shared LBP utilities ──────────────────────────────────────────────────────

def _center_crop(img, size):
    h, w = img.shape[:2]
    if h < size or w < size:
        pad_h = max(0, size - h)
        pad_w = max(0, size - w)
        img = cv2.copyMakeBorder(img, pad_h//2, pad_h-pad_h//2,
                                  pad_w//2, pad_w-pad_w//2, cv2.BORDER_REFLECT)
        h, w = img.shape[:2]
    cy, cx = h//2, w//2
    half = size//2
    return img[cy-half:cy+half, cx-half:cx+half]


def _apply_clahe(gray):
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)


def _lbp_hist(gray, P, R):
    try:
        from skimage.feature import local_binary_pattern
        lbp = local_binary_pattern(gray, P, R, method="uniform")
        hist, _ = np.histogram(lbp, bins=P+2, range=(0, P+2))
    except ImportError:
        hist = np.zeros(P+2, dtype=int)
    return hist.astype(float) / (hist.sum() + 1e-9)


def extract_lbp_features(img):
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    gray = _center_crop(gray, CROP_SIZE)
    gray = _apply_clahe(gray)
    feats = [
        _lbp_hist(gray, 8, 1), _lbp_hist(gray, 16, 2),
        _lbp_hist(cv2.resize(gray, (CROP_SIZE//2, CROP_SIZE//2)), 8, 1),
        _lbp_hist(cv2.resize(gray, (CROP_SIZE//2, CROP_SIZE//2)), 16, 2),
        _lbp_hist(cv2.resize(gray, (CROP_SIZE//4, CROP_SIZE//4)), 8, 1),
    ]
    return np.concatenate(feats)


# ── Dataset loading ───────────────────────────────────────────────────────────

def load_lbp_dataset(dataset_root: Path):
    classes = sorted([d.name for d in dataset_root.iterdir() if d.is_dir()])
    X, y, paths = [], [], []
    for label_idx, cls_name in enumerate(tqdm(classes, desc="LBP features")):
        cls_dir = dataset_root / cls_name
        for img_path in list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.png")):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            X.append(extract_lbp_features(img))
            y.append(label_idx)
            paths.append(img_path)
    return np.array(X), np.array(y), classes, paths


# ── Visualizations ────────────────────────────────────────────────────────────

def _plot_confusion_matrix(cm, classes, save_path):
    if not HAS_VIZ:
        return
    n = len(classes)
    fig_size = max(10, n // 2)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    annot = n <= 25
    sns.heatmap(
        cm, annot=annot, fmt="d", cmap="Blues",
        xticklabels=classes, yticklabels=classes, ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150)
    plt.close()
    print(f"Confusion matrix → {save_path}")


def _plot_per_class(report_dict, classes, save_path):
    if not HAS_VIZ:
        return
    recalls = [report_dict.get(cls, {}).get("recall", 0.0) for cls in classes]
    mean_r  = float(np.mean(recalls))

    fig, ax = plt.subplots(figsize=(max(12, len(classes) // 2), 5))
    colors  = ["#d62728" if r < 0.6 else "#ff7f0e" if r < 0.8 else "#2ca02c"
               for r in recalls]
    ax.bar(classes, recalls, color=colors)
    ax.axhline(mean_r, color="navy", linestyle="--", linewidth=2,
               label=f"Mean: {mean_r:.3f}")
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("Class")
    ax.set_ylabel("Recall (per-class accuracy)")
    ax.set_title("Per-Class Accuracy")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    ax.legend()
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150)
    plt.close()
    print(f"Per-class accuracy → {save_path}")


def _plot_tsne(X, y, classes, save_path, n_samples=3000):
    if not HAS_VIZ:
        return
    if len(X) > n_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X), n_samples, replace=False)
        X, y = X[idx], y[idx]

    print("Computing t-SNE (may take 1–2 minutes) …")
    X_2d = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000).fit_transform(X)

    fig, ax = plt.subplots(figsize=(13, 11))
    cmap = plt.cm.get_cmap("tab20", len(classes))
    for label_idx, cls_name in enumerate(classes):
        mask = y == label_idx
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   c=[cmap(label_idx)], label=cls_name, alpha=0.55, s=18)
    ax.set_title("t-SNE — Feature Space", fontsize=14)
    ax.legend(loc="upper right", fontsize=6, ncol=max(1, len(classes)//15))
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150)
    plt.close()
    print(f"t-SNE → {save_path}")


# ── LBP+SVM evaluation ───────────────────────────────────────────────────────

def evaluate_lbp(model_path: Path, dataset_root: Path, output_dir: Path):
    print("\nLoading LBP model …")
    with open(model_path, "rb") as f:
        data = pickle.load(f)
    model   = data["model"]
    classes = data["classes"]

    print("Extracting features …")
    X, y, _, paths = load_lbp_dataset(dataset_root)

    print(f"Predicting {len(X)} samples …")
    y_pred  = model.predict(X)
    y_proba = model.predict_proba(X)

    acc = accuracy_score(y, y_pred)
    print(f"\nOverall accuracy : {acc:.4f}")

    try:
        top3 = top_k_accuracy_score(y, y_proba, k=min(3, len(classes) - 1))
        print(f"Top-3 accuracy   : {top3:.4f}")
    except Exception:
        pass

    report_str  = classification_report(y, y_pred, target_names=classes)
    report_dict = classification_report(y, y_pred, target_names=classes, output_dict=True)
    print("\n" + report_str)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.txt").write_text(
        f"Overall accuracy: {acc:.4f}\n\n{report_str}"
    )

    cm = confusion_matrix(y, y_pred)
    _plot_confusion_matrix(cm, classes, output_dir / "confusion_matrix.png")
    _plot_per_class(report_dict, classes, output_dir / "per_class_accuracy.png")
    _plot_tsne(X, y, classes, output_dir / "tsne.png")

    # 5 hardest classes
    recalls = {cls: report_dict.get(cls, {}).get("recall", 0.0) for cls in classes}
    worst   = sorted(recalls.items(), key=lambda x: x[1])[:5]
    print("\n5 hardest classes (lowest recall):")
    for cls, r in worst:
        print(f"  {cls}: {r:.3f}")


# ── CNN evaluation ────────────────────────────────────────────────────────────

def evaluate_cnn(model_path: Path, dataset_root: Path, output_dir: Path, classes_file: Path):
    try:
        import torch
        from torchvision import models, transforms
        import torch.nn as nn
        from PIL import Image
    except ImportError:
        print("PyTorch not installed. Run: pip install torch torchvision")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Classes
    if classes_file.exists():
        classes = [l.strip() for l in classes_file.read_text().splitlines() if l.strip()]
    else:
        classes = sorted([d.name for d in dataset_root.iterdir() if d.is_dir()])

    num_classes = len(classes)
    print(f"\nClasses: {num_classes}")

    # Rebuild model
    ckpt = torch.load(str(model_path), map_location=device)
    m = models.mobilenet_v3_small(weights=None)
    in_features = m.classifier[-1].in_features
    m.classifier[-1] = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_features, num_classes))
    m.load_state_dict(ckpt["model_state"])
    m = m.to(device).eval()

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Hook to extract penultimate-layer features (for t-SNE)
    feats_cache = []
    def _hook(module, inp, out):
        feats_cache.append(out.detach().cpu().numpy().flatten())
    handle = m.classifier[0].register_forward_hook(_hook)

    y_all, probs_all, feats_all = [], [], []
    print("CNN inference …")
    with torch.no_grad():
        for label_idx, cls_name in enumerate(tqdm(classes, desc="Classes")):
            cls_dir = dataset_root / cls_name
            for img_path in list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.png")):
                try:
                    img = Image.open(str(img_path)).convert("RGB")
                except Exception:
                    continue
                tensor = transform(img).unsqueeze(0).to(device)
                feats_cache.clear()
                logits = m(tensor)
                probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]
                y_all.append(label_idx)
                probs_all.append(probs)
                if feats_cache:
                    feats_all.append(feats_cache[0])

    handle.remove()
    y_all     = np.array(y_all)
    probs_all = np.array(probs_all)
    feats_all = np.array(feats_all)

    y_pred = probs_all.argmax(axis=1)
    acc    = accuracy_score(y_all, y_pred)
    print(f"\nOverall accuracy : {acc:.4f}")

    try:
        top3 = top_k_accuracy_score(y_all, probs_all, k=min(3, num_classes - 1))
        print(f"Top-3 accuracy   : {top3:.4f}")
    except Exception:
        pass

    report_str  = classification_report(y_all, y_pred, target_names=classes)
    report_dict = classification_report(y_all, y_pred, target_names=classes, output_dict=True)
    print("\n" + report_str)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.txt").write_text(
        f"Overall accuracy: {acc:.4f}\n\n{report_str}"
    )

    cm = confusion_matrix(y_all, y_pred)
    _plot_confusion_matrix(cm, classes, output_dir / "confusion_matrix.png")
    _plot_per_class(report_dict, classes, output_dir / "per_class_accuracy.png")
    if len(feats_all) > 0:
        _plot_tsne(feats_all, y_all, classes, output_dir / "tsne.png")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive evaluation of a texture recognition model"
    )
    parser.add_argument("--model", required=True,
                        help="Model file (.pkl for LBP, .pth for CNN)")
    parser.add_argument("--dataset", required=True,
                        help="Test dataset directory")
    parser.add_argument("--type", choices=["lbp", "cnn"], required=True,
                        help="Model type: 'lbp' or 'cnn'")
    parser.add_argument("--output", default="eval_output",
                        help="Output directory for plots (default: eval_output)")
    parser.add_argument("--classes", default=None,
                        help="Path to classes.txt (CNN only, auto-detected otherwise)")
    args = parser.parse_args()

    model_path   = Path(args.model)
    dataset_root = Path(args.dataset)
    output_dir   = Path(args.output)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_root}")

    if args.type == "lbp":
        evaluate_lbp(model_path, dataset_root, output_dir)
    else:
        classes_file = (
            Path(args.classes) if args.classes
            else model_path.parent / "classes.txt"
        )
        evaluate_cnn(model_path, dataset_root, output_dir, classes_file)


if __name__ == "__main__":
    main()
