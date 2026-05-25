#!/usr/bin/env python3
"""
02_preprocess_and_augment.py — Dataset validation, preprocessing and augmentation.

Steps applied in order:
  1. Validation  : detect corrupt / too blurry / badly exposed images
  2. Preprocessing: centered crop (175 px) + CLAHE + unsharp mask
  3. Augmentation : generate N augmented copies per original image

Usage:
    python 02_preprocess_and_augment.py --dataset my_dataset --augment-factor 5
    python 02_preprocess_and_augment.py --dataset my_dataset --validate-only
    python 02_preprocess_and_augment.py --dataset my_dataset --no-augment
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw):
        return it


CROP_SIZE = 175   # must match the model's recognition_crop_size parameter


# ── Image utilities ───────────────────────────────────────────────────────────

def center_crop(img: np.ndarray, size: int) -> np.ndarray:
    """Centered square crop. Adds reflection padding if the image is too small."""
    h, w = img.shape[:2]
    if h < size or w < size:
        pad_h = max(0, size - h)
        pad_w = max(0, size - w)
        img = cv2.copyMakeBorder(
            img, pad_h // 2, pad_h - pad_h // 2,
            pad_w // 2, pad_w - pad_w // 2,
            cv2.BORDER_REFLECT,
        )
        h, w = img.shape[:2]
    cy, cx = h // 2, w // 2
    half = size // 2
    return img[cy - half: cy + half, cx - half: cx + half]


def apply_clahe(gray: np.ndarray, clip: float = 2.0, grid: int = 8) -> np.ndarray:
    """CLAHE — normalizes non-uniform LED illumination."""
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
    return clahe.apply(gray)


def unsharp_mask(gray: np.ndarray, kernel: int = 3, strength: float = 1.5) -> np.ndarray:
    """Recover perceived sharpness lost to low-res optics without inventing detail."""
    blurred = cv2.GaussianBlur(gray, (kernel, kernel), 0)
    return cv2.addWeighted(gray, 1.0 + strength, blurred, -strength, 0)


def elastic_deform(img: np.ndarray, alpha: float = 20.0, sigma: float = 4.0) -> np.ndarray:
    """Smooth random warp — simulates surface curvature under the glove."""
    h, w = img.shape[:2]
    rng = np.random.default_rng()
    dx = cv2.GaussianBlur(
        (rng.random((h, w)) * 2 - 1).astype(np.float32), (0, 0), sigma
    ) * alpha
    dy = cv2.GaussianBlur(
        (rng.random((h, w)) * 2 - 1).astype(np.float32), (0, 0), sigma
    ) * alpha
    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
    map_x = np.clip(xs + dx, 0, w - 1).astype(np.float32)
    map_y = np.clip(ys + dy, 0, h - 1).astype(np.float32)
    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def preprocess(img: np.ndarray) -> np.ndarray:
    """Full pipeline: BGR → grayscale → centered crop → CLAHE → unsharp mask."""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    cropped = center_crop(gray, CROP_SIZE)
    enhanced = apply_clahe(cropped)
    return unsharp_mask(enhanced)  # uint8 grayscale


# ── Augmentations ─────────────────────────────────────────────────────────────

def augment_image(img: np.ndarray) -> list:
    """
    Generate a list of augmented versions of the input image.
    Augmentations are chosen to reflect realistic acquisition variability
    (glove / endoscope on a surface).

    Input : numpy array (H, W) grayscale.
    Returns: list of numpy arrays of the same size.
    """
    augmented = []
    h, w = img.shape[:2]

    # 1. Random rotations — textures are orientation-invariant relative to the glove
    for angle in random.sample(range(-180, 181, 5), 6):
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        rot = cv2.warpAffine(img, M, (w, h),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT)
        augmented.append(rot)

    # 2. Flips (textures are often symmetric)
    augmented.append(cv2.flip(img, 1))   # horizontal
    augmented.append(cv2.flip(img, 0))   # vertical
    augmented.append(cv2.flip(img, -1))  # both

    # 3. Brightness / contrast jitter (LED illumination variation)
    for _ in range(3):
        alpha = random.uniform(0.65, 1.35)   # contrast
        beta  = random.randint(-40, 40)       # brightness
        jitter = np.clip(alpha * img.astype(float) + beta, 0, 255).astype(np.uint8)
        augmented.append(jitter)

    # 4. Gaussian noise (sensor noise, especially common on miniature cameras)
    for _ in range(3):
        sigma = random.uniform(3, 18)
        noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
        noisy = np.clip(img.astype(float) + noise, 0, 255).astype(np.uint8)
        augmented.append(noisy)

    # 5. Slight blur (endoscope focus variation)
    ksize = random.choice([3, 5])
    augmented.append(cv2.GaussianBlur(img, (ksize, ksize), 0))

    # 6. Slight perspective warp (endoscope not always perfectly perpendicular)
    pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    jitter_px = random.randint(5, 15)
    pts2 = np.float32([
        [random.randint(0, jitter_px),      random.randint(0, jitter_px)],
        [w - random.randint(0, jitter_px),  random.randint(0, jitter_px)],
        [random.randint(0, jitter_px),      h - random.randint(0, jitter_px)],
        [w - random.randint(0, jitter_px),  h - random.randint(0, jitter_px)],
    ])
    M_persp = cv2.getPerspectiveTransform(pts1, pts2)
    warped = cv2.warpPerspective(img, M_persp, (w, h), borderMode=cv2.BORDER_REFLECT)
    augmented.append(warped)

    # 7. Elastic deformation (surface curvature / glove pressure variation)
    augmented.append(elastic_deform(img))

    return augmented


# ── Validation ────────────────────────────────────────────────────────────────

def validate_dataset(dataset_root: Path) -> dict:
    """
    Scan the dataset, validate each image, report issues.
    Returns a statistics dictionary.
    """
    classes = sorted([d for d in dataset_root.iterdir() if d.is_dir()])
    print(f"\nValidating {len(classes)} classes in {dataset_root} …")

    stats = {
        "total": 0, "valid": 0, "corrupt": 0,
        "blurry": 0, "dark": 0, "classes": {}
    }

    for cls_dir in tqdm(classes, desc="Validating"):
        images = sorted(cls_dir.glob("*.jpg")) + sorted(cls_dir.glob("*.png"))
        cls_stats = {"total": len(images), "valid": 0, "issues": []}

        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None:
                cls_stats["issues"].append(f"CORRUPT: {img_path.name}")
                stats["corrupt"] += 1
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            brightness = gray.mean()
            # Measure sharpness on center only — avoids LED ring artifacts
            h, w = gray.shape
            cy, cx = h // 2, w // 2
            r = min(h, w) // 3
            center = gray[cy - r:cy + r, cx - r:cx + r]
            blur = cv2.Laplacian(center, cv2.CV_64F).var()

            if blur < 5.0:
                cls_stats["issues"].append(f"BLURRY ({blur:.1f}): {img_path.name}")
                stats["blurry"] += 1
                continue
            if brightness < 15:
                cls_stats["issues"].append(f"DARK ({brightness:.1f}): {img_path.name}")
                stats["dark"] += 1
                continue

            cls_stats["valid"] += 1
            stats["valid"] += 1

        stats["total"] += len(images)
        stats["classes"][cls_dir.name] = cls_stats

    print(
        f"\nTotal: {stats['total']} | Valid: {stats['valid']} | "
        f"Corrupt: {stats['corrupt']} | Blurry: {stats['blurry']} | "
        f"Too dark: {stats['dark']}"
    )

    if stats["classes"]:
        min_cls = min(stats["classes"], key=lambda k: stats["classes"][k]["valid"])
        max_cls = max(stats["classes"], key=lambda k: stats["classes"][k]["valid"])
        print(f"Smallest class: '{min_cls}' "
              f"({stats['classes'][min_cls]['valid']} valid images)")
        print(f"Largest class:  '{max_cls}' "
              f"({stats['classes'][max_cls]['valid']} valid images)")

        # Show issues for the 3 worst classes
        worst = sorted(
            stats["classes"].items(),
            key=lambda x: x[1]["valid"]
        )[:3]
        for cls_name, cstats in worst:
            if cstats["issues"]:
                print(f"\n  Issues in '{cls_name}':")
                for issue in cstats["issues"][:5]:
                    print(f"    • {issue}")

    return stats


# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess_dataset(dataset_root: Path, output_root: Path):
    """Apply crop + CLAHE to all images and save to output_root."""
    classes = sorted([d for d in dataset_root.iterdir() if d.is_dir()])
    output_root.mkdir(parents=True, exist_ok=True)
    total_processed = 0

    print(f"\nPreprocessing {len(classes)} classes → {output_root} …")

    for cls_dir in tqdm(classes, desc="Preprocessing"):
        out_cls = output_root / cls_dir.name
        out_cls.mkdir(parents=True, exist_ok=True)

        images = sorted(cls_dir.glob("*.jpg")) + sorted(cls_dir.glob("*.png"))
        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            processed = preprocess(img)
            cv2.imwrite(str(out_cls / img_path.name), processed)
            total_processed += 1

    print(f"  {total_processed} images preprocessed.")


# ── Augmentation ──────────────────────────────────────────────────────────────

def augment_dataset(processed_root: Path, factor: int):
    """
    Augment the preprocessed dataset in-place.
    For each original image (no '_aug' in filename), generate `factor` copies.
    """
    classes = sorted([d for d in processed_root.iterdir() if d.is_dir()])
    print(f"\nAugmenting {len(classes)} classes (factor={factor}) …")
    total_generated = 0

    for cls_dir in tqdm(classes, desc="Augmenting"):
        # Only process original images (not previous augmentations)
        originals = [
            f for f in sorted(cls_dir.glob("*.jpg"))
            if "_aug" not in f.stem
        ] + [
            f for f in sorted(cls_dir.glob("*.png"))
            if "_aug" not in f.stem
        ]

        aug_idx = 0
        for img_path in originals:
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            augs = augment_image(img)
            random.shuffle(augs)

            for aug in augs[:factor]:
                out_path = cls_dir / f"{img_path.stem}_aug{aug_idx:05d}.jpg"
                cv2.imwrite(str(out_path), aug)
                aug_idx += 1
                total_generated += 1

    print(f"  {total_generated} augmented images generated.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Validate, preprocess and augment a texture dataset"
    )
    parser.add_argument("--dataset", required=True,
                        help="Root directory of the raw dataset")
    parser.add_argument("--output", default=None,
                        help="Preprocessed output directory "
                             "(default: <dataset>_preprocessed)")
    parser.add_argument("--augment-factor", type=int, default=5,
                        help="Augmented copies per original image (default: 5)")
    parser.add_argument("--validate-only", action="store_true",
                        help="Validate only, without preprocessing")
    parser.add_argument("--no-augment", action="store_true",
                        help="Skip the augmentation step")
    args = parser.parse_args()

    dataset_root = Path(args.dataset)
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_root}")

    output_root = (
        Path(args.output) if args.output
        else Path(str(dataset_root) + "_preprocessed")
    )

    validate_dataset(dataset_root)

    if args.validate_only:
        return

    preprocess_dataset(dataset_root, output_root)

    if not args.no_augment:
        augment_dataset(output_root, args.augment_factor)

    # Final summary
    total = sum(
        len(list((output_root / d.name).glob("*.jpg")))
        for d in dataset_root.iterdir() if d.is_dir()
    )
    print(f"\nDone. Total images in preprocessed dataset: {total}")
    print(f"Output directory: {output_root}")


if __name__ == "__main__":
    main()
