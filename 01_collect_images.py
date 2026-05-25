#!/usr/bin/env python3
"""
01_collect_images.py — Guided image collection from the endoscope camera.

Usage:
    python 01_collect_images.py --label cotton --camera 0 --n 80
    python 01_collect_images.py --label wood   --camera 0 --n 80 --auto

Keys (interactive mode):
    SPACE : save the current frame
    A     : toggle auto-save mode
    Q     : quit
    R     : reset (keep saved files)
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np


# ── Image quality ─────────────────────────────────────────────────────────────

def laplacian_variance(gray: np.ndarray) -> float:
    """Laplacian variance — sharpness metric. Higher = sharper."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def brightness_score(gray: np.ndarray) -> float:
    """Mean pixel intensity (0–255)."""
    return float(gray.mean())


def quality_check(frame: np.ndarray, blur_thr: float = 5.0) -> tuple:
    """
    Check whether a frame is usable.
    Returns (ok: bool, message: str).
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Measure sharpness on the central third only — avoids LED ring artifacts
    h, w = gray.shape
    cy, cx = h // 2, w // 2
    r = min(h, w) // 3
    center = gray[cy - r:cy + r, cx - r:cx + r]
    blur = laplacian_variance(center)
    bright = brightness_score(gray)

    if blur < blur_thr:
        return False, f"TOO BLURRY (sharpness={blur:.1f} < {blur_thr})"
    if bright < 20:
        return False, f"TOO DARK (brightness={bright:.1f})"
    if bright > 235:
        return False, f"OVEREXPOSED (brightness={bright:.1f})"
    return True, f"OK  sharpness={blur:.1f}  brightness={bright:.1f}"


# ── HUD overlay ───────────────────────────────────────────────────────────────

def draw_overlay(
    frame: np.ndarray,
    label: str,
    saved: int,
    target: int,
    quality_ok: bool,
    quality_msg: str,
    auto: bool,
) -> np.ndarray:
    vis = frame.copy()
    h, w = vis.shape[:2]
    crop_size = min(h, w) * 2 // 3
    cx, cy = w // 2, h // 2

    # Crop indicator rectangle
    color = (0, 220, 0) if quality_ok else (0, 0, 220)
    cv2.rectangle(
        vis,
        (cx - crop_size // 2, cy - crop_size // 2),
        (cx + crop_size // 2, cy + crop_size // 2),
        color, 2,
    )

    # Top info bar
    cv2.rectangle(vis, (0, 0), (w, 62), (0, 0, 0), -1)
    cv2.putText(vis, f"Class: {label}  [{saved}/{target}]",
                (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)
    mode_str = "AUTO (fixed interval)" if auto else "MANUAL (SPACE=save)"
    cv2.putText(vis, f"Mode: {mode_str}  | Q=quit | A=toggle auto",
                (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (190, 190, 190), 1)

    # Bottom quality bar
    bar_color = (0, 200, 0) if quality_ok else (0, 0, 200)
    cv2.rectangle(vis, (0, h - 32), (w, h), (0, 0, 0), -1)
    cv2.putText(vis, quality_msg,
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bar_color, 1)

    # Progress bar
    prog_w = int((w - 4) * saved / max(target, 1))
    cv2.rectangle(vis, (2, h - 36), (2 + prog_w, h - 33), (0, 200, 100), -1)

    return vis


# ── Main collection loop ──────────────────────────────────────────────────────

def open_camera(camera_idx: int, camera_name: str | None):
    """
    Open camera. Tries vidgear first (avoids MSMF device-locking on Windows),
    then falls back to OpenCV DSHOW / name-based capture.
    Returns (stream_obj, is_vidgear: bool).
    """
    # 1. vidgear — must come first on Windows to avoid MSMF locking the device
    try:
        from vidgear.gears import CamGear
        stream = CamGear(source=camera_idx, THREADED_QUEUE_MODE=False).start()
        time.sleep(0.3)
        frame = stream.read()
        if frame is not None:
            print(f"  [INFO] Using vidgear backend (camera {camera_idx})")
            return stream, True
        stream.stop()
    except Exception:
        pass

    # 2. OpenCV by name (DSHOW)
    if camera_name:
        cap = cv2.VideoCapture(f"video={camera_name}", cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                return cap, False
            cap.release()

    # 3. OpenCV by index (DSHOW only — skip MSMF to avoid device locking)
    cap = cv2.VideoCapture(camera_idx, cv2.CAP_DSHOW)
    if cap.isOpened():
        ret, _ = cap.read()
        if ret:
            return cap, False
        cap.release()

    raise RuntimeError(
        f"Cannot open camera index={camera_idx}. "
        "Make sure the endoscope is plugged in and no other app is using it."
    )


def read_frame(stream, is_vidgear: bool):
    """Read one frame regardless of backend."""
    if is_vidgear:
        frame = stream.read()
        return (frame is not None), frame
    return stream.read()


def release_camera(stream, is_vidgear: bool):
    if is_vidgear:
        stream.stop()
    else:
        stream.release()


def collect(
    label: str,
    camera_idx: int,
    camera_name: str | None,
    n_images: int,
    auto: bool,
    auto_interval: float,
    blur_threshold: float,
    output_root: str,
):
    out_dir = Path(output_root) / label
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resume after existing images
    existing = sorted(out_dir.glob("*.jpg"))
    start_idx = len(existing)
    saved = 0

    stream, is_vidgear = open_camera(camera_idx, camera_name)

    if not is_vidgear:
        stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    src = f"'{camera_name}'" if camera_name else f"index {camera_idx}"
    print(f"\nCollecting '{label}' from camera {src} → {out_dir}")
    print(f"Starting at index {start_idx}, need {n_images} more images.")
    print("Keys: SPACE=save  A=auto  Q=quit  R=reset\n")

    auto_mode = auto
    last_save_time = 0.0

    while saved < n_images:
        ret, frame = read_frame(stream, is_vidgear)
        if not ret or frame is None:
            print("Camera read failed.")
            break

        quality_ok, quality_msg = quality_check(frame, blur_threshold)
        vis = draw_overlay(frame, label, saved, n_images, quality_ok, quality_msg, auto_mode)
        cv2.imshow("TextureVision — Collection", vis)

        key = cv2.waitKey(1) & 0xFF

        # Auto-save
        if auto_mode and quality_ok:
            now = time.time()
            if now - last_save_time >= auto_interval:
                img_path = out_dir / f"{start_idx + saved:04d}.jpg"
                cv2.imwrite(str(img_path), frame)
                saved += 1
                last_save_time = now
                print(f"  [AUTO] {img_path.name}  ({saved}/{n_images})")

        if key == ord(" ") and not auto_mode:
            if quality_ok:
                img_path = out_dir / f"{start_idx + saved:04d}.jpg"
                cv2.imwrite(str(img_path), frame)
                saved += 1
                print(f"  [OK]  {img_path.name}  ({saved}/{n_images}) — {quality_msg}")
            else:
                print(f"  [REJECTED] {quality_msg}")
        elif key == ord("a"):
            auto_mode = not auto_mode
            print(f"  Auto mode: {'ON' if auto_mode else 'OFF'}")
        elif key == ord("r"):
            print("  Reset (saved files are kept).")
        elif key == ord("q"):
            print("  Quit requested.")
            break

    release_camera(stream, is_vidgear)
    cv2.destroyAllWindows()
    print(f"\nDone. {saved} images saved to {out_dir}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Guided image collection for TextureVision"
    )
    parser.add_argument("--label", required=True,
                        help="Texture class name (e.g. 'cotton', 'wood')")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera device index (default: 0)")
    parser.add_argument("--camera-name", default=None,
                        help="Open camera by DirectShow name instead of index "
                             "(e.g. 'USB Camera'). Takes priority over --camera.")
    parser.add_argument("--n", type=int, default=80,
                        help="Number of images to collect (default: 80)")
    parser.add_argument("--auto", action="store_true",
                        help="Start directly in auto-save mode")
    parser.add_argument("--interval", type=float, default=1.5,
                        help="Seconds between auto-saves (default: 1.5)")
    parser.add_argument("--blur-threshold", type=float, default=5.0,
                        help="Minimum Laplacian variance to accept a frame (default: 5)")
    parser.add_argument("--output", default="my_dataset",
                        help="Root output directory (default: my_dataset)")
    args = parser.parse_args()

    collect(
        args.label,
        args.camera,
        args.camera_name,
        args.n,
        args.auto,
        args.interval,
        args.blur_threshold,
        args.output,
    )


if __name__ == "__main__":
    main()
