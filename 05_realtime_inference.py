#!/usr/bin/env python3
"""
06_realtime_inference.py — Real-time texture recognition from the endoscope camera.

Features:
    - Live camera feed with HUD (Heads-Up Display)
    - Contact detection (hysteretic threshold on LED contrast metric)
    - Temporal smoothing (majority vote over N frames)
    - Confidence bar + top-3 predictions
    - Compatible with LBP (.pkl) and CNN (.pth) models

Usage:
    python 06_realtime_inference.py --model model_lbp.pkl --type lbp --camera 0
    python 06_realtime_inference.py --model cnn_output/model_cnn_best.pth --type cnn
    python 06_realtime_inference.py --model model_lbp.pkl --type lbp --no-contact

Keys during execution:
    Q   : quit
    R   : reset smoothing history
    S   : save a screenshot
    +/- : adjust the contact threshold
"""

import argparse
import collections
import pickle
import time
from pathlib import Path

import cv2
import numpy as np


CROP_SIZE = 175
IMG_SIZE  = 128   # CNN input size

# Contact detection thresholds (metric = contrast variation in the central patch)
_CONTACT_PATCH = 60    # central patch size (pixels)
_CONTACT_DOWN  = 15.0  # threshold to enter contact (rising)
_CONTACT_UP    = 9.0   # threshold to leave contact (hysteresis, lower)
_DEBOUNCE      = 3     # consecutive frames required before state change


# ── LBP utilities (self-contained so this script has no external imports) ──────

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


def _clahe(gray):
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)


def _lbp_hist(gray, P, R):
    try:
        from skimage.feature import local_binary_pattern
        lbp = local_binary_pattern(gray, P, R, method="uniform")
        hist, _ = np.histogram(lbp, bins=P+2, range=(0, P+2))
    except ImportError:
        hist = np.zeros(P+2, dtype=int)
    return hist.astype(float) / (hist.sum() + 1e-9)


def _extract_lbp(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame.copy()
    gray = _center_crop(gray, CROP_SIZE)
    gray = _clahe(gray)
    feats = [
        _lbp_hist(gray, 8, 1), _lbp_hist(gray, 16, 2),
        _lbp_hist(cv2.resize(gray, (CROP_SIZE//2, CROP_SIZE//2)), 8, 1),
        _lbp_hist(cv2.resize(gray, (CROP_SIZE//2, CROP_SIZE//2)), 16, 2),
        _lbp_hist(cv2.resize(gray, (CROP_SIZE//4, CROP_SIZE//4)), 8, 1),
    ]
    return np.concatenate(feats)


# ── Contact detection ─────────────────────────────────────────────────────────

def _contrast_metric(gray) -> float:
    """
    Measures contrast variation in the central patch.
    High when the endoscope is in contact with a surface
    (LED light reveals texture microrelief); low in air.
    """
    h, w = gray.shape
    cy, cx = h//2, w//2
    half = _CONTACT_PATCH // 2
    patch = gray[cy-half:cy+half, cx-half:cx+half].astype(int)
    return float(np.mean(np.abs(patch - np.roll(patch, 1, axis=1))))


class ContactDetector:
    """Contact detection with hysteresis and debounce."""

    def __init__(self, down_thr=_CONTACT_DOWN, up_thr=_CONTACT_UP, debounce=_DEBOUNCE):
        self.down_thr   = down_thr
        self.up_thr     = up_thr
        self.debounce   = debounce
        self.in_contact = False
        self._cnt       = 0
        self.just_touched = False  # True for exactly 1 frame on touch-down

    def update(self, metric: float):
        self.just_touched = False
        if not self.in_contact and metric > self.down_thr:
            self._cnt += 1
            if self._cnt >= self.debounce:
                self.in_contact   = True
                self.just_touched = True
                self._cnt = 0
        elif self.in_contact and metric < self.up_thr:
            self._cnt += 1
            if self._cnt >= self.debounce:
                self.in_contact = False
                self._cnt = 0
        else:
            self._cnt = 0


# ── Predictors ────────────────────────────────────────────────────────────────

class LBPPredictor:
    def __init__(self, model_path: str):
        with open(model_path, "rb") as f:
            data = pickle.load(f)
        self.model   = data["model"]
        self.classes = data["classes"]

    def predict(self, frame) -> tuple:
        feats = _extract_lbp(frame).reshape(1, -1)
        idx   = self.model.predict(feats)[0]
        proba = self.model.predict_proba(feats)[0]
        label = self.classes[idx]
        conf  = float(proba.max())
        top3  = [(self.classes[i], float(proba[i]))
                 for i in proba.argsort()[::-1][:3]]
        return label, conf, top3


class CNNPredictor:
    def __init__(self, model_path: str, classes_file: str = None):
        import torch
        import torch.nn as nn
        from torchvision import models, transforms

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Classes
        if classes_file and Path(classes_file).exists():
            self.classes = [l.strip() for l in
                            Path(classes_file).read_text().splitlines() if l.strip()]
        else:
            raise FileNotFoundError(
                "classes.txt not found. Use --classes-file or place it in "
                "the same directory as the .pth file."
            )

        num_classes = len(self.classes)
        ckpt = torch.load(model_path, map_location=self.device)
        m = models.mobilenet_v3_small(weights=None)
        in_feat = m.classifier[-1].in_features
        m.classifier[-1] = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_feat, num_classes))
        m.load_state_dict(ckpt["model_state"])
        self.model = m.to(self.device).eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self._torch = torch

    def predict(self, frame) -> tuple:
        tensor = self.transform(frame).unsqueeze(0).to(self.device)
        with self._torch.no_grad():
            proba = self._torch.softmax(self.model(tensor), dim=1).cpu().numpy()[0]
        idx   = proba.argmax()
        conf  = float(proba.max())
        label = self.classes[idx]
        top3  = [(self.classes[i], float(proba[i]))
                 for i in proba.argsort()[::-1][:3]]
        return label, conf, top3


# ── HUD ───────────────────────────────────────────────────────────────────────

def draw_hud(frame, label, conf, top3, in_contact, metric, fps, contact_thr) -> np.ndarray:
    vis = frame.copy()
    h, w = vis.shape[:2]

    # ── Top bar ───────────────────────────────────────────────────────────────
    cv2.rectangle(vis, (0, 0), (w, 38), (15, 15, 15), -1)
    dot_color = (0, 210, 0) if in_contact else (80, 80, 80)
    cv2.circle(vis, (22, 19), 9, dot_color, -1)
    cv2.putText(
        vis,
        f"Contact: {'YES' if in_contact else 'NO'}  "
        f"metric={metric:.1f}  threshold={contact_thr:.1f}  FPS={fps:.1f}",
        (38, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (210, 210, 210), 1,
    )

    # ── Crop indicator rectangle ──────────────────────────────────────────────
    cs = min(h - 38, w) * 2 // 3
    cx, cy = w // 2, (h + 38) // 2
    color = (0, 210, 255) if in_contact else (80, 80, 80)
    cv2.rectangle(vis, (cx - cs//2, cy - cs//2), (cx + cs//2, cy + cs//2), color, 2)

    # ── Bottom panel ──────────────────────────────────────────────────────────
    panel_h = 140
    cv2.rectangle(vis, (0, h - panel_h), (w, h), (12, 12, 12), -1)

    if in_contact and label:
        # Smoothed predicted class
        cv2.putText(vis, label.upper(),
                    (12, h - panel_h + 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 255, 150), 2)

        # Confidence bar
        bar_w = int((w - 24) * conf)
        bar_color = ((0, 200, 0) if conf > 0.70
                     else (0, 165, 255) if conf > 0.45
                     else (0, 0, 200))
        cv2.rectangle(vis, (12, h - panel_h + 42), (12 + bar_w, h - panel_h + 60),
                      bar_color, -1)
        cv2.putText(vis, f"{conf*100:.0f}%",
                    (16, h - panel_h + 57),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Top-3
        for i, (cls_name, prob) in enumerate(top3):
            y_pos = h - panel_h + 80 + i * 19
            blen  = int((w - 24) * prob)
            bar_c = (60, 200, 60) if i == 0 else (80, 80, 80)
            cv2.rectangle(vis, (12, y_pos - 11), (12 + blen, y_pos + 3), bar_c, -1)
            cv2.putText(vis, f"{i+1}. {cls_name}: {prob*100:.0f}%",
                        (16, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (215, 215, 215), 1)
    else:
        cv2.putText(vis, "Place glove on a surface …",
                    (12, h - panel_h + 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, (140, 140, 140), 1)

    # Key hints
    cv2.putText(vis, "Q=quit  R=reset  S=screenshot  +/-=threshold",
                (12, h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (90, 90, 90), 1)

    return vis


# ── Main loop ─────────────────────────────────────────────────────────────────

def open_camera(camera_idx: int, camera_name: str | None):
    """Returns (stream, is_vidgear: bool). vidgear is tried first to avoid MSMF locking."""
    # 1. vidgear first — avoids MSMF device-locking on Windows
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

    # 3. OpenCV by index (DSHOW only)
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
    if is_vidgear:
        frame = stream.read()
        return (frame is not None), frame
    return stream.read()


def release_camera(stream, is_vidgear: bool):
    stream.stop() if is_vidgear else stream.release()


def run(
    model_path: str,
    model_type: str,
    camera_idx: int,
    camera_name: str | None,
    classes_file: str,
    no_contact: bool,
    history_len: int,
):
    print(f"Loading {model_type.upper()} model …")
    predictor = (LBPPredictor(model_path)
                 if model_type == "lbp"
                 else CNNPredictor(model_path, classes_file))
    print(f"Classes ({len(predictor.classes)}): {predictor.classes[:8]} …")

    stream, is_vidgear = open_camera(camera_idx, camera_name)

    if not is_vidgear:
        stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    detector       = ContactDetector()
    history        = collections.deque(maxlen=history_len)
    label          = ""
    conf           = 0.0
    top3           = []
    fps            = 0.0
    last_time      = time.time()
    contact_thr    = _CONTACT_DOWN
    screenshot_idx = 0

    print("\nRunning … (Q=quit, R=reset, S=screenshot, +/-=contact threshold)")

    while True:
        ret, frame = read_frame(stream, is_vidgear)
        if not ret or frame is None:
            print("Camera read failed.")
            break

        gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        metric = _contrast_metric(gray)

        if no_contact:
            in_contact = True
        else:
            detector.down_thr = contact_thr
            detector.update(metric)
            in_contact = detector.in_contact

        if in_contact:
            new_label, new_conf, new_top3 = predictor.predict(frame)
            history.append(new_label)
            # Majority vote to smooth frame-to-frame predictions
            smoothed = collections.Counter(history).most_common(1)[0][0]
            label, conf, top3 = smoothed, new_conf, new_top3
        else:
            if not in_contact:
                label = ""

        # Exponential moving average FPS
        now  = time.time()
        fps  = 0.9 * fps + 0.1 / max(now - last_time, 1e-6)
        last_time = now

        vis = draw_hud(frame, label, conf, top3, in_contact, metric, fps, contact_thr)
        cv2.imshow("TextureVision — Real-time", vis)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            history.clear()
            label = ""
            print("  History reset.")
        elif key == ord("s"):
            fname = f"screenshot_{screenshot_idx:04d}.jpg"
            cv2.imwrite(fname, vis)
            screenshot_idx += 1
            print(f"  Screenshot → {fname}")
        elif key == ord("+") or key == ord("="):
            contact_thr = min(contact_thr + 1.0, 100.0)
            print(f"  Contact threshold: {contact_thr:.1f}")
        elif key == ord("-"):
            contact_thr = max(contact_thr - 1.0, 1.0)
            print(f"  Contact threshold: {contact_thr:.1f}")

    release_camera(stream, is_vidgear)
    cv2.destroyAllWindows()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Real-time texture recognition"
    )
    parser.add_argument("--model", required=True,
                        help="Model file (.pkl or .pth)")
    parser.add_argument("--type", choices=["lbp", "cnn"], required=True,
                        help="Model type")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera device index (default: 0)")
    parser.add_argument("--camera-name", default=None,
                        help="Open camera by DirectShow name (e.g. 'USB Camera'). "
                             "Takes priority over --camera.")
    parser.add_argument("--classes-file", default=None,
                        help="Path to classes.txt (CNN only)")
    parser.add_argument("--no-contact", action="store_true",
                        help="Disable contact detection (classify continuously)")
    parser.add_argument("--history", type=int, default=9,
                        help="Temporal smoothing window size (default: 9)")
    args = parser.parse_args()

    # Auto-detect classes.txt for CNN
    if args.type == "cnn" and args.classes_file is None:
        auto = Path(args.model).parent / "classes.txt"
        if auto.exists():
            args.classes_file = str(auto)

    run(
        args.model, args.type, args.camera, args.camera_name,
        args.classes_file, args.no_contact, args.history,
    )


if __name__ == "__main__":
    main()
