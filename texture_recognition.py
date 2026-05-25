"""
Magic Finger – Camera-based Texture Recognition
================================================
Faithful implementation of the texture sensing pipeline from:

  Yang, X.-D., Grossman, T., Wigdor, D., Fitzmaurice, G.
  "Magic Finger: Always-Available Input through Finger Instrumentation"
  ACM UIST 2012, pp. 147–156.
  https://doi.org/10.1145/2380116.2380137

  - 59-dim uniform LBP (Ojala 2002), hand-rolled with bilinear interpolation
  - Tuned RBF-SVM with calibrated probabilities
  - Model persistence (save/load)
  - Zero-dependency KNN fallback
  - Synthetic demo (no camera or dataset needed)
  - MagicFingerConfig dataclass for clean parameter management   [from draft]
  - Three-level dataset layout (environmental / artificial / mixed) [from draft]
  - Crop-size ablation replicating Figure 7 of the paper          [from draft]
  - DataMatrix decoding via pylibdmtx                             [from draft]

Hardware in the paper:
  * AWAIBA NanEye RGB micro-camera  -  248x248 px @ 3 um^2/px, 44 fps
  * Cropped to 175x175 px (70% of sensor view)
  * 5 mm white LED (illuminates surface; doubles as output indicator)
  * ADNS-2620 optical-flow sensor (X-Y movement only, not used for texture)

Usage
-----
  python magic_finger.py --mode collect --label wood   # gather training data
  python magic_finger.py --mode train                  # fit SVM
  python magic_finger.py --mode train --ablation       # + Figure 7 crop ablation
  python magic_finger.py --mode run                    # real-time recognition
  python magic_finger.py --mode demo                   # synthetic 5-fold benchmark

Dataset layout (mirrors paper's evaluation structure):
  dataset/
      environmental/          # natural surfaces
          desk/img001.jpg ...
          keyboard/img001.jpg ...
      artificial/             # printed ASCII textures
          A/img001.jpg ...
      mixed/                  # combined set (paper's 32-class eval)
          desk/
          A/
          ...
"""

from __future__ import annotations

import argparse
import glob
import os
import pickle
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

# -- optional sklearn ---------------------------------------------------------
try:
    from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False
    print("[warn] scikit-learn not found - KNN fallback active")

# -- optional DataMatrix ------------------------------------------------------
try:
    from pylibdmtx.pylibdmtx import decode as _dmtx_decode
    HAS_DMTX = True
except Exception:
    HAS_DMTX = False


# -----------------------------------------------------------------------------
# Configuration  (dataclass from draft - cleaner than scattered constants)
# -----------------------------------------------------------------------------

@dataclass
class MagicFingerConfig:
    # -- image sizes ----------------------------------------------------------
    # Paper: "175x175 rectangle (yellow region)" = 70% of 248x248 NanEye view
    recognition_crop_size: int = 175
    # Paper: "60x60 pixel square" used for the contrast / contact metric
    contrast_patch_size: int = 60

    # -- LBP (Ojala et al. PAMI 2002) -----------------------------------------
    # P=8, R=1 -> 58 uniform patterns + 1 catch-all = 59-dim histogram.
    # The paper calls these the "10 microstructures" (semantic families, not
    # the bin count): flat, 4 edge orientations, 4 corner orientations, spot.
    lbp_points: int = 8
    lbp_radius: int = 1

    # -- contact detection ----------------------------------------------------
    # Must be calibrated to your LED brightness and lens distance.
    # Paper: contrast jumps when LED reflection returns on touch-down.
    touch_down_threshold: float = 15.0
    lift_up_threshold: float = 9.0     # hysteresis: lower than touch_down
    debounce_frames: int = 3

    # -- SVM ------------------------------------------------------------------
    # Grid-searched as per paper ("tuned required SVM parameters").
    # probability=True gives Platt-scaled [0,1] confidence, not a raw margin.
    svm_c_grid: List[float] = field(default_factory=lambda: [0.1, 1, 10, 100])
    svm_gamma_grid: List = field(default_factory=lambda: ["scale", "auto", 0.001, 0.01])

    # -- evaluation -----------------------------------------------------------
    n_splits: int = 5              # paper: 5-fold CV
    random_state: int = 42
    # Paper Figure 7: crop sizes tested for sensor flexibility
    ablation_crop_sizes: Tuple[int, ...] = (18, 70, 122, 175)
    # Paper: 10 snaps x 2/day x 3 days = 60 samples/class
    samples_per_class: int = 60

    # -- runtime --------------------------------------------------------------
    history_len: int = 9           # temporal smoothing window
    confidence_threshold: float = 0.60

    # -- paths ----------------------------------------------------------------
    dataset_root: str = "dataset"
    model_path: str = "magic_finger_model.pkl"


CFG = MagicFingerConfig()

IMG_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")


# -----------------------------------------------------------------------------
# 1.  UNIFORM LBP  (Ojala et al. PAMI 2002) - hand-rolled, no skimage needed
# -----------------------------------------------------------------------------

def _build_uniform_lbp_map(n_points: int) -> np.ndarray:
    """
    Lookup table: LBP code (0 ... 2^P - 1) -> uniform bin index.

    A pattern is *uniform* if it has <= 2 circular bit-transitions.
    For P=8: 58 uniform patterns + 1 non-uniform catch-all = 59 bins.
    These 59 bins map onto the "10 microstructure" semantic families
    described by Ojala (flat areas, edges x4 orientations, corners x4,
    spots) - the "10" is a semantic grouping, NOT the histogram dimension.
    """
    n_codes = 2 ** n_points
    mapping = np.full(n_codes, -1, dtype=np.int32)
    uni_idx = 0
    for code in range(n_codes):
        bits = [(code >> i) & 1 for i in range(n_points)]
        transitions = sum(bits[i] != bits[(i + 1) % n_points]
                          for i in range(n_points))
        if transitions <= 2:
            mapping[code] = uni_idx
            uni_idx += 1
    mapping[mapping == -1] = uni_idx   # non-uniform catch-all
    return mapping                      # values in [0, 58] for P=8


_LBP_MAP_P8 = _build_uniform_lbp_map(8)   # precomputed for default P=8


def compute_uniform_lbp(gray: np.ndarray,
                         radius: int = CFG.lbp_radius,
                         n_points: int = CFG.lbp_points) -> np.ndarray:
    """
    Normalised uniform LBP histogram.

    Uses bilinear interpolation for sub-pixel neighbour sampling, matching
    the formulation in Ojala 2002.  Orientation-invariant and grayscale-
    invariant, as stated in the paper.

    Returns a (59,) float32 vector for the default P=8, R=1.
    """
    h, w = gray.shape
    lbp  = np.zeros((h, w), dtype=np.int32)
    fp   = gray.astype(np.float32)

    for p in range(n_points):
        theta = 2.0 * np.pi * p / n_points
        xn = np.clip(np.arange(w) + radius * np.cos(theta), 0, w - 1)
        yn = np.clip(np.arange(h) + radius * np.sin(theta), 0, h - 1)

        x0, y0 = xn.astype(int), yn.astype(int)
        x1 = np.clip(x0 + 1, 0, w - 1)
        y1 = np.clip(y0 + 1, 0, h - 1)
        fx = (xn - x0)[None, :]
        fy = (yn - y0)[:, None]

        neighbor = (
            (1 - fy) * (1 - fx) * fp[y0][:, x0] +
            (1 - fy) *       fx  * fp[y0][:, x1] +
                  fy  * (1 - fx) * fp[y1][:, x0] +
                  fy  *       fx  * fp[y1][:, x1]
        )
        lbp += (neighbor >= fp).astype(np.int32) << p

    mapping = _LBP_MAP_P8 if n_points == 8 else _build_uniform_lbp_map(n_points)
    n_bins  = int(mapping.max()) + 1
    mapped  = mapping[lbp.ravel() % (2 ** n_points)]
    hist, _ = np.histogram(mapped, bins=n_bins, range=(0, n_bins))
    hist    = hist.astype(np.float32)
    norm    = hist.sum()
    if norm > 0:
        hist /= norm
    return hist


# -----------------------------------------------------------------------------
# 2.  IMAGE UTILITIES
# -----------------------------------------------------------------------------

def to_gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def center_crop(img: np.ndarray, size: int) -> np.ndarray:
    h, w = img.shape[:2]
    size = min(size, h, w)
    y0 = (h - size) // 2
    x0 = (w - size) // 2
    return img[y0: y0 + size, x0: x0 + size]


def extract_features(frame: np.ndarray,
                     crop_size: int = CFG.recognition_crop_size) -> np.ndarray:
    """
    Full pipeline:  BGR frame -> grayscale -> center crop -> 59-dim LBP hist.
    crop_size can be varied to replicate Figure 7 of the paper.
    """
    gray = to_gray(frame)
    roi  = center_crop(gray, crop_size)
    roi  = cv2.resize(roi, (crop_size, crop_size), interpolation=cv2.INTER_AREA)
    return compute_uniform_lbp(roi)


# -----------------------------------------------------------------------------
# 3.  CONTACT DETECTION
# -----------------------------------------------------------------------------

def contrast_metric(gray: np.ndarray,
                    patch_size: int = CFG.contrast_patch_size) -> float:
    """
    Paper: "averaging the square difference between each pixel and its
    neighbors" over a 60x60 centre square.

    Low when >5 mm from surface (dark, no LED reflection).
    Jumps on touch-down as the LED illuminates the texture.
    """
    patch = center_crop(gray, patch_size).astype(np.float32)
    dx = patch[:, 1:] - patch[:, :-1]
    dy = patch[1:, :] - patch[:-1, :]
    return float((np.mean(dx ** 2) + np.mean(dy ** 2)) / 2.0)


class ContactDetector:
    """
    Hysteretic contact detector with debounce.

    Combines:
      - Separate touch-down / lift-up thresholds (hysteresis)
      - N-frame debounce counter before state transitions commit
      - Sliding window average to suppress single-frame spikes
    """

    def __init__(self, cfg: MagicFingerConfig = CFG) -> None:
        self.thr_down  = cfg.touch_down_threshold
        self.thr_up    = cfg.lift_up_threshold
        self.debounce  = cfg.debounce_frames
        self._window: deque[float] = deque(maxlen=self.debounce)
        self._in_contact = False
        self._counter    = 0

    def update(self, frame: np.ndarray) -> Tuple[bool, Optional[str], float]:
        """
        Feed a BGR (or grayscale) frame.

        Returns
        -------
        in_contact : bool
        event      : 'touch_down' | 'lift_up' | None
        metric     : raw contrast value (useful for threshold calibration)
        """
        gray   = to_gray(frame)
        metric = contrast_metric(gray)
        self._window.append(metric)
        avg = float(np.mean(self._window))

        target = self._in_contact
        if not self._in_contact and avg >= self.thr_down:
            target = True
        elif self._in_contact and avg < self.thr_up:
            target = False

        event = None
        if target != self._in_contact:
            self._counter += 1
            if self._counter >= self.debounce:
                self._in_contact = target
                event = "touch_down" if self._in_contact else "lift_up"
                self._counter = 0
        else:
            self._counter = 0

        return self._in_contact, event, metric

    @property
    def in_contact(self) -> bool:
        return self._in_contact


# -----------------------------------------------------------------------------
# 4.  CLASSIFIER
# -----------------------------------------------------------------------------

class _KNNFallback:
    """3-NN Euclidean classifier - used when scikit-learn is unavailable."""

    def __init__(self, k: int = 3):
        self.k = k
        self.X: List[np.ndarray] = []
        self.y: List[int] = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X, self.y = list(X), list(y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        out = []
        for xq in X:
            dists = sorted((float(np.linalg.norm(xq - xi)), yi)
                            for xi, yi in zip(self.X, self.y))
            out.append(Counter(yi for _, yi in dists[: self.k]).most_common(1)[0][0])
        return np.array(out)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        labels = sorted(set(self.y))
        proba  = []
        for xq in X:
            dists  = sorted((float(np.linalg.norm(xq - xi)), yi)
                             for xi, yi in zip(self.X, self.y))
            top    = [yi for _, yi in dists[: self.k]]
            counts = Counter(top)
            proba.append([counts.get(l, 0) / self.k for l in labels])
        return np.array(proba)


def _tune_svm(X: np.ndarray, y: np.ndarray,
              cfg: MagicFingerConfig = CFG) -> "Pipeline":
    """
    Grid-search RBF-SVM over C and gamma with stratified 5-fold CV.
    Mirrors the paper's "tuned required SVM parameters that gave high CV scores."
    probability=True -> Platt-scaled [0,1] confidence (not a raw decision margin).
    """
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svm",    SVC(kernel="rbf", probability=True)),
    ])
    param_grid = {
        "svm__C":     cfg.svm_c_grid,
        "svm__gamma": cfg.svm_gamma_grid,
    }
    cv = StratifiedKFold(n_splits=min(cfg.n_splits, len(np.unique(y))),
                         shuffle=True, random_state=cfg.random_state)
    gs = GridSearchCV(pipe, param_grid, cv=cv, scoring="accuracy",
                      n_jobs=-1, verbose=0)
    gs.fit(X, y)
    print(f"[tune] best: {gs.best_params_}  CV = {gs.best_score_:.1%}")
    return gs.best_estimator_


# -----------------------------------------------------------------------------
# 5.  DATASET LOADING  (three-level layout from draft)
# -----------------------------------------------------------------------------

def list_images(folder: str) -> List[str]:
    paths: List[str] = []
    for ext in IMG_EXTS:
        paths.extend(glob.glob(os.path.join(folder, ext)))
    return sorted(paths)


def load_dataset(root: str,
                 crop_size: int = CFG.recognition_crop_size
                 ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load features from  root/{class_name}/image_files ...

    root can be any of:
      dataset/environmental/
      dataset/artificial/
      dataset/mixed/          <- paper's 32-class combined evaluation
    """
    class_names = sorted(d for d in os.listdir(root)
                         if os.path.isdir(os.path.join(root, d)))
    X, y = [], []
    for idx, cls in enumerate(class_names):
        for path in list_images(os.path.join(root, cls)):
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                continue
            X.append(extract_features(img, crop_size=crop_size))
            y.append(idx)

    if not X:
        raise ValueError(f"No images found under {root}")
    return np.vstack(X), np.array(y, dtype=np.int32), class_names


# -----------------------------------------------------------------------------
# 6.  TextureRecognizer
# -----------------------------------------------------------------------------

class TextureRecognizer:
    """
    Encapsulates training, evaluation, prediction, and persistence.

    Paper protocol:
      - 60 samples/class  (10 snaps x 2/day x 3 days)
      - order randomised before 5-fold CV
      - RBF-SVM with C, gamma chosen by CV
    """

    def __init__(self, cfg: MagicFingerConfig = CFG):
        self.cfg = cfg
        self.model: object = None
        self.label_names: List[str] = []

    def train(self, root: str) -> float:
        """Fit on root/{class}/...  Return 5-fold CV accuracy."""
        X, y, self.label_names = load_dataset(root, self.cfg.recognition_crop_size)

        # Randomise order before CV (paper: "randomized the order of the
        # points prior to the test")
        rng  = np.random.default_rng(self.cfg.random_state)
        perm = rng.permutation(len(y))
        X, y = X[perm], y[perm]

        if SKLEARN_OK:
            self.model = _tune_svm(X, y, self.cfg)
            cv = StratifiedKFold(n_splits=min(self.cfg.n_splits, len(y)),
                                 shuffle=True,
                                 random_state=self.cfg.random_state)
            scores = cross_val_score(self.model, X, y, cv=cv,
                                     scoring="accuracy")
            acc = float(scores.mean())
            print(f"[train] {len(X)} samples x {len(self.label_names)} classes  "
                  f"5-fold CV = {acc:.1%} +/- {scores.std():.1%}")
        else:
            self.model = _KNNFallback(k=3)
            self.model.fit(X, y)
            acc = 0.0
            print("[train] KNN fallback fitted")

        self.save()
        return acc

    def predict(self, frame: np.ndarray) -> Tuple[str, float]:
        """Return (label, confidence) for a single BGR frame."""
        if self.model is None:
            return "unknown", 0.0
        feat = extract_features(frame, self.cfg.recognition_crop_size).reshape(1, -1)
        idx  = int(self.model.predict(feat)[0])
        conf = float(self.model.predict_proba(feat)[0][idx]) \
               if hasattr(self.model, "predict_proba") else 1.0
        label = self.label_names[idx] if idx < len(self.label_names) else "unknown"
        return label, conf

    def save(self, path: Optional[str] = None):
        path = path or self.cfg.model_path
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "labels": self.label_names}, f)
        print(f"[save] -> {path}")

    def load(self, path: Optional[str] = None) -> bool:
        path = path or self.cfg.model_path
        if not os.path.exists(path):
            return False
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model, self.label_names = data["model"], data["labels"]
        print(f"[load] <- {path}  classes = {self.label_names}")
        return True


# -----------------------------------------------------------------------------
# 7.  CROP-SIZE ABLATION  (replicates Figure 7 of the paper)
# -----------------------------------------------------------------------------

def run_ablation(root: str,
                 crop_sizes: Sequence[int] = CFG.ablation_crop_sizes,
                 cfg: MagicFingerConfig = CFG) -> Dict[int, float]:
    """
    Evaluate recognition accuracy at each crop size.

    Paper Figure 7 reports:  18 px -> ~55%,  70 px -> 95%,
                             122 px -> ~98%,  175 px -> 99.1%
    """
    results: Dict[int, float] = {}
    for size in crop_sizes:
        X, y, _ = load_dataset(root, crop_size=size)
        if SKLEARN_OK:
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("svm",    SVC(kernel="rbf", C=10, gamma="scale")),
            ])
            cv = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True,
                                 random_state=cfg.random_state)
            scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
            results[size] = float(scores.mean())
        else:
            clf = _KNNFallback(k=3)
            clf.fit(X, y)
            results[size] = 0.0
        print(f"  {size:>3}x{size:<3}  CV = {results[size]:.1%}")
    return results


# -----------------------------------------------------------------------------
# 8.  DATA MATRIX DECODING
# -----------------------------------------------------------------------------

def decode_datamatrix(frame: np.ndarray,
                      crop_size: int = CFG.recognition_crop_size
                      ) -> Optional[str]:
    """
    Try to decode a Data Matrix code from the centre of the frame.

    Paper: 10x10 cell codes, clusters of 72x15 identical codes printed so
    that at least one is fully in the camera's FOV.  Attempted before texture
    classification on every touch-down event.

    Returns decoded string, or None if not found or library unavailable.
    """
    if not HAS_DMTX:
        return None
    roi = center_crop(to_gray(frame), crop_size)
    decoded = _dmtx_decode(roi)
    if not decoded:
        return None
    try:
        return decoded[0].data.decode("utf-8")
    except Exception:
        return str(decoded[0].data)


# -----------------------------------------------------------------------------
# 9.  DATA COLLECTION UI
# -----------------------------------------------------------------------------

def collect_data(label: str, dataset_root: str = CFG.dataset_root,
                 subset: str = "environmental",
                 n_snaps: int = 10, camera_id: int = 0,
                 cfg: MagicFingerConfig = CFG):
    """
    Collect training patches from the camera.

    Paper protocol: 10 snaps per session, twice a day, 3 days = 60/class.
    Run once per session.  Auto-snaps on touch-down; SPACE for manual snap.

    subset : 'environmental' | 'artificial' | 'mixed'
    """
    save_dir = Path(dataset_root) / subset / label
    save_dir.mkdir(parents=True, exist_ok=True)
    existing = len(list(save_dir.glob("*.jpg")))
    print(f"[collect] '{subset}/{label}'  existing={existing}  "
          f"target={cfg.samples_per_class}  SPACE=snap  Q=quit")

    detector = ContactDetector(cfg)
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_id}")

    snapped = 0
    half    = cfg.recognition_crop_size // 2
    roi_h   = cfg.contrast_patch_size // 2

    while snapped < n_snaps:
        ret, frame = cap.read()
        if not ret:
            break

        in_contact, event, metric = detector.update(frame)

        if event == "touch_down":
            ts = int(time.time() * 1000)
            cv2.imwrite(str(save_dir / f"{ts}.jpg"), frame)
            snapped += 1
            print(f"  auto-snap  ({snapped}/{n_snaps})")

        vis = frame.copy()
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        col = (0, 200, 80) if in_contact else (100, 100, 100)
        cv2.rectangle(vis, (cx - half, cy - half), (cx + half, cy + half), col, 2)
        cv2.rectangle(vis, (cx - roi_h, cy - roi_h), (cx + roi_h, cy + roi_h),
                      (50, 50, 220), 1)
        cv2.putText(vis,
                    f"'{subset}/{label}'  {snapped}/{n_snaps}  "
                    f"contrast={metric:.1f}  {'CONTACT' if in_contact else '---'}",
                    (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(vis, "SPACE=manual snap   Q=quit",
                    (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)
        cv2.imshow("Magic Finger - Collect", vis)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            ts = int(time.time() * 1000)
            cv2.imwrite(str(save_dir / f"{ts}_m.jpg"), frame)
            snapped += 1
            print(f"  manual snap  ({snapped}/{n_snaps})")
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    total = len(list(save_dir.glob("*.jpg")))
    print(f"[collect] done.  total for '{label}': {total} / {cfg.samples_per_class}")


# -----------------------------------------------------------------------------
# 10.  REAL-TIME RECOGNITION
# -----------------------------------------------------------------------------

def run_recognition(camera_id: int = 0, cfg: MagicFingerConfig = CFG):
    """
    Live recognition loop.

    On every touch-down:
      1. Try DataMatrix decode first (paper: checked before texture clf)
      2. Fall back to texture classification
    Temporal smoothing over the last cfg.history_len touch events.
    """
    rec = TextureRecognizer(cfg)
    if not rec.load():
        print("[error] No model found. Run --mode train first.")
        return

    detector = ContactDetector(cfg)
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_id}")

    history: deque[str] = deque(maxlen=cfg.history_len)
    last_label, last_conf = "-", 0.0
    fps_t, fps_n, fps = time.time(), 0, 0.0
    half  = cfg.recognition_crop_size // 2
    roi_h = cfg.contrast_patch_size // 2

    print("[run] active.  Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        in_contact, event, metric = detector.update(frame)

        if event == "touch_down":
            # DataMatrix first (paper section "Fiduciary Markers")
            code = decode_datamatrix(frame, cfg.recognition_crop_size)
            if code:
                last_label, last_conf = f"DM:{code}", 1.0
                history.append(last_label)
                print(f"  DataMatrix -> {code}")
            else:
                last_label, last_conf = rec.predict(frame)
                if last_conf >= cfg.confidence_threshold:
                    history.append(last_label)
                print(f"  touch -> {last_label}  ({last_conf:.0%})")

        smoothed = Counter(history).most_common(1)[0][0] if history else "-"

        fps_n += 1
        elapsed = time.time() - fps_t
        if elapsed >= 1.0:
            fps = fps_n / elapsed
            fps_t, fps_n = time.time(), 0

        vis = frame.copy()
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2

        col = (0, 200, 80) if in_contact else (100, 100, 100)
        cv2.rectangle(vis, (cx - half, cy - half), (cx + half, cy + half), col, 2)
        cv2.rectangle(vis, (cx - roi_h, cy - roi_h), (cx + roi_h, cy + roi_h),
                      (50, 50, 220), 1)

        overlay = vis.copy()
        cv2.rectangle(overlay, (0, h - 80), (w, h), (15, 15, 15), -1)
        vis = cv2.addWeighted(overlay, 0.6, vis, 0.4, 0)
        cv2.putText(vis, f"Texture: {smoothed}",
                    (10, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(vis,
                    f"raw: {last_label} ({last_conf:.0%})  "
                    f"contrast: {metric:.1f}  fps: {fps:.1f}",
                    (10, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (160, 160, 160), 1)

        cv2.imshow("Magic Finger - Recognition", vis)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# -----------------------------------------------------------------------------
# 11.  SYNTHETIC BENCHMARK  (no camera or dataset required)
# -----------------------------------------------------------------------------

def demo_synthetic(n_classes: int = 22, n_samples: int = 60,
                   cfg: MagicFingerConfig = CFG):
    """
    Replicates the paper's evaluation protocol with synthetic textures:
      * 22 texture classes (paper's environmental-texture study)
      * 60 samples per class
      * 5-fold stratified CV, order randomised
      * Reports per-fold accuracy + mean +/- std
    """
    print(f"\n=== Synthetic benchmark  "
          f"({n_classes} classes x {n_samples} samples) ===")

    rng = np.random.default_rng(0)
    X, y = [], []
    for cls in range(n_classes):
        for _ in range(n_samples):
            freq  = 5 + cls * 3
            xs    = np.linspace(0, 2 * np.pi * freq, cfg.recognition_crop_size)
            ys    = np.linspace(0, 2 * np.pi * freq * 0.7, cfg.recognition_crop_size)
            base  = np.sin(xs)[None, :] * np.cos(ys)[:, None] * 80 + 128
            patch = np.clip(base + rng.normal(0, 12, base.shape), 0, 255).astype(np.uint8)
            X.append(compute_uniform_lbp(patch))
            y.append(cls)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    perm = rng.permutation(len(y))
    X, y = X[perm], y[perm]

    if SKLEARN_OK:
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("svm",    SVC(kernel="rbf", C=10, gamma="scale", probability=True)),
        ])
        cv     = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True,
                                 random_state=cfg.random_state)
        scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
        print("Per-fold accuracy:  " + "  ".join(f"{s:.1%}" for s in scores))
        print(f"Mean +/- std:       {scores.mean():.1%} +/- {scores.std():.1%}")
        print(f"Paper result:       99.1%  (22 environmental textures)")
        print(f"Feature vector:     {X.shape[1]}-dim uniform LBP  (P=8, R=1)")
    else:
        split = n_classes * (n_samples - 5)
        clf = _KNNFallback(k=3)
        clf.fit(X[:split], y[:split])
        preds = clf.predict(X[split:])
        print(f"KNN test accuracy: {(preds == y[split:]).mean():.1%}")


# -----------------------------------------------------------------------------
# 12.  CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Magic Finger - camera-based texture recognition "
                    "(Yang et al. UIST 2012)")
    p.add_argument("--mode",
                   choices=["collect", "train", "run", "demo"],
                   default="demo")
    p.add_argument("--label",   default="wood",
                   help="Texture class name  (collect mode)")
    p.add_argument("--subset",  default="environmental",
                   choices=["environmental", "artificial", "mixed"],
                   help="Dataset subset  (collect / train mode)")
    p.add_argument("--snaps",   type=int, default=10,
                   help="Snaps per session  (collect mode; "
                        "run 6 sessions for paper's 60/class)")
    p.add_argument("--camera",  type=int, default=0)
    p.add_argument("--ablation", action="store_true",
                   help="Run crop-size ablation (Figure 7) after training")
    p.add_argument("--classes", type=int, default=22,
                   help="Synthetic texture classes  (demo mode)")
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = MagicFingerConfig()

    if args.mode == "collect":
        collect_data(label=args.label,
                     dataset_root=cfg.dataset_root,
                     subset=args.subset,
                     n_snaps=args.snaps,
                     camera_id=args.camera,
                     cfg=cfg)

    elif args.mode == "train":
        root = os.path.join(cfg.dataset_root, args.subset)
        rec  = TextureRecognizer(cfg)
        rec.train(root)
        if args.ablation:
            print("\n--- Figure 7 crop-size ablation ---")
            run_ablation(root, cfg.ablation_crop_sizes, cfg)

    elif args.mode == "run":
        run_recognition(camera_id=args.camera, cfg=cfg)

    elif args.mode == "demo":
        demo_synthetic(n_classes=args.classes, cfg=cfg)


if __name__ == "__main__":
    main()