"""
Flask server for TextureVision CNN texture recognition interface.
Compatible with index_cnn.html.

Endpoints:
    GET  /              → serves index_cnn.html
    GET  /model-info    → model metadata
    POST /analyze       → predict from base64 images
    GET  /camera        → single live frame prediction from endoscope
"""

import base64
import pickle
import time
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

# ── Optional CNN imports ───────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    from torchvision import models, transforms
    from PIL import Image
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

# ── Config ────────────────────────────────────────────────────────────────────
CNN_MODEL_PATH  = Path("cnn_output/model_cnn_best.pth")
LBP_MODEL_PATH  = Path("model_lbp.pkl")
CAMERA_IDX      = 0
CROP_SIZE       = 175
IMG_SIZE        = 128

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

app = Flask(__name__, static_folder=".")
CORS(app)


# ── Preprocessing ─────────────────────────────────────────────────────────────

def center_crop(gray, size):
    h, w = gray.shape
    if h < size or w < size:
        pad_h = max(0, size - h)
        pad_w = max(0, size - w)
        gray = cv2.copyMakeBorder(
            gray, pad_h // 2, pad_h - pad_h // 2,
            pad_w // 2, pad_w - pad_w // 2, cv2.BORDER_REFLECT
        )
        h, w = gray.shape
    cy, cx = h // 2, w // 2
    half = size // 2
    return gray[cy - half:cy + half, cx - half:cx + half]


def preprocess(frame: np.ndarray) -> np.ndarray:
    """BGR frame → 175×175 CLAHE + unsharp-masked grayscale."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    gray = center_crop(gray, CROP_SIZE)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.addWeighted(gray, 2.5, blurred, -1.5, 0)
    return gray


# ── CNN predictor ─────────────────────────────────────────────────────────────

class CNNPredictor:
    def __init__(self, model_path: Path):
        self.classes = []
        self.model = None
        self.transform = None
        self.device = None

        if not HAS_TORCH:
            print("[CNN] PyTorch not available.")
            return
        if not model_path.exists():
            print(f"[CNN] Model not found: {model_path}")
            return

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(str(model_path), map_location=self.device)
        self.classes = ckpt["classes"]
        num_classes = len(self.classes)

        m = models.mobilenet_v3_small(weights=None)
        in_features = m.classifier[-1].in_features
        m.classifier[-1] = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_features, num_classes))
        m.load_state_dict(ckpt["model_state"])
        self.model = m.to(self.device).eval()

        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ])
        print(f"[CNN] Loaded — {num_classes} classes on {self.device}")

    @property
    def loaded(self):
        return self.model is not None

    def predict(self, frame: np.ndarray):
        """Returns list of (class_name, confidence) sorted by confidence desc."""
        gray = preprocess(frame)
        pil  = Image.fromarray(gray)
        tensor = self.transform(pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.model(tensor), dim=1).cpu().numpy()[0]
        order = probs.argsort()[::-1]
        return [(self.classes[i], float(probs[i])) for i in order]


# ── LBP fallback predictor ────────────────────────────────────────────────────

class LBPPredictor:
    def __init__(self, model_path: Path):
        self.classes = []
        self.model = None
        if not model_path.exists():
            return
        with open(model_path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.classes = data["classes"]
        print(f"[LBP] Loaded — {len(self.classes)} classes")

    @property
    def loaded(self):
        return self.model is not None

    def predict(self, frame: np.ndarray):
        from skimage.feature import local_binary_pattern
        gray = preprocess(frame)
        feats = []
        for p, r in [(8, 1), (16, 2)]:
            lbp = local_binary_pattern(gray, p, r, method="uniform")
            hist, _ = np.histogram(lbp, bins=p + 2, range=(0, p + 2))
            feats.append(hist.astype(float) / (hist.sum() + 1e-9))
        feat = np.concatenate(feats).reshape(1, -1)
        probs = self.model.predict_proba(feat)[0]
        order = probs.argsort()[::-1]
        return [(self.classes[i], float(probs[i])) for i in order]


# ── Load models ───────────────────────────────────────────────────────────────

cnn = CNNPredictor(CNN_MODEL_PATH)
lbp = LBPPredictor(LBP_MODEL_PATH)
predictor = cnn if cnn.loaded else lbp
mode = "CNN (MobileNetV3-Small)" if cnn.loaded else ("LBP+SVM" if lbp.loaded else "none")
print(f"[server] Active predictor: {mode}")


# ── Camera helper ─────────────────────────────────────────────────────────────

def _open_camera():
    try:
        from vidgear.gears import CamGear
        stream = CamGear(source=CAMERA_IDX, THREADED_QUEUE_MODE=False).start()
        time.sleep(0.3)
        return stream, True
    except Exception:
        pass
    cap = cv2.VideoCapture(CAMERA_IDX, cv2.CAP_DSHOW)
    return (cap, False) if cap.isOpened() else (None, False)


def _read_frame(stream, is_vidgear):
    if is_vidgear:
        f = stream.read()
        return f is not None, f
    return stream.read()


def _release(stream, is_vidgear):
    stream.stop() if is_vidgear else stream.release()


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(".", "index_cnn.html")


@app.route("/model-info")
def model_info():
    if not predictor.loaded:
        return jsonify({"loaded": False, "classes": [],
                        "message": "No model found. Run 04_train_cnn.py first."})
    return jsonify({
        "loaded": True,
        "classes": predictor.classes,
        "n_classes": len(predictor.classes),
        "algorithm": mode,
        "img_size": IMG_SIZE,
        "crop_size": CROP_SIZE,
    })


@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        images = request.json.get("images", [])
        if not images:
            return jsonify({"error": "No images provided"}), 400

        results = []
        for img_data in images:
            raw = img_data["src"].split(",")[1]
            buf = np.frombuffer(base64.b64decode(raw), dtype=np.uint8)
            frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if frame is None:
                results.append({"name": img_data["name"], "error": "Decode failed"})
                continue

            if not predictor.loaded:
                results.append({"name": img_data["name"],
                                "textures": [{"name": "NO_MODEL", "confidence": 0}]})
                continue

            top = predictor.predict(frame)[:3]
            results.append({
                "name": img_data["name"],
                "src":  img_data["src"],
                "textures": [{"name": cls.upper(), "confidence": int(conf * 100)}
                             for cls, conf in top],
            })

        return jsonify({"results": results})

    except Exception as e:
        print(f"[server] /analyze error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/camera")
def camera_predict():
    """Grab one frame from the endoscope and return a prediction."""
    if not predictor.loaded:
        return jsonify({"error": "No model loaded"}), 503

    stream, is_vg = _open_camera()
    if stream is None:
        return jsonify({"error": "Cannot open camera"}), 503

    ret, frame = _read_frame(stream, is_vg)
    _release(stream, is_vg)

    if not ret or frame is None:
        return jsonify({"error": "Camera read failed"}), 503

    top = predictor.predict(frame)[:3]

    # Encode frame as base64 JPEG for display
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
    src = "data:image/jpeg;base64," + base64.b64encode(buf).decode()

    return jsonify({
        "src": src,
        "textures": [{"name": cls.upper(), "confidence": int(conf * 100)}
                     for cls, conf in top],
    })


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("TextureVision CNN Server")
    print("=" * 60)
    print(f"Model    : {mode}")
    if predictor.loaded:
        print(f"Classes  : {', '.join(predictor.classes)}")
    print(f"Crop     : {CROP_SIZE} px  |  CNN input: {IMG_SIZE}×{IMG_SIZE}")
    print(f"Camera   : index {CAMERA_IDX}")
    print("=" * 60)
    print("Open http://localhost:5000 in your browser\n")
    app.run(debug=False, port=5000)
