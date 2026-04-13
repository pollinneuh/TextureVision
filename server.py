"""
Flask server for Magic Finger texture recognition web interface
"""
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import cv2
import numpy as np
import base64
import os
from pathlib import Path

# Import your texture_recognition code
from texture_recognition import (
    TextureRecognizer,
    MagicFingerConfig,
    extract_features
)

app = Flask(__name__, static_folder='.')
CORS(app)

# Initialize the recognizer
CFG = MagicFingerConfig()
recognizer = TextureRecognizer(CFG)

# Load pre-trained model
MODEL_PATH = "TV_model.pkl"
if os.path.exists(MODEL_PATH):
    recognizer.load(MODEL_PATH)
    print(f"[server] Model loaded: {recognizer.label_names}")
else:
    print("[server] WARNING: No model found! Train first with: python texture_recognition.py --mode train")

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/analyze', methods=['POST'])
def analyze_texture():
    """
    Receive base64-encoded images and return LBP texture predictions
    """
    try:
        data = request.json
        images = data.get('images', [])
        
        if not images:
            return jsonify({'error': 'No images provided'}), 400
        
        results = []
        
        for img_data in images:
            # Decode base64 image
            img_b64 = img_data['src'].split(',')[1]  # Remove data:image/...;base64,
            img_bytes = base64.b64decode(img_b64)
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if frame is None:
                results.append({
                    'name': img_data['name'],
                    'error': 'Failed to decode image'
                })
                continue
            
            # Get prediction from your LBP algorithm
            if recognizer.model is None:
                # Fallback if no model
                results.append({
                    'name': img_data['name'],
                    'textures': [{
                        'name': 'NO_MODEL',
                        'confidence': 0
                    }]
                })
            else:
                label, confidence = recognizer.predict(frame)
                
                # Get top predictions if using sklearn
                if hasattr(recognizer.model, 'predict_proba'):
                    feat = extract_features(frame, CFG.recognition_crop_size).reshape(1, -1)
                    proba = recognizer.model.predict_proba(feat)[0]
                    
                    # Get top 3 predictions
                    top_indices = np.argsort(proba)[::-1][:3]
                    textures = []
                    for idx in top_indices:
                        if idx < len(recognizer.label_names):
                            textures.append({
                                'name': recognizer.label_names[idx].upper(),
                                'confidence': int(proba[idx] * 100)
                            })
                else:
                    textures = [{
                        'name': label.upper(),
                        'confidence': int(confidence * 100)
                    }]
                
                results.append({
                    'name': img_data['name'],
                    'textures': textures,
                    'src': img_data['src']  # Send back for display
                })
        
        return jsonify({'results': results})
    
    except Exception as e:
        print(f"[server] Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/model-info', methods=['GET'])
def model_info():
    """Return information about the loaded model"""
    if recognizer.model is None:
        return jsonify({
            'loaded': False,
            'classes': [],
            'message': 'No model loaded. Train with: python texture_recognition.py --mode train'
        })
    
    return jsonify({
        'loaded': True,
        'classes': recognizer.label_names,
        'n_classes': len(recognizer.label_names),
        'algorithm': 'LBP (Local Binary Patterns)',
        'feature_dim': 59
    })

if __name__ == '__main__':
    print("\n" + "="*60)
    print("Texture Recognition Server")
    print("="*60)
    print(f"Model: {'✓ Loaded' if recognizer.model else '✗ Not found'}")
    if recognizer.model:
        print(f"Classes: {', '.join(recognizer.label_names)}")
    print(f"Algorithm: LBP (59-dim uniform, P=8, R=1)")
    print("="*60 + "\n")
    
    app.run(debug=True, port=5000)