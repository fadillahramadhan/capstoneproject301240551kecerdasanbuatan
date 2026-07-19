"""
inference.py
Memuat model terlatih (default: model improved / EfficientNetB0, fallback ke
baseline jika improved belum ada) dan menyediakan fungsi prediksi untuk
digunakan oleh app.py.
"""

import os
import json
import numpy as np
import tensorflow as tf
from PIL import Image

DEFAULT_IMG_SIZE = (224, 224)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

IMPROVED_MODEL_PATH = os.path.join(MODELS_DIR, "improved_efficientnetb0.keras")
BASELINE_MODEL_PATH = os.path.join(MODELS_DIR, "baseline_mobilenetv2.keras")
CLASS_INDEX_PATH = os.path.join(MODELS_DIR, "class_indices.json")
TRAIN_CONFIG_PATH = os.path.join(MODELS_DIR, "train_config.json")

_model = None
_model_name = None
_class_names = None
_img_size = None


def _load_class_names():
    with open(CLASS_INDEX_PATH, "r") as f:
        mapping = json.load(f)
    return [mapping[str(i)] for i in range(len(mapping))]


def _load_img_size():
    """Reads the img_size used during training (saved by utils.build_datasets),
    so inference always matches the model's actual input shape - even if
    training used a custom --img_size (e.g. 128 for faster CPU training)."""
    if os.path.exists(TRAIN_CONFIG_PATH):
        with open(TRAIN_CONFIG_PATH, "r") as f:
            config = json.load(f)
        return tuple(config["img_size"])
    return DEFAULT_IMG_SIZE


def get_model():
    """Lazily loads the model once and caches it (singleton pattern)."""
    global _model, _model_name, _class_names, _img_size

    if _model is not None:
        return _model, _model_name, _class_names

    if os.path.exists(IMPROVED_MODEL_PATH):
        _model = tf.keras.models.load_model(IMPROVED_MODEL_PATH)
        _model_name = "EfficientNetB0 (improved)"
    elif os.path.exists(BASELINE_MODEL_PATH):
        _model = tf.keras.models.load_model(BASELINE_MODEL_PATH)
        _model_name = "MobileNetV2 (baseline)"
    else:
        raise FileNotFoundError(
            "Belum ada model terlatih di folder models/. "
            "Jalankan scripts/train_baseline.py atau scripts/train_improved.py terlebih dahulu."
        )

    _class_names = _load_class_names()
    
    # Try to detect image size from model input shape first
    try:
        shape = _model.input_shape
        if isinstance(shape, list):
            shape = shape[0]
        if len(shape) == 4 and shape[1] is not None and shape[2] is not None:
            _img_size = (shape[1], shape[2])
            print(f"[Inference] Detected image size {_img_size} from model input shape.")
        else:
            _img_size = _load_img_size()
    except Exception as e:
        print(f"[Inference] Failed to detect shape from model: {e}. Falling back to config.")
        _img_size = _load_img_size()

    return _model, _model_name, _class_names


def preprocess_image(image_file):
    """Converts an uploaded image file into a batch-ready numpy array,
    resized to match whatever img_size the active model was trained with."""
    img = Image.open(image_file).convert("RGB").resize(_img_size or DEFAULT_IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    return np.expand_dims(arr, axis=0)


def predict(image_file, top_k=3):
    """Runs prediction and returns top-k (label, probability) pairs plus the
    active model name."""
    model, model_name, class_names = get_model()
    batch = preprocess_image(image_file)
    preds = model.predict(batch, verbose=0)[0]

    top_indices = preds.argsort()[-top_k:][::-1]
    results = [
        {"label": class_names[i], "probability": float(preds[i])}
        for i in top_indices
    ]
    return results, model_name
