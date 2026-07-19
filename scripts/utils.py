"""
utils.py
Shared configuration and data-loading helpers used by both the baseline
(MobileNetV2) and improved (EfficientNetB0) training scripts.

Dataset expected layout (ImageFolder format, one folder per class):

    data/
        raw/
            organik/
                img001.jpg
                img002.jpg
                ...
            anorganik/
                ...
            b3/
                ...

Lihat README.md untuk sumber dataset yang direkomendasikan (Kaggle: dataset
sampah organik/anorganik + tambahan kelas B3).
"""

import os
import json
import tensorflow as tf

# ---------------------------------------------------------------------------
# Global config
# ---------------------------------------------------------------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

# Resolve paths relative to the project root (parent of scripts/), not the
# current working directory, so training works whether you run it from
# the project root or from inside scripts/.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
CLASS_INDEX_PATH = os.path.join(MODELS_DIR, "class_indices.json")
TRAIN_CONFIG_PATH = os.path.join(MODELS_DIR, "train_config.json")

AUTOTUNE = tf.data.AUTOTUNE


def build_datasets(data_dir=RAW_DATA_DIR, img_size=IMG_SIZE, batch_size=BATCH_SIZE,
                    val_split=0.2, seed=SEED, cache=True):
    """
    Builds train/validation tf.data.Dataset objects directly from a folder
    structure using Keras' image_dataset_from_directory, then applies
    caching + prefetching. Also saves the class name -> index mapping to
    disk so the Flask app can load it later for inference.

    cache=True keeps decoded images in memory after the first epoch, which
    speeds up epoch 2+ significantly on CPU (no repeated JPEG decode/resize).
    Turn it off only if the dataset is too large to fit in RAM.
    """
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=val_split,
        subset="training",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="categorical",
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=val_split,
        subset="validation",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="categorical",
    )

    class_names = train_ds.class_names

    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(CLASS_INDEX_PATH, "w") as f:
        json.dump({i: name for i, name in enumerate(class_names)}, f, indent=2)

    if cache:
        train_ds = train_ds.cache()
        val_ds = val_ds.cache()

    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, class_names


def get_data_augmentation():
    """Light augmentation shared by both models (kept modest since leaves
    are somewhat orientation/color sensitive for disease symptoms)."""
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.08),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomContrast(0.1),
    ], name="data_augmentation")


def load_class_names(path=CLASS_INDEX_PATH):
    with open(path, "r") as f:
        mapping = json.load(f)
    # keys are saved as strings in JSON; sort by int key to preserve order
    return [mapping[str(i)] for i in range(len(mapping))]
