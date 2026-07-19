"""
train_improved.py
Pendekatan 2 (Improved): Transfer Learning dengan EfficientNetB0 + fine-tuning
dua tahap.

Berbeda dengan baseline (MobileNetV2, hanya feature extraction), di sini
digunakan:
    1. Arsitektur yang lebih dalam & lebih akurat (EfficientNetB0).
    2. Fine-tuning dua tahap:
         Stage 1 - freeze base, latih head saja (feature extraction).
         Stage 2 - unfreeze sebagian besar layer teratas base model,
                    lanjutkan training dengan learning rate kecil.

Hasil akhir dibandingkan dengan baseline pada scripts/evaluate.py.

Jalankan:
    python scripts/train_improved.py
"""

import os
import json
import argparse
import tensorflow as tf
from tensorflow.keras import layers, models

from utils import build_datasets, get_data_augmentation, IMG_SIZE, MODELS_DIR, RAW_DATA_DIR

STAGE1_EPOCHS = 10
STAGE2_EPOCHS = 10
STAGE1_LR = 1e-3
STAGE2_LR = 1e-5
FINE_TUNE_AT = 100  # unfreeze layers from this index onward in stage 2

MODEL_OUT_PATH = os.path.join(MODELS_DIR, "improved_efficientnetb0.keras")
CHECKPOINT_PATH = os.path.join(MODELS_DIR, "improved_efficientnetb0_checkpoint.keras")
STATE_PATH = os.path.join(MODELS_DIR, "improved_training_state.json")
HISTORY_OUT_PATH = os.path.join(MODELS_DIR, "history_improved.json")


class EpochStateSaver(tf.keras.callbacks.Callback):
    """Tracks the last completed epoch (and which stage) so training can
    resume from where it left off if interrupted."""

    def __init__(self, state_path, stage):
        super().__init__()
        self.state_path = state_path
        self.stage = stage

    def on_epoch_end(self, epoch, logs=None):
        with open(self.state_path, "w") as f:
            json.dump({"last_completed_epoch": epoch, "stage": self.stage}, f)


def build_model(num_classes, img_size=IMG_SIZE):
    data_augmentation = get_data_augmentation()

    base_model = tf.keras.applications.EfficientNetB0(
        input_shape=img_size + (3,),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False  # stage 1: frozen

    inputs = layers.Input(shape=img_size + (3,))
    x = data_augmentation(inputs)
    # EfficientNet preprocessing expects raw [0-255] pixel values; no manual
    # rescale needed since preprocess_input for efficientnet is identity-like
    # scaling handled internally by the Rescaling layer inside the model.
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="improved_efficientnetb0")
    return model, base_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=None,
                         help="Path ke folder dataset (default: data/raw). "
                              "Gunakan ../data/raw_subset untuk uji cepat dengan subset kelas.")
    parser.add_argument("--stage1_epochs", type=int, default=STAGE1_EPOCHS)
    parser.add_argument("--stage2_epochs", type=int, default=STAGE2_EPOCHS)
    parser.add_argument("--img_size", type=int, default=IMG_SIZE[0],
                         help="Ukuran gambar (persegi). Default: 224. "
                              "Coba 128 atau 96 untuk CPU yang lambat.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--steps_per_epoch", type=int, default=None,
                         help="Batasi step per epoch untuk smoke test cepat.")
    args = parser.parse_args()

    img_size = (args.img_size, args.img_size)

    train_ds, val_ds, class_names = build_datasets(
        data_dir=args.data_dir if args.data_dir else RAW_DATA_DIR,
        img_size=img_size,
        batch_size=args.batch_size,
    )
    num_classes = len(class_names)
    print(f"Ditemukan {num_classes} kelas: {class_names}")
    print(f"Konfigurasi: img_size={img_size}, batch_size={args.batch_size}, "
          f"steps_per_epoch={args.steps_per_epoch or 'semua data'}")

    model, base_model = build_model(num_classes, img_size=img_size)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=4, restore_best_weights=True
        ),
    ]

    # ---------------- Stage 1: feature extraction ----------------
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=STAGE1_LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    print("== Stage 1: training classification head (base frozen) ==")
    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.stage1_epochs,
        steps_per_epoch=args.steps_per_epoch,
        callbacks=callbacks,
    )

    # ---------------- Stage 2: fine-tuning ----------------
    base_model.trainable = True
    for layer in base_model.layers[:FINE_TUNE_AT]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=STAGE2_LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    print("== Stage 2: fine-tuning top layers of EfficientNetB0 ==")
    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.stage1_epochs + args.stage2_epochs,
        initial_epoch=history1.epoch[-1] + 1,
        steps_per_epoch=args.steps_per_epoch,
        callbacks=callbacks,
    )

    os.makedirs(MODELS_DIR, exist_ok=True)
    model.save(MODEL_OUT_PATH)

    combined_history = {}
    for key in history1.history:
        combined_history[key] = history1.history[key] + history2.history.get(key, [])
    with open(HISTORY_OUT_PATH, "w") as f:
        json.dump(combined_history, f, indent=2)

    print(f"Model improved disimpan di {MODEL_OUT_PATH}")


if __name__ == "__main__":
    main()
