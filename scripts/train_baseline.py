"""
train_baseline.py
Pendekatan 1 (Baseline): Transfer Learning dengan MobileNetV2.

MobileNetV2 dipilih sebagai baseline karena arsitekturnya ringan (depthwise
separable convolutions) sehingga cepat dilatih dan cocok sebagai pembanding
terhadap model yang lebih besar (EfficientNetB0) pada train_improved.py.
Ringan juga berarti cocok untuk kasus ini: klasifikasi 3 kelas sampah
(organik / anorganik / B3) dengan dataset relatif kecil (ribuan gambar).

Strategi:
    1. Muat MobileNetV2 (ImageNet weights), freeze seluruh base.
    2. Tambahkan classification head (GAP -> Dropout -> Dense softmax).
    3. Latih head saja beberapa epoch (feature extraction).

Jalankan:
    python scripts/train_baseline.py
"""

import os
import json
import argparse
import tensorflow as tf
from tensorflow.keras import layers, models

from utils import build_datasets, get_data_augmentation, IMG_SIZE, MODELS_DIR, RAW_DATA_DIR

EPOCHS = 15
LEARNING_RATE = 1e-3
MODEL_OUT_PATH = os.path.join(MODELS_DIR, "baseline_mobilenetv2.keras")
CHECKPOINT_PATH = os.path.join(MODELS_DIR, "baseline_mobilenetv2_checkpoint.keras")
STATE_PATH = os.path.join(MODELS_DIR, "baseline_training_state.json")
HISTORY_OUT_PATH = os.path.join(MODELS_DIR, "history_baseline.json")


class EpochStateSaver(tf.keras.callbacks.Callback):
    """Tracks the last completed epoch so training can resume from where it
    left off if interrupted (Ctrl+C, crash, power loss, etc.)."""

    def __init__(self, state_path):
        super().__init__()
        self.state_path = state_path

    def on_epoch_end(self, epoch, logs=None):
        with open(self.state_path, "w") as f:
            json.dump({"last_completed_epoch": epoch}, f)


def build_model(num_classes, img_size=IMG_SIZE):
    data_augmentation = get_data_augmentation()

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=img_size + (3,),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False  # feature extraction only for baseline

    inputs = layers.Input(shape=img_size + (3,))
    x = data_augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="baseline_mobilenetv2")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=None,
                         help="Path ke folder dataset (default: data/raw). "
                              "Gunakan ../data/raw_subset untuk uji cepat dengan subset kelas.")
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                         help=f"Jumlah epoch (default: {EPOCHS})")
    parser.add_argument("--img_size", type=int, default=IMG_SIZE[0],
                         help="Ukuran gambar (persegi), makin kecil makin cepat "
                              "tapi akurasi bisa sedikit turun. Default: 224. "
                              "Coba 128 atau 96 untuk CPU yang lambat.")
    parser.add_argument("--batch_size", type=int, default=32,
                         help="Batch size. Batch lebih besar = lebih sedikit step/epoch "
                              "(butuh RAM lebih banyak). Default: 32.")
    parser.add_argument("--steps_per_epoch", type=int, default=None,
                         help="Batasi jumlah step per epoch untuk smoke test super cepat "
                              "(mis. 50). Tidak melihat seluruh data tiap epoch. "
                              "Kosongkan untuk training normal.")
    parser.add_argument("--fresh", action="store_true",
                         help="Mulai training dari awal, abaikan checkpoint yang ada "
                              "(default: otomatis lanjut dari checkpoint terakhir jika ada).")
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

    initial_epoch = 0
    if not args.fresh and os.path.exists(CHECKPOINT_PATH) and os.path.exists(STATE_PATH):
        print(f"[Resume] Checkpoint ditemukan di {CHECKPOINT_PATH}, melanjutkan training...")
        model = tf.keras.models.load_model(CHECKPOINT_PATH)
        with open(STATE_PATH, "r") as f:
            state = json.load(f)
        initial_epoch = state["last_completed_epoch"] + 1
        print(f"[Resume] Melanjutkan dari epoch {initial_epoch + 1}/{args.epochs}")
    else:
        if args.fresh and os.path.exists(CHECKPOINT_PATH):
            print("[Fresh] --fresh diaktifkan, mengabaikan checkpoint lama.")
        model = build_model(num_classes, img_size=img_size)
        model.summary()

    if initial_epoch >= args.epochs:
        print(
            f"[!] Checkpoint sudah mencapai epoch {initial_epoch}, "
            f">= target --epochs {args.epochs}. Naikkan --epochs untuk melanjutkan, "
            f"atau pakai --fresh untuk mulai ulang."
        )
        return

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=4, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2
        ),
        tf.keras.callbacks.ModelCheckpoint(
            CHECKPOINT_PATH, save_freq="epoch"
        ),
        EpochStateSaver(STATE_PATH),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        initial_epoch=initial_epoch,
        steps_per_epoch=args.steps_per_epoch,
        callbacks=callbacks,
    )

    os.makedirs(MODELS_DIR, exist_ok=True)
    model.save(MODEL_OUT_PATH)

    with open(HISTORY_OUT_PATH, "w") as f:
        json.dump(history.history, f, indent=2)

    print(f"Model baseline disimpan di {MODEL_OUT_PATH}")
    print(f"(Checkpoint sementara di {CHECKPOINT_PATH} masih ada, aman dihapus manual jika perlu.)")


if __name__ == "__main__":
    main()
