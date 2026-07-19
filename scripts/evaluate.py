"""
evaluate.py
Membandingkan performa Pendekatan 1 (MobileNetV2 baseline) vs
Pendekatan 2 (EfficientNetB0 improved) pada validation set yang sama.

Menghasilkan:
    - models/comparison_report.json  (accuracy, precision, recall, f1 per model)
    - models/confusion_matrix_baseline.png
    - models/confusion_matrix_improved.png

Jalankan setelah kedua model selesai dilatih:
    python scripts/evaluate.py
"""

import os
import json
import argparse
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from utils import build_datasets, MODELS_DIR, TRAIN_CONFIG_PATH, IMG_SIZE, RAW_DATA_DIR

BASELINE_PATH = os.path.join(MODELS_DIR, "baseline_mobilenetv2.keras")
IMPROVED_PATH = os.path.join(MODELS_DIR, "improved_efficientnetb0.keras")
REPORT_OUT_PATH = os.path.join(MODELS_DIR, "comparison_report.json")


def evaluate_model(model, val_ds, class_names, model_name):
    y_true = []
    y_pred = []

    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, xticks_rotation=90, colorbar=False)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    out_path = os.path.join(MODELS_DIR, f"confusion_matrix_{model_name}.png")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix disimpan di {out_path}")

    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=None,
                         help="Path ke folder dataset yang dipakai saat training (default: data/raw)")
    parser.add_argument("--img_size", type=int, default=None,
                         help="Ukuran gambar (persegi). Kalau tidak diisi, otomatis dibaca dari "
                              "models/train_config.json (ukuran yang dipakai saat training terakhir).")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    if args.img_size:
        img_size = (args.img_size, args.img_size)
    elif os.path.exists(TRAIN_CONFIG_PATH):
        with open(TRAIN_CONFIG_PATH, "r") as f:
            config = json.load(f)
        img_size = tuple(config["img_size"])
        print(f"[Info] Memakai img_size={img_size} dari models/train_config.json "
              f"(hasil training terakhir). Pakai --img_size untuk override.")
    else:
        img_size = IMG_SIZE

    _, val_ds, class_names = build_datasets(
        data_dir=args.data_dir if args.data_dir else RAW_DATA_DIR,
        img_size=img_size,
        batch_size=args.batch_size,
    )

    results = {}

    if os.path.exists(BASELINE_PATH):
        print("Mengevaluasi baseline (MobileNetV2)...")
        baseline_model = tf.keras.models.load_model(BASELINE_PATH)
        results["baseline_mobilenetv2"] = evaluate_model(
            baseline_model, val_ds, class_names, "baseline"
        )
    else:
        print(f"[!] {BASELINE_PATH} tidak ditemukan, lewati evaluasi baseline.")

    if os.path.exists(IMPROVED_PATH):
        print("Mengevaluasi improved (EfficientNetB0)...")
        improved_model = tf.keras.models.load_model(IMPROVED_PATH)
        results["improved_efficientnetb0"] = evaluate_model(
            improved_model, val_ds, class_names, "improved"
        )
    else:
        print(f"[!] {IMPROVED_PATH} tidak ditemukan, lewati evaluasi improved.")

    with open(REPORT_OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nRingkasan perbandingan disimpan di {REPORT_OUT_PATH}")
    for name, report in results.items():
        acc = report.get("accuracy", None)
        macro_f1 = report.get("macro avg", {}).get("f1-score", None)
        print(f"  {name}: accuracy={acc:.4f}, macro_f1={macro_f1:.4f}")


if __name__ == "__main__":
    main()