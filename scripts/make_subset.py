"""
make_subset.py
Membuat subset kecil dari dataset penuh (data/raw/) supaya training di CPU
laptop jauh lebih cepat untuk uji coba awal, sebelum scale up ke seluruh
dataset.

Cara kerja:
    - Ambil N kelas pertama (atau kelas yang kamu tentukan lewat --classes)
    - Ambil maksimal M gambar per kelas (default 150)
    - Copy ke folder data/raw_subset/

Setelah itu, jalankan training dengan menunjuk ke folder subset:
    python train_baseline.py --data_dir ../data/raw_subset

Contoh:
    # Ambil 4 kelas pertama, maksimal 100 gambar per kelas
    python make_subset.py --num_classes 4 --max_per_class 100

    # Ambil kelas tertentu saja
    python make_subset.py --classes Tomato___healthy Tomato___Early_blight
"""

import os
import shutil
import argparse
import random

SEED = 42


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", default=None,
                         help="Folder dataset penuh (default: <project_root>/data/raw)")
    parser.add_argument("--target_dir", default=None,
                         help="Folder output subset (default: <project_root>/data/raw_subset)")
    parser.add_argument("--num_classes", type=int, default=4,
                         help="Jumlah kelas yang diambil jika --classes tidak diisi (default: 4)")
    parser.add_argument("--max_per_class", type=int, default=150,
                         help="Maksimal jumlah gambar per kelas (default: 150)")
    parser.add_argument("--classes", nargs="*", default=None,
                         help="Nama kelas spesifik yang mau diambil (opsional, override --num_classes)")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    source_dir = args.source_dir or os.path.join(project_root, "data", "raw")
    target_dir = args.target_dir or os.path.join(project_root, "data", "raw_subset")

    if not os.path.isdir(source_dir):
        raise FileNotFoundError(
            f"Folder dataset tidak ditemukan: {source_dir}\n"
            f"Pastikan dataset sudah diekstrak ke data/raw/ (lihat README)."
        )

    all_classes = sorted([
        d for d in os.listdir(source_dir)
        if os.path.isdir(os.path.join(source_dir, d))
    ])

    if not all_classes:
        raise ValueError(f"Tidak ada folder kelas ditemukan di {source_dir}")

    if args.classes:
        missing = [c for c in args.classes if c not in all_classes]
        if missing:
            raise ValueError(
                f"Kelas berikut tidak ditemukan di dataset: {missing}\n"
                f"Kelas yang tersedia: {all_classes}"
            )
        selected_classes = args.classes
    else:
        selected_classes = all_classes[:args.num_classes]

    print(f"Kelas yang dipilih ({len(selected_classes)}): {selected_classes}")

    if os.path.exists(target_dir):
        print(f"Menghapus subset lama di {target_dir}...")
        shutil.rmtree(target_dir)
    os.makedirs(target_dir, exist_ok=True)

    random.seed(SEED)

    for cls in selected_classes:
        src_cls_dir = os.path.join(source_dir, cls)
        dst_cls_dir = os.path.join(target_dir, cls)
        os.makedirs(dst_cls_dir, exist_ok=True)

        images = [
            f for f in os.listdir(src_cls_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        random.shuffle(images)
        selected_images = images[:args.max_per_class]

        for img_name in selected_images:
            shutil.copy2(
                os.path.join(src_cls_dir, img_name),
                os.path.join(dst_cls_dir, img_name),
            )

        print(f"  {cls}: {len(selected_images)} gambar disalin")

    total_images = sum(
        len(os.listdir(os.path.join(target_dir, cls))) for cls in selected_classes
    )
    print(f"\nSelesai. Subset dataset ada di: {target_dir}")
    print(f"Total: {len(selected_classes)} kelas, {total_images} gambar.")
    print(
        "\nJalankan training dengan menunjuk ke folder subset ini, contoh:\n"
        "  python train_baseline.py --data_dir ../data/raw_subset\n"
        "  python train_improved.py --data_dir ../data/raw_subset"
    )


if __name__ == "__main__":
    main()
