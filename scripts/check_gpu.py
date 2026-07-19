"""
check_gpu.py
Cek apakah TensorFlow mendeteksi GPU di komputer kamu. Kalau hasilnya kosong
([]), training akan jalan di CPU saja - itu sebabnya training terasa lambat
(~1-2 detik/step, ~25-30 menit/epoch untuk MobileNetV2 pada dataset penuh).

Jalankan:
    python scripts/check_gpu.py
"""

import tensorflow as tf

print("TensorFlow version:", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
print("GPU terdeteksi:", gpus)

if not gpus:
    print(
        "\n[!] Tidak ada GPU terdeteksi. Training akan berjalan di CPU.\n"
        "    Ini normal untuk laptop tanpa GPU NVIDIA + CUDA/cuDNN terpasang.\n"
        "    Opsi: (1) latih dengan subset kelas lebih sedikit (lihat make_subset.py),\n"
        "          (2) pindah ke Google Colab / Kaggle Notebook (GPU gratis)."
    )
else:
    print("\n[OK] GPU siap dipakai untuk training.")
