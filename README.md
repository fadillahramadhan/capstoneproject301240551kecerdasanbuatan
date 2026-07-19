# Klasifikasi Sampah — Organik / Anorganik / B3

Capstone project mata kuliah Kecerdasan Buatan (Computer Vision) — klasifikasi
citra untuk memilah jenis sampah ke dalam 3 kategori: **organik**,
**anorganik**, dan **B3** (Bahan Berbahaya dan Beracun — baterai, elektronik,
lampu neon, obat-obatan, dsb).

## Kenapa Tema Ini

Dataset untuk tema ini jauh lebih kecil dari dataset citra medis/tanaman
(ribuan, bukan puluhan ribu gambar per kelas), jadi training jauh lebih cepat
walau di CPU laptop biasa — cocok untuk timeline capstone 2–4 minggu.

## Dua Pendekatan yang Dibandingkan

Sesuai ketentuan capstone (wajib membandingkan minimal dua pendekatan):

| | Pendekatan 1 — Baseline | Pendekatan 2 — Improved |
|---|---|---|
| Arsitektur | MobileNetV2 | EfficientNetB0 |
| Strategi training | Feature extraction (base freeze total) | 2 tahap: feature extraction → fine-tuning sebagian layer atas |
| Tujuan | Baseline cepat & ringan | Akurasi lebih tinggi dengan biaya komputasi lebih besar |
| Script | `scripts/train_baseline.py` | `scripts/train_improved.py` |

Hasil kedua model dibandingkan otomatis oleh `scripts/evaluate.py`
(accuracy, precision, recall, F1-score per kelas, dan confusion matrix).

## Dataset

Belum ada satu dataset siap pakai yang sudah punya ketiga kelas sekaligus
(organik, anorganik, B3) dalam satu sumber. Cara paling praktis: gabungkan
2 sumber Kaggle berikut ke dalam struktur folder yang sama.

**1. Organik + Anorganik** (dasar):
- https://www.kaggle.com/datasets/eldadvikorian/dataset-sampah-organik-dan-anorganik
  (organik ~14rb gambar, anorganik ~11rb gambar)
- Alternatif: https://www.kaggle.com/datasets/agustian132/dataset-sampah-organik-anorganik

**2. B3 (Bahan Berbahaya dan Beracun)** — biasanya diambil dari kelas
"battery"/e-waste pada dataset sampah multi-kelas, mis:
- Kaggle "Garbage Classification (12 classes)" — pakai subset kelas
  `battery` sebagai representasi B3. Cari dengan kata kunci
  "garbage classification 12 classes kaggle".
- Atau kumpulkan sendiri foto baterai bekas, lampu neon, kemasan obat,
  e-waste kecil (20–50 foto per sub-kategori sudah cukup untuk subset kecil).

Susun ke dalam struktur ImageFolder berikut di `data/raw/`:

```
data/raw/
    organik/
        img001.jpg
        ...
    anorganik/
        img001.jpg
        ...
    b3/
        img001.jpg
        ...
```

> Karena dataset organik/anorganik aslinya besar (~25rb gambar total),
> **sangat disarankan langsung pakai `make_subset.py`** (lihat bagian di
> bawah) supaya training di CPU tetap cepat — mis. ambil 300-500 gambar per
> kelas dulu untuk eksperimen, baru scale up kalau sudah yakin pipeline-nya
> jalan.

## Struktur Proyek

```
waste-classifier/
├── app/
│   ├── app.py              # Flask app (routing + upload handling)
│   ├── inference.py         # Load model + preprocessing + prediksi
│   ├── templates/
│   │   └── index.html       # UI upload & hasil klasifikasi
│   └── static/
├── scripts/
│   ├── utils.py               # Data pipeline & augmentasi bersama
│   ├── train_baseline.py      # Pendekatan 1: MobileNetV2
│   ├── train_improved.py      # Pendekatan 2: EfficientNetB0 + fine-tuning
│   ├── evaluate.py            # Perbandingan kedua model
│   ├── make_subset.py         # Bikin subset kecil dataset (untuk uji cepat)
│   └── check_gpu.py           # Cek apakah GPU terdeteksi
├── data/
│   └── raw/                    # Taruh dataset di sini (lihat di atas)
├── models/                      # Output: model .keras, class_indices.json, hasil evaluasi
├── notebooks/                    # (opsional) eksplorasi/EDA
├── requirements.txt
└── README.md
```

## Uji Cepat dengan Subset (Sangat Direkomendasikan)

Dataset organik/anorganik aslinya besar (~25rb gambar), jadi tetap disarankan
mulai dari subset kecil dulu:

1. Cek GPU:
   ```bash
   cd scripts
   python check_gpu.py
   ```

2. Buat subset (ambil semua 3 kelas kalau namanya persis `organik`,
   `anorganik`, `b3`; atau tentukan manual):
   ```bash
   python make_subset.py --classes organik anorganik b3 --max_per_class 300
   ```
   Hasilnya ada di `data/raw_subset/`.

3. Latih dengan setting cepat (gambar diperkecil, batch lebih besar):
   ```bash
   python train_baseline.py --data_dir ../data/raw_subset --epochs 8 \
       --img_size 128 --batch_size 64
   ```
   Dengan 3 kelas & ~300 gambar/kelas, ukuran 128px, ini biasanya hanya
   perlu beberapa menit per epoch di CPU — jauh lebih cepat dibanding
   dataset besar dengan ukuran gambar penuh (224px).

4. Training otomatis membuat checkpoint tiap epoch (`models/*_checkpoint.keras`
   + `models/*_training_state.json`). Kalau training terhenti (Ctrl+C, laptop
   mati, dll), tinggal jalankan command yang sama lagi — otomatis lanjut dari
   epoch terakhir, bukan mulai dari nol. Pakai `--fresh` untuk sengaja mulai
   ulang dari awal.

5. Kalau sudah oke, scale up: naikkan `--max_per_class` di `make_subset.py`,
   atau langsung pakai `--data_dir ../data/raw` (dataset penuh tanpa subset).

## Cara Menjalankan (Lengkap)

1. Buat virtual environment & install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate        # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Siapkan dataset di `data/raw/` (lihat bagian Dataset di atas).

3. Latih kedua model:
   ```bash
   cd scripts
   python train_baseline.py --data_dir ../data/raw_subset --img_size 128 --batch_size 64
   python train_improved.py --data_dir ../data/raw_subset --img_size 128 --batch_size 64
   ```

4. Bandingkan performa keduanya:
   ```bash
   python evaluate.py --data_dir ../data/raw_subset
   ```
   Hasil: `models/comparison_report.json` + 2 gambar confusion matrix.

5. Jalankan web app demo (menggunakan model improved secara default, fallback
   ke baseline jika belum ada):
   ```bash
   cd ../app
   python app.py
   ```
   Buka `http://127.0.0.1:5000`, unggah foto sampah, lihat top-3 prediksi
   kategori.

## Deployment

Untuk deploy ke Railway/Render, gunakan `app/app.py` sebagai entry point dan
pastikan file model (`models/*.keras`) ikut ter-deploy (gunakan Git LFS jika
ukurannya besar, karena `.gitignore` saat ini mengecualikan file model dari
repo standar).

## Rencana Commit (mengikuti timeline capstone)

- Minggu 1: struktur proyek awal, `utils.py`, README, riset & kumpul dataset
- Minggu 2: `train_baseline.py` + hasil eksperimen awal (subset kecil)
- Minggu 3: `train_improved.py`, `evaluate.py`, perbandingan hasil, scale up dataset
- Minggu 4: Flask app, polish UI, dokumentasi final

## Status

- [x] Kerangka proyek (scripts, app, struktur folder)
- [x] Pipeline data & augmentasi (`utils.py`)
- [x] Script training baseline (MobileNetV2) + resume dari checkpoint
- [x] Script training improved (EfficientNetB0, fine-tuning 2 tahap)
- [x] Script evaluasi & perbandingan
- [x] Script subset dataset & cek GPU (buat training cepat di CPU)
- [x] Flask web app untuk demo
- [ ] Kumpulkan/gabungkan dataset organik + anorganik + B3 ke `data/raw/`
- [ ] Training dijalankan dengan dataset asli
- [ ] Video demo (5–8 menit)
- [ ] Laporan ilmiah (belum dikerjakan sesuai permintaan — fokus kode dulu)
