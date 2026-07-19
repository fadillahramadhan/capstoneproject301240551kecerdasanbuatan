"""
app.py
Flask web app untuk demo klasifikasi jenis sampah (organik / anorganik / B3).
Upload foto sampah -> menampilkan probabilitas tiap kategori beserta confidence.

Jalankan:
    cd app
    python app.py
Lalu buka http://127.0.0.1:5000
"""

import os
import sys

# allow importing inference.py when run from project root or from app/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, render_template, request, jsonify
from inference import predict

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB upload limit

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_route():
    if "image" not in request.files:
        return jsonify({"error": "Tidak ada file gambar yang diunggah."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Nama file kosong."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Format file harus PNG/JPG/JPEG."}), 400

    try:
        results, model_name = predict(file)
        return jsonify({"predictions": results, "model": model_name})
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        return jsonify({"error": f"Terjadi kesalahan saat prediksi: {e}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
