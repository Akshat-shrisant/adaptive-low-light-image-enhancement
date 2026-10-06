"""
Faculty Demonstration Application Server
Course: Image and Video Processing (IVP) Mini-Project

A lightweight, self-contained local web demonstration for presenting the
Content-Aware Adaptive Low-Light Image Enhancement framework to faculty and peers.

Features:
    - Pre-loaded representative benchmark scenes + custom image file upload.
    - Real-time content-aware feature extraction (Brightness, Contrast, Dynamic Range, Noise, INU).
    - Transparent decision explanation generated directly by the adaptive engine.
    - Side-by-side and interactive before/after visual comparison.
    - Execution latency timing (ms).
    - On-demand comparison with all 6 baseline enhancement methods.
    - 100% local, no external internet/CDN dependencies.
"""

import base64
import io
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Tuple

import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
from PIL import Image

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset_loader import LOLDatasetLoader
from src.image_analysis import analyze_image_characteristics
from src.adaptive_enhancement import adaptive_enhance
from src.enhancement_methods import (
    gamma_correction,
    histogram_equalization,
    clahe_enhancement,
    denoising_enhancement,
    single_scale_retinex,
    multi_scale_retinex,
)

app = Flask(__name__, template_folder="templates")
app.config["TEMPLATES_AUTO_RELOAD"] = True

# Initialize dataset loader for pre-loaded samples
try:
    loader = LOLDatasetLoader()
except Exception:
    loader = None


def array_to_base64_jpeg(img_rgb: np.ndarray, quality: int = 92) -> str:
    """Convert a uint8 RGB numpy array to base64 JPEG data URL."""
    pil_img = Image.fromarray(img_rgb)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG", quality=quality)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


# Representative benchmark scenes covering the 4 degradation profiles
CURATED_SAMPLES = [
    {
        "id": "1.png",
        "subset": "eval15",
        "label": "1.png — Severe Illumination Non-Uniformity (eval15)",
        "expected": "Multi-Scale Retinex (MSR)",
    },
    {
        "id": "22.png",
        "subset": "eval15",
        "label": "22.png — Mild Underexposure / Clean Signal (eval15)",
        "expected": "Adaptive Gamma Correction",
    },
    {
        "id": "55.png",
        "subset": "eval15",
        "label": "55.png — Uniform Low-Contrast Underexposure (eval15)",
        "expected": "Adaptive CLAHE",
    },
    {
        "id": "493.png",
        "subset": "eval15",
        "label": "493.png — Severe Underexposure & Compressed Range (eval15)",
        "expected": "Multi-Scale Retinex (MSR)",
    },
    {
        "id": "780.png",
        "subset": "eval15",
        "label": "780.png — High Sensor Noise Degradation (eval15)",
        "expected": "Denoising (Bilateral) + CLAHE",
    },
    {
        "id": "111.png",
        "subset": "eval15",
        "label": "111.png — Moderate Underexposure (Balanced Fallback) (eval15)",
        "expected": "Standard CLAHE",
    },
]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/samples", methods=["GET"])
def get_samples():
    """Return available curated samples for 1-click loading."""
    return jsonify({"samples": CURATED_SAMPLES})


def load_input_image_from_request() -> Tuple[np.ndarray, str]:
    """Extract RGB image array from either uploaded file or sample ID."""
    if "file" in request.files and request.files["file"].filename:
        file = request.files["file"]
        file_bytes = np.frombuffer(file.read(), np.uint8)
        bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("Could not decode uploaded image file.")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        name = file.filename
        return rgb, name

    sample_id = request.form.get("sample_id") or request.json.get("sample_id")
    if sample_id and loader:
        # Determine subset (default eval15)
        subset = request.form.get("subset") or (request.json.get("subset") if request.json else None) or "eval15"
        pair = loader.get_pair_by_id(sample_id, subset=subset)
        rgb = pair.load_low(color_mode="RGB")
        return rgb, sample_id

    raise ValueError("No valid image file or sample_id provided.")


@app.route("/api/analyze_and_enhance", methods=["POST"])
def analyze_and_enhance():
    """
    Core demonstration endpoint:
    1. Reads input image
    2. Runs content-aware analysis
    3. Executes adaptive decision engine
    4. Measures execution latency
    5. Returns transparent decision report and base64 rendered images
    """
    try:
        rgb, name = load_input_image_from_request()

        # Step 1: Content-Aware Analysis
        analysis = analyze_image_characteristics(rgb, image_name=name, is_rgb=True)

        # Step 2: Adaptive Decision & Enhancement (Timed)
        t_start = time.perf_counter()
        decision_res = adaptive_enhance(rgb, image_name=name, is_rgb=True)
        t_end = time.perf_counter()
        elapsed_ms = (t_end - t_start) * 1000.0

        # Render base64 images
        input_b64 = array_to_base64_jpeg(rgb)
        enhanced_b64 = array_to_base64_jpeg(decision_res.enhanced_image)

        chars = analysis.to_dict()

        return jsonify({
            "status": "success",
            "image_name": name,
            "dimensions": f"{rgb.shape[1]}x{rgb.shape[0]} px",
            "image_characteristics": {
                "mean_brightness": round(float(chars["mean_brightness"]), 2),
                "global_contrast": round(float(chars["global_contrast"]), 2),
                "dynamic_range": round(float(chars["dynamic_range"]), 1),
                "noise_estimate": round(float(chars["noise_estimate"]), 2),
                "illumination_nonuniformity": round(float(chars["illumination_nonuniformity"]), 3),
            },
            "adaptive_decision": {
                "selected_method": decision_res.selected_method,
                "degradation_profile": decision_res.degradation_profile,
                "decision_strength": decision_res.decision_strength,
                "decision_reason": decision_res.decision_reason,
                "selected_parameters": decision_res.selected_parameters,
            },
            "processing_time_ms": round(float(elapsed_ms), 1),
            "input_image_b64": input_b64,
            "enhanced_image_b64": enhanced_b64,
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route("/api/run_baseline", methods=["POST"])
def run_baseline():
    """
    Run an individual baseline enhancement method for interactive comparison.
    Methods: 'gamma', 'he', 'clahe', 'denoise_clahe', 'ssr', 'msr'
    """
    try:
        rgb, name = load_input_image_from_request()
        method_key = request.form.get("method") or (request.json.get("method") if request.json else "clahe")

        t_start = time.perf_counter()
        if method_key == "gamma":
            out = gamma_correction(rgb, gamma=0.5)
            display_name = "Gamma Correction (gamma=0.5)"
        elif method_key == "he":
            out = histogram_equalization(rgb)
            display_name = "Histogram Equalization (HE)"
        elif method_key == "clahe":
            out = clahe_enhancement(rgb, clip_limit=2.0, tile_grid_size=(8, 8))
            display_name = "CLAHE (clip=2.0, grid=8x8)"
        elif method_key == "denoise_clahe":
            out = denoising_enhancement(
                rgb,
                denoise_method="bilateral",
                denoise_kwargs={"d": 5, "sigma_color": 50.0, "sigma_space": 50.0},
                enhancement_method="clahe",
                enhancement_kwargs={"clip_limit": 2.0}
            )
            display_name = "Denoise (Bilateral) + CLAHE"
        elif method_key == "ssr":
            out = single_scale_retinex(rgb, sigma=80.0)
            display_name = "Single-Scale Retinex (SSR, sigma=80)"
        elif method_key == "msr":
            out = multi_scale_retinex(rgb, sigmas=(15.0, 80.0, 250.0))
            display_name = "Multi-Scale Retinex (MSR, sigmas=15,80,250)"
        elif method_key == "adaptive":
            res = adaptive_enhance(rgb, image_name=name)
            out = res.enhanced_image
            display_name = f"Proposed Adaptive ({res.selected_method})"
        else:
            raise ValueError(f"Unknown baseline method '{method_key}'")

        t_end = time.perf_counter()
        elapsed_ms = (t_end - t_start) * 1000.0

        out_b64 = array_to_base64_jpeg(out)

        return jsonify({
            "status": "success",
            "method_key": method_key,
            "display_name": display_name,
            "processing_time_ms": round(float(elapsed_ms), 1),
            "enhanced_image_b64": out_b64,
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("=" * 70)
    print("STARTING FACULTY DEMONSTRATION SERVER")
    print(f"URL: http://127.0.0.1:{port}")
    print("Press Ctrl+C to stop.")
    print("=" * 70)
    app.run(host="127.0.0.1", port=port, debug=False)
