"""
Command-Line Demonstration Script
Course: Image and Video Processing (IVP) Mini-Project

Usage:
    python scripts/run_demo_cli.py [--sample 1.png] [--image path/to/image.png]

Performs headless demonstration:
    1. Loads input low-light image
    2. Runs content-aware degradation analysis
    3. Executes adaptive decision engine with transparent rationale
    4. Runs and compares baseline enhancement algorithms
    5. Saves output images to results/demo_outputs/
"""

import argparse
import sys
import time
from pathlib import Path
import cv2
import numpy as np

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


def run_cli_demo(sample_id: str = "1.png", custom_path: str = None) -> None:
    print("=" * 78)
    print("ADAPTIVE LOW-LIGHT ENHANCEMENT — CLI DEMONSTRATION")
    print("=" * 78)

    out_dir = PROJECT_ROOT / "results" / "demo_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    if custom_path:
        img_p = Path(custom_path).resolve()
        if not img_p.exists():
            raise FileNotFoundError(f"Custom image not found: {img_p}")
        bgr = cv2.imread(str(img_p))
        if bgr is None:
            raise IOError(f"Could not read image: {img_p}")
        img_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        name = img_p.name
    else:
        loader = LOLDatasetLoader()
        pair = loader.get_pair_by_id(sample_id, subset="eval15")
        img_rgb = pair.load_low(color_mode="RGB")
        name = sample_id

    print(f"\n[1] Loaded Input Image: '{name}' | Dimensions: {img_rgb.shape[1]}x{img_rgb.shape[0]} px (RGB)")

    # Step 1: Content-Aware Analysis
    print("\n[2] Extracting Content-Aware Image Characteristics...")
    t0 = time.perf_counter()
    analysis = analyze_image_characteristics(img_rgb, image_name=name, is_rgb=True)
    chars = analysis.to_dict()

    print("    ----------------------------------------------------------")
    print(f"    * Mean Luminance (Y)        : {chars['mean_brightness']:>7.2f}  [Reference Range: 0 - 255]")
    print(f"    * Global RMS Contrast       : {chars['global_contrast']:>7.2f}  [Dev Q1: 6.61, Med: 10.15, Q3: 12.95]")
    print(f"    * Dynamic Range (P95 - P5)  : {chars['dynamic_range']:>7.1f}  [Dev Q1: 18.0, Med: 30.0]")
    print(f"    * Noise Estimate (sigma_n)  : {chars['noise_estimate']:>7.2f}  [Dev Med: 1.14, Q3: 1.23]")
    print(f"    * Illumination Non-Uniform. : {chars['illumination_nonuniformity']:>7.3f}  [Dev Med: 0.49, Q3: 0.64]")
    print("    ----------------------------------------------------------")

    # Step 2: Adaptive Enhancement Pipeline
    print("\n[3] Executing Content-Aware Adaptive Decision Engine...")
    t_adapt_start = time.perf_counter()
    adapt_res = adaptive_enhance(img_rgb, image_name=name, is_rgb=True)
    t_adapt_end = time.perf_counter()
    adapt_ms = (t_adapt_end - t_adapt_start) * 1000.0

    print("    ----------------------------------------------------------")
    print(f"    * SELECTED METHOD       : {adapt_res.selected_method.upper()}")
    print(f"    * DEGRADATION PROFILE   : {adapt_res.degradation_profile}")
    print(f"    * MATCH CONFIDENCE      : {adapt_res.decision_strength}")
    print(f"    * CONFIGURED PARAMETERS : {adapt_res.selected_parameters}")
    print(f"    * EXECUTION LATENCY     : {adapt_ms:.1f} ms")
    print("    * DECISION REASON:")
    print(f"      \"{adapt_res.decision_reason}\"")
    print("    ----------------------------------------------------------")

    # Save adaptive output
    adapt_out_path = out_dir / f"demo_{Path(name).stem}_adaptive_{adapt_res.selected_method}.png"
    cv2.imwrite(str(adapt_out_path), cv2.cvtColor(adapt_res.enhanced_image, cv2.COLOR_RGB2BGR))
    print(f"\n[4] Saved Adaptive Enhanced Image to: {adapt_out_path.relative_to(PROJECT_ROOT)}")

    # Step 3: Fast Baseline Comparison
    print("\n[5] Comparing Against Fixed Baseline Methods on the Same Image:")
    baselines = {
        "Gamma Correction (0.5)": lambda img: gamma_correction(img, gamma=0.5),
        "Histogram Equalization": lambda img: histogram_equalization(img),
        "CLAHE (2.0)": lambda img: clahe_enhancement(img, clip_limit=2.0, tile_grid_size=(8, 8)),
        "Denoise (Bilateral) + CLAHE": lambda img: denoising_enhancement(img, denoise_method="bilateral", enhancement_method="clahe"),
        "Single-Scale Retinex (SSR)": lambda img: single_scale_retinex(img, sigma=80.0),
        "Multi-Scale Retinex (MSR)": lambda img: multi_scale_retinex(img, sigmas=(15.0, 80.0, 250.0)),
    }

    print("    " + "-" * 70)
    print(f"    {'Enhancement Pipeline':<32} | {'Output Mean Y':<15} | {'Latency (ms)':<12}")
    print("    " + "-" * 70)

    # Adaptive row
    ad_mean = float(np.mean(cv2.cvtColor(adapt_res.enhanced_image, cv2.COLOR_RGB2YCrCb)[:, :, 0]))
    print(f"    {'Proposed Adaptive (Ours)':<32} | {ad_mean:>15.2f} | {adapt_ms:>10.1f} ms")

    for b_name, b_fn in baselines.items():
        tb0 = time.perf_counter()
        b_out = b_fn(img_rgb)
        tb1 = time.perf_counter()
        b_ms = (tb1 - tb0) * 1000.0
        b_mean = float(np.mean(cv2.cvtColor(b_out, cv2.COLOR_RGB2YCrCb)[:, :, 0]))
        print(f"    {b_name:<32} | {b_mean:>15.2f} | {b_ms:>10.1f} ms")

    print("    " + "-" * 70)
    print("\n" + "=" * 78)
    print("CLI DEMONSTRATION COMPLETE")
    print("=" * 78)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Adaptive Low-Light Enhancement CLI Demo")
    parser.add_argument("--sample", type=str, default="1.png", help="Sample ID from eval15 (default: 1.png)")
    parser.add_argument("--image", type=str, default=None, help="Custom image path")
    args = parser.parse_args()

    run_cli_demo(sample_id=args.sample, custom_path=args.image)
