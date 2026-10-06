"""
Verification and Runtime Benchmarking of Baseline Enhancement Algorithms
Course: Image and Video Processing (IVP) Mini-Project

This script:
    1. Loads representative low-light images from the development subset ('our485').
    2. Executes all five baseline enhancement techniques:
       - Gamma Correction (gamma=0.5)
       - Histogram Equalization (HE on YCrCb luminance)
       - CLAHE (clip_limit=2.0, tile_grid_size=(8, 8))
       - Denoising + CLAHE (Bilateral filter + CLAHE)
       - Single-Scale Retinex (SSR, sigma=80.0)
       - Multi-Scale Retinex (MSR, sigmas=(15.0, 80.0, 250.0))
    3. Benchmarks computational execution times across samples.
    4. Generates a multi-image comparison panel and runtime summary CSV.

IMPORTANT:
    - Strictly uses 'our485' development samples.
    - 'eval15' is reserved for held-out evaluation and is NOT accessed.
    - No quantitative PSNR/SSIM claims are asserted until Phase 8 (Evaluation Module).
"""

import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset_loader import LOLDatasetLoader
from src.enhancement_methods import (
    gamma_correction,
    histogram_equalization,
    clahe_enhancement,
    denoising_enhancement,
    single_scale_retinex,
    multi_scale_retinex,
)


def run_baseline_verification():
    print("=" * 70)
    print("BASELINE ENHANCEMENT METHODS VERIFICATION & RUNTIME BENCHMARK")
    print("=" * 70)

    comparisons_dir = PROJECT_ROOT / "results" / "comparisons"
    metrics_dir = PROJECT_ROOT / "results" / "metrics"
    comparisons_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    comp_img_path = comparisons_dir / "phase6_baseline_comparison.png"
    runtime_csv_path = metrics_dir / "phase6_runtime_summary.csv"

    loader = LOLDatasetLoader()
    # Select 3 representative development pairs from our485
    all_dev_pairs = loader.get_pairs("our485")
    sample_pairs = [all_dev_pairs[0], all_dev_pairs[5], all_dev_pairs[10]]
    print(f"Selected {len(sample_pairs)} development samples from 'our485': {[p.pair_id for p in sample_pairs]}")

    methods_dict = {
        "Gamma Correction (gamma=0.5)": lambda img: gamma_correction(img, gamma=0.5),
        "Histogram Equalization": lambda img: histogram_equalization(img),
        "CLAHE (clip=2.0)": lambda img: clahe_enhancement(img, clip_limit=2.0, tile_grid_size=(8, 8)),
        "Denoise (Bilateral) + CLAHE": lambda img: denoising_enhancement(
            img,
            denoise_method="bilateral",
            denoise_kwargs={"d": 5, "sigma_color": 50.0, "sigma_space": 50.0},
            enhancement_method="clahe",
            enhancement_kwargs={"clip_limit": 2.0}
        ),
        "Single-Scale Retinex (SSR)": lambda img: single_scale_retinex(img, sigma=80.0),
        "Multi-Scale Retinex (MSR)": lambda img: multi_scale_retinex(img, sigmas=(15.0, 80.0, 250.0)),
    }

    # Benchmark runtime on 10 development samples (repeated 3 times per image for timing precision)
    bench_samples = all_dev_pairs[:10]
    timing_records = {m_name: [] for m_name in methods_dict}

    print("\nBenchmarking execution runtimes over 10 development samples (3 trials each)...")
    for pair in bench_samples:
        low_img = pair.load_low(color_mode="RGB")
        for m_name, fn in methods_dict.items():
            # Warmup
            _ = fn(low_img)
            # Timed runs
            t_start = time.perf_counter()
            for _ in range(3):
                _ = fn(low_img)
            t_elapsed = (time.perf_counter() - t_start) / 3.0  # seconds per image
            timing_records[m_name].append(t_elapsed * 1000.0)  # in milliseconds

    # Compute runtime summary
    runtime_summary = []
    for m_name, times in timing_records.items():
        mean_ms = float(np.mean(times))
        std_ms = float(np.std(times))
        min_ms = float(np.min(times))
        max_ms = float(np.max(times))
        fps = 1000.0 / mean_ms if mean_ms > 0 else 0.0
        runtime_summary.append({
            "method": m_name,
            "mean_runtime_ms": round(mean_ms, 2),
            "std_runtime_ms": round(std_ms, 2),
            "min_runtime_ms": round(min_ms, 2),
            "max_runtime_ms": round(max_ms, 2),
            "approx_fps": round(fps, 1),
        })

    df_runtime = pd.DataFrame(runtime_summary)
    df_runtime.to_csv(runtime_csv_path, index=False)
    print(f"\n[SAVED] Runtime summary saved to: {runtime_csv_path.resolve()}")
    print(df_runtime.to_string(index=False))

    # Generate Visual Comparison Grid (Rows = samples, Columns = methods + input + reference)
    # Selected visual columns:
    col_names = [
        "Low-Light Input",
        "Gamma (gamma=0.5)",
        "Histogram Eq.",
        "CLAHE (clip=2.0)",
        "Denoise + CLAHE",
        "MSR (Multi-Scale)",
        "Reference (High)",
    ]

    fig, axes = plt.subplots(len(sample_pairs), len(col_names), figsize=(20, 3.2 * len(sample_pairs)))
    if len(sample_pairs) == 1:
        axes = np.expand_dims(axes, axis=0)

    for r_idx, pair in enumerate(sample_pairs):
        low_rgb, high_rgb = pair.load_pair(color_mode="RGB")
        out_gamma = gamma_correction(low_rgb, gamma=0.5)
        out_he = histogram_equalization(low_rgb)
        out_clahe = clahe_enhancement(low_rgb, clip_limit=2.0)
        out_denoise = denoising_enhancement(
            low_rgb,
            denoise_method="bilateral",
            enhancement_method="clahe"
        )
        out_msr = multi_scale_retinex(low_rgb, sigmas=(15.0, 80.0, 250.0))

        row_images = [low_rgb, out_gamma, out_he, out_clahe, out_denoise, out_msr, high_rgb]

        for c_idx, (col_title, img) in enumerate(zip(col_names, row_images)):
            ax = axes[r_idx, c_idx]
            ax.imshow(img)
            if r_idx == 0:
                ax.set_title(col_title, fontsize=11, fontweight="bold")
            if c_idx == 0:
                ax.set_ylabel(f"Sample: {pair.pair_id}", fontsize=11, fontweight="bold")
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(comp_img_path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"\n[SAVED] Visual comparison grid saved to: {comp_img_path.resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    run_baseline_verification()
