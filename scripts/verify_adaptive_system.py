"""
Visual Verification and Decision Inspection of the Adaptive Enhancement System
Course: Image and Video Processing (IVP) Mini-Project

This script:
    1. Loads 4 representative scenes from the development subset ('our485') that span
       the distinct degradation profiles (Severe Underexposure, High Noise, Low Contrast, Mild Underexposure).
    2. Runs all five fixed baseline enhancement techniques alongside the Proposed Adaptive Engine.
    3. Produces a side-by-side comparative panel (4 scenes x 8 columns) highlighting
       the adaptively selected method.
    4. Prints complete transparent decision records for all tested scenes.

IMPORTANT:
    - Strictly operates on 'our485'.
    - 'eval15' is reserved for held-out evaluation and is NOT accessed.
"""

from pathlib import Path
import time
# Matplotlib may be blocked by Windows Application Control policy.
# Visual plotting is optional; core adaptive verification does not depend on it.
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    plt = None
    MATPLOTLIB_AVAILABLE = False
import numpy as np

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
    multi_scale_retinex,
)
from src.adaptive_enhancement import adaptive_enhance


def run_adaptive_verification():
    print("=" * 75)
    print("PROPOSED ADAPTIVE ENHANCEMENT SYSTEM - VISUAL VERIFICATION")
    print("=" * 75)

    comp_dir = PROJECT_ROOT / "results" / "comparisons"
    comp_dir.mkdir(parents=True, exist_ok=True)
    out_fig_path = comp_dir / "phase7_adaptive_comparison.png"

    loader = LOLDatasetLoader()
    # 4 distinct degradation profile samples discovered in our485
    target_sample_ids = ["2.png", "5.png", "17.png", "21.png"]
    sample_pairs = [loader.get_pair_by_id(pid, subset="our485") for pid in target_sample_ids]

    col_headers = [
        "Low-Light Input",
        "Gamma (0.5)",
        "Histogram Eq.",
        "CLAHE (2.0)",
        "Denoise + CLAHE",
        "MSR (Retinex)",
        "PROPOSED ADAPTIVE",
        "Reference Ground Truth"
    ]

    print("\n--- Processing and Inspecting Adaptive Decisions ---")
    
    # Store results for visualization if matplotlib is available
    all_results = []
    
    for r_idx, pair in enumerate(sample_pairs):
        low_rgb, high_rgb = pair.load_pair(color_mode="RGB")

        # Run fixed baselines
        out_gamma = gamma_correction(low_rgb, gamma=0.5)
        out_he = histogram_equalization(low_rgb)
        out_clahe = clahe_enhancement(low_rgb, clip_limit=2.0)
        out_denoise = denoising_enhancement(low_rgb, denoise_method="bilateral", enhancement_method="clahe")
        out_msr = multi_scale_retinex(low_rgb, sigmas=(15.0, 80.0, 250.0))

        # Run Proposed Adaptive Engine (timed)
        t0 = time.perf_counter()
        adaptive_result = adaptive_enhance(low_rgb, image_name=pair.pair_id)
        adaptive_ms = (time.perf_counter() - t0) * 1000.0

        out_adaptive = adaptive_result.enhanced_image

        print(f"\n[SCENE: {pair.pair_id}] (Runtime: {adaptive_ms:.1f} ms)")
        print(f"  Degradation Profile: {adaptive_result.degradation_profile}")
        print(f"  Selected Method:     {adaptive_result.selected_method}")
        print(f"  Decision Strength:   {adaptive_result.decision_strength}")
        print(f"  Configured Params:   {adaptive_result.selected_parameters}")
        print(f"  Decision Reason:     {adaptive_result.decision_reason}")

        # Store results for visualization
        all_results.append({
            'pair': pair,
            'low_rgb': low_rgb,
            'high_rgb': high_rgb,
            'out_gamma': out_gamma,
            'out_he': out_he,
            'out_clahe': out_clahe,
            'out_denoise': out_denoise,
            'out_msr': out_msr,
            'out_adaptive': out_adaptive,
            'adaptive_result': adaptive_result
        })

    if MATPLOTLIB_AVAILABLE:
        fig, axes = plt.subplots(len(sample_pairs), len(col_headers), figsize=(24, 3.4 * len(sample_pairs)))

        for r_idx, result in enumerate(all_results):
            pair = result['pair']
            low_rgb = result['low_rgb']
            high_rgb = result['high_rgb']
            out_gamma = result['out_gamma']
            out_he = result['out_he']
            out_clahe = result['out_clahe']
            out_denoise = result['out_denoise']
            out_msr = result['out_msr']
            out_adaptive = result['out_adaptive']
            adaptive_result = result['adaptive_result']

            row_images = [
                low_rgb,
                out_gamma,
                out_he,
                out_clahe,
                out_denoise,
                out_msr,
                out_adaptive,
                high_rgb
            ]

            for c_idx, (col_title, img) in enumerate(zip(col_headers, row_images)):
                ax = axes[r_idx, c_idx]
                ax.imshow(img)

                # Header row
                if r_idx == 0:
                    if c_idx == 6:
                        ax.set_title(col_title, fontsize=10, fontweight="bold", color="darkgreen")
                    else:
                        ax.set_title(col_title, fontsize=10, fontweight="bold")

                # Row label
                if c_idx == 0:
                    ax.set_ylabel(f"Sample: {pair.pair_id}", fontsize=11, fontweight="bold")

                # Highlight the adaptive output with a distinct green border
                if c_idx == 6:
                    for spine in ax.spines.values():
                        spine.set_edgecolor("#2ca02c")
                        spine.set_linewidth(3.0)
                    # Display selected method subcaption
                    short_name = adaptive_result.selected_method.replace("_enhancement", "").replace("_correction", "")
                    ax.set_xlabel(f"[{short_name.upper()}]", fontsize=9, fontweight="bold", color="darkgreen")
                else:
                    ax.set_xticks([])
                    ax.set_yticks([])

                ax.set_xticks([])
                ax.set_yticks([])

        plt.tight_layout()
        plt.savefig(out_fig_path, dpi=160, bbox_inches="tight")
        plt.close()

        print(f"\n[SAVED] Adaptive comparison panel saved to: {out_fig_path.resolve()}")
    else:
        print("\nMatplotlib unavailable; skipping visual comparison figure. Core adaptive verification continues.")
    print("=" * 75)


if __name__ == "__main__":
    run_adaptive_verification()
