"""
Dataset Characteristic Analysis and Feature Distribution Profiling
Course: Image and Video Processing (IVP) Mini-Project

This script analyzes low-light images from the development subset ('our485'),
computes objective degradation metrics, saves the resulting structured dataset
to CSV, and produces empirical distribution plots.

IMPORTANT:
    - This script strictly operates on 'our485'.
    - 'eval15' is reserved as a held-out benchmark and is NOT accessed.
    - All outputs are saved in results/metrics/ and results/plots/.
    - The original lol_dataset/ is never modified.
"""

import argparse
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset_loader import LOLDatasetLoader
from src.image_analysis import analyze_image_characteristics


def run_dataset_analysis(sample_size: int = 100, analyze_all: bool = False):
    print("=" * 70)
    print("LOL DEVELOPMENT DATASET ('our485') CHARACTERISTIC ANALYSIS")
    print("=" * 70)

    # 1. Setup output directories
    metrics_dir = PROJECT_ROOT / "results" / "metrics"
    plots_dir = PROJECT_ROOT / "results" / "plots"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    csv_path = metrics_dir / "our485_characteristics.csv"
    plot_path = plots_dir / "our485_feature_distributions.png"

    # 2. Load pairs from our485 (strictly development subset)
    loader = LOLDatasetLoader()
    all_pairs = loader.get_pairs("our485")
    total_available = len(all_pairs)

    if analyze_all or sample_size >= total_available:
        target_pairs = all_pairs
        print(f"Profiling ALL {len(target_pairs)} images in 'our485' development set...")
    else:
        target_pairs = all_pairs[:sample_size]
        print(f"Profiling representative sample of {len(target_pairs)}/{total_available} images in 'our485'...")

    records = []
    for i, pair in enumerate(target_pairs, start=1):
        low_rgb = pair.load_low(color_mode="RGB")
        res = analyze_image_characteristics(low_rgb, image_name=pair.pair_id, is_rgb=True)
        records.append(res.to_dict())
        if i % 50 == 0 or i == len(target_pairs):
            print(f"  Processed {i:3d}/{len(target_pairs)} images...")

    df = pd.DataFrame(records)

    # 3. Save CSV report
    df.to_csv(csv_path, index=False)
    print(f"\n[SAVED] Analysis metrics saved to CSV: {csv_path.resolve()}")
    print(f"  Total records: {len(df)}")
    print(f"  Columns: {list(df.columns)}")

    # 4. Summary statistics table
    numeric_cols = [
        "mean_brightness", "median_brightness", "global_contrast",
        "local_contrast", "dynamic_range", "noise_estimate",
        "illumination_nonuniformity", "entropy"
    ]
    summary_stats = df[numeric_cols].describe().T[["mean", "std", "min", "50%", "max"]]
    summary_stats.rename(columns={"50%": "median"}, inplace=True)
    print("\n--- Empirical Distribution Summary (our485) ---")
    print(summary_stats.to_string())

    # 5. Generate Distribution Plots
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(
        f"Degradation Characteristic Distributions (LOL Development Split 'our485', N={len(df)})",
        fontsize=14,
        fontweight="bold",
        y=0.98
    )

    plot_configs = [
        ("mean_brightness", "Mean Luminance (Y)", "Intensity [0-255]", axes[0, 0], "#1f77b4"),
        ("global_contrast", "Global Contrast (RMS Std Dev)", "Std Dev", axes[0, 1], "#2ca02c"),
        ("local_contrast", "Local Contrast (16x16 Blocks)", "Local Std Dev", axes[0, 2], "#ff7f0e"),
        ("dynamic_range", "Effective Dynamic Range (P95 - P5)", "Intensity Span", axes[1, 0], "#9467bd"),
        ("noise_estimate", "High-Freq Noise Estimate (Immerkaer)", r"Estimated $\sigma_n$", axes[1, 1], "#d62728"),
        ("illumination_nonuniformity", "Illumination Non-Uniformity Index", "CV Score", axes[1, 2], "#8c564b"),
    ]

    for col, title, xlabel, ax, color in plot_configs:
        series = df[col]
        ax.hist(series, bins=25, color=color, alpha=0.7, edgecolor="black")
        mean_val = series.mean()
        med_val = series.median()
        ax.axvline(mean_val, color="red", linestyle="--", linewidth=1.5, label=f"Mean: {mean_val:.2f}")
        ax.axvline(med_val, color="blue", linestyle=":", linewidth=1.5, label=f"Median: {med_val:.2f}")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel("Image Frequency", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] Distribution multi-panel plot saved to: {plot_path.resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze LOL Development Set Image Characteristics")
    parser.add_argument("--sample_size", type=int, default=100, help="Number of images from our485 to analyze")
    parser.add_argument("--all", action="store_true", help="Analyze all 485 images in our485")
    args = parser.parse_args()

    run_dataset_analysis(sample_size=args.sample_size, analyze_all=args.all)
