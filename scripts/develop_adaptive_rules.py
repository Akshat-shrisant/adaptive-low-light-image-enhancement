"""
Development and Empirical Calibration of Adaptive Enhancement Rules
Course: Image and Video Processing (IVP) Mini-Project

This script:
    1. Loads the development dataset characteristics from 'results/metrics/our485_characteristics.csv'.
    2. Calculates distribution quartiles (Q1, Median, Q3) for all degradation features.
    3. Formulates multi-feature degradation profiles and corresponding enhancement assignments.
    4. Evaluates the selection distribution across development images.
    5. Saves detailed rule development records and summary statistics.
    6. Generates diagnostic feature-method relationship plots.

IMPORTANT:
    - Strictly calibrated on 'our485'.
    - 'eval15' is reserved for held-out evaluation and is NOT accessed.
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def develop_adaptive_rules():
    print("=" * 70)
    print("EMPIRICAL ADAPTIVE RULE CALIBRATION (DEVELOPMENT SET 'our485')")
    print("=" * 70)

    input_csv = PROJECT_ROOT / "results" / "metrics" / "our485_characteristics.csv"
    if not input_csv.exists():
        raise FileNotFoundError(f"Input characteristics file not found: {input_csv}")

    df = pd.read_csv(input_csv)
    n_samples = len(df)
    print(f"Loaded {n_samples} development samples from: {input_csv.resolve()}")

    # 1. Compute empirical quartiles
    stats = {}
    features = [
        "mean_brightness", "global_contrast", "local_contrast",
        "dynamic_range", "noise_estimate", "illumination_nonuniformity"
    ]
    for feat in features:
        q1 = float(df[feat].quantile(0.25))
        med = float(df[feat].quantile(0.50))
        q3 = float(df[feat].quantile(0.75))
        stats[feat] = {"q1": q1, "med": med, "q3": q3}

    print("\n--- Empirical Quartile Boundaries (our485) ---")
    for feat, q in stats.items():
        print(f"  {feat:<28}: Q1={q['q1']:6.2f} | Med={q['med']:6.2f} | Q3={q['q3']:6.2f}")

    # 2. Decision Logic Calibration
    # Define multi-feature decision rules based on quartiles
    decisions = []
    for _, row in df.iterrows():
        b = row["mean_brightness"]
        c = row["global_contrast"]
        n = row["noise_estimate"]
        inu = row["illumination_nonuniformity"]
        dr = row["dynamic_range"]

        # Rule 1: High Sensor Noise Degraded
        # High noise (>= Q3) that would be aggressively amplified by contrast expansion
        if n >= stats["noise_estimate"]["q3"] and b < stats["mean_brightness"]["q3"]:
            method = "denoising_enhancement"
            profile = "High-Noise Underexposure"
            strength = "Strong Match"
            reason = (f"Elevated high-frequency noise (sigma_n={n:.2f} >= Q3:{stats['noise_estimate']['q3']:.2f}) "
                      f"with underexposure (mean_Y={b:.2f}); bilateral pre-filtering required before contrast expansion.")
            params = {
                "denoise_method": "bilateral",
                "d": 5,
                "sigma_color": round(40.0 + 15.0 * (n - stats["noise_estimate"]["q3"]), 1),
                "sigma_space": 50.0,
                "enhancement_method": "clahe",
                "clip_limit": 2.0
            }

        # Rule 2: Severe Underexposure (< Q1 Brightness AND < Q1 Dynamic Range)
        elif b <= stats["mean_brightness"]["q1"] and dr <= stats["dynamic_range"]["med"]:
            method = "multi_scale_retinex"
            profile = "Severe Low-Light & Compressed Range"
            strength = "Strong Match"
            reason = (f"Severe underexposure (mean_Y={b:.2f} <= Q1:{stats['mean_brightness']['q1']:.2f}) "
                      f"and compressed range (DR={dr:.1f} <= Med:{stats['dynamic_range']['med']:.1f}); "
                      f"multi-scale illumination decomposition required to reveal buried structures.")
            params = {
                "sigmas": (15.0, 80.0, 250.0),
                "color_space": "ycrcb"
            }

        # Rule 3: Highly Non-Uniform Illumination (INU >= Q3)
        elif inu >= stats["illumination_nonuniformity"]["q3"]:
            method = "multi_scale_retinex"
            profile = "Severe Illumination Non-Uniformity"
            strength = "Strong Match"
            reason = (f"Pronounced spatial illumination gradient (INU={inu:.3f} >= Q3:{stats['illumination_nonuniformity']['q3']:.3f}); "
                      f"Retinex surround decomposition required to normalize spatial lighting field.")
            params = {
                "sigmas": (15.0, 80.0, 250.0),
                "color_space": "ycrcb"
            }

        # Rule 4: Flat Uniform Low Contrast (Contrast < Median, Low Noise, Uniform Illumination)
        elif c < stats["global_contrast"]["med"] and n < stats["noise_estimate"]["q3"]:
            method = "clahe_enhancement"
            profile = "Uniform Low-Contrast Underexposure"
            strength = "Strong Match"
            reason = (f"Suppressed global contrast (std_Y={c:.2f} < Med:{stats['global_contrast']['med']:.2f}) "
                      f"under relatively even illumination (INU={inu:.3f} < Q3) and acceptable noise (sigma_n={n:.2f}); "
                      f"local adaptive histogram tile equalization optimal.")
            # Dynamically adapt clip limit: lower dynamic range gets slightly higher clip limit
            adapted_clip = round(float(np.clip(2.0 + (stats["dynamic_range"]["med"] - dr) / 25.0, 1.5, 3.2)), 2)
            params = {
                "clip_limit": adapted_clip,
                "tile_grid_size": (8, 8),
                "color_space": "ycrcb"
            }

        # Rule 5: Mild Underexposure with Preserved Contrast & Clean Signal
        elif b >= stats["mean_brightness"]["med"] and c >= stats["global_contrast"]["med"] and n < stats["noise_estimate"]["med"]:
            method = "gamma_correction"
            profile = "Mild Underexposure (Clean Signal)"
            strength = "Strong Match"
            reason = (f"Mild underexposure (mean_Y={b:.2f} >= Med:{stats['mean_brightness']['med']:.2f}) "
                      f"with preserved contrast (std_Y={c:.2f}) and low noise (sigma_n={n:.2f}); "
                      f"smooth monotonic power-law expansion provides natural brightening without over-processing.")
            adapted_gamma = round(float(np.clip(0.35 + 0.015 * b, 0.45, 0.70)), 2)
            params = {
                "gamma": adapted_gamma,
                "color_space": "ycrcb"
            }

        # Rule 6: Moderate Underexposure Fallback
        else:
            method = "clahe_enhancement"
            profile = "Moderate Underexposure (Balanced)"
            strength = "Moderate Match"
            reason = (f"Balanced intermediate degradation (mean_Y={b:.2f}, std_Y={c:.2f}, INU={inu:.3f}, sigma_n={n:.2f}); "
                      f"standard CLAHE provides robust local dynamic expansion.")
            params = {
                "clip_limit": 2.0,
                "tile_grid_size": (8, 8),
                "color_space": "ycrcb"
            }

        decisions.append({
            "image_name": row["image_name"],
            "mean_brightness": b,
            "global_contrast": c,
            "local_contrast": row["local_contrast"],
            "noise_estimate": n,
            "illumination_nonuniformity": inu,
            "dynamic_range": dr,
            "degradation_profile": profile,
            "selected_method": method,
            "decision_strength": strength,
            "decision_reason": reason,
            "parameters_summary": str(params)
        })

    df_decisions = pd.DataFrame(decisions)

    # 3. Save detailed decision development records
    output_dev_csv = PROJECT_ROOT / "results" / "metrics" / "adaptive_rule_development.csv"
    output_summary_csv = PROJECT_ROOT / "results" / "metrics" / "adaptive_rule_summary.csv"

    df_decisions.to_csv(output_dev_csv, index=False)
    print(f"\n[SAVED] Rule development records saved to: {output_dev_csv.resolve()}")

    # 4. Generate Summary Distribution
    summary_counts = df_decisions["selected_method"].value_counts().reset_index()
    summary_counts.columns = ["selected_method", "count"]
    summary_counts["percentage"] = (summary_counts["count"] / n_samples * 100.0).round(1)

    profile_counts = df_decisions["degradation_profile"].value_counts().reset_index()
    profile_counts.columns = ["degradation_profile", "count"]
    profile_counts["percentage"] = (profile_counts["count"] / n_samples * 100.0).round(1)

    summary_df = pd.merge(summary_counts, df_decisions.groupby("selected_method")[features].mean().round(2), on="selected_method")
    summary_df.to_csv(output_summary_csv, index=False)
    print(f"[SAVED] Rule selection summary saved to: {output_summary_csv.resolve()}")

    print("\n--- Selection Distribution across Development Set (our485) ---")
    print(summary_counts.to_string(index=False))

    print("\n--- Degradation Profile Breakdown ---")
    print(profile_counts.to_string(index=False))

    # 5. Visual diagnostic plot: Feature vs Method Relationships
    plot_path = PROJECT_ROOT / "results" / "plots" / "adaptive_feature_method_relationships.png"
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Adaptive Rule Selection Relationships (LOL Development Set 'our485')", fontsize=13, fontweight="bold")

    method_colors = {
        "denoising_enhancement": "#d62728",
        "multi_scale_retinex": "#9467bd",
        "clahe_enhancement": "#2ca02c",
        "gamma_correction": "#1f77b4"
    }

    # Plot 1: Mean Brightness vs Global Contrast
    ax = axes[0]
    for m, col in method_colors.items():
        subset = df_decisions[df_decisions["selected_method"] == m]
        ax.scatter(subset["mean_brightness"], subset["global_contrast"], c=col, label=m, alpha=0.7, edgecolors="k", s=45)
    ax.axvline(stats["mean_brightness"]["q1"], color="gray", linestyle="--", alpha=0.6, label="B Q1 (8.2)")
    ax.axhline(stats["global_contrast"]["med"], color="gray", linestyle=":", alpha=0.6, label="C Med (10.1)")
    ax.set_title("Brightness vs Global Contrast", fontsize=11, fontweight="bold")
    ax.set_xlabel("Mean Luminance (Y)")
    ax.set_ylabel("Global Contrast (RMS)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.4)

    # Plot 2: Noise Estimate vs Brightness
    ax = axes[1]
    for m, col in method_colors.items():
        subset = df_decisions[df_decisions["selected_method"] == m]
        ax.scatter(subset["noise_estimate"], subset["mean_brightness"], c=col, label=m, alpha=0.7, edgecolors="k", s=45)
    ax.axvline(stats["noise_estimate"]["q3"], color="red", linestyle="--", alpha=0.7, label=f"Noise Q3 ({stats['noise_estimate']['q3']:.2f})")
    ax.set_title("Noise Estimate vs Mean Brightness", fontsize=11, fontweight="bold")
    ax.set_xlabel(r"Noise Estimate $\sigma_n$")
    ax.set_ylabel("Mean Luminance (Y)")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)

    # Plot 3: Illumination Non-Uniformity vs Dynamic Range
    ax = axes[2]
    for m, col in method_colors.items():
        subset = df_decisions[df_decisions["selected_method"] == m]
        ax.scatter(subset["illumination_nonuniformity"], subset["dynamic_range"], c=col, label=m, alpha=0.7, edgecolors="k", s=45)
    ax.axvline(stats["illumination_nonuniformity"]["q3"], color="purple", linestyle="--", alpha=0.7, label=f"INU Q3 ({stats['illumination_nonuniformity']['q3']:.2f})")
    ax.set_title("Illumination Non-Uniformity vs Dynamic Range", fontsize=11, fontweight="bold")
    ax.set_xlabel("Illumination Non-Uniformity (INU)")
    ax.set_ylabel("Effective Dynamic Range (P95 - P5)")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"\n[SAVED] Feature-method relationship plot saved to: {plot_path.resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    develop_adaptive_rules()
