"""
Evaluation Benchmark Script for Held-Out eval15 Dataset
Course: Image and Video Processing (IVP) Mini-Project

This script performs rigorous, reproducible, objective evaluation across the completely
held-out 'eval15' subset of the LOL dataset:
    - Evaluates 6 baseline methods + Proposed Adaptive method across all 15 image pairs (105 runs).
    - Measures PSNR, SSIM, output brightness, output contrast, noise estimate, and runtime.
    - Generates:
        1. results/metrics/eval15_per_image.csv (105 rows: 15 images x 7 methods)
        2. results/metrics/eval15_summary.csv (aggregate mean, std, median, min, max)
        3. results/metrics/eval15_pairwise_comparison.csv (paired differences, win/loss rates)
        4. results/metrics/eval15_adaptive_decisions.csv (adaptive strategy & feature log)
        5. results/plots/eval15_metric_comparison.png (PSNR, SSIM, Runtime comparison)
        6. results/comparisons/eval15_visual_comparison.png (Multi-scene visual panel)

ACADEMIC INTEGRITY RULES:
    - eval15 was NEVER used for parameter tuning, calibration, or threshold development.
    - The ground-truth high-light image is NEVER seen by the adaptive system during decision-making.
    - All algorithms are evaluated with frozen parameters established in Phases 6 and 7.
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
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
from src.adaptive_enhancement import adaptive_enhance
from src.evaluation import (
    evaluate_single_output,
    aggregate_results,
    compare_adaptive_against_baselines,
)


def run_eval15_benchmark() -> None:
    print("=" * 78)
    print("PHASE 8: QUANTITATIVE EVALUATION ON HELD-OUT 'eval15' DATASET")
    print("=" * 78)

    # 1. Directory Setup
    metrics_dir = PROJECT_ROOT / "results" / "metrics"
    plots_dir = PROJECT_ROOT / "results" / "plots"
    comparisons_dir = PROJECT_ROOT / "results" / "comparisons"

    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    comparisons_dir.mkdir(parents=True, exist_ok=True)

    # 2. Dataset Loading & Validation
    loader = LOLDatasetLoader()
    eval_pairs = loader.get_pairs("eval15")
    num_pairs = len(eval_pairs)
    print(f"\n[1/7] Loading held-out benchmark pairs...")
    print(f"      Discovered {num_pairs} verified image pairs in 'lol_dataset/eval15/'.")
    if num_pairs != 15:
        raise ValueError(f"Expected exactly 15 pairs in eval15, but found {num_pairs}!")

    pair_names = [p.pair_id for p in eval_pairs]
    print(f"      Pair IDs: {', '.join(pair_names)}")

    # 3. Define Standard Evaluation Pipelines (Frozen Parameters)
    baselines = {
        "Gamma Correction": lambda img: gamma_correction(img, gamma=0.5),
        "Histogram Equalization": lambda img: histogram_equalization(img),
        "CLAHE": lambda img: clahe_enhancement(img, clip_limit=2.0, tile_grid_size=(8, 8)),
        "Denoise + CLAHE": lambda img: denoising_enhancement(
            img,
            denoise_method="bilateral",
            denoise_kwargs={"d": 5, "sigma_color": 50.0, "sigma_space": 50.0},
            enhancement_method="clahe",
            enhancement_kwargs={"clip_limit": 2.0, "tile_grid_size": (8, 8)},
        ),
        "Single-Scale Retinex (SSR)": lambda img: single_scale_retinex(img, sigma=80.0),
        "Multi-Scale Retinex (MSR)": lambda img: multi_scale_retinex(img, sigmas=(15.0, 80.0, 250.0)),
    }

    # 4. Execute Benchmark Evaluation (105 Runs)
    print(f"\n[2/7] Executing quantitative benchmark across 7 methods x 15 images (105 runs)...")

    per_image_records: List[Dict[str, Any]] = []
    adaptive_decision_records: List[Dict[str, Any]] = []
    cached_visual_samples: Dict[str, Dict[str, Any]] = {}

    # Representative sample IDs to cache for multi-scene visual comparison panel
    visual_sample_ids = ["1.png", "111.png", "780.png"]

    for idx, pair in enumerate(eval_pairs, 1):
        pid = pair.pair_id
        low_rgb = pair.load_low(color_mode="RGB")
        high_rgb = pair.load_high(color_mode="RGB")

        if pid in visual_sample_ids:
            cached_visual_samples[pid] = {
                "Low-Light Input": low_rgb,
                "Ground Truth Reference": high_rgb,
            }

        # --- A. Evaluate 6 Baselines ---
        for method_name, method_fn in baselines.items():
            # Warmup run to avoid initialization jitter
            _ = method_fn(low_rgb)

            # Precise execution timing (enhancement latency only)
            t_start = time.perf_counter()
            enhanced_b = method_fn(low_rgb)
            t_end = time.perf_counter()
            exec_time_ms = (t_end - t_start) * 1000.0

            rec = evaluate_single_output(
                image_name=pid,
                method=method_name,
                ref_rgb=high_rgb,
                enhanced_rgb=enhanced_b,
                processing_time_ms=exec_time_ms
            )
            per_image_records.append(rec)

            if pid in visual_sample_ids:
                cached_visual_samples[pid][method_name] = enhanced_b

        # --- B. Evaluate Proposed Adaptive Method ---
        # Note: Input is strictly low_rgb only; ground truth is never seen during decision
        _ = adaptive_enhance(low_rgb, image_name=pid)  # warmup

        t_start = time.perf_counter()
        adaptive_res = adaptive_enhance(low_rgb, image_name=pid)
        t_end = time.perf_counter()
        adaptive_exec_ms = (t_end - t_start) * 1000.0

        enhanced_a = adaptive_res.enhanced_image
        rec_a = evaluate_single_output(
            image_name=pid,
            method="Proposed Adaptive",
            ref_rgb=high_rgb,
            enhanced_rgb=enhanced_a,
            processing_time_ms=adaptive_exec_ms
        )
        per_image_records.append(rec_a)

        if pid in visual_sample_ids:
            cached_visual_samples[pid]["Proposed Adaptive"] = enhanced_a
            cached_visual_samples[pid]["adaptive_decision"] = adaptive_res.selected_method
            cached_visual_samples[pid]["degradation_profile"] = adaptive_res.degradation_profile

        # Record adaptive decision details
        chars = adaptive_res.detected_characteristics
        adaptive_decision_records.append({
            "image_name": pid,
            "selected_strategy": adaptive_res.selected_method,
            "degradation_profile": adaptive_res.degradation_profile,
            "decision_strength": adaptive_res.decision_strength,
            "brightness_mean": chars.get("mean_brightness", np.nan),
            "contrast_rms": chars.get("global_contrast", np.nan),
            "dynamic_range": chars.get("dynamic_range", np.nan),
            "noise_sigma": chars.get("noise_estimate", np.nan),
            "illumination_nonuniformity": chars.get("illumination_nonuniformity", np.nan),
            "decision_reason": adaptive_res.decision_reason,
        })

        print(f"      Processed pair [{idx:02d}/15] '{pid}' | Adaptive Choice: {adaptive_res.selected_method} | "
              f"Adaptive PSNR: {rec_a['psnr']:.2f} dB, SSIM: {rec_a['ssim']:.4f}")

    # 5. Export Tabular Results
    print(f"\n[3/7] Formatting and exporting metric CSV files...")
    df_per_image = pd.DataFrame(per_image_records)
    csv_per_image_path = metrics_dir / "eval15_per_image.csv"
    df_per_image.to_csv(csv_per_image_path, index=False)
    print(f"      -> Saved per-image metrics to '{csv_per_image_path.relative_to(PROJECT_ROOT)}' ({len(df_per_image)} rows)")

    # Validate dataframe integrity
    if df_per_image.isnull().values.any():
        raise ValueError("Critical Error: Null or NaN values detected in eval15_per_image.csv!")
    if len(df_per_image) != 105:
        raise ValueError(f"Expected exactly 105 rows, found {len(df_per_image)}!")

    # Summary table
    df_summary = aggregate_results(df_per_image)
    csv_summary_path = metrics_dir / "eval15_summary.csv"
    df_summary.to_csv(csv_summary_path, index=False)
    print(f"      -> Saved aggregate summary to '{csv_summary_path.relative_to(PROJECT_ROOT)}'")

    # Pairwise comparison table
    df_pairwise = compare_adaptive_against_baselines(df_per_image, adaptive_name="Proposed Adaptive")
    csv_pairwise_path = metrics_dir / "eval15_pairwise_comparison.csv"
    df_pairwise.to_csv(csv_pairwise_path, index=False)
    print(f"      -> Saved pairwise comparison to '{csv_pairwise_path.relative_to(PROJECT_ROOT)}'")

    # Adaptive decisions table
    df_decisions = pd.DataFrame(adaptive_decision_records)
    csv_decisions_path = metrics_dir / "eval15_adaptive_decisions.csv"
    df_decisions.to_csv(csv_decisions_path, index=False)
    print(f"      -> Saved adaptive decision log to '{csv_decisions_path.relative_to(PROJECT_ROOT)}'")

    # 6. Generate Metric Comparison Plots
    print(f"\n[4/7] Generating quantitative metric comparison plots...")
    plot_path = plots_dir / "eval15_metric_comparison.png"
    _generate_metric_plots(df_summary, plot_path)
    print(f"      -> Saved metric plots to '{plot_path.relative_to(PROJECT_ROOT)}'")

    # 7. Generate Multi-Scene Visual Comparison Panel
    print(f"\n[5/7] Generating multi-scene visual comparison figure...")
    visual_comp_path = comparisons_dir / "eval15_visual_comparison.png"
    _generate_visual_comparison(cached_visual_samples, visual_sample_ids, visual_comp_path)
    print(f"      -> Saved visual comparison to '{visual_comp_path.relative_to(PROJECT_ROOT)}'")

    # 8. Print Executive Summary in Terminal
    print("\n" + "=" * 78)
    print("PHASE 8 BENCHMARK RESULTS SUMMARY (eval15 held-out test set, n = 15)")
    print("=" * 78)

    print("\n--- Aggregate Method Performance (Mean +/- Std, Median [Min, Max]) ---")
    header_fmt = "{:<28} | {:<16} | {:<18} | {:<15}"
    print(header_fmt.format("Method", "PSNR (dB)", "SSIM", "Latency (ms)"))
    print("-" * 84)
    for _, r in df_summary.iterrows():
        psnr_str = f"{r['psnr_mean']:.2f} +/- {r['psnr_std']:.2f}"
        ssim_str = f"{r['ssim_mean']:.4f} +/- {r['ssim_std']:.4f}"
        time_str = f"{r['processing_time_ms_mean']:.1f} ms"
        print(header_fmt.format(r['method'], psnr_str, ssim_str, time_str))

    print("\n--- Paired Comparison: Proposed Adaptive vs. Baselines (n = 15 pairs) ---")
    pair_fmt = "{:<32} | {:<14} | {:<14} | {:<16} | {:<12}"
    print(pair_fmt.format("Comparison", "Mean dPSNR(dB)", "Mean dSSIM", "PSNR Wins/Ties/Loss", "SSIM Win Pct"))
    print("-" * 96)
    for _, r in df_pairwise.iterrows():
        comp_title = r['comparison']
        d_psnr = f"{r['mean_psnr_diff']:+.3f}"
        d_ssim = f"{r['mean_ssim_diff']:+.4f}"
        wtl = f"{r['psnr_wins']}/{r['psnr_ties']}/{r['psnr_losses']} ({r['psnr_win_pct']}%)"
        ssim_pct = f"{r['ssim_win_pct']:.1f}%"
        print(pair_fmt.format(comp_title, d_psnr, d_ssim, wtl, ssim_pct))

    print("\n--- Adaptive Decision Distribution Across eval15 ---")
    strat_counts = df_decisions["selected_strategy"].value_counts()
    for strat, count in strat_counts.items():
        pct = (count / 15.0) * 100.0
        print(f"   * {strat:<24}: {count:2d} / 15 images ({pct:.1f}%)")

    print("\n" + "=" * 78)
    print("PHASE 8 BENCHMARK COMPLETED SUCCESSFULLY!")
    print("=" * 78)


def _load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    """Safely load Windows system font or fallback to default."""
    font_names = ["arialbd.ttf" if bold else "arial.ttf", "segoeuib.ttf" if bold else "segoeui.ttf"]
    for fn in font_names:
        try:
            return ImageFont.truetype(fn, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _generate_metric_plots(df_summary: pd.DataFrame, save_path: Path) -> None:
    """
    Generate a 3-panel academic benchmark plot using high-resolution Pillow rendering:
        1. Reconstruction Fidelity (PSNR in dB) +/- Std Dev
        2. Structural Similarity (SSIM index [0, 1]) +/- Std Dev
        3. Computational Latency (Mean execution time in ms)
    """
    width = 1800
    height = 650
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    title_font = _load_font(24, bold=True)
    panel_title_font = _load_font(18, bold=True)
    axis_font = _load_font(13, bold=False)
    tick_font = _load_font(12, bold=False)
    label_font = _load_font(12, bold=True)

    # Master Title
    master_title = "Quantitative Evaluation on Held-Out LOL 'eval15' Benchmark (n = 15)"
    draw.text((width // 2, 28), master_title, fill=(20, 20, 20), font=title_font, anchor="mm")

    # Methods and colors
    short_labels = ["Gamma", "HE", "CLAHE", "Denoise+CL", "SSR", "MSR", "Adaptive (Ours)"]
    bar_colors = [
        (74, 123, 183),   # Gamma
        (89, 154, 211),   # HE
        (114, 183, 178),  # CLAHE
        (85, 168, 104),   # Denoise+CLAHE
        (236, 161, 84),   # SSR
        (225, 87, 89),    # MSR
        (148, 103, 189),  # Proposed Adaptive (Highlight Purple)
    ]

    panels = [
        {
            "title": "Reconstruction Fidelity (PSNR)",
            "ylabel": "PSNR (dB)",
            "mean_col": "psnr_mean",
            "std_col": "psnr_std",
            "y_max": 25.0,
            "y_ticks": [0, 5, 10, 15, 20, 25],
            "fmt": "{:.2f}",
        },
        {
            "title": "Structural Similarity (SSIM)",
            "ylabel": "SSIM Index [0, 1]",
            "mean_col": "ssim_mean",
            "std_col": "ssim_std",
            "y_max": 1.0,
            "y_ticks": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            "fmt": "{:.3f}",
        },
        {
            "title": "Computational Latency",
            "ylabel": "Mean Execution Time (ms)",
            "mean_col": "processing_time_ms_mean",
            "std_col": "processing_time_ms_std",
            "y_max": float(np.ceil(df_summary["processing_time_ms_mean"].max() / 50.0) * 50.0 + 25.0),
            "y_ticks": None,  # Computed dynamically
            "fmt": "{:.1f}",
        },
    ]

    panel_w = 520
    panel_h = 420
    top_margin = 90
    spacing = 50

    for p_idx, p in enumerate(panels):
        p_left = 60 + p_idx * (panel_w + spacing)
        p_top = top_margin
        chart_left = p_left + 65
        chart_right = p_left + panel_w - 20
        chart_bottom = p_top + panel_h - 60
        chart_top = p_top + 45
        c_w = chart_right - chart_left
        c_h = chart_bottom - chart_top

        # Panel Title
        draw.text(((chart_left + chart_right) // 2, p_top + 16), p["title"], fill=(30, 30, 30), font=panel_title_font, anchor="mm")

        # Y Axis label
        draw.text((p_left + 15, (chart_top + chart_bottom) // 2), p["ylabel"], fill=(60, 60, 60), font=axis_font, anchor="mm")

        # Y-ticks and gridlines
        y_max = p["y_max"]
        if p["y_ticks"] is None:
            # Dynamic ticks for latency
            step = 50 if y_max >= 150 else 20
            p["y_ticks"] = [i * step for i in range(int(y_max // step) + 1)]

        for y_val in p["y_ticks"]:
            norm_y = y_val / y_max
            y_pixel = int(chart_bottom - norm_y * c_h)
            # Gridline
            draw.line([(chart_left, y_pixel), (chart_right, y_pixel)], fill=(230, 230, 230), width=1)
            # Tick label
            tick_text = f"{y_val:.1f}" if isinstance(y_val, float) else f"{int(y_val)}"
            draw.text((chart_left - 8, y_pixel), tick_text, fill=(100, 100, 100), font=tick_font, anchor="rm")

        # Chart Axes
        draw.line([(chart_left, chart_bottom), (chart_right, chart_bottom)], fill=(80, 80, 80), width=2)
        draw.line([(chart_left, chart_top), (chart_left, chart_bottom)], fill=(80, 80, 80), width=2)

        # Plot bars
        num_methods = len(df_summary)
        bar_width = int(c_w / (num_methods * 1.45))
        gap = int((c_w - (num_methods * bar_width)) / (num_methods + 1))

        means = df_summary[p["mean_col"]].values
        stds = df_summary[p["std_col"]].values if p["std_col"] in df_summary else np.zeros(num_methods)

        for m_idx in range(num_methods):
            m_val = means[m_idx]
            s_val = stds[m_idx]
            bx0 = chart_left + gap + m_idx * (bar_width + gap)
            bx1 = bx0 + bar_width
            by1 = chart_bottom
            bar_h = int((m_val / y_max) * c_h)
            by0 = chart_bottom - bar_h

            color = bar_colors[m_idx % len(bar_colors)]
            draw.rectangle([bx0, by0, bx1, by1], fill=color, outline=(34, 34, 34), width=1)

            # Error bar if standard deviation present
            if s_val > 0.001 and p["std_col"] != "processing_time_ms_std":
                err_h = int((s_val / y_max) * c_h)
                ey_top = max(chart_top, by0 - err_h)
                ey_bot = min(chart_bottom, by0 + err_h)
                cx = (bx0 + bx1) // 2
                draw.line([(cx, ey_top), (cx, ey_bot)], fill=(30, 30, 30), width=1)
                draw.line([(cx - 4, ey_top), (cx + 4, ey_top)], fill=(30, 30, 30), width=1)
                draw.line([(cx - 4, ey_bot), (cx + 4, ey_bot)], fill=(30, 30, 30), width=1)

            # Bar text label
            label_y = by0 - 8 if (s_val == 0 or p["std_col"] == "processing_time_ms_std") else max(chart_top - 2, by0 - int((s_val / y_max) * c_h) - 10)
            txt = p["fmt"].format(m_val)
            draw.text(((bx0 + bx1) // 2, label_y), txt, fill=(20, 20, 20), font=label_font, anchor="mm")

            # X-axis method label
            draw.text(((bx0 + bx1) // 2, chart_bottom + 18), short_labels[m_idx], fill=(40, 40, 40), font=tick_font, anchor="mm")

    # Footnote
    draw.text(
        (width // 2, height - 25),
        "Error bars represent +/- 1 standard deviation across all 15 test pairs. Proposed Adaptive execution includes full feature extraction + decision engine.",
        fill=(110, 110, 110),
        font=tick_font,
        anchor="mm"
    )

    img.save(save_path, "PNG", quality=95)


def _generate_visual_comparison(
    samples_data: Dict[str, Dict[str, Any]],
    sample_ids: List[str],
    save_path: Path
) -> None:
    """
    Generate a clean multi-scene visual grid comparing all 7 methods alongside
    the original low-light input and ground-truth reference using Pillow.
    """
    col_names = [
        "Low-Light Input",
        "Ground Truth Reference",
        "Gamma Correction",
        "Histogram Equalization",
        "CLAHE",
        "Denoise + CLAHE",
        "Single-Scale Retinex (SSR)",
        "Multi-Scale Retinex (MSR)",
        "Proposed Adaptive",
    ]

    short_headers = [
        "Low-Light",
        "Ground Truth",
        "Gamma (0.5)",
        "Hist Eq",
        "CLAHE",
        "Denoise+CL",
        "SSR",
        "MSR",
        "Adaptive",
    ]

    thumb_w = 210
    thumb_h = 140
    n_rows = len(sample_ids)
    n_cols = len(col_names)

    left_margin = 170
    top_margin = 85
    gap_x = 8
    gap_y = 45

    total_w = left_margin + n_cols * thumb_w + (n_cols - 1) * gap_x + 30
    total_h = top_margin + n_rows * thumb_h + (n_rows - 1) * gap_y + 40

    canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    title_font = _load_font(22, bold=True)
    header_font = _load_font(13, bold=True)
    row_font = _load_font(12, bold=True)
    sub_font = _load_font(11, bold=False)

    # Master Title
    draw.text(
        (total_w // 2, 28),
        "Qualitative Visual Comparison Across Representative Held-Out eval15 Scenes",
        fill=(20, 20, 20),
        font=title_font,
        anchor="mm"
    )

    # Column Headers
    for c_idx, header in enumerate(short_headers):
        x = left_margin + c_idx * (thumb_w + gap_x) + (thumb_w // 2)
        # Highlight adaptive column
        color = (148, 103, 189) if "Adaptive" in header else (40, 40, 40)
        draw.text((x, 62), header, fill=color, font=header_font, anchor="mm")

    # Draw rows
    for r_idx, pid in enumerate(sample_ids):
        data = samples_data[pid]
        strat = data.get("adaptive_decision", "N/A")
        profile = data.get("degradation_profile", "")

        y_pos = top_margin + r_idx * (thumb_h + gap_y)

        # Row Label
        draw.text((25, y_pos + 30), f"Sample: {pid}", fill=(20, 20, 20), font=row_font)
        draw.text((25, y_pos + 52), f"Decision:", fill=(90, 90, 90), font=sub_font)
        draw.text((25, y_pos + 70), f"{strat}", fill=(148, 50, 160), font=row_font)
        if len(profile) > 20:
            p_short = profile[:18] + "..."
        else:
            p_short = profile
        draw.text((25, y_pos + 92), p_short, fill=(110, 110, 110), font=sub_font)

        # Images
        for c_idx, c_name in enumerate(col_names):
            x_pos = left_margin + c_idx * (thumb_w + gap_x)
            arr = data.get(c_name)
            if arr is not None:
                # Resize array to thumbnail
                pil_thumb = Image.fromarray(arr).resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
                canvas.paste(pil_thumb, (x_pos, y_pos))
                # Border
                border_color = (148, 103, 189) if c_name == "Proposed Adaptive" else (180, 180, 180)
                b_width = 2 if c_name == "Proposed Adaptive" else 1
                draw.rectangle([x_pos, y_pos, x_pos + thumb_w - 1, y_pos + thumb_h - 1], outline=border_color, width=b_width)

    canvas.save(save_path, "PNG", quality=95)


if __name__ == "__main__":
    run_eval15_benchmark()
