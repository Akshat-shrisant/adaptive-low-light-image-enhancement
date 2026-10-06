"""
Quantitative Objective Evaluation Module
Course: Image and Video Processing (IVP) Mini-Project

This module provides standardized, reproducible evaluation functions for assessing
low-light image enhancement quality using full-reference ground truth and no-reference
image statistics.

Supported Metrics:
    1. PSNR (Peak Signal-to-Noise Ratio, dB): Full-reference pixel-level reconstruction fidelity.
    2. SSIM (Structural Similarity Index): Full-reference perceptual structure, luminance, and contrast similarity.
    3. Output Luminance Statistics: Mean and median brightness of enhanced image.
    4. Output Contrast: Global RMS contrast (standard deviation of luminance).
    5. Output Noise Estimate: High-frequency residual standard deviation (Immerkaer, 1996).
    6. Runtime Latency: Algorithm execution duration in milliseconds.

Evaluation Protocol:
    - Images must be uint8 [0, 255] RGB arrays of identical dimensions.
    - PSNR uses data_range=255.
    - SSIM uses data_range=255 with channel_axis=-1 for 3-channel RGB.
    - Robust handling for edge cases (e.g., identical images producing infinite PSNR).
"""

from typing import Dict, Any, List, Optional, Union
import numpy as np
import pandas as pd

try:
    from skimage.metrics import structural_similarity
except Exception:
    from skimage.metrics._structural_similarity import structural_similarity

from src.preprocessing import validate_image, extract_luminance
from src.image_analysis import (
    compute_brightness_statistics,
    compute_global_contrast,
    estimate_noise_level,
)


def calculate_psnr(
    ref_rgb: np.ndarray,
    enhanced_rgb: np.ndarray,
    data_range: float = 255.0
) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR) between reference and enhanced image.
    
    Mathematical Definition:
        MSE = (1 / (3 * H * W)) * sum_{c,y,x} (I_ref(x,y,c) - I_enh(x,y,c))^2
        PSNR = 10 * log10( (data_range^2) / MSE )
        
    Args:
        ref_rgb: Ground-truth reference image (uint8 RGB).
        enhanced_rgb: Enhanced output image (uint8 RGB).
        data_range: Maximum possible intensity dynamic range (default: 255.0).
        
    Returns:
        float: PSNR in decibels (dB). Returns 100.0 if images are mathematically identical (MSE == 0)
               to avoid non-finite values in tabular exports while preserving numerical ranking.
    """
    validate_image(ref_rgb, allow_gray=False, name="PSNR ref_rgb")
    validate_image(enhanced_rgb, allow_gray=False, name="PSNR enhanced_rgb")

    if ref_rgb.shape != enhanced_rgb.shape:
        raise ValueError(
            f"Shape mismatch in PSNR calculation: reference {ref_rgb.shape} "
            f"vs enhanced {enhanced_rgb.shape}"
        )

    # Fast identical check
    if np.array_equal(ref_rgb, enhanced_rgb):
        return 100.0

    mse = np.mean((ref_rgb.astype(np.float64) - enhanced_rgb.astype(np.float64)) ** 2)
    if mse == 0.0:
        return 100.0

    psnr_val = 10.0 * np.log10((float(data_range) ** 2) / mse)
    return round(float(psnr_val), 4)


def calculate_ssim(
    ref_rgb: np.ndarray,
    enhanced_rgb: np.ndarray,
    data_range: float = 255.0
) -> float:
    """
    Calculate Structural Similarity Index (SSIM) between reference and enhanced image.
    
    Mathematical Definition (Wang et al., 2004):
        SSIM(x, y) = [ (2*mu_x*mu_y + C1)(2*sigma_xy + C2) ] / [ (mu_x^2 + mu_y^2 + C1)(sigma_x^2 + sigma_y^2 + C2) ]
        
    Args:
        ref_rgb: Ground-truth reference image (uint8 RGB).
        enhanced_rgb: Enhanced output image (uint8 RGB).
        data_range: Dynamic range of pixel values (default: 255.0).
        
    Returns:
        float: Mean SSIM index in range [-1.0, 1.0].
    """
    validate_image(ref_rgb, allow_gray=False, name="SSIM ref_rgb")
    validate_image(enhanced_rgb, allow_gray=False, name="SSIM enhanced_rgb")

    if ref_rgb.shape != enhanced_rgb.shape:
        raise ValueError(
            f"Shape mismatch in SSIM calculation: reference {ref_rgb.shape} "
            f"vs enhanced {enhanced_rgb.shape}"
        )

    ssim_val = structural_similarity(
        ref_rgb,
        enhanced_rgb,
        data_range=data_range,
        channel_axis=-1
    )
    return round(float(ssim_val), 4)


def calculate_output_statistics(img_rgb: np.ndarray) -> Dict[str, float]:
    """
    Compute no-reference quality and distribution statistics on an enhanced image.
    
    Args:
        img_rgb: Enhanced image (uint8 RGB).
        
    Returns:
        Dict[str, float] with:
            - output_brightness: Mean luminance (Y channel)
            - output_contrast: Global RMS contrast (std dev of luminance)
            - output_noise: High-frequency noise estimate (sigma_n)
    """
    validate_image(img_rgb, allow_gray=False, name="calculate_output_statistics input")
    lum = extract_luminance(img_rgb, method="ycrcb", is_rgb=True)

    b_stats = compute_brightness_statistics(lum)
    g_contrast, _ = compute_global_contrast(lum)
    noise_est = estimate_noise_level(lum)

    return {
        "output_brightness": b_stats["mean_brightness"],
        "output_contrast": g_contrast,
        "output_noise": noise_est,
    }


def evaluate_single_output(
    image_name: str,
    method: str,
    ref_rgb: np.ndarray,
    enhanced_rgb: np.ndarray,
    processing_time_ms: float
) -> Dict[str, Any]:
    """
    Execute complete quantitative evaluation for a single (image, method) run.
    
    Args:
        image_name: Sample filename or ID.
        method: Name of enhancement method.
        ref_rgb: Ground-truth reference image (uint8 RGB).
        enhanced_rgb: Enhanced output image (uint8 RGB).
        processing_time_ms: Enhancement runtime in milliseconds.
        
    Returns:
        Dict[str, Any] containing all primary and secondary evaluation metrics.
    """
    psnr = calculate_psnr(ref_rgb, enhanced_rgb)
    ssim = calculate_ssim(ref_rgb, enhanced_rgb)
    stats = calculate_output_statistics(enhanced_rgb)

    return {
        "image_name": image_name,
        "method": method,
        "psnr": psnr,
        "ssim": ssim,
        "output_brightness": stats["output_brightness"],
        "output_contrast": stats["output_contrast"],
        "output_noise": stats["output_noise"],
        "processing_time_ms": round(float(processing_time_ms), 2),
    }


def aggregate_results(df_results: pd.DataFrame) -> pd.DataFrame:
    """
    Compute aggregate summary statistics (mean, std, median, min, max) for every method.
    
    Args:
        df_results: DataFrame with columns ['image_name', 'method', 'psnr', 'ssim', ...].
        
    Returns:
        pd.DataFrame: Summary table grouped by method.
    """
    metrics = ["psnr", "ssim", "output_brightness", "output_contrast", "output_noise", "processing_time_ms"]
    summary_rows = []

    for method, group in df_results.groupby("method"):
        row = {"method": method, "num_images": len(group)}
        for m in metrics:
            row[f"{m}_mean"] = round(float(group[m].mean()), 3)
            row[f"{m}_std"] = round(float(group[m].std()), 3)
            row[f"{m}_median"] = round(float(group[m].median()), 3)
            row[f"{m}_min"] = round(float(group[m].min()), 3)
            row[f"{m}_max"] = round(float(group[m].max()), 3)
        summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows)
    # Order methods logically: Baselines first, then Adaptive
    preferred_order = [
        "Gamma Correction",
        "Histogram Equalization",
        "CLAHE",
        "Denoise + CLAHE",
        "Single-Scale Retinex (SSR)",
        "Multi-Scale Retinex (MSR)",
        "Proposed Adaptive",
    ]
    df_summary["order"] = df_summary["method"].map(lambda m: preferred_order.index(m) if m in preferred_order else 99)
    df_summary = df_summary.sort_values("order").drop(columns=["order"]).reset_index(drop=True)
    return df_summary


def compare_adaptive_against_baselines(
    df_results: pd.DataFrame,
    adaptive_name: str = "Proposed Adaptive"
) -> pd.DataFrame:
    """
    Perform paired image-by-image comparisons between the Proposed Adaptive method
    and each fixed baseline technique.
    
    Computes:
        - Mean and Median PSNR difference (Adaptive - Baseline)
        - Mean and Median SSIM difference (Adaptive - Baseline)
        - Win / Tie / Loss counts for PSNR and SSIM
        - Percentage of images where Adaptive outperforms the baseline
        
    Args:
        df_results: DataFrame with columns ['image_name', 'method', 'psnr', 'ssim'].
        adaptive_name: Exact method string for adaptive method.
        
    Returns:
        pd.DataFrame: Paired comparative summary table.
    """
    adaptive_df = df_results[df_results["method"] == adaptive_name].set_index("image_name")
    if adaptive_df.empty:
        raise ValueError(f"Adaptive method '{adaptive_name}' not found in results dataframe.")

    other_methods = [m for m in df_results["method"].unique() if m != adaptive_name]
    comparison_records = []

    for base_method in other_methods:
        base_df = df_results[df_results["method"] == base_method].set_index("image_name")

        # Align on common images
        common_imgs = adaptive_df.index.intersection(base_df.index)
        n_imgs = len(common_imgs)
        if n_imgs == 0:
            continue

        a_psnr = adaptive_df.loc[common_imgs, "psnr"]
        b_psnr = base_df.loc[common_imgs, "psnr"]
        diff_psnr = a_psnr - b_psnr

        a_ssim = adaptive_df.loc[common_imgs, "ssim"]
        b_ssim = base_df.loc[common_imgs, "ssim"]
        diff_ssim = a_ssim - b_ssim

        psnr_wins = int((diff_psnr > 0.01).sum())
        psnr_ties = int((diff_psnr.abs() <= 0.01).sum())
        psnr_losses = int((diff_psnr < -0.01).sum())

        ssim_wins = int((diff_ssim > 0.005).sum())
        ssim_ties = int((diff_ssim.abs() <= 0.005).sum())
        ssim_losses = int((diff_ssim < -0.005).sum())

        psnr_win_rate = round(psnr_wins / n_imgs * 100.0, 1)
        ssim_win_rate = round(ssim_wins / n_imgs * 100.0, 1)

        comparison_records.append({
            "comparison": f"Adaptive vs {base_method}",
            "baseline_method": base_method,
            "num_images": n_imgs,
            "mean_psnr_diff": round(float(diff_psnr.mean()), 3),
            "median_psnr_diff": round(float(diff_psnr.median()), 3),
            "psnr_wins": psnr_wins,
            "psnr_ties": psnr_ties,
            "psnr_losses": psnr_losses,
            "psnr_win_pct": psnr_win_rate,
            "mean_ssim_diff": round(float(diff_ssim.mean()), 4),
            "median_ssim_diff": round(float(diff_ssim.median()), 4),
            "ssim_wins": ssim_wins,
            "ssim_ties": ssim_ties,
            "ssim_losses": ssim_losses,
            "ssim_win_pct": ssim_win_rate,
        })

    return pd.DataFrame(comparison_records)
