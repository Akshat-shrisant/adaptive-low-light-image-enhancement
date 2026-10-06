"""
Preprocessing Visual Verification Script
Course: Image and Video Processing (IVP) Mini-Project

This script loads a sample image from the development subset ('our485'),
applies modular preprocessing operations (luminance extraction, edge-preserving
filtering, color space conversions), and saves a diagnostic visual verification panel.

NOTE: Denoising is inspected here solely for preprocessing filter verification;
it is NOT an enhancement result.
"""

from pathlib import Path
import matplotlib.pyplot as plt

# Ensure project root is accessible
PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset_loader import LOLDatasetLoader
from src.preprocessing import (
    extract_luminance,
    gaussian_denoise,
    bilateral_denoise,
    rgb_to_gray,
)


def run_visual_verification():
    output_dir = PROJECT_ROOT / "samples"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "preprocessing_verification.png"

    # Use development subset 'our485' (do NOT touch eval15)
    loader = LOLDatasetLoader()
    sample_pair = loader.get_pairs("our485")[0]
    low_rgb, high_rgb = sample_pair.load_pair(color_mode="RGB")

    print(f"Loaded development sample: '{sample_pair.pair_id}' (subset: {sample_pair.subset})")
    print(f"Image shape: {low_rgb.shape}, dtype: {low_rgb.dtype}")

    # Extract components
    lum_ycrcb = extract_luminance(low_rgb, method="ycrcb")
    lum_lab = extract_luminance(low_rgb, method="lab")
    gray_bt601 = rgb_to_gray(low_rgb)
    gauss = gaussian_denoise(low_rgb, ksize=(5, 5), sigma=1.2)
    bilat = bilateral_denoise(low_rgb, d=5, sigma_color=50, sigma_space=50)

    # Plot 2x3 diagnostic panel
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    axes[0, 0].imshow(low_rgb)
    axes[0, 0].set_title(f"Original Low-Light (RGB)\n[{sample_pair.pair_id}]", fontsize=11)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(lum_ycrcb, cmap="gray")
    axes[0, 1].set_title("Luminance (Y Channel - YCrCb)", fontsize=11)
    axes[0, 1].axis("off")

    axes[0, 2].imshow(lum_lab, cmap="gray")
    axes[0, 2].set_title("Lightness (L* Channel - CIE LAB)", fontsize=11)
    axes[0, 2].axis("off")

    axes[1, 0].imshow(gray_bt601, cmap="gray")
    axes[1, 0].set_title("Grayscale (ITU-R BT.601)", fontsize=11)
    axes[1, 0].axis("off")

    axes[1, 1].imshow(gauss)
    axes[1, 1].set_title(r"Gaussian Smoothing (5x5, $\sigma$=1.2)", fontsize=11)
    axes[1, 1].axis("off")

    axes[1, 2].imshow(bilat)
    axes[1, 2].set_title("Bilateral Edge-Preserving Filter", fontsize=11)
    axes[1, 2].axis("off")

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[SUCCESS] Preprocessing visual verification saved to: {output_file.resolve()}")


if __name__ == "__main__":
    run_visual_verification()
