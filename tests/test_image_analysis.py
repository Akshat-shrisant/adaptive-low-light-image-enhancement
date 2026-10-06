"""
Unit and Integration Tests for Image Characteristic Analysis Module
Course: Image and Video Processing (IVP) Mini-Project
"""

import sys
import contextlib
from pathlib import Path
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset_loader import LOLDatasetLoader
from src.image_analysis import (
    ImageAnalysisResult,
    compute_brightness_statistics,
    compute_global_contrast,
    compute_local_contrast,
    compute_effective_dynamic_range,
    estimate_noise_level,
    compute_illumination_nonuniformity,
    compute_image_entropy,
    analyze_image_characteristics,
)


@contextlib.contextmanager
def assert_raises(exc_type):
    try:
        yield
    except exc_type:
        return
    except Exception as e:
        raise AssertionError(f"Expected {exc_type.__name__}, but caught {type(e).__name__}: {e}")
    raise AssertionError(f"Expected {exc_type.__name__}, but no exception was raised.")


def test_constant_images():
    print("--- Test 1: Synthetic Constant Images ---")
    # Constant dark image (intensity 10)
    dark_const = np.full((100, 100), 10, dtype=np.uint8)
    res_dark = analyze_image_characteristics(dark_const, "dark_const")

    assert res_dark.mean_brightness == 10.0
    assert res_dark.median_brightness == 10.0
    assert res_dark.global_contrast == 0.0
    assert res_dark.local_contrast == 0.0
    assert res_dark.dynamic_range == 0.0
    assert res_dark.noise_estimate == 0.0
    assert res_dark.entropy == 0.0
    assert res_dark.illumination_nonuniformity == 0.0

    # Constant bright image (intensity 220)
    bright_const = np.full((100, 100), 220, dtype=np.uint8)
    res_bright = analyze_image_characteristics(bright_const, "bright_const")
    assert res_bright.mean_brightness == 220.0
    assert res_bright.global_contrast == 0.0
    assert res_bright.noise_estimate == 0.0

    print("[PASS] Constant images produce expected theoretical zero-dispersion metrics.")


def test_contrast_and_dynamic_range():
    print("\n--- Test 2: Contrast & Dynamic Range on Synthetic Patterns ---")
    # High contrast split image: half 0, half 255
    high_cont = np.zeros((100, 100), dtype=np.uint8)
    high_cont[:, 50:] = 255
    res_hc = analyze_image_characteristics(high_cont, "high_contrast")

    assert np.isclose(res_hc.mean_brightness, 127.5, atol=0.1)
    assert np.isclose(res_hc.global_contrast, 127.5, atol=0.1)
    assert res_hc.dynamic_range == 255.0

    # Low contrast image: intensities uniformly between 30 and 40
    low_cont = np.random.RandomState(42).randint(30, 41, (100, 100), dtype=np.uint8)
    res_lc = analyze_image_characteristics(low_cont, "low_contrast")

    assert 30.0 <= res_lc.mean_brightness <= 40.0
    assert res_lc.dynamic_range <= 10.0
    assert res_lc.global_contrast < res_hc.global_contrast

    print("[PASS] Contrast and dynamic range metrics reflect ground-truth synthetic intensity spans.")


def test_noise_estimation():
    print("\n--- Test 3: Noise Level Estimation with Known Additive Noise ---")
    rng = np.random.RandomState(42)
    # Clean smooth gradient
    x = np.linspace(50, 150, 200)
    y = np.linspace(50, 150, 200)
    xx, yy = np.meshgrid(x, y)
    clean_grad = (0.5 * xx + 0.5 * yy).astype(np.uint8)

    noise_clean = estimate_noise_level(clean_grad)
    assert noise_clean < 0.5, f"Smooth gradient should have near-zero noise, got {noise_clean}"

    # Inject Gaussian noise with standard deviation sigma=15.0
    known_sigma = 15.0
    noise_pattern = rng.normal(0, known_sigma, clean_grad.shape)
    noisy_img = np.clip(clean_grad.astype(float) + noise_pattern, 0, 255).astype(np.uint8)

    estimated_sigma = estimate_noise_level(noisy_img)
    # Immerkaer estimator is typically accurate within 10-15% of true additive Gaussian noise
    error_pct = abs(estimated_sigma - known_sigma) / known_sigma
    assert error_pct < 0.15, f"Estimated noise {estimated_sigma} deviated too far from known sigma {known_sigma} ({error_pct:.1%})"

    print(f"[PASS] Noise estimation accurately measured synthetic noise: true={known_sigma}, estimated={estimated_sigma:.2f} (error={error_pct:.1%}).")


def test_illumination_nonuniformity():
    print("\n--- Test 4: Illumination Non-Uniformity Detection ---")
    # 1. Perfectly flat uniform image
    uniform_img = np.full((200, 200), 50, dtype=np.uint8)
    inu_uniform = compute_illumination_nonuniformity(uniform_img, sigma=20.0)
    assert inu_uniform == 0.0

    # 2. Linear ramp lighting (strong gradient from 10 to 180 across scene)
    ramp = np.tile(np.linspace(10, 180, 200, dtype=np.uint8), (200, 1))
    inu_ramp = compute_illumination_nonuniformity(ramp, sigma=20.0)

    assert inu_ramp > 0.3, f"Ramp illumination non-uniformity should be high, got {inu_ramp}"
    print(f"[PASS] Illumination non-uniformity successfully separated uniform ({inu_uniform}) from gradient ({inu_ramp:.4f}).")


def test_actual_lol_image_analysis():
    print("\n--- Test 5: Analysis on Actual LOL Dataset Images ---")
    loader = LOLDatasetLoader()
    # Use development subset 'our485' (never touch eval15 during development)
    pairs = loader.get_pairs("our485")
    sample_pair = pairs[0]

    low_rgb, high_rgb = sample_pair.load_pair(color_mode="RGB")
    copy_low = low_rgb.copy()

    res_low = analyze_image_characteristics(low_rgb, image_name=sample_pair.pair_id, is_rgb=True)
    res_high = analyze_image_characteristics(high_rgb, image_name=sample_pair.pair_id, is_rgb=True)

    # Verify input immutability
    assert np.array_equal(low_rgb, copy_low), "Input image was mutated during analysis!"

    # Verify return types and data integrity
    assert isinstance(res_low, ImageAnalysisResult)
    for k, v in res_low.to_dict().items():
        if k != "image_name":
            assert not np.isnan(v), f"Metric {k} is NaN"
            assert not np.isinf(v), f"Metric {k} is Inf"
            assert isinstance(v, float), f"Metric {k} is not float: {type(v)}"

    # Low light must have significantly lower mean/median brightness than high-light reference
    assert res_low.mean_brightness < res_high.mean_brightness
    assert res_low.median_brightness < res_high.median_brightness
    assert res_low.dynamic_range < res_high.dynamic_range

    print(f"  Sample ID: {sample_pair.pair_id}")
    print(f"  Low-Light  -> Mean: {res_low.mean_brightness:.2f}, Global Contrast: {res_low.global_contrast:.2f}, Noise: {res_low.noise_estimate:.2f}, INU: {res_low.illumination_nonuniformity:.3f}")
    print(f"  Reference  -> Mean: {res_high.mean_brightness:.2f}, Global Contrast: {res_high.global_contrast:.2f}, Noise: {res_high.noise_estimate:.2f}, INU: {res_high.illumination_nonuniformity:.3f}")
    print("[PASS] LOL sample analysis successfully verified with expected comparative relationships.")


def test_determinism():
    print("\n--- Test 6: Determinism Verification ---")
    loader = LOLDatasetLoader()
    pair = loader.get_pairs("our485")[0]
    low_rgb = pair.load_low(color_mode="RGB")

    res1 = analyze_image_characteristics(low_rgb, "test_pair")
    res2 = analyze_image_characteristics(low_rgb, "test_pair")

    assert res1.to_dict() == res2.to_dict(), "Analysis must be 100% deterministic"
    print("[PASS] Analysis pipeline is perfectly deterministic.")


if __name__ == "__main__":
    print("=" * 65)
    print("RUNNING IMAGE ANALYSIS MODULE TEST SUITE")
    print("=" * 65)
    test_constant_images()
    test_contrast_and_dynamic_range()
    test_noise_estimation()
    test_illumination_nonuniformity()
    test_actual_lol_image_analysis()
    test_determinism()
    print("\n" + "=" * 65)
    print("ALL IMAGE ANALYSIS TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 65)
