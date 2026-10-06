"""
Unit and Integration Tests for Quantitative Evaluation Module
Course: Image and Video Processing (IVP) Mini-Project
"""

import sys
import contextlib
from pathlib import Path
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import (
    calculate_psnr,
    calculate_ssim,
    calculate_output_statistics,
    evaluate_single_output,
    aggregate_results,
    compare_adaptive_against_baselines,
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


def test_psnr_metrics():
    print("--- Test 1: PSNR Calculation & Edge Cases ---")
    img_a = np.full((100, 100, 3), 120, dtype=np.uint8)
    img_identical = np.full((100, 100, 3), 120, dtype=np.uint8)

    # 1. Identical images
    psnr_identical = calculate_psnr(img_a, img_identical)
    assert psnr_identical >= 100.0, f"Expected capped high PSNR for identical images, got {psnr_identical}"

    # 2. Known constant shift of delta = 10
    # Expected PSNR = 10 * log10( 255^2 / 10^2 ) = 20 * log10( 25.5 ) = 28.1308 dB
    img_shifted = np.full((100, 100, 3), 130, dtype=np.uint8)
    psnr_shifted = calculate_psnr(img_a, img_shifted)
    expected_psnr = 20.0 * np.log10(255.0 / 10.0)
    assert np.isclose(psnr_shifted, expected_psnr, atol=0.01), f"Expected ~{expected_psnr:.2f}, got {psnr_shifted}"

    # 3. Shape mismatch error
    diff_shape = np.full((100, 80, 3), 120, dtype=np.uint8)
    with assert_raises(ValueError):
        calculate_psnr(img_a, diff_shape)

    print("[PASS] PSNR calculation verified for identical images, known shifts, and shape mismatches.")


def test_ssim_metrics():
    print("\n--- Test 2: SSIM Calculation & Bounds ---")
    rng = np.random.RandomState(42)
    img = rng.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    img_same = img.copy()

    # 1. Identical image SSIM must be 1.0
    ssim_identical = calculate_ssim(img, img_same)
    assert np.isclose(ssim_identical, 1.0, atol=1e-3), f"Expected SSIM=1.0, got {ssim_identical}"

    # 2. Degraded/noisy image SSIM must be strictly lower
    noisy = np.clip(img.astype(float) + rng.normal(0, 25.0, img.shape), 0, 255).astype(np.uint8)
    ssim_noisy = calculate_ssim(img, noisy)
    assert 0.0 <= ssim_noisy < 1.0, f"Expected 0 <= SSIM < 1, got {ssim_noisy}"
    assert not np.isnan(ssim_noisy)
    assert not np.isinf(ssim_noisy)

    print(f"[PASS] SSIM verified: identical={ssim_identical}, degraded={ssim_noisy:.4f}.")


def test_output_statistics():
    print("\n--- Test 3: Output Statistics Verification ---")
    img = np.full((100, 100, 3), 50, dtype=np.uint8)
    stats = calculate_output_statistics(img)

    expected_keys = {"output_brightness", "output_contrast", "output_noise"}
    assert set(stats.keys()) == expected_keys
    assert np.isclose(stats["output_brightness"], 50.0, atol=0.5)
    assert stats["output_contrast"] == 0.0
    assert stats["output_noise"] == 0.0

    print("[PASS] Output statistics return expected keys and values.")


def test_input_immutability():
    print("\n--- Test 4: Input Immutability Check ---")
    ref = np.random.randint(0, 256, (80, 80, 3), dtype=np.uint8)
    enh = np.random.randint(0, 256, (80, 80, 3), dtype=np.uint8)

    ref_copy = ref.copy()
    enh_copy = enh.copy()

    _ = evaluate_single_output("img_01", "test_method", ref, enh, 12.5)

    assert np.array_equal(ref, ref_copy), "Reference image was mutated during evaluation!"
    assert np.array_equal(enh, enh_copy), "Enhanced image was mutated during evaluation!"
    print("[PASS] Evaluation functions are non-destructive and preserve input arrays.")


def test_aggregation_and_pairwise_logic():
    print("\n--- Test 5: Result Aggregation and Pairwise Comparison Logic ---")
    mock_data = [
        {"image_name": "img1", "method": "Baseline A", "psnr": 20.0, "ssim": 0.80, "output_brightness": 50.0, "output_contrast": 15.0, "output_noise": 1.1, "processing_time_ms": 10.0},
        {"image_name": "img2", "method": "Baseline A", "psnr": 22.0, "ssim": 0.82, "output_brightness": 55.0, "output_contrast": 16.0, "output_noise": 1.2, "processing_time_ms": 12.0},
        {"image_name": "img1", "method": "Proposed Adaptive", "psnr": 21.0, "ssim": 0.85, "output_brightness": 52.0, "output_contrast": 18.0, "output_noise": 1.0, "processing_time_ms": 15.0},
        {"image_name": "img2", "method": "Proposed Adaptive", "psnr": 22.0, "ssim": 0.82, "output_brightness": 55.0, "output_contrast": 16.0, "output_noise": 1.2, "processing_time_ms": 14.0},
    ]
    df_mock = pd.DataFrame(mock_data)

    # Aggregation
    summary = aggregate_results(df_mock)
    assert len(summary) == 2
    assert "psnr_mean" in summary.columns
    assert "ssim_mean" in summary.columns

    # Pairwise comparison
    pairwise = compare_adaptive_against_baselines(df_mock, adaptive_name="Proposed Adaptive")
    assert len(pairwise) == 1
    p_row = pairwise.iloc[0]

    # img1: Adaptive (21.0) > Baseline (20.0) -> Win
    # img2: Adaptive (22.0) == Baseline (22.0) -> Tie
    assert p_row["psnr_wins"] == 1
    assert p_row["psnr_ties"] == 1
    assert p_row["psnr_losses"] == 0
    assert p_row["psnr_win_pct"] == 50.0

    # SSIM img1: 0.85 > 0.80 (Win), img2: 0.82 == 0.82 (Tie)
    assert p_row["ssim_wins"] == 1
    assert p_row["ssim_ties"] == 1
    assert p_row["ssim_losses"] == 0

    print("[PASS] Aggregation and pairwise comparison logic accurately compute summary statistics and win/loss/tie tallies.")


if __name__ == "__main__":
    print("=" * 65)
    print("RUNNING QUANTITATIVE EVALUATION TEST SUITE")
    print("=" * 65)
    test_psnr_metrics()
    test_ssim_metrics()
    test_output_statistics()
    test_input_immutability()
    test_aggregation_and_pairwise_logic()
    print("\n" + "=" * 65)
    print("ALL EVALUATION TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 65)
