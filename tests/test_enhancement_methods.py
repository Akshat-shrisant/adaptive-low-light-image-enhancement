"""
Unit and Integration Tests for Baseline Enhancement Algorithms
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
from src.enhancement_methods import (
    gamma_correction,
    histogram_equalization,
    clahe_enhancement,
    denoising_enhancement,
    single_scale_retinex,
    multi_scale_retinex,
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


def test_parameter_validation():
    print("--- Test 1: Parameter Validation Checks ---")
    dummy_rgb = np.full((50, 50, 3), 30, dtype=np.uint8)

    # Invalid Gamma
    with assert_raises(ValueError):
        gamma_correction(dummy_rgb, gamma=0.0)
    with assert_raises(ValueError):
        gamma_correction(dummy_rgb, gamma=-0.5)

    # Invalid CLAHE
    with assert_raises(ValueError):
        clahe_enhancement(dummy_rgb, clip_limit=0.0)
    with assert_raises(ValueError):
        clahe_enhancement(dummy_rgb, tile_grid_size=(0, 8))

    # Invalid Denoising + Enhancement
    with assert_raises(ValueError):
        denoising_enhancement(dummy_rgb, denoise_method="invalid_denoiser")
    with assert_raises(ValueError):
        denoising_enhancement(dummy_rgb, enhancement_method="invalid_enhancer")

    # Invalid Retinex
    with assert_raises(ValueError):
        single_scale_retinex(dummy_rgb, sigma=0.0)
    with assert_raises(ValueError):
        multi_scale_retinex(dummy_rgb, sigmas=())
    with assert_raises(ValueError):
        multi_scale_retinex(dummy_rgb, sigmas=(15.0, -80.0))

    print("[PASS] Parameter validation successfully rejected invalid configurations.")


def test_synthetic_dark_image():
    print("\n--- Test 2: Synthetic Dark Image Enhancement ---")
    # Low-light synthetic test image with varying gradient and local detail
    x = np.linspace(5, 35, 100)
    y = np.linspace(5, 35, 100)
    xx, yy = np.meshgrid(x, y)
    dark_synth = np.zeros((100, 100, 3), dtype=np.uint8)
    for c in range(3):
        dark_synth[:, :, c] = (xx * (0.8 + 0.1 * c)).astype(np.uint8)

    orig_mean = dark_synth.mean()
    assert orig_mean < 40.0

    methods = {
        "Gamma (0.5)": gamma_correction(dark_synth, gamma=0.5),
        "Histogram Equalization": histogram_equalization(dark_synth),
        "CLAHE (2.0)": clahe_enhancement(dark_synth, clip_limit=2.0),
        "Denoise + CLAHE": denoising_enhancement(dark_synth, denoise_method="bilateral", enhancement_method="clahe"),
        "Single-Scale Retinex": single_scale_retinex(dark_synth, sigma=30.0),
        "Multi-Scale Retinex": multi_scale_retinex(dark_synth, sigmas=(15.0, 30.0, 60.0)),
    }

    for name, out in methods.items():
        assert out.shape == dark_synth.shape, f"{name}: Shape mismatch {out.shape}"
        assert out.dtype == np.uint8, f"{name}: dtype mismatch {out.dtype}"
        assert not np.isnan(out).any(), f"{name}: Contains NaN"
        assert not np.isinf(out).any(), f"{name}: Contains Inf"
        # Check that low-light scene was effectively brightened
        assert out.mean() > orig_mean, f"{name}: Failed to increase brightness (in: {orig_mean:.1f}, out: {out.mean():.1f})"
        print(f"  [PASS] {name:<25}: in_mean={orig_mean:.2f} -> out_mean={out.mean():.2f}")


def test_input_immutability():
    print("\n--- Test 3: Input Array Immutability Check ---")
    loader = LOLDatasetLoader()
    pair = loader.get_pairs("our485")[0]
    low_rgb = pair.load_low(color_mode="RGB")
    pristine_copy = low_rgb.copy()

    # Run all methods on low_rgb
    _ = gamma_correction(low_rgb, gamma=0.6)
    _ = histogram_equalization(low_rgb)
    _ = clahe_enhancement(low_rgb, clip_limit=2.0)
    _ = denoising_enhancement(low_rgb)
    _ = single_scale_retinex(low_rgb)
    _ = multi_scale_retinex(low_rgb)

    assert np.array_equal(low_rgb, pristine_copy), "Input array was mutated by an enhancement method!"
    print("[PASS] Input image array remained 100% untouched across all enhancement operations.")


def test_determinism():
    print("\n--- Test 4: Determinism Across Invocations ---")
    loader = LOLDatasetLoader()
    pair = loader.get_pairs("our485")[0]
    low_rgb = pair.load_low(color_mode="RGB")

    for method_name, fn in [
        ("Gamma", lambda: gamma_correction(low_rgb, gamma=0.5)),
        ("HE", lambda: histogram_equalization(low_rgb)),
        ("CLAHE", lambda: clahe_enhancement(low_rgb, clip_limit=2.0)),
        ("Denoise+CLAHE", lambda: denoising_enhancement(low_rgb)),
        ("SSR", lambda: single_scale_retinex(low_rgb, sigma=80.0)),
        ("MSR", lambda: multi_scale_retinex(low_rgb, sigmas=(15.0, 80.0, 250.0))),
    ]:
        run1 = fn()
        run2 = fn()
        assert np.array_equal(run1, run2), f"{method_name} is not deterministic!"
        print(f"  [PASS] {method_name} is perfectly deterministic.")


def test_our485_real_images():
    print("\n--- Test 5: Execution on Real LOL Dataset Images ('our485') ---")
    loader = LOLDatasetLoader()
    pairs = loader.get_pairs("our485")[:3]  # First 3 development samples

    for pair in pairs:
        low_rgb = pair.load_low(color_mode="RGB")
        assert low_rgb.shape == (400, 600, 3)

        out_gamma = gamma_correction(low_rgb, gamma=0.5)
        out_he = histogram_equalization(low_rgb)
        out_clahe = clahe_enhancement(low_rgb, clip_limit=2.0)
        out_denoise = denoising_enhancement(low_rgb, denoise_method="bilateral", enhancement_method="clahe")
        out_ssr = single_scale_retinex(low_rgb, sigma=80.0)
        out_msr = multi_scale_retinex(low_rgb, sigmas=(15.0, 80.0, 250.0))

        for name, out in [
            ("Gamma", out_gamma),
            ("HE", out_he),
            ("CLAHE", out_clahe),
            ("Denoise+CLAHE", out_denoise),
            ("SSR", out_ssr),
            ("MSR", out_msr),
        ]:
            assert out.shape == (400, 600, 3)
            assert out.dtype == np.uint8
            assert 0 <= out.min() and out.max() <= 255

        print(f"  [PASS] Sample '{pair.pair_id}' successfully enhanced by all 6 algorithms (dims: {out_msr.shape}).")


if __name__ == "__main__":
    print("=" * 65)
    print("RUNNING BASELINE ENHANCEMENT METHODS TEST SUITE")
    print("=" * 65)
    test_parameter_validation()
    test_synthetic_dark_image()
    test_input_immutability()
    test_determinism()
    test_our485_real_images()
    print("\n" + "=" * 65)
    print("ALL BASELINE ENHANCEMENT TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 65)
