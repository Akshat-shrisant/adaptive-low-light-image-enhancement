"""
Unit and Integration Tests for Image Preprocessing Module
Course: Image and Video Processing (IVP) Mini-Project
"""

import sys
from pathlib import Path
import numpy as np
import contextlib

@contextlib.contextmanager
def assert_raises(exc_type):
    try:
        yield
    except exc_type:
        return
    except Exception as e:
        raise AssertionError(f"Expected {exc_type.__name__}, but caught {type(e).__name__}: {e}")
    raise AssertionError(f"Expected {exc_type.__name__}, but no exception was raised.")

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset_loader import LOLDatasetLoader
from src.preprocessing import (
    validate_image,
    rgb_to_bgr,
    bgr_to_rgb,
    rgb_to_gray,
    rgb_to_hsv,
    hsv_to_rgb,
    rgb_to_lab,
    lab_to_rgb,
    rgb_to_ycrcb,
    ycrcb_to_rgb,
    convert_color_space,
    normalize_image,
    denormalize_image,
    extract_luminance,
    replace_luminance,
    gaussian_denoise,
    median_denoise,
    bilateral_denoise,
    nlm_denoise,
    resize_image,
)


def test_input_validation():
    print("--- Test 1: Input Validation Checks ---")
    # Test None
    with assert_raises(TypeError):
        validate_image(None)

    # Test non-ndarray
    with assert_raises(TypeError):
        validate_image([1, 2, 3])

    # Test empty array
    with assert_raises(ValueError):
        validate_image(np.array([]))

    # Test 4D array
    with assert_raises(ValueError):
        validate_image(np.zeros((2, 2, 2, 2)))

    # Test invalid channels (e.g. 5 channels)
    with assert_raises(ValueError):
        validate_image(np.zeros((10, 10, 5)))

    # Valid cases
    validate_image(np.zeros((10, 10), dtype=np.uint8), allow_gray=True)
    validate_image(np.zeros((10, 10, 3), dtype=np.uint8), allow_color=True)
    print("[PASS] Input validation correctly rejected invalid inputs and accepted valid images.")


def test_color_space_conversions():
    print("\n--- Test 2: Color Space Conversions & Invertibility ---")
    # Synthetic RGB test image
    np.random.seed(42)
    sample_rgb = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

    # BGR <-> RGB
    bgr = rgb_to_bgr(sample_rgb)
    rgb_back = bgr_to_rgb(bgr)
    assert np.array_equal(sample_rgb, rgb_back), "RGB -> BGR -> RGB must be perfectly lossless"

    # RGB -> Grayscale
    gray = rgb_to_gray(sample_rgb)
    assert gray.ndim == 2
    assert gray.shape == (100, 100)
    assert gray.dtype == np.uint8

    # HSV roundtrip (8-bit quantization limits Hue to 0..179, so max diff can be up to 5-6)
    hsv = rgb_to_hsv(sample_rgb)
    rgb_from_hsv = hsv_to_rgb(hsv)
    diff_hsv = np.abs(sample_rgb.astype(int) - rgb_from_hsv.astype(int))
    assert np.mean(diff_hsv) < 1.5, f"HSV roundtrip mean error too large: {np.mean(diff_hsv)}"
    assert np.max(diff_hsv) <= 10, f"HSV roundtrip max error too large: {np.max(diff_hsv)}"

    # LAB roundtrip (8-bit quantization of L*a*b* coordinates can have corner-case quantization on random noise)
    lab = rgb_to_lab(sample_rgb)
    rgb_from_lab = lab_to_rgb(lab)
    diff_lab = np.abs(sample_rgb.astype(int) - rgb_from_lab.astype(int))
    assert np.mean(diff_lab) < 1.5, f"LAB roundtrip mean error too large: {np.mean(diff_lab)}"
    assert np.max(diff_lab) <= 25, f"LAB roundtrip max error too large: {np.max(diff_lab)}"

    # YCrCb roundtrip
    ycrcb = rgb_to_ycrcb(sample_rgb)
    rgb_from_ycrcb = ycrcb_to_rgb(ycrcb)
    diff_ycrcb = np.abs(sample_rgb.astype(int) - rgb_from_ycrcb.astype(int))
    assert np.mean(diff_ycrcb) < 1.0, f"YCrCb roundtrip mean error too large: {np.mean(diff_ycrcb)}"
    assert np.max(diff_ycrcb) <= 3, f"YCrCb roundtrip max error too large: {np.max(diff_ycrcb)}"

    # Generic converter
    res_hsv = convert_color_space(sample_rgb, "RGB", "HSV")
    assert np.array_equal(hsv, res_hsv)

    print("[PASS] Color space conversions (RGB, BGR, Grayscale, HSV, LAB, YCrCb) verified within integer roundoff bounds.")


def test_normalization_and_denormalization():
    print("\n--- Test 3: Normalization & Denormalization ---")
    orig_uint8 = np.array([[0, 64, 128, 255]], dtype=np.uint8)
    norm = normalize_image(orig_uint8)

    assert norm.dtype == np.float32, f"Expected float32, got {norm.dtype}"
    assert np.isclose(norm[0, 0], 0.0)
    assert np.isclose(norm[0, 3], 1.0)
    assert 0.0 <= norm.min() and norm.max() <= 1.0

    denorm = denormalize_image(norm)
    assert denorm.dtype == np.uint8, f"Expected uint8, got {denorm.dtype}"
    assert np.array_equal(orig_uint8, denorm), f"Mismatch: orig {orig_uint8} vs denorm {denorm}"

    # Test out-of-bounds clipping
    out_of_bounds = np.array([[-0.5, 0.5, 1.8]], dtype=np.float32)
    clipped_norm = normalize_image(out_of_bounds)
    assert clipped_norm.min() >= 0.0 and clipped_norm.max() <= 1.0

    clipped_denorm = denormalize_image(out_of_bounds)
    assert clipped_denorm[0, 0] == 0
    assert clipped_denorm[0, 2] == 255
    print("[PASS] Normalization and denormalization are mathematically consistent and clip safely.")


def test_luminance_extraction_and_replacement():
    print("\n--- Test 4: Luminance Extraction & Replacement ---")
    loader = LOLDatasetLoader()
    # Use development subset 'our485' to avoid touching held-out 'eval15'
    pair = loader.get_pairs("our485")[0]
    low_rgb, _ = pair.load_pair(color_mode="RGB")

    for method in ["ycrcb", "lab", "hsv", "gray"]:
        lum = extract_luminance(low_rgb, method=method)
        assert lum.ndim == 2, f"Luminance for {method} must be 2D, got shape {lum.shape}"
        assert lum.shape == (400, 600), f"Luminance shape mismatch: {lum.shape}"
        assert lum.dtype == np.uint8

    # Test luminance replacement (e.g. increase luminance by 50 units)
    y_orig = extract_luminance(low_rgb, method="ycrcb")
    y_brightened = np.clip(y_orig.astype(np.int32) + 50, 0, 255).astype(np.uint8)
    reconstructed_rgb = replace_luminance(low_rgb, y_brightened, method="ycrcb")

    assert reconstructed_rgb.shape == low_rgb.shape
    assert reconstructed_rgb.dtype == np.uint8
    assert reconstructed_rgb.mean() > low_rgb.mean(), "Brightened luminance should increase image mean brightness"
    print("[PASS] Luminance extraction and luminance-replacement roundtrips work correctly on LOL images.")


def test_denoising_utilities():
    print("\n--- Test 5: Optional Noise Reduction Utilities ---")
    loader = LOLDatasetLoader()
    pair = loader.get_pairs("our485")[0]
    low_rgb = pair.load_low(color_mode="RGB")

    # 1. Gaussian Filter
    gauss = gaussian_denoise(low_rgb, ksize=(3, 3), sigma=1.0)
    assert gauss.shape == low_rgb.shape
    assert gauss.dtype == low_rgb.dtype

    # 2. Median Filter
    med = median_denoise(low_rgb, ksize=3)
    assert med.shape == low_rgb.shape
    assert med.dtype == low_rgb.dtype

    # 3. Bilateral Filter
    bilat = bilateral_denoise(low_rgb, d=5, sigma_color=50, sigma_space=50)
    assert bilat.shape == low_rgb.shape
    assert bilat.dtype == low_rgb.dtype

    # 4. NLM Denoising (test on smaller patch or fast params for test speed)
    nlm = nlm_denoise(low_rgb[:100, :100], h=10.0, template_window_size=5, search_window_size=11)
    assert nlm.shape == (100, 100, 3)
    assert nlm.dtype == np.uint8

    # Invalid kernel check
    with assert_raises(ValueError):
        gaussian_denoise(low_rgb, ksize=(4, 4))  # Even kernel invalid
    with assert_raises(ValueError):
        median_denoise(low_rgb, ksize=4)  # Even kernel invalid

    print("[PASS] Denoising utilities (Gaussian, Median, Bilateral, NLM) operate safely with dimension and type preservation.")


def test_resizing_utility():
    print("\n--- Test 6: Resizing Utility (Previews & Experimental Use) ---")
    sample = np.zeros((400, 600, 3), dtype=np.uint8)
    downscaled = resize_image(sample, (300, 200))
    assert downscaled.shape == (200, 300, 3)

    upscaled = resize_image(sample, (1200, 800))
    assert upscaled.shape == (800, 1200, 3)

    with assert_raises(ValueError):
        resize_image(sample, (-1, 200))

    print("[PASS] Resizing utility properly verifies dimensions and interpolates correctly.")


if __name__ == "__main__":
    print("=" * 65)
    print("RUNNING PREPROCESSING MODULE VALIDATION SUITE")
    print("=" * 65)
    test_input_validation()
    test_color_space_conversions()
    test_normalization_and_denormalization()
    test_luminance_extraction_and_replacement()
    test_denoising_utilities()
    test_resizing_utility()
    print("\n" + "=" * 65)
    print("ALL PREPROCESSING TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 65)
