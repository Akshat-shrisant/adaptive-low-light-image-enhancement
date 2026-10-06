"""
Unit and Integration Tests for Content-Aware Adaptive Enhancement Module
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
from src.adaptive_enhancement import (
    AdaptiveDecisionResult,
    AdaptiveThresholds,
    select_enhancement_strategy,
    adaptive_enhance,
)

VALID_METHODS = {
    "gamma_correction",
    "histogram_equalization",
    "clahe_enhancement",
    "denoising_enhancement",
    "single_scale_retinex",
    "multi_scale_retinex",
}


def test_synthetic_degradation_profiles():
    print("--- Test 1: Synthetic Degradation Condition Response ---")

    # 1. High Noise Image (Clean gradient + high noise sigma=18)
    rng = np.random.RandomState(42)
    base = np.full((100, 100, 3), 15, dtype=np.uint8)
    noisy_img = np.clip(base.astype(float) + rng.normal(0, 18.0, base.shape), 0, 255).astype(np.uint8)
    res_noisy = adaptive_enhance(noisy_img, "synthetic_noisy")
    assert res_noisy.selected_method == "denoising_enhancement", f"Expected denoising_enhancement, got {res_noisy.selected_method}"
    print(f"  [PASS] High-Noise Image correctly assigned to: {res_noisy.selected_method} ({res_noisy.degradation_profile})")

    # 2. Severely Dark Image (Mean ~ 5)
    dark_img = np.full((100, 100, 3), 5, dtype=np.uint8)
    res_dark = adaptive_enhance(dark_img, "synthetic_dark")
    assert res_dark.selected_method == "multi_scale_retinex", f"Expected multi_scale_retinex, got {res_dark.selected_method}"
    print(f"  [PASS] Severe Dark Image correctly assigned to: {res_dark.selected_method} ({res_dark.degradation_profile})")

    # 3. Non-Uniform Illumination Image (Localized spotlight on dark background)
    spot = np.full((200, 200, 3), 8, dtype=np.uint8)
    spot[40:120, 40:120] = 70
    res_spot = adaptive_enhance(spot, "synthetic_spot")
    assert res_spot.selected_method == "multi_scale_retinex", f"Expected multi_scale_retinex, got {res_spot.selected_method}"
    print(f"  [PASS] Non-Uniform Lighting correctly assigned to: {res_spot.selected_method} ({res_spot.degradation_profile})")

    # 4. Clean Mild Underexposure with Preserved Contrast
    x = np.linspace(15, 30, 100)
    xx, yy = np.meshgrid(x, x)
    clean_mild = np.stack([xx.astype(np.uint8), xx.astype(np.uint8), xx.astype(np.uint8)], axis=-1)
    res_mild = adaptive_enhance(clean_mild, "synthetic_mild")
    assert res_mild.selected_method in {"gamma_correction", "clahe_enhancement"}
    print(f"  [PASS] Mild Clean Image correctly assigned to: {res_mild.selected_method} ({res_mild.degradation_profile})")


def test_input_immutability_and_types():
    print("\n--- Test 2: Input Immutability and Output Contract Verification ---")
    loader = LOLDatasetLoader()
    pair = loader.get_pairs("our485")[0]
    low_rgb = pair.load_low(color_mode="RGB")
    pristine_copy = low_rgb.copy()

    result = adaptive_enhance(low_rgb, image_name=pair.pair_id)

    # 1. Input immutability
    assert np.array_equal(low_rgb, pristine_copy), "Input array was mutated by adaptive_enhance!"

    # 2. Output shapes and types
    assert isinstance(result, AdaptiveDecisionResult)
    assert result.enhanced_image.shape == (400, 600, 3)
    assert result.enhanced_image.dtype == np.uint8
    assert 0 <= result.enhanced_image.min() and result.enhanced_image.max() <= 255
    assert not np.isnan(result.enhanced_image).any()
    assert not np.isinf(result.enhanced_image).any()

    # 3. Decision record validation
    assert result.selected_method in VALID_METHODS
    assert isinstance(result.selected_parameters, dict)
    assert len(result.decision_reason) > 20
    assert result.decision_strength in {"Strong Match", "Moderate Match"}

    print("[PASS] Input immutability, array shapes, types, and decision record integrity verified.")


def test_determinism():
    print("\n--- Test 3: Determinism Verification ---")
    loader = LOLDatasetLoader()
    pair = loader.get_pairs("our485")[0]
    low_rgb = pair.load_low(color_mode="RGB")

    res1 = adaptive_enhance(low_rgb, "test_pair")
    res2 = adaptive_enhance(low_rgb, "test_pair")

    assert np.array_equal(res1.enhanced_image, res2.enhanced_image), "Enhanced images differ between repeated runs!"
    assert res1.selected_method == res2.selected_method
    assert res1.degradation_profile == res2.degradation_profile
    assert res1.selected_parameters == res2.selected_parameters
    print("[PASS] Adaptive enhancement pipeline is 100% deterministic.")


def test_our485_real_images():
    print("\n--- Test 4: End-to-End Execution on Diverse 'our485' Samples ---")
    loader = LOLDatasetLoader()
    # Test on 5 development samples
    dev_pairs = loader.get_pairs("our485")[:5]

    for pair in dev_pairs:
        low_rgb = pair.load_low(color_mode="RGB")
        res = adaptive_enhance(low_rgb, image_name=pair.pair_id)

        assert res.enhanced_image.shape == (400, 600, 3)
        assert res.enhanced_image.dtype == np.uint8
        assert res.selected_method in VALID_METHODS
        assert res.enhanced_image.mean() > low_rgb.mean(), f"Enhanced image brightness did not increase for {pair.pair_id}"

        print(f"  [PASS] Sample '{pair.pair_id}': Profile='{res.degradation_profile}' -> Method='{res.selected_method}' "
              f"(in_mean={low_rgb.mean():.1f}, out_mean={res.enhanced_image.mean():.1f})")


if __name__ == "__main__":
    print("=" * 65)
    print("RUNNING ADAPTIVE ENHANCEMENT TEST SUITE")
    print("=" * 65)
    test_synthetic_degradation_profiles()
    test_input_immutability_and_types()
    test_determinism()
    test_our485_real_images()
    print("\n" + "=" * 65)
    print("ALL ADAPTIVE ENHANCEMENT TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 65)
