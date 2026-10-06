"""
Unit and Integration Tests for LOL Dataset Loader
Course: Image and Video Processing (IVP) Mini-Project
"""

import sys
from pathlib import Path
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset_loader import LOLDatasetLoader, ImagePairInfo


def test_loader_initialization():
    print("--- Test 1: Loader Initialization & Path Discovery ---")
    loader = LOLDatasetLoader()
    assert loader.dataset_root.exists(), f"Dataset root does not exist: {loader.dataset_root}"
    print(f"[PASS] Successfully resolved dataset root: {loader.dataset_root}")


def test_eval15_pairing_and_loading():
    print("\n--- Test 2: 'eval15' Benchmark Subset Verification ---")
    loader = LOLDatasetLoader()
    pairs = loader.get_pairs("eval15")
    
    assert len(pairs) == 15, f"Expected 15 pairs in eval15, found {len(pairs)}"
    print(f"[PASS] Pair count in eval15: {len(pairs)} pairs.")

    # Test loading a few specific pairs
    sample_ids = ["1.png", "22.png", "111.png"]
    for pid in sample_ids:
        pair = loader.get_pair_by_id(pid, subset="eval15")
        assert pair.low_path.exists(), f"Low image missing: {pair.low_path}"
        assert pair.high_path.exists(), f"High image missing: {pair.high_path}"
        
        # Test RGB loading
        low_rgb, high_rgb = pair.load_pair(color_mode="RGB")
        assert isinstance(low_rgb, np.ndarray), "Low image is not a numpy array"
        assert isinstance(high_rgb, np.ndarray), "High image is not a numpy array"
        assert low_rgb.shape == (400, 600, 3), f"Unexpected low shape {low_rgb.shape}"
        assert high_rgb.shape == (400, 600, 3), f"Unexpected high shape {high_rgb.shape}"
        assert low_rgb.dtype == np.uint8, f"Unexpected dtype {low_rgb.dtype}"
        
        # Test BGR loading and RGB/BGR difference
        low_bgr = pair.load_low(color_mode="BGR")
        assert low_bgr.shape == (400, 600, 3)
        # Verify RGB is swapped with BGR for color channels (channel 0 vs 2)
        assert np.array_equal(low_rgb[:, :, 0], low_bgr[:, :, 2]), "RGB Red channel should match BGR Red channel"
        assert np.array_equal(low_rgb[:, :, 2], low_bgr[:, :, 0]), "RGB Blue channel should match BGR Blue channel"

        # Verify low-light is noticeably darker than high-light reference
        mean_low = low_rgb.mean()
        mean_high = high_rgb.mean()
        assert mean_low < mean_high, f"Low mean ({mean_low:.1f}) should be lower than high mean ({mean_high:.1f})"
        
        print(f"  [PASS] Successfully loaded eval15 pair '{pid}' (dims: {low_rgb.shape}, low mean: {mean_low:.2f}, high mean: {mean_high:.2f})")


def test_our485_pairing_and_loading():
    print("\n--- Test 3: 'our485' Development Subset Verification ---")
    loader = LOLDatasetLoader()
    pairs = loader.get_pairs("our485")
    
    assert len(pairs) == 485, f"Expected 485 pairs in our485, found {len(pairs)}"
    print(f"[PASS] Pair count in our485: {len(pairs)} pairs.")

    # Test first 5 and last 5 pairs for readability and dimension check
    test_samples = pairs[:5] + pairs[-5:]
    for pair in test_samples:
        low, high = pair.load_pair(color_mode="RGB")
        assert low.shape == (400, 600, 3), f"Unexpected shape {low.shape} for {pair.pair_id}"
        assert high.shape == (400, 600, 3), f"Unexpected shape {high.shape} for {pair.pair_id}"
        assert low.dtype == np.uint8
        assert high.dtype == np.uint8
        
    print(f"  [PASS] Successfully verified sample pairs from our485 (first 5 and last 5).")


def test_full_dataset_integrity():
    print("\n--- Test 4: Full Dataset Exhaustive Integrity Scan ---")
    loader = LOLDatasetLoader()
    
    # 1. eval15 scan
    valid_eval15, report_eval15 = loader.validate_all("eval15")
    print(f"  eval15 summary: {report_eval15['valid_pairs']} pairs, 0 read errors: {len(report_eval15['read_errors']) == 0}")
    assert valid_eval15, f"eval15 validation failed: {report_eval15}"
    assert report_eval15["valid_pairs"] == 15
    assert report_eval15["is_exact_paired"] is True

    # 2. our485 scan
    valid_our485, report_our485 = loader.validate_all("our485")
    print(f"  our485 summary: {report_our485['valid_pairs']} pairs, 0 read errors: {len(report_our485['read_errors']) == 0}")
    assert valid_our485, f"our485 validation failed: {report_our485}"
    assert report_our485["valid_pairs"] == 485
    assert report_our485["is_exact_paired"] is True

    print("[PASS] Full dataset exhaustive scan passed with 100% integrity.")


if __name__ == "__main__":
    print("=" * 65)
    print("RUNNING DATASET LOADER VALIDATION SUITE")
    print("=" * 65)
    test_loader_initialization()
    test_eval15_pairing_and_loading()
    test_our485_pairing_and_loading()
    test_full_dataset_integrity()
    print("\n" + "=" * 65)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 65)
