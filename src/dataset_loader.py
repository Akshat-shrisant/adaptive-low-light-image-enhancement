"""
Dataset Loader Module for the LOL (LOw-Light) Dataset
Course: Image and Video Processing (IVP) Mini-Project

This module provides structured, read-only loading and validation of the paired
LOL dataset, supporting:
    - 'our485': Development & experimental training subset (485 pairs)
    - 'eval15': Benchmark & held-out test subset (15 pairs)

Features:
    - Automatic project-relative dataset root resolution
    - Lazy on-demand image reading using OpenCV
    - BGR to RGB conversion support (default RGB for academic visualization & processing)
    - Pairwise dimension and channel consistency checks
    - Integrity validation of all image files
    - Strict read-only dataset access (never modifies/renames/overwrites original files)
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import cv2
import numpy as np


VALID_SUBSETS = {"our485", "eval15"}
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


@dataclass(frozen=True)
class ImagePairInfo:
    """Metadata describing a single paired low-light / normal-light image sample."""
    pair_id: str
    subset: str
    low_path: Path
    high_path: Path

    def load_low(self, color_mode: str = "RGB") -> np.ndarray:
        """
        Load the low-light input image from disk.
        
        Args:
            color_mode: Desired color format ('RGB', 'BGR', or 'GRAY').
            
        Returns:
            np.ndarray: Loaded image array uint8 [H, W, C] or [H, W].
        """
        return _read_image(self.low_path, color_mode=color_mode)

    def load_high(self, color_mode: str = "RGB") -> np.ndarray:
        """
        Load the reference/ground-truth normal-light image from disk.
        
        Args:
            color_mode: Desired color format ('RGB', 'BGR', or 'GRAY').
            
        Returns:
            np.ndarray: Loaded image array uint8 [H, W, C] or [H, W].
        """
        return _read_image(self.high_path, color_mode=color_mode)

    def load_pair(self, color_mode: str = "RGB") -> Tuple[np.ndarray, np.ndarray]:
        """
        Load both low-light and reference images, verifying identical dimensions.
        
        Args:
            color_mode: Desired color format ('RGB', 'BGR', or 'GRAY').
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (low_image, high_image).
        """
        low = self.load_low(color_mode=color_mode)
        high = self.load_high(color_mode=color_mode)
        if low.shape != high.shape:
            raise ValueError(
                f"Dimension mismatch for pair '{self.pair_id}': "
                f"low shape {low.shape} != high shape {high.shape}"
            )
        return low, high


def _read_image(image_path: Path, color_mode: str = "RGB") -> np.ndarray:
    """
    Safely load an image from disk using OpenCV with color space handling.
    
    Args:
        image_path: Path to the image file.
        color_mode: 'RGB', 'BGR', or 'GRAY'.
        
    Returns:
        np.ndarray: Image array.
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # cv2.imread loads 8-bit BGR by default
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Failed to read image (corrupted or unreadable format): {image_path}")

    mode = color_mode.upper()
    if mode == "RGB":
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif mode == "BGR":
        return img
    elif mode == "GRAY":
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(f"Unsupported color_mode '{color_mode}'. Use 'RGB', 'BGR', or 'GRAY'.")


class LOLDatasetLoader:
    """
    Robust Dataset Loader for the LOL (LOw-Light) Image Dataset.
    
    Automatically handles subset distinction ('our485' development vs 'eval15' test),
    pairing validation, lazy loading, and dataset integrity checks.
    """

    def __init__(self, dataset_root: Optional[Union[str, Path]] = None) -> None:
        """
        Initialize the dataset loader.
        
        Args:
            dataset_root: Path to the 'lol_dataset' directory.
                          If None, resolves relative to project root.
        """
        if dataset_root is None:
            # Locate relative to project root: <project_root>/lol_dataset
            project_root = Path(__file__).resolve().parent.parent
            self.dataset_root = project_root / "lol_dataset"
        else:
            self.dataset_root = Path(dataset_root).resolve()

        if not self.dataset_root.exists():
            raise FileNotFoundError(
                f"Dataset root directory does not exist: {self.dataset_root}"
            )

    def get_subset_path(self, subset: str) -> Path:
        """
        Resolve and validate the path for a given subset ('our485' or 'eval15').
        """
        if subset not in VALID_SUBSETS:
            raise ValueError(
                f"Invalid subset '{subset}'. Must be one of: {VALID_SUBSETS}"
            )
        subset_path = self.dataset_root / subset
        if not subset_path.exists():
            raise FileNotFoundError(
                f"Subset directory does not exist: {subset_path}"
            )
        return subset_path

    def get_pairs(self, subset: str) -> List[ImagePairInfo]:
        """
        Discover and return all validated image pairs in the specified subset.
        
        Args:
            subset: 'our485' or 'eval15'.
            
        Returns:
            List[ImagePairInfo]: Sorted list of image pair descriptors.
        """
        subset_path = self.get_subset_path(subset)
        low_dir = subset_path / "low"
        high_dir = subset_path / "high"

        if not low_dir.exists() or not high_dir.exists():
            raise FileNotFoundError(
                f"Both 'low' and 'high' subfolders must exist in {subset_path}"
            )

        # Collect files matching supported image extensions
        low_files = {
            f.name: f for f in low_dir.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        }
        high_files = {
            f.name: f for f in high_dir.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        }

        # Pair files by common filename
        common_filenames = sorted(
            list(low_files.keys() & high_files.keys()),
            key=lambda x: (len(x), x)  # natural sort for numeric filenames like 1.png, 22.png
        )

        pairs = [
            ImagePairInfo(
                pair_id=name,
                subset=subset,
                low_path=low_files[name],
                high_path=high_files[name],
            )
            for name in common_filenames
        ]
        return pairs

    def get_pair_by_id(self, pair_id: str, subset: str = "our485") -> ImagePairInfo:
        """
        Retrieve a specific ImagePairInfo by filename or ID.
        
        Args:
            pair_id: e.g., '1.png' or '1'
            subset: 'our485' or 'eval15'
            
        Returns:
            ImagePairInfo
        """
        if not pair_id.endswith(".png"):
            pair_id = f"{pair_id}.png"

        pairs = {p.pair_id: p for p in self.get_pairs(subset)}
        if pair_id not in pairs:
            raise KeyError(f"Pair '{pair_id}' not found in subset '{subset}'.")
        return pairs[pair_id]

    def get_subset_summary(self, subset: str, validate_images: bool = False) -> Dict[str, Any]:
        """
        Generate a summary of image counts, pairing status, extensions, and dimensions.
        
        Args:
            subset: 'our485' or 'eval15'.
            validate_images: If True, reads every file with OpenCV to verify readability and dimensions.
            
        Returns:
            Dict[str, Any] with summary metrics.
        """
        subset_path = self.get_subset_path(subset)
        low_dir = subset_path / "low"
        high_dir = subset_path / "high"

        low_files = [f for f in low_dir.iterdir() if f.is_file()] if low_dir.exists() else []
        high_files = [f for f in high_dir.iterdir() if f.is_file()] if high_dir.exists() else []

        low_names = {f.name for f in low_files}
        high_names = {f.name for f in high_files}

        common_names = low_names & high_names
        missing_in_high = sorted(list(low_names - high_names))
        missing_in_low = sorted(list(high_names - low_names))

        exts = sorted(list({f.suffix.lower() for f in (low_files + high_files)}))

        dimensions = set()
        read_errors = []

        if validate_images:
            for name in sorted(list(common_names)):
                p_low = low_dir / name
                p_high = high_dir / name
                img_l = cv2.imread(str(p_low))
                img_h = cv2.imread(str(p_high))
                if img_l is None:
                    read_errors.append(f"Unreadable low-light image: {p_low.name}")
                if img_h is None:
                    read_errors.append(f"Unreadable reference image: {p_high.name}")
                if img_l is not None and img_h is not None:
                    if img_l.shape != img_h.shape:
                        read_errors.append(
                            f"Dimension mismatch for {name}: {img_l.shape} vs {img_h.shape}"
                        )
                    dimensions.add((img_l.shape[1], img_l.shape[0]))  # (width, height)
        else:
            # Sample first 5 to get representative dimension without reading all
            sample_names = sorted(list(common_names))[:5]
            for name in sample_names:
                img_l = cv2.imread(str(low_dir / name))
                if img_l is not None:
                    dimensions.add((img_l.shape[1], img_l.shape[0]))

        return {
            "subset": subset,
            "low_count": len(low_files),
            "high_count": len(high_files),
            "valid_pairs": len(common_names),
            "missing_in_high": missing_in_high,
            "missing_in_low": missing_in_low,
            "image_extensions": exts,
            "dimensions": sorted(list(dimensions)),
            "is_exact_paired": (len(low_files) == len(high_files) == len(common_names)),
            "read_errors": read_errors,
        }

    def validate_all(self, subset: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Perform a thorough integrity scan of all images in a subset.
        
        Returns:
            Tuple[bool, Dict[str, Any]]: (is_valid, summary_report)
        """
        summary = self.get_subset_summary(subset, validate_images=True)
        is_valid = summary["is_exact_paired"] and len(summary["read_errors"]) == 0
        return is_valid, summary

    def print_summary(self) -> None:
        """Print a formatted academic summary of all dataset subsets."""
        print("=" * 65)
        print("LOL DATASET LOADER SUMMARY")
        print(f"Root: {self.dataset_root}")
        print("=" * 65)
        for subset in sorted(list(VALID_SUBSETS)):
            info = self.get_subset_summary(subset, validate_images=False)
            dim_str = ", ".join([f"{w}x{h}" for w, h in info["dimensions"]])
            print(f"Subset: {subset:7s} | Low: {info['low_count']:3d} | High: {info['high_count']:3d} | "
                  f"Pairs: {info['valid_pairs']:3d} | Dims: {dim_str} | Exts: {info['image_extensions']}")
        print("=" * 65)


if __name__ == "__main__":
    # Self-test routine when executed directly
    loader = LOLDatasetLoader()
    loader.print_summary()

    for sub in ["eval15", "our485"]:
        valid, report = loader.validate_all(sub)
        status = "PASSED" if valid else "FAILED"
        print(f"\nFull Validation ({sub}): [{status}]")
        print(f"  Valid Pairs: {report['valid_pairs']}")
        print(f"  Missing Pairs: {len(report['missing_in_high']) + len(report['missing_in_low'])}")
        print(f"  Read Errors: {len(report['read_errors'])}")
        if report['read_errors']:
            print(f"  Errors detail: {report['read_errors'][:3]}")
