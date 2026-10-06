"""
Image Preprocessing Module for Low-Light Image Enhancement
Course: Image and Video Processing (IVP) Mini-Project

This module provides foundational, modular, and non-destructive image preprocessing
utilities for LOL dataset images, including:
    - Input validation and bounds checking
    - Color space conversions (RGB, BGR, Grayscale, HSV, LAB, YCrCb)
    - Precision intensity normalization and denormalization ([0, 255] <-> [0.0, 1.0])
    - Luminance/Intensity extraction and reconstruction
    - Optional noise reduction filters (Gaussian, Median, Bilateral, Non-Local Means)
    - Aspect-preserving and explicit resizing utilities (for previews/UI)

Design Principles:
    - Non-destructive: All functions return copies/new arrays; inputs are never mutated.
    - Explicit color representation: Functions clearly differentiate between RGB and BGR.
    - Decoupled pipeline: Denoising and resizing are strictly optional and modular.
    - Strict validation: Clear exceptions raised on invalid types, shapes, or ranges.
"""

from typing import Optional, Tuple, Union
import cv2
import numpy as np


# ==============================================================================
# 1. INPUT VALIDATION UTILITIES
# ==============================================================================

def validate_image(
    img: np.ndarray,
    allow_gray: bool = True,
    allow_color: bool = True,
    name: str = "Input image"
) -> None:
    """
    Validate that an input object is a valid 2D or 3D image NumPy array.
    
    Args:
        img: Array to validate.
        allow_gray: Whether single-channel 2D arrays are permitted.
        allow_color: Whether 3-channel 3D arrays are permitted.
        name: Identifier name for descriptive error reporting.
        
    Raises:
        TypeError: If img is None or not a numpy.ndarray.
        ValueError: If array dimensions, channels, or size are invalid.
    """
    if img is None:
        raise TypeError(f"{name} cannot be None.")
    if not isinstance(img, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray, got {type(img).__name__}.")
    if img.size == 0:
        raise ValueError(f"{name} is empty (size is 0).")
    
    ndim = img.ndim
    if ndim == 2:
        if not allow_gray:
            raise ValueError(f"{name} is single-channel grayscale (2D), but only color images are allowed.")
    elif ndim == 3:
        if not allow_color:
            raise ValueError(f"{name} is multi-channel (3D), but only grayscale images are allowed.")
        channels = img.shape[2]
        if channels not in (1, 3):
            raise ValueError(f"{name} must have 1 or 3 channels, found {channels}.")
    else:
        raise ValueError(f"{name} must be a 2D or 3D array, but got {ndim} dimensions (shape: {img.shape}).")


# ==============================================================================
# 2. COLOR SPACE CONVERSIONS
# ==============================================================================
# NOTE: OpenCV loads images from disk as BGR by default. Our dataset loader
# returns standard RGB images by default. The functions below operate on RGB
# unless explicitly stated otherwise.

def rgb_to_bgr(img: np.ndarray) -> np.ndarray:
    """Convert RGB image to OpenCV BGR format."""
    validate_image(img, allow_gray=False, name="rgb_to_bgr input")
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    """Convert OpenCV BGR image to standard RGB format."""
    validate_image(img, allow_gray=False, name="bgr_to_rgb input")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def rgb_to_gray(img: np.ndarray) -> np.ndarray:
    """
    Convert RGB image to single-channel Grayscale using ITU-R BT.601 standard
    weights (Y = 0.299*R + 0.587*G + 0.114*B).
    """
    validate_image(img, allow_gray=False, name="rgb_to_gray input")
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def rgb_to_hsv(img: np.ndarray) -> np.ndarray:
    """Convert RGB image to HSV (Hue [0, 179], Saturation [0, 255], Value [0, 255])."""
    validate_image(img, allow_gray=False, name="rgb_to_hsv input")
    return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)


def hsv_to_rgb(img: np.ndarray) -> np.ndarray:
    """Convert HSV image back to standard RGB format."""
    validate_image(img, allow_gray=False, name="hsv_to_rgb input")
    return cv2.cvtColor(img, cv2.COLOR_HSV2RGB)


def rgb_to_lab(img: np.ndarray) -> np.ndarray:
    """
    Convert RGB image to CIE LAB color space.
    L represents lightness [0, 255], A and B represent color-opponent dimensions.
    """
    validate_image(img, allow_gray=False, name="rgb_to_lab input")
    return cv2.cvtColor(img, cv2.COLOR_RGB2LAB)


def lab_to_rgb(img: np.ndarray) -> np.ndarray:
    """Convert CIE LAB image back to standard RGB format."""
    validate_image(img, allow_gray=False, name="lab_to_rgb input")
    return cv2.cvtColor(img, cv2.COLOR_LAB2RGB)


def rgb_to_ycrcb(img: np.ndarray) -> np.ndarray:
    """
    Convert RGB image to YCrCb color space.
    Y represents luminance [0, 255], Cr and Cb represent chrominance components.
    """
    validate_image(img, allow_gray=False, name="rgb_to_ycrcb input")
    return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)


def ycrcb_to_rgb(img: np.ndarray) -> np.ndarray:
    """Convert YCrCb image back to standard RGB format."""
    validate_image(img, allow_gray=False, name="ycrcb_to_rgb input")
    return cv2.cvtColor(img, cv2.COLOR_YCrCb2RGB)


def convert_color_space(img: np.ndarray, source: str, target: str) -> np.ndarray:
    """
    Generic high-level color space converter supporting RGB, BGR, GRAY, HSV, LAB, YCRCB.
    
    Args:
        img: Input image array.
        source: Source color format ('RGB', 'BGR', 'HSV', 'LAB', 'YCRCB').
        target: Target color format ('RGB', 'BGR', 'GRAY', 'HSV', 'LAB', 'YCRCB').
        
    Returns:
        np.ndarray: Converted image.
    """
    src = source.upper()
    dst = target.upper()
    if src == dst:
        return img.copy()

    # Normalize to RGB first if source is not RGB
    if src == "RGB":
        rgb_img = img
    elif src == "BGR":
        rgb_img = bgr_to_rgb(img)
    elif src == "HSV":
        rgb_img = hsv_to_rgb(img)
    elif src == "LAB":
        rgb_img = lab_to_rgb(img)
    elif src == "YCRCB":
        rgb_img = ycrcb_to_rgb(img)
    else:
        raise ValueError(f"Unsupported source color space: '{source}'")

    # Convert from RGB to target
    if dst == "RGB":
        return rgb_img.copy()
    elif dst == "BGR":
        return rgb_to_bgr(rgb_img)
    elif dst == "GRAY":
        return rgb_to_gray(rgb_img)
    elif dst == "HSV":
        return rgb_to_hsv(rgb_img)
    elif dst == "LAB":
        return rgb_to_lab(rgb_img)
    elif dst == "YCRCB":
        return rgb_to_ycrcb(rgb_img)
    else:
        raise ValueError(f"Unsupported target color space: '{target}'")


# ==============================================================================
# 3. NORMALIZATION & INTENSITY SCALING
# ==============================================================================

def normalize_image(img: np.ndarray) -> np.ndarray:
    """
    Convert an image from integer intensity [0, 255] to floating point [0.0, 1.0].
    
    If the image is already floating point and within [0.0, 1.0], returns a float32 copy.
    Guarantees output dtype np.float32 clipped to [0.0, 1.0].
    
    Args:
        img: Input image array.
        
    Returns:
        np.ndarray: Normalized image as np.float32 in [0.0, 1.0].
    """
    validate_image(img, name="normalize_image input")
    
    if np.issubdtype(img.dtype, np.floating):
        # Already floating point
        if img.max() > 1.0:
            # Float in [0, 255] range
            normalized = (img / 255.0).astype(np.float32)
        else:
            normalized = img.astype(np.float32)
    elif img.dtype == np.uint8:
        normalized = (img.astype(np.float32) / 255.0)
    elif img.dtype == np.uint16:
        normalized = (img.astype(np.float32) / 65535.0)
    else:
        # Generic numeric conversion
        min_val = float(img.min())
        max_val = float(img.max())
        if max_val > min_val:
            normalized = ((img.astype(np.float32) - min_val) / (max_val - min_val))
        else:
            normalized = np.zeros_like(img, dtype=np.float32)

    return np.clip(normalized, 0.0, 1.0)


def denormalize_image(
    img: np.ndarray,
    target_dtype: type = np.uint8
) -> np.ndarray:
    """
    Convert a floating point image in [0.0, 1.0] back to standard integer range [0, 255].
    
    Clips values safely to prevent numerical wrap-around or overflow before casting.
    
    Args:
        img: Floating point array in [0.0, 1.0].
        target_dtype: Desired integer dtype, defaults to np.uint8.
        
    Returns:
        np.ndarray: Denormalized array with values in [0, 255] uint8.
    """
    validate_image(img, name="denormalize_image input")
    
    if not np.issubdtype(img.dtype, np.floating):
        if img.dtype == np.uint8:
            return img.copy()
        img = img.astype(np.float32)

    clipped = np.clip(img, 0.0, 1.0)
    denorm = np.round(clipped * 255.0).astype(target_dtype)
    return denorm


# ==============================================================================
# 4. LUMINANCE / INTENSITY EXTRACTION
# ==============================================================================

def extract_luminance(
    img: np.ndarray,
    method: str = "ycrcb",
    is_rgb: bool = True
) -> np.ndarray:
    """
    Extract a 2D single-channel luminance / intensity map from a 3-channel color image.
    
    This separates the achromatic intensity component from the chromatic information,
    which is essential for low-light enhancement without color hue distortion.
    
    Supported methods:
        - 'ycrcb': Y channel from YCrCb (standard luminance).
        - 'lab': L channel from CIE LAB (perceptual lightness).
        - 'hsv': V channel from HSV (maximum channel value / brightness).
        - 'gray': Grayscale intensity via standard ITU-R BT.601 perceptual weights.
        
    Args:
        img: 3-channel color image array.
        method: Color representation to derive luminance ('ycrcb', 'lab', 'hsv', 'gray').
        is_rgb: True if input is RGB, False if OpenCV BGR.
        
    Returns:
        np.ndarray: 2D array [H, W] of luminance/lightness.
    """
    validate_image(img, allow_gray=False, name="extract_luminance input")
    
    rgb_img = img if is_rgb else bgr_to_rgb(img)
    meth = method.lower()
    
    if meth == "ycrcb":
        ycrcb = rgb_to_ycrcb(rgb_img)
        return ycrcb[:, :, 0]  # Y channel
    elif meth == "lab":
        lab = rgb_to_lab(rgb_img)
        return lab[:, :, 0]    # L channel
    elif meth == "hsv":
        hsv = rgb_to_hsv(rgb_img)
        return hsv[:, :, 2]    # V channel
    elif meth == "gray":
        return rgb_to_gray(rgb_img)
    else:
        raise ValueError(
            f"Unsupported luminance extraction method '{method}'. "
            f"Choose from: 'ycrcb', 'lab', 'hsv', 'gray'."
        )


def replace_luminance(
    original_rgb: np.ndarray,
    enhanced_lum: np.ndarray,
    method: str = "ycrcb"
) -> np.ndarray:
    """
    Recombine an enhanced 2D luminance map with original chromatic channels
    to produce an enhanced 3-channel RGB image without hue shifts.
    
    Args:
        original_rgb: Original 3-channel RGB image (uint8).
        enhanced_lum: Enhanced 2D luminance map [H, W] (uint8 matching original size).
        method: Color space used ('ycrcb', 'lab', 'hsv').
        
    Returns:
        np.ndarray: Enhanced 3-channel RGB image (uint8).
    """
    validate_image(original_rgb, allow_gray=False, name="original_rgb")
    validate_image(enhanced_lum, allow_color=False, name="enhanced_lum")
    
    if original_rgb.shape[:2] != enhanced_lum.shape[:2]:
        raise ValueError(
            f"Spatial dimensions mismatch: original {original_rgb.shape[:2]} "
            f"vs luminance {enhanced_lum.shape[:2]}"
        )

    # Ensure enhanced_lum is uint8
    if enhanced_lum.dtype != np.uint8:
        enhanced_lum = np.clip(enhanced_lum, 0, 255).astype(np.uint8)

    meth = method.lower()
    if meth == "ycrcb":
        ycrcb = rgb_to_ycrcb(original_rgb)
        ycrcb[:, :, 0] = enhanced_lum
        return ycrcb_to_rgb(ycrcb)
    elif meth == "lab":
        lab = rgb_to_lab(original_rgb)
        lab[:, :, 0] = enhanced_lum
        return lab_to_rgb(lab)
    elif meth == "hsv":
        hsv = rgb_to_hsv(original_rgb)
        hsv[:, :, 2] = enhanced_lum
        return hsv_to_rgb(hsv)
    else:
        raise ValueError(f"Unsupported luminance replacement method '{method}'.")


# ==============================================================================
# 4. OPTIONAL NOISE REDUCTION UTILITIES
# ==============================================================================
# IMPORTANT NOTE: Denoising is NOT automatically applied to all images.
# These functions provide modular filters for the adaptive enhancement engine
# to call selectively when noise estimation indicates noise degradation.

def gaussian_denoise(
    img: np.ndarray,
    ksize: Tuple[int, int] = (3, 3),
    sigma: float = 1.0
) -> np.ndarray:
    """
    Apply Gaussian spatial smoothing filter.
    
    Args:
        img: Input image (grayscale or color).
        ksize: Kernel dimensions (must be positive odd integers, e.g. (3, 3) or (5, 5)).
        sigma: Gaussian kernel standard deviation.
        
    Returns:
        np.ndarray: Denoised image array.
    """
    validate_image(img, name="gaussian_denoise input")
    if ksize[0] % 2 == 0 or ksize[1] % 2 == 0 or ksize[0] <= 0 or ksize[1] <= 0:
        raise ValueError(f"Kernel dimensions must be positive odd integers, got {ksize}.")
    return cv2.GaussianBlur(img, ksize, sigmaX=sigma, sigmaY=sigma)


def median_denoise(
    img: np.ndarray,
    ksize: int = 3
) -> np.ndarray:
    """
    Apply Median filter for non-linear impulse / salt-and-pepper noise suppression
    while preserving sharp edge boundaries.
    
    Args:
        img: Input image (grayscale or color uint8).
        ksize: Kernel aperture linear size (must be an odd integer > 1, e.g., 3, 5).
        
    Returns:
        np.ndarray: Filtered image array.
    """
    validate_image(img, name="median_denoise input")
    if ksize % 2 == 0 or ksize <= 1:
        raise ValueError(f"Median ksize must be an odd integer greater than 1, got {ksize}.")
    return cv2.medianBlur(img, ksize)


def bilateral_denoise(
    img: np.ndarray,
    d: int = 5,
    sigma_color: float = 50.0,
    sigma_space: float = 50.0
) -> np.ndarray:
    """
    Apply Bilateral filter for edge-preserving noise smoothing.
    
    Weights pixels based on both spatial distance and radiometric (color) similarity,
    preventing edge blur commonly caused by standard linear filters.
    
    Args:
        img: Input image (uint8).
        d: Diameter of pixel neighborhood (e.g., 5 or 9).
        sigma_color: Filter sigma in the color space.
        sigma_space: Filter sigma in the coordinate space.
        
    Returns:
        np.ndarray: Filtered image array.
    """
    validate_image(img, name="bilateral_denoise input")
    if d <= 0:
        raise ValueError(f"Diameter d must be positive, got {d}.")
    return cv2.bilateralFilter(img, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)


def nlm_denoise(
    img: np.ndarray,
    h: float = 10.0,
    template_window_size: int = 7,
    search_window_size: int = 21
) -> np.ndarray:
    """
    Apply Non-Local Means (NLM) Denoising (Buades et al.).
    
    Exploits redundancy in repetitive image patterns across non-local search windows.
    Effective for low-light noise where photon noise is amplified during brightening.
    
    Args:
        img: Input image (uint8, 2D grayscale or 3D RGB).
        h: Filter strength parameter regulating noise decay. Higher h removes more noise
           but may smooth fine texture.
        template_window_size: Size in pixels of template patch used to compute weights (odd).
        search_window_size: Size in pixels of window used to search patches (odd).
        
    Returns:
        np.ndarray: Denoised image array (uint8).
    """
    validate_image(img, name="nlm_denoise input")
    if img.dtype != np.uint8:
        img = denormalize_image(img)

    if template_window_size % 2 == 0 or template_window_size <= 0:
        raise ValueError(f"template_window_size must be positive odd, got {template_window_size}.")
    if search_window_size % 2 == 0 or search_window_size <= 0:
        raise ValueError(f"search_window_size must be positive odd, got {search_window_size}.")

    if img.ndim == 2:
        return cv2.fastNlMeansDenoising(
            img,
            h=h,
            templateWindowSize=template_window_size,
            searchWindowSize=search_window_size
        )
    else:
        # For RGB color image
        # Note: cv2.fastNlMeansDenoisingColored expects BGR or RGB consistently
        return cv2.fastNlMeansDenoisingColored(
            img,
            h=h,
            hColor=h,
            templateWindowSize=template_window_size,
            searchWindowSize=search_window_size
        )


# ==============================================================================
# 5. RESIZING UTILITY (PREVIEW / EXPERIMENTAL USE ONLY)
# ==============================================================================

def resize_image(
    img: np.ndarray,
    target_size: Tuple[int, int],
    interpolation: Optional[int] = None
) -> np.ndarray:
    """
    Resize image to a specified (width, height) using appropriate interpolation.
    
    IMPORTANT: LOL dataset images are NOT resized during primary evaluation.
    This utility is reserved for demonstration previews and computational profiling.
    
    Args:
        img: Input image array.
        target_size: Desired (width, height) tuple.
        interpolation: OpenCV interpolation flag. Defaults to cv2.INTER_AREA for
                       downsampling and cv2.INTER_LINEAR for upsampling.
                       
    Returns:
        np.ndarray: Resized image array.
    """
    validate_image(img, name="resize_image input")
    w_target, h_target = target_size
    if w_target <= 0 or h_target <= 0:
        raise ValueError(f"Target dimensions must be positive, got {target_size}.")

    h_orig, w_orig = img.shape[:2]
    if interpolation is None:
        if w_target < w_orig or h_target < h_orig:
            interp = cv2.INTER_AREA
        else:
            interp = cv2.INTER_LINEAR
    else:
        interp = interpolation

    return cv2.resize(img, (w_target, h_target), interpolation=interp)
