"""
Content-Aware Image Characteristic Analysis Module
Course: Image and Video Processing (IVP) Mini-Project

This module computes quantitative, reproducible metrics to characterize the degradation
profile of low-light images. The resulting feature vector enables evidence-based adaptive
enhancement selection.

Analyzed Characteristics:
    1. Mean & Median Luminance: Global brightness level.
    2. Intensity Percentiles: Robust distribution profiling (P5, P10, P50, P90, P95).
    3. Global Contrast: Standard deviation of luminance (RMS contrast).
    4. Local Contrast: Average patch-wise standard deviation (16x16 blocks).
    5. Percentile Dynamic Range: Effective dynamic range (P95 - P5) immune to outlier spikes.
    6. Noise Estimation: High-frequency Laplacian-based noise variance estimation (Immerkaer, 1996).
    7. Illumination Non-Uniformity: Coefficient of variation of low-frequency illumination profile.
    8. Shannon Entropy: Information distribution across intensity levels.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple, Any, List
import cv2
import numpy as np

from src.preprocessing import validate_image, extract_luminance


@dataclass(frozen=True)
class ImageAnalysisResult:
    """Structured container for all computed image characteristics."""
    image_name: str
    mean_brightness: float
    median_brightness: float
    min_brightness: float
    max_brightness: float
    p5: float
    p10: float
    p50: float
    p90: float
    p95: float
    global_contrast: float
    normalized_contrast: float
    local_contrast: float
    dynamic_range: float
    noise_estimate: float
    illumination_nonuniformity: float
    entropy: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert analysis result to a standard dictionary."""
        return asdict(self)


# ==============================================================================
# 1. BRIGHTNESS AND INTENSITY DISTRIBUTION
# ==============================================================================

def compute_brightness_statistics(lum: np.ndarray) -> Dict[str, float]:
    """
    Compute global brightness central tendencies and percentile distribution.
    
    Args:
        lum: 2D uint8 or float luminance array.
        
    Returns:
        Dict[str, float] with mean, median, min, max, and percentiles (P5, P10, P50, P90, P95).
    """
    validate_image(lum, allow_color=False, name="compute_brightness_statistics input")
    lum_f = lum.astype(np.float64)

    mean_val = float(np.mean(lum_f))
    median_val = float(np.median(lum_f))
    min_val = float(np.min(lum_f))
    max_val = float(np.max(lum_f))

    p5, p10, p50, p90, p95 = np.percentile(lum_f, [5, 10, 50, 90, 95])

    return {
        "mean_brightness": round(mean_val, 4),
        "median_brightness": round(median_val, 4),
        "min_brightness": round(min_val, 4),
        "max_brightness": round(max_val, 4),
        "p5": round(float(p5), 4),
        "p10": round(float(p10), 4),
        "p50": round(float(p50), 4),
        "p90": round(float(p90), 4),
        "p95": round(float(p95), 4),
    }


# ==============================================================================
# 2. CONTRAST ANALYSIS
# ==============================================================================

def compute_global_contrast(lum: np.ndarray) -> Tuple[float, float]:
    """
    Compute global Root Mean Square (RMS) contrast and normalized contrast.
    
    RMS contrast is defined as the standard deviation of pixel intensities:
        sigma = sqrt( (1 / N) * sum((Y_i - mu)^2) )
    
    Normalized contrast is normalized by 255.0 for scale invariance:
        C_norm = sigma / 255.0
        
    Args:
        lum: 2D luminance array.
        
    Returns:
        Tuple[float, float]: (global_contrast, normalized_contrast).
    """
    validate_image(lum, allow_color=False, name="compute_global_contrast input")
    lum_f = lum.astype(np.float64)
    std_val = float(np.std(lum_f))
    norm_val = std_val / 255.0
    return round(std_val, 4), round(norm_val, 4)


def compute_local_contrast(lum: np.ndarray, block_size: int = 16) -> float:
    """
    Compute local contrast as the average standard deviation across non-overlapping blocks.
    
    This captures fine textural contrast that might be obscured by low global standard deviation.
    
    Args:
        lum: 2D luminance array.
        block_size: Size of square non-overlapping blocks (default: 16).
        
    Returns:
        float: Mean local standard deviation across all valid blocks.
    """
    validate_image(lum, allow_color=False, name="compute_local_contrast input")
    h, w = lum.shape[:2]
    if block_size <= 1 or block_size > min(h, w):
        raise ValueError(f"block_size must be between 2 and {min(h, w)}, got {block_size}.")

    lum_f = lum.astype(np.float64)
    # Number of full blocks along each dimension
    n_h = h // block_size
    n_w = w // block_size

    # Crop to integer block multiples
    cropped = lum_f[:n_h * block_size, :n_w * block_size]
    # Reshape into blocks: (n_h, block_size, n_w, block_size) -> (n_h * n_w, block_size * block_size)
    blocks = cropped.reshape(n_h, block_size, n_w, block_size).swapaxes(1, 2).reshape(-1, block_size * block_size)
    local_stds = np.std(blocks, axis=1)
    mean_local_contrast = float(np.mean(local_stds))
    return round(mean_local_contrast, 4)


# ==============================================================================
# 3. DYNAMIC RANGE
# ==============================================================================

def compute_effective_dynamic_range(lum: np.ndarray, lower_p: float = 5.0, upper_p: float = 95.0) -> float:
    """
    Compute robust percentile-based dynamic range: P_upper - P_lower.
    
    Using P95 - P5 prevents isolated dead pixels, hot pixels, or specular outlier spikes
    from artificially inflating the measured dynamic range.
    
    Args:
        lum: 2D luminance array.
        lower_p: Lower percentile (default 5.0).
        upper_p: Upper percentile (default 95.0).
        
    Returns:
        float: Effective dynamic range in intensity units.
    """
    validate_image(lum, allow_color=False, name="compute_effective_dynamic_range input")
    if not (0.0 <= lower_p < upper_p <= 100.0):
        raise ValueError(f"Invalid percentiles: lower={lower_p}, upper={upper_p}")

    lum_f = lum.astype(np.float64)
    p_low, p_high = np.percentile(lum_f, [lower_p, upper_p])
    dynamic_range = float(p_high - p_low)
    return round(dynamic_range, 4)


# ==============================================================================
# 4. NOISE ESTIMATION
# ==============================================================================

def estimate_noise_level(lum: np.ndarray) -> float:
    """
    Estimate image-based high-frequency additive noise standard deviation using Immerkaer's
    fast noise estimation algorithm (Immerkaer, 1996).
    
    Algorithm:
        Convolves luminance with a 3x3 Laplacian mask M:
            M = [ 1  -2   1 ]
                [-2   4  -2 ]
                [ 1  -2   1 ]
        The noise sigma is estimated as:
            sigma_n = sqrt(pi / 2) * (1 / (6 * (W - 2) * (H - 2))) * sum(|lum * M|)
            
    This metric isolates high-frequency noise from smooth illumination trends and textures.
    
    Args:
        lum: 2D luminance array (uint8).
        
    Returns:
        float: Estimated noise standard deviation sigma_n.
    """
    validate_image(lum, allow_color=False, name="estimate_noise_level input")
    h, w = lum.shape[:2]
    if h < 3 or w < 3:
        raise ValueError(f"Image too small for 3x3 noise filter: ({h}, {w})")

    lum_f = lum.astype(np.float64)
    # Immerkaer kernel
    kernel = np.array([
        [1, -2, 1],
        [-2, 4, -2],
        [1, -2, 1]
    ], dtype=np.float64)

    # Convolve with valid region
    conv = cv2.filter2D(lum_f, cv2.CV_64F, kernel, borderType=cv2.BORDER_CONSTANT)
    # Exclude borders where padding might introduce artifacts
    valid_conv = conv[1:-1, 1:-1]

    sum_abs = np.sum(np.abs(valid_conv))
    scale = np.sqrt(np.pi / 2.0) / (6.0 * (w - 2) * (h - 2))
    sigma_noise = float(sum_abs * scale)
    return round(sigma_noise, 4)


# ==============================================================================
# 5. ILLUMINATION NON-UNIFORMITY
# ==============================================================================

def compute_illumination_nonuniformity(
    lum: np.ndarray,
    sigma: float = 30.0,
    eps: float = 1e-4
) -> float:
    """
    Measure spatial illumination non-uniformity across the image.
    
    Method:
        1. Extract the low-frequency illumination profile L_smooth by applying a strong
           Gaussian filter (sigma=30, capturing large-scale spatial trends).
        2. Compute the coefficient of variation (CV) of the illumination profile:
               INU = std(L_smooth) / (mean(L_smooth) + eps)
               
        When lighting is even throughout the scene, std(L_smooth) is very low -> INU near 0.
        When strong directional light, glare, or severe vignetting exists, INU is significantly higher.
        
    Args:
        lum: 2D luminance array.
        sigma: Standard deviation for large-scale Gaussian smoothing (default: 30.0).
        eps: Small constant to prevent division by zero in near-black images.
        
    Returns:
        float: Illumination non-uniformity index (dimensionless).
    """
    validate_image(lum, allow_color=False, name="compute_illumination_nonuniformity input")
    lum_f = lum.astype(np.float64)

    # Smooth luminance to isolate the ambient illumination component
    # Using a 61x61 kernel for sigma=30
    ksize = int(round(sigma * 2)) * 2 + 1
    illum_profile = cv2.GaussianBlur(lum_f, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)

    profile_std = float(np.std(illum_profile))
    profile_mean = float(np.mean(illum_profile))

    inu_score = profile_std / (profile_mean + eps)
    return round(inu_score, 4)


# ==============================================================================
# 6. SHANNON ENTROPY
# ==============================================================================

def compute_image_entropy(lum: np.ndarray, bins: int = 256) -> float:
    """
    Calculate the Shannon entropy of the luminance histogram:
        H = - sum( p_i * log2(p_i) )
    
    Measures information density and dispersion across available grayscale levels.
    Severely degraded images typically exhibit low entropy clustered in low bins.
    
    Args:
        lum: 2D luminance array (uint8).
        bins: Number of histogram bins (default: 256).
        
    Returns:
        float: Entropy in bits [0.0, 8.0].
    """
    validate_image(lum, allow_color=False, name="compute_image_entropy input")
    hist, _ = np.histogram(lum.flatten(), bins=bins, range=(0, 256), density=True)
    # Keep only non-zero probabilities
    non_zero = hist[hist > 0]
    entropy = -np.sum(non_zero * np.log2(non_zero))
    return round(float(entropy), 4)


# ==============================================================================
# 7. HIGH-LEVEL CONTENT-AWARE ANALYSIS PIPELINE
# ==============================================================================

def analyze_image_characteristics(
    image: np.ndarray,
    image_name: str = "unnamed",
    is_rgb: bool = True
) -> ImageAnalysisResult:
    """
    Execute complete content-aware quantitative characterization on an input image.
    
    Args:
        image: 3-channel color image (RGB or BGR) or 2D single-channel luminance map.
        image_name: Identifier for tracking and logging.
        is_rgb: True if input 3-channel image is RGB; False if BGR.
        
    Returns:
        ImageAnalysisResult: Strongly typed dataclass containing all metric values.
    """
    validate_image(image, name="analyze_image_characteristics input")

    # Extract single-channel luminance (Y from YCrCb for perceptual accuracy)
    if image.ndim == 3:
        lum = extract_luminance(image, method="ycrcb", is_rgb=is_rgb)
    else:
        lum = image.copy()

    # Compute all characteristics
    b_stats = compute_brightness_statistics(lum)
    g_contrast, norm_contrast = compute_global_contrast(lum)
    l_contrast = compute_local_contrast(lum, block_size=16)
    dyn_range = compute_effective_dynamic_range(lum, lower_p=5.0, upper_p=95.0)
    noise = estimate_noise_level(lum)
    inu = compute_illumination_nonuniformity(lum, sigma=30.0)
    entropy = compute_image_entropy(lum)

    return ImageAnalysisResult(
        image_name=image_name,
        mean_brightness=b_stats["mean_brightness"],
        median_brightness=b_stats["median_brightness"],
        min_brightness=b_stats["min_brightness"],
        max_brightness=b_stats["max_brightness"],
        p5=b_stats["p5"],
        p10=b_stats["p10"],
        p50=b_stats["p50"],
        p90=b_stats["p90"],
        p95=b_stats["p95"],
        global_contrast=g_contrast,
        normalized_contrast=norm_contrast,
        local_contrast=l_contrast,
        dynamic_range=dyn_range,
        noise_estimate=noise,
        illumination_nonuniformity=inu,
        entropy=entropy,
    )
