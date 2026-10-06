"""
Baseline Image Enhancement Algorithms Module
Course: Image and Video Processing (IVP) Mini-Project

This module implements five canonical, academically defensible baseline low-light
image enhancement techniques:
    1. Configurable Gamma Correction (Power-Law Luminance Transformation)
    2. Luminance Histogram Equalization (HE)
    3. Contrast Limited Adaptive Histogram Equalization (CLAHE)
    4. Denoising + Contrast Enhancement Pipeline
    5. Classical Retinex (Single-Scale and Multi-Scale Retinex with Luminance Preservation)

Design Principles:
    - Luminance-Centric Processing: Processing is applied primarily to the achromatic
      luminance channel (Y in YCrCb) to preserve original chromatic ratios and prevent
      the chromatic distortion/color shifts typical of independent RGB-channel processing.
    - Pure Functions: Inputs are never modified in-place; all functions return new arrays.
    - Deterministic & Configurable: Parameters are explicit, validated, and documented.
    - Standard I/O Contract: Accepts RGB uint8 [H, W, 3], returns RGB uint8 [H, W, 3].
"""

from typing import Optional, Tuple, Union
import cv2
import numpy as np

from src.preprocessing import (
    validate_image,
    extract_luminance,
    replace_luminance,
    gaussian_denoise,
    median_denoise,
    bilateral_denoise,
    nlm_denoise,
)


# ==============================================================================
# 1. GAMMA CORRECTION
# ==============================================================================

def gamma_correction(
    img_rgb: np.ndarray,
    gamma: float = 0.5,
    color_space: str = "ycrcb"
) -> np.ndarray:
    """
    Apply power-law (Gamma) transformation to the luminance channel of an RGB image.
    
    Mathematical Formulation:
        Y_norm = Y / 255.0
        Y_enhanced = (Y_norm ** gamma) * 255.0
        
    When gamma < 1.0, dark pixel intensities are mapped to brighter values, expanding
    the shadows while compressing the highlights.
    When gamma > 1.0, intensities are darkened.
    When gamma == 1.0, the transformation is identity.
    
    Color Space Strategy:
        Operates on the isolated luminance channel (Y in YCrCb) and reassembles with
        original chrominance to prevent severe color saturation distortions caused by
        independent per-channel RGB gamma scaling.
        
    Args:
        img_rgb: 3-channel RGB image (uint8).
        gamma: Power-law exponent. Must be positive (default: 0.5 for low-light brightening).
        color_space: Color space for luminance separation ('ycrcb' or 'lab').
        
    Returns:
        np.ndarray: Enhanced RGB image (uint8).
        
    Limitations:
        - Global transformation: Treats all spatial regions uniformly.
        - May over-amplify already bright regions if gamma is too aggressive (< 0.3).
        - Does not perform local contrast enhancement or noise suppression.
    """
    validate_image(img_rgb, allow_gray=False, name="gamma_correction input")
    if gamma <= 0:
        raise ValueError(f"Gamma exponent must be strictly positive, got {gamma}.")

    # Fast LUT (Look-Up Table) for 8-bit integer luminance
    inv_gamma = gamma
    lut = np.array([((i / 255.0) ** inv_gamma) * 255.0 for i in range(256)]).astype(np.uint8)

    y_channel = extract_luminance(img_rgb, method=color_space, is_rgb=True)
    y_enhanced = cv2.LUT(y_channel, lut)

    return replace_luminance(img_rgb, y_enhanced, method=color_space)


# ==============================================================================
# 2. HISTOGRAM EQUALIZATION (HE)
# ==============================================================================

def histogram_equalization(
    img_rgb: np.ndarray,
    color_space: str = "ycrcb"
) -> np.ndarray:
    """
    Apply global Histogram Equalization (HE) to the luminance channel of an RGB image.
    
    Mathematical Formulation:
        Computes the Cumulative Distribution Function (CDF) of luminance Y:
            CDF(k) = sum_{i=0}^k p_Y(i)
        Maps intensities:
            Y_equalized(k) = round( (CDF(k) - CDF_min) / (N - CDF_min) * 255 )
            
    This flattens and spreads the luminance histogram across the full [0, 255] range.
    
    Color Space Strategy:
        Applies equalization exclusively to the luminance channel. Applying HE
        independently to R, G, and B alters color ratios, causing severe unnatural
        color casts. Recombining with original chromatic channels preserves hue.
        
    Args:
        img_rgb: 3-channel RGB image (uint8).
        color_space: Color space for luminance separation ('ycrcb' or 'lab').
        
    Returns:
        np.ndarray: Enhanced RGB image (uint8).
        
    Limitations:
        - Global operation: Amplifies noise heavily in uniformly dark backgrounds.
        - Tends to over-saturate scenes with non-uniform lighting (e.g. wash out highlights).
        - Can result in an unnatural, harsh appearance with lost subtle gradations.
    """
    validate_image(img_rgb, allow_gray=False, name="histogram_equalization input")

    y_channel = extract_luminance(img_rgb, method=color_space, is_rgb=True)
    y_enhanced = cv2.equalizeHist(y_channel)

    return replace_luminance(img_rgb, y_enhanced, method=color_space)


# ==============================================================================
# 3. CONTRAST LIMITED ADAPTIVE HISTOGRAM EQUALIZATION (CLAHE)
# ==============================================================================

def clahe_enhancement(
    img_rgb: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
    color_space: str = "ycrcb"
) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) (Zuiderveld, 1994).
    
    Mathematical Formulation:
        1. Divides the image into a grid of contextual tiles (e.g. 8x8 tiles).
        2. Computes the histogram for each tile.
        3. Clips histogram bins exceeding 'clip_limit' and redistributes clipped pixels
           uniformly among all bins to prevent local over-amplification of noise.
        4. Applies equalization to each tile and combines neighboring tiles via bilinear
           interpolation to eliminate artificial tile boundary artifacts.
           
    Color Space Strategy:
        CLAHE is applied to the isolated luminance plane (Y in YCrCb), preserving
        original color balance while boosting local contrast in dark shadow regions.
        
    Args:
        img_rgb: 3-channel RGB image (uint8).
        clip_limit: Threshold for contrast limiting (default: 2.0). Higher values increase
                    contrast but may amplify noise.
        tile_grid_size: Grid size for local histogram equalization (default: (8, 8)).
        color_space: Color space for luminance extraction ('ycrcb' or 'lab').
        
    Returns:
        np.ndarray: Enhanced RGB image (uint8).
        
    Limitations:
        - If clip_limit is set too high, background photon noise is amplified.
        - Grid size choice introduces a scale dependency: very small grids over-enhance
          micro-texture, while very large grids approach standard global HE.
    """
    validate_image(img_rgb, allow_gray=False, name="clahe_enhancement input")
    if clip_limit <= 0:
        raise ValueError(f"clip_limit must be positive, got {clip_limit}.")
    if tile_grid_size[0] <= 0 or tile_grid_size[1] <= 0:
        raise ValueError(f"tile_grid_size dimensions must be positive, got {tile_grid_size}.")

    y_channel = extract_luminance(img_rgb, method=color_space, is_rgb=True)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    y_enhanced = clahe.apply(y_channel)

    return replace_luminance(img_rgb, y_enhanced, method=color_space)


# ==============================================================================
# 4. DENOISING + ENHANCEMENT PIPELINE
# ==============================================================================

def denoising_enhancement(
    img_rgb: np.ndarray,
    denoise_method: str = "bilateral",
    denoise_kwargs: Optional[dict] = None,
    enhancement_method: str = "clahe",
    enhancement_kwargs: Optional[dict] = None
) -> np.ndarray:
    """
    Two-stage enhancement pipeline: Denoising pre-filter followed by contrast enhancement.
    
    Motivation:
        Low-light image sensors produce low signal-to-noise ratios (SNR). Applying
        aggressive contrast enhancement (CLAHE, HE) amplifies sensor noise.
        Preprocessing with an edge-preserving filter before enhancement suppresses
        noise spikes before they can be boosted.
        
    Pipeline:
        Input Low-Light RGB -> Spatial Edge-Preserving Denoising -> Luminance Contrast Enhancement -> Enhanced RGB
        
    Args:
        img_rgb: 3-channel RGB image (uint8).
        denoise_method: Denoising operator ('bilateral', 'gaussian', 'median', 'nlm').
        denoise_kwargs: Parameters for the chosen denoiser. Defaults:
            - bilateral: {'d': 5, 'sigma_color': 50.0, 'sigma_space': 50.0}
            - gaussian: {'ksize': (3, 3), 'sigma': 1.0}
            - median: {'ksize': 3}
            - nlm: {'h': 8.0, 'template_window_size': 5, 'search_window_size': 15}
        enhancement_method: Downstream contrast method ('clahe' or 'gamma').
        enhancement_kwargs: Parameters for the chosen enhancer. Defaults:
            - clahe: {'clip_limit': 2.0, 'tile_grid_size': (8, 8)}
            - gamma: {'gamma': 0.5}
            
    Returns:
        np.ndarray: Denoised and enhanced RGB image (uint8).
    """
    validate_image(img_rgb, allow_gray=False, name="denoising_enhancement input")

    # 1. Apply Denoising
    d_meth = denoise_method.lower()
    d_kw = denoise_kwargs or {}

    if d_meth == "bilateral":
        d = d_kw.get("d", 5)
        sc = d_kw.get("sigma_color", 50.0)
        ss = d_kw.get("sigma_space", 50.0)
        denoised_rgb = bilateral_denoise(img_rgb, d=d, sigma_color=sc, sigma_space=ss)
    elif d_meth == "gaussian":
        ksize = d_kw.get("ksize", (3, 3))
        sigma = d_kw.get("sigma", 1.0)
        denoised_rgb = gaussian_denoise(img_rgb, ksize=ksize, sigma=sigma)
    elif d_meth == "median":
        ksize = d_kw.get("ksize", 3)
        denoised_rgb = median_denoise(img_rgb, ksize=ksize)
    elif d_meth == "nlm":
        h = d_kw.get("h", 8.0)
        tws = d_kw.get("template_window_size", 5)
        sws = d_kw.get("search_window_size", 15)
        denoised_rgb = nlm_denoise(img_rgb, h=h, template_window_size=tws, search_window_size=sws)
    else:
        raise ValueError(f"Unsupported denoise_method '{denoise_method}'. Choose from: 'bilateral', 'gaussian', 'median', 'nlm'.")

    # 2. Apply Enhancement
    e_meth = enhancement_method.lower()
    e_kw = enhancement_kwargs or {}

    if e_meth == "clahe":
        clip = e_kw.get("clip_limit", 2.0)
        tgs = e_kw.get("tile_grid_size", (8, 8))
        return clahe_enhancement(denoised_rgb, clip_limit=clip, tile_grid_size=tgs)
    elif e_meth == "gamma":
        g = e_kw.get("gamma", 0.5)
        return gamma_correction(denoised_rgb, gamma=g)
    else:
        raise ValueError(f"Unsupported enhancement_method '{enhancement_method}'. Choose from: 'clahe', 'gamma'.")


# ==============================================================================
# 5. CLASSICAL RETINEX (SINGLE-SCALE AND MULTI-SCALE)
# ==============================================================================

def single_scale_retinex(
    img_rgb: np.ndarray,
    sigma: float = 80.0,
    eps: float = 1e-4,
    color_space: str = "ycrcb"
) -> np.ndarray:
    """
    Apply Single-Scale Retinex (SSR) (Land & McCann, 1971; Jobson et al., 1997).
    
    Retinex Theory:
        An image I(x, y) is decomposed into the product of illumination L(x, y)
        and intrinsic surface reflectance R(x, y):
            I(x, y) = R(x, y) * L(x, y)
            
        In the logarithmic domain:
            log R(x, y) = log I(x, y) - log L(x, y)
            
        The illumination L is estimated via spatial convolution with a Gaussian surround:
            L(x, y) = I(x, y) * G(x, y; sigma)
            
        Therefore:
            R_SSR(x, y) = log(I(x, y) + eps) - log((I(x, y) * G_sigma) + eps)
            
    Dynamic Range Compression & Normalization:
        The logarithmic reflectance R_SSR is normalized back to the [0, 255] display
        range using a robust statistical dynamic range stretch (mean +/- 2*std).
        
    Color Preservation:
        Retinex is applied to the isolated luminance channel (Y in YCrCb). This decouples
        illumination from surface reflectance without causing the gray-world color distortion
        typical of per-channel RGB Retinex.
        
    Args:
        img_rgb: 3-channel RGB image (uint8).
        sigma: Gaussian surround standard deviation (default: 80.0 for medium-scale illumination).
        eps: Small positive constant preventing log(0).
        color_space: Color space for luminance isolation ('ycrcb' or 'lab').
        
    Returns:
        np.ndarray: Enhanced RGB image (uint8).
    """
    validate_image(img_rgb, allow_gray=False, name="single_scale_retinex input")
    if sigma <= 0:
        raise ValueError(f"Sigma must be positive, got {sigma}.")

    y_channel = extract_luminance(img_rgb, method=color_space, is_rgb=True)
    y_float = y_channel.astype(np.float64) + eps

    # Gaussian surround filter for illumination estimation
    ksize = int(round(sigma * 2)) * 2 + 1
    illumination = cv2.GaussianBlur(y_float, (ksize, ksize), sigmaX=sigma, sigmaY=sigma) + eps

    # Logarithmic reflectance
    log_reflectance = np.log(y_float) - np.log(illumination)

    # Statistical dynamic range mapping to [0, 255]
    mean_r = np.mean(log_reflectance)
    std_r = np.std(log_reflectance)
    min_bound = mean_r - 2.0 * std_r
    max_bound = mean_r + 2.0 * std_r

    if max_bound > min_bound:
        y_enhanced = np.clip((log_reflectance - min_bound) / (max_bound - min_bound) * 255.0, 0, 255).astype(np.uint8)
    else:
        y_enhanced = np.clip(log_reflectance * 255.0, 0, 255).astype(np.uint8)

    return replace_luminance(img_rgb, y_enhanced, method=color_space)


def multi_scale_retinex(
    img_rgb: np.ndarray,
    sigmas: Tuple[float, ...] = (15.0, 80.0, 250.0),
    weights: Optional[Tuple[float, ...]] = None,
    eps: float = 1e-4,
    color_space: str = "ycrcb"
) -> np.ndarray:
    """
    Apply Multi-Scale Retinex (MSR) (Jobson, Rahman, Woodell, 1997).
    
    Mathematical Principle:
        Single-Scale Retinex presents an inherent trade-off:
            - Small sigma (e.g. 15): Strong dynamic range compression, sharp details, but halo artifacts.
            - Large sigma (e.g. 250): Good color rendition and natural tonal gradation, but weaker dynamic compression.
        MSR resolves this by computing a weighted average across multiple Gaussian scales:
            R_MSR(x, y) = sum_{k=1}^K w_k * [ log(Y + eps) - log(Y * G_{sigma_k} + eps) ]
            where sum_{k=1}^K w_k = 1.0.
            
    Args:
        img_rgb: 3-channel RGB image (uint8).
        sigmas: Tuple of Gaussian surround scales (default: (15.0, 80.0, 250.0) covering fine, medium, and wide scales).
        weights: Tuple of relative weights for each scale. Defaults to equal weighting (1/K, ..., 1/K).
        eps: Regularization constant preventing log(0).
        color_space: Color space for luminance separation ('ycrcb' or 'lab').
        
    Returns:
        np.ndarray: Enhanced RGB image (uint8).
        
    Limitations:
        - Higher computational complexity due to multiple large Gaussian convolutions.
        - Can create subtle halo transitions along high-contrast object silhouettes.
    """
    validate_image(img_rgb, allow_gray=False, name="multi_scale_retinex input")
    if not sigmas:
        raise ValueError("At least one sigma scale must be provided.")

    for s in sigmas:
        if s <= 0:
            raise ValueError(f"All sigmas must be positive, got {s} in {sigmas}.")

    if weights is None:
        num_scales = len(sigmas)
        weights = tuple(1.0 / num_scales for _ in range(num_scales))
    else:
        if len(weights) != len(sigmas):
            raise ValueError(f"Mismatch between number of sigmas ({len(sigmas)}) and weights ({len(weights)}).")
        w_sum = sum(weights)
        if w_sum <= 0:
            raise ValueError("Weights sum must be strictly positive.")
        weights = tuple(w / w_sum for w in weights)

    y_channel = extract_luminance(img_rgb, method=color_space, is_rgb=True)
    y_float = y_channel.astype(np.float64) + eps
    log_y = np.log(y_float)

    msr_reflectance = np.zeros_like(y_float, dtype=np.float64)

    for sigma, weight in zip(sigmas, weights):
        # Gaussian surround
        ksize = int(round(sigma * 2)) * 2 + 1
        illumination = cv2.GaussianBlur(y_float, (ksize, ksize), sigmaX=sigma, sigmaY=sigma) + eps
        ssr = log_y - np.log(illumination)
        msr_reflectance += weight * ssr

    # Statistical dynamic range mapping to [0, 255]
    mean_r = np.mean(msr_reflectance)
    std_r = np.std(msr_reflectance)
    min_bound = mean_r - 2.0 * std_r
    max_bound = mean_r + 2.0 * std_r

    if max_bound > min_bound:
        y_enhanced = np.clip((msr_reflectance - min_bound) / (max_bound - min_bound) * 255.0, 0, 255).astype(np.uint8)
    else:
        y_enhanced = np.clip(msr_reflectance * 255.0, 0, 255).astype(np.uint8)

    return replace_luminance(img_rgb, y_enhanced, method=color_space)
