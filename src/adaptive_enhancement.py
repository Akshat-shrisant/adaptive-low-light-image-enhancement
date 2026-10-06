"""
Content-Aware Adaptive Low-Light Image Enhancement Decision Engine
Course: Image and Video Processing (IVP) Mini-Project

This module implements the core contribution of the project:
An explainable, deterministic, rule-based decision engine that analyzes input image
characteristics, classifies the underlying degradation profile, and automatically selects
and configures the most appropriate classical enhancement pipeline.

Degradation Profiles & Method Selection:
    1. High Sensor Noise Degraded -> Denoising (Bilateral) + CLAHE
    2. Severe Low-Light & Compressed Range -> Multi-Scale Retinex (MSR)
    3. Severe Illumination Non-Uniformity -> Multi-Scale Retinex (MSR)
    4. Flat Uniform Low-Contrast -> Adaptively Tuned CLAHE
    5. Mild Underexposure with Preserved Contrast -> Adaptive Gamma Correction
    6. Balanced Intermediate Degradation -> Robust Standard CLAHE (Fallback)

Threshold Calibration:
    All decision boundaries are empirically calibrated from the development dataset ('our485')
    using distribution quartiles (Q1, Median, Q3). The benchmark subset ('eval15') remains held-out.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple, Any
import numpy as np

from src.preprocessing import validate_image
from src.image_analysis import analyze_image_characteristics, ImageAnalysisResult
from src.enhancement_methods import (
    gamma_correction,
    histogram_equalization,
    clahe_enhancement,
    denoising_enhancement,
    single_scale_retinex,
    multi_scale_retinex,
)


@dataclass(frozen=True)
class AdaptiveThresholds:
    """
    Empirically grounded decision boundaries calibrated from 'our485'.
    
    Default values correspond to empirical quartiles (Q1, Median, Q3) computed
    across the LOL development set.
    """
    brightness_q1: float = 8.19
    brightness_med: float = 13.70
    brightness_q3: float = 19.32
    contrast_q1: float = 6.61
    contrast_med: float = 10.15
    contrast_q3: float = 12.95
    dynamic_range_q1: float = 18.00
    dynamic_range_med: float = 30.00
    noise_q3: float = 1.23
    noise_med: float = 1.14
    inu_q3: float = 0.64


@dataclass(frozen=True)
class AdaptiveDecisionResult:
    """Transparent structured decision record detailing the adaptive enhancement."""
    image_name: str
    detected_characteristics: Dict[str, float]
    degradation_profile: str
    selected_method: str
    selected_parameters: Dict[str, Any]
    decision_strength: str
    decision_reason: str
    enhanced_image: np.ndarray

    def to_dict(self, include_image: bool = False) -> Dict[str, Any]:
        """Convert decision record to a serializable dictionary."""
        d = {
            "image_name": self.image_name,
            "detected_characteristics": self.detected_characteristics,
            "degradation_profile": self.degradation_profile,
            "selected_method": self.selected_method,
            "selected_parameters": self.selected_parameters,
            "decision_strength": self.decision_strength,
            "decision_reason": self.decision_reason,
        }
        if include_image:
            d["enhanced_image"] = self.enhanced_image
        return d


def select_enhancement_strategy(
    analysis: ImageAnalysisResult,
    thresholds: Optional[AdaptiveThresholds] = None
) -> Tuple[str, str, str, Dict[str, Any], str]:
    """
    Evaluate image characteristics against empirical degradation profiles.
    
    Args:
        analysis: Quantitative image characteristics from Phase 5.
        thresholds: Calibrated decision boundaries (defaults to our485 quartiles).
        
    Returns:
        Tuple[str, str, str, Dict[str, Any], str]:
            (profile, selected_method, decision_strength, parameters, decision_reason)
    """
    t = thresholds or AdaptiveThresholds()

    b = analysis.mean_brightness
    c = analysis.global_contrast
    n = analysis.noise_estimate
    inu = analysis.illumination_nonuniformity
    dr = analysis.dynamic_range

    # Rule 1: High Sensor Noise Degraded
    # Condition: Elevated noise (>= Q3) in underexposed scene (< Q3)
    if n >= t.noise_q3 and b < t.brightness_q3:
        profile = "High-Noise Underexposure"
        method = "denoising_enhancement"
        strength = "Strong Match"
        # Dynamic filter adaptation: scale color sigma with noise level
        adapted_sigma_c = round(float(np.clip(40.0 + 20.0 * (n - t.noise_q3), 35.0, 70.0)), 1)
        params = {
            "denoise_method": "bilateral",
            "denoise_kwargs": {
                "d": 5,
                "sigma_color": adapted_sigma_c,
                "sigma_space": 50.0
            },
            "enhancement_method": "clahe",
            "enhancement_kwargs": {
                "clip_limit": 2.0,
                "tile_grid_size": (8, 8)
            }
        }
        reason = (
            f"Elevated sensor noise detected (sigma_n={n:.2f} >= Q3 threshold {t.noise_q3:.2f}) "
            f"under low-light (mean_Y={b:.2f}). Direct contrast expansion would severely amplify grain. "
            f"Selected bilateral edge-preserving pre-filtering (sigma_color={adapted_sigma_c}) followed by CLAHE."
        )

    # Rule 2: Severe Underexposure & Compressed Range
    # Condition: Deep darkness (mean_Y <= Q1) with severely compressed dynamic range (DR <= Median)
    elif b <= t.brightness_q1 and dr <= t.dynamic_range_med:
        profile = "Severe Low-Light & Compressed Range"
        method = "multi_scale_retinex"
        strength = "Strong Match"
        params = {
            "sigmas": (15.0, 80.0, 250.0),
            "color_space": "ycrcb"
        }
        reason = (
            f"Severe underexposure detected (mean_Y={b:.2f} <= Q1 {t.brightness_q1:.2f}) with "
            f"severely compressed dynamic range (DR={dr:.1f} <= Median {t.dynamic_range_med:.1f}). "
            f"Single-curve gamma or standard CLAHE cannot recover buried structural details. "
            f"Selected Multi-Scale Retinex (MSR) to decompose multi-scale illumination and recover intrinsic reflectance."
        )

    # Rule 3: Severe Illumination Non-Uniformity
    # Condition: High spatial lighting gradient (INU >= Q3)
    elif inu >= t.inu_q3:
        profile = "Severe Illumination Non-Uniformity"
        method = "multi_scale_retinex"
        strength = "Strong Match"
        params = {
            "sigmas": (15.0, 80.0, 250.0),
            "color_space": "ycrcb"
        }
        reason = (
            f"Pronounced spatial illumination non-uniformity detected (INU={inu:.3f} >= Q3 {t.inu_q3:.3f}). "
            f"Scene contains strong directional light or sharp localized shadowing. Global methods cause highlight "
            f"burnout or underexposed shadows. Selected Multi-Scale Retinex to normalize the spatial lighting field."
        )

    # Rule 4: Uniform Low-Contrast Underexposure
    # Condition: Suppressed contrast (< Median) under relatively uniform lighting (INU < Q3) and moderate noise (< Q3)
    elif c < t.contrast_med and n < t.noise_q3:
        profile = "Uniform Low-Contrast Underexposure"
        method = "clahe_enhancement"
        strength = "Strong Match"
        # Adapt clip limit: narrower dynamic range receives slightly higher contrast boost
        adapted_clip = round(float(np.clip(2.0 + (t.dynamic_range_med - dr) / 25.0, 1.5, 3.2)), 2)
        params = {
            "clip_limit": adapted_clip,
            "tile_grid_size": (8, 8),
            "color_space": "ycrcb"
        }
        reason = (
            f"Low global contrast (std_Y={c:.2f} < Median {t.contrast_med:.2f}) under uniform illumination "
            f"(INU={inu:.3f} < Q3) with low noise (sigma_n={n:.2f} < Q3). Selected CLAHE with adaptively tuned "
            f"clip_limit={adapted_clip} to maximize local textural visibility without highlight burnout."
        )

    # Rule 5: Mild Underexposure with Preserved Contrast & Clean Signal
    # Condition: Moderate/high brightness (>= Median) and contrast (>= Median) with low noise (< Median)
    elif b >= t.brightness_med and c >= t.contrast_med and n < t.noise_med:
        profile = "Mild Underexposure (Clean Signal)"
        method = "gamma_correction"
        strength = "Strong Match"
        # Adapt gamma: darker scenes get slightly lower gamma (< 1 brightens)
        adapted_gamma = round(float(np.clip(0.35 + 0.015 * b, 0.45, 0.70)), 2)
        params = {
            "gamma": adapted_gamma,
            "color_space": "ycrcb"
        }
        reason = (
            f"Mild underexposure (mean_Y={b:.2f} >= Median {t.brightness_med:.2f}) with intact contrast "
            f"(std_Y={c:.2f} >= Median) and clean signal (sigma_n={n:.2f} < Median {t.noise_med:.2f}). "
            f"Selected smooth monotonic Gamma Correction (gamma={adapted_gamma}) to expand dark midtones "
            f"without introducing local contrast or halo artifacts."
        )

    # Rule 6: Moderate Underexposure (Balanced Intermediate Fallback)
    else:
        profile = "Moderate Underexposure (Balanced)"
        method = "clahe_enhancement"
        strength = "Moderate Match"
        params = {
            "clip_limit": 2.0,
            "tile_grid_size": (8, 8),
            "color_space": "ycrcb"
        }
        reason = (
            f"Balanced intermediate degradation detected (mean_Y={b:.2f}, std_Y={c:.2f}, INU={inu:.3f}, sigma_n={n:.2f}). "
            f"No single extreme degradation dominates. Selected baseline CLAHE (clip_limit=2.0, grid=8x8) as the most "
            f"stable, generalizable enhancement."
        )

    return profile, method, strength, params, reason


def apply_selected_enhancement(
    img_rgb: np.ndarray,
    method: str,
    params: Dict[str, Any]
) -> np.ndarray:
    """
    Execute the assigned enhancement algorithm with its selected parameters.
    
    Args:
        img_rgb: Input 3-channel RGB image (uint8).
        method: Method name string.
        params: Parameter dictionary.
        
    Returns:
        np.ndarray: Enhanced RGB image (uint8).
    """
    if method == "gamma_correction":
        return gamma_correction(img_rgb, **params)
    elif method == "histogram_equalization":
        return histogram_equalization(img_rgb, **params)
    elif method == "clahe_enhancement":
        return clahe_enhancement(img_rgb, **params)
    elif method == "denoising_enhancement":
        return denoising_enhancement(img_rgb, **params)
    elif method == "single_scale_retinex":
        return single_scale_retinex(img_rgb, **params)
    elif method == "multi_scale_retinex":
        return multi_scale_retinex(img_rgb, **params)
    else:
        raise ValueError(f"Unknown enhancement method: '{method}'")


def adaptive_enhance(
    img_rgb: np.ndarray,
    image_name: str = "unnamed_image",
    is_rgb: bool = True,
    thresholds: Optional[AdaptiveThresholds] = None
) -> AdaptiveDecisionResult:
    """
    High-level content-aware adaptive low-light image enhancement pipeline.
    
    Workflow:
        1. Validate input RGB image.
        2. Compute comprehensive quantitative characteristics (Phase 5).
        3. Classify degradation profile and select optimal baseline pipeline (Phase 7).
        4. Configure dynamically adapted parameters.
        5. Execute enhancement without modifying input.
        6. Return complete transparent decision record and enhanced image.
        
    Args:
        img_rgb: 3-channel RGB image (uint8).
        image_name: Identifier for tracking.
        is_rgb: True if RGB, False if OpenCV BGR.
        thresholds: Custom or default empirical thresholds.
        
    Returns:
        AdaptiveDecisionResult: Container with metrics, decision, and enhanced image.
    """
    validate_image(img_rgb, allow_gray=False, name="adaptive_enhance input")

    # Step 1: Content-Aware Feature Extraction
    analysis = analyze_image_characteristics(img_rgb, image_name=image_name, is_rgb=is_rgb)

    # Step 2: Adaptive Decision & Parameter Calibration
    profile, method, strength, params, reason = select_enhancement_strategy(
        analysis=analysis,
        thresholds=thresholds
    )

    # Step 3: Execute Selected Enhancement
    enhanced = apply_selected_enhancement(img_rgb, method, params)

    return AdaptiveDecisionResult(
        image_name=image_name,
        detected_characteristics=analysis.to_dict(),
        degradation_profile=profile,
        selected_method=method,
        selected_parameters=params,
        decision_strength=strength,
        decision_reason=reason,
        enhanced_image=enhanced
    )
