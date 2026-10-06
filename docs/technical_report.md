# Adaptive Low-Light Image Enhancement Using Content-Aware Image Processing Techniques

**Course**: Image and Video Processing (IVP) — Mini-Project  
**Academic Level**: 3rd-Year B.Tech Computer Engineering  
**Project Group**: 2 Students  
**Academic Year**: 2026–2027  

---

## 1. Title Page & Project Metadata

* **Project Title**: Adaptive Low-Light Image Enhancement Using Content-Aware Image Processing Techniques
* **Domain**: Digital Image Processing / Computer Vision (Classical Methods)
* **Dataset**: LOL (LOw-Light) Paired Dataset (Wei et al., BMVC 2018)
* **Implementation Language**: Python 3.14+ (NumPy, OpenCV, Scikit-Image, SciPy, Pandas, Pillow, Flask)
* **Target Hardware**: Standard CPU (Personal Computer / Laptop), Deterministic & Non-Deep-Learning Execution

---

## 2. Abstract
Low-light digital image capture suffers from severe visual degradation caused by insufficient photon flux, sensor noise, dynamic range compression, and spatial non-uniformity of artificial or natural illumination. Classical enhancement techniques—such as Gamma Correction, Histogram Equalization (HE), Contrast Limited Adaptive Histogram Equalization (CLAHE), and Retinex-based models—apply static mathematical transformations uniformly across all input images regardless of individual degradation characteristics. Consequently, a single fixed enhancement pipeline often causes highlight blowout in non-uniformly lit scenes, severe noise amplification in grain-dominated images, or insufficient brightness expansion in severely underexposed environments.

To address this limitation, this project investigates the research question: *"Can a content-aware adaptive selection of classical image-processing techniques provide more consistent low-light image enhancement than applying one fixed enhancement technique to all images?"* We propose an explainable, deterministic, content-aware enhancement framework that first extracts quantitative degradation indicators—including mean luminance, global RMS contrast, dynamic range span ($P_{95}-P_5$), high-frequency sensor noise ($\sigma_n$), and spatial illumination non-uniformity ($\text{INU}$)—and then dynamically maps the image to a matched enhancement pipeline with dynamically adapted parameters. 

The decision boundaries were empirically calibrated on the 485 training pairs (`our485`) of the LOL benchmark, and quantitatively validated on the strictly held-out test subset (`eval15`, 15 paired scenes, 105 evaluations across 7 methods). Full-reference objective evaluation reveals that while Multi-Scale Retinex (MSR) achieved the highest average reconstruction metrics ($\text{PSNR} = 15.03 \pm 3.30$ dB, $\text{SSIM} = 0.577 \pm 0.123$) due to substantial global brightness shift, the Adaptive System exhibited relatively low PSNR variability (σ = 3.08 dB), lower than Gamma Correction (3.60 dB), SSR (3.28 dB), MSR (3.30 dB), HE (3.14 dB), and CLAHE (3.11 dB), although Denoise + CLAHE showed slightly lower variability (3.02 dB). The Adaptive System produced substantially lower estimated output noise than Histogram Equalization (4.41 vs. 13.61), although Denoise + CLAHE, CLAHE, and Gamma produced lower average noise estimates. It outperformed fixed CLAHE and Denoise+CLAHE baselines in pairwise comparisons (60.0% win rate over Denoise+CLAHE with zero SSIM losses). These findings provide qualified empirical support for content-adaptive selection in classical low-light enhancement.

---

## 3. Keywords
Low-Light Image Enhancement, Content-Aware Processing, Adaptive Decision Engine, Luminance Processing, CLAHE, Multi-Scale Retinex, Noise Estimation, Illumination Non-Uniformity, LOL Dataset.

---

## 4. Introduction
Digital cameras operating in low-light environments face fundamental optical and sensor constraints. When ambient illuminance is low, the sensor receives an insufficient number of photons, leading to a low signal-to-noise ratio (SNR), compressed dynamic range, and loss of critical textural details. Modern consumer imaging often encounters severe underexposure in indoor surveillance, nighttime autonomous driving, computational photography, and forensic inspection.

Classical digital image processing provides several well-established enhancement paradigms:
1. **Point Transformations**: Power-law (Gamma) transformations adjust brightness non-linearly across the dynamic range.
2. **Global Histogram Modeling**: Histogram Equalization (HE) flattens the cumulative intensity distribution to expand contrast.
3. **Local Histogram Modeling**: CLAHE operates on localized contextual tiles while enforcing an empirical clipping limit to prevent over-amplification of noise.
4. **Retinex Physical Illumination Models**: Single-Scale and Multi-Scale Retinex decompose images into low-frequency spatial illumination and high-frequency reflectance components.

However, each of these methods assumes a specific degradation model. When an image violates these assumptions (e.g., applying HE to a scene with elevated sensor noise), the result is visually unnatural and mathematically degraded. This project investigates a content-aware framework that quantitatively evaluates the input image before selecting and parameterizing an enhancement pipeline.

---

## 5. Problem Statement
Fixed-parameter classical image enhancement algorithms fail to generalize across diverse low-light degradations because they lack awareness of the underlying physical degradation profile. A single fixed pipeline cannot simultaneously handle:
* High sensor noise without introducing filtering blur on clean scenes.
* Severe underexposure without producing highlight clipping in scenes with localized light sources.
* Compressed dynamic range without washing out midtone textural details.

There is a need for an automated, lightweight, rule-based decision mechanism that estimates degradation metrics directly from the input low-light image and adaptively routes the image to the most appropriate classical processing branch.

---

## 6. Motivation
While deep-learning-based low-light enhancement architectures (such as RetinexNet, Zero-DCE, and EnlightenGAN) have achieved notable visual fidelity, they impose significant computational barriers: high GPU memory footprints, dependence on massive training corpora, vulnerability to out-of-domain hallucinations, and black-box uninterpretability. 

In contrast, classical image processing techniques are deterministic, lightweight, mathematically transparent, and capable of executing in near-real-time on resource-constrained consumer CPUs. Developing an intelligent, content-aware meta-controller that orchestrates classical methods offers an academically defensible, practical, and interpretable engineering solution for edge devices, surveillance cameras, and embedded platforms.

---

## 7. Objectives
1. **Empirical Dataset Profiling**: Characterize the statistical distributions of brightness, contrast, noise, dynamic range, and illumination non-uniformity across the LOL dataset.
2. **Baseline Implementation**: Implement and standardize six canonical classical enhancement baselines operating primarily in the luminance domain (YCrCb).
3. **Adaptive Decision Engine**: Design an explainable, rule-based decision engine that classifies low-light degradation and adapts enhancement parameters.
4. **Strict Held-Out Validation**: Quantitatively evaluate all baselines and the proposed system across the unseen `eval15` benchmark using PSNR, SSIM, output contrast, residual noise, and runtime latency.
5. **Interactive Demonstration**: Build a clean, responsive demonstration application for faculty presentation and experimental reproducibility.

---

## 8. Research Question
> *"Can a content-aware adaptive selection of classical image-processing techniques provide more consistent low-light image enhancement than applying one fixed enhancement technique to all images?"*

---

## 9. Literature Review

The development of this project builds upon foundational literature in digital image processing:

1. **Retinex Theory and Multi-Scale Illumination Modeling**:
   Land (1977) formulated the Retinex theory, positing that human color constancy decomposes visual perception into reflectance $R(x,y)$ and illuminance $L(x,y)$, where $I(x,y) = R(x,y) \cdot L(x,y)$. Jobson, Rahman, and Woodell (1997) extended this to Multi-Scale Retinex (MSR), utilizing a weighted sum of Gaussian surround filters across small, medium, and large scales ($\sigma \in \{15, 80, 250\}$) to achieve dynamic range compression and color rendition simultaneously.

2. **Adaptive Contrast Enhancement**:
   Pizer et al. (1987) and Zuiderveld (1994) introduced Contrast Limited Adaptive Histogram Equalization (CLAHE). By dividing images into contextual grids and clipping the local histogram before redistribution, CLAHE mitigates the noise amplification intrinsic to global histogram equalization while enhancing localized detail.

3. **Fast Image Noise Estimation**:
   Immerkaer (1996) formulated a fast, uncalibrated noise variance estimation algorithm using a 2D Laplacian operator $N = I * M$, where $M$ is a $3 \times 3$ high-pass mask. The standard deviation of the high-frequency residual reliably reflects additive sensor noise without requiring homogeneous patch segmentation.

4. **Edge-Preserving Denoising**:
   Tomasi and Manduchi (1998) introduced the Bilateral Filter, which combines spatial domain closeness with photometric range similarity, smoothing sensor grain while preserving sharp structural boundaries.

5. **Structural Similarity and Perceptual Metrics**:
   Wang et al. (2004) developed the Structural Similarity Index (SSIM), establishing that human visual perception is adapted to extract structural information rather than absolute point-by-point differences.

6. **The LOL Benchmark Dataset**:
   Wei et al. (BMVC 2018) released the LOw-Light (LOL) dataset, providing 500 paired low/normal light photographs captured under real shooting conditions with varying ISO and exposure times.

```mermaid
graph TD
    A[Land 1977: Retinex Theory] --> B[Jobson 1997: Multi-Scale Retinex MSR]
    C[Zuiderveld 1994: CLAHE] --> D[Adaptive Contrast Enhancement]
    E[Immerkaer 1996: Laplacian Noise Est.] --> F[Degradation Feature Profiling]
    G[Tomasi 1998: Bilateral Filtering] --> H[Edge-Preserving Pre-Denoising]
    B & D & F & H --> I[Proposed Content-Aware Adaptive System]
```

---

## 10. Dataset Description & Partition Hygiene

The project uses the paired **LOL (LOw-Light) Dataset** (Wei et al., 2018):
* **Resolution**: All images are $600 \times 400$ pixels, 24-bit RGB in lossless PNG format.
* **Pairs**: 500 low-light images paired with 500 corresponding normal-light ground-truth references.
* **Storage Footprint**: Total size is 331.93 MB (348,058,979 bytes).

### Partition Protocol
* **`our485` (Development & Calibration)**: 485 image pairs used exclusively in Phases 1–7 to extract feature distributions, establish quartile thresholds, and formulate adaptive decision rules.
* **`eval15` (Held-Out Benchmark)**: 15 image pairs reserved exclusively for Phase 8 evaluation. **`eval15` was never accessed during threshold tuning or rule design.**
* **Non-Oracle Flow**: The adaptive decision engine processes strictly the low-light input image. Ground-truth reference images are accessed only after enhancement to compute objective error metrics.

---

## 11. Proposed Methodology & Architecture

The system pipeline comprises six discrete, deterministic stages:

```
Low-Light Input Image [uint8 RGB]
             │
             ▼
   [1. Preprocessing]
   • Dimension & Type Validation
   • YCrCb Luminance Channel Extraction
             │
             ▼
   [2. Content-Aware Feature Analysis]
   • Mean Luminance (μY)
   • Global RMS Contrast (σY)
   • Dynamic Range Span (P95 - P5)
   • Noise Estimate (Immerkaer σn)
   • Illumination Non-Uniformity (INU)
             │
             ▼
   [3. Adaptive Decision Engine]
   • Rule-Based Profile Matching (Frozen Quartiles)
   • Dynamic Parameter Calibration
             │
             ▼
   [4. Enhancement Execution]
   • Gamma / CLAHE / Denoise+CLAHE / MSR
   • Luminance Recombination & Denormalization
             │
             ▼
   [5. Enhanced Output Image]
             │
             ▼
   [6. Objective Evaluation]
   • Full-Reference Comparison against High Reference (eval15)
   • PSNR, SSIM, Residual Noise, Runtime Latency
```

---

## 12. Image Preprocessing Pipeline
To prevent chromatic distortion and saturation clipping, all enhancement operations are conducted primarily on the achromatic luminance channel:
1. Input validation: Asserts uint8 array format, 3 color channels, non-empty dimensions.
2. Color conversion: $\text{RGB} \rightarrow \text{YCrCb}$.
3. Channel separation: Luminance $Y$ is processed; chrominance components $Cr$ and $Cb$ are retained unaltered.
4. Channel recombination: Enhanced luminance $Y'$ is merged with original chrominance and inverted back to RGB with strict $[0, 255]$ dynamic range clipping.

---

## 13. Content-Aware Image Analysis

The analysis module computes five numerical features:

1. **Mean Luminance ($\mu_Y$)**:
   $$\mu_Y = \frac{1}{H \cdot W} \sum_{x=1}^{W} \sum_{y=1}^{H} Y(x, y)$$

2. **Global RMS Contrast ($\sigma_Y$)**:
   $$\sigma_Y = \sqrt{\frac{1}{H \cdot W} \sum_{x=1}^{W} \sum_{y=1}^{H} (Y(x, y) - \mu_Y)^2}$$

3. **Effective Dynamic Range ($\text{DR}$)**:
   $$\text{DR} = P_{95}(Y) - P_{5}(Y)$$

4. **Sensor Noise Level ($\sigma_n$)**:
   Computed using the Immerkaer Laplacian mask $M$:
   $$M = \begin{bmatrix} 1 & -2 & 1 \\ -2 & 4 & -2 \\ 1 & -2 & 1 \end{bmatrix}, \quad \sigma_n = \sqrt{\frac{\pi}{2}} \cdot \frac{1}{6(W-2)(H-2)} \sum |Y * M|$$

5. **Illumination Non-Uniformity ($\text{INU}$)**:
   Estimated via heavy Gaussian low-pass spatial filtering ($\sigma = 30$):
   $$L_{\text{bg}} = Y * G_{\sigma=30}, \quad \text{INU} = \frac{\sigma(L_{\text{bg}})}{\mu(L_{\text{bg}})}$$

---

## 14. Baseline Enhancement Methods

Six canonical baselines were implemented with fixed, standardized parameters:

1. **Gamma Correction ($\gamma = 0.5$)**:
   Power-law luminance expansion: $Y_{\text{out}} = 255 \cdot (Y / 255)^\gamma$.
   *Role*: Fast monotonic expansion of dark midtones.
   *Limitation*: Cannot enhance local contrast or suppress noise.

2. **Histogram Equalization (HE)**:
   Transforms luminance to yield a uniform cumulative probability distribution.
   *Role*: Maximizes global contrast across the full dynamic range.
   *Limitation*: Causes severe noise amplification and highlight blowout.

3. **CLAHE ($\text{clip\_limit} = 2.0$, $8 \times 8$ grid)**:
   Tiles the image, limits local histogram slope, and interpolates bilinearly.
   *Role*: Enhances localized textural visibility without excessive blowout.
   *Limitation*: May amplify sensor grain in noisy low-light regions.

4. **Denoise + CLAHE**:
   Applies an edge-preserving Bilateral filter ($d=5, \sigma_{\text{color}}=50, \sigma_{\text{space}}=50$) before CLAHE.
   *Role*: Suppresses high-frequency sensor noise prior to contrast expansion.
   *Limitation*: Introduces minor textural smoothing and execution overhead.

5. **Single-Scale Retinex (SSR, $\sigma = 80.0$)**:
   Logarithmic reflectance separation: $R(x,y) = \log(Y(x,y)) - \log(Y(x,y) * G_\sigma)$.
   *Role*: Dynamic range compression and shadow detail recovery.
   *Limitation*: Causes halo artifacts near strong luminance boundaries.

6. **Multi-Scale Retinex (MSR, $\sigma \in \{15, 80, 250\}$)**:
   Weighted combination of three surround scales:
   $$R_{\text{MSR}} = \sum_{k=1}^{3} w_k \cdot (\log Y - \log(Y * G_{\sigma_k})), \quad w_k = \frac{1}{3}$$
   *Role*: Balances dynamic range compression, local contrast, and edge sharpness.
   *Limitation*: High computational latency ($>160$ ms per frame).

---

## 15. Proposed Adaptive Enhancement Framework & Decision Logic

### Empirically Calibrated Development Boundaries (`our485`)
Quartile distributions established from the 485 training pairs:
* **Brightness**: $Q_1 = 8.19$, $\text{Median} = 13.70$, $Q_3 = 19.32$
* **Global Contrast**: $Q_1 = 6.61$, $\text{Median} = 10.15$, $Q_3 = 12.95$
* **Dynamic Range**: $Q_1 = 18.00$, $\text{Median} = 30.00$
* **Sensor Noise**: $\text{Median} = 1.14$, $Q_3 = 1.23$
* **Illumination Non-Uniformity**: $\text{Median} = 0.49$, $Q_3 = 0.64$

### Decision Rules
1. **Rule 1: High-Noise Underexposure**
   * *Condition*: $\sigma_n \ge 1.23$ and $\mu_Y < 19.32$
   * *Assigned Method*: `denoising_enhancement` (Bilateral + CLAHE)
   * *Parameter Adaptation*: $\sigma_{\text{color}} = \text{clip}(40.0 + 20.0 \cdot (\sigma_n - 1.23), 35.0, 70.0)$
2. **Rule 2: Severe Underexposure & Compressed Range**
   * *Condition*: $\mu_Y \le 8.19$ and $\text{DR} \le 30.0$
   * *Assigned Method*: `multi_scale_retinex` ($\sigma \in \{15, 80, 250\}$)
3. **Rule 3: Severe Illumination Non-Uniformity**
   * *Condition*: $\text{INU} \ge 0.64$
   * *Assigned Method*: `multi_scale_retinex`
4. **Rule 4: Uniform Low-Contrast Underexposure**
   * *Condition*: $\sigma_Y < 10.15$ and $\sigma_n < 1.23$
   * *Assigned Method*: `clahe_enhancement`
   * *Parameter Adaptation*: $\text{clip\_limit} = \text{clip}(2.0 + (30.0 - \text{DR}) / 25.0, 1.5, 3.2)$
5. **Rule 5: Mild Underexposure / Clean Signal**
   * *Condition*: $\mu_Y \ge 13.70$, $\sigma_Y \ge 10.15$, $\sigma_n < 1.14$
   * *Assigned Method*: `gamma_correction`
   * *Parameter Adaptation*: $\gamma = \text{clip}(0.35 + 0.015 \cdot \mu_Y, 0.45, 0.70)$
6. **Rule 6: Moderate Underexposure (Balanced Fallback)**
   * *Condition*: All other conditions
   * *Assigned Method*: `clahe_enhancement` ($\text{clip\_limit}=2.0$, $8 \times 8$ grid)

---

## 16. Experimental Setup & Evaluation Protocol

* **Evaluation Subset**: `lol_dataset/eval15/` (15 pairs, $600 \times 400$ PNG).
* **Total Executions**: 15 images $\times$ 7 algorithms = 105 runs.
* **Objective Metrics**:
  * **PSNR (dB)**: Full-reference pixel-level reconstruction accuracy ($\text{data\_range}=255.0$).
  * **SSIM**: Full-reference structural similarity ($\text{data\_range}=255.0$, 3 channels).
  * **Output Noise ($\sigma_n$)**: High-frequency residual standard deviation.
  * **Computational Latency (ms)**: Pure algorithm execution time measured via `time.perf_counter()`.

---

## 17. Empirical Results

All values in the following tables are derived directly from [`results/metrics/eval15_summary.csv`](../results/metrics/eval15_summary.csv), [`results/metrics/eval15_pairwise_comparison.csv`](../results/metrics/eval15_pairwise_comparison.csv), and [`results/metrics/eval15_adaptive_decisions.csv`](../results/metrics/eval15_adaptive_decisions.csv).

### Table 1: Method vs. Full-Reference Reconstruction Fidelity (PSNR, dB)
| Method | Mean PSNR | Std Dev | Median | Min | Max |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Multi-Scale Retinex (MSR) | **15.029** | 3.302 | 14.709 | 9.642 | 20.834 |
| Single-Scale Retinex (SSR) | 14.830 | 3.283 | 14.718 | 9.677 | 20.865 |
| Histogram Equalization (HE) | 14.236 | 3.138 | 14.991 | 9.139 | 19.859 |
| Gamma Correction ($\gamma=0.5$) | 12.054 | 3.603 | 11.361 | 6.899 | 18.341 |
| **Proposed Adaptive (Ours)** | **10.240** | **3.083** | **9.642** | **5.736** | **15.598** |
| CLAHE ($2.0$) | 9.627 | 3.109 | 8.581 | 5.635 | 15.598 |
| Denoise + CLAHE | 9.388 | 3.021 | 8.322 | 5.564 | 15.107 |

### Table 2: Method vs. Structural Similarity Index (SSIM)
| Method | Mean SSIM | Std Dev | Median | Min | Max |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Single-Scale Retinex (SSR) | **0.585** | 0.123 | 0.615 | 0.361 | 0.763 |
| Multi-Scale Retinex (MSR) | 0.577 | 0.123 | 0.618 | 0.360 | 0.761 |
| Gamma Correction ($\gamma=0.5$) | 0.576 | 0.124 | 0.610 | 0.369 | 0.739 |
| **Proposed Adaptive (Ours)** | **0.429** | **0.128** | **0.440** | **0.221** | **0.655** |
| Histogram Equalization (HE) | 0.425 | 0.102 | 0.418 | 0.255 | 0.601 |
| CLAHE ($2.0$) | 0.400 | 0.120 | 0.412 | 0.222 | 0.655 |
| Denoise + CLAHE | 0.375 | 0.116 | 0.392 | 0.202 | 0.606 |

### Table 3: Method vs. Computational Execution Latency
| Method | Mean Latency (ms) | Std Dev | Median (ms) | Min (ms) | Max (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Gamma Correction ($\gamma=0.5$) | **0.6** | 0.1 | 0.6 | 0.4 | 0.8 |
| Histogram Equalization (HE) | 0.7 | 0.2 | 0.6 | 0.5 | 1.2 |
| CLAHE ($2.0$) | 0.8 | 0.2 | 0.8 | 0.5 | 1.0 |
| Denoise + CLAHE | 3.0 | 0.2 | 3.0 | 2.7 | 3.4 |
| Single-Scale Retinex (SSR) | 42.5 | 0.9 | 42.2 | 41.4 | 44.9 |
| **Proposed Adaptive (Ours)** | **61.9** | **67.2** | **28.9** | **26.2** | **197.9** |
| Multi-Scale Retinex (MSR) | 163.7 | 3.4 | 164.0 | 157.4 | 170.8 |

### Table 4: Method vs. Estimated Output Noise Level ($\sigma_n$)
| Method | Mean Output Noise ($\sigma_n$) | Std Dev | Median | Min | Max |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Denoise + CLAHE | **1.035** | 0.153 | 0.983 | 0.730 | 1.306 |
| CLAHE ($2.0$) | 3.299 | 0.553 | 3.265 | 2.021 | 4.212 |
| Gamma Correction ($\gamma=0.5$) | 3.362 | 0.777 | 3.179 | 2.288 | 5.323 |
| Proposed Adaptive (Ours) | 4.414 | 4.356 | 3.102 | 1.110 | 14.310 |
| Single-Scale Retinex (SSR) | 9.864 | 3.498 | 9.910 | 4.829 | 16.547 |
| Multi-Scale Retinex (MSR) | 10.288 | 3.437 | 10.148 | 5.205 | 16.682 |
| Histogram Equalization (HE) | 13.611 | 3.742 | 13.336 | 8.386 | 19.931 |

### Table 5: Paired Image-Level Comparison (Adaptive vs. Each Baseline, $n=15$)
| Comparison | Mean $\Delta$PSNR | Median $\Delta$PSNR | PSNR Win / Tie / Loss | Mean $\Delta$SSIM | Median $\Delta$SSIM | SSIM Win / Tie / Loss | SSIM Win % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Adaptive vs. Gamma Correction | $-1.814$ dB | $-2.300$ dB | $2\ /\ 0\ /\ 13$ | $-0.1471$ | $-0.1948$ | $1\ /\ 0\ /\ 14$ | $6.7\%$ |
| Adaptive vs. Histogram Equalization | $-3.996$ dB | $-3.429$ dB | $3\ /\ 0\ /\ 12$ | $+0.0039$ | $+0.0031$ | $7\ /\ 1\ /\ 7$ | $46.7\%$ |
| Adaptive vs. CLAHE | $+0.612$ dB | $0.000$ dB | $5\ /\ 4\ /\ 6$ | $+0.0295$ | $0.0000$ | $6\ /\ 5\ /\ 4$ | $40.0\%$ |
| **Adaptive vs. Denoise + CLAHE** | **$+0.852$ dB** | **$+0.171$ dB** | **$9\ /\ 4\ /\ 2$** | **$+0.0541$** | **$+0.0206$** | **$9\ /\ 6\ /\ 0$** | **$60.0\%$** |
| Adaptive vs. Single-Scale Retinex | $-4.591$ dB | $-4.836$ dB | $4\ /\ 0\ /\ 11$ | $-0.1561$ | $-0.2252$ | $2\ /\ 3\ /\ 10$ | $13.3\%$ |
| Adaptive vs. Multi-Scale Retinex | $-4.790$ dB | $-5.011$ dB | $2\ /\ 3\ /\ 10$ | $-0.1478$ | $-0.2107$ | $2\ /\ 3\ /\ 10$ | $13.3\%$ |

### Table 6: Adaptive Strategy Selection Distribution Across eval15
| Selected Enhancement Strategy | Image Count | Percentage | Targeted Degradation Profile |
| :--- | :---: | :---: | :--- |
| `clahe_enhancement` | 6 | 40.0% | Moderate Underexposure / Uniform Low-Contrast |
| `denoising_enhancement` | 5 | 33.3% | High Sensor Noise ($\sigma_n \ge 1.23$) |
| `multi_scale_retinex` | 3 | 20.0% | Severe Illumination Non-Uniformity / Compressed Range |
| `gamma_correction` | 1 | 6.7% | Mild Underexposure with Clean Signal |
| **Total** | **15** | **100.0%** | **4 Distinct Profiles Activated** |

---

## 18. Comparative Analysis & Trade-Offs

The empirical data illustrates distinct trade-offs across the six evaluated paradigms:

1. **Reconstruction Metrics vs. Ground-Truth Bias**:
   The LOL benchmark ground-truth reference images have an elevated mean brightness ($\mu \approx 110 - 135$). Algorithms that aggressively shift the entire luminance plane upward (e.g., Multi-Scale Retinex: out mean $\mu = 131.4$, Histogram Equalization: out mean $\mu = 130.8$) achieve higher average PSNR ($15.03$ dB and $14.24$ dB) mechanically by minimizing mean squared error relative to bright references.
2. **Noise Amplification Behavior**:
   Histogram Equalization produced substantially higher estimated output noise ($\sigma_n = 13.61$), indicating considerable noise amplification compared with the other enhancement methods. Retinex methods similarly elevated noise ($\sigma_n \approx 10.3$). The Adaptive System produced substantially lower estimated output noise than Histogram Equalization, with an average of $\sigma_n = 4.41$, although Denoise + CLAHE, CLAHE, and Gamma produced lower average noise estimates.
3. **Performance Against Local Contrast Baselines**:
   Against fixed CLAHE, the Adaptive System achieved positive gains ($\Delta\text{PSNR} = +0.612$ dB, $\Delta\text{SSIM} = +0.0295$). Against Denoise+CLAHE, it achieved a 60.0% win rate on PSNR and 60.0% win rate on SSIM with zero losses.
4. **Computational Efficiency**:
   Point operations (Gamma, HE, CLAHE) execute in under $1.0$ ms. The Adaptive System averaged 61.9 ms per image, approximately 2.6× lower than the 163.7 ms average measured for always applying MSR. However, Adaptive latency varies with the selected branch and can approach or exceed fixed MSR latency for individual images.

---

## 19. Qualitative Visual Analysis

Visual inspection of the multi-scene comparison panel ([`results/comparisons/eval15_visual_comparison.png`](../results/comparisons/eval15_visual_comparison.png)) highlights how the adaptive engine handles distinct degradation profiles:

* **Scene `1.png` (Severe Spatial Illumination Non-Uniformity, $\text{INU} = 0.822$)**:
  The scene features a directional light source illuminating a shelf, leaving surrounding areas deeply shadowed. Global methods (HE, Gamma) produce extreme blowout on the bright cabinet while leaving the dark corners murky. The Adaptive Engine correctly selected `multi_scale_retinex`, compressing the dynamic range to normalize the illumination field without local blowout.
* **Scene `111.png` (Moderate Underexposure, Balanced Fallback)**:
  Lacking extreme noise ($\sigma_n = 1.15$) or severe spatial gradients ($\text{INU} = 0.338$), the engine routed the image to standard `clahe_enhancement`. This enhanced local edge contrast without the computational penalty or color desaturation of Retinex.
* **Scene `780.png` (High Sensor Noise, $\sigma_n = 1.24$, Underexposed Interior)**:
  Direct application of HE or MSR amplified sensor grain into severe visual artifacts. The Adaptive Engine correctly triggered `denoising_enhancement` (Bilateral + CLAHE), smoothing the background floor and ceiling grain while enhancing doorframe edges.

---

## 20. Discussion & Research Findings

### Addressing the Research Question
*"Can a content-aware adaptive selection of classical image-processing techniques provide more consistent low-light image enhancement than applying one fixed enhancement technique to all images?"*

**The experimental evidence provides partial and qualified support for the adaptive hypothesis:**
1. **Consistency (Lower Variance)**: The Adaptive System exhibited a PSNR standard deviation of **$3.08$ dB**, lower than fixed Gamma Correction ($3.60$ dB), SSR ($3.28$ dB), and MSR ($3.30$ dB), indicating relatively stable PSNR performance across the evaluated scenes.
2. **Superiority Over Local-Contrast Baselines**: On average, the Adaptive System achieved higher PSNR and SSIM than fixed CLAHE and Denoise + CLAHE, with the strongest pairwise advantage observed against Denoise + CLAHE.
3. **Retinex Dominance in Global Metrics**: Fixed Retinex methods (MSR and SSR) achieved higher average PSNR and SSIM across `eval15`. The adaptive system did not outperform Retinex on global averages because Retinex aggressively shifts intensities toward the bright LOL ground-truth target. However, Retinex incurs a substantial computational penalty ($163.7$ ms) and causes substantial noise amplification ($\sigma_n = 10.29$).

---

## 21. Limitations & Threats to Validity

1. **Sample Size**: The held-out benchmark `eval15` contains 15 paired scenes. While statistically informative and strictly held out from development, broad generalization claims across arbitrary real-world imaging conditions require caution.
2. **Development-Set Boundary Calibration**: The decision boundaries ($Q_1, \text{Median}, Q_3$) were empirically derived from `our485`. While effective, hard threshold boundaries can introduce edge-case sensitivity near boundary values.
3. **Information Loss**: Classical algorithms operate by redistributing or filtering existing pixel intensities; they cannot hallucinate information physically absent from zero-photon regions.
4. **Computational Overhead**: Feature analysis introduces additional computational overhead; the total Adaptive pipeline averaged 61.9 ms per image in the final benchmark, while individual branch execution times varied substantially.
5. **Full-Reference Metric Limitations**: PSNR and SSIM reward brightness proximity to a specific ground truth and do not fully quantify perceptual naturalness or artistic preference.

---

## 22. Conclusion
This project developed and evaluated an explainable, content-aware adaptive framework for low-light image enhancement. By extracting quantitative degradation indicators including mean luminance, contrast, dynamic range, estimated noise, and illumination non-uniformity, the system dynamically maps input scenes to an appropriate classical enhancement pipeline. Evaluation on the held-out LOL eval15 subset showed that fixed Retinex methods achieved higher average full-reference reconstruction scores, while the Adaptive System achieved higher average PSNR and SSIM than CLAHE and Denoise + CLAHE and exhibited relatively low PSNR variability across the evaluated scenes. It also produced substantially lower estimated output noise than Histogram Equalization, although it did not achieve the lowest noise or highest reconstruction scores overall. These results provide qualified support for content-aware adaptive selection as an interpretable engineering strategy for condition-dependent low-light enhancement.

---

## 23. Future Scope
1. **Soft / Fuzzy Decision Boundaries**: Replace hard threshold logic with fuzzy membership functions or probabilistic gating to eliminate boundary switching jitter.
2. **Video Sequence Extension**: Integrate temporal smoothing across consecutive video frames to prevent inter-frame flicker in video surveillance applications.
3. **Perceptual Metric Optimization**: Incorporate no-reference perceptual metrics (e.g., NIQE, BRISQUE) into the objective decision loop.
4. **Hybrid Classical-Neural Pipelines**: Explore lightweight neural gating networks to parameterize classical filters in real time.
5. **Hardware Acceleration**: Implement OpenCL / SIMD vectorization for real-time edge embedded deployment at $>60$ FPS.

---

## 24. References
1. **Land, E. H.** (1977). "The Retinex theory of color vision." *Scientific American*, 237(6), 108–129.
2. **Jobson, D. J., Rahman, Z., & Woodell, G. A.** (1997). "A multiscale retinex for bridging the gap between color images and the human observation of scenes." *IEEE Transactions on Image Processing*, 6(7), 965–976. [DOI: 10.1109/83.597272](https://doi.org/10.1109/83.597272).
3. **Pizer, S. M., Amburn, E. P., Austin, J. D., et al.** (1987). "Adaptive histogram equalization and its variations." *Computer Vision, Graphics, and Image Processing*, 39(3), 355–368.
4. **Zuiderveld, K.** (1994). "Contrast Limited Adaptive Histogram Equalization." *Graphics Gems IV*, Academic Press, 474–485.
5. **Immerkaer, J.** (1996). "Fast noise variance estimation." *Computer Vision and Image Understanding*, 64(2), 300–302. [DOI: 10.1006/cviu.1996.0060](https://doi.org/10.1006/cviu.1996.0060).
6. **Tomasi, C., & Manduchi, R.** (1998). "Bilateral filtering for gray and color images." *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*, 839–846.
7. **Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P.** (2004). "Image quality assessment: from error visibility to structural similarity." *IEEE Transactions on Image Processing*, 13(4), 600–612. [DOI: 10.1109/TIP.2003.819861](https://doi.org/10.1109/TIP.2003.819861).
8. **Wei, C., Wang, W., Yang, W., & Liu, J.** (2018). "Deep Retinex Decomposition for Low-Light Enhancement." *Proceedings of the British Machine Vision Conference (BMVC)*, 1–12.
