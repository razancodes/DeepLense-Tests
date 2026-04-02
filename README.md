# ML4SCI DeepLense — GSoC 2026 Application

**Project:** DeepLense Foundation Model for Gravitational Lensing [DEEPLENSE4]  
**Contributor:** Muhammed Razan  
**Mentors:** [Michael Toomey](https://www.michael-toomey.com/) · [Sergei Gleyzer](https://sergeigleyzer.com/) · [Pranath Reddy](https://scholar.google.com/citations?user=sq-LU5kAAAAJ&hl=en) · Anna Parul · [J Rishi](https://scholar.google.com/citations?user=7Kb5PhsAAAAJ&hl=en)

---
<img width="1280" height="720" alt="image" src="https://github.com/user-attachments/assets/eac297db-986f-4434-9d1e-05a7bd9db575" />

## Overview

This repository contains my test submissions for the **ML4SCI DeepLense GSoC 2026** project. The goal is to build a physics-aware foundation model for dark matter substructure detection in simulated gravitational lensing images. Three experiments are presented, progressing from a strong transfer-learning baseline to a self-supervised foundation model with embedded physics.

Link to the Model Weights: [Google Drive](https://drive.google.com/drive/folders/1aNzzPyFXHWAkzCvKtPckG-IOeGRsyiNH?usp=sharing)

| # | Experiment | Task | Key Result |
|---|---|---|---|
| 1 | [Common Test I — ResNet34](Common_Test_I) | 3-class classification | AUC **0.9947** |
| 2 | [Specific Task VII — PINN](SPECIFIC_TEST_VII) | Physics-guided classification | AUC **0.9958** |
| 3 | [Specific Test IX — MAE Foundation Model](Specific_Test_IX) | Self-supervised pre-training + classification + super-resolution | AUC **0.9955** · PSNR **38.97 dB** |

---

## Common Test I — ResNet34 Baseline

**Task:** 3-class classification (`no_sub`, `subhalo`, `vortex`) — 37,500 balanced samples.

### Architecture

**ResNet34** with ImageNet1K pre-trained weights, fine-tuned end-to-end. The final FC head is replaced with a 512→3 layer.

```
Input (B, 3, 224, 224)  [arcsinh-normalized, 3-channel duplicate]
  → Conv1 → MaxPool
  → 4× Residual Stage (64/128/256/512 filters)
  → Global Avg Pool → FC(512→3)
```

**21.8M parameters total.**

### Key Configuration

| Hyperparameter | Value |
|---|---|
| Image size | 224×224 (upsampled for ImageNet compat.) |
| Optimizer | AdamW, LR=1e-4, WD=0.05 |
| LR schedule | Linear warmup (5 ep) → Cosine Annealing |
| Loss | CrossEntropy + Label Smoothing (ε=0.1) |
| AMP |  Mixed precision |
| Preprocessing | arcsinh stretch (a=5.0) + Gaussian noise (σ=0.005) |

The **arcsinh stretch** (`f(x) = ln(a·x + √((a·x)²+1))`) is a physics-motivated preprocessing step that suppresses saturated arcs while amplifying faint substructure signals — key for distinguishing dark matter models.

### Results

| Metric | Value |
|---|---|
| **Best Val AUC** | **0.9947** (Epoch 27) |
| Best Val Accuracy | ~96.9% |
| Best Val Loss | ~0.358 |
| Train AUC (ep 30) | 0.9983 |

Strong convergence with no overfitting observed across 30 epochs, demonstrating effective regularization (label smoothing, weight decay, augmentation).

---

## Specific Task VII — Physics-Informed Neural Network

**Task:** Same 3-class classification, but with a **differentiable physics constraint** embedded in the architecture.  

### Architecture

The PINN extends ResNet34 with a **SIS-ansatz physics branch** and a **differentiable InverseLensLayer**:

```
Input image (1-ch, 150×150)
    │
    ▼
ResNet34 backbone  →  feature_map [B, 512, H', W']
    │
    ├─────────────────────────────┐
    ▼                             ▼
Classification Head          Physics Branch (SIS-ansatz)
(GAP → FC → 3 logits)       ↓
                          k(x,y) map  →  deflection field α
                          InverseLensLayer  →  source image S
                          L_phys = TV(S) + Compactness(S)
```

The physics loss enforces the **Singular Isothermal Sphere (SIS)** lensing equation:

```
α(x, y) = k(x, y) · (x, y) / |(x, y)|   [SIS deflection]
S(x, y) = I(x + α_x, y + α_y)            [Source reconstruction]
L_total  = L_ce + λ_phys · L_phys
```

### Two-Stage Training

| Stage | Epochs | Backbone | λ_phys | Goal |
|---|---|---|---|---|
| 1 — Physics Warmup | 20 | Frozen | 0.1 | Converge physics branch |
| 2 — Joint Fine-tuning | 40 | Unfrozen | 0.01 | End-to-end optimization |

### Results

| Metric | Stage 1 Best | Stage 2 Best |
|---|---|---|
| Val AUC | 0.9271 (ep 20) | **0.9958** (ep 40) |
| Val Accuracy | 76.3% | **96.71%** |
| Val Loss | 0.5718 | 0.1348 |

**vs. ResNet34 baseline:**

| Model | Val Acc | Val AUC | Physics-Consistent |
|---|---|---|---|
| ResNet34 (baseline) | ~96.9% | 0.9947 | ✗ |
| **PINN-ResNet34** | 96.71% | **0.9958** | ✓ |

The PINN surpasses the baseline AUC while also producing interpretable **k(x,y) lens-strength maps**, **source-plane reconstructions**, and **deflection vector fields** — grounding predictions in the physical lensing equation.

---

## Specific Test IX — MAE Foundation Model
**Tasks:** A) 3-class dark matter classification (`no_sub`, `cdm`, `axion`) · B) 4× gravitational lens super-resolution  

### Architecture

A custom **Lensing ViT** pre-trained with a **Masked Autoencoder (MAE) + LatentMIM Lite** objective, then fine-tuned on two downstream tasks:

```
[Pre-training]  no_sub images (29,449 unlabeled)
    MAE encoder (2.67M) + decoder (2 blocks)
    LatentMIM: L_total = L_pixel + 0.1 · ||P·z_i − z_j||²

[Downstream A]  Full classification dataset (89,104 samples)
    Encoder + [CLS] token + MLP head  →  3 logits

[Downstream B]  SR pairs: LR (75×75) → HR (150×150)
    Encoder + conv decoder  →  reconstructed HR image
```

| Property | Value |
|---|---|
| Input size | 64×64 (center-cropped) |
| Patch size | 4×4 → 256 tokens |
| ViT embed dim / depth / heads | 192 / 6 / 3 |
| Encoder params | 2.67M |
| Full MAE params | 3.74M |
| LatentMIM projection P | [16, 16], λ=0.1 |

### Masking Ratio Comparison

Two configurations are benchmarked — **MAE-75** (75% masking, 64 visible patches) and **MAE-90** (90% masking, 25 visible patches):

#### Task IX.A — Classification

| Model | MAE-75 AUC | MAE-90 AUC |
|---|---|---|
| Scratch ViT (no pretraining) | 0.9619 | 0.9633 |
| MAE + LatentMIM | **0.9896** | **0.9955** |
| Improvement over scratch | +0.0277 | **+0.0322** |

**Winner: MAE-90 (AUC = 0.9955)** — extreme masking forces global semantic learning, producing richer representations for classification.

Per-class breakdown (MAE-90 best model, 95.94% overall accuracy):

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| `no_sub` | 0.975 | 0.978 | 0.976 |
| `cdm` | 0.925 | 0.956 | 0.940 |
| `axion` | 0.981 | 0.944 | 0.962 |

#### Task IX.B — Super-Resolution

| Metric | MAE-75 | MAE-90 |
|---|---|---|
| **PSNR (dB)** | **38.97** | 29.48 |
| **SSIM** | **0.9949** | 0.9880 |
| MSE | 0.000135 | 0.001153 |

**Winner: MAE-75 (PSNR = 38.97 dB)** — moderate masking preserves local spatial detail critical for pixel-level reconstruction. The 9.5 dB gap between configurations illustrates why task-specific tuning of the masking ratio matters.

### Key Takeaway

> The optimal masking ratio is **task-dependent**: `90%` for classification (semantic understanding), `75%` for super-resolution (spatial fidelity). This motivates a multi-task foundation model approach.

---

## Overall Results at a Glance

| Experiment | Best Model | Metric 1 | Metric 2 | Notes |
|---|---|---|---|---|
| Common Test I — Classification | ResNet34 | AUC **0.9947** | Acc **96.9%** | Transfer learning baseline |
| Specific Task VII — Classification | PINN-ResNet34 | AUC **0.9958** | Acc **96.71%** | SIS physics embedded in architecture |
| Specific Test IX — Classification | MAE-90 ViT | AUC **0.9955** | Acc **95.94%** | Self-supervised pre-training, 3 new classes |
| Specific Test IX — Super-Resolution | MAE-75 ViT | PSNR **38.97 dB** | SSIM **0.9949** | 4× upsampling, MAE encoder fine-tuned ||

---
## Reproducing Results

All notebooks are designed to run on **Google Colab** with a T4 GPU runtime.

### Common Test I & Specific Task VII (PINN)

Both experiments share the same dataset. Download and place it in the notebook's working directory before running:

**Dataset:** [Download from Google Drive](https://drive.google.com/file/d/1ZEyNMEO43u3qhJAwJeBZxFBEYc_pVYZQ/view)

## Environment

| Dependency | Version |
|---|---|
| PyTorch | 2.10.0+cu128 |
| CUDA | Available (Tesla T4 on Colab) |
| timm | Latest |
| scikit-learn | Latest |
| numpy / matplotlib / seaborn | Latest |

All notebooks are self-contained and run on Google Colab with GPU runtime.

---

*By Muhammed Razan — ML4SCI DeepLense GSoC 2026 Application*
