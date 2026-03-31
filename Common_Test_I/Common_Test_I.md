# Common Test I: Multi-Class Gravitational Lensing Classification — ResNet34 Experiment Report

**Project:** ML4SCI DeepLense Foundation Model (GSoC 2026)  
**Author:** Muhammed Razan  
**Task:** Common Test I — Multi-Class Classification  
**Model:** ResNet34 (Transfer Learning)

---

## 1. Task Overview

This notebook tackles **Common Test I** of the ML4SCI DeepLense GSoC challenge: a **3-class image classification** task on simulated gravitational lensing images. The goal is to distinguish between three classes of lensing substructure:

| Class Label | Physical Description |
|---|---|
| `no_sub` | No substructure |
| `subhalo` | Subhalo (sphere) dark matter substructure |
| `vortex` | Vortex dark matter substructure |

The images are stored as `.npy` files (NumPy arrays), representing simulated single-channel gravitational lens data from different dark matter models.

---

## 2. Environment and Dependencies

| Dependency | Purpose |
|---|---|
| `PyTorch` | Deep learning framework |
| `torchvision` | Pre-trained models (ResNet34) |
| `timm` | Model utilities |
| `scikit-learn` | Metrics (AUC, confusion matrix, classification report) |
| `matplotlib` / `seaborn` | Visualization |
| `einops` | Tensor manipulation |
| `numpy` | Array operations |
| `tqdm` | Progress bars |

**Hardware:** CUDA GPU (confirmed at runtime: `Device: cuda`)

---

## 3. Configuration

All hyperparameters are managed via a centralized `Config` dataclass:

```python
@dataclass
class Config:
    data_root: str = "./dataset"
    classes: tuple = ("no_sub", "subhalo", "vortex")
    num_classes: int = 3
    img_size: int = 224           # Upsampled for transfer learning
    val_split: float = 0.10

    arcsinh_a: float = 5.0        # Physics-informed normalization
    noise_std: float = 0.005      # Data augmentation noise

    batch_size: int = 64
    num_epochs: int = 30
    lr: float = 1e-4
    weight_decay: float = 0.05
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    label_smoothing: float = 0.1
    grad_clip: float = 1.0
    use_amp: bool = True          # Mixed-precision training (AMP)
```

### Key Hyperparameter Choices

- **Image size 224x224**: Upsampled to be compatible with ImageNet pre-trained ResNet34 weights.
- **`arcsinh` normalization** (`a=5.0`): Physics-informed preprocessing that maps the wide dynamic range of lensing flux values more uniformly, improving feature learning.
- **Label smoothing (0.1)**: Regularization to prevent overconfident predictions.
- **Warmup epochs (5)**: Linear learning rate warmup to stabilize early training.
- **Cosine Annealing LR schedule**: After warmup, LR decays from `1e-4` down to `min_lr=1e-6`.
- **Mixed Precision (AMP)**: Enabled for faster training and reduced memory usage on GPU.
- **Gradient clipping (1.0)**: Prevents gradient explosion.

---

## 4. Dataset

### Statistics

| Split | no_sub | subhalo | vortex | Total |
|---|---|---|---|---|
| (train + val combined) | 12,500 | 12,500 | 12,500 | **37,500** |

The dataset is **perfectly balanced** across all three classes (12,500 samples each), eliminating any class imbalance concerns.

### Dataset Structure

```
dataset/
  train/
    no/       -> maps to "no_sub"
    sphere/   -> maps to "subhalo"
    vort/     -> maps to "vortex"
  val/
    no/
    sphere/
    vort/
```

Files are `.npy` format (NumPy binary arrays). Directory names map to logical class names via: `{"no": "no_sub", "sphere": "subhalo", "vort": "vortex"}`.

### Train/Val Split

The dataset uses a **pre-defined directory split** (`train/` and `val/`) with a ~90/10 proportion. This eliminates data leakage between splits.

---

## 5. Data Preprocessing and Augmentation

### Physics-Informed Preprocessing using ArcSinh

ArcSinh Strech Formula:

\begin{equation}
    f(x) = \ln\left(a \cdot x + \sqrt{(a \cdot x)^2 + 1}\right)
\end{equation}

here $a = 5.0$ was used, it is the softening parameter controlling the degree of compression. This transformation suppresses bright arc saturation while simultaneously amplifying faint substructure signals in the low-intensity regime, making it particularly effective for differentiating dark-matter models that manifest as subtle perturbations to the lensing signal.
<img width="1607" height="709" alt="image" src="https://github.com/user-attachments/assets/3bd9db99-46d7-4114-b5fa-01d119213046" />


### Augmentation Strategy

| Augmentation | Details |
|---|---|
| Gaussian Noise | `noise_std = 0.005` added during training |
| Resize | 224x224 (for ImageNet compatibility) |
| Normalization | Applied after arcsinh |
| geometric augmentation | Random 90° Rotations, Horizontal and Vertical Flips |

### Batch Size

`batch_size = 64` — moderate size facilitating effective gradient estimation.

---

## 6. Model Architecture

**ResNet34** (34-layer Residual Network) with **ImageNet pre-trained weights**, fine-tuned end-to-end on the lensing dataset.

### Architecture Summary

```
Input: (B, 3, 224, 224)
|
+-- Conv1 (7x7, stride 2) -> BN -> ReLU
+-- MaxPool (3x3, stride 2)
|
+-- Layer1: 3x Residual Blocks (64 filters)
+-- Layer2: 4x Residual Blocks (128 filters)
+-- Layer3: 6x Residual Blocks (256 filters)
+-- Layer4: 3x Residual Blocks (512 filters)
|
+-- Global Avg Pool
+-- FC (512 -> 3 classes)
```

**Total parameters (ResNet34):** ~21.8M  
**Modified head:** 512 -> 3 (replacing standard 512->1000 ImageNet head)

### Why ResNet34?

- Lightweight enough for fast iteration (~21M parameters)
- Well-studied architecture with strong performance on ImageNet
- Residual connections prevent vanishing gradients
- Pre-trained weights transfer well to new visual domains

---

## 7. Loss Function

**Cross-Entropy Loss with Label Smoothing** (`label_smoothing=0.1`)

With `eps=0.1` and `K=3` classes, the soft target is `(1-eps)` for the true class and `eps/K` for others. Label smoothing prevents the model from becoming overconfident, acting as a regularizer.

---

## 8. Training Protocol

### Optimizer and Learning Rate Schedule

- **Optimizer:** AdamW (Adam with decoupled weight decay)
- **Initial LR:** `1e-4`
- **Warmup:** Linear warmup over 5 epochs (`2e-5` -> `1e-4`)
- **After warmup:** Cosine Annealing LR decay from `1e-4` to `1e-6`
- **Weight decay:** `0.05`

### Training Loop Details

For each epoch:
1. Forward pass with AMP (`autocast`)
2. Loss computation (cross-entropy + label smoothing)
3. Backward pass with `GradScaler` (AMP gradient scaling)
4. Gradient clipping (max norm = 1.0)
5. Parameter update (AdamW step)
6. Validation pass (no gradient, AUC tracking)
7. Checkpoint saving if new best validation AUC is achieved

---

## 9. Training Metrics — Epoch-by-Epoch Results

<img width="2144" height="616" alt="image" src="https://github.com/user-attachments/assets/730607f2-9578-49bc-9049-12bed7e942f2" />

## 10. Final Results Summary

| Metric | Value |
|---|---|
| **Best Validation AUC** | **0.9947** (Epoch 27) |
| Best Validation Accuracy | ~96.9% |
| Best Validation Loss | ~0.3575 |
| Train AUC (final, epoch 30) | 0.9983 |
| Train Accuracy (final, epoch 30) | 98.4% |
| Train Loss (final, epoch 30) | 0.3239 |

---
<img width="944" height="824" alt="image" src="https://github.com/user-attachments/assets/4f701fd3-44a7-4541-b329-d92852b025d5" />
<img width="1605" height="677" alt="image" src="https://github.com/user-attachments/assets/684deb32-6f8d-476b-b7da-732f1eebc398" />



## 11. Observations and Discussion

### Strengths
1. **High AUC (0.9947):** Near-ceiling performance, demonstrating effective transfer learning from ImageNet to gravitational lensing.
2. **No overfitting:** The regularization strategy (label smoothing, weight decay, noise, AMP) is well-calibrated.
3. **Physics-aware preprocessing:** Arcsinh normalization contributed to faster convergence and higher final performance.
4. **Balanced dataset:** Equal class distribution simplifies metric interpretation.

### Limitations and Areas for Improvement
1. **Single-channel to 3-channel duplication:** Native lensing images are single-channel; channel duplication introduces a slight domain gap vs. ImageNet.
2. **Limited augmentation:** No rotation, flipping, or cropping — physics-preserving augmentations could improve robustness.
3. **No architecture ablation:** Comparing ResNet50, EfficientNet, or ViT would establish whether ResNet34 is truly optimal.
4. **Label smoothing interaction:** With balanced classes, epsilon=0.1 may slightly suppress precision on hard examples.

---

## 12. Relation to the Foundation Model Goal

This experiment establishes the **supervised ResNet34 baseline** (Common Test I) for the ML4SCI DeepLense GSoC 2026 proposal. The AUC of **0.9947** is the reference point against which more advanced approaches are compared:

- **Physics-Informed Neural Networks (PINNs):** Integrate the SIS gravitational lensing equation as a physics constraint
- **Masked Autoencoder (MAE) Foundation Models:** Self-supervised pre-training at 75% and 90% masking ratios, followed by fine-tuning
- **Vision Transformer (ViT):** Attention-based architectures for capturing long-range spatial dependencies

The goal across these experiments is to show that physics-aware and self-supervised methods can match or surpass the strong ResNet34 baseline while generalizing better to new lensing configurations.

---

## 13. Reproducibility Checklist

| Item | Status |
|---|---|
| Random seeds fixed (`SEED=42`) | Yes |
| Pre-split train/val directories | Yes |
| Hyperparameters in `Config` dataclass | Yes |
| Best checkpoint saved to Drive | Yes |
| Figures saved to `./figures/` | Yes |
| GPU device logged at startup | Yes |
| Dataset existence check before extraction | Yes |

---

*Report based on: `Common_Test_I_ResNet34 (1).ipynb`*  
*Best Validation AUC: **0.9947** | Total Epochs: 30 | Dataset: 37,500 balanced lensing images (3 classes)*
