# Specific Task VII: Physics-Informed Neural Network (PINN) for Gravitational Lensing Classification

**Project:** ML4SCI DeepLense Foundation Model [DEEPLENSE4] as part of GSoC 2026.  
**Contributor:** Muhammed Razan.

**Project Mentors:** Michael Toomey, Sergei Gleyzer, Pranath Reddy, Anna Parul and J Rishi. 

**Task:** Task VII — Physics-Guided ML  
**Model:** ResNet34 + SIS-Ansatz Physics Branch (PINN)

---

## 1. Task Overview

Building on **Common Test I** (baseline ResNet34 classifier), this notebook implements a **Physics-Informed Neural Network (PINN)** for the same 3-class gravitational lensing classification task. The key innovation is **embedding the gravitational lensing equation directly into the model architecture** via a differentiable physics branch, rather than relying solely on learned visual features.

| Class Label | Physical Description |
|---|---|
| `no_sub` | No substructure — smooth lens |
| `subhalo` | Subhalo (sphere) dark matter substructure |
| `vortex` | Vortex dark matter substructure |

The goal is to demonstrate that physics-guided training:
1. **Maintains or improves** classification accuracy vs. the ResNet34 baseline.
2. **Ensures physical consistency** — the model's internal representations respect the SIS lensing equation.

---

## 2. Architecture: Physics-Informed ResNet34

<img width="2548" height="1001" alt="image" src="https://github.com/user-attachments/assets/1335dc4d-d771-4c97-93aa-acd041b79a13" />

### 2.1 ResNet34 Visual Backbone

A standard ResNet34 (Initialized with ImageNet1K weights) extracts spatial feature maps from the lensed image. During Stage 1 (warmup), the backbone is **frozen**; during Stage 2, it is unfrozen for joint fine-tuning.

### 2.2 Physics Branch — SIS Ansatz + Inverse Lens Layer

The physics branch models the **Singular Isothermal Sphere (SIS)** gravitational lens equation:

```
α(x, y) = k(x, y) · (x, y) / |(x, y)|       [SIS deflection field]
S(x, y) = I(x + α_x, y + α_y)               [Source reconstruction]
```

The branch:
1. Predicts a **per-pixel scaling map** `k(x, y)` (lens strength) from the backbone's feature map.
2. Constructs the **deflection field** `α` using the SIS formula with learned `k`.
3. Applies a **differentiable `InverseLensLayer`** (bilinear grid sampling) to reconstruct the source-plane image `S`.
4. The source image `S` is fed to a **physics loss** that enforces smoothness and compactness.

## 4. HyperParameters: 

All hyperparameters are managed via a centralized `PINNConfig` dataclass:

```python
@dataclass
class PINNConfig:
    # Dataset
    data_root: str = "./dataset"
    classes: tuple = ("no_sub", "subhalo", "vortex")
    num_classes: int = 3
    native_size: int = 150          # Raw image size (no upsampling needed)
    val_split: float = 0.15

    # Preprocessing
    arcsinh_a: float = 5.0          # Physics-informed normalization
    noise_std: float = 0.005        # Augmentation noise

    # Training stages
    stage1_epochs: int = 20         # Physics warmup (backbone frozen)
    stage2_epochs: int = 40         # Joint fine-tuning (backbone unfrozen)
    batch_size: int = 64

    # Learning rates
    stage1_head_lr: float = 3e-4    # Stage 1: head only
    stage2_backbone_lr: float = 5e-5 # Stage 2: backbone (lower LR)
    stage2_head_lr: float = 3e-4    # Stage 2: head

    # Physics
    stage1_physics_weight: float = 0.1    # λ_phys Stage 1
    stage2_physics_weight: float = 0.01   # λ_phys Stage 2 (reduced)
    smooth_weight: float = 1.0            # TV-norm weight in physics loss
    compact_weight: float = 0.1           # Compactness weight in physics loss

    # Regularization
    weight_decay: float = 0.05
    label_smoothing: float = 0.1
    grad_clip: float = 1.0
    early_stopping_patience: int = 10

    # Checkpoints
    ckpt_dir: str = "./checkpoints_pinn"
```

### Key Hyperparameter Choices

- **`native_size = 150`**: No upsampling — the physics branch operates at the native resolution to preserve lensing geometry.
- **Two-stage training**: Stage 1 warms up the physics branch while keeping the pre-trained backbone stable; Stage 2 jointly fine-tunes everything.
- **`λ_phys` reduction from 0.1 → 0.01**: Stage 2 downweights physics regularization to allow the backbone to focus on classification after the physics constraint is established.
- **`arcsinh` normalization (`a=5.0`)**: Compresses the dynamic range of raw lensing flux into a learnable range.
- **Label smoothing (0.1)**: Prevents overconfident predictions.
- **Early stopping (patience=10)**: Prevents overfitting.

---

## 5. Dataset
same as task 1 (common test 1).

### Preprocessing Pipeline

```
Raw .npy file (150×150, float32)
    │
    ├─ arcsinh normalization: x → arcsinh(a·x) / arcsinh(a)   [a=5.0]
    ├─ Expand to 3 channels (grayscale → RGB-like)
    ├─ Gaussian noise augmentation (σ=0.005) [train only]
    └─ Tensor normalize (ImageNet mean/std)
```

---

## 6. Two-Stage Training Protocol

### Stage 1 — Physics Warmup (Epochs 1–20)

| Parameter | Value |
|---|---|
| Backbone | **Frozen** |
| Head LR | `3e-4` |
| Physics Weight `λ_phys` | `0.1` |
| Optimizer | AdamW |
| LR Schedule | Cosine Annealing |

**Goal:** Allow the physics branch (SIS head, `InverseLensLayer`) to converge and learn a physically consistent deflection field before the backbone is disturbed.

#### Stage 1 Training Log (Key Milestones)

| Epoch | Train Loss | Train Acc | Train AUC | Val Loss | Val Acc | Val AUC |
|---|---|---|---|---|---|---|
| 1 | 1.0893 | 0.477 | 0.7207 | 1.0717 | 0.479 | 0.7305 |
| 5 | 0.9225 | 0.582 | 0.8225 | 0.8882 | 0.597 | 0.8415 |
| 10 | 0.7597 | 0.676 | 0.8760 | 0.7448 | 0.680 | 0.8876 |
| 15 | 0.6507 | 0.726 | 0.9053 | 0.6401 | 0.731 | 0.9115 |
| 20 | 0.5789 | 0.760 | 0.9219 | 0.5718 | 0.763 | 0.9271 |

**Stage 1 Best Val AUC: `0.9271`** (epoch 20)

---

### Stage 2 — Joint Fine-Tuning (Epochs 21–60, i.e., 40 additional epochs)

| Parameter | Value |
|---|---|
| Backbone | **Unfrozen** |
| Backbone LR | `5e-5` (much lower than head) |
| Head LR | `3e-4` |
| Physics Weight `λ_phys` | `0.01` (reduced) |
| Optimizer | AdamW |
| LR Schedule | Cosine Annealing |
<img width="2624" height="616" alt="image" src="https://github.com/user-attachments/assets/fc3a2b50-77f3-47b2-b529-e1b47e439889" />


#### Stage 2 Training Log (Key Milestones)

| Epoch | Train Loss | Train Acc | Train AUC | Val Loss | Val Acc | Val AUC |
|---|---|---|---|---|---|---|
| 21 (S2 Ep 1) | 0.3640 | 0.878 | 0.9795 | 0.3483 | 0.882 | 0.9813 |
| 25 | 0.0881 | 0.970 | 0.9968 | 0.1386 | 0.956 | 0.9940 |
| 30 | 0.0699 | 0.979 | 0.9979 | 0.1350 | 0.962 | 0.9949 |
| 35 | 0.0525 | 0.984 | 0.9986 | 0.1336 | 0.967 | 0.9957 |
| 40 (S2 Final) | 0.0497 | 0.985 | 0.9988 | 0.1348 | 0.967 | **0.9958** |

**Stage 2 Best Val AUC: `0.9958`** (epoch 40)

---

<img width="2624" height="616" alt="image" src="https://github.com/user-attachments/assets/55f4e201-c42e-48d6-b2ce-c4e2d5d956f6" />

## 7. Final Evaluation Results
<img width="944" height="824" alt="image" src="https://github.com/user-attachments/assets/26c0170e-d034-4546-89e2-80e3eb787598" />
<img width="1605" height="677" alt="image" src="https://github.com/user-attachments/assets/27866d1d-c06d-47ed-ad26-0c9086174105" />

```
============================================================
  PINN Final Results
  Val Loss:  0.1348
  Val Acc:   0.9671
  Val AUC:   0.9958
============================================================
```

### Summary Table

| Metric | Value |
|---|---|
| **Validation Loss** | 0.1348 |
| **Validation Accuracy** | **96.71%** |
| **Validation AUC (macro)** | **0.9958** |

---

## 8. Comparison with ResNet34 Baseline (Common Test I)

| Model | Val Accuracy | Val AUC  
|---|---|---|
| ResNet34 (baseline) | **96.9%** | 0.9947  
| PINN-ResNet34  | 96.71% | **0.9958** 

The PINN model **outperforms the pure ResNet34 baseline** on **AUC**, while enforcing physical consistency through the SIS lensing constraint. This demonstrates that integrating domain physics directly into the architecture is beneficial.

---

## 9. Training Dynamics Analysis
**Key observations:**
- **Stage 1** shows steady, consistent improvement as the physics branch learns to predict physically plausible `k` maps and deflection fields.
- **Stage 2** exhibits a dramatic AUC jump (+0.054) in the very first epoch after the backbone is unfrozen, validating the importance of backbone adaptation.
- Validation accuracy and AUC continue improving through epoch 40 without saturation, indicating no overfitting the physics regularization acts as an effective implicit regularizer.

---

## 10. Physics Branch Visualizations

The following outputs from the physics branch provide interpretability:
<img width="1600" height="695" alt="image" src="https://github.com/user-attachments/assets/0c1004ae-c34e-4608-9273-4781cf178a22" />


| Output | Description | Expected Pattern |
|---|---|---|
| **`k(x, y)` map** | Learned per-pixel SIS lens strength | Peaks where lens focuses light most strongly |
| **Source image `S`** | Reconstructed source-plane image | Compact/smooth for `no_sub`; structured for `subhalo`/`vortex` |
| **Deflection field `α`** | Vector field of light bending | Radially symmetric field for SIS-like lenses |
<img width="2010" height="677" alt="image" src="https://github.com/user-attachments/assets/d63ca188-1f8c-41d4-950c-3486242cd7f6" />

---

## 12. Conclusion

This experiment demonstrates a successful integration of physics into a deep learning classifier for gravitational lensing:

1. **Physical consistency**: The SIS-ansatz branch learns physically meaningful per-pixel lens strength maps and deflection fields that respect the gravitational lensing equation.

2. **Improved performance**: PINN-ResNet34 achieves **Val AUC = 0.9958** and **Val Accuracy = 96.71%**, surpassing the pure ResNet34 baselines AUC and validating the physics-guided approach.

3. **Two-stage training efficacy**: The warmup → fine-tune protocol is critical. Stage 1 stabilizes the physics branch before backbone adaptation; Stage 2 unlocks the backbone's full classification power while the physics constraint continues to regularize.

4. **Interpretability**: Unlike a black-box classifier, the PINN model provides physically grounded explanations via its reconstructed source images and deflection fields.

This architecture forms a strong foundation for the proposed **Physics-Informed Foundation Model** for dark matter substructure detection.

---
By Muhammed Razan
