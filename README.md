# EEG-Based Hand Grasp Classification — EEGNet

> **3-class motor imagery BCI** — Cylindrical · Spherical · Lumbrical grasps  
> BCI Competition IV 2020, Track 4 · **64.97% accuracy** (our run, EEGNet)

---

## Overview

This project implements **EEGNet** (Lawhern et al., 2018) in PyTorch for decoding hand grasp motor imagery from scalp EEG, applied to the BCI Competition IV 2020 Track 4 dataset. The classifier discriminates between three grasp types — cylindrical, spherical, and lumbrical — from EEG recorded during the motor imagery phase of each trial.

The task is framed as a **session-to-session decoding problem**: the model trains on session 1 (day 1) and is evaluated on session 2 (day 2), recorded 7 days apart. This reflects realistic BCI deployment conditions where EEG signals are non-stationary across time and sessions.

---

## Dataset

**BCI Competition IV 2020 — Track 4: Upper-Limb Motor Imagery**  
[https://osf.io/pq7vb/](https://osf.io/pq7vb/)

| Property | Details |
|---|---|
| **Subjects** | 15 (S1–S15; aged 20–34; all right-handed) |
| **Classes** | Cylindrical grasp · Spherical grasp · Lumbrical grasp |
| **Trials** | 150 per subject per session (3 classes × 50 trials) |
| **Sessions** | 3 sessions per subject, each 7 days apart |
| **EEG system** | BrainAmp (BrainProducts); 60 electrodes, 10-20 system |
| **Sampling rate** | 250 Hz |
| **Notch filter** | 60 Hz (applied at recording) |
| **Electrode impedance** | < 15 kΩ |
| **Reference / Ground** | FCz / Fpz |

### Experimental Protocol

Each trial lasts 10 seconds across three consecutive stages:

```
0 ──────── 3s ──────── 6s ──────────────── 10s
│  Relaxation  │  Preparation/Cue  │  Motor Imagery  │
```

Subjects imagined grasping one of three objects (cup → cylindrical, ball → spherical, card → lumbrical) prompted by a flashing visual cue. Object positions were randomised across trials to minimise spatial artifact confounds.

### Analysis Window

Only the **motor imagery phase** (6–10 s) is used for classification:

```
epo.x[1500:2500, :, :]   →   1000 samples × 60 channels × 150 trials
                                  (4 s at 250 Hz)
```

### Data Structure

```
data/
├── training/       # Session 1 (day 1) — training
│   ├── sample01.mat  …  sample15.mat
└── validation/     # Session 2 (day 2) — cross-session evaluation
    ├── sample01.mat  …  sample15.mat
```

Each `.mat` file contains:

| Field | Shape | Description |
|---|---|---|
| `epo.x` | (2500, 60, 150) | Pre-processed EEG (samples × channels × trials) |
| `epo.y` | (3, 150) | One-hot class labels |
| `epo.fs` | scalar | Sampling frequency — 250 Hz |
| `epo.clab` | (1, 60) | Channel labels |
| `mnt.pos_3d` | (3, 60) | 3D electrode coordinates |

Class label rows: `epo.y[0,:]` = Cylindrical, `epo.y[1,:]` = Spherical, `epo.y[2,:]` = Lumbrical.

---

## Architecture — EEGNet

EEGNet (Lawhern et al., 2018) is a compact depthwise separable CNN purpose-built for EEG-based BCIs. Its core principle is the **factorisation of standard 2D convolution** into three neurophysiologically-motivated operations, which simultaneously reduces parameters and preserves interpretability.

```
Input: (batch, 1, n_channels=60, n_times=1000)
       │
       ▼
╔══════════════════════════════════════════════════════════════╗
║  BLOCK 1 — Temporal Convolution                              ║
║                                                              ║
║  Conv2d(1 → F1=8, kernel=(1,125), padding='same', bias=F)   ║
║  ├─ Learns F1=8 spectral filters over time                   ║
║  ├─ kern_length = sfreq/2 = 125 → frequency resolution 2 Hz ║
║  └─ padding='same' guarantees output_time = input_time       ║
║  BatchNorm2d(8)                                              ║
║  Output: (B, 8, 60, 1000)                                    ║
╚══════════════════════════════════════════════════════════════╝
       │
       ▼
╔══════════════════════════════════════════════════════════════╗
║  BLOCK 1 — Depthwise Spatial Convolution                     ║
║                                                              ║
║  Conv2d(8 → 16, kernel=(60,1), groups=8, bias=F)            ║
║  ├─ D=2 spatial filters per temporal feature map            ║
║  ├─ Collapses electrode axis → learned spatial beamformer   ║
║  └─ max_norm ≤ 1.0 applied post-step                        ║
║  BatchNorm2d(16) → ELU → AvgPool2d(1,4) → Dropout(0.5)     ║
║  Output: (B, 16, 1, 250)                                     ║
╚══════════════════════════════════════════════════════════════╝
       │
       ▼
╔══════════════════════════════════════════════════════════════╗
║  BLOCK 2 — Separable Convolution                             ║
║                                                              ║
║  Depthwise:  Conv2d(16→16, kernel=(1,16), padding='same')   ║
║              ├─ Temporal mixing independently per map        ║
║              └─ padding='same' preserves 250 time points     ║
║  Pointwise:  Conv2d(16→16, kernel=(1,1))                    ║
║              └─ Cross-map feature integration                ║
║  BatchNorm2d(16) → ELU → AvgPool2d(1,8) → Dropout(0.5)     ║
║  Output: (B, 16, 1, 31)                                      ║
╚══════════════════════════════════════════════════════════════╝
       │
       ▼
  Flatten → (B, 496)
       │
       ▼
╔══════════════════════════════════════════════════════════════╗
║  CLASSIFIER                                                  ║
║  Linear(496 → 3)  [max_norm ≤ 0.25 applied post-step]       ║
╚══════════════════════════════════════════════════════════════╝
       │
       ▼
  Logits: (B, 3)
```

### Verified Tensor Shapes

| Stage | Operation | Output shape |
|---|---|---|
| Input | — | (B, 1, 60, 1000) |
| Block 1 temporal conv | kernel=(1,125), `padding='same'` | (B, 8, 60, 1000) |
| Block 1 depthwise conv | kernel=(60,1), no padding | (B, 16, 1, 1000) |
| Block 1 AvgPool | (1,4) | (B, 16, 1, 250) |
| Block 2 depthwise conv | kernel=(1,16), `padding='same'` | (B, 16, 1, 250) |
| Block 2 pointwise conv | kernel=(1,1) | (B, 16, 1, 250) |
| Block 2 AvgPool | (1,8) | (B, 16, 1, 31) |
| Flatten | — | (B, 496) |
| Classifier | Linear | (B, 3) |

### Verified Parameter Count

| Component | Layer | Parameters |
|---|---|---|
| Block 1 | Temporal Conv2d(1,8,(1,125)) | 1,000 |
| Block 1 | BatchNorm2d(8) | 16 |
| Block 1 | Depthwise Conv2d(8,16,(60,1)) | 960 |
| Block 1 | BatchNorm2d(16) | 32 |
| Block 2 | Depthwise Conv2d(16,16,(1,16)) | 256 |
| Block 2 | Pointwise Conv2d(16,16,(1,1)) | 256 |
| Block 2 | BatchNorm2d(16) | 32 |
| Classifier | Linear(496,3) | 1,491 |
| **Total** | | **4,043** |

All parameter counts verified by explicit shape arithmetic (no bias terms in Conv2d layers; bias in classifier only).

### Design Rationale

**Temporal convolution first.** The first convolution operates solely along the time axis with `kern_length = sfreq/2 = 125`. This acts as a bank of learned bandpass filters, decomposing the broadband EEG into spectral components (analogous to µ/β isolation, but data-driven). Applying temporal filtering before spatial filtering ensures each spatial filter operates on spectrally-defined signals rather than raw broadband EEG.

**Depthwise spatial convolution.** `groups=F1` enforces that each temporal feature map receives its own independent set of `D=2` spatial filters. This is equivalent to learning a separate spatial beamformer per spectral band. The `max_norm ≤ 1.0` constraint after each weight update prevents filter explosion — critical when only 50 trials per class are available.

**`padding='same'` throughout.** All temporal convolutions use `padding='same'` (PyTorch ≥ 1.9), which guarantees `output_time = input_time` regardless of kernel size or stride. Manual padding (e.g. `padding=(0, kern_length//2)`) produces correct results only for odd kernel lengths and silently introduces off-by-one errors for even kernels (e.g. `padding=(0,8)` for `kernel=16` gives 251 instead of 250 time points).

**Average pooling as regularisation.** Two `AvgPool` layers decimate the temporal axis by ×4 and ×8 (×32 total: 1000→31), acting as low-pass filters. Without pooling, all 1000 time points would feed the classifier directly, increasing the flat size from 496 to ~16,000 and making overfitting near-certain at 150 trials per subject.

**Separable convolution in Block 2.** Factorising into depthwise + pointwise reduces the parameter count by ~8× versus an equivalent standard Conv2d while retaining the ability to model interactions between spatial feature maps.

**Max-norm constraints.** Applied after every optimiser step via `apply_constraints()`. The projection `w ← w · min(1, c/‖w‖₂)` is a no-op when ‖w‖₂ ≤ c and scales down otherwise. This is a form of implicit regularisation beyond L2 weight decay, constraining the geometry of the weight space rather than just its scale.

---

## Results

| Model | Accuracy (%) |
|---|---|
| CSP + Random Forest | 35.36 |
| CSP + SVM | 36.35 |
| CSP + LDA | 36.01 |
| Filter Bank CSP (FBCSP) | 43.06 |
| Early Region-Based CNN | 57.34 |
| Deep ConvNet | 62.58 |
| Shallow ConvNet | 63.39 |
| EEG-Transformer | 64.32 |
| Multi-Level CNN | 64.59 |
| **EEGNet (this work)** | **64.97** |
| EEG Conformer | 67.44 |

Chance level: **33.33%** (3-class uniform baseline)  
Our result: **+31.6 pp above chance**

### Key Observations

**CSP-based methods fail (~35–43%).** Hand-crafted spatial filters cannot capture the fine-grained spectrotemporal patterns that distinguish cylindrical, spherical, and lumbrical grasp imagery. These three classes involve largely overlapping cortical motor representations; their discriminability lies in subtle temporal dynamics rather than gross spatial differences that CSP is designed to exploit.

**EEGNet (64.97%) outperforms Deep ConvNet (62.58%) and Shallow ConvNet (63.39%)** with 4,043 parameters versus ~170,000+ for those baselines. The inductive biases baked into EEGNet — temporal-first factorisation, depthwise spatial filtering, progressive average pooling — are well-matched to the low-data EEG regime (50 trials/class).

**EEG Conformer (67.44%)** marginally outperforms EEGNet, likely because self-attention can capture long-range temporal dependencies within the 4 s imagery window that fixed-kernel convolution cannot. This comes at a substantially higher parameter and compute cost.

---

## Training Protocol

| Hyperparameter | Value | Rationale |
|---|---|---|
| **Optimiser** | Adam | Adaptive LR; standard for EEG deep learning |
| **Learning rate** | 1e-3 | EEGNet paper default |
| **Weight decay** | 1e-4 | L2 regularisation |
| **Loss** | Cross-entropy | Standard multi-class |
| **Max epochs** | 300 | Hard ceiling; early stopping governs actual length |
| **Batch size** | 32 | — |
| **LR schedule** | ReduceLROnPlateau (factor=0.5, patience=20) | Adapts to actual convergence; preferred over StepLR |
| **Early stopping** | patience=50, best-weight restoration | Restores checkpoint with lowest val loss |
| **Gradient clipping** | max_norm=1.0 | Stabilises early training on small datasets |
| **Dropout** | 0.5 (both blocks) | EEGNet paper default |
| **max_norm (spatial conv)** | 1.0 | Constrains depthwise filter growth |
| **max_norm (classifier)** | 0.25 | Constrains classifier weight growth |
| **Normalisation** | Per-trial, per-channel z-score | Removes amplitude drift; applied over time axis per (channel, trial) |
| **Cross-validation** | 5-fold StratifiedKFold | Preserves 50-trials-per-class balance across folds |
| **Seed** | 42 (global + cuDNN deterministic) | Full reproducibility |

---

## Repository Structure

```
├── train.py               # EEGNet, training loop, CV, cross-session eval
├── requirements.txt       # Python dependencies
├── results/
│   └── cv_results.csv     # Per-fold accuracy
└── README.md
```

---

## Setup & Usage

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download dataset from https://osf.io/pq7vb/
#    Place under:
#      data/training/sample01.mat  …  sample15.mat
#      data/validation/sample01.mat … sample15.mat

# 3a. Within-session 5-fold CV — single subject
python train.py --subject 1

# 3b. Within-session 5-fold CV — all 15 subjects pooled
python train.py --all

# 3c. Cross-session generalisation: train day1 → evaluate day2
#     (reflects the competition protocol)
python train.py --subject 1 --cross_session
```

---

## Limitations & Future Work

- **Session-to-session non-stationarity.** EEG statistics shift substantially between sessions recorded 7 days apart. Domain adaptation methods (e.g. Euclidean alignment, CORAL) could reduce the distribution gap without re-calibration.
- **No bandpass preprocessing.** The dataset applies a 60 Hz notch at recording, but no µ/β band isolation (8–30 Hz) is performed prior to the CNN. A bandpass filter may improve SNR by attenuating slow drift and high-frequency muscle artifacts.
- **Subject-independent generalisation.** The current protocol is within-subject. Leave-one-subject-out cross-validation would test whether learned filters transfer across individuals.
- **Fixed analysis window.** The 6–10 s window is applied uniformly. Adaptive windowing aligned to individual motor imagery onset could improve temporal consistency.
- **No artifact rejection.** Trials contaminated by eye movements or muscle artifacts are not screened. Threshold-based or ICA-based rejection could improve signal quality.

---

## References

-  [EEGNet: A Compact Convolutional Neural Network for EEG-based Brain-Computer Interfaces](https://arxiv.org/abs/1611.08024).
-  **BCI Competition IV 2020 Track 4** — [Dataset & description (OSF)](https://osf.io/pq7vb/)

