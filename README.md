# EEG Grasp Classification

3-class hand grasp motor imagery decoding from scalp EEG using **EEGNet**, applied to the BCI Competition IV 2020 Track 4 dataset.

**64.97% accuracy**  · session-to-session evaluation

---

## Dataset

**BCI Competition IV 2020 — Track 4** · [osf.io/pq7vb](https://osf.io/pq7vb/)

| | |
|---|---|
| **Classes** | Cylindrical · Spherical · Lumbrical grasp |
| **Subjects** | 15 (right-handed, aged 20–35) |
| **Trials** | 150 per subject per session (50 per class) |
| **EEG** | 60 channels, 250 Hz|
| **Analysis window** | Motor imagery phase: 6–10 s → 1000 samples |
| **Sessions** | 3 sessions per subject, 7 days apart |

---

## Model — EEGNet

EEGNet factorises 2D convolution into three operations that map directly onto EEG signal structure:

```
Input: (B, 1, 60, 1000)
       │
       ▼  BLOCK 1
          Temporal Conv (1×125, F1=8, padding='same') + BatchNorm
          Depthwise Spatial Conv (60×1, D=2) + BatchNorm → ELU → AvgPool(1,4) → Dropout
          output: (B, 16, 1, 250)
       │
       ▼  BLOCK 2
          Depthwise Conv (1×16, padding='same') + Pointwise Conv (1×1, F2=16)
          BatchNorm → ELU → AvgPool(1,8) → Dropout
          output: (B, 16, 1, 31)
       │
       ▼  Flatten → Linear(496 → 3)
```


4,043 parameters - kept stable during training via max-norm constraints on the spatial conv (≤ 1.0) and classifier (≤ 0.25).

---

## Results

| Model | Accuracy (%) |
|---|---|
| CSP + RF / SVM / LDA | 35.36 |
| Filter Bank CSP | 43.06 |
| Deep ConvNet | 62.58 |
| Shallow ConvNet | 63.39 |
| EEG-Transformer | 64.32 |
| **EEGNet** * | **64.97** |
| EEG Conformer | 67.44 |

Baseline model accuracies source:
[Iteratively Calibratable Network for Reliable EEG-Based Robotic Arm Control](https://doi.org/10.1109/tnsre.2024.3434983)


---

## Training

| | |
|---|---|
| **Optimiser** | Adam (lr=1e-3, weight_decay=1e-4) |
| **LR schedule** | ReduceLROnPlateau (factor=0.5, patience=20) |
| **Early stopping** | patience=50, best-weight restoration |
| **Regularisation** | Dropout=0.5, gradient clipping (max_norm=1.0), max-norm constraints |
| **Normalisation** | Per-trial, per-channel z-score over the time axis |
| **Validation** | 5-fold StratifiedKFold |

---


Download `sample01.mat` – `sample15.mat` from [osf.io/pq7vb](https://osf.io/pq7vb/) into:
```
data/training/
data/validation/
```

```bash
# Within-session 5-fold CV
python train.py --subject 
python train.py --all                         # all 15 subjects pooled

# Cross-session: train day  
python train.py --subject  --cross_session
```



