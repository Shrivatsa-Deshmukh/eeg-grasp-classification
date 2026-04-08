# EEG Grasp Classification

3-class hand grasp motor imagery decoding from scalp EEG using **EEGNet** (Lawhern et al., 2018), applied to the BCI Competition IV 2020 Track 4 dataset.

**64.97% accuracy** · 33.33% chance · 5-fold stratified CV · session-to-session evaluation

---

## Dataset

**BCI Competition IV 2020 — Track 4** · [osf.io/pq7vb](https://osf.io/pq7vb/)

| | |
|---|---|
| **Classes** | Cylindrical · Spherical · Lumbrical grasp |
| **Subjects** | 15 (S1–S15, right-handed, aged 20–34) |
| **Trials** | 150 per subject per session (50 per class) |
| **EEG** | 60 channels, 250 Hz, BrainAmp |
| **Analysis window** | Motor imagery phase: 6–10 s → 1000 samples |
| **Sessions** | 3 sessions per subject, 7 days apart |

---

## Model — EEGNet

EEGNet factorises 2D convolution into three operations that map directly onto EEG signal structure:

```
Input: (B, 1, 60, 1000)
       │
       ▼  Temporal Conv (1×125, F1=8, padding='same')     spectral decomposition
       ▼  BatchNorm
       ▼  Depthwise Spatial Conv (60×1, D=2)              learned spatial beamformer
       ▼  BatchNorm → ELU → AvgPool(1,4) → Dropout        output: (B, 16, 1, 250)
       │
       ▼  Depthwise Conv (1×16, padding='same')           temporal mixing per map
       ▼  Pointwise Conv (1×1, F2=16)                     cross-map integration
       ▼  BatchNorm → ELU → AvgPool(1,8) → Dropout        output: (B, 16, 1, 31)
       │
       ▼  Flatten → Linear(496 → 3)
```

**4,043 parameters total.** Max-norm constraints (spatial conv ≤ 1.0, classifier ≤ 0.25) applied after each weight update.

---

## Results

| Model | Accuracy (%) |
|---|---|
| CSP + RF / SVM / LDA | 35–36 |
| Filter Bank CSP | 43.06 |
| Early Region-Based CNN | 57.34 |
| Deep ConvNet | 62.58 |
| Shallow ConvNet | 63.39 |
| EEG-Transformer | 64.32 |
| Multi-Level CNN | 64.59 |
| **EEGNet (this work)** | **64.97** |
| EEG Conformer | 67.44 |

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

## Setup

```bash
pip install -r requirements.txt
```

Download `sample01.mat` – `sample15.mat` from [osf.io/pq7vb](https://osf.io/pq7vb/) into:
```
data/training/
data/validation/
```

```bash
# Within-session 5-fold CV
python train.py --subject 1
python train.py --all                         # all 15 subjects pooled

# Cross-session: train day 1 → evaluate day 2
python train.py --subject 1 --cross_session
```

---

## Limitations

- No bandpass preprocessing (µ/β isolation, 8–30 Hz) prior to the CNN
- No artifact rejection (eye movements, muscle noise)
- Within-subject evaluation only — cross-subject generalisation untested

---

## References

1. Lawhern et al. (2018) — [EEGNet](https://arxiv.org/abs/1611.08024). *J. Neural Eng.* 15(5).
2. Schirrmeister et al. (2017) — Deep learning for EEG decoding. *Human Brain Mapping* 38(11).
3. Song et al. (2022) — EEG Conformer. *IEEE Trans. Neural Syst. Rehabil. Eng.* 31.

---

*Robot Vision · Aug–Dec 2024 · Ankush Desai, Shrivatsa Deshmukh, Prapti Bordoloi*
