"""
train.py — EEG Grasp Classification via EEGNet
BCI Competition IV 2020, Track 4

Task:    3-class motor imagery (Cylindrical, Spherical, Lumbrical)
Data:    60-channel EEG, 250 Hz, 150 trials/subject/session
Window:  Motor imagery phase 6–10 s → epo.x[1500:2500] → 1000 samples
Model:   EEGNet 

Usage:
    python train.py --subject n              # within-session 5-fold CV
    python train.py --all                    # all 15 subjects pooled
    python train.py --subject n --cross_session  # train day1 → test day2

"""

import argparse
import csv
import os

import numpy as np
import scipy.io
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset

# ── Reproducibility ────────────────────────────────────────────────────────────

torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── Config ─────────────────────────────────────────────────────────────────────

CFG = {
    "train_dir":    "data/training",
    "val_dir":      "data/validation",
    "sfreq":        250,
    "n_channels":   60,
    "n_classes":    3,
    "class_names":  ["Cylindrical", "Spherical", "Lumbrical"],
    "sample_start": 1500,       # 6 s × 250 Hz
    "sample_end":   2500,       # 10 s × 250 Hz  →  1000 samples
    # EEGNet (Lawhern et al. 2018 defaults)
    "F1":           8,
    "D":            2,
    "F2":           16,         # must equal F1 × D
    "kern_length":  125,        # sfreq / 2
    "dropout":      0.5,
    # Training
    "epochs":       300,
    "batch_size":   32,
    "lr":           1e-3,
    "weight_decay": 1e-4,
    "patience":     50,
    "k_folds":      5,
    "results_dir":  "results",
}

# ── EEGNet ─────────────────────────────────────────────────────────────────────

class EEGNet(nn.Module):
    """
    Input:  (B, 1, n_channels, n_times)
    Output: (B, n_classes)

    Block 1 — Temporal conv learns spectral filters (kern_length = sfreq/2).
              Depthwise spatial conv learns D beamformers per spectral band.
    Block 2 — Separable conv (depthwise + pointwise) integrates features.
    Both blocks use padding='same' and AvgPool for temporal decimation.
    Max-norm constraints on spatial conv (≤1.0) and classifier (≤0.25)
    stabilise training on small EEG datasets (50 trials/class).

    Verified shapes for n_channels=60, n_times=1000:
        After Block 1: (B, 16, 1, 250)
        After Block 2: (B, 16, 1,  31)
        Flat:          (B, 496)
        Total params:  4,043
    """

    def __init__(self, n_channels=60, n_times=1000, n_classes=3,
                 F1=8, D=2, F2=16, kern_length=125, dropout=0.5):
        super().__init__()
        assert F2 == F1 * D, f"F2 must equal F1×D={F1*D}, got {F2}"

        self.block1 = nn.Sequential(
            # Temporal: spectral decomposition across time
            nn.Conv2d(1, F1, (1, kern_length), padding="same", bias=False),
            nn.BatchNorm2d(F1),
            # Depthwise spatial: learned beamformer per spectral band
            nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )

        self.block2 = nn.Sequential(
            # Separable: depthwise temporal + pointwise cross-map
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding="same", groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )

        # F2 * floor(n_times / 32) = 16 * 31 = 496 for n_times=1000
        flat = F2 * (n_times // 4 // 8)
        self.classifier = nn.Linear(flat, n_classes)

        # Keep references for max-norm projection
        self._spatial_conv  = self.block1[2]
        self._depthwise_max = 1.0
        self._classifier_max = 0.25

    def apply_constraints(self):
        """Project weights onto max-norm ball after each optimiser step."""
        with torch.no_grad():
            for w, c in [(self._spatial_conv.weight, self._depthwise_max),
                         (self.classifier.weight,    self._classifier_max)]:
                norms = w.norm(2, dim=tuple(range(1, w.dim())), keepdim=True).clamp(min=1e-8)
                w.mul_((norms.clamp(max=c) / norms))

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return self.classifier(x.flatten(1))


# ── Data ───────────────────────────────────────────────────────────────────────

def load_subject(path: str) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Load one subject's .mat file and return normalised EEG tensors.

    Normalisation: per-trial, per-channel z-score over the time axis.
    Removes amplitude drift across trials and sessions without requiring
    any population statistics — valid at single-trial inference time.

    Returns
    -------
    X : FloatTensor (n_trials, 1, n_channels, n_times)
    y : LongTensor  (n_trials,)  0=Cyl, 1=Sph, 2=Lum
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"'{path}' not found.\n"
            "Download from https://osf.io/pq7vb/ into data/training/ and data/validation/"
        )

    mat    = scipy.io.loadmat(path, squeeze_me=False)
    eeg    = mat["epo"]["x"][0][0].astype(np.float64)  # (2500, 60, 150)
    labels = mat["epo"]["y"][0][0].astype(np.int32)    # (3, 150)

    # Validate sampling frequency
    fs = int(np.asarray(mat["epo"]["fs"][0][0]).flat[0])
    if fs != CFG["sfreq"]:
        raise ValueError(f"Expected {CFG['sfreq']} Hz, got {fs} Hz in '{path}'")

    # Extract motor imagery window: epo.x[1500:2500]
    X = eeg[CFG["sample_start"]:CFG["sample_end"], :, :]  # (1000, 60, 150)

    # Per-trial, per-channel z-score  (stats over 1000 time points)
    mu  = X.mean(axis=0, keepdims=True)
    sig = X.std(axis=0,  keepdims=True).clip(min=1e-8)
    X   = (X - mu) / sig

    # (1000, 60, 150) → (150, 1, 60, 1000)
    X = torch.from_numpy(X).float().permute(2, 1, 0).unsqueeze(1)
    y = torch.from_numpy(np.argmax(labels, axis=0)).long()

    return X, y


def load_subjects(data_dir: str, subject_ids: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    all_X, all_y = [], []
    for sid in subject_ids:
        X, y = load_subject(os.path.join(data_dir, f"sample{sid:02d}.mat"))
        counts = torch.bincount(y, minlength=CFG["n_classes"]).tolist()
        print(f"  S{sid:02d}: {len(y)} trials [Cyl={counts[0]}, Sph={counts[1]}, Lum={counts[2]}]")
        all_X.append(X)
        all_y.append(y)
    return torch.cat(all_X), torch.cat(all_y)


# ── Training ───────────────────────────────────────────────────────────────────

def train(model, train_loader, val_loader):
    """
    Train with Adam + ReduceLROnPlateau + early stopping.
    Returns best validation accuracy (restored from best checkpoint).
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG["lr"],
                                 weight_decay=CFG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=20, min_lr=1e-6
    )

    best_loss, best_state, patience_count = float("inf"), None, 0

    for epoch in range(CFG["epochs"]):

        # Train
        model.train()
        correct, total = 0, 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(X), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            model.apply_constraints()
            correct += (model(X).argmax(1) == y).sum().item()
            total   += y.size(0)
        train_acc = correct / total

        # Validate
        model.eval()
        val_loss, vc, vt = 0.0, 0, 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                logits    = model(X)
                val_loss += criterion(logits, y).item()
                vc       += (logits.argmax(1) == y).sum().item()
                vt       += y.size(0)
        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        if (epoch + 1) % 50 == 0:
            print(f"    ep {epoch+1:3d} | train {train_acc*100:.1f}% | "
                  f"val {vc/vt*100:.1f}% | loss {val_loss:.4f} | "
                  f"lr {optimizer.param_groups[0]['lr']:.1e}")

        # Early stopping
        if val_loss < best_loss - 1e-4:
            best_loss, patience_count = val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_count += 1
            if patience_count >= CFG["patience"]:
                print(f"    Early stop at epoch {epoch+1}")
                break

    # Restore best checkpoint
    model.load_state_dict(best_state)
    model.eval()
    bc, bt = 0, 0
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            bc += (model(X).argmax(1) == y).sum().item()
            bt += y.size(0)
    return bc / bt


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate(model, loader):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for X, y in loader:
            preds.append(model(X.to(device)).argmax(1).cpu())
            labels.append(y)
    preds  = torch.cat(preds).numpy()
    labels = torch.cat(labels).numpy()

    print(classification_report(labels, preds,
                                 target_names=CFG["class_names"], digits=3))
    cm    = confusion_matrix(labels, preds)
    names = CFG["class_names"]
    w     = max(len(n) for n in names) + 2
    print("  Confusion matrix (rows=true, cols=predicted):")
    print("  " + "".join(f"{n:>{w}}" for n in names))
    for i, row in enumerate(cm):
        print(f"  {names[i]:<{w}}" + "".join(f"{v:>{w}}" for v in row))
    print()
    return float((preds == labels).mean())


# ── Cross-validation ───────────────────────────────────────────────────────────

def run_cv(subject_ids: list[int]):
    print(f"\n{'='*55}\n  Within-session CV | subjects: {subject_ids}\n{'='*55}")
    X, y = load_subjects(CFG["train_dir"], subject_ids)

    skf     = StratifiedKFold(CFG["k_folds"], shuffle=True, random_state=42)
    results = []

    for fold, (tr, val) in enumerate(skf.split(X, y.numpy())):
        print(f"\n  Fold {fold+1}/{CFG['k_folds']} | train={len(tr)}, val={len(val)}")

        train_loader = DataLoader(TensorDataset(X[tr], y[tr]),
                                  batch_size=CFG["batch_size"], shuffle=True)
        val_loader   = DataLoader(TensorDataset(X[val], y[val]),
                                  batch_size=CFG["batch_size"])

        model   = EEGNet(**{k: CFG[k] for k in
                            ("n_channels","n_classes","F1","D","F2","kern_length","dropout")},
                         n_times=CFG["sample_end"] - CFG["sample_start"]).to(device)
        val_acc = train(model, train_loader, val_loader)

        print(f"\n  Fold {fold+1} evaluation:")
        acc = evaluate(model, val_loader)
        results.append({"fold": fold+1, "val_acc": acc})

    # Summary
    accs = [r["val_acc"] for r in results]
    print(f"\n{'='*55}")
    print(f"  {'Fold':<8} {'Val Acc':>10}")
    print(f"  {'─'*20}")
    for r in results:
        print(f"  {r['fold']:<8} {r['val_acc']*100:>9.2f}%")
    print(f"  {'─'*20}")
    print(f"  {'Mean':<8} {np.mean(accs)*100:>9.2f}%")
    print(f"  {'Std':<8} {np.std(accs)*100:>9.2f}%")
    print(f"  {'Chance':<8} {'33.33%':>10}")
    print(f"{'='*55}")

    os.makedirs(CFG["results_dir"], exist_ok=True)
    out = os.path.join(CFG["results_dir"], "cv_results.csv")
    with open(out, "w", newline="") as f:
        csv.DictWriter(f, ["fold","val_acc"]).writerows(results)
    print(f"\n  Saved → {out}")


def run_cross_session(subject_id: int):
    print(f"\n{'='*55}\n  Cross-session | S{subject_id:02d} | train=day1, test=day2\n{'='*55}")

    X_train, y_train = load_subject(
        os.path.join(CFG["train_dir"], f"sample{subject_id:02d}.mat"))
    X_val,   y_val   = load_subject(
        os.path.join(CFG["val_dir"],   f"sample{subject_id:02d}.mat"))

    train_loader = DataLoader(TensorDataset(X_train, y_train),
                              batch_size=CFG["batch_size"], shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val, y_val),
                              batch_size=CFG["batch_size"])

    model = EEGNet(**{k: CFG[k] for k in
                      ("n_channels","n_classes","F1","D","F2","kern_length","dropout")},
                   n_times=CFG["sample_end"] - CFG["sample_start"]).to(device)
    train(model, train_loader, val_loader)

    print(f"\n  Cross-session evaluation (S{subject_id:02d}):")
    acc = evaluate(model, val_loader)
    print(f"  Accuracy: {acc*100:.2f}%")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--subject", type=int, metavar="N")
    group.add_argument("--all",     action="store_true")
    parser.add_argument("--cross_session", action="store_true")
    args = parser.parse_args()

    if args.cross_session:
        if not args.subject:
            parser.error("--cross_session requires --subject N")
        run_cross_session(args.subject)
    elif args.subject:
        run_cv([args.subject])
    else:
        run_cv(list(range(1, 16)))
