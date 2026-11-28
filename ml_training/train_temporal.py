"""
MindTrack AI v5 — Mamba SSM Temporal Model Training
Trains Mamba-style Selective State Space Model on temporal feature sequences
extracted from RAVDESS + CREMA-D + TESS audio datasets (all freely available).

The temporal model learns to classify emotions from time-varying audio features
(energy, spectral centroid, pitch, ZCR, etc.) extracted over sliding windows.

Usage:
  # Local (with GPU):
  python train_temporal.py --ravdess ../data/datasets/ravdess --cremad ../data/datasets/cremad --tess ../data/datasets/tess

  # Or use the Colab notebook: ml_training/colab_train_temporal.ipynb
"""
import os
import sys
import json
import glob
import math
import logging
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import librosa
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

EMOTION_LABELS = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
NUM_CLASSES = len(EMOTION_LABELS)
SEQ_LEN = 10  # temporal windows per audio clip
FEATURE_DIM = 7  # features per window (matches inference code)


# ─── Model Architecture (must match backend/app/models/temporal_model.py) ───

class SelectiveScanSSM(nn.Module):
    """Mamba-style Selective Scan State Space Model."""

    def __init__(self, d_model=64, state_dim=16, dt_rank=8, d_inner=32):
        super().__init__()
        self.d_model = d_model
        self.state_dim = state_dim
        self.dt_rank = dt_rank
        self.d_inner = d_inner

        self.in_proj = nn.Linear(d_model, d_inner * 2)
        self.A_log = nn.Parameter(torch.randn(d_inner, state_dim))
        self.D = nn.Parameter(torch.ones(d_inner))
        self.x_proj = nn.Linear(d_inner, dt_rank + state_dim * 2, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)
        self.out_proj = nn.Linear(d_inner, d_model)
        self.norm = nn.LayerNorm(d_inner)
        self._init_weights()

    def _init_weights(self):
        A = torch.arange(1, self.state_dim + 1, dtype=torch.float32).unsqueeze(0).repeat(self.d_inner, 1)
        self.A_log.data = torch.log(A)
        nn.init.uniform_(self.dt_proj.bias, -4.0, -2.0)

    def forward(self, x):
        batch, seq_len, _ = x.shape
        xz = self.in_proj(x)
        x_in, z = xz.chunk(2, dim=-1)
        x_dbl = self.x_proj(x_in)
        dt, B, C = x_dbl.split([self.dt_rank, self.state_dim, self.state_dim], dim=-1)
        dt = torch.softplus(self.dt_proj(dt))
        A = -torch.exp(self.A_log)
        y = self._selective_scan(x_in, dt, A, B, C)
        y = self.norm(y)
        y = y * torch.silu(z)
        y = self.out_proj(y)
        return y

    def _selective_scan(self, u, dt, A, B, C):
        batch, seq_len, d_inner = u.shape
        h = torch.zeros(batch, d_inner, self.state_dim, device=u.device, dtype=u.dtype)
        outputs = []
        for t in range(seq_len):
            dt_t = dt[:, t, :].unsqueeze(-1)
            A_bar = torch.exp(A.unsqueeze(0) * dt_t)
            B_t = B[:, t, :].unsqueeze(1).expand(-1, d_inner, -1)
            B_bar = dt_t * B_t
            u_t = u[:, t, :].unsqueeze(-1)
            h = A_bar * h + B_bar * u_t
            C_t = C[:, t, :].unsqueeze(1).expand(-1, d_inner, -1)
            y_t = (C_t * h).sum(dim=-1) + self.D * u[:, t, :]
            outputs.append(y_t)
        return torch.stack(outputs, dim=1)


class MambaTemporalModel(nn.Module):
    """Full Mamba-based temporal analysis model."""

    def __init__(self, input_dim=FEATURE_DIM, d_model=64, n_layers=4, state_dim=16, d_inner=32, num_classes=NUM_CLASSES):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, 100, d_model) * 0.02)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "ssm": SelectiveScanSSM(d_model, state_dim, d_inner=d_inner),
                "norm": nn.LayerNorm(d_model),
                "ffn": nn.Sequential(
                    nn.Linear(d_model, d_model * 4),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(d_model * 4, d_model),
                    nn.Dropout(0.1),
                ),
                "ffn_norm": nn.LayerNorm(d_model),
            })
            for _ in range(n_layers)
        ])
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        batch, seq_len, _ = x.shape
        x = self.input_proj(x)
        x = x + self.pos_embed[:, :seq_len, :]
        for layer in self.layers:
            residual = x
            x = layer["norm"](x)
            x = layer["ssm"](x) + residual
            residual = x
            x = layer["ffn_norm"](x)
            x = layer["ffn"](x) + residual
        x = x.mean(dim=1)
        return self.head(x)


# ─── Feature Extraction ───

def extract_temporal_features(audio_path, sr=16000, seq_len=SEQ_LEN):
    """Extract a sequence of features from sliding windows over audio.

    Features per window (7 dims, matching inference code input_dim=7):
      0: energy (RMS) → proxy for stress
      1: spectral centroid → proxy for anxiety
      2: spectral rolloff → proxy for stability
      3: zero crossing rate → proxy for depression_risk
      4: spectral bandwidth → proxy for overall_risk
      5: fundamental frequency → voice_freq
      6: MFCC variance → confidence
    """
    try:
        y, actual_sr = librosa.load(audio_path, sr=sr, mono=True)
    except Exception:
        return None

    if len(y) < sr * 0.5:
        return None

    window_size = len(y) // seq_len
    hop = max(1, (len(y) - window_size) // max(1, seq_len - 1))

    features = []
    for i in range(seq_len):
        start = i * hop
        end = min(start + window_size, len(y))
        segment = y[start:end]

        if len(segment) < 256:
            segment = np.pad(segment, (0, 256 - len(segment)))

        rms = float(np.sqrt(np.mean(segment ** 2)))

        spec_cent = librosa.feature.spectral_centroid(y=segment, sr=sr)
        centroid = float(np.mean(spec_cent)) / 4000.0

        spec_roll = librosa.feature.spectral_rolloff(y=segment, sr=sr)
        rolloff = float(np.mean(spec_roll)) / 8000.0

        zcr = librosa.feature.zero_crossing_rate(segment)
        zcr_val = float(np.mean(zcr))

        spec_bw = librosa.feature.spectral_bandwidth(y=segment, sr=sr)
        bandwidth = float(np.mean(spec_bw)) / 4000.0

        try:
            pitches, magnitudes = librosa.piptrack(y=segment, sr=sr)
            pitch_vals = pitches[magnitudes > np.median(magnitudes)]
            pitch_vals = pitch_vals[pitch_vals > 50]
            f0 = float(np.median(pitch_vals)) / 500.0 if len(pitch_vals) > 0 else 0.36
        except Exception:
            f0 = 0.36

        mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=13)
        mfcc_var = float(np.var(mfcc)) / 100.0

        features.append([rms, centroid, rolloff, zcr_val, bandwidth, f0, min(1.0, mfcc_var)])

    return np.array(features, dtype=np.float32)


# ─── Data Loading ───

def load_ravdess(data_dir):
    files, labels = [], []
    for wav in glob.glob(os.path.join(data_dir, "**/*.wav"), recursive=True):
        parts = os.path.basename(wav).split("-")
        if len(parts) >= 3:
            try:
                ec = int(parts[2]) - 1
                if 0 <= ec < NUM_CLASSES:
                    files.append(wav); labels.append(ec)
            except ValueError:
                continue
    logger.info(f"RAVDESS: {len(files)} files")
    return files, labels


def load_cremad(data_dir):
    MAP = {"ANG": 4, "DIS": 6, "FEA": 5, "HAP": 2, "NEU": 0, "SAD": 3}
    files, labels = [], []
    for wav in glob.glob(os.path.join(data_dir, "**/*.wav"), recursive=True):
        parts = os.path.basename(wav).replace(".wav", "").split("_")
        if len(parts) >= 3 and parts[2] in MAP:
            files.append(wav); labels.append(MAP[parts[2]])
    logger.info(f"CREMA-D: {len(files)} files")
    return files, labels


def load_tess(data_dir):
    MAP = {"angry": 4, "disgust": 6, "fear": 5, "happy": 2, "neutral": 0, "ps": 7, "sad": 3}
    files, labels = [], []
    for wav in glob.glob(os.path.join(data_dir, "**/*.wav"), recursive=True):
        parts = os.path.basename(wav).lower().replace(".wav", "").split("_")
        if parts[-1] in MAP:
            files.append(wav); labels.append(MAP[parts[-1]])
    logger.info(f"TESS: {len(files)} files")
    return files, labels


class TemporalDataset(Dataset):
    def __init__(self, sequences, labels, augment=False):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        seq = self.sequences[idx].clone()
        if self.augment:
            if np.random.random() < 0.4:
                seq += torch.randn_like(seq) * 0.05
            if np.random.random() < 0.3:
                t = np.random.randint(0, seq.shape[0])
                seq[t] = 0
            if np.random.random() < 0.3:
                scale = torch.FloatTensor(1).uniform_(0.9, 1.1)
                seq *= scale
        return seq, self.labels[idx]


def main():
    parser = argparse.ArgumentParser(description="Train Mamba SSM Temporal Model")
    parser.add_argument("--ravdess", type=str, default="../data/datasets/ravdess")
    parser.add_argument("--cremad", type=str, default="../data/datasets/cremad")
    parser.add_argument("--tess", type=str, default="../data/datasets/tess")
    parser.add_argument("--output-dir", type=str, default="../data/pretrained")
    parser.add_argument("--results-dir", type=str, default="../results")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seq-len", type=int, default=10)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--state-dim", type=int, default=16)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load file paths
    all_files, all_labels = [], []
    for loader, path in [(load_ravdess, args.ravdess), (load_cremad, args.cremad), (load_tess, args.tess)]:
        f, l = loader(path)
        all_files.extend(f); all_labels.extend(l)

    if len(all_files) == 0:
        logger.error("No audio data found! Download datasets from Kaggle first.")
        sys.exit(1)

    # Extract temporal features
    logger.info(f"Extracting temporal features from {len(all_files)} audio files...")
    sequences, seq_labels = [], []
    skipped = 0
    for fp, label in zip(all_files, all_labels):
        feat = extract_temporal_features(fp, seq_len=args.seq_len)
        if feat is not None:
            sequences.append(feat)
            seq_labels.append(label)
        else:
            skipped += 1

    sequences = np.array(sequences)
    seq_labels = np.array(seq_labels)
    logger.info(f"Extracted: {len(sequences)} sequences (skipped {skipped}), shape: {sequences.shape}")

    # Split
    train_seq, test_seq, train_lab, test_lab = train_test_split(
        sequences, seq_labels, test_size=0.15, random_state=42, stratify=seq_labels
    )
    train_seq, val_seq, train_lab, val_lab = train_test_split(
        train_seq, train_lab, test_size=0.15, random_state=42, stratify=train_lab
    )

    logger.info(f"Train: {len(train_seq)}, Val: {len(val_seq)}, Test: {len(test_seq)}")

    train_loader = DataLoader(TemporalDataset(train_seq, train_lab, augment=True), batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(TemporalDataset(val_seq, val_lab), batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Model
    model = MambaTemporalModel(
        input_dim=FEATURE_DIM, d_model=args.d_model,
        n_layers=args.n_layers, state_dim=args.state_dim, num_classes=NUM_CLASSES,
    ).to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_f1 = 0.0
    history = []

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        for seqs, labels in train_loader:
            seqs, labels = seqs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(seqs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            _, pred = outputs.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()
        train_loss /= len(train_loader)
        train_acc = 100.0 * correct / total

        # Validate
        model.eval()
        val_loss, all_preds, all_true = 0.0, [], []
        with torch.no_grad():
            for seqs, labels in val_loader:
                seqs, labels = seqs.to(device), labels.to(device)
                outputs = model(seqs)
                val_loss += criterion(outputs, labels).item()
                _, pred = outputs.max(1)
                all_preds.extend(pred.cpu().numpy())
                all_true.extend(labels.cpu().numpy())
        val_loss /= len(val_loader)
        val_acc = 100.0 * np.mean(np.array(all_preds) == np.array(all_true))
        val_f1 = f1_score(all_true, all_preds, average="macro") * 100
        scheduler.step()

        logger.info(f"Epoch {epoch}/{args.epochs} — Train: {train_loss:.4f}/{train_acc:.1f}% | Val: {val_loss:.4f}/{val_acc:.1f}%/F1={val_f1:.1f}%")
        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc, "val_f1": val_f1})

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), output_dir / "temporal_mamba.pt")
            logger.info(f"New best F1: {val_f1:.2f}%")

    logger.info(f"\nTraining complete. Best F1: {best_f1:.2f}%")

    with open(results_dir / "temporal_training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Test evaluation
    test_loader = DataLoader(TemporalDataset(test_seq, test_lab), batch_size=args.batch_size, shuffle=False, num_workers=2)
    model.load_state_dict(torch.load(output_dir / "temporal_mamba.pt", map_location=device, weights_only=True))
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for seqs, labels in test_loader:
            seqs, labels = seqs.to(device), labels.to(device)
            outputs = model(seqs)
            _, pred = outputs.max(1)
            all_preds.extend(pred.cpu().numpy())
            all_true.extend(labels.cpu().numpy())
    test_acc = 100.0 * np.mean(np.array(all_preds) == np.array(all_true))
    test_f1 = f1_score(all_true, all_preds, average="macro") * 100
    logger.info(f"Test: Acc={test_acc:.2f}%, F1={test_f1:.2f}%")

    results = {
        "model": "MambaTemporalSSM",
        "datasets": ["RAVDESS", "CREMA-D", "TESS"],
        "total_samples": len(sequences),
        "seq_len": args.seq_len,
        "feature_dim": FEATURE_DIM,
        "num_classes": NUM_CLASSES,
        "classes": EMOTION_LABELS,
        "test_accuracy": round(test_acc, 2),
        "test_f1_macro": round(test_f1, 2),
        "best_val_f1": round(best_f1, 2),
        "parameters": params,
        "epochs_trained": args.epochs,
    }
    with open(results_dir / "temporal_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
