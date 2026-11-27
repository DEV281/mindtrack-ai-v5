"""
MindTrack AI v5 — Voice Emotion CNN Training
Trains Raw Spectrogram CNN on RAVDESS + CREMA-D + TESS datasets (all freely available).

Datasets:
  - RAVDESS: https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio
  - CREMA-D: https://www.kaggle.com/datasets/ejlok1/cremad
  - TESS:    https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess

Usage:
  # Local (with GPU):
  python train_voice.py --ravdess ../data/datasets/ravdess --cremad ../data/datasets/cremad --tess ../data/datasets/tess

  # Or use the Colab notebook: ml_training/colab_train_voice.ipynb
"""
import os
import sys
import json
import glob
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

# 8 emotion classes (RAVDESS standard)
EMOTION_LABELS = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
NUM_CLASSES = len(EMOTION_LABELS)


class SpectrogramCNN(nn.Module):
    """Raw Spectrogram CNN for voice emotion classification.

    Architecture MUST match backend/app/models/voice_model.py exactly.
    """

    def __init__(self, num_classes: int = 8):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


class VoiceDataset(Dataset):
    """Dataset for audio files with mel spectrogram extraction."""

    def __init__(self, file_paths: list, labels: list, sr: int = 16000, n_mels: int = 128,
                 max_len: int = 128, augment: bool = False):
        self.file_paths = file_paths
        self.labels = labels
        self.sr = sr
        self.n_mels = n_mels
        self.max_len = max_len
        self.augment = augment

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int):
        try:
            y, sr = librosa.load(self.file_paths[idx], sr=self.sr, mono=True)
        except Exception:
            return torch.zeros(1, self.n_mels, self.max_len), self.labels[idx]

        if len(y) < 1024:
            y = np.pad(y, (0, 1024 - len(y)))

        # Augmentation (training only)
        if self.augment:
            if np.random.random() < 0.3:
                rate = np.random.uniform(0.85, 1.15)
                y = librosa.effects.time_stretch(y, rate=rate)
            if np.random.random() < 0.3:
                n_steps = np.random.uniform(-2, 2)
                y = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
            if np.random.random() < 0.3:
                noise = np.random.randn(len(y)) * 0.005
                y = y + noise

        # Mel spectrogram
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=self.n_mels, fmax=8000, hop_length=512)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Pad/truncate
        if mel_spec_db.shape[1] < self.max_len:
            mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, self.max_len - mel_spec_db.shape[1])), mode="constant")
        else:
            mel_spec_db = mel_spec_db[:, :self.max_len]

        # Normalize
        mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-6)
        tensor = torch.FloatTensor(mel_spec_db).unsqueeze(0)
        return tensor, self.labels[idx]


def load_ravdess(data_dir: str) -> tuple:
    """Load RAVDESS dataset. Filename: 03-01-{emotion}-{intensity}-...-{actor}.wav"""
    files, labels = [], []
    for wav in glob.glob(os.path.join(data_dir, "**/*.wav"), recursive=True):
        parts = os.path.basename(wav).split("-")
        if len(parts) >= 3:
            try:
                ec = int(parts[2]) - 1
                if 0 <= ec < NUM_CLASSES:
                    files.append(wav)
                    labels.append(ec)
            except ValueError:
                continue
    logger.info(f"RAVDESS: {len(files)} files")
    return files, labels


def load_cremad(data_dir: str) -> tuple:
    """Load CREMA-D dataset. Filename: {ActorID}_{Statement}_{Emotion}_{Level}.wav"""
    MAP = {"ANG": 4, "DIS": 6, "FEA": 5, "HAP": 2, "NEU": 0, "SAD": 3}
    files, labels = [], []
    for wav in glob.glob(os.path.join(data_dir, "**/*.wav"), recursive=True):
        parts = os.path.basename(wav).replace(".wav", "").split("_")
        if len(parts) >= 3 and parts[2] in MAP:
            files.append(wav)
            labels.append(MAP[parts[2]])
    logger.info(f"CREMA-D: {len(files)} files")
    return files, labels


def load_tess(data_dir: str) -> tuple:
    """Load TESS dataset. Filename: OAF_back_angry.wav — emotion is last word."""
    MAP = {"angry": 4, "disgust": 6, "fear": 5, "happy": 2, "neutral": 0, "ps": 7, "sad": 3}
    files, labels = [], []
    for wav in glob.glob(os.path.join(data_dir, "**/*.wav"), recursive=True):
        parts = os.path.basename(wav).lower().replace(".wav", "").split("_")
        if parts[-1] in MAP:
            files.append(wav)
            labels.append(MAP[parts[-1]])
    logger.info(f"TESS: {len(files)} files")
    return files, labels


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for batch_idx, (specs, labels) in enumerate(dataloader):
        specs, labels = specs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(specs)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        if batch_idx % 20 == 0:
            logger.info(f"Epoch {epoch} [{batch_idx}/{len(dataloader)}] Loss: {loss.item():.4f}")
    return total_loss / len(dataloader), 100.0 * correct / total


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    for specs, labels in dataloader:
        specs, labels = specs.to(device), labels.to(device)
        outputs = model(specs)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * np.mean(np.array(all_preds) == np.array(all_labels))
    f1 = f1_score(all_labels, all_preds, average="macro") * 100
    return avg_loss, accuracy, f1


def main():
    parser = argparse.ArgumentParser(description="Train Voice Emotion CNN")
    parser.add_argument("--ravdess", type=str, default="../data/datasets/ravdess", help="RAVDESS directory")
    parser.add_argument("--cremad", type=str, default="../data/datasets/cremad", help="CREMA-D directory")
    parser.add_argument("--tess", type=str, default="../data/datasets/tess", help="TESS directory")
    parser.add_argument("--output-dir", type=str, default="../data/pretrained")
    parser.add_argument("--results-dir", type=str, default="../results")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load all available datasets
    all_files, all_labels = [], []

    r_files, r_labels = load_ravdess(args.ravdess)
    all_files.extend(r_files)
    all_labels.extend(r_labels)

    c_files, c_labels = load_cremad(args.cremad)
    all_files.extend(c_files)
    all_labels.extend(c_labels)

    t_files, t_labels = load_tess(args.tess)
    all_files.extend(t_files)
    all_labels.extend(t_labels)

    if len(all_files) == 0:
        logger.error("No audio data found! Download datasets:")
        logger.error("  RAVDESS: kaggle datasets download uwrfkaggler/ravdess-emotional-speech-audio")
        logger.error("  CREMA-D: kaggle datasets download ejlok1/cremad")
        logger.error("  TESS:    kaggle datasets download ejlok1/toronto-emotional-speech-set-tess")
        sys.exit(1)

    logger.info(f"Total: {len(all_files)} audio files across {NUM_CLASSES} classes")

    # Split: 70% train, 15% val, 15% test
    train_files, test_files, train_labels, test_labels = train_test_split(
        all_files, all_labels, test_size=0.15, random_state=42, stratify=all_labels
    )
    train_files, val_files, train_labels, val_labels = train_test_split(
        train_files, train_labels, test_size=0.15, random_state=42, stratify=train_labels
    )

    train_dataset = VoiceDataset(train_files, train_labels, augment=True)
    val_dataset = VoiceDataset(val_files, val_labels, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_files)}")

    # Model
    model = SpectrogramCNN(num_classes=NUM_CLASSES).to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_f1 = 0.0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        logger.info(
            f"Epoch {epoch}/{args.epochs} — "
            f"Train: {train_loss:.4f}/{train_acc:.1f}% | "
            f"Val: {val_loss:.4f}/{val_acc:.1f}%/F1={val_f1:.1f}%"
        )
        history.append({
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc, "val_f1": val_f1,
        })

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), output_dir / "voice_cnn.pt")
            logger.info(f"New best F1: {val_f1:.2f}% — Saved")

    logger.info(f"\nTraining complete. Best F1: {best_f1:.2f}%")

    with open(results_dir / "voice_training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Final test evaluation
    test_dataset = VoiceDataset(test_files, test_labels, augment=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    model.load_state_dict(torch.load(output_dir / "voice_cnn.pt", map_location=device, weights_only=True))
    test_loss, test_acc, test_f1 = evaluate(model, test_loader, criterion, device)
    logger.info(f"Test: Acc={test_acc:.2f}%, F1={test_f1:.2f}%")

    results = {
        "model": "SpectrogramCNN",
        "datasets": ["RAVDESS", "CREMA-D", "TESS"],
        "total_samples": len(all_files),
        "num_classes": NUM_CLASSES,
        "classes": EMOTION_LABELS,
        "test_accuracy": round(test_acc, 2),
        "test_f1_macro": round(test_f1, 2),
        "best_val_f1": round(best_f1, 2),
        "parameters": params,
        "epochs_trained": args.epochs,
    }
    with open(results_dir / "voice_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
