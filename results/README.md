# MindTrack AI v5 — Training Results

This directory contains training output artifacts for all three ML models.

## Models

### 1. Facial Emotion (ViT-Base on FER2013)

| Metric | Value |
|--------|-------|
| **Architecture** | ViT-Base-Patch16-224 |
| **Dataset** | FER2013 (35,887 images, 7 emotions) |
| **Parameters** | 85.8M |
| **Weights File** | `data/pretrained/vit_fer2013_best.pt` (327 MB) |
| **Training Notebook** | `ml_training/colab_train.ipynb` |

**Output Files:**
- `confusion_matrix.png` — 7×7 confusion matrix heatmap
- `classification_report.json` — Per-class precision, recall, F1
- `training_curves.png` — Loss and F1 curves over epochs
- `metrics_summary.json` — Final metrics summary

### 2. Voice Emotion (SpectrogramCNN)

| Metric | Value |
|--------|-------|
| **Architecture** | 4-layer Conv2d + BatchNorm + AdaptiveAvgPool |
| **Datasets** | RAVDESS (1,440) + CREMA-D (7,442) + TESS (2,800) = ~11,682 samples |
| **Classes** | 8 (neutral, calm, happy, sad, angry, fearful, disgust, surprised) |
| **Input** | Mel spectrogram (128 mels × 128 time steps) |
| **Weights File** | `data/pretrained/voice_cnn.pt` (~4 MB) |
| **Training Notebook** | `ml_training/colab_train_voice.ipynb` |

**Output Files:**
- `voice_confusion_matrix.png` — 8×8 confusion matrix
- `voice_results.json` — Test accuracy and F1 scores
- `voice_training_history.json` — Per-epoch training metrics
- `voice_training_curves.png` — Loss and accuracy curves

### 3. Temporal (Mamba SSM)

| Metric | Value |
|--------|-------|
| **Architecture** | 4-layer Selective State Space Model + FFN |
| **Datasets** | Temporal features from RAVDESS + CREMA-D + TESS |
| **Input** | Sequence of 10 windows × 7 features (energy, centroid, pitch, ZCR, bandwidth, F0, MFCC variance) |
| **Classes** | 8 emotions |
| **Weights File** | `data/pretrained/temporal_mamba.pt` (~1 MB) |
| **Training Notebook** | `ml_training/colab_train_temporal.ipynb` |

**Output Files:**
- `temporal_confusion_matrix.png` — 8×8 confusion matrix
- `temporal_results.json` — Test accuracy and F1 scores
- `temporal_training_history.json` — Per-epoch training metrics
- `temporal_training_curves.png` — Loss and accuracy curves

## Datasets (All Free)

| Dataset | Samples | Source |
|---------|---------|--------|
| FER2013 | 35,887 images | [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013) |
| RAVDESS | 1,440 audio clips | [Kaggle](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio) |
| CREMA-D | 7,442 audio clips | [Kaggle](https://www.kaggle.com/datasets/ejlok1/cremad) |
| TESS | 2,800 audio clips | [Kaggle](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess) |

## Reproducing Results

```bash
# Download datasets
python ml_training/download_datasets.py

# Train all models (requires GPU — use Colab notebooks for free T4)
python ml_training/train_facial_vit.py
python ml_training/train_voice.py
python ml_training/train_temporal.py

# Evaluate a saved model
python ml_training/evaluate_saved_model.py --checkpoint data/pretrained/vit_fer2013_best.pt
```
