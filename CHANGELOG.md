# Changelog

All notable changes to MindTrack AI are documented here.

## [5.0.0] — 2026-12-04

### Added
- **Voice Emotion CNN** — SpectrogramCNN trained on RAVDESS + CREMA-D + TESS (~11.6K samples, 8 emotions)
- **Temporal Mamba SSM** — 4-layer Selective State Space Model trained on temporal audio features
- Google Colab notebooks for training both voice and temporal models (`colab_train_voice.ipynb`, `colab_train_temporal.ipynb`)
- Multi-model status endpoint (`/api/analysis/model-status`) showing all 3 models
- Real-time voice analysis in WebSocket when voice weights are loaded
- Real-time temporal analysis in WebSocket when temporal weights are loaded
- Data augmentation for voice training (time stretch, pitch shift, noise)
- Data augmentation for temporal training (Gaussian noise, time masking, scaling)
- librosa + soundfile dependencies for audio processing

### Changed
- Voice model no longer requires restricted DAIC-WOZ dataset — uses freely available datasets
- Temporal model no longer requires restricted IEMOCAP dataset — uses freely available datasets
- Aligned temporal_model.py inference architecture with training (4-layer MambaTemporalModel)
- Voice inference now normalizes spectrograms to match training pipeline
- WebSocket analysis blends facial + voice + temporal when all models loaded
- Updated README to reflect all 3 models as trainable on free datasets
- Version bump: 4.0.0 → 5.0.0 across all files

### Fixed
- Spectrogram normalization mismatch between training and inference
- Temporal model architecture mismatch (was 2-layer, now 4-layer like training)

## [4.0.0] — 2026-11-16

### Added
- **Facial Emotion ViT** — ViT-Base fine-tuned on FER2013 (7 emotions, 85.8M params)
- **Groq LLM Chatbot** — LLaMA 3.3 70B with real-time streaming
- **Safety Filter** — 20 crisis keywords, biometric triggers, deterministic
- Full-stack web app: React 18 + TypeScript + FastAPI + PostgreSQL + Redis
- JWT auth, Google OAuth2, OTP email verification
- WebSocket streaming for real-time analysis
- Docker Compose deployment with NGINX reverse proxy
- Render Blueprint for cloud deployment
- Google Colab notebook for facial model training
- Comprehensive test suite (57 tests)
