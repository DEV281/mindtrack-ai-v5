# Contributing to MindTrack AI

Thank you for your interest in contributing to MindTrack AI! This document provides guidelines and instructions.

## 🛠️ Development Setup

```bash
# Clone the repository
git clone https://github.com/DEV281/mindtrack-ai-v5.git
cd mindtrack-ai-v5

# Backend setup
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your credentials

# Frontend setup
cd ../frontend
npm install
```

## 🧪 Running Tests

```bash
# Backend tests
cd backend
python -m pytest ../tests/ -v

# Frontend type check
cd frontend
npx tsc --noEmit
```

## 🏗️ Architecture

| Component | Technology | Location |
|-----------|-----------|----------|
| Backend API | FastAPI + Python 3.11+ | `backend/` |
| Frontend | React 18 + TypeScript | `frontend/` |
| ML Training | PyTorch + librosa | `ml_training/` |
| ML Inference | PyTorch (CPU/GPU) | `backend/app/models/` |

## 📋 Commit Convention

We use [Conventional Commits](https://www.conventionalcommits.org/):

| Prefix | Usage |
|--------|-------|
| `feat:` | New feature |
| `fix:` | Bug fix |
| `docs:` | Documentation only |
| `test:` | Adding or updating tests |
| `refactor:` | Code refactoring (no behavior change) |
| `chore:` | Build, CI, dependencies |
| `perf:` | Performance improvement |

## 🔒 Important Rules

1. **GROQ ONLY** — No OpenAI, no Anthropic API calls anywhere
2. **Safety filter** must run before every AI call — it cannot be bypassed
3. **Demo mode** must be clearly indicated — never present synthetic data as real
4. **TypeScript zero `any`** — All React components and hooks must be fully typed
5. **No secrets in code** — All API keys come from environment variables

## 🧠 ML Models

| Model | Architecture | Dataset | File |
|-------|-------------|---------|------|
| Facial | ViT-Base (FER2013) | 35,887 images | `vit_fer2013_best.pt` |
| Voice | SpectrogramCNN | RAVDESS + CREMA-D + TESS | `voice_cnn.pt` |
| Temporal | Mamba SSM (4-layer) | Temporal audio features | `temporal_mamba.pt` |

Training notebooks are in `ml_training/colab_train_*.ipynb`.

## 📝 Pull Request Process

1. Fork the repository
2. Create a feature branch: `git checkout -b feat/my-feature`
3. Make your changes with tests
4. Ensure all tests pass
5. Submit a PR with a clear description

## 🐛 Reporting Issues

- Use GitHub Issues
- Include: steps to reproduce, expected vs actual behavior, environment info
- For security issues, email directly — do not create public issues
