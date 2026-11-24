<div align="center">

# 🧠 MindTrack AI v5

### Real-Time Mental Health Stress Detection & AI Consultation Platform

<br/>

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18.3-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://react.dev)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.5-3178C6?style=for-the-badge&logo=typescript&logoColor=white)](https://typescriptlang.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Groq](https://img.shields.io/badge/Groq-LLaMA_3.3_70B-F55036?style=for-the-badge)](https://groq.com)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

<br/>

**A full-stack AI platform that combines facial emotion recognition (ViT), real-time biometric analysis, and an AI-powered mental health chatbot — all running on Groq's ultra-fast inference.**

[Features](#-features) · [Quick Start](#-quick-start) · [Architecture](#-architecture) · [Tech Stack](#-tech-stack) · [Training](#-training-the-facial-emotion-model) · [Deploy](#-deployment) · [API Docs](#-api-documentation)

</div>

---

## ✨ Features

### 🎯 Core Capabilities

- **🤖 AI Mental Health Chatbot** — Powered by Groq LLaMA 3.3 70B with real-time streaming responses. The AI receives live biometric context (heart rate, stress level, detected emotion) with every message to provide context-aware support.

- **📸 Facial Emotion Recognition** — ViT-Base model fine-tuned on FER2013 dataset (35,887 images, 7 emotion classes). Real PyTorch training pipeline with reproducible results.

- **🛡️ Crisis Safety Filter** — Deterministic pure-Python safety system that runs *before* every AI call. Detects 20+ crisis keywords, harm indicators, and dangerous biometric combinations. Cannot be bypassed or overridden.

- **📡 Real-Time WebSocket Streaming** — Bidirectional communication for live facial analysis and AI consultation with keepalive heartbeats.

- **🔐 Full Authentication System** — JWT tokens, refresh tokens, Google OAuth2, OTP email verification, rate limiting, and session management.

- **📊 Interactive Dashboard** — Weather-metaphor stress visualization, session history, trend analysis, and downloadable PDF reports.

### 🔬 What's Real vs. What's Simulated

> **Transparency is a core principle.** This project clearly labels every component.

| Component | Status | Details |
|-----------|--------|---------|
| Full-Stack Web App | ✅ **Real** | React 18 + TypeScript, FastAPI, PostgreSQL, Redis, JWT auth, OAuth2, OTP |
| AI Chatbot (Groq) | ✅ **Real** | Every response generated live via Groq API — no canned replies |
| Facial Emotion Model (ViT) | ✅ **Real** | Trained on FER2013 with confusion matrix, classification report, ROC AUC |
| Safety Filter | ✅ **Real** | 20 crisis keywords, biometric triggers, deterministic — zero AI dependency |
| Voice Stress Model | 🔬 **Demo** | Architecture defined (Raw Spectrogram CNN). Needs DAIC-WOZ dataset (restricted) |
| Temporal Sequence Model | 🔬 **Demo** | Architecture defined (Mamba SSM). Needs IEMOCAP dataset (USC license) |

> Demo-mode modalities show a clear **"Demo Mode"** indicator in the UI. Synthetic data is never presented as real model output.

---

## 🚀 Quick Start

### Docker (Recommended)

```bash
# 1. Clone the repo
git clone https://github.com/DEV281/mindtrack-ai-v5.git
cd mindtrack-ai-v5

# 2. Configure environment
cp backend/.env.example backend/.env
# Edit backend/.env — add your GROQ_API_KEY (free at console.groq.com)

# 3. Launch everything
docker-compose up --build

# 4. Open the app
open http://localhost:80
```

### Manual Setup

<details>
<summary><strong>Backend</strong></summary>

```bash
cd backend
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your GROQ_API_KEY and database credentials

# Start PostgreSQL and Redis (or use Docker for just these)
docker-compose up postgres redis -d

# Run database migrations
alembic upgrade head

# Start the backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

</details>

<details>
<summary><strong>Frontend</strong></summary>

```bash
cd frontend
npm install
npm run dev
# → http://localhost:3000
```

</details>

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        NGINX (port 80)                              │
│                   Reverse Proxy + Rate Limiting                     │
├──────────────────┬──────────────────────────────┬───────────────────┤
│                  │                              │                   │
│  ┌───────────────▼──────────────┐  ┌────────────▼───────────────┐  │
│  │      React Frontend          │  │      FastAPI Backend        │  │
│  │      (Vite + TypeScript)     │  │      (Uvicorn ASGI)         │  │
│  │      Port 3000               │  │      Port 8000              │  │
│  │                              │  │                             │  │
│  │  • Auth (Login/Register)     │  │  • REST API + WebSocket     │  │
│  │  • Live Analysis Session     │  │  • JWT / OTP / OAuth2       │  │
│  │  • AI Consultation Room      │  │  • Groq LLM Engine          │  │
│  │  • Dashboard + Reports       │  │  • Safety Filter (pre-AI)   │  │
│  │  • Voice Status Indicator    │  │  • ML Inference Pipeline    │  │
│  └──────────────────────────────┘  │                             │  │
│                                    │  ML Models:                 │  │
│                                    │  ├─ ViT-Base (Facial) ✅    │  │
│                                    │  ├─ CNN (Voice) 🔬 Planned  │  │
│                                    │  └─ Mamba SSM (Temporal) 🔬 │  │
│                                    └──────────┬──────────────────┘  │
│                                               │                     │
│               ┌───────────────────────────────┼───────────┐        │
│      ┌────────▼────────┐         ┌────────────▼────────┐  │        │
│      │   PostgreSQL    │         │       Redis          │  │        │
│      │   Port 5432     │         │    Port 6379         │  │        │
│      │  • Users        │         │  • OTP Codes (TTL)   │  │        │
│      │  • Sessions     │         │  • Session Cache     │  │        │
│      │  • Readings     │         │  • Refresh Tokens    │  │        │
│      │  • Reports      │         │                      │  │        │
│      └─────────────────┘         └─────────────────────┘  │        │
│               └───────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | React 18, TypeScript 5.5, Vite, Tailwind CSS, Framer Motion, Zustand, Lucide Icons |
| **Backend** | FastAPI 0.111, Python 3.11+, Uvicorn, SQLAlchemy (async), Alembic, Pydantic v2 |
| **AI/ML** | Groq (LLaMA 3.3 70B), PyTorch 2.3, HuggingFace Transformers, ViT-Base, timm |
| **Database** | PostgreSQL 16, Redis 7 |
| **Auth** | JWT (access + refresh), Google OAuth2, OTP via SMTP, bcrypt |
| **Infra** | Docker Compose, NGINX, Render Blueprint |
| **Design** | Nunito font, CSS custom properties (`--primary: #5b9bd5`) |

---

## 🧪 Training the Facial Emotion Model

The ViT model is trained on **FER2013** (35,887 grayscale 48×48 images → 7 emotions). The training pipeline runs on Google Colab free tier (T4 GPU):

```bash
cd ml_training

# Option 1: Run locally with GPU
python train_facial_vit.py

# Option 2: Use the Colab notebook
# Open ml_training/colab_train.ipynb in Google Colab
```

### Training Outputs

| File | Description |
|------|-------------|
| `data/pretrained/vit_fer2013_best.pt` | Trained model weights (327 MB) |
| `results/confusion_matrix.png` | 7×7 confusion matrix heatmap |
| `results/classification_report.json` | Per-class precision, recall, F1 |
| `results/training_curves.png` | Loss and F1 curves over epochs |
| `results/metrics_summary.json` | Final metrics — proof training happened |

### Evaluate a Saved Model

```bash
cd ml_training
python evaluate_saved_model.py --checkpoint ../data/pretrained/vit_fer2013_best.pt
```

### Model Selection Research

A systematic comparison of **30 multi-modal model combinations** was conducted:

| Facial | Voice | Temporal | Combined F1 |
|--------|-------|----------|:-----------:|
| **ViT-Base** | Raw Spectrogram CNN | Mamba SSM | **79.5%** |

See [results/README.md](./results/README.md) for full training documentation.

---

## 🔐 Environment Variables

| Variable | Required | Description |
|----------|:--------:|-------------|
| `GROQ_API_KEY` | ✅ | Groq API key — powers ALL AI features ([console.groq.com](https://console.groq.com)) |
| `DATABASE_URL` | ✅ | PostgreSQL connection string |
| `REDIS_URL` | ✅ | Redis connection string |
| `SECRET_KEY` | ✅ | JWT signing key (`openssl rand -hex 32`) |
| `SMTP_HOST` | ❌ | Email SMTP host (default: `smtp.gmail.com`) |
| `SMTP_USER` | ❌ | SMTP username |
| `SMTP_PASSWORD` | ❌ | SMTP app password |
| `GOOGLE_CLIENT_ID` | ❌ | Google OAuth client ID |
| `GOOGLE_CLIENT_SECRET` | ❌ | Google OAuth secret |
| `FRONTEND_URL` | ❌ | Frontend origin for CORS (default: `http://localhost:3000`) |
| `REAL_MODEL_PATH` | ❌ | Path to trained ViT weights (auto-detected) |

<details>
<summary><strong>How to Get Your Groq API Key (Free)</strong></summary>

1. Go to [console.groq.com](https://console.groq.com)
2. Create a free account
3. Navigate to **API Keys**
4. Create a new key
5. Add to your `.env` file: `GROQ_API_KEY=gsk_...`

Groq provides free API access with generous rate limits. MindTrack AI uses `llama-3.3-70b-versatile` for all AI responses.

</details>

---

## 🚢 Deployment

A [render.yaml](./render.yaml) Render Blueprint is included for one-click deployment:

```
Services:
  ├── mindtrack-backend   (Python, Web Service)
  ├── mindtrack-frontend  (Static Site)
  ├── PostgreSQL          (Managed Database)
  └── Redis               (Key-Value Store)
```

See [DEPLOY.md](./DEPLOY.md) for full deployment instructions (Render, Vercel, Railway).

---

## 📚 API Documentation

Once the backend is running, interactive API docs are available at:

| Docs | URL |
|------|-----|
| **Swagger UI** | `http://localhost:8000/docs` |
| **ReDoc** | `http://localhost:8000/redoc` |

### Key Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/auth/register` | Register new user |
| `POST` | `/api/auth/login` | Login (returns JWT) |
| `POST` | `/api/auth/verify-otp` | Verify email OTP |
| `GET` | `/api/users/me` | Get current user profile |
| `WS` | `/api/analysis/ws/{session_id}` | Real-time facial analysis |
| `WS` | `/api/consultation/ws/{session_id}` | AI chatbot with streaming |
| `GET` | `/api/analysis/model-status` | Check ML model status |
| `GET` | `/api/sessions/` | List user sessions |
| `GET` | `/api/reports/{id}` | Download session report |

---

## 📁 Project Structure

```
mindtrack-ai-v5/
├── backend/                    # FastAPI backend
│   ├── app/
│   │   ├── api/                # Route handlers
│   │   │   ├── auth.py         #   Registration, login, OTP, OAuth2
│   │   │   ├── analysis.py     #   WebSocket facial analysis + model status
│   │   │   ├── consultation.py #   AI chatbot with safety filter
│   │   │   ├── sessions.py     #   Session CRUD
│   │   │   ├── reports.py      #   PDF report generation
│   │   │   └── users.py        #   User profile management
│   │   ├── core/               # Core modules
│   │   │   ├── ai_engine.py    #   Groq LLM integration with streaming
│   │   │   ├── safety_filter.py#   Crisis detection (runs pre-AI)
│   │   │   ├── config.py       #   App configuration
│   │   │   ├── database.py     #   Async SQLAlchemy + Render compat
│   │   │   ├── redis.py        #   Redis connection
│   │   │   ├── security.py     #   JWT, hashing, OAuth
│   │   │   └── websocket.py    #   WebSocket manager
│   │   ├── db/                 # SQLAlchemy ORM models
│   │   ├── models/             # ML inference
│   │   │   ├── facial_inference.py  # FaceAnalyzer (ViT + OpenCV)
│   │   │   ├── facial_model.py      # Model architecture
│   │   │   ├── fusion.py            # Multi-modal fusion
│   │   │   └── model_loader.py      # Model loading utilities
│   │   └── schemas/            # Pydantic request/response schemas
│   ├── main.py                 # FastAPI app entry point
│   ├── requirements.txt        # Python dependencies
│   └── requirements-deploy.txt # Production deps (includes PyTorch)
├── frontend/                   # React + TypeScript (Vite)
│   └── src/
│       ├── api/                # Axios client with JWT interceptors
│       ├── components/         # ChatInputBar, VoiceStatusIndicator, etc.
│       ├── hooks/              # useCamera, useMicrophone, useWebSocket, etc.
│       ├── pages/              # Auth, Dashboard, LiveSession, Consultation
│       ├── store/              # Zustand state management
│       └── styles/             # Tailwind CSS + design system
├── ml_training/                # Training pipeline
│   ├── train_facial_vit.py     #   Full ViT training script
│   ├── evaluate_saved_model.py #   Standalone model evaluation
│   └── colab_train.ipynb       #   Google Colab notebook
├── results/                    # Training output artifacts
├── data/                       # Datasets + pretrained weights
├── nginx/                      # NGINX reverse proxy config
├── docker-compose.yml          # Full-stack Docker setup
├── render.yaml                 # Render deployment blueprint
├── DEPLOY.md                   # Deployment guide
├── SETUP.md                    # Setup guide
└── TEAM_CONTRIBUTIONS.md       # Team contributions
```

---

## ⚠️ Honest Limitations

1. **Voice and temporal models are not yet trained.** They require restricted datasets (DAIC-WOZ, IEMOCAP) with academic licenses. The app clearly indicates demo mode when unavailable.
2. **Facial emotion recognition has known limitations.** FER2013 has ~65% human inter-annotator agreement. Our model achieves competitive results, but face-based emotion recognition alone is not a clinical tool.
3. **This is a research project, not a medical device.** Nothing in this app should be used for clinical diagnosis. The safety filter provides emergency resources for crisis situations, but is not a substitute for professional help.
4. **Groq API dependency.** The chatbot requires an active Groq API key. Without it, consultation won't work (the rest of the app functions normally).

---

## 📖 Citations

- **FER2013** — Goodfellow, I. J. et al. *"Challenges in representation learning: A report on three machine learning contests."* ICML 2013.
- **ViT** — Dosovitskiy, A. et al. *"An Image is Worth 16×16 Words."* ICLR, 2021.
- **Mamba** — Gu, A. & Dao, T. *"Mamba: Linear-Time Sequence Modeling with Selective State Spaces."* arXiv, 2023.
- **DAIC-WOZ** — Gratch, J. et al. *"The Distress Analysis Interview Corpus."* LREC, 2014.
- **IEMOCAP** — Busso, C. et al. *"IEMOCAP: Interactive Emotional Dyadic Motion Capture Database."* Language Resources & Evaluation, 2008.

---

<div align="center">

**Made with ❤️ for mental health research**

**MindTrack AI v5.0** · Powered by [Groq](https://groq.com) · Real ML · Honest about what's real

[⬆ Back to Top](#-mindtrack-ai-v5)

</div>
