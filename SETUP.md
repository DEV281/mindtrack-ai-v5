# MindTrack AI v5 — Setup Guide (Any System)

## Prerequisites

| Software | Version | Install Command |
|----------|---------|-----------------|
| Python | 3.10+ | [python.org](https://python.org) |
| Node.js | 18+ | [nodejs.org](https://nodejs.org) |
| PostgreSQL | 14+ | `brew install postgresql@15` (mac) / `sudo apt install postgresql` (linux) |
| Redis | 7+ | `brew install redis` (mac) / `sudo apt install redis-server` (linux) |
| Git | any | pre-installed on most systems |

---

## Step 1: Clone the Project

```bash
git clone <your-repo-url> mindtrack-ai
cd mindtrack-ai
```

Or copy the project folder to the new machine.

---

## Step 2: Start Database Services

### macOS (Homebrew)
```bash
brew services start postgresql@15
brew services start redis
```

### Linux (systemd)
```bash
sudo systemctl start postgresql
sudo systemctl start redis-server
```

### Windows
Start PostgreSQL and Redis from their respective installers or use WSL2.

---

## Step 3: Create the Database

```bash
# macOS
/opt/homebrew/opt/postgresql@15/bin/createdb mindtrack

# Linux / Windows
createdb mindtrack
# Or: psql -c "CREATE DATABASE mindtrack;"
```

---

## Step 4: Backend Setup

```bash
cd backend

# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
```

### Edit `backend/.env`:
```env
# IMPORTANT: Change the database URL to match YOUR system username
DATABASE_URL=postgresql+asyncpg://YOUR_USERNAME@localhost:5432/mindtrack
DATABASE_URL_SYNC=postgresql://YOUR_USERNAME@localhost:5432/mindtrack
REDIS_URL=redis://localhost:6379
SECRET_KEY=your-secret-key-here-change-this

# Gmail SMTP (for OTP emails)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=your-gmail-app-password

FRONTEND_URL=http://localhost:3000
DEBUG=true
```

> **Gmail App Password**: Go to https://myaccount.google.com/apppasswords → Select "Mail" → Generate. Use the 16-char password above.

---

## Step 5: Frontend Setup

```bash
cd ../frontend   # from backend directory

# Install dependencies
npm install
```

---

## Step 6: Run Everything

### Option A: Two Terminal Tabs (Recommended for Development)

**Tab 1 — Backend:**
```bash
cd mindtrack-ai/backend
source venv/bin/activate
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**Tab 2 — Frontend:**
```bash
cd mindtrack-ai/frontend
npm run dev
```

### Option B: Single Command (macOS only)
```bash
cd mindtrack-ai
./start.sh
```

### Option C: Docker (Any System — Recommended for Deployment)
```bash
cd mindtrack-ai
docker-compose up --build
# → http://localhost:80
```
Docker starts all 5 services automatically: PostgreSQL, Redis, Backend, Frontend, Nginx.
No need to install Python, Node.js, PostgreSQL, or Redis — everything runs in containers.

---

## Step 7: Open the App

| Service | URL (Local Dev) | URL (Docker) |
|---------|-----------------|--------------|
| **Frontend** | http://localhost:3000 | http://localhost:80 |
| **Backend API** | http://localhost:8000 | http://localhost:80/api |
| **API Docs** | http://localhost:8000/docs | http://localhost:80/docs |

### Demo Login
- Click **"Demo Login"** on the login page
- Credentials: `demo@mindtrack.ai` / `Demo@1234`
- Demo user skips OTP — instant access

### Register a New User
- Fill in email, name, institution, password
- OTP will be sent to the email (if Gmail SMTP is configured)
- In dev mode, OTP `000000` always works as a bypass

---

## Stopping the App

```bash
# Local dev: Ctrl+C in each terminal

# Docker:
docker-compose down

# Stop database services (local only, optional):
brew services stop postgresql@15   # macOS
brew services stop redis            # macOS
```

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `connection refused` on port 5432 | Start PostgreSQL: `brew services start postgresql@15` |
| `connection refused` on port 6379 | Start Redis: `brew services start redis` |
| `createdb: error: role does not exist` | Run: `createuser -s $(whoami)` |
| `MODULE_NOT_FOUND` in frontend | Run `npm install` in `frontend/` |
| OTP not arriving via email | Check `SMTP_USER` and `SMTP_PASSWORD` in `.env` |
| Session start fails | Ensure backend is running on port 8000 |
| Docker build fails on M1/M2 Mac | Add `platform: linux/amd64` to services in `docker-compose.yml` |


---

## Training ML Models

### 1. Facial Emotion Model (ViT-Base on FER2013)
```bash
# Google Colab (recommended — free T4 GPU):
# Open ml_training/colab_train.ipynb

# Local with GPU:
cd ml_training && python train_facial_vit.py
# Output: data/pretrained/vit_fer2013_best.pt (327MB)
```

### 2. Voice Emotion Model (SpectrogramCNN)
```bash
# Google Colab (recommended):
# Open ml_training/colab_train_voice.ipynb

# Local with GPU:
cd ml_training
python train_voice.py --ravdess ../data/datasets/ravdess --cremad ../data/datasets/cremad --tess ../data/datasets/tess
# Output: data/pretrained/voice_cnn.pt (~4MB)
```

### 3. Temporal Model (Mamba SSM)
```bash
# Google Colab (recommended):
# Open ml_training/colab_train_temporal.ipynb

# Local with GPU:
cd ml_training
python train_temporal.py --ravdess ../data/datasets/ravdess --cremad ../data/datasets/cremad --tess ../data/datasets/tess
# Output: data/pretrained/temporal_mamba.pt (~1MB)
```

### Download Datasets (All Free)
```bash
pip install kaggle
kaggle datasets download -d uwrfkaggler/ravdess-emotional-speech-audio -p data/datasets/ravdess --unzip
kaggle datasets download -d ejlok1/cremad -p data/datasets/cremad --unzip
kaggle datasets download -d ejlok1/toronto-emotional-speech-set-tess -p data/datasets/tess --unzip
```
