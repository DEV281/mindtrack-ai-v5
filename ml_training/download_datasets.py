#!/usr/bin/env python3
"""Download training datasets from Kaggle for MindTrack AI v5.

Requires: pip install kaggle
Set KAGGLE_USERNAME and KAGGLE_KEY environment variables.
"""

import os
import sys
import subprocess
from pathlib import Path


DATASETS = [
    {
        "name": "RAVDESS (Speech Emotion)",
        "kaggle_id": "uwrfkaggler/ravdess-emotional-speech-audio",
        "target_dir": "ravdess",
    },
    {
        "name": "CREMA-D (Speech Emotion)",
        "kaggle_id": "ejlok1/cremad",
        "target_dir": "cremad",
    },
    {
        "name": "TESS (Speech Emotion)",
        "kaggle_id": "ejlok1/toronto-emotional-speech-set-tess",
        "target_dir": "tess",
    },
    {
        "name": "FER2013 (Facial Emotion)",
        "kaggle_id": "msambare/fer2013",
        "target_dir": "fer2013",
    },
]


def check_kaggle():
    """Check if Kaggle CLI is installed and configured."""
    if not os.environ.get("KAGGLE_USERNAME") and not Path.home().joinpath(".kaggle/kaggle.json").exists():
        print("❌ Kaggle not configured. Either:")
        print("   1. Set KAGGLE_USERNAME and KAGGLE_KEY environment variables")
        print("   2. Place kaggle.json in ~/.kaggle/")
        print("   Get your API key at: https://www.kaggle.com/settings")
        sys.exit(1)

    try:
        subprocess.run(["kaggle", "--version"], capture_output=True, check=True)
    except FileNotFoundError:
        print("❌ Kaggle CLI not installed. Run: pip install kaggle")
        sys.exit(1)


def download_dataset(dataset: dict, base_dir: Path):
    """Download and extract a Kaggle dataset."""
    target = base_dir / dataset["target_dir"]
    if target.exists() and any(target.iterdir()):
        print(f"  ✅ {dataset['name']} already downloaded")
        return True

    target.mkdir(parents=True, exist_ok=True)
    print(f"  📥 Downloading {dataset['name']}...")

    result = subprocess.run(
        ["kaggle", "datasets", "download", "-d", dataset["kaggle_id"], "-p", str(target), "--unzip"],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"  ❌ Failed: {result.stderr.strip()}")
        return False

    print(f"  ✅ {dataset['name']} downloaded to {target}")
    return True


def main():
    print("🧠 MindTrack AI v5 — Dataset Downloader\n")
    check_kaggle()

    base_dir = Path(__file__).parent.parent / "data" / "datasets"
    base_dir.mkdir(parents=True, exist_ok=True)

    print(f"Download directory: {base_dir}\n")

    success = 0
    for ds in DATASETS:
        if download_dataset(ds, base_dir):
            success += 1

    print(f"\n{'='*50}")
    print(f"Downloaded {success}/{len(DATASETS)} datasets")

    if success == len(DATASETS):
        print("\n✅ All datasets ready! You can now train:")
        print("   python ml_training/train_voice.py")
        print("   python ml_training/train_temporal.py")
        print("   python ml_training/train_facial_vit.py")
    else:
        print("\n⚠️  Some datasets failed. Check your Kaggle credentials.")


if __name__ == "__main__":
    main()
