"""
StegGuard SOC — Dataset Downloader
Downloads BOSSbase 1.01 (images) and LibriSpeech (audio)
CSY4022 Computing Dissertation | Liza Gurung 24812928

Run from: C:\steganalysis_project\
Command:  python download_datasets.py
"""

import os
import urllib.request
import tarfile
import zipfile

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")
IMG  = os.path.join(DATA, "images")
AUD  = os.path.join(DATA, "audio")

os.makedirs(IMG, exist_ok=True)
os.makedirs(AUD, exist_ok=True)

def progress(count, block, total):
    pct = min(int(count * block / total * 100), 100)
    print(f"\r  Downloading... {pct}%", end="", flush=True)

# ── 1. BOSSbase 1.01 (images) ─────────────────────────────────────────────────
print("\n=== BOSSbase 1.01 Image Dataset ===")
print("NOTE: BOSSbase requires registration at:")
print("  http://agents.fel.cvut.cz/boss/index.php?article=overview")
print()
print("After registering, download 'BOSSbase_1.01.zip' manually")
print("and extract it to:", IMG)
print()
print("Alternatively, use the ALASKA2 dataset (no registration needed):")
alaska_url = "https://storage.googleapis.com/kaggle-data-sets/alaska2-image-steganalysis/alaska2.zip"
print(f"  URL: {alaska_url}")
print("  Or download from Kaggle: https://www.kaggle.com/c/alaska2-image-steganalysis/data")

# ── 2. LibriSpeech (audio) ────────────────────────────────────────────────────
print("\n=== LibriSpeech Audio Dataset ===")
libri_url  = "https://www.openslr.org/resources/12/dev-clean.tar.gz"
libri_file = os.path.join(AUD, "dev-clean.tar.gz")

if not os.path.exists(libri_file):
    print(f"Downloading LibriSpeech dev-clean (~337MB)...")
    try:
        urllib.request.urlretrieve(libri_url, libri_file, progress)
        print("\n✅ Downloaded!")
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        print("Manual download: https://www.openslr.org/12")
        print("Download 'dev-clean.tar.gz' and save to:", AUD)
else:
    print("✅ Already downloaded!")

# Extract
libri_extract = os.path.join(AUD, "LibriSpeech")
if os.path.exists(libri_file) and not os.path.exists(libri_extract):
    print("Extracting...")
    with tarfile.open(libri_file, "r:gz") as tar:
        tar.extractall(AUD)
    print("✅ Extracted to:", libri_extract)
elif os.path.exists(libri_extract):
    print("✅ Already extracted!")

# Count WAV files
wav_count = 0
if os.path.exists(libri_extract):
    for root, dirs, files in os.walk(libri_extract):
        wav_count += sum(1 for f in files if f.endswith('.flac') or f.endswith('.wav'))
    print(f"✅ Found {wav_count} audio files")

print("\n=== Summary ===")
print(f"Audio files available: {wav_count}")
print(f"Target dataset size:   50,000 samples")
print(f"Recommended split:     25,000 clean + 25,000 stego")
print()
print("Next step: Run prepare_dataset.py to extract features and retrain models")