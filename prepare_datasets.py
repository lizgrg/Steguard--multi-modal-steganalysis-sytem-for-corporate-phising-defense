"""
StegGuard SOC — Dataset Preparation & Model Retraining
CSY4022 Computing Dissertation | Liza Gurung 24812928

This script:
1. Loads LibriSpeech WAV files
2. Creates stego versions using LSB embedding
3. Extracts MFCC + LSB features from all files
4. Saves new audio_features_dataset.csv
5. Retrains audio classifiers
6. Saves new .pkl model files

Run from: C:\steganalysis_project\
Command:  python prepare_dataset.py
"""

import os
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')

BASE      = os.path.dirname(os.path.abspath(__file__))
LIBRI_DIR = os.path.join(BASE, "data", "audio", "LibriSpeech")
MODEL_DIR = os.path.join(BASE, "models")
STEGO_DIR = os.path.join(BASE, "data", "audio", "stego")
os.makedirs(STEGO_DIR, exist_ok=True)

# ── Step 1: Collect all FLAC/WAV files from LibriSpeech ───────────────────────
print("\n=== Step 1: Collecting audio files ===")
audio_files = []
for root, dirs, files in os.walk(LIBRI_DIR):
    for f in files:
        if f.endswith('.flac') or f.endswith('.wav'):
            audio_files.append(os.path.join(root, f))

print(f"Found {len(audio_files)} audio files")

# Limit to 2000 for speed (1000 clean + 1000 stego)
if len(audio_files) > 2000:
    audio_files = audio_files[:2000]
    print(f"Using first 2000 files for balanced dataset")

# ── Step 2: LSB Steganography embedding ───────────────────────────────────────
print("\n=== Step 2: Creating stego versions ===")

def embed_lsb(audio, message_bits=None):
    """Embed random bits into LSB of audio samples."""
    samples = (audio * 32767).astype(np.int16)
    if message_bits is None:
        message_bits = np.random.randint(0, 2, len(samples))
    # Clear LSB and set new value
    samples = (samples & ~1) | message_bits[:len(samples)]
    return samples.astype(np.float32) / 32767.0

stego_paths = []
clean_paths = []

for i, fpath in enumerate(audio_files):
    if i % 100 == 0:
        print(f"  Processing {i}/{len(audio_files)}...", end='\r')
    try:
        y, sr = librosa.load(fpath, sr=16000, mono=True)
        clean_paths.append((fpath, y, sr, 0))  # label 0 = clean

        # Create stego version
        stego_y = embed_lsb(y)
        stego_name = f"stego_{i:04d}.wav"
        stego_path = os.path.join(STEGO_DIR, stego_name)
        sf.write(stego_path, stego_y, sr)
        stego_paths.append((stego_path, stego_y, sr, 1))  # label 1 = stego
    except Exception as e:
        continue

print(f"\n✅ Created {len(stego_paths)} stego files")
print(f"✅ {len(clean_paths)} clean files ready")

# ── Step 3: Feature extraction ────────────────────────────────────────────────
print("\n=== Step 3: Extracting features ===")

def extract_features(y, sr):
    """Extract 45 features matching audio_scaler.pkl exactly."""
    samples = (y * 32767).astype(np.int16)
    lsb     = samples & 1
    lsb2    = (samples >> 1) & 1

    lsb_mean = float(np.mean(lsb))
    lsb_std  = float(np.std(lsb))
    p = np.clip(lsb_mean, 1e-10, 1-1e-10)
    lsb_entropy     = float(-p*np.log2(p) - (1-p)*np.log2(1-p))
    lsb_transitions = float(np.sum(np.diff(lsb.astype(int)) != 0) / max(len(lsb)-1, 1))

    lsb2_mean = float(np.mean(lsb2))
    p2 = np.clip(lsb2_mean, 1e-10, 1-1e-10)
    lsb2_entropy = float(-p2*np.log2(p2) - (1-p2)*np.log2(1-p2))

    hist, _ = np.histogram(y, bins=256, density=True)
    hist    = hist + 1e-10
    hist_std      = float(np.std(hist))
    hist_entropy  = float(-np.sum(hist * np.log2(hist)))
    hist_flatness = float(np.exp(np.mean(np.log(hist))) / np.mean(hist))

    yc = y[:1000] if len(y) >= 1000 else y
    ac = np.correlate(yc, yc, mode='full')
    ac = ac[len(ac)//2:]
    ac = ac / (ac[0] + 1e-10)
    autocorr_1 = float(ac[1]) if len(ac) > 1 else 0.0
    autocorr_2 = float(ac[2]) if len(ac) > 2 else 0.0

    even = samples[0::2].astype(np.float64)
    odd  = samples[1::2].astype(np.float64)
    n    = min(len(even), len(odd))
    even, odd = even[:n], odd[:n]
    even_odd_diff = float(np.mean(np.abs(even - odd)))
    even_odd_corr = float(np.corrcoef(even, odd)[0,1]) if n > 1 else 0.0

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_feats = []
    for i in range(13):
        mfcc_feats += [float(np.mean(mfcc[i])), float(np.std(mfcc[i]))]

    yf = y.astype(np.float64)
    sample_var      = float(np.var(yf))
    sample_kurtosis = float(kurtosis(yf))
    sample_skew     = float(skew(yf))

    residual = np.diff(yf)
    residual_std      = float(np.std(residual))
    residual_kurtosis = float(kurtosis(residual))
    r_hist, _ = np.histogram(residual, bins=256, density=True)
    r_hist = r_hist + 1e-10
    residual_entropy = float(-np.sum(r_hist * np.log2(r_hist)))

    return [
        lsb_mean, lsb_std, lsb_entropy, lsb_transitions,
        lsb2_mean, lsb2_entropy,
        hist_std, hist_entropy, hist_flatness,
        autocorr_1, autocorr_2,
        even_odd_diff, even_odd_corr,
        *mfcc_feats,
        sample_var, sample_kurtosis, sample_skew,
        residual_std, residual_kurtosis, residual_entropy
    ]

COLUMNS = [
    'lsb_mean','lsb_std','lsb_entropy','lsb_transitions',
    'lsb2_mean','lsb2_entropy',
    'hist_std','hist_entropy','hist_flatness',
    'autocorr_1','autocorr_2',
    'even_odd_diff','even_odd_corr',
    *[f'mfcc_{s}_{i}' for i in range(13) for s in ['mean','std']],
    'sample_var','sample_kurtosis','sample_skew',
    'residual_std','residual_kurtosis','residual_entropy',
    'label'
]

rows = []
all_files = clean_paths + stego_paths
for i, (path, y, sr, label) in enumerate(all_files):
    if i % 100 == 0:
        print(f"  Extracting features {i}/{len(all_files)}...", end='\r')
    try:
        feats = extract_features(y, sr)
        feats.append(label)
        rows.append(feats)
    except Exception as e:
        continue

print(f"\n✅ Extracted features from {len(rows)} files")

# Load existing dataset and combine
existing_path = os.path.join(MODEL_DIR, "audio_features_dataset.csv")
new_df = pd.DataFrame(rows, columns=COLUMNS)

if os.path.exists(existing_path):
    existing_df = pd.read_csv(existing_path)
    # Align columns
    for col in COLUMNS:
        if col not in existing_df.columns:
            existing_df[col] = 0.0
    existing_df = existing_df[COLUMNS]
    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    print(f"✅ Combined with existing: {len(combined_df)} total samples")
else:
    combined_df = new_df
    print(f"✅ New dataset: {len(combined_df)} samples")

# Save
out_path = os.path.join(MODEL_DIR, "audio_features_dataset.csv")
combined_df.to_csv(out_path, index=False)
print(f"✅ Saved to {out_path}")

# ── Step 4: Retrain models ────────────────────────────────────────────────────
print("\n=== Step 4: Retraining audio models ===")

feature_cols = [c for c in combined_df.columns if c != 'label']
X = combined_df[feature_cols].values
y = combined_df['label'].values

print(f"Dataset: {len(X)} samples, {X.shape[1]} features")
print(f"Class distribution: Clean={sum(y==0)}, Stego={sum(y==1)}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# Train Gradient Boosting
print("\nTraining Gradient Boosting...")
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb.fit(X_train_s, y_train)
gb_pred = gb.predict(X_test_s)
gb_f1   = f1_score(y_test, gb_pred)
print(f"  GB F1-Score: {gb_f1:.3f}")
print(classification_report(y_test, gb_pred, target_names=['Clean','Stego']))

# Train Random Forest
print("Training Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_s, y_train)
rf_pred = rf.predict(X_test_s)
rf_f1   = f1_score(y_test, rf_pred)
print(f"  RF F1-Score: {rf_f1:.3f}")

# Train SVM
print("Training SVM...")
svm = SVC(kernel='rbf', probability=True, random_state=42)
svm.fit(X_train_s, y_train)
svm_pred = svm.predict(X_test_s)
svm_f1   = f1_score(y_test, svm_pred)
print(f"  SVM F1-Score: {svm_f1:.3f}")

# Pick best
scores = {'gb': (gb_f1, gb), 'rf': (rf_f1, rf), 'svm': (svm_f1, svm)}
best_name = max(scores, key=lambda k: scores[k][0])
best_f1, best_model = scores[best_name]
print(f"\n🏆 Best model: {best_name.upper()} with F1={best_f1:.3f}")

# Save models
joblib.dump(scaler,      os.path.join(MODEL_DIR, "audio_scaler.pkl"))
joblib.dump(gb,          os.path.join(MODEL_DIR, "audio_classifier_gb.pkl"))
joblib.dump(rf,          os.path.join(MODEL_DIR, "audio_classifier_rf.pkl"))
joblib.dump(svm,         os.path.join(MODEL_DIR, "audio_classifier_svm.pkl"))
joblib.dump(best_model,  os.path.join(MODEL_DIR, "audio_classifier_best.pkl"))

print("\n✅ All models saved!")
print("\n=== Final Summary ===")
print(f"Total audio samples: {len(combined_df)}")
print(f"Clean: {sum(combined_df['label']==0)}")
print(f"Stego: {sum(combined_df['label']==1)}")
print(f"Best model F1: {best_f1:.3f}")
print(f"\nModels saved to: {MODEL_DIR}")
print("Restart uvicorn to use new models!")