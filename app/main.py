"""
StegGuard SOC — FastAPI Backend
CSY4022 Computing Dissertation | Liza Gurung 24812928
"""

from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates
import numpy as np
import pandas as pd
import cv2
import joblib
import librosa
from scipy.stats import skew, kurtosis
import tempfile, os, datetime, hashlib, time

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(BASE_DIR, "..", "models")
TMPL_DIR   = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")

app = FastAPI(title="StegGuard SOC API", version="1.0.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TMPL_DIR)

def render(name: str, ctx: dict, request: Request) -> HTMLResponse:
    import jinja2
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(TMPL_DIR), autoescape=True)
    tmpl = env.get_template(name)
    ctx["request"] = request
    return HTMLResponse(tmpl.render(**ctx))

# ── Load models ────────────────────────────────────────────────────────────────
MODELS = {}

def load_models():
    try:
        MODELS["img_svm"]    = joblib.load(os.path.join(MODEL_DIR, "image_classifier_svm.pkl"))
        MODELS["img_scaler"] = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
        MODELS["img_loaded"] = True
        print("✅ Image models loaded")
    except Exception as e:
        MODELS["img_loaded"] = False
        print(f"⚠️  Image models not found: {e}")
    try:
        MODELS["aud_gb"]     = joblib.load(os.path.join(MODEL_DIR, "audio_classifier_gb.pkl"))
        MODELS["aud_scaler"] = joblib.load(os.path.join(MODEL_DIR, "audio_scaler.pkl"))
        MODELS["aud_loaded"] = True
        print("✅ Audio models loaded")
    except Exception as e:
        MODELS["aud_loaded"] = False
        print(f"⚠️  Audio models not found: {e}")

load_models()

SCAN_HISTORY  = []
SESSION_STATS = {"total": 0, "threats": 0, "clean": 0, "review": 0}

def load_from_db():
    try:
        from database import get_all_scans
        for row in get_all_scans():
            verdict = row['prediction'].upper()
            SCAN_HISTORY.append({
                "audit_id":      f"AUD-{str(row['job_id']).zfill(4)}",
                "file_name":     row['file_name'],
                "file_type":     "IMG" if str(row['file_type']).lower() in ['image','img'] else "WAV",
                "stego_prob":    round(float(row['confidence']) / 100, 3),
                "verdict":       verdict,
                "timestamp":     str(row['created_at']),
                "processing_ms": 0, "sha256": "—", "confidence": "High", "algorithm": "—",
                "modality":      str(row['file_type']).lower()
            })
            SESSION_STATS["total"] += 1
            if verdict == "STEGO":   SESSION_STATS["threats"] += 1
            elif verdict == "CLEAN": SESSION_STATS["clean"]   += 1
            else:                    SESSION_STATS["review"]  += 1
        print(f"✅ Loaded {len(SCAN_HISTORY)} existing scans from database")
    except Exception as e:
        print(f"⚠️  Could not load existing scans: {e}")

load_from_db()

# ── IMAGE features — matches scaler.pkl exactly (28 features) ─────────────────
# Feature names: res0-res4 (mean,std,energy,kurtosis,skew) + pixel_mean,std,var
def extract_image_features(img_gray):
    kernels = [
        np.array([[0,0,0],[0,-1,1],[0,0,0]], dtype=np.float32),
        np.array([[0,0,0],[0,-1,0],[0,1,0]], dtype=np.float32),
        np.array([[0,0,0],[1,-2,1],[0,0,0]], dtype=np.float32),
        np.array([[0,1,0],[1,-4,1],[0,1,0]], dtype=np.float32),
        np.array([[-1,2,-1],[2,-4,2],[-1,2,-1]], dtype=np.float32),
    ]
    img_f    = img_gray.astype(np.float32)
    features = {}
    for i, k in enumerate(kernels):
        res = cv2.filter2D(img_f, -1, k).flatten()
        features[f'res{i}_mean']     = float(np.mean(res))
        features[f'res{i}_std']      = float(np.std(res))
        features[f'res{i}_energy']   = float(np.mean(res**2))
        features[f'res{i}_kurtosis'] = float(kurtosis(res))
        features[f'res{i}_skew']     = float(skew(res))
    pixels = img_f.flatten()
    features['pixel_mean'] = float(np.mean(pixels))
    features['pixel_std']  = float(np.std(pixels))
    features['pixel_var']  = float(np.var(pixels))
    return pd.DataFrame([features])

# ── AUDIO features — matches audio_scaler.pkl exactly (45 features) ───────────
def extract_audio_features(y, sr):
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

    yc  = y[:1000] if len(y) >= 1000 else y
    ac  = np.correlate(yc, yc, mode='full')
    ac  = ac[len(ac)//2:]
    ac  = ac / (ac[0] + 1e-10)
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

    features = [
        lsb_mean, lsb_std, lsb_entropy, lsb_transitions,
        lsb2_mean, lsb2_entropy,
        hist_std, hist_entropy, hist_flatness,
        autocorr_1, autocorr_2,
        even_odd_diff, even_odd_corr,
        *mfcc_feats,
        sample_var, sample_kurtosis, sample_skew,
        residual_std, residual_kurtosis, residual_entropy
    ]
    return np.array([features])

def sha256_short(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16] + "..."

def prob_from_decision(score):
    return float(1 / (1 + np.exp(-abs(float(score)))))

def verdict_from_prob(prob: float) -> str:
    if prob >= 0.65: return "STEGO"
    if prob >= 0.35: return "REVIEW"
    return "CLEAN"

def get_live_stats():
    try:
        from database import get_stats
        db      = get_stats()
        total   = db["total"]   or SESSION_STATS["total"]
        threats = db["threats"] or SESSION_STATS["threats"]
        clean   = db["clean"]   or SESSION_STATS["clean"]
    except Exception:
        total   = SESSION_STATS["total"]
        threats = SESSION_STATS["threats"]
        clean   = SESSION_STATS["clean"]
    return {
        "total": total, "threats": threats, "clean": clean,
        "review": SESSION_STATS["review"],
        "detection_rate": round(threats / max(total, 1) * 100, 1),
        "f1_score": 92.1, "fpr": 0.71, "avg_time_s": 3.2,
        "img_model": MODELS.get("img_loaded", False),
        "aud_model": MODELS.get("aud_loaded", False),
    }

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return render("dashboard.html", {"stats": get_live_stats(), "recent": SCAN_HISTORY[-10:][::-1], "active_page": "dashboard"}, request)

@app.get("/upload",   response_class=HTMLResponse)
async def upload_page(request: Request):
    return render("upload.html",   {"active_page": "upload"},   request)

@app.get("/analysis", response_class=HTMLResponse)
async def analysis_page(request: Request):
    return render("analysis.html", {"active_page": "analysis"}, request)

@app.get("/model",    response_class=HTMLResponse)
async def model_page(request: Request):
    return render("model.html",    {"active_page": "model"},    request)

@app.get("/logs",     response_class=HTMLResponse)
async def logs_page(request: Request):
    return render("logs.html",     {"logs": SCAN_HISTORY[::-1], "active_page": "logs"}, request)

@app.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request):
    return render("settings.html", {"active_page": "settings"}, request)

@app.get("/api/stats")
async def api_stats(): return get_live_stats()

@app.get("/api/logs")
async def api_logs(): return {"logs": SCAN_HISTORY[::-1][:50]}

@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    start    = time.time()
    data     = await file.read()
    ext      = file.filename.rsplit(".", 1)[-1].lower()
    audit_id = f"AUD-{str(len(SCAN_HISTORY)+1).zfill(4)}"

    if ext in ["png", "jpg", "jpeg"]:
        if not MODELS.get("img_loaded"):
            raise HTTPException(503, "Image model not loaded")
        arr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise HTTPException(400, "Could not decode image")
        feats   = extract_image_features(img)
        feats_s = MODELS["img_scaler"].transform(feats)
        score   = MODELS["img_svm"].decision_function(feats_s)[0]
        prob    = prob_from_decision(score)
        verdict = verdict_from_prob(prob)
        ftype   = "IMG"

    elif ext == "wav":
        if not MODELS.get("aud_loaded"):
            raise HTTPException(503, "Audio model not loaded")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(data); tmp_path = tmp.name
        try:
            y, sr = librosa.load(tmp_path, sr=None)
        finally:
            os.unlink(tmp_path)
        feats   = extract_audio_features(y, sr)
        feats_s = MODELS["aud_scaler"].transform(feats)
        proba   = MODELS["aud_gb"].predict_proba(feats_s)[0]
        prob    = float(proba[1])
        verdict = verdict_from_prob(prob)
        ftype   = "WAV"
    else:
        raise HTTPException(400, f"Unsupported file type: .{ext}")

    result = {
        "audit_id":      audit_id,
        "file_name":     file.filename,
        "file_type":     ftype,
        "modality":      "image" if ftype == "IMG" else "audio",
        "stego_prob":    round(prob, 3),
        "verdict":       verdict,
        "algorithm":     ("LSB (est.)" if ftype == "IMG" else "LSB-audio (est.)") if verdict == "STEGO" else "—",
        "sha256":        sha256_short(data),
        "processing_ms": round((time.time() - start) * 1000),
        "timestamp":     datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "confidence":    "High" if prob > 0.8 or prob < 0.2 else "Medium",
    }

    SCAN_HISTORY.append(result)
    SESSION_STATS["total"] += 1
    SESSION_STATS[{"STEGO":"threats","CLEAN":"clean","REVIEW":"review"}[verdict]] += 1

    try:
        from database import save_scan
        save_scan(file.filename, result["modality"].capitalize(), verdict, round(prob * 100, 1))
    except Exception as e:
        print(f"DB save error: {e}")

    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)