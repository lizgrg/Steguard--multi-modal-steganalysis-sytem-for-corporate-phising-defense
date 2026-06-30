"""
BLACKBOX TESTS — StegGuard SOC
Tests the system purely from the outside: valid inputs → expected output contracts.
No knowledge of internal implementation assumed.
Run: pytest tests/test_blackbox.py -v
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app")))

import pytest
import numpy as np
import cv2
import io
import wave
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


# ══════════════════════════════════════════════════════════════════════════════
# Helpers — all use sine waves / random data, never constant signals
# ══════════════════════════════════════════════════════════════════════════════

def make_png(h=64, w=64) -> bytes:
    rng = np.random.RandomState(7)
    img = rng.randint(0, 256, (h, w), dtype=np.uint8)
    _, buf = cv2.imencode('.png', img)
    return buf.tobytes()

def make_jpg(h=64, w=64) -> bytes:
    rng = np.random.RandomState(8)
    img = rng.randint(0, 256, (h, w), dtype=np.uint8)
    _, buf = cv2.imencode('.jpg', img)
    return buf.tobytes()

def make_wav(sr=16000, duration=0.5) -> bytes:
    n = int(sr * duration)
    t = np.linspace(0, duration, n)
    # Mix of two frequencies for richer signal
    audio = ((np.sin(2 * np.pi * 440 * t) + np.sin(2 * np.pi * 880 * t)) * 16383).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
        wf.writeframes(audio.tobytes())
    return buf.getvalue()


# ══════════════════════════════════════════════════════════════════════════════
# App fixture
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def client():
    img_scaler = MagicMock(); img_scaler.transform.return_value = np.zeros((1, 28))
    img_svm    = MagicMock(); img_svm.decision_function.return_value = np.array([1.2])
    aud_scaler = MagicMock(); aud_scaler.transform.return_value = np.zeros((1, 45))
    aud_gb     = MagicMock(); aud_gb.predict_proba.return_value = np.array([[0.4, 0.6]])

    with patch("joblib.load", return_value=MagicMock()), \
         patch("database.get_all_scans", return_value=[]), \
         patch("database.get_stats", return_value={"total": 0, "threats": 0, "clean": 0}), \
         patch("database.save_scan", return_value=1):

        import main as m
        m.MODELS.update({
            "img_svm":    img_svm,    "img_scaler": img_scaler, "img_loaded": True,
            "aud_gb":     aud_gb,     "aud_scaler": aud_scaler, "aud_loaded": True,
        })
        with TestClient(m.app) as c:
            yield c


# ══════════════════════════════════════════════════════════════════════════════
# 1. Response contract — all required fields must be present
# ══════════════════════════════════════════════════════════════════════════════

REQUIRED_FIELDS = {
    "audit_id", "file_name", "file_type", "stego_prob",
    "verdict", "sha256", "processing_ms", "timestamp",
    "confidence", "algorithm", "modality"
}

class TestResponseContract:
    def test_png_has_all_fields(self, client):
        data = client.post("/api/predict", files={"file": ("f.png", make_png(), "image/png")}).json()
        assert REQUIRED_FIELDS.issubset(data.keys())

    def test_jpg_has_all_fields(self, client):
        data = client.post("/api/predict", files={"file": ("f.jpg", make_jpg(), "image/jpeg")}).json()
        assert REQUIRED_FIELDS.issubset(data.keys())

    def test_wav_has_all_fields(self, client):
        data = client.post("/api/predict", files={"file": ("f.wav", make_wav(), "audio/wav")}).json()
        assert REQUIRED_FIELDS.issubset(data.keys())

    def test_verdict_is_always_valid(self, client):
        for _ in range(3):
            data = client.post("/api/predict", files={"file": ("f.png", make_png(), "image/png")}).json()
            assert data["verdict"] in {"STEGO", "CLEAN", "REVIEW"}

    def test_stego_prob_between_0_and_1(self, client):
        data = client.post("/api/predict", files={"file": ("f.png", make_png(), "image/png")}).json()
        assert 0.0 <= data["stego_prob"] <= 1.0

    def test_confidence_label_valid(self, client):
        data = client.post("/api/predict", files={"file": ("f.png", make_png(), "image/png")}).json()
        assert data["confidence"] in {"High", "Medium"}

    def test_audit_id_starts_with_AUD(self, client):
        data = client.post("/api/predict", files={"file": ("f.png", make_png(), "image/png")}).json()
        assert data["audit_id"].startswith("AUD-")

    def test_sha256_ends_with_ellipsis(self, client):
        data = client.post("/api/predict", files={"file": ("f.png", make_png(), "image/png")}).json()
        assert data["sha256"].endswith("...")

    def test_processing_ms_non_negative(self, client):
        data = client.post("/api/predict", files={"file": ("f.png", make_png(), "image/png")}).json()
        assert data["processing_ms"] >= 0

    def test_file_name_echoed_back(self, client):
        data = client.post("/api/predict", files={"file": ("myfile.png", make_png(), "image/png")}).json()
        assert data["file_name"] == "myfile.png"

    def test_timestamp_present_and_string(self, client):
        data = client.post("/api/predict", files={"file": ("f.png", make_png(), "image/png")}).json()
        assert isinstance(data["timestamp"], str)
        assert len(data["timestamp"]) > 0


# ══════════════════════════════════════════════════════════════════════════════
# 2. /api/stats contract
# ══════════════════════════════════════════════════════════════════════════════

STATS_FIELDS = {"total", "threats", "clean", "review", "detection_rate", "f1_score", "fpr", "avg_time_s"}

class TestStatsContract:
    def test_has_required_fields(self, client):
        data = client.get("/api/stats").json()
        assert STATS_FIELDS.issubset(data.keys())

    def test_total_non_negative(self, client):
        data = client.get("/api/stats").json()
        assert isinstance(data["total"], int)
        assert data["total"] >= 0

    def test_detection_rate_in_range(self, client):
        data = client.get("/api/stats").json()
        assert 0.0 <= data["detection_rate"] <= 100.0

    def test_f1_score_present(self, client):
        assert "f1_score" in client.get("/api/stats").json()

    def test_fpr_present(self, client):
        assert "fpr" in client.get("/api/stats").json()


# ══════════════════════════════════════════════════════════════════════════════
# 3. Input validation — unsupported formats must be rejected
# ══════════════════════════════════════════════════════════════════════════════

class TestInputValidation:
    @pytest.mark.parametrize("filename,content,mime", [
        ("file.pdf",  b"%PDF-1.4",   "application/pdf"),
        ("file.txt",  b"hello",       "text/plain"),
        ("file.mp3",  b"ID3",         "audio/mpeg"),
        ("file.gif",  b"GIF89a",      "image/gif"),
        ("file.docx", b"PK\x03\x04", "application/vnd.openxmlformats"),
    ])
    def test_unsupported_format_returns_400(self, client, filename, content, mime):
        r = client.post("/api/predict", files={"file": (filename, content, mime)})
        assert r.status_code == 400

    def test_corrupt_png_returns_400(self, client):
        r = client.post("/api/predict", files={"file": ("x.png", b"\x89PNG\x0d\x0a", "image/png")})
        assert r.status_code == 400

    def test_mp3_extension_rejected(self, client):
        # .mp3 is not a supported extension — should return 400
        r = client.post("/api/predict", files={"file": ("audio.mp3", b"ID3\x00", "audio/mpeg")})
        assert r.status_code == 400

    def test_error_body_has_detail(self, client):
        r = client.post("/api/predict", files={"file": ("x.txt", b"hi", "text/plain")})
        assert "detail" in r.json()


# ══════════════════════════════════════════════════════════════════════════════
# 4. Various image sizes accepted
# ══════════════════════════════════════════════════════════════════════════════

class TestImageSizes:
    @pytest.mark.parametrize("h,w", [
        (32, 32), (64, 64), (128, 128), (256, 256), (100, 200)
    ])
    def test_png_size_accepted(self, client, h, w):
        r = client.post("/api/predict", files={"file": ("t.png", make_png(h, w), "image/png")})
        assert r.status_code == 200

    def test_wide_image_accepted(self, client):
        r = client.post("/api/predict", files={"file": ("wide.png", make_png(32, 512), "image/png")})
        assert r.status_code == 200


# ══════════════════════════════════════════════════════════════════════════════
# 5. Various WAV durations accepted
# ══════════════════════════════════════════════════════════════════════════════

class TestAudioLengths:
    @pytest.mark.parametrize("duration", [0.1, 0.5, 1.0, 3.0])
    def test_wav_duration_accepted(self, client, duration):
        r = client.post("/api/predict", files={"file": ("a.wav", make_wav(duration=duration), "audio/wav")})
        assert r.status_code == 200


# ══════════════════════════════════════════════════════════════════════════════
# 6. Page routes return HTML
# ══════════════════════════════════════════════════════════════════════════════

class TestPageContentTypes:
    @pytest.mark.parametrize("path", ["/", "/upload", "/analysis", "/model", "/logs", "/settings"])
    def test_page_returns_html(self, client, path):
        r = client.get(path)
        assert r.status_code == 200
        assert "text/html" in r.headers.get("content-type", "")


# ══════════════════════════════════════════════════════════════════════════════
# 7. API routes return JSON
# ══════════════════════════════════════════════════════════════════════════════

class TestApiContentTypes:
    @pytest.mark.parametrize("path", ["/api/stats", "/api/logs"])
    def test_api_returns_json(self, client, path):
        r = client.get(path)
        assert r.status_code == 200
        assert "application/json" in r.headers.get("content-type", "")


# ══════════════════════════════════════════════════════════════════════════════
# 8. Idempotency — same file must give same sha256 and stego_prob
# ══════════════════════════════════════════════════════════════════════════════

class TestIdempotency:
    def test_same_png_twice_both_200(self, client):
        png = make_png()
        r1 = client.post("/api/predict", files={"file": ("s.png", png, "image/png")})
        r2 = client.post("/api/predict", files={"file": ("s.png", png, "image/png")})
        assert r1.status_code == 200
        assert r2.status_code == 200

    def test_same_png_same_sha256(self, client):
        png = make_png()
        d1 = client.post("/api/predict", files={"file": ("s.png", png, "image/png")}).json()
        d2 = client.post("/api/predict", files={"file": ("s.png", png, "image/png")}).json()
        assert d1["sha256"] == d2["sha256"]

    def test_same_png_same_stego_prob(self, client):
        png = make_png()
        d1 = client.post("/api/predict", files={"file": ("s.png", png, "image/png")}).json()
        d2 = client.post("/api/predict", files={"file": ("s.png", png, "image/png")}).json()
        assert d1["stego_prob"] == d2["stego_prob"]

    def test_same_wav_same_sha256(self, client):
        wav = make_wav()
        d1 = client.post("/api/predict", files={"file": ("a.wav", wav, "audio/wav")}).json()
        d2 = client.post("/api/predict", files={"file": ("a.wav", wav, "audio/wav")}).json()
        assert d1["sha256"] == d2["sha256"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])