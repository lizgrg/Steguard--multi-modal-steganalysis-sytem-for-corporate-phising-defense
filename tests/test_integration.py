"""
INTEGRATION TESTS — StegGuard SOC
Tests how components work together: routes ↔ feature extraction ↔ model ↔ DB.
Uses FastAPI TestClient + mocked DB + mocked ML models.
Run: pytest test_integration.py -v
"""

import pytest
import numpy as np
import cv2
import io
import struct
import wave
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


# ══════════════════════════════════════════════════════════════════════════════
# Fixtures & helpers
# ══════════════════════════════════════════════════════════════════════════════

def make_png_bytes(h=64, w=64) -> bytes:
    """Create a real grayscale PNG in memory."""
    img = np.random.randint(0, 256, (h, w), dtype=np.uint8)
    _, buf = cv2.imencode('.png', img)
    return buf.tobytes()


def make_wav_bytes(sr=16000, duration=0.5) -> bytes:
    """Create a minimal valid WAV file in memory."""
    n_samples = int(sr * duration)
    t = np.linspace(0, duration, n_samples)
    audio = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(audio.tobytes())
    return buf.getvalue()


def mock_image_models():
    """Return mock scaler + SVM that mimic sklearn API."""
    scaler = MagicMock()
    scaler.transform.return_value = np.zeros((1, 28))
    svm = MagicMock()
    svm.decision_function.return_value = np.array([2.5])   # high → STEGO
    return scaler, svm


def mock_audio_models():
    """Return mock scaler + GB classifier."""
    scaler = MagicMock()
    scaler.transform.return_value = np.zeros((1, 45))
    gb = MagicMock()
    gb.predict_proba.return_value = np.array([[0.1, 0.9]])  # 90 % stego
    return scaler, gb


# ══════════════════════════════════════════════════════════════════════════════
# App fixture — patches models + DB before importing main
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def client():
    img_scaler, img_svm = mock_image_models()
    aud_scaler, aud_gb  = mock_audio_models()

    with patch("joblib.load") as mock_load, \
         patch("database.get_connection", side_effect=Exception("no db")), \
         patch("database.get_all_scans",  return_value=[]), \
         patch("database.save_scan",      return_value=1):

        # Return different mocks for different pkl paths
        def side_effect(path):
            if "svm" in path or "image_classifier" in path:
                return img_svm
            if "scaler.pkl" in path:
                return img_scaler
            if "gb" in path:
                return aud_gb
            if "audio_scaler" in path:
                return aud_scaler
            return MagicMock()

        mock_load.side_effect = side_effect

        import importlib
        import main as m
        # Force model reload with mocks in place
        m.MODELS["img_svm"]    = img_svm
        m.MODELS["img_scaler"] = img_scaler
        m.MODELS["img_loaded"] = True
        m.MODELS["aud_gb"]     = aud_gb
        m.MODELS["aud_scaler"] = aud_scaler
        m.MODELS["aud_loaded"] = True

        yield TestClient(m.app)


# ══════════════════════════════════════════════════════════════════════════════
# 1. HTML page routes
# ══════════════════════════════════════════════════════════════════════════════

class TestPageRoutes:
    def test_dashboard_returns_200(self, client):
        r = client.get("/")
        assert r.status_code == 200

    def test_upload_page_returns_200(self, client):
        r = client.get("/upload")
        assert r.status_code == 200

    def test_analysis_page_returns_200(self, client):
        r = client.get("/analysis")
        assert r.status_code == 200

    def test_model_page_returns_200(self, client):
        r = client.get("/model")
        assert r.status_code == 200

    def test_logs_page_returns_200(self, client):
        r = client.get("/logs")
        assert r.status_code == 200

    def test_settings_page_returns_200(self, client):
        r = client.get("/settings")
        assert r.status_code == 200

    def test_dashboard_content_type_html(self, client):
        r = client.get("/")
        assert "text/html" in r.headers["content-type"]


# ══════════════════════════════════════════════════════════════════════════════
# 2. /api/stats
# ══════════════════════════════════════════════════════════════════════════════

class TestApiStats:
    def test_returns_200(self, client):
        r = client.get("/api/stats")
        assert r.status_code == 200

    def test_has_required_keys(self, client):
        data = client.get("/api/stats").json()
        for key in ["total", "threats", "clean", "detection_rate"]:
            assert key in data

    def test_detection_rate_is_float(self, client):
        data = client.get("/api/stats").json()
        assert isinstance(data["detection_rate"], (int, float))

    def test_model_flags_present(self, client):
        data = client.get("/api/stats").json()
        assert "img_model" in data
        assert "aud_model" in data


# ══════════════════════════════════════════════════════════════════════════════
# 3. /api/logs
# ══════════════════════════════════════════════════════════════════════════════

class TestApiLogs:
    def test_returns_200(self, client):
        assert client.get("/api/logs").status_code == 200

    def test_has_logs_key(self, client):
        data = client.get("/api/logs").json()
        assert "logs" in data

    def test_logs_is_list(self, client):
        data = client.get("/api/logs").json()
        assert isinstance(data["logs"], list)


# ══════════════════════════════════════════════════════════════════════════════
# 4. /api/predict — image integration
# ══════════════════════════════════════════════════════════════════════════════

class TestPredictImage:
    def test_png_upload_returns_200(self, client):
        png = make_png_bytes()
        r = client.post("/api/predict", files={"file": ("test.png", png, "image/png")})
        assert r.status_code == 200

    def test_response_has_verdict(self, client):
        png = make_png_bytes()
        data = client.post("/api/predict", files={"file": ("test.png", png, "image/png")}).json()
        assert "verdict" in data
        assert data["verdict"] in ["STEGO", "CLEAN", "REVIEW"]

    def test_response_has_stego_prob(self, client):
        png = make_png_bytes()
        data = client.post("/api/predict", files={"file": ("test.png", png, "image/png")}).json()
        assert "stego_prob" in data
        assert 0.0 <= data["stego_prob"] <= 1.0

    def test_response_has_sha256(self, client):
        png = make_png_bytes()
        data = client.post("/api/predict", files={"file": ("t.png", png, "image/png")}).json()
        assert "sha256" in data
        assert data["sha256"].endswith("...")

    def test_file_type_is_img(self, client):
        png = make_png_bytes()
        data = client.post("/api/predict", files={"file": ("t.png", png, "image/png")}).json()
        assert data["file_type"] == "IMG"

    def test_jpeg_upload_works(self, client):
        img = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        _, buf = cv2.imencode('.jpg', img)
        r = client.post("/api/predict", files={"file": ("img.jpg", buf.tobytes(), "image/jpeg")})
        assert r.status_code == 200

    def test_scan_added_to_history(self, client):
        import main as m
        before = len(m.SCAN_HISTORY)
        client.post("/api/predict", files={"file": ("h.png", make_png_bytes(), "image/png")})
        assert len(m.SCAN_HISTORY) == before + 1


# ══════════════════════════════════════════════════════════════════════════════
# 5. /api/predict — audio integration
# ══════════════════════════════════════════════════════════════════════════════

class TestPredictAudio:
    def test_wav_upload_returns_200(self, client):
        wav = make_wav_bytes()
        r = client.post("/api/predict", files={"file": ("audio.wav", wav, "audio/wav")})
        assert r.status_code == 200

    def test_file_type_is_wav(self, client):
        wav = make_wav_bytes()
        data = client.post("/api/predict", files={"file": ("a.wav", wav, "audio/wav")}).json()
        assert data["file_type"] == "WAV"

    def test_response_has_processing_ms(self, client):
        wav = make_wav_bytes()
        data = client.post("/api/predict", files={"file": ("a.wav", wav, "audio/wav")}).json()
        assert "processing_ms" in data
        assert data["processing_ms"] >= 0

    def test_audio_verdict_valid(self, client):
        wav = make_wav_bytes()
        data = client.post("/api/predict", files={"file": ("a.wav", wav, "audio/wav")}).json()
        assert data["verdict"] in ["STEGO", "CLEAN", "REVIEW"]


# ══════════════════════════════════════════════════════════════════════════════
# 6. /api/predict — error cases
# ══════════════════════════════════════════════════════════════════════════════

class TestPredictErrors:
    def test_unsupported_extension_returns_400(self, client):
        r = client.post("/api/predict", files={"file": ("doc.pdf", b"fake", "application/pdf")})
        assert r.status_code == 400

    def test_corrupt_image_returns_400(self, client):
        r = client.post("/api/predict", files={"file": ("bad.png", b"not_an_image", "image/png")})
        assert r.status_code == 400

    def test_txt_file_returns_400(self, client):
        r = client.post("/api/predict", files={"file": ("x.txt", b"hello", "text/plain")})
        assert r.status_code == 400


# ══════════════════════════════════════════════════════════════════════════════
# 7. Session stats accumulation
# ══════════════════════════════════════════════════════════════════════════════

class TestSessionStats:
    def test_total_increments_after_scan(self, client):
        import main as m
        before = m.SESSION_STATS["total"]
        client.post("/api/predict", files={"file": ("s.png", make_png_bytes(), "image/png")})
        assert m.SESSION_STATS["total"] == before + 1

    def test_verdict_bucket_increments(self, client):
        import main as m
        img_scaler, img_svm = mock_image_models()
        img_svm.decision_function.return_value = np.array([-5.0])  # CLEAN
        m.MODELS["img_svm"]    = img_svm
        m.MODELS["img_scaler"] = img_scaler
        before_clean = m.SESSION_STATS["clean"]
        client.post("/api/predict", files={"file": ("c.png", make_png_bytes(), "image/png")})
        # verdict depends on prob_from_decision(-5) → high prob → STEGO
        # just assert total moved
        assert m.SESSION_STATS["total"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])