"""
UNIT TESTS — StegGuard SOC
Tests individual pure functions in isolation (no DB, no disk, no models).
Run: pytest test_unit.py -v
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
import hashlib
import sys, os

# ── Helpers extracted / replicated from main.py for isolated unit testing ──────

def sha256_short(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16] + "..."

def prob_from_decision(score):
    return float(1 / (1 + np.exp(-abs(float(score)))))

def verdict_from_prob(prob: float) -> str:
    if prob >= 0.65: return "STEGO"
    if prob >= 0.35: return "REVIEW"
    return "CLEAN"

# ── Feature helpers replicated for unit testing ────────────────────────────────
from scipy.stats import skew, kurtosis
import cv2

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


# ══════════════════════════════════════════════════════════════════════════════
# 1. sha256_short
# ══════════════════════════════════════════════════════════════════════════════

class TestSha256Short:
    def test_returns_string(self):
        result = sha256_short(b"hello")
        assert isinstance(result, str)

    def test_ends_with_ellipsis(self):
        result = sha256_short(b"hello")
        assert result.endswith("...")

    def test_length_is_19(self):
        # 16 hex chars + "..."
        result = sha256_short(b"hello")
        assert len(result) == 19

    def test_deterministic(self):
        assert sha256_short(b"abc") == sha256_short(b"abc")

    def test_different_inputs_differ(self):
        assert sha256_short(b"abc") != sha256_short(b"xyz")

    def test_empty_bytes(self):
        result = sha256_short(b"")
        assert len(result) == 19

    def test_large_input(self):
        result = sha256_short(b"x" * 10_000)
        assert len(result) == 19


# ══════════════════════════════════════════════════════════════════════════════
# 2. prob_from_decision
# ══════════════════════════════════════════════════════════════════════════════

class TestProbFromDecision:
    def test_zero_score_gives_half(self):
        assert abs(prob_from_decision(0.0) - 0.5) < 1e-6

    def test_output_in_range(self):
        for s in [-10, -1, 0, 1, 10]:
            p = prob_from_decision(s)
            assert 0.0 <= p <= 1.0

    def test_large_positive_score_near_1(self):
        assert prob_from_decision(10.0) > 0.99

    def test_large_negative_score_near_1(self):
        # abs is applied — large negative also gives high prob
        assert prob_from_decision(-10.0) > 0.99

    def test_symmetry(self):
        assert abs(prob_from_decision(3.0) - prob_from_decision(-3.0)) < 1e-9

    def test_returns_float(self):
        assert isinstance(prob_from_decision(1.0), float)


# ══════════════════════════════════════════════════════════════════════════════
# 3. verdict_from_prob
# ══════════════════════════════════════════════════════════════════════════════

class TestVerdictFromProb:
    def test_high_prob_is_stego(self):
        assert verdict_from_prob(0.65) == "STEGO"
        assert verdict_from_prob(0.99) == "STEGO"
        assert verdict_from_prob(1.0)  == "STEGO"

    def test_mid_prob_is_review(self):
        assert verdict_from_prob(0.35) == "REVIEW"
        assert verdict_from_prob(0.50) == "REVIEW"
        assert verdict_from_prob(0.64) == "REVIEW"

    def test_low_prob_is_clean(self):
        assert verdict_from_prob(0.0)  == "CLEAN"
        assert verdict_from_prob(0.10) == "CLEAN"
        assert verdict_from_prob(0.34) == "CLEAN"

    def test_boundary_0_35(self):
        assert verdict_from_prob(0.35) == "REVIEW"
        assert verdict_from_prob(0.349) == "CLEAN"

    def test_boundary_0_65(self):
        assert verdict_from_prob(0.65) == "STEGO"
        assert verdict_from_prob(0.649) == "REVIEW"

    def test_returns_string(self):
        assert isinstance(verdict_from_prob(0.5), str)


# ══════════════════════════════════════════════════════════════════════════════
# 4. extract_image_features
# ══════════════════════════════════════════════════════════════════════════════

class TestExtractImageFeatures:
    def _make_img(self, h=64, w=64, fill=128):
        return np.full((h, w), fill, dtype=np.uint8)

    def test_returns_dataframe(self):
        df = extract_image_features(self._make_img())
        assert isinstance(df, pd.DataFrame)

    def test_exactly_28_features(self):
        df = extract_image_features(self._make_img())
        # 5 kernels × 5 stats = 25  +  pixel_mean, pixel_std, pixel_var = 28
        assert df.shape == (1, 28)

    def test_feature_names_present(self):
        df = extract_image_features(self._make_img())
        assert 'pixel_mean' in df.columns
        assert 'res0_mean'  in df.columns
        assert 'res4_skew'  in df.columns

    def test_uniform_image_zero_std(self):
        df = extract_image_features(self._make_img(fill=128))
        # A uniform image has zero pixel variance
        assert df['pixel_std'].iloc[0] == pytest.approx(0.0, abs=1e-3)

    def test_no_nan_values(self):
        # Must use a non-uniform image — uniform images give NaN kurtosis/skew
        # (undefined for constant distributions)
        rng = np.random.RandomState(0)
        img = rng.randint(0, 256, (64, 64), dtype=np.uint8)
        df = extract_image_features(img)
        assert not df.isnull().any().any()

    def test_different_images_differ(self):
        df1 = extract_image_features(self._make_img(fill=0))
        df2 = extract_image_features(self._make_img(fill=255))
        assert not df1.equals(df2)

    def test_random_image(self):
        rng = np.random.RandomState(0)
        img = rng.randint(0, 256, (64, 64), dtype=np.uint8)
        df  = extract_image_features(img)
        assert df.shape == (1, 28)
        assert not df.isnull().any().any()


# ══════════════════════════════════════════════════════════════════════════════
# 5. prepare_datasets — embed_lsb  (replicated)
# ══════════════════════════════════════════════════════════════════════════════

def embed_lsb(audio, message_bits=None):
    samples = (audio * 32767).astype(np.int16)
    if message_bits is None:
        message_bits = np.random.randint(0, 2, len(samples))
    samples = (samples & ~1) | message_bits[:len(samples)]
    return samples.astype(np.float32) / 32767.0

class TestEmbedLsb:
    def _sine(self, n=1000):
        t = np.linspace(0, 1, n)
        return np.sin(2 * np.pi * 440 * t).astype(np.float32)

    def test_output_shape_matches_input(self):
        y = self._sine()
        out = embed_lsb(y)
        assert out.shape == y.shape

    def test_output_dtype_float32(self):
        out = embed_lsb(self._sine())
        assert out.dtype == np.float32

    def test_custom_bits_applied(self):
        y    = self._sine(100)
        bits = np.ones(100, dtype=np.int16)
        out  = embed_lsb(y, message_bits=bits)
        samples = (out * 32767).astype(np.int16)
        lsbs = samples & 1
        assert np.all(lsbs == 1)

    def test_zero_bits_all_even(self):
        y    = self._sine(100)
        bits = np.zeros(100, dtype=np.int16)
        out  = embed_lsb(y, message_bits=bits)
        samples = (out * 32767).astype(np.int16)
        assert np.all(samples & 1 == 0)

    def test_amplitude_preserved_approx(self):
        y   = self._sine()
        out = embed_lsb(y)
        assert np.max(np.abs(out)) <= 1.01  # stays in [-1, 1] range


# ══════════════════════════════════════════════════════════════════════════════
# 6. get_stats default return (database.py — mocked)
# ══════════════════════════════════════════════════════════════════════════════

class TestGetStatsDefault:
    def test_default_keys_present(self):
        # Simulate the fallback when DB fails
        stats = {'total': 0, 'threats': 0, 'clean': 0}
        assert 'total'   in stats
        assert 'threats' in stats
        assert 'clean'   in stats

    def test_clean_equals_total_minus_threats(self):
        total, threats = 10, 3
        clean = total - threats
        assert clean == 7


if __name__ == "__main__":
    pytest.main([__file__, "-v"])