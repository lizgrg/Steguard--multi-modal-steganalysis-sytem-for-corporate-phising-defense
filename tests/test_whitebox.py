"""
WHITEBOX TESTS — StegGuard SOC
Tests internal logic, branches, and edge cases with full knowledge of the code.
Covers: boundary conditions, all if/elif branches, error paths, internal state.
Run: pytest test_whitebox.py -v
"""

import pytest
import numpy as np
import pandas as pd
import cv2
import io
import wave
import json
import hashlib
from unittest.mock import patch, MagicMock, call
from scipy.stats import skew, kurtosis


# ══════════════════════════════════════════════════════════════════════════════
# Replicated internals (whitebox — we know the implementation)
# ══════════════════════════════════════════════════════════════════════════════

def sha256_short(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16] + "..."

def prob_from_decision(score):
    return float(1 / (1 + np.exp(-abs(float(score)))))

def verdict_from_prob(prob: float) -> str:
    if prob >= 0.65: return "STEGO"
    if prob >= 0.35: return "REVIEW"
    return "CLEAN"

def extract_image_features(img_gray):
    kernels = [
        np.array([[0,0,0],[0,-1,1],[0,0,0]], dtype=np.float32),
        np.array([[0,0,0],[0,-1,0],[0,1,0]], dtype=np.float32),
        np.array([[0,0,0],[1,-2,1],[0,0,0]], dtype=np.float32),
        np.array([[0,1,0],[1,-4,1],[0,1,0]], dtype=np.float32),
        np.array([[-1,2,-1],[2,-4,2],[-1,2,-1]], dtype=np.float32),
    ]
    img_f = img_gray.astype(np.float32)
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

def extract_audio_features(y, sr):
    import librosa
    samples = (y * 32767).astype(np.int16)
    lsb  = samples & 1
    lsb2 = (samples >> 1) & 1
    lsb_mean = float(np.mean(lsb))
    lsb_std  = float(np.std(lsb))
    p = np.clip(lsb_mean, 1e-10, 1-1e-10)
    lsb_entropy     = float(-p*np.log2(p) - (1-p)*np.log2(1-p))
    lsb_transitions = float(np.sum(np.diff(lsb.astype(int)) != 0) / max(len(lsb)-1, 1))
    lsb2_mean = float(np.mean(lsb2))
    p2 = np.clip(lsb2_mean, 1e-10, 1-1e-10)
    lsb2_entropy = float(-p2*np.log2(p2) - (1-p2)*np.log2(1-p2))
    hist, _ = np.histogram(y, bins=256, density=True)
    hist = hist + 1e-10
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
    return np.array([[
        lsb_mean, lsb_std, lsb_entropy, lsb_transitions,
        lsb2_mean, lsb2_entropy,
        hist_std, hist_entropy, hist_flatness,
        autocorr_1, autocorr_2,
        even_odd_diff, even_odd_corr,
        *mfcc_feats,
        sample_var, sample_kurtosis, sample_skew,
        residual_std, residual_kurtosis, residual_entropy
    ]])

def embed_lsb(audio, message_bits=None):
    samples = (audio * 32767).astype(np.int16)
    if message_bits is None:
        message_bits = np.random.randint(0, 2, len(samples))
    samples = (samples & ~1) | message_bits[:len(samples)]
    return samples.astype(np.float32) / 32767.0


# ══════════════════════════════════════════════════════════════════════════════
# 1. verdict_from_prob — every branch
# ══════════════════════════════════════════════════════════════════════════════

class TestVerdictBranches:
    """Cover all three if/elif/else branches explicitly."""

    def test_branch_stego_exact_boundary(self):
        # Branch: prob >= 0.65
        assert verdict_from_prob(0.65) == "STEGO"

    def test_branch_stego_above_boundary(self):
        assert verdict_from_prob(0.66) == "STEGO"

    def test_branch_review_lower_exact(self):
        # Branch: prob >= 0.35 (but < 0.65)
        assert verdict_from_prob(0.35) == "REVIEW"

    def test_branch_review_upper_just_below(self):
        assert verdict_from_prob(0.6499) == "REVIEW"

    def test_branch_clean_just_below_review(self):
        # Branch: else (prob < 0.35)
        assert verdict_from_prob(0.3499) == "CLEAN"

    def test_branch_clean_zero(self):
        assert verdict_from_prob(0.0) == "CLEAN"

    def test_branch_stego_one(self):
        assert verdict_from_prob(1.0) == "STEGO"


# ══════════════════════════════════════════════════════════════════════════════
# 2. prob_from_decision — internal sigmoid with abs()
# ══════════════════════════════════════════════════════════════════════════════

class TestProbFromDecisionInternals:
    def test_abs_applied_negative(self):
        """Negative score treated same as positive due to abs()."""
        assert prob_from_decision(-3.0) == prob_from_decision(3.0)

    def test_sigmoid_formula(self):
        """Verify exact sigmoid formula: 1/(1+exp(-|score|))."""
        score = 2.0
        expected = 1 / (1 + np.exp(-abs(score)))
        assert abs(prob_from_decision(score) - expected) < 1e-9

    def test_score_0_exactly_half(self):
        assert abs(prob_from_decision(0.0) - 0.5) < 1e-9

    def test_very_large_score_approaches_1(self):
        assert prob_from_decision(100.0) > 0.9999

    def test_small_score_near_half(self):
        p = prob_from_decision(0.001)
        assert 0.499 < p < 0.501


# ══════════════════════════════════════════════════════════════════════════════
# 3. extract_image_features — internal kernel loop (5 kernels × 5 stats)
# ══════════════════════════════════════════════════════════════════════════════

class TestImageFeaturesInternals:
    def _img(self, fill=128, h=64, w=64):
        return np.full((h, w), fill, dtype=np.uint8)

    def test_exactly_5_kernel_groups(self):
        df = extract_image_features(self._img())
        kernel_means = [c for c in df.columns if c.endswith('_mean') and c.startswith('res')]
        assert len(kernel_means) == 5  # res0…res4

    def test_each_kernel_has_5_stats(self):
        df = extract_image_features(self._img())
        for i in range(5):
            for stat in ['mean', 'std', 'energy', 'kurtosis', 'skew']:
                assert f'res{i}_{stat}' in df.columns

    def test_pixel_stats_present(self):
        df = extract_image_features(self._img())
        assert 'pixel_mean' in df.columns
        assert 'pixel_std'  in df.columns
        assert 'pixel_var'  in df.columns

    def test_pixel_var_equals_std_squared(self):
        rng = np.random.RandomState(42)
        img = rng.randint(0, 256, (64, 64), dtype=np.uint8)
        df  = extract_image_features(img)
        assert abs(df['pixel_var'].iloc[0] - df['pixel_std'].iloc[0]**2) < 1e-3

    def test_energy_non_negative(self):
        df = extract_image_features(self._img(fill=100))
        for i in range(5):
            assert df[f'res{i}_energy'].iloc[0] >= 0

    def test_uniform_image_all_residuals_zero(self):
        """A flat image has zero response to edge-detecting kernels."""
        df = extract_image_features(self._img(fill=200))
        for i in range(5):
            assert abs(df[f'res{i}_mean'].iloc[0]) < 1e-3

    def test_total_feature_count_is_28(self):
        df = extract_image_features(self._img())
        assert len(df.columns) == 28  # 5*5 + 3


# ══════════════════════════════════════════════════════════════════════════════
# 4. extract_audio_features — internal LSB logic
# ══════════════════════════════════════════════════════════════════════════════

class TestAudioFeaturesInternals:
    def _sine(self, n=4000, sr=16000):
        t = np.linspace(0, n/sr, n)
        return np.sin(2 * np.pi * 440 * t).astype(np.float32), sr

    def test_feature_vector_length_45(self):
        y, sr = self._sine()
        feats = extract_audio_features(y, sr)
        assert feats.shape == (1, 45)

    def test_lsb_mean_in_0_1_range(self):
        """LSB values are 0 or 1, so mean must be in [0,1]."""
        y, sr = self._sine()
        feats = extract_audio_features(y, sr)
        lsb_mean = feats[0, 0]
        assert 0.0 <= lsb_mean <= 1.0

    def test_lsb_entropy_non_negative(self):
        y, sr = self._sine()
        feats = extract_audio_features(y, sr)
        lsb_entropy = feats[0, 2]
        assert lsb_entropy >= 0.0

    def test_lsb_transitions_in_0_1(self):
        """Transition rate is proportion, must be in [0,1]."""
        y, sr = self._sine()
        feats = extract_audio_features(y, sr)
        transitions = feats[0, 3]
        assert 0.0 <= transitions <= 1.0

    def test_autocorr_indices_not_nan(self):
        y, sr = self._sine()
        feats = extract_audio_features(y, sr)
        assert not np.isnan(feats[0, 9])   # autocorr_1
        assert not np.isnan(feats[0, 10])  # autocorr_2

    def test_very_short_audio_handled(self):
        """Audio shorter than 1000 samples triggers the autocorr short-path branch.
        Use a sine wave — constant signals produce NaN kurtosis/corrcoef."""
        t  = np.linspace(0, 500/16000, 500)
        y  = (np.sin(2 * np.pi * 440 * t)).astype(np.float32) * 0.5
        sr = 16000
        feats = extract_audio_features(y, sr)
        assert feats.shape == (1, 45)

    def test_stego_audio_has_higher_lsb_entropy(self):
        """LSB-embedded audio should have higher LSB entropy than silent audio."""
        silence = np.zeros(4000, dtype=np.float32)
        stego   = embed_lsb(np.zeros(4000, dtype=np.float32),
                            message_bits=np.random.randint(0, 2, 4000))
        f_silence = extract_audio_features(silence + 0.001, 16000)
        f_stego   = extract_audio_features(stego,           16000)
        assert f_stego[0, 2] >= f_silence[0, 2]  # lsb_entropy index=2

    def test_no_nan_in_output(self):
        y, sr = self._sine()
        feats = extract_audio_features(y, sr)
        assert not np.isnan(feats).any()


# ══════════════════════════════════════════════════════════════════════════════
# 5. embed_lsb — internal bit manipulation
# ══════════════════════════════════════════════════════════════════════════════

class TestEmbedLsbInternals:
    def test_lsb_cleared_before_setting(self):
        """AND with ~1 clears LSB, then OR sets it."""
        audio = np.full(10, 0.5, dtype=np.float32)
        bits  = np.zeros(10, dtype=np.int16)
        out   = embed_lsb(audio, message_bits=bits)
        samples = (out * 32767).astype(np.int16)
        assert np.all(samples & 1 == 0)  # all LSBs = 0

    def test_bit_1_sets_lsb(self):
        audio = np.full(10, 0.5, dtype=np.float32)
        bits  = np.ones(10, dtype=np.int16)
        out   = embed_lsb(audio, message_bits=bits)
        samples = (out * 32767).astype(np.int16)
        assert np.all(samples & 1 == 1)

    def test_only_lsb_changed(self):
        """Upper bits should be unchanged (within ±1 due to int16 floor)."""
        audio = np.full(100, 0.5, dtype=np.float32)
        bits  = np.zeros(100, dtype=np.int16)
        out   = embed_lsb(audio, message_bits=bits)
        original = (audio * 32767).astype(np.int16)
        result   = (out  * 32767).astype(np.int16)
        diff     = np.abs(original.astype(int) - result.astype(int))
        assert np.all(diff <= 1)

    def test_default_bits_random(self):
        """Without explicit bits, output should differ from input."""
        np.random.seed(99)
        audio = np.full(1000, 0.3, dtype=np.float32)
        out   = embed_lsb(audio)
        assert not np.array_equal(audio, out)


# ══════════════════════════════════════════════════════════════════════════════
# 6. database.py — internal branching (mocked)
# ══════════════════════════════════════════════════════════════════════════════

class TestDatabaseInternals:
    def test_save_scan_returns_none_on_exception(self):
        with patch("mysql.connector.connect", side_effect=Exception("conn fail")):
            import database
            result = database.save_scan("file.png", "image", "stego", 85.0)
            assert result is None

    def test_get_all_scans_returns_empty_on_exception(self):
        with patch("mysql.connector.connect", side_effect=Exception("fail")):
            import database
            result = database.get_all_scans()
            assert result == []

    def test_get_stats_returns_zeros_on_exception(self):
        with patch("mysql.connector.connect", side_effect=Exception("fail")):
            import database
            result = database.get_stats()
            assert result == {'total': 0, 'threats': 0, 'clean': 0}

    def test_init_db_returns_false_on_failure(self):
        with patch("mysql.connector.connect", side_effect=Exception("fail")):
            import database
            assert database.init_db() == False

    def test_init_db_returns_true_on_success(self):
        mock_conn = MagicMock()
        with patch("mysql.connector.connect", return_value=mock_conn):
            import database
            assert database.init_db() == True

    def test_save_scan_image_uses_model_id_1(self):
        """Whitebox: image file_type maps to model_id=1."""
        mock_conn  = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.lastrowid = 42
        mock_conn.cursor.return_value = mock_cursor
        with patch("mysql.connector.connect", return_value=mock_conn):
            import database
            database.save_scan("img.png", "image", "stego", 90.0)
            # Second execute: DETECTION_RESULT insert with model_id=1
            second_call_args = mock_cursor.execute.call_args_list[1]
            assert 1 in second_call_args[0][1]  # model_id=1 for image

    def test_save_scan_audio_uses_model_id_2(self):
        """Whitebox: audio file_type maps to model_id=2."""
        mock_conn   = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.lastrowid = 43
        mock_conn.cursor.return_value = mock_cursor
        with patch("mysql.connector.connect", return_value=mock_conn):
            import database
            database.save_scan("audio.wav", "audio", "clean", 10.0)
            second_call_args = mock_cursor.execute.call_args_list[1]
            assert 2 in second_call_args[0][1]  # model_id=2 for audio

    def test_save_scan_commits_transaction(self):
        mock_conn   = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.lastrowid = 1
        mock_conn.cursor.return_value = mock_cursor
        with patch("mysql.connector.connect", return_value=mock_conn):
            import database
            database.save_scan("f.png", "image", "clean", 5.0)
            mock_conn.commit.assert_called_once()

    def test_features_json_serialized(self):
        """Whitebox: features dict should be JSON-serialised before insert."""
        mock_conn   = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.lastrowid = 5
        mock_conn.cursor.return_value = mock_cursor
        feats = {"pixel_mean": 128.0}
        with patch("mysql.connector.connect", return_value=mock_conn):
            import database
            database.save_scan("f.png", "image", "stego", 80.0, features=feats)
            second_call_args = mock_cursor.execute.call_args_list[1]
            params = second_call_args[0][1]
            json_arg = params[-1]
            assert json.loads(json_arg) == feats

    def test_save_scan_none_features_inserts_null(self):
        mock_conn   = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.lastrowid = 6
        mock_conn.cursor.return_value = mock_cursor
        with patch("mysql.connector.connect", return_value=mock_conn):
            import database
            database.save_scan("f.png", "image", "clean", 5.0, features=None)
            second_call_args = mock_cursor.execute.call_args_list[1]
            params = second_call_args[0][1]
            assert params[-1] is None


# ══════════════════════════════════════════════════════════════════════════════
# 7. get_stats — whitebox: clean = total - threats
# ══════════════════════════════════════════════════════════════════════════════

class TestGetStatsCalculation:
    def test_clean_derived_correctly(self):
        mock_conn   = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.side_effect = [
            {'total': 20},
            {'threats': 7}
        ]
        mock_conn.cursor.return_value = mock_cursor
        with patch("mysql.connector.connect", return_value=mock_conn):
            import database
            result = database.get_stats()
            assert result['total']   == 20
            assert result['threats'] == 7
            assert result['clean']   == 13  # 20 - 7


# ══════════════════════════════════════════════════════════════════════════════
# 8. sha256_short — internal slicing
# ══════════════════════════════════════════════════════════════════════════════

class TestSha256Internals:
    def test_exactly_first_16_hex_chars(self):
        data = b"test"
        full = hashlib.sha256(data).hexdigest()
        result = sha256_short(data)
        assert result == full[:16] + "..."

    def test_hex_chars_only_in_prefix(self):
        result = sha256_short(b"hello")
        prefix = result[:-3]  # strip "..."
        assert all(c in "0123456789abcdef" for c in prefix)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])