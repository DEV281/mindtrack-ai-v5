"""Tests for voice stress model inference."""

import numpy as np
import pytest


class TestVoiceStressModel:
    """Test suite for VoiceStressModel."""

    def test_import(self):
        from backend.app.models.voice_model import VoiceStressModel, VOICE_EMOTION_LABELS
        assert len(VOICE_EMOTION_LABELS) == 8
        assert "neutral" in VOICE_EMOTION_LABELS
        assert "angry" in VOICE_EMOTION_LABELS

    def test_instantiate(self):
        from backend.app.models.voice_model import VoiceStressModel
        model = VoiceStressModel()
        assert model.model is None
        assert model._loaded is False

    def test_default_result(self):
        from backend.app.models.voice_model import VoiceStressModel
        model = VoiceStressModel()
        result = model._default_result()
        assert result["stress_score"] == 0.0
        assert result["voice_emotion"] == "neutral"
        assert result["frequency"] == 0.0
        assert result["amplitude"] == 0.0
        assert result["confidence"] == 0.0
        assert result["mfcc_features"] == []

    def test_predict_empty_bytes(self):
        from backend.app.models.voice_model import VoiceStressModel
        model = VoiceStressModel()
        result = model.predict(b"")
        assert "stress_score" in result
        assert "voice_emotion" in result
        assert result["stress_score"] == 0.0

    def test_predict_short_audio(self):
        from backend.app.models.voice_model import VoiceStressModel
        model = VoiceStressModel()
        # Very short audio — should return default
        short_audio = np.zeros(100, dtype=np.float32).tobytes()
        result = model.predict(short_audio)
        assert result["stress_score"] == 0.0

    def test_heuristic_predict_high_pitch(self):
        from backend.app.models.voice_model import VoiceStressModel
        model = VoiceStressModel()
        audio = np.random.randn(16000).astype(np.float32)
        result = model._heuristic_predict(audio, 16000, 300.0, 0.5, None)
        assert result["voice_emotion"] in ["angry", "surprised"]
        assert result["stress_score"] >= 0

    def test_heuristic_predict_low_pitch(self):
        from backend.app.models.voice_model import VoiceStressModel
        model = VoiceStressModel()
        audio = np.random.randn(16000).astype(np.float32)
        result = model._heuristic_predict(audio, 16000, 100.0, 0.1, None)
        assert result["voice_emotion"] == "sad"

    def test_heuristic_predict_neutral(self):
        from backend.app.models.voice_model import VoiceStressModel
        model = VoiceStressModel()
        audio = np.random.randn(16000).astype(np.float32)
        result = model._heuristic_predict(audio, 16000, 170.0, 0.15, None)
        assert result["voice_emotion"] in ["neutral", "calm"]

    def test_extract_spectrogram_valid_audio(self):
        from backend.app.models.voice_model import VoiceStressModel
        model = VoiceStressModel()
        audio = np.random.randn(16000).astype(np.float32)
        spec = model.extract_spectrogram(audio, sr=16000)
        if spec is not None:
            assert spec.ndim == 2
            assert spec.shape[0] == 128  # n_mels

    def test_extract_mfcc_valid_audio(self):
        from backend.app.models.voice_model import VoiceStressModel
        model = VoiceStressModel()
        audio = np.random.randn(16000).astype(np.float32)
        mfcc = model.extract_mfcc(audio, sr=16000)
        if mfcc is not None:
            assert mfcc.ndim == 2
            assert mfcc.shape[0] == 40  # n_mfcc

    def test_stress_score_range(self):
        from backend.app.models.voice_model import VoiceStressModel
        model = VoiceStressModel()
        audio = np.random.randn(16000).astype(np.float32)
        result = model._heuristic_predict(audio, 16000, 200.0, 0.3, None)
        assert 0 <= result["stress_score"] <= 100

    def test_emotion_labels_complete(self):
        from backend.app.models.voice_model import VOICE_EMOTION_LABELS
        expected = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
        assert VOICE_EMOTION_LABELS == expected
