"""Tests for temporal Mamba SSM model inference."""

import numpy as np
import pytest


class TestTemporalAnalysisModel:
    """Test suite for TemporalAnalysisModel."""

    def test_import(self):
        from backend.app.models.temporal_model import TemporalAnalysisModel, EMOTION_LABELS
        assert len(EMOTION_LABELS) == 8

    def test_instantiate(self):
        from backend.app.models.temporal_model import TemporalAnalysisModel
        model = TemporalAnalysisModel()
        assert model.model is None
        assert model._loaded is False
        assert model._weights_loaded is False

    def test_predict_insufficient_data(self):
        from backend.app.models.temporal_model import TemporalAnalysisModel
        model = TemporalAnalysisModel()
        result = model.predict([])
        assert result["pattern"] == "insufficient_data"
        assert result["temporal_stress"] == 0.0

    def test_predict_single_frame(self):
        from backend.app.models.temporal_model import TemporalAnalysisModel
        model = TemporalAnalysisModel()
        result = model.predict([{"stress": 50}])
        assert result["pattern"] == "insufficient_data"

    def test_heuristic_predict_stable(self):
        from backend.app.models.temporal_model import TemporalAnalysisModel
        model = TemporalAnalysisModel()
        history = [{"stress": 40, "anxiety": 30, "stability": 60, "depression_risk": 20,
                     "overall_risk": 30, "voice_freq": 180, "confidence": 0.8}] * 5
        result = model.predict(history)
        assert result["trend"] == "stable"
        assert result["pattern"] == "stable"

    def test_heuristic_predict_increasing(self):
        from backend.app.models.temporal_model import TemporalAnalysisModel
        model = TemporalAnalysisModel()
        history = []
        for i in range(5):
            history.append({"stress": 20 + i * 15, "anxiety": 20 + i * 10,
                            "stability": 70 - i * 10, "depression_risk": 10 + i * 8,
                            "overall_risk": 20 + i * 12, "voice_freq": 180, "confidence": 0.7})
        result = model.predict(history)
        assert result["trend"] == "increasing"

    def test_heuristic_predict_decreasing(self):
        from backend.app.models.temporal_model import TemporalAnalysisModel
        model = TemporalAnalysisModel()
        history = []
        for i in range(5):
            history.append({"stress": 80 - i * 15, "anxiety": 70 - i * 10,
                            "stability": 30 + i * 10, "depression_risk": 60 - i * 8,
                            "overall_risk": 70 - i * 12, "voice_freq": 180, "confidence": 0.7})
        result = model.predict(history)
        assert result["trend"] == "decreasing"

    def test_classify_pattern_stable(self):
        from backend.app.models.temporal_model import TemporalAnalysisModel
        model = TemporalAnalysisModel()
        features = np.array([[0.4, 0.3, 0.6, 0.2, 0.3, 0.36, 0.5]] * 5, dtype=np.float32)
        assert model._classify_pattern(features) == "stable"

    def test_classify_pattern_escalating(self):
        from backend.app.models.temporal_model import TemporalAnalysisModel
        model = TemporalAnalysisModel()
        features = np.array([[0.1 + i * 0.15, 0.3, 0.6, 0.2, 0.3, 0.36, 0.5]
                              for i in range(5)], dtype=np.float32)
        assert model._classify_pattern(features) == "escalating"

    def test_classify_pattern_recovering(self):
        from backend.app.models.temporal_model import TemporalAnalysisModel
        model = TemporalAnalysisModel()
        features = np.array([[0.8 - i * 0.15, 0.3, 0.6, 0.2, 0.3, 0.36, 0.5]
                              for i in range(5)], dtype=np.float32)
        assert model._classify_pattern(features) == "recovering"

    def test_predict_returns_all_fields(self):
        from backend.app.models.temporal_model import TemporalAnalysisModel
        model = TemporalAnalysisModel()
        history = [{"stress": 50, "anxiety": 40, "stability": 55, "depression_risk": 25,
                     "overall_risk": 35, "voice_freq": 190, "confidence": 0.75}] * 5
        result = model.predict(history)
        assert "temporal_stress" in result
        assert "pattern" in result
        assert "trend" in result

    def test_temporal_stress_range(self):
        from backend.app.models.temporal_model import TemporalAnalysisModel
        model = TemporalAnalysisModel()
        history = [{"stress": 90, "anxiety": 85, "stability": 15, "depression_risk": 70,
                     "overall_risk": 80, "voice_freq": 300, "confidence": 0.9}] * 5
        result = model.predict(history)
        assert 0 <= result["temporal_stress"] <= 100
