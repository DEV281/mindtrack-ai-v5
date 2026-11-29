"""Mamba SSM (Selective State Space Model) for temporal pattern analysis.

Architecture matches ml_training/train_temporal.py exactly.
When trained weights are available (temporal_mamba.pt), the model uses them.
Otherwise, it falls back to heuristic analysis.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

EMOTION_LABELS = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]


class TemporalAnalysisModel:
    """Temporal analysis using Mamba SSM for sequence patterns.

    When trained weights are loaded, runs real Mamba SSM inference on
    temporal feature sequences extracted from analysis history.

    When no weights are available, falls back to heuristic analysis.
    """

    def __init__(self) -> None:
        self.model = None
        self.device = "cpu"
        self._loaded = False
        self._weights_loaded = False

    def load(self, weights_path: Optional[str] = None) -> None:
        """Build the Mamba SSM model and optionally load trained weights."""
        try:
            import torch
            import torch.nn as nn

            self.device = "cuda" if torch.cuda.is_available() else "cpu"

            # ── SSM Block (matches training script) ──

            class SelectiveScanSSM(nn.Module):
                def __init__(self, d_model=64, state_dim=16, dt_rank=8, d_inner=32):
                    super().__init__()
                    self.d_model = d_model
                    self.state_dim = state_dim
                    self.dt_rank = dt_rank
                    self.d_inner = d_inner

                    self.in_proj = nn.Linear(d_model, d_inner * 2)
                    self.A_log = nn.Parameter(torch.randn(d_inner, state_dim))
                    self.D = nn.Parameter(torch.ones(d_inner))
                    self.x_proj = nn.Linear(d_inner, dt_rank + state_dim * 2, bias=False)
                    self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)
                    self.out_proj = nn.Linear(d_inner, d_model)
                    self.norm = nn.LayerNorm(d_inner)

                    # HiPPO init
                    A = torch.arange(1, state_dim + 1, dtype=torch.float32).unsqueeze(0).repeat(d_inner, 1)
                    self.A_log.data = torch.log(A)
                    nn.init.uniform_(self.dt_proj.bias, -4.0, -2.0)

                def forward(self, x):
                    batch, seq_len, _ = x.shape
                    xz = self.in_proj(x)
                    x_in, z = xz.chunk(2, dim=-1)
                    x_dbl = self.x_proj(x_in)
                    dt, B, C = x_dbl.split([self.dt_rank, self.state_dim, self.state_dim], dim=-1)
                    dt = torch.softplus(self.dt_proj(dt))
                    A = -torch.exp(self.A_log)
                    y = self._selective_scan(x_in, dt, A, B, C)
                    y = self.norm(y)
                    y = y * torch.silu(z)
                    y = self.out_proj(y)
                    return y

                def _selective_scan(self, u, dt, A, B, C):
                    batch, seq_len, d_inner = u.shape
                    h = torch.zeros(batch, d_inner, self.state_dim, device=u.device, dtype=u.dtype)
                    outputs = []
                    for t in range(seq_len):
                        dt_t = dt[:, t, :].unsqueeze(-1)
                        A_bar = torch.exp(A.unsqueeze(0) * dt_t)
                        B_t = B[:, t, :].unsqueeze(1).expand(-1, d_inner, -1)
                        B_bar = dt_t * B_t
                        u_t = u[:, t, :].unsqueeze(-1)
                        h = A_bar * h + B_bar * u_t
                        C_t = C[:, t, :].unsqueeze(1).expand(-1, d_inner, -1)
                        y_t = (C_t * h).sum(dim=-1) + self.D * u[:, t, :]
                        outputs.append(y_t)
                    return torch.stack(outputs, dim=1)

            # ── Full Model (matches training script MambaTemporalModel) ──

            class MambaTemporalModel(nn.Module):
                def __init__(self, input_dim=7, d_model=64, n_layers=4, state_dim=16, d_inner=32, num_classes=8):
                    super().__init__()
                    self.input_proj = nn.Linear(input_dim, d_model)
                    self.pos_embed = nn.Parameter(torch.randn(1, 100, d_model) * 0.02)
                    self.layers = nn.ModuleList([
                        nn.ModuleDict({
                            "ssm": SelectiveScanSSM(d_model, state_dim, d_inner=d_inner),
                            "norm": nn.LayerNorm(d_model),
                            "ffn": nn.Sequential(
                                nn.Linear(d_model, d_model * 4),
                                nn.GELU(),
                                nn.Dropout(0.1),
                                nn.Linear(d_model * 4, d_model),
                                nn.Dropout(0.1),
                            ),
                            "ffn_norm": nn.LayerNorm(d_model),
                        })
                        for _ in range(n_layers)
                    ])
                    self.head = nn.Sequential(
                        nn.LayerNorm(d_model),
                        nn.Linear(d_model, 128),
                        nn.GELU(),
                        nn.Dropout(0.2),
                        nn.Linear(128, num_classes),
                    )

                def forward(self, x):
                    batch, seq_len, _ = x.shape
                    x = self.input_proj(x)
                    x = x + self.pos_embed[:, :seq_len, :]
                    for layer in self.layers:
                        residual = x
                        x = layer["norm"](x)
                        x = layer["ssm"](x) + residual
                        residual = x
                        x = layer["ffn_norm"](x)
                        x = layer["ffn"](x) + residual
                    x = x.mean(dim=1)
                    return self.head(x)

            self.model = MambaTemporalModel(
                input_dim=7, d_model=64, n_layers=4,
                state_dim=16, d_inner=32, num_classes=8,
            ).to(self.device)

            if weights_path:
                try:
                    state_dict = torch.load(weights_path, map_location=self.device, weights_only=True)
                    self.model.load_state_dict(state_dict)
                    self._weights_loaded = True
                    logger.info(f"Temporal Mamba SSM weights loaded from {weights_path}")
                except Exception as e:
                    logger.warning(f"Could not load temporal weights: {e}")
                    self._weights_loaded = False

            self.model.eval()
            self._loaded = True
            logger.info(f"Temporal Mamba SSM model built on {self.device}")

        except Exception as e:
            logger.warning(f"Mamba SSM model not available: {e}")
            self._loaded = False

    def predict(self, history: List[Dict]) -> Dict:
        """Predict temporal patterns from analysis history.

        Args:
            history: List of recent analysis results (at least 2 frames).

        Returns:
            Dict with temporal_stress, pattern, trend.
        """
        if len(history) < 2:
            return {"temporal_stress": 0.0, "pattern": "insufficient_data", "trend": "stable"}

        # Extract feature sequences (7 dims matching training)
        features = []
        for frame in history[-10:]:  # up to 10 frames
            feat = [
                frame.get("stress", 0) / 100.0,
                frame.get("anxiety", 0) / 100.0,
                frame.get("stability", 50) / 100.0,
                frame.get("depression_risk", 0) / 100.0,
                frame.get("overall_risk", 0) / 100.0,
                frame.get("voice_freq", 180) / 500.0,
                frame.get("confidence", 0.5),
            ]
            features.append(feat)

        # Pad to at least 5 frames
        while len(features) < 5:
            features.insert(0, features[0])

        feature_array = np.array(features, dtype=np.float32)

        # Run Mamba SSM inference if model + weights available
        if self._loaded and self._weights_loaded and self.model is not None:
            try:
                import torch
                x = torch.from_numpy(feature_array).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    logits = self.model(x)[0].cpu().numpy()

                probs = np.exp(logits) / np.sum(np.exp(logits))  # softmax
                pred_class = int(np.argmax(probs))
                pred_emotion = EMOTION_LABELS[pred_class]

                # Stress-related classes: sad(3), angry(4), fearful(5), disgust(6)
                stress_indices = [3, 4, 5, 6]
                temporal_stress = float(sum(probs[i] for i in stress_indices)) * 100

                pattern = self._classify_pattern(feature_array)

                # Trend from feature sequence
                stresses = feature_array[:, 0]
                diff = np.diff(stresses)
                mean_diff = float(np.mean(diff)) if len(diff) > 0 else 0
                if mean_diff > 0.05:
                    trend = "increasing"
                elif mean_diff < -0.05:
                    trend = "decreasing"
                else:
                    trend = "stable"

                return {
                    "temporal_stress": min(100, temporal_stress),
                    "pattern": pattern,
                    "trend": trend,
                    "predicted_emotion": pred_emotion,
                    "confidence": float(probs[pred_class]),
                }
            except Exception as e:
                logger.warning(f"Temporal inference error: {e}")

        # Heuristic fallback
        return self._heuristic_predict(feature_array)

    def _classify_pattern(self, features: np.ndarray) -> str:
        """Classify the temporal pattern from feature sequence."""
        stresses = features[:, 0]
        if len(stresses) < 2:
            return "stable"

        diff = np.diff(stresses)
        mean_diff = float(np.mean(diff))
        std_diff = float(np.std(diff))

        if abs(mean_diff) < 0.05 and std_diff < 0.1:
            return "stable"
        elif mean_diff > 0.1:
            return "escalating"
        elif mean_diff < -0.1:
            return "recovering"
        elif std_diff > 0.15:
            return "volatile"
        else:
            return "fluctuating"

    def _heuristic_predict(self, features: np.ndarray) -> Dict:
        """Heuristic temporal analysis fallback."""
        stresses = features[:, 0]
        mean_stress = float(np.mean(stresses)) * 100
        diff = np.diff(stresses)
        mean_diff = float(np.mean(diff)) if len(diff) > 0 else 0

        if mean_diff > 0.05:
            trend = "increasing"
        elif mean_diff < -0.05:
            trend = "decreasing"
        else:
            trend = "stable"

        pattern = self._classify_pattern(features)

        return {
            "temporal_stress": mean_stress,
            "pattern": pattern,
            "trend": trend,
        }
