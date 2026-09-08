import logging
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch

from app.config import settings
from app.data import OceanEmbedDataLoader
from app.schemas import DepthPrediction, PredictionRequest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from scripts.inference.oceanembed_inference import (
    DEPTHS_M,
    HISTORY_DAYS,
    INPUT_CHANNELS,
    INPUT_HEIGHT,
    INPUT_WIDTH,
    OUTPUT_CHANNELS,
    OUTPUT_HEIGHT,
    OUTPUT_WIDTH,
    INPUT_FEATURES,
    OceanEmbedEnsemble,
)

logger = logging.getLogger("oceanembed.model")


class OceanEmbedModel:
    """Service wrapper around the existing validated OceanEmbed V1 ensemble."""

    def __init__(self):
        self.model_version = settings.model_version
        self.loaded = False
        self._ensemble: OceanEmbedEnsemble | None = None
        self._data: OceanEmbedDataLoader | None = None

    def load(self) -> None:
        self.loaded = False
        self._ensemble = None
        if self._data is not None:
            self._data.close()
        self._data = None

        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"

            # OceanEmbedEnsemble is the source of truth for:
            # architecture, V1 config/statistics, checkpoint loading,
            # normalization and three-seed inference.
            self._ensemble = OceanEmbedEnsemble(device=device)

            # Fail startup if the real harmonized data are unavailable.
            self._data = OceanEmbedDataLoader()

            self.loaded = True
            logger.info(
                "OceanEmbed V1 ensemble and real harmonized data loaded on %s",
                device,
            )
        except Exception:
            logger.exception("Failed to load real OceanEmbed inference resources")
            if self._data is not None:
                self._data.close()
            self._data = None
            self._ensemble = None
            self.loaded = False

    def predict(self, request: PredictionRequest) -> List[DepthPrediction]:
        if not self.loaded or self._ensemble is None or self._data is None:
            raise RuntimeError("Real OceanEmbed model/data are not loaded")

        feature_arrays, metadata = self._data.get_window(
            request.date,
            request.latitude,
            request.longitude,
        )

        # This is the authoritative V1 preprocessing path.
        normalized_input = self._ensemble.normalize_input_window(feature_arrays)
        if tuple(normalized_input.shape) != (
            INPUT_CHANNELS,
            INPUT_HEIGHT,
            INPUT_WIDTH,
        ):
            raise RuntimeError(
                f"Unexpected normalized input shape: {tuple(normalized_input.shape)}"
            )

        if not torch.isfinite(normalized_input).all().item():
            raise RuntimeError("Normalized OceanEmbed input contains non-finite values")

        logger.info(
            "OceanEmbed input: shape=%s date=%s window=%s..%s tile=(%s,%s) "
            "grid=(%.2f,%.2f)",
            tuple(normalized_input.shape),
            metadata["target_date"],
            metadata["window_start"],
            metadata["window_end"],
            metadata["tile_row"],
            metadata["tile_col"],
            metadata["snapped_latitude"],
            metadata["snapped_longitude"],
        )

        # Verified interface in scripts/inference/oceanembed_inference.py.
        prediction = self._ensemble.predict_single(normalized_input)

        expected_output = (
            OUTPUT_CHANNELS,
            OUTPUT_HEIGHT,
            OUTPUT_WIDTH,
        )
        if tuple(prediction.shape) != expected_output:
            raise RuntimeError(
                f"Unexpected OceanEmbed output shape: {tuple(prediction.shape)}; "
                f"expected {expected_output}"
            )
        if not torch.isfinite(prediction).all().item():
            raise RuntimeError("OceanEmbed prediction contains non-finite values")

        row = int(metadata["output_row"])
        col = int(metadata["output_col"])
        prediction_array = prediction.detach().cpu().numpy()

        depth_to_index = {depth: i for i, depth in enumerate(DEPTHS_M)}
        results: list[DepthPrediction] = []

        for depth in request.depths:
            if depth not in depth_to_index:
                raise ValueError(
                    f"Unsupported depth {depth}m. Supported depths: {DEPTHS_M}"
                )
            depth_index = depth_to_index[depth]
            results.append(
                DepthPrediction(
                    depth_m=depth,
                    temperature_c=round(
                        float(prediction_array[depth_index, row, col]),
                        3,
                    ),
                    uncertainty_c=None,
                )
            )

        logger.info(
            "OceanEmbed output: shape=%s requested_depths=%s",
            tuple(prediction.shape),
            request.depths,
        )
        return results

    @torch.no_grad()
    def predict_input(self, model_input: torch.Tensor | np.ndarray) -> np.ndarray:
        """Run the verified ensemble on a normalized [49,64,64] tensor."""
        if not self.loaded or self._ensemble is None:
            raise RuntimeError("Model is not loaded")

        tensor = torch.as_tensor(model_input, dtype=torch.float32)
        if tuple(tensor.shape) != (INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH):
            raise ValueError(
                f"Expected model input shape "
                f"{(INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH)}, "
                f"got {tuple(tensor.shape)}"
            )

        prediction = self._ensemble.predict_single(tensor)
        result = prediction.detach().cpu().numpy()
        if result.shape != (OUTPUT_CHANNELS, OUTPUT_HEIGHT, OUTPUT_WIDTH):
            raise RuntimeError(f"Unexpected prediction shape: {result.shape}")
        return result

    def get_model_info(self) -> dict:
        return {
            "project": "OceanF",
            "experiment": "OceanEmbed-CNN",
            "experiment_variant": "E2_7day_retrospective",
            "checkpoint_version": "V1",
            "ensemble_type": "three_seed_mean",
            "seeds": [42, 123, 2024],
            "history_days": HISTORY_DAYS,
            "input_features": list(INPUT_FEATURES),
            "input_channels": INPUT_CHANNELS,
            "input_tile_size": INPUT_WIDTH,
            "output_channels": OUTPUT_CHANNELS,
            "output_tile_size": OUTPUT_WIDTH,
            "grid_resolution_deg": settings.grid_resolution_deg,
            "depths_m": list(DEPTHS_M),
            "normalization_source": "data/processed/ML/ml_config.json",
            "model_loaded": self.loaded,
        }


model = OceanEmbedModel()
