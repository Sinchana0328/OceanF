import logging
import math
from typing import List

from app.config import settings
from app.schemas import DepthPrediction, PredictionRequest

logger = logging.getLogger("oceanembed.model")


class OceanEmbedModel:
    """
    Wrapper around the OceanEmbed subsurface temperature reconstruction model.

    Reference approach (from the research papers this service is built against):
      - Subsurface Temperature Reconstruction for the Global Ocean from 1993-2020
        Using Satellite Observations and Deep Learning (MDPI Remote Sensing 14(13):3198)
      - A Deep Learning Method for Inversing 3D Temperature Fields Using Sea Surface
        Data in Offshore China and the Northwest Pacific Ocean (MDPI JMSE 12(12):2337)

    Model contract:
      INPUT  -> lat, lon, date, surface variables (SST, SSS, SSH/SLA, wind) on a
                0.25 deg x 0.25 deg daily grid
      OUTPUT -> predicted subsurface temperature at requested depth levels

    Training data is GLORYS reanalysis; ARGO float profiles are used purely for
    independent validation and are never mixed into training, to avoid leakage.

    This class is the ONLY place that should be touched to plug in the real
    trained PyTorch model. Everything above/below it (FastAPI routes, Spring
    Boot orchestration) is agnostic to how predict() is implemented.
    """

    def __init__(self):
        self.model_version = settings.model_version
        self.loaded = False
        self._torch_model = None

    def load(self) -> None:
        """
        Load model weights. Currently runs in stub mode (physically-plausible
        synthetic profile) since no trained weights are present. Swap in:

            import torch
            self._torch_model = torch.load(settings.model_weights_path, map_location="cpu")
            self._torch_model.eval()

        and self.loaded = True once the file exists.
        """
        try:
            self.loaded = True
            logger.info(
                "OceanEmbed model '%s' ready (stub mode - no trained weights found, "
                "using placeholder physical profile).", self.model_version
            )
        except Exception as exc:
            logger.error("Failed to load model weights: %s", exc)
            self.loaded = False

    def predict(self, request: PredictionRequest) -> List[DepthPrediction]:
        if not self.loaded:
            raise RuntimeError("Model is not loaded")

        sst = request.surface.sst
        out: List[DepthPrediction] = []
        for depth in request.depths:
            temp = self._synthetic_profile(sst, depth, request.latitude)
            uncertainty = round(0.15 + depth * 0.0005, 3)
            out.append(DepthPrediction(depth_m=depth, temperature_c=round(temp, 3), uncertainty_c=uncertainty))
        return out

    @staticmethod
    def _synthetic_profile(sst: float, depth: int, latitude: float) -> float:
        """
        Placeholder ONLY: mimics thermocline decay (temperature falls off
        roughly exponentially with depth toward a latitude-dependent deep
        water temperature). Replace with real model inference.
        """
        deep_water_temp = 4.0 - 0.03 * abs(latitude)
        decay_scale = 150.0
        return deep_water_temp + (sst - deep_water_temp) * math.exp(-depth / decay_scale)


model = OceanEmbedModel()
