from contextlib import nullcontext
from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np
import torch
import torchvision.transforms as T

if TYPE_CHECKING:
    from src.utils.profiler import PipelineProfiler


# Mapping from torchreid model names to timm equivalents
_TIMM_FALLBACK_MODELS = {
    "osnet_x1_0": "mobilenetv3_large_100.ra_in1k",
    "osnet_x0_75": "mobilenetv3_small_100.lamb_in1k",
    "osnet_x0_5": "mobilenetv3_small_050.lamb_in1k",
    "resnet50": "resnet50.a1_in1k",
    "resnet50_mid": "resnet50.a1_in1k",
}


class BodyEmbedder:
    """Body appearance embedding extractor for person re-identification.

    Uses torchreid's FeatureExtractor (OSNet) if available, otherwise
    falls back to a ResNet50 backbone via timm.
    """

    def __init__(self, config: dict, profiler: Optional["PipelineProfiler"] = None):
        body_cfg = config["features"]["body"]
        self.enabled = body_cfg["enabled"]
        self.device = torch.device(config["device"])
        self.input_size = tuple(body_cfg["input_size"])  # (H, W)
        self._profiler = profiler

        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize(self.input_size),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        self._use_torchreid = False
        if self.enabled:
            self.model = self._load_model(body_cfg["model"])
            if profiler is not None:
                self._profile_model()

    def _profile_model(self) -> None:
        """Profile FLOPs and params for the body backbone using thop."""
        try:
            # torchreid wraps the nn.Module; access it via .model.model
            if self._use_torchreid:
                nn_model = self.model.model
            else:
                nn_model = self.model

            H, W = self.input_size
            dummy = torch.zeros(1, 3, H, W).to(self.device)
            self._profiler.profile_torch_model("OSNet", nn_model, dummy)
        except Exception as exc:
            self._profiler.record_model_profile("OSNet", None, None)
            print(f"[Profiler] OSNet FLOPs profiling failed: {exc}")

    def _load_model(self, model_name: str):
        """Load model, trying torchreid first, then timm as fallback."""
        # Try torchreid
        try:
            from torchreid.utils import FeatureExtractor
            extractor = FeatureExtractor(
                model_name=model_name,
                model_path="",
                device=str(self.device),
            )
            self._use_torchreid = True
            print(f"[BodyEmbedder] Loaded torchreid model: {model_name}")
            return extractor
        except (ImportError, Exception):
            pass

        # Fallback to timm with mapped model name
        import timm
        timm_name = _TIMM_FALLBACK_MODELS.get(model_name, "resnet50.a1_in1k")
        model = timm.create_model(timm_name, pretrained=True, num_classes=0)
        model = model.eval().to(self.device)
        self._use_torchreid = False
        print(f"[BodyEmbedder] Loaded timm fallback model: {timm_name}")
        return model

    @torch.no_grad()
    def extract(self, crop: np.ndarray) -> np.ndarray:
        """Extract body appearance embedding from a person crop.

        Args:
            crop: BGR person crop image.

        Returns:
            Normalized embedding vector.
        """
        if not self.enabled:
            return None

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        ctx = self._profiler.time_stage("osnet") if self._profiler else nullcontext()
        with ctx:
            if self._use_torchreid:
                features = self.model([rgb])
                embedding = features.cpu().numpy().flatten()
            else:
                tensor = self.transform(rgb).unsqueeze(0).to(self.device)
                embedding = self.model(tensor).cpu().numpy().flatten()

        # L2 normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding.astype(np.float32)

    @torch.no_grad()
    def extract_batch(self, crops: list[np.ndarray]) -> np.ndarray:
        """Extract embeddings for multiple crops in a single forward pass."""
        if not self.enabled or len(crops) == 0:
            return None

        rgb_crops = [cv2.cvtColor(c, cv2.COLOR_BGR2RGB) for c in crops]

        ctx = self._profiler.time_stage("osnet") if self._profiler else nullcontext()
        with ctx:
            if self._use_torchreid:
                features = self.model(rgb_crops)
                embeddings = features.cpu().numpy()
            else:
                tensors = torch.stack(
                    [self.transform(c) for c in rgb_crops]
                ).to(self.device)
                embeddings = self.model(tensors).cpu().numpy()

        # L2 normalize each row
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        embeddings = embeddings / norms

        return embeddings.astype(np.float32)
