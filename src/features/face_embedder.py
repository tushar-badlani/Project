import os
from contextlib import nullcontext
from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np

if TYPE_CHECKING:
    from src.utils.profiler import PipelineProfiler


class FaceEmbedder:
    """ArcFace-based face embedding extractor using InsightFace.

    Uses InsightFace's buffalo_l model for face detection + ArcFace
    embedding extraction (512-d). Falls back gracefully if no face
    is detected or if the face is too small.
    """

    def __init__(self, config: dict, profiler: Optional["PipelineProfiler"] = None):
        face_cfg = config["features"]["face"]
        self.enabled = face_cfg["enabled"]
        self.min_face_size = face_cfg["min_face_size"]
        self._profiler = profiler
        self._device = config["device"]

        if self.enabled:
            import insightface  # noqa: F401
            from insightface.app import FaceAnalysis

            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if self._device == "cpu":
                providers = ["CPUExecutionProvider"]

            # buffalo_sc is much faster on CPU than buffalo_l
            model_name = "buffalo_sc" if self._device == "cpu" else "buffalo_l"
            self._model_name = model_name
            self.app = FaceAnalysis(
                name=model_name,
                providers=providers,
            )
            self.app.prepare(
                ctx_id=0 if self._device != "cpu" else -1,
                det_size=(320, 320),  # smaller det input for speed
            )

            if profiler is not None:
                self._profile_models()

    def _profile_models(self) -> None:
        """Count parameters for InsightFace ONNX models (detection + recognition)."""
        try:
            import numpy as _np
            import onnx  # type: ignore

            insightface_root = os.path.expanduser("~/.insightface/models")
            model_dir = os.path.join(insightface_root, self._model_name)

            total_params = 0
            found_files = []
            if os.path.isdir(model_dir):
                for fname in os.listdir(model_dir):
                    if fname.endswith(".onnx"):
                        fpath = os.path.join(model_dir, fname)
                        try:
                            proto = onnx.load(fpath)
                            n = sum(
                                int(_np.prod(init.dims))
                                for init in proto.graph.initializer
                                if len(init.dims) > 0
                            )
                            found_files.append((fname, n))
                            total_params += n
                        except Exception:
                            pass

            if found_files:
                self._profiler.record_model_profile(
                    "InsightFace (ArcFace)", None, total_params
                )
            else:
                self._profiler.record_model_profile("InsightFace (ArcFace)", None, None)

        except Exception as exc:
            self._profiler.record_model_profile("InsightFace (ArcFace)", None, None)
            print(f"[Profiler] InsightFace param count failed: {exc}")

    def extract(self, crop: np.ndarray) -> Optional[np.ndarray]:
        """Extract face embedding from a person crop.

        Args:
            crop: BGR person crop image.

        Returns:
            512-d normalized embedding, or None if face not found/too small.
        """
        if not self.enabled:
            return None

        ctx = self._profiler.time_stage("insightface") if self._profiler else nullcontext()
        with ctx:
            # InsightFace expects BGR input (which OpenCV provides)
            faces = self.app.get(crop)

        if not faces:
            return None

        # Pick the largest face
        best_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

        # Check face size
        face_w = best_face.bbox[2] - best_face.bbox[0]
        face_h = best_face.bbox[3] - best_face.bbox[1]
        if face_w < self.min_face_size or face_h < self.min_face_size:
            return None

        embedding = best_face.embedding
        if embedding is None:
            return None

        # L2 normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding.astype(np.float32)
