from typing import Optional

import cv2
import numpy as np


class FaceEmbedder:
    """ArcFace-based face embedding extractor using InsightFace.

    Uses InsightFace's buffalo_l model for face detection + ArcFace
    embedding extraction (512-d). Falls back gracefully if no face
    is detected or if the face is too small.
    """

    def __init__(self, config: dict):
        face_cfg = config["features"]["face"]
        self.enabled = face_cfg["enabled"]
        self.min_face_size = face_cfg["min_face_size"]

        if self.enabled:
            import insightface
            from insightface.app import FaceAnalysis

            # Use buffalo_l model (includes detection + recognition)
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if config["device"] == "cpu":
                providers = ["CPUExecutionProvider"]

            # buffalo_sc is much faster on CPU than buffalo_l
            model_name = "buffalo_sc" if config["device"] == "cpu" else "buffalo_l"
            self.app = FaceAnalysis(
                name=model_name,
                providers=providers,
            )
            self.app.prepare(
                ctx_id=0 if config["device"] != "cpu" else -1,
                det_size=(320, 320),  # smaller det input for speed
            )

    def extract(self, crop: np.ndarray) -> Optional[np.ndarray]:
        """Extract face embedding from a person crop.

        Args:
            crop: BGR person crop image.

        Returns:
            512-d normalized embedding, or None if face not found/too small.
        """
        if not self.enabled:
            return None

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
