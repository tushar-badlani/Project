from typing import Optional

import cv2
import numpy as np


class GaitEmbedder:
    """Gait feature extractor using Gait Energy Image (GEI).

    Computes silhouettes from person crops over a tracklet, averages them
    into a Gait Energy Image, and reduces dimensionality via PCA to produce
    a compact gait embedding.
    """

    def __init__(self, config: dict):
        gait_cfg = config["features"]["gait"]
        self.enabled = gait_cfg["enabled"]
        self.min_tracklet_length = gait_cfg["min_tracklet_length"]
        self.embedding_dim = gait_cfg["embedding_dim"]
        self.pca_components = gait_cfg.get("pca_components", self.embedding_dim)

        # Standardized silhouette size for GEI computation
        self._sil_h = 128
        self._sil_w = 64

        # PCA projection matrix — fitted lazily on first batch of GEIs
        self._pca_matrix: Optional[np.ndarray] = None
        self._pca_mean: Optional[np.ndarray] = None

        # Background subtractor for silhouette extraction
        self._bg_subtractors: dict[int, cv2.BackgroundSubtractorMOG2] = {}

    def _get_silhouette(self, crop: np.ndarray) -> np.ndarray:
        """Convert a person crop to a binary silhouette."""
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (self._sil_w, self._sil_h))

        # Adaptive threshold for foreground extraction
        _, binary = cv2.threshold(
            resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        return binary.astype(np.float32) / 255.0

    def _compute_gei(self, tracklet: list[np.ndarray]) -> np.ndarray:
        """Compute Gait Energy Image from a tracklet of crops."""
        silhouettes = [self._get_silhouette(crop) for crop in tracklet]
        gei = np.mean(silhouettes, axis=0)
        return gei

    def _reduce_dim(self, gei: np.ndarray) -> np.ndarray:
        """Reduce GEI to a compact embedding vector.

        Uses simple flattening + normalization if PCA is not fitted,
        otherwise applies PCA projection.
        """
        flat = gei.flatten()

        if self._pca_matrix is not None:
            centered = flat - self._pca_mean
            embedding = centered @ self._pca_matrix.T
        else:
            # Simple dimensionality reduction: resize GEI to small size
            small = cv2.resize(
                gei,
                (8, 16),  # 8*16=128 dimensions
                interpolation=cv2.INTER_AREA,
            )
            embedding = small.flatten()

        # L2 normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding.astype(np.float32)

    def extract(self, tracklet: list[np.ndarray]) -> Optional[np.ndarray]:
        """Extract gait embedding from a tracklet of person crops.

        Args:
            tracklet: List of BGR person crop images over consecutive frames.

        Returns:
            128-d normalized gait embedding, or None if tracklet is too short.
        """
        if not self.enabled:
            return None

        if len(tracklet) < self.min_tracklet_length:
            return None

        gei = self._compute_gei(tracklet)
        embedding = self._reduce_dim(gei)

        return embedding
