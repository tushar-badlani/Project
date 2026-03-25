from typing import Optional

import numpy as np
from scipy.spatial.distance import cosine as cosine_distance

from .gallery import Gallery


class Matcher:
    """Weighted multi-modal embedding matcher.

    Fuses face, body, and gait similarity scores with dynamic weight
    renormalization when some modalities are unavailable.
    """

    def __init__(self, config: dict):
        match_cfg = config["matching"]
        self.weights = match_cfg["weights"]  # {"face": 0.5, "body": 0.35, "gait": 0.15}
        self.threshold = match_cfg["similarity_threshold"]

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        return 1.0 - cosine_distance(a, b)

    def match(
        self,
        query_embeddings: dict[str, Optional[np.ndarray]],
        gallery: Gallery,
    ) -> tuple[str, float]:
        """Match query embeddings against all gallery identities.

        Args:
            query_embeddings: {"face": ndarray|None, "body": ndarray|None, "gait": ndarray|None}
            gallery: Gallery of known identities.

        Returns:
            (person_id or "unknown", best_score)
        """
        identities = gallery.get_all()
        if not identities:
            return ("unknown", 0.0)

        # Check if query has any embeddings at all
        has_any = any(v is not None for v in query_embeddings.values())
        if not has_any:
            return ("unknown", 0.0)

        best_id = "unknown"
        best_score = 0.0

        for identity in identities:
            score = self._compute_fused_score(query_embeddings, identity)
            if score > best_score:
                best_score = score
                best_id = identity.person_id

        if best_score < self.threshold:
            return ("unknown", best_score)

        return (best_id, best_score)

    def _compute_fused_score(
        self,
        query: dict[str, Optional[np.ndarray]],
        identity,
    ) -> float:
        """Compute weighted fusion score between query and one identity.

        Weights are renormalized based on which modalities are available
        in BOTH query and gallery.
        """
        modality_scores = {}
        available_weights = {}

        # Face
        if query.get("face") is not None and identity.face_embedding is not None:
            sim = self._cosine_similarity(query["face"], identity.face_embedding)
            modality_scores["face"] = sim
            available_weights["face"] = self.weights["face"]

        # Body
        if query.get("body") is not None and identity.body_embedding is not None:
            sim = self._cosine_similarity(query["body"], identity.body_embedding)
            modality_scores["body"] = sim
            available_weights["body"] = self.weights["body"]

        # Gait
        if query.get("gait") is not None and identity.gait_embedding is not None:
            sim = self._cosine_similarity(query["gait"], identity.gait_embedding)
            modality_scores["gait"] = sim
            available_weights["gait"] = self.weights["gait"]

        if not modality_scores:
            return 0.0

        # Renormalize weights so they sum to 1.0
        total_weight = sum(available_weights.values())
        fused_score = sum(
            (available_weights[mod] / total_weight) * modality_scores[mod]
            for mod in modality_scores
        )

        return fused_score
