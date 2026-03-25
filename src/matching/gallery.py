import os
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np


@dataclass
class Identity:
    person_id: str
    face_embedding: Optional[np.ndarray] = None
    body_embedding: Optional[np.ndarray] = None
    gait_embedding: Optional[np.ndarray] = None


class Gallery:
    """Gallery of known identities with their reference embeddings.

    Loads query/reference images from a directory structure where each
    subdirectory or image represents one identity.
    """

    def __init__(self, config: dict, face_embedder, body_embedder):
        self.config = config
        self.face_embedder = face_embedder
        self.body_embedder = body_embedder
        self.identities: list[Identity] = []

    def load_from_directory(self, path: str):
        """Load reference identities from a directory.

        Supports two layouts:
          1. Flat: each image in the directory is one identity
             (filename minus extension = person_id).
          2. Nested: each subdirectory is one identity containing 1+ images
             (directory name = person_id, embeddings averaged across images).
        """
        if not os.path.isdir(path):
            print(f"[Gallery] Query directory not found: {path}")
            return

        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        entries = sorted(os.listdir(path))

        for entry in entries:
            full_path = os.path.join(path, entry)

            if os.path.isdir(full_path):
                # Nested layout: directory of images for one person
                self._load_person_directory(entry, full_path, image_extensions)
            elif os.path.isfile(full_path):
                ext = os.path.splitext(entry)[1].lower()
                if ext in image_extensions:
                    person_id = os.path.splitext(entry)[0]
                    self._load_single_image(person_id, full_path)

        print(f"[Gallery] Loaded {len(self.identities)} identities")

    def _load_single_image(self, person_id: str, image_path: str):
        """Load a single reference image and compute embeddings."""
        img = cv2.imread(image_path)
        if img is None:
            print(f"[Gallery] Failed to read image: {image_path}")
            return

        face_emb = self.face_embedder.extract(img)
        body_emb = self.body_embedder.extract(img)

        identity = Identity(
            person_id=person_id,
            face_embedding=face_emb,
            body_embedding=body_emb,
        )
        self.identities.append(identity)

    def _load_person_directory(
        self, person_id: str, dir_path: str, extensions: set
    ):
        """Load multiple images for one person and average embeddings."""
        face_embeddings = []
        body_embeddings = []

        for fname in sorted(os.listdir(dir_path)):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in extensions:
                continue

            img = cv2.imread(os.path.join(dir_path, fname))
            if img is None:
                continue

            face_emb = self.face_embedder.extract(img)
            body_emb = self.body_embedder.extract(img)

            if face_emb is not None:
                face_embeddings.append(face_emb)
            body_embeddings.append(body_emb)

        if not body_embeddings:
            return

        # Average and re-normalize embeddings
        avg_face = None
        if face_embeddings:
            avg_face = np.mean(face_embeddings, axis=0)
            norm = np.linalg.norm(avg_face)
            if norm > 0:
                avg_face = avg_face / norm

        avg_body = np.mean(body_embeddings, axis=0)
        norm = np.linalg.norm(avg_body)
        if norm > 0:
            avg_body = avg_body / norm

        identity = Identity(
            person_id=person_id,
            face_embedding=avg_face,
            body_embedding=avg_body,
        )
        self.identities.append(identity)

    def add_identity(self, person_id: str, embeddings: dict):
        """Dynamically add a new identity."""
        identity = Identity(
            person_id=person_id,
            face_embedding=embeddings.get("face"),
            body_embedding=embeddings.get("body"),
            gait_embedding=embeddings.get("gait"),
        )
        self.identities.append(identity)

    def get_all(self) -> list[Identity]:
        return self.identities
