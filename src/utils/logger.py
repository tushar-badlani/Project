import csv
import os
import logging
from datetime import datetime


class MatchLogger:
    """Logs re-identification match events to CSV and stdout."""

    def __init__(self, config: dict):
        log_cfg = config["logging"]
        self.log_file = log_cfg["log_file"]
        self.level = getattr(logging, log_cfg["level"].upper(), logging.INFO)

        # Setup Python logger
        self.logger = logging.getLogger("person_reid")
        self.logger.setLevel(self.level)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
            )
            self.logger.addHandler(handler)

        # Setup CSV file
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        self._csv_file = open(self.log_file, "w", newline="")
        self._writer = csv.writer(self._csv_file)
        self._writer.writerow([
            "timestamp", "frame_number", "track_id",
            "matched_person_id", "similarity_score",
            "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
        ])

    def log(
        self,
        frame_number: int,
        track_id: int,
        person_id: str,
        score: float,
        bbox,
    ):
        """Log a single match event."""
        timestamp = datetime.now().isoformat()
        x1, y1, x2, y2 = bbox

        self._writer.writerow([
            timestamp, frame_number, track_id,
            person_id, f"{score:.4f}",
            x1, y1, x2, y2,
        ])
        self._csv_file.flush()

        if person_id != "unknown":
            self.logger.info(
                f"Frame {frame_number}: Track {track_id} matched "
                f"'{person_id}' (score={score:.3f})"
            )

    def info(self, msg: str):
        self.logger.info(msg)

    def close(self):
        self._csv_file.close()
