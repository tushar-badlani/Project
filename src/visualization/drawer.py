import cv2
import numpy as np


class Visualizer:
    """Draw bounding boxes, person IDs, and confidence scores on frames."""

    def __init__(self, config: dict):
        vis_cfg = config["visualization"]
        self.draw_boxes = vis_cfg["draw_boxes"]
        self.draw_ids = vis_cfg["draw_ids"]
        self.thickness = vis_cfg["box_thickness"]
        self.matched_color = tuple(vis_cfg["matched_color"])
        self.unknown_color = tuple(vis_cfg["unknown_color"])
        self.font_scale = vis_cfg["font_scale"]
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def draw(
        self,
        frame: np.ndarray,
        tracked_persons: list,
        matches: dict[int, tuple[str, float]],
    ) -> np.ndarray:
        """Annotate frame with detection results.

        Args:
            frame: BGR frame to annotate.
            tracked_persons: List of TrackedPerson objects.
            matches: Dict mapping track_id -> (person_id, score).

        Returns:
            Annotated frame copy.
        """
        annotated = frame.copy()

        if not self.draw_boxes:
            return annotated

        for person in tracked_persons:
            track_id = person.track_id
            bbox = person.bbox
            x1, y1, x2, y2 = bbox

            person_id, score = matches.get(track_id, ("unknown", 0.0))
            is_matched = person_id != "unknown"
            color = self.matched_color if is_matched else self.unknown_color

            # Draw bounding box
            cv2.rectangle(
                annotated, (x1, y1), (x2, y2), color, self.thickness
            )

            if self.draw_ids:
                # Build label
                if is_matched:
                    label = f"{person_id} ({score:.2f})"
                else:
                    label = f"ID:{track_id}"

                # Background rectangle for text
                (text_w, text_h), baseline = cv2.getTextSize(
                    label, self.font, self.font_scale, 1
                )
                cv2.rectangle(
                    annotated,
                    (x1, y1 - text_h - baseline - 4),
                    (x1 + text_w, y1),
                    color,
                    -1,
                )

                # Text
                cv2.putText(
                    annotated,
                    label,
                    (x1, y1 - baseline - 2),
                    self.font,
                    self.font_scale,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        return annotated
