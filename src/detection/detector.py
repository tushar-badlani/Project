import numpy as np
from ultralytics import YOLO


class PersonDetector:
    """YOLOv8 / RT-DETR based person detector."""

    def __init__(self, config: dict):
        det_cfg = config["detection"]
        self.model = YOLO(det_cfg["model"])
        self.conf_threshold = det_cfg["conf_threshold"]
        self.iou_threshold = det_cfg["iou_threshold"]
        self.person_class_id = det_cfg["person_class_id"]
        self.imgsz = det_cfg["imgsz"]
        self.device = config["device"]

    def detect(self, frame: np.ndarray) -> np.ndarray:
        """Detect persons in a frame.

        Returns:
            np.ndarray of shape (N, 5) with columns [x1, y1, x2, y2, confidence].
            Empty array of shape (0, 5) if no detections.
        """
        results = self.model(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.imgsz,
            device=self.device,
            classes=[self.person_class_id],
            verbose=False,
        )

        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return np.empty((0, 5), dtype=np.float32)

        boxes = result.boxes.xyxy.cpu().numpy()      # (N, 4)
        confs = result.boxes.conf.cpu().numpy()[:, None]  # (N, 1)
        return np.hstack([boxes, confs]).astype(np.float32)
