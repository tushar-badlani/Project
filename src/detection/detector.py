from typing import TYPE_CHECKING, Optional

import numpy as np
import torch
from ultralytics import YOLO

if TYPE_CHECKING:
    from src.utils.profiler import PipelineProfiler


class PersonDetector:
    """YOLOv8 / RT-DETR based person detector."""

    def __init__(self, config: dict, profiler: Optional["PipelineProfiler"] = None):
        det_cfg = config["detection"]
        self.model = YOLO(det_cfg["model"])
        self.conf_threshold = det_cfg["conf_threshold"]
        self.iou_threshold = det_cfg["iou_threshold"]
        self.person_class_id = det_cfg["person_class_id"]
        self.imgsz = det_cfg["imgsz"]
        self.device = config["device"]
        self._profiler = profiler

        if profiler is not None:
            self._profile_model()

    def _profile_model(self) -> None:
        """Measure FLOPs and param count for the YOLOv8 backbone using thop."""
        try:
            from thop import profile as thop_profile  # type: ignore

            nn_model = self.model.model
            imgsz = self.imgsz if isinstance(self.imgsz, int) else self.imgsz[0]
            dummy = torch.zeros(1, 3, imgsz, imgsz)
            device = next(nn_model.parameters()).device
            dummy = dummy.to(device)

            nn_model.eval()
            flops, params = thop_profile(nn_model, inputs=(dummy,), verbose=False)
            self._profiler.record_model_profile("YOLOv8", int(flops * 2), int(params))
        except Exception as exc:
            self._profiler.record_model_profile("YOLOv8", None, None)
            print(f"[Profiler] YOLOv8 FLOPs profiling failed: {exc}")

    def detect(self, frame: np.ndarray) -> np.ndarray:
        """Detect persons in a frame.

        Returns:
            np.ndarray of shape (N, 5) with columns [x1, y1, x2, y2, confidence].
            Empty array of shape (0, 5) if no detections.
        """
        if self._profiler is not None:
            ctx = self._profiler.time_stage("yolov8")
        else:
            from contextlib import nullcontext
            ctx = nullcontext()

        with ctx:
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
