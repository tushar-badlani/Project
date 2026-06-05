from dataclasses import dataclass
from collections import defaultdict
from contextlib import nullcontext
from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np
import supervision as sv

if TYPE_CHECKING:
    from src.utils.profiler import PipelineProfiler


@dataclass
class TrackedPerson:
    track_id: int
    bbox: np.ndarray       # [x1, y1, x2, y2]
    crop: np.ndarray       # BGR image crop
    confidence: float


def _compute_hist(crop: np.ndarray) -> np.ndarray:
    """Compute a compact color histogram descriptor for appearance matching."""
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


class Tracker:
    """ByteTrack wrapper with re-ID based track stitching.

    When a track is lost and later a new track appears nearby with similar
    appearance, the old ID is reused to reduce fragmentation.
    """

    def __init__(self, config: dict, profiler: Optional["PipelineProfiler"] = None):
        trk_cfg = config["tracking"]
        self._profiler = profiler
        self.tracker = sv.ByteTrack(
            track_activation_threshold=trk_cfg["track_thresh"],
            lost_track_buffer=trk_cfg["track_buffer"],
            minimum_matching_threshold=trk_cfg["match_thresh"],
        )
        self.max_tracklet_length = trk_cfg.get("max_tracklet_length", 120)
        self.reid_stitch = trk_cfg.get("reid_stitch", True)

        self._tracklets: dict[int, list[np.ndarray]] = defaultdict(list)

        # Re-ID stitching state
        self._id_map: dict[int, int] = {}               # bytetrack_id -> stable_id
        self._known_bt_ids: set[int] = set()             # currently active bytetrack IDs
        self._lost_tracks: dict[int, dict] = {}          # stable_id -> {hist, bbox, lost_frame}
        self._next_stable_id = 1
        self._frame_count = 0
        self._lost_track_ttl = trk_cfg.get("track_buffer", 90) * 2  # how long to keep lost tracks for stitching
        self._stitch_hist_thresh = 0.4   # min histogram correlation to stitch
        self._stitch_dist_thresh = 200   # max bbox center distance (pixels) to stitch

        # Per-track appearance histograms (running average)
        self._track_hists: dict[int, np.ndarray] = {}
        # Last known bbox center per stable_id
        self._center_cache: dict[int, tuple[float, float]] = {}

    def _get_stable_id(self, bt_id: int, bbox: np.ndarray, crop: np.ndarray) -> int:
        """Map a ByteTrack ID to a stable ID, stitching with lost tracks if possible."""
        if bt_id in self._id_map:
            return self._id_map[bt_id]

        # New ByteTrack ID — try to match against lost tracks
        if self.reid_stitch and self._lost_tracks:
            hist = _compute_hist(crop)
            cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2

            best_match_id = None
            best_score = -1.0

            for stable_id, info in self._lost_tracks.items():
                # Spatial check — don't stitch tracks too far apart
                lcx, lcy = info["center"]
                dist = np.sqrt((cx - lcx) ** 2 + (cy - lcy) ** 2)
                if dist > self._stitch_dist_thresh:
                    continue

                # Appearance check
                score = cv2.compareHist(
                    hist, info["hist"], cv2.HISTCMP_CORREL
                )
                if score > self._stitch_hist_thresh and score > best_score:
                    best_score = score
                    best_match_id = stable_id

            if best_match_id is not None:
                self._id_map[bt_id] = best_match_id
                del self._lost_tracks[best_match_id]
                return best_match_id

        # No match — assign new stable ID
        stable_id = self._next_stable_id
        self._next_stable_id += 1
        self._id_map[bt_id] = stable_id
        return stable_id

    def update(
        self, detections: np.ndarray, frame: np.ndarray
    ) -> list[TrackedPerson]:
        """Update tracker with new detections and return tracked persons."""
        self._frame_count += 1

        ctx = self._profiler.time_stage("bytetrack") if self._profiler else nullcontext()
        with ctx:
            if len(detections) == 0:
                self.tracker.update_with_detections(sv.Detections.empty())
                self._handle_lost_tracks(set())
                return []

            sv_detections = sv.Detections(
                xyxy=detections[:, :4],
                confidence=detections[:, 4],
            )
            tracked = self.tracker.update_with_detections(sv_detections)

        results = []
        h, w = frame.shape[:2]
        current_bt_ids = set()

        for i in range(len(tracked)):
            bbox = tracked.xyxy[i].astype(int)
            bt_id = int(tracked.tracker_id[i])
            conf = float(tracked.confidence[i])

            # Clamp bbox to frame bounds
            x1 = max(0, bbox[0])
            y1 = max(0, bbox[1])
            x2 = min(w, bbox[2])
            y2 = min(h, bbox[3])

            if x2 <= x1 or y2 <= y1:
                continue

            crop = frame[y1:y2, x1:x2].copy()
            clamped_bbox = np.array([x1, y1, x2, y2])

            # Map to stable ID
            stable_id = self._get_stable_id(bt_id, clamped_bbox, crop)
            current_bt_ids.add(bt_id)

            # Update appearance histogram (running average)
            hist = _compute_hist(crop)
            if stable_id in self._track_hists:
                self._track_hists[stable_id] = (
                    0.8 * self._track_hists[stable_id] + 0.2 * hist
                )
            else:
                self._track_hists[stable_id] = hist

            # Store bbox center for lost-track stitching
            self._center_cache[stable_id] = (
                (x1 + x2) / 2.0, (y1 + y2) / 2.0
            )

            # Store in tracklet history
            tracklet = self._tracklets[stable_id]
            tracklet.append(crop)
            if len(tracklet) > self.max_tracklet_length:
                tracklet.pop(0)

            results.append(TrackedPerson(
                track_id=stable_id,
                bbox=clamped_bbox,
                crop=crop,
                confidence=conf,
            ))

        self._handle_lost_tracks(current_bt_ids)
        self._known_bt_ids = current_bt_ids

        return results

    def _handle_lost_tracks(self, current_bt_ids: set):
        """Move disappeared tracks to lost pool for future stitching."""
        disappeared = self._known_bt_ids - current_bt_ids

        for bt_id in disappeared:
            stable_id = self._id_map.get(bt_id)
            if stable_id is None:
                continue

            hist = self._track_hists.get(stable_id)
            tracklet = self._tracklets.get(stable_id, [])
            if hist is not None and tracklet:
                last_crop = tracklet[-1]
                # Estimate center from last known crop position
                # Use the histogram for appearance matching
                self._lost_tracks[stable_id] = {
                    "hist": hist.copy(),
                    "center": self._center_cache.get(stable_id, (0, 0)),
                    "lost_frame": self._frame_count,
                }

            # Clean up the bytetrack->stable mapping
            del self._id_map[bt_id]

        # Expire old lost tracks
        expired = [
            sid for sid, info in self._lost_tracks.items()
            if (self._frame_count - info["lost_frame"]) > self._lost_track_ttl
        ]
        for sid in expired:
            del self._lost_tracks[sid]

    def get_tracklet(self, track_id: int) -> list[np.ndarray]:
        """Get stored crops for a given track ID."""
        return self._tracklets.get(track_id, [])
