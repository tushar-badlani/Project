#!/usr/bin/env python
"""Two-Pass Person Re-Identification Pipeline.

Pass 1: Detect + track all frames, collect N best crops per track.
Resolve: Match each track against gallery using best crops, pick highest score.
Pass 2: Re-read video, draw boxes with resolved identities.

Usage:
    python pipeline.py                              # use config/default.yaml
    python pipeline.py --config config/custom.yaml  # custom config
    python pipeline.py --source video.mp4           # override video source
    python pipeline.py --query query_images/        # override query directory
"""

import argparse
import heapq
import os
import time
from collections import defaultdict

import cv2
import numpy as np
import yaml

from src.detection import PersonDetector
from src.tracking import Tracker
from src.tracking.bytetrack import TrackedPerson
from src.features import FaceEmbedder, BodyEmbedder, GaitEmbedder
from src.matching import Gallery, Matcher
from src.visualization import Visualizer
from src.utils import MatchLogger


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(description="Person Re-ID Pipeline (Two-Pass)")
    parser.add_argument(
        "--config", default="config/default.yaml", help="Path to config YAML"
    )
    parser.add_argument("--source", default=None, help="Video source override")
    parser.add_argument("--query", default=None, help="Query images dir override")
    parser.add_argument("--device", default=None, help="Device override (cuda/cpu)")
    parser.add_argument(
        "--no-display", action="store_true", help="Disable video display"
    )
    parser.add_argument(
        "--max-crops", type=int, default=10,
        help="Max crops to keep per track for matching (default: 10)"
    )
    return parser.parse_args()


def run_pipeline(config: dict, max_crops: int = 10):
    """Two-pass pipeline: collect then match then draw."""

    # ── Initialize modules ───────────────────────────────────────────
    print("Initializing modules...")
    detector = PersonDetector(config)
    tracker = Tracker(config)
    face_embedder = FaceEmbedder(config)
    body_embedder = BodyEmbedder(config)
    gait_embedder = GaitEmbedder(config)
    matcher = Matcher(config)
    gallery = Gallery(config, face_embedder, body_embedder)
    visualizer = Visualizer(config)
    logger = MatchLogger(config)

    # ── Load gallery ─────────────────────────────────────────────────
    query_dir = config["input"]["query_dir"]
    gallery.load_from_directory(query_dir)

    if not gallery.get_all():
        logger.info("WARNING: Gallery is empty. No identities to match against.")

    # ── Open video ───────────────────────────────────────────────────
    source = config["input"]["source"]
    if source.isdigit():
        source = int(source)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    process_every_n = config["input"].get("process_every_n", 1)

    logger.info(f"Video: {width}x{height} @ {fps:.1f} FPS, {total_frames} frames")
    logger.info(f"Two-pass mode: collecting top {max_crops} crops per track")

    # ══════════════════════════════════════════════════════════════════
    # PASS 1: Detect + Track, collect crops and per-frame bbox data
    # ══════════════════════════════════════════════════════════════════
    logger.info("=== PASS 1: Collecting detections and crops ===")

    # Per-frame storage: frame_num -> [(track_id, bbox)]
    frame_data: dict[int, list[tuple[int, np.ndarray]]] = {}
    # Per-track top-N crops: track_id -> min-heap of (confidence, index, crop)
    # Using a min-heap so we can efficiently evict the lowest-confidence crop
    track_crops: dict[int, list] = defaultdict(list)
    crop_counter = 0  # tiebreaker for heap

    frame_num = 0
    t_start = time.time()
    last_frame_data: list[tuple[int, np.ndarray]] = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_num += 1

            # Skip frames — reuse last detections for annotation positions
            if frame_num % process_every_n != 0:
                frame_data[frame_num] = last_frame_data
                continue

            # Detect + track
            detections = detector.detect(frame)
            tracked_persons = tracker.update(detections, frame)

            current_frame_data = []
            for person in tracked_persons:
                tid = person.track_id
                current_frame_data.append((tid, person.bbox.copy()))

                # Keep top-N crops by confidence (min-heap)
                heap = track_crops[tid]
                if len(heap) < max_crops:
                    heapq.heappush(heap, (person.confidence, crop_counter, person.crop))
                elif person.confidence > heap[0][0]:
                    heapq.heapreplace(heap, (person.confidence, crop_counter, person.crop))
                crop_counter += 1

            frame_data[frame_num] = current_frame_data
            last_frame_data = current_frame_data

            if frame_num % 100 == 0:
                elapsed = time.time() - t_start
                proc_fps = frame_num / elapsed
                progress = f" ({frame_num}/{total_frames})" if total_frames > 0 else ""
                logger.info(
                    f"[Pass 1] Frame {frame_num}{progress} | "
                    f"{proc_fps:.1f} FPS | "
                    f"{len(tracked_persons)} persons | "
                    f"{len(track_crops)} tracks collected"
                )

    except KeyboardInterrupt:
        logger.info("Pass 1 interrupted by user")

    pass1_elapsed = time.time() - t_start
    logger.info(
        f"[Pass 1] Done: {frame_num} frames in {pass1_elapsed:.1f}s, "
        f"{len(track_crops)} unique tracks"
    )

    # ══════════════════════════════════════════════════════════════════
    # RESOLVE: Match each track against gallery using best crops
    # ══════════════════════════════════════════════════════════════════
    logger.info("=== RESOLVING: Matching tracks against gallery ===")
    t_resolve = time.time()

    # track_id -> (person_id, best_score)
    resolved_ids: dict[int, tuple[str, float]] = {}

    for tid, heap in track_crops.items():
        best_id = "unknown"
        best_score = 0.0

        # Try each crop, keep the highest-scoring match
        for _conf, _idx, crop in heap:
            query_embs: dict[str, np.ndarray | None] = {}
            query_embs["face"] = face_embedder.extract(crop)
            query_embs["body"] = body_embedder.extract(crop)

            # Gait from full tracklet
            tracklet = tracker.get_tracklet(tid)
            query_embs["gait"] = gait_embedder.extract(tracklet)

            person_id, score = matcher.match(query_embs, gallery)
            if score > best_score:
                best_score = score
                best_id = person_id

        resolved_ids[tid] = (best_id, best_score)
        logger.info(f"  Track {tid}: {best_id} (score={best_score:.4f})")

    resolve_elapsed = time.time() - t_resolve
    logger.info(f"[Resolve] Done: {len(resolved_ids)} tracks in {resolve_elapsed:.1f}s")

    # Free crop memory
    del track_crops

    # ══════════════════════════════════════════════════════════════════
    # PASS 2: Re-read video, draw boxes with resolved identities
    # ══════════════════════════════════════════════════════════════════
    logger.info("=== PASS 2: Drawing annotations ===")
    t_pass2 = time.time()

    # Setup video writer
    writer = None
    if config["input"]["save_video"]:
        output_dir = config["input"]["output_dir"]
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, "output.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        logger.info(f"Saving output to: {out_path}")

    display = config["input"]["display"]

    cap.release()
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot reopen video source: {source}")

    frame_num = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_num += 1

            # Build matches for this frame from resolved IDs
            persons_in_frame = frame_data.get(frame_num, [])
            matches: dict[int, tuple[str, float]] = {}
            tracked_persons = []

            for tid, bbox in persons_in_frame:
                matches[tid] = resolved_ids.get(tid, ("unknown", 0.0))
                tracked_persons.append(TrackedPerson(
                    track_id=tid,
                    bbox=bbox,
                    crop=np.empty((0, 0, 3), dtype=np.uint8),  # not needed for drawing
                    confidence=0.0,
                ))

                # Log on processed frames only
                if frame_num % process_every_n == 0:
                    person_id, score = matches[tid]
                    logger.log(frame_num, tid, person_id, score, bbox)

            # Draw
            annotated = visualizer.draw(frame, tracked_persons, matches)

            if display:
                cv2.imshow("Person Re-ID", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break

            if writer is not None:
                writer.write(annotated)

            if frame_num % 100 == 0:
                elapsed = time.time() - t_pass2
                proc_fps = frame_num / elapsed
                progress = f" ({frame_num}/{total_frames})" if total_frames > 0 else ""
                logger.info(f"[Pass 2] Frame {frame_num}{progress} | {proc_fps:.1f} FPS")

    except KeyboardInterrupt:
        logger.info("Pass 2 interrupted by user")

    finally:
        total_elapsed = time.time() - t_start
        logger.info(
            f"Total: {frame_num} frames in {total_elapsed:.1f}s "
            f"({frame_num / max(total_elapsed, 1e-6):.1f} FPS avg)"
        )
        cap.release()
        if writer is not None:
            writer.release()
        if display:
            cv2.destroyAllWindows()
        logger.close()


def main():
    args = parse_args()
    config = load_config(args.config)

    # Apply CLI overrides
    if args.source:
        config["input"]["source"] = args.source
    if args.query:
        config["input"]["query_dir"] = args.query
    if args.device:
        config["device"] = args.device
    if args.no_display:
        config["input"]["display"] = False

    run_pipeline(config, max_crops=args.max_crops)


if __name__ == "__main__":
    main()
