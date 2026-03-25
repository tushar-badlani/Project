"""FastAPI server for Person Re-Identification Pipeline.

Upload a video and a query photo, get back an annotated output video
with the person identified and highlighted.
"""

import heapq
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from collections import defaultdict

import cv2
import numpy as np
import yaml
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from src.detection import PersonDetector
from src.tracking import Tracker
from src.tracking.bytetrack import TrackedPerson
from src.features import FaceEmbedder, BodyEmbedder, GaitEmbedder
from src.matching import Gallery, Matcher
from src.visualization import Visualizer
from src.utils import MatchLogger

app = FastAPI(title="Person Re-ID API", version="1.0.0")
app.mount("/static", StaticFiles(directory="static"), name="static")

CONFIG_PATH = "config/default.yaml"
MAX_CROPS = 10


def load_config() -> dict:
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    # Force server-appropriate settings
    config["input"]["display"] = False
    config["input"]["save_video"] = True
    return config


# Load config and initialize models once at startup
print("Loading config and initializing models...")
config = load_config()
detector = PersonDetector(config)
tracker_template_config = config  # tracker needs to be per-request (stateful)
face_embedder = FaceEmbedder(config)
body_embedder = BodyEmbedder(config)
gait_embedder = GaitEmbedder(config)
matcher = Matcher(config)
visualizer = Visualizer(config)
print("Models loaded.")


def save_upload(upload: UploadFile, dest: str):
    with open(dest, "wb") as f:
        shutil.copyfileobj(upload.file, f)


def run_pipeline(video_path: str, query_dir: str, output_path: str):
    """Run the two-pass re-id pipeline on given video and query image."""

    # Per-request stateful modules
    request_config = load_config()
    request_config["input"]["source"] = video_path
    request_config["input"]["query_dir"] = query_dir
    request_config["input"]["output_dir"] = os.path.dirname(output_path)
    request_config["logging"]["log_file"] = os.path.join(
        os.path.dirname(output_path), "reid_log.csv"
    )

    tracker = Tracker(request_config)
    gallery = Gallery(request_config, face_embedder, body_embedder)
    logger = MatchLogger(request_config)

    # Load gallery from the uploaded query image
    gallery.load_from_directory(query_dir)
    if not gallery.get_all():
        raise RuntimeError("Gallery is empty - could not process query photo.")

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    process_every_n = request_config["input"].get("process_every_n", 1)

    logger.info(f"Video: {width}x{height} @ {fps:.1f} FPS, {total_frames} frames")

    # ── PASS 1: Detect + Track, collect crops ──
    logger.info("=== PASS 1: Collecting detections and crops ===")

    frame_data: dict[int, list[tuple[int, np.ndarray]]] = {}
    track_crops: dict[int, list] = defaultdict(list)
    crop_counter = 0

    frame_num = 0
    t_start = time.time()
    last_frame_data: list[tuple[int, np.ndarray]] = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1

        if frame_num % process_every_n != 0:
            frame_data[frame_num] = last_frame_data
            continue

        detections = detector.detect(frame)
        tracked_persons = tracker.update(detections, frame)

        current_frame_data = []
        for person in tracked_persons:
            tid = person.track_id
            current_frame_data.append((tid, person.bbox.copy()))

            heap = track_crops[tid]
            if len(heap) < MAX_CROPS:
                heapq.heappush(heap, (person.confidence, crop_counter, person.crop))
            elif person.confidence > heap[0][0]:
                heapq.heapreplace(heap, (person.confidence, crop_counter, person.crop))
            crop_counter += 1

        frame_data[frame_num] = current_frame_data
        last_frame_data = current_frame_data

    pass1_elapsed = time.time() - t_start
    logger.info(
        f"[Pass 1] Done: {frame_num} frames in {pass1_elapsed:.1f}s, "
        f"{len(track_crops)} unique tracks"
    )

    # ── RESOLVE: Match each track against gallery ──
    logger.info("=== RESOLVING: Matching tracks against gallery ===")
    t_resolve = time.time()

    resolved_ids: dict[int, tuple[str, float]] = {}

    for tid, heap in track_crops.items():
        best_id = "unknown"
        best_score = 0.0

        for _conf, _idx, crop in heap:
            query_embs: dict[str, np.ndarray | None] = {}
            query_embs["face"] = face_embedder.extract(crop)
            query_embs["body"] = body_embedder.extract(crop)

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

    del track_crops

    # ── PASS 2: Re-read video, draw annotations ──
    logger.info("=== PASS 2: Drawing annotations ===")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    cap.release()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot reopen video: {video_path}")

    frame_num = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1

        persons_in_frame = frame_data.get(frame_num, [])
        matches: dict[int, tuple[str, float]] = {}
        tracked_persons = []

        for tid, bbox in persons_in_frame:
            matches[tid] = resolved_ids.get(tid, ("unknown", 0.0))
            tracked_persons.append(TrackedPerson(
                track_id=tid,
                bbox=bbox,
                crop=np.empty((0, 0, 3), dtype=np.uint8),
                confidence=0.0,
            ))

            if frame_num % process_every_n == 0:
                person_id, score = matches[tid]
                logger.log(frame_num, tid, person_id, score, bbox)

        annotated = visualizer.draw(frame, tracked_persons, matches)
        writer.write(annotated)

    cap.release()
    writer.release()
    logger.close()

    total_elapsed = time.time() - t_start
    logger.info(
        f"Total: {frame_num} frames in {total_elapsed:.1f}s "
        f"({frame_num / max(total_elapsed, 1e-6):.1f} FPS avg)"
    )


@app.get("/", response_class=HTMLResponse)
async def index():
    with open("static/index.html", "r") as f:
        return f.read()


@app.post("/process", summary="Upload video + query photo, get annotated video back")
async def process_video(
    video: UploadFile = File(..., description="Input video file"),
    photo: UploadFile = File(..., description="Query photo of the person to identify"),
):
    """
    Upload a video and a reference photo of a person.
    Returns an annotated video with the person identified and highlighted.
    """
    # Validate file types
    video_ext = os.path.splitext(video.filename or "video.mp4")[1].lower()
    photo_ext = os.path.splitext(photo.filename or "photo.jpg")[1].lower()

    if video_ext not in (".mp4", ".avi", ".mov", ".mkv", ".webm"):
        raise HTTPException(400, f"Unsupported video format: {video_ext}")
    if photo_ext not in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
        raise HTTPException(400, f"Unsupported image format: {photo_ext}")

    # Create temp workspace
    job_id = uuid.uuid4().hex[:12]
    work_dir = tempfile.mkdtemp(prefix=f"reid_{job_id}_")
    video_path = os.path.join(work_dir, f"input{video_ext}")
    query_dir = os.path.join(work_dir, "query")
    os.makedirs(query_dir)
    photo_path = os.path.join(query_dir, f"query{photo_ext}")
    output_path = os.path.join(work_dir, "output.mp4")

    try:
        # Save uploaded files
        save_upload(video, video_path)
        save_upload(photo, photo_path)

        # Run pipeline (writes mp4v codec, not browser-playable)
        raw_output = os.path.join(work_dir, "output_raw.mp4")
        run_pipeline(video_path, query_dir, raw_output)

        if not os.path.exists(raw_output):
            raise HTTPException(500, "Pipeline failed to produce output video.")

        # Re-encode to H.264 so browsers can play it
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", raw_output,
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "23",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                "-an",
                output_path,
            ],
            check=True,
            capture_output=True,
        )

        return FileResponse(
            output_path,
            media_type="video/mp4",
            filename=f"reid_output_{job_id}.mp4",
            background=None,
        )

    except RuntimeError as e:
        raise HTTPException(500, str(e))

    finally:
        # Clean up input files (output cleaned after response via middleware or manually)
        for f in [video_path, photo_path]:
            if os.path.exists(f):
                os.unlink(f)


@app.on_event("shutdown")
def cleanup_temp():
    """Best-effort cleanup of any leftover temp dirs."""
    import glob
    for d in glob.glob(os.path.join(tempfile.gettempdir(), "reid_*")):
        shutil.rmtree(d, ignore_errors=True)


@app.get("/health")
async def health():
    return {"status": "ok"}
