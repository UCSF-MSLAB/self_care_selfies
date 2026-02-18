"""
self_care_selfies.py
====================
Video analysis pipeline for the Self Care Selfies project.

Analyzes videos of patients performing tasks (gait, hand activities, facial
expression) and computes motion metrics that correlate with clinical disease
measures (e.g., MS progression).

Updated from the 2022 original to use the current MediaPipe Tasks API
(mediapipe >= 0.10) and to fix several calculation errors. Core analysis
functions are importable for integration into a production backend pipeline.

CLI Usage
---------
    python self_care_selfies.py --help
    python self_care_selfies.py --video-dir videos --output output.csv
    python self_care_selfies.py --video-dir videos --output output.csv --display
    python self_care_selfies.py --video-dir videos --output output.csv --save-video annotated/
    python self_care_selfies.py --video-dir videos --output output.csv --input-csv schedule.csv

Importable Usage
----------------
    from self_care_selfies import analyze_video_file, VideoAnalysisResult

    results = analyze_video_file("path/to/video.mp4", activity="BrushL")
    for r in results:
        print(r.metrics)

Output CSV Columns
------------------
    activity        - name of video / task (e.g., BrushL, Gait)
    hand            - Left, Right, or Pose
    landmark        - name of landmark (e.g., index_finger_tip, left_foot_index)
    participant     - patient/participant ID
    date            - date the video was taken
    displacement    - straight-line distance from first to last position (normalised)
    total_travel    - cumulative path length across all frames (normalised)
    average_velocity- total_travel / (num_frames - 1): mean per-frame displacement
    peak_velocity   - maximum displacement between any two consecutive frames
    normed_velocity - average_velocity / peak_velocity  [0, 1]
    velocity_peaks  - direction-reversal count in velocity signal / num_frames
    fps             - video frame rate (use to convert velocity to real-world units)
    num_frames      - number of frames analysed

Coordinate System
-----------------
All (x, y) coordinates are MediaPipe-normalised: x ∈ [0, 1] (left → right),
y ∈ [0, 1] (top → bottom). Using normalised coordinates makes metrics
device- and resolution-independent, which is important since videos may come
from different patient devices.

Python & MediaPipe Versions
---------------------------
Requires Python 3.11+ (3.13 recommended).
mediapipe >= 0.10.30 is required — this release switched to universal py3
wheels, adding Python 3.13 support. Earlier releases capped at Python 3.12.

MediaPipe Models
----------------
The Tasks API requires separate model bundle files (.task). These are
downloaded automatically to ~/.cache/self_care_selfies/models/ on first run.
You can override paths with --hand-model and --pose-model CLI flags, or by
passing model_path= to the analysis functions.

Change Log (vs 2022 original)
------------------------------
- Migrated from deprecated mp.solutions.* to mediapipe.tasks.python.vision
- Removed mediapipe.framework and mediapipe.solutions imports (gone in 0.10.30+);
  drawing now uses plain cv2 + HandLandmarksConnections / PoseLandmarksConnections
- Fixed currentSec bug: was (frame * fps) / 1000 → corrected to frame / fps
- Fixed average_velocity denominator: N → N-1 (correct number of intervals)
- Fixed hand landmark pre-allocation: was 33 (pose count) → corrected to 21
- peak_velocity computed once, not twice, in compute_metrics
- Landmarker contexts properly managed (no resource leaks)
- cv2.imshow gated behind --display flag; headless-safe by default
- Replaced manual sys.argv parsing with argparse
- Removed local re-implementation of str.removeprefix() (Python 3.9+ built-in)
- Uses normalised coordinates for resolution-independent metrics
- Added fps and num_frames to output for downstream per-second conversion
- Added structured logging, type hints, and docstrings throughout
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import math
import os
import ssl
import sys
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import HandLandmarksConnections, PoseLandmarksConnections

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
log = logging.getLogger(__name__)


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logging for CLI-style use.

    This function is intentionally not called at import time so that
    applications importing this module remain in control of logging
    configuration. Call it explicitly from a CLI entry point if needed.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DETECTION_CONFIDENCE: float = 0.5
TRACKING_CONFIDENCE: float = 0.5

# MediaPipe hand detection reports handedness mirrored (selfie convention).
# Flipping restores anatomically correct left/right.
FLIP_HAND: dict[str, str] = {"Left": "Right", "Right": "Left"}

# Default model cache directory
_MODEL_CACHE_DIR = Path.home() / ".cache" / "self_care_selfies" / "models"

# MediaPipe model download URLs (float16 / lite variants — good accuracy/speed balance)
_HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
_POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)

# SHA256 checksums for model integrity verification
# TODO: Populate with actual checksums from official MediaPipe releases.
# For now, checksum verification is skipped with a warning logged.
# To enable verification, download models manually and compute their SHA256 hashes.
_MODEL_CHECKSUMS: dict[str, str] = {
    # "hand_landmarker.task": "actual_sha256_hash_here",
    # "pose_landmarker_lite.task": "actual_sha256_hash_here",
}

# Download timeout in seconds
_DOWNLOAD_TIMEOUT = 300  # 5 minutes

# ---------------------------------------------------------------------------
# Landmark dictionaries
# ---------------------------------------------------------------------------
# MediaPipe BlazePose — 33 landmarks
POSE_LANDMARKS: dict[int, str] = {
    0: "nose",
    1: "left_eye_inner",
    2: "left_eye",
    3: "left_eye_outer",
    4: "right_eye_inner",
    5: "right_eye",
    6: "right_eye_outer",
    7: "left_ear",
    8: "right_ear",
    9: "mouth_left",
    10: "mouth_right",
    11: "left_shoulder",
    12: "right_shoulder",
    13: "left_elbow",
    14: "right_elbow",
    15: "left_wrist",
    16: "right_wrist",
    17: "left_pinky",
    18: "right_pinky",
    19: "left_index",
    20: "right_index",
    21: "left_thumb",
    22: "right_thumb",
    23: "left_hip",
    24: "right_hip",
    25: "left_knee",
    26: "right_knee",
    27: "left_ankle",
    28: "right_ankle",
    29: "left_heel",
    30: "right_heel",
    31: "left_foot_index",
    32: "right_foot_index",
}

# MediaPipe HandLandmarker — 21 landmarks
HAND_LANDMARKS: dict[int, str] = {
    0: "wrist",
    1: "thumb_cmc",
    2: "thumb_mcp",
    3: "thumb_ip",
    4: "thumb_tip",
    5: "index_finger_mcp",
    6: "index_finger_pip",
    7: "index_finger_dip",
    8: "index_finger_tip",
    9: "middle_finger_mcp",
    10: "middle_finger_pip",
    11: "middle_finger_dip",
    12: "middle_finger_tip",
    13: "ring_finger_mcp",
    14: "ring_finger_pip",
    15: "ring_finger_dip",
    16: "ring_finger_tip",
    17: "pinky_mcp",
    18: "pinky_pip",
    19: "pinky_dip",
    20: "pinky_tip",
}

# ---------------------------------------------------------------------------
# Activity configuration
# ---------------------------------------------------------------------------
# Features are landmark indices selected for each activity type.
# Chosen landmarks capture the primary motion of interest.
@dataclass
class ActivityConfig:
    modality: str          # "hand" or "pose"
    features: list[int]    # landmark indices to extract metrics for
    landmark_dict: dict[int, str]


def get_activity_config(activity: str) -> ActivityConfig:
    """
    Return the analysis configuration for a given activity name.

    Activity names are matched case-insensitively by prefix:
      Gait   → PoseLandmarker, foot indices (31, 32)
      Talk   → PoseLandmarker, eyes + mouth (1, 5, 9, 10)
      Button, Eat, Brush, (or any other) → HandLandmarker, wrist + index tip (0, 8)
    """
    name = activity.lower()
    if name.startswith("gait"):
        return ActivityConfig(
            modality="pose",
            features=[31, 32],           # left_foot_index, right_foot_index
            landmark_dict=POSE_LANDMARKS,
        )
    if name.startswith("talk"):
        return ActivityConfig(
            modality="pose",
            features=[1, 5, 9, 10],      # left_eye, right_eye, mouth_left, mouth_right
            landmark_dict=POSE_LANDMARKS,
        )
    # Button, Eat, Brush (with optional L/R suffix)
    return ActivityConfig(
        modality="hand",
        features=[0, 8],                 # wrist, index_finger_tip
        landmark_dict=HAND_LANDMARKS,
    )


# ---------------------------------------------------------------------------
# Core metric functions  (pure — no I/O, no MediaPipe dependency)
# ---------------------------------------------------------------------------
Point2D = tuple[float, float]   # (x, y) in normalised landmark coordinates


def _distance(p1: Point2D, p2: Point2D) -> float:
    """Euclidean distance between two 2-D points."""
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return math.sqrt(dx * dx + dy * dy)


def compute_displacement(track: list[Point2D]) -> float:
    """Straight-line distance between the first and last positions."""
    return _distance(track[0], track[-1])


def compute_total_travel(track: list[Point2D]) -> float:
    """Cumulative path length: sum of distances between consecutive frame positions."""
    return sum(_distance(track[i], track[i + 1]) for i in range(len(track) - 1))


def compute_average_velocity(track: list[Point2D]) -> float:
    """
    Mean per-interval displacement: total_travel / (N - 1) intervals.

    FIX vs original: denominator is N-1 (number of inter-frame intervals),
    not N (number of frames).
    
    Note: This assumes consecutive detections. If landmark detection is intermittent
    (gaps where the landmark is not detected), velocity calculations will treat 
    detected positions as consecutive even if they span multiple frames, which may
    overestimate velocity. Consider filtering tracks with too many gaps if this
    affects your analysis.
    """
    n_intervals = len(track) - 1
    if n_intervals <= 0:
        return 0.0
    return compute_total_travel(track) / n_intervals


def compute_peak_velocity(track: list[Point2D]) -> float:
    """Maximum displacement between any two consecutive frames."""
    if len(track) < 2:
        return 0.0
    return max(_distance(track[i], track[i + 1]) for i in range(len(track) - 1))


def _inter_frame_velocities(track: list[Point2D]) -> list[float]:
    """Return the list of per-interval displacement values."""
    return [_distance(track[i], track[i + 1]) for i in range(len(track) - 1)]


def _count_direction_reversals(signal: list[float]) -> int:
    """
    Count the number of times the signal changes direction (peak or trough).

    A reversal is detected when consecutive differences change sign.
    """
    if len(signal) < 2:
        return 0
    reversals = 0
    # Compare each pair of consecutive differences
    diffs = [signal[i + 1] - signal[i] for i in range(len(signal) - 1)]
    for i in range(len(diffs) - 1):
        if diffs[i] == 0 or diffs[i + 1] == 0:
            continue
        if (diffs[i] > 0) != (diffs[i + 1] > 0):
            reversals += 1
    return reversals


def compute_velocity_peaks(track: list[Point2D]) -> float:
    """
    Direction-reversal rate in the velocity signal.

    Returns: count_of_reversals / len(track)
    
    Note: This normalizes by the number of detected landmark positions (len(track)),
    not by the total number of video frames. If some frames don't detect the landmark,
    len(track) may be less than the total frame count. This metric represents reversal
    density per detected position, not per video frame.
    """
    velocities = _inter_frame_velocities(track)
    reversals = _count_direction_reversals(velocities)
    return reversals / len(track) if len(track) > 0 else 0.0


def compute_metrics(track: list[Point2D]) -> dict[str, float] | None:
    """
    Compute all motion metrics for a single landmark's trajectory.

    Returns None if the track is too short or shows no motion.
    """
    if len(track) < 2:
        return None

    peak_vel = compute_peak_velocity(track)
    if peak_vel == 0.0:
        # Landmark never moved — uninformative, skip.
        return None

    avg_vel = compute_average_velocity(track)

    return {
        "displacement": compute_displacement(track),
        "total_travel": compute_total_travel(track),
        "average_velocity": avg_vel,
        "peak_velocity": peak_vel,
        "normed_velocity": avg_vel / peak_vel,      # ∈ [0, 1]
        "velocity_peaks": compute_velocity_peaks(track),
    }


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class LandmarkTrack:
    """Time-ordered (x, y) positions for a single landmark across frames."""
    hand: str               # "Left", "Right", or "Pose"
    landmark_idx: int
    landmark_name: str
    frames: list[Point2D] = field(default_factory=list)


@dataclass
class VideoAnalysisResult:
    """All per-landmark results extracted from a single video."""
    participant: str
    activity_date: str
    activity: str
    fps: float
    num_frames: int
    tracks: list[LandmarkTrack]
    metrics: list[dict]     # one dict per track that had computable metrics


# ---------------------------------------------------------------------------
# Model management
# ---------------------------------------------------------------------------
def _compute_sha256(file_path: Path) -> str:
    """Compute SHA256 checksum of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def _ensure_model(url: str, filename: str, model_dir: Path | None = None) -> str:
    """
    Return path to a model file, downloading it if necessary.

    Downloads to model_dir (default: ~/.cache/self_care_selfies/models/).
    Uses SSL verification and timeout for secure downloads.
    
    Note: Checksum verification is skipped if no checksum is defined for the model.
    To enable checksum verification, update _MODEL_CHECKSUMS with the correct hash.
    """
    cache_dir = model_dir or _MODEL_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_path = cache_dir / filename
    
    if not model_path.exists():
        log.info("Downloading model %s ...", filename)
        log.info("  from: %s", url)
        log.info("  to:   %s", model_path)
        
        # Create SSL context with certificate verification enabled
        ssl_context = ssl.create_default_context()
        
        # Download with SSL verification and timeout
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "self_care_selfies/1.0"})
            with urllib.request.urlopen(req, timeout=_DOWNLOAD_TIMEOUT, context=ssl_context) as response:
                with open(model_path, "wb") as out_file:
                    out_file.write(response.read())
        except urllib.error.URLError as e:
            log.error("Failed to download model: %s", e)
            raise
        except Exception as e:
            log.error("Unexpected error during model download: %s", e)
            raise
            
        log.info("Download complete.")
        
        # Verify checksum if available
        expected_checksum = _MODEL_CHECKSUMS.get(filename)
        if expected_checksum:
            log.info("Verifying file integrity...")
            actual_checksum = _compute_sha256(model_path)
            if actual_checksum != expected_checksum:
                model_path.unlink()  # Remove corrupted file
                raise ValueError(
                    f"Checksum mismatch for {filename}. "
                    f"Expected: {expected_checksum}, Got: {actual_checksum}. "
                    "The downloaded file may be corrupted or compromised."
                )
            log.info("Checksum verified successfully.")
        else:
            log.warning(
                "No checksum defined for %s; skipping integrity verification. "
                "Consider adding a checksum to _MODEL_CHECKSUMS for security.",
                filename
            )
    
    return str(model_path)


def ensure_hand_model(model_path: str | None = None) -> str:
    """Return the path to the HandLandmarker .task file, downloading if needed."""
    if model_path:
        return model_path
    return _ensure_model(_HAND_MODEL_URL, "hand_landmarker.task")


def ensure_pose_model(model_path: str | None = None) -> str:
    """Return the path to the PoseLandmarker .task file, downloading if needed."""
    if model_path:
        return model_path
    return _ensure_model(_POSE_MODEL_URL, "pose_landmarker_lite.task")


# ---------------------------------------------------------------------------
# MediaPipe frame processing (Tasks API)
# ---------------------------------------------------------------------------
def _draw_landmarks_cv2(
    image,
    landmarks: list,
    connections: list,
    point_color: tuple = (0, 255, 0),
    line_color: tuple = (255, 255, 255),
    point_radius: int = 4,
    line_thickness: int = 2,
) -> None:
    """
    Draw landmark points and skeleton connections on a BGR image using cv2.

    Works with any mediapipe Tasks API landmark list whose items have .x/.y
    attributes in normalised [0, 1] coordinates.
    Uses cv2 directly — no dependency on mediapipe.framework or mp.solutions.
    """
    h, w = image.shape[:2]
    # Convert normalised coords to pixel coords once
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    # Draw connections
    for conn in connections:
        pt1 = pts[conn.start]
        pt2 = pts[conn.end]
        cv2.line(image, pt1, pt2, line_color, line_thickness, cv2.LINE_AA)
    # Draw landmark points on top
    for pt in pts:
        cv2.circle(image, pt, point_radius, point_color, -1, cv2.LINE_AA)


def _draw_hand_landmarks(image, result: mp_vision.HandLandmarkerResult) -> None:
    """Overlay hand skeleton on a BGR image in-place (display only)."""
    connections = HandLandmarksConnections.HAND_CONNECTIONS
    for hand_lm_list in result.hand_landmarks:
        _draw_landmarks_cv2(
            image, hand_lm_list, connections,
            point_color=(0, 255, 0), line_color=(255, 255, 255),
        )


def _draw_pose_landmarks(image, result: mp_vision.PoseLandmarkerResult) -> None:
    """Overlay pose skeleton on a BGR image in-place (display only)."""
    connections = PoseLandmarksConnections.POSE_LANDMARKS
    for pose_lm_list in result.pose_landmarks:
        _draw_landmarks_cv2(
            image, pose_lm_list, connections,
            point_color=(0, 0, 255), line_color=(0, 255, 255),
        )


def _frames_from_video(
    video_path: str,
    start_sec: float = 0.0,
    end_sec: float = float("inf"),
) -> Iterator[tuple[int, float, float, object]]:
    """
    Yield (frame_index, timestamp_ms, fps, bgr_frame) for each video frame
    within [start_sec, end_sec).

    FIX vs original: timestamp in seconds = frame_index / fps (not frame * fps / 1000).
    """
    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            log.warning(
                "FPS metadata missing or invalid for video '%s'; "
                "defaulting to 30.0 FPS. Timing and velocity estimates may be inaccurate.",
                video_path,
            )
            fps = 30.0
        frame_idx = 0
        
        while True:
            success, frame = cap.read()
            if not success:
                break
            # Correct conversion — frame_index / fps gives seconds
            current_sec = frame_idx / fps
            if current_sec < start_sec:
                frame_idx += 1
                continue
            if current_sec >= end_sec:
                break
            timestamp_ms = int(current_sec * 1000)
            yield frame_idx, timestamp_ms, fps, frame
            frame_idx += 1
    finally:
        cap.release()


# ---------------------------------------------------------------------------
# High-level analysis functions (importable for production pipeline)
# ---------------------------------------------------------------------------
def analyze_video_file(
    video_path: str,
    activity: str,
    participant: str = "unknown",
    activity_date: str = "unknown",
    start_sec: float = 0.0,
    end_sec: float = float("inf"),
    hand_model_path: str | None = None,
    pose_model_path: str | None = None,
    display: bool = False,
    save_video_dir: str | None = None,
) -> VideoAnalysisResult:
    """
    Analyse a single video file and return motion metrics.

    This is the primary importable entry point for production pipelines.

    Parameters
    ----------
    video_path       : Path to the video file (.mp4, .mov, etc.)
    activity         : Activity name determining which landmarker to use.
                       Prefix-matched: "Gait*", "Talk*", or hand activities
                       (Button, Eat, Brush with optional L/R suffix).
    participant      : Participant/patient identifier for output tagging.
    activity_date    : Date string for output tagging.
    start_sec        : Skip frames before this time (seconds). Default 0.
    end_sec          : Stop after this time (seconds). Default: process all.
    hand_model_path  : Path to HandLandmarker .task file. Auto-downloaded if None.
    pose_model_path  : Path to PoseLandmarker .task file. Auto-downloaded if None.
    display          : If True, show annotated video in a cv2 window (local use only).
    save_video_dir   : If set, save each annotated video to
                       <save_video_dir>/<participant>/<date>/<activity>_annotated.mp4.
                       Landmarks are drawn even when display=False.

    Returns
    -------
    VideoAnalysisResult with per-landmark tracks and computed metrics.
    """
    config = get_activity_config(activity)

    # Pre-allocate per-landmark tracking lists, keyed by (hand_label, landmark_idx).
    # "hand_label" is "Left", "Right" for hand modality; "Pose" for pose.
    # Note: This pre-allocates tracks for all landmarks (21 for hands, 33 for pose),
    # even though metrics are only computed for the subset in config.features.
    # For large-scale processing, consider optimizing to only allocate needed landmarks.
    n_landmarks = (
        len(HAND_LANDMARKS) if config.modality == "hand" else len(POSE_LANDMARKS)
    )
    tracks: dict[tuple[str, int], list[Point2D]] = {}
    if config.modality == "hand":
        for side in ("Left", "Right"):
            for idx in range(n_landmarks):
                tracks[(side, idx)] = []
    else:
        for idx in range(n_landmarks):
            tracks[("Pose", idx)] = []

    fps_value = 30.0
    frame_count = 0
    video_writer: cv2.VideoWriter | None = None
    annotate = display or (save_video_dir is not None)

    if config.modality == "hand":
        model_path = ensure_hand_model(hand_model_path)
        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=DETECTION_CONFIDENCE,
            min_hand_presence_confidence=DETECTION_CONFIDENCE,
            min_tracking_confidence=TRACKING_CONFIDENCE,
        )
        landmarker_cls = mp_vision.HandLandmarker

    else:  # pose
        model_path = ensure_pose_model(pose_model_path)
        options = mp_vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=mp_vision.RunningMode.VIDEO,
            min_pose_detection_confidence=DETECTION_CONFIDENCE,
            min_pose_presence_confidence=DETECTION_CONFIDENCE,
            min_tracking_confidence=TRACKING_CONFIDENCE,
        )
        landmarker_cls = mp_vision.PoseLandmarker

    with landmarker_cls.create_from_options(options) as landmarker:
        for frame_idx, timestamp_ms, fps_value, bgr_frame in _frames_from_video(
            video_path, start_sec, end_sec
        ):
            frame_count += 1
            rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            if config.modality == "hand":
                result: mp_vision.HandLandmarkerResult = (
                    landmarker.detect_for_video(mp_image, timestamp_ms)
                )
                # Accumulate normalised (x, y) per hand / landmark
                for hand_i, hand_lm_list in enumerate(result.hand_landmarks):
                    # Flip handedness: MediaPipe uses selfie/mirror convention
                    raw_label = result.handedness[hand_i][0].category_name
                    side = FLIP_HAND.get(raw_label, raw_label)
                    for lm_idx, lm in enumerate(hand_lm_list):
                        tracks[(side, lm_idx)].append((lm.x, lm.y))
                if annotate:
                    _draw_hand_landmarks(bgr_frame, result)

            else:  # pose
                result: mp_vision.PoseLandmarkerResult = (
                    landmarker.detect_for_video(mp_image, timestamp_ms)
                )
                for pose_lm_list in result.pose_landmarks:
                    for lm_idx, lm in enumerate(pose_lm_list):
                        tracks[("Pose", lm_idx)].append((lm.x, lm.y))
                if annotate:
                    _draw_pose_landmarks(bgr_frame, result)

            # Lazy VideoWriter init on first frame (needs frame dimensions)
            if save_video_dir is not None and video_writer is None:
                h, w = bgr_frame.shape[:2]
                out_path = (
                    Path(save_video_dir) / participant / activity_date
                    / f"{activity}_annotated.mp4"
                )
                out_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Prefer H.264/avc1 for better compatibility, with mp4v fallback
                codecs_to_try = ["avc1", "mp4v"]
                for codec in codecs_to_try:
                    fourcc = cv2.VideoWriter_fourcc(*codec)
                    vw = cv2.VideoWriter(str(out_path), fourcc, fps_value, (w, h))
                    if vw is not None and vw.isOpened():
                        video_writer = vw
                        log.info(
                            "Saving annotated video to %s using codec %s",
                            out_path,
                            codec,
                        )
                        break
                    # Clean up a writer that failed to open
                    if vw is not None:
                        vw.release()

                if video_writer is None:
                    log.error(
                        "Failed to initialize VideoWriter for %s with codecs %s; "
                        "annotated video will not be saved.",
                        out_path,
                        ", ".join(codecs_to_try),
                    )

            if video_writer is not None:
                video_writer.write(bgr_frame)

            if display:
                cv2.imshow(f"Self Care Selfies — {activity}", bgr_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    if video_writer is not None:
        video_writer.release()
    if display:
        cv2.destroyAllWindows()

    # Build LandmarkTrack objects for the selected feature landmarks only
    landmark_tracks: list[LandmarkTrack] = []
    metrics_list: list[dict] = []

    if config.modality == "hand":
        for side in ("Left", "Right"):
            for lm_idx in config.features:
                t = tracks.get((side, lm_idx), [])
                m = compute_metrics(t) if t else None
                lt = LandmarkTrack(
                    hand=side,
                    landmark_idx=lm_idx,
                    landmark_name=config.landmark_dict.get(lm_idx, str(lm_idx)),
                    frames=t,
                )
                landmark_tracks.append(lt)
                if m:
                    metrics_list.append(
                        {
                            "activity": activity,
                            "hand": side,
                            "landmark": lt.landmark_name,
                            "participant": participant,
                            "date": activity_date,
                            **m,
                            "fps": fps_value,
                            "num_frames": frame_count,
                        }
                    )
    else:  # pose
        for lm_idx in config.features:
            t = tracks.get(("Pose", lm_idx), [])
            m = compute_metrics(t) if t else None
            lt = LandmarkTrack(
                hand="Pose",
                landmark_idx=lm_idx,
                landmark_name=config.landmark_dict.get(lm_idx, str(lm_idx)),
                frames=t,
            )
            landmark_tracks.append(lt)
            if m:
                metrics_list.append(
                    {
                        "activity": activity,
                        "hand": "Pose",
                        "landmark": lt.landmark_name,
                        "participant": participant,
                        "date": activity_date,
                        **m,
                        "fps": fps_value,
                        "num_frames": frame_count,
                    }
                )

    return VideoAnalysisResult(
        participant=participant,
        activity_date=activity_date,
        activity=activity,
        fps=fps_value,
        num_frames=frame_count,
        tracks=landmark_tracks,
        metrics=metrics_list,
    )


# ---------------------------------------------------------------------------
# Directory crawling & CSV I/O
# ---------------------------------------------------------------------------
_CSV_HEADERS = [
    "activity", "hand", "landmark", "participant", "date",
    "displacement", "total_travel", "average_velocity",
    "peak_velocity", "normed_velocity", "velocity_peaks",
    "fps", "num_frames",
]

_SUPPORTED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}


def _make_ignore_key(participant: str, activity_date: str, activity: str) -> str:
    return f"{participant}|{activity_date}|{activity}"


def _read_existing_output(output_file: str) -> tuple[set[str], list[list]]:
    """
    Read an existing output CSV to build an ignore set (videos already processed)
    and retain existing rows for appending.
    """
    ignore_set: set[str] = set()
    rows: list[list] = []
    if not os.path.exists(output_file):
        return ignore_set, rows
    with open(output_file, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            return ignore_set, rows
        # Map column names to indices for forward-compatibility
        try:
            act_i = header.index("activity")
            par_i = header.index("participant")
            dat_i = header.index("date")
        except ValueError:
            # Unrecognised format — skip ignore logic
            return ignore_set, rows
        for row in reader:
            if len(row) <= max(act_i, par_i, dat_i):
                continue
            rows.append(row)
            key = _make_ignore_key(row[par_i], row[dat_i], row[act_i])
            ignore_set.add(key)
    return ignore_set, rows


def _crawl_video_dir(
    video_dir: str,
    ignore_set: set[str],
    **analysis_kwargs,
) -> tuple[list[dict], int, int]:
    """
    Walk video_dir (expected structure: video_dir/<participant>/<date>/<activity>.*),
    skip videos already in ignore_set, and return a tuple of:
    (metrics list, success count, failure count).
    """
    all_metrics: list[dict] = []
    success_count = 0
    failure_count = 0
    video_dir_path = Path(video_dir).resolve()

    for path in sorted(video_dir_path.rglob("*")):
        if path.suffix.lower() not in _SUPPORTED_EXTENSIONS:
            continue
        # Expect exactly two parent directories between video_dir and the file
        try:
            rel = path.relative_to(video_dir_path)
        except ValueError:
            continue
        parts = rel.parts  # (participant, date, filename)
        if len(parts) != 3:
            log.warning("Skipping %s — expected <participant>/<date>/<file>", path)
            continue
        participant, activity_date, filename = parts
        activity = path.stem   # filename without extension

        key = _make_ignore_key(participant, activity_date, activity)
        if key in ignore_set:
            log.debug("Already processed: %s", key)
            continue

        log.info(
            "Processing participant=%s  date=%s  activity=%s",
            participant, activity_date, activity,
        )
        try:
            result = analyze_video_file(
                str(path),
                activity=activity,
                participant=participant,
                activity_date=activity_date,
                **analysis_kwargs,
            )
            all_metrics.extend(result.metrics)
            success_count += 1
        except Exception as exc:
            log.error("Failed to process %s: %s", path, exc)
            failure_count += 1

    return all_metrics, success_count, failure_count


def _process_from_csv(
    input_csv: str,
    video_dir: str,
    **analysis_kwargs,
) -> tuple[list[dict], int, int]:
    """
    Process a scheduled list of videos defined in a CSV manifest.

    Expected columns: participant, date, activity, extension,
                      start_sec, end_sec
                      
    Returns: (metrics list, success count, failure count)
    """
    all_metrics: list[dict] = []
    success_count = 0
    failure_count = 0
    with open(input_csv, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        
        # Validate required columns exist
        required_columns = {"participant", "date", "activity"}
        if reader.fieldnames is None:
            log.error("CSV file %s appears to be empty or malformed", input_csv)
            return all_metrics, 0, 0
            
        missing_columns = required_columns - set(reader.fieldnames)
        if missing_columns:
            log.error(
                "CSV file %s is missing required columns: %s. "
                "Required columns are: %s",
                input_csv,
                ", ".join(sorted(missing_columns)),
                ", ".join(sorted(required_columns)),
            )
            return all_metrics, 0, 0
        
        # start=2 because DictReader reads the header as line 1; data rows start at line 2
        for row_index, row in enumerate(reader, start=2):
            participant = row["participant"]
            activity_date = row["date"]
            activity = row["activity"]
            ext = row.get("extension", "mp4").lstrip(".")
            raw_start = row.get("start_sec", 0)
            raw_end = row.get("end_sec", float("inf"))
            try:
                start_sec = float(raw_start)
            except (TypeError, ValueError):
                log.error(
                    "Invalid numeric value for start_sec in CSV %s at data row %d: %r",
                    input_csv,
                    row_index,
                    raw_start,
                )
                continue
            try:
                end_sec = float(raw_end)
            except (TypeError, ValueError):
                log.error(
                    "Invalid numeric value for end_sec in CSV %s at data row %d: %r",
                    input_csv,
                    row_index,
                    raw_end,
                )
                continue

            video_path = (
                Path(video_dir) / participant / activity_date / f"{activity}.{ext}"
            )
            if not video_path.exists():
                log.warning("Video not found: %s", video_path)
                continue

            log.info(
                "Processing (csv) participant=%s  date=%s  activity=%s",
                participant, activity_date, activity,
            )
            try:
                result = analyze_video_file(
                    str(video_path),
                    activity=activity,
                    participant=participant,
                    activity_date=activity_date,
                    start_sec=start_sec,
                    end_sec=end_sec,
                    **analysis_kwargs,
                )
                all_metrics.extend(result.metrics)
                success_count += 1
            except Exception as exc:
                log.error("Failed to process %s: %s", video_path, exc)
                failure_count += 1

    return all_metrics, success_count, failure_count


def process_all_videos(
    video_dir: str,
    output_file: str,
    input_csv: str | None = None,
    hand_model_path: str | None = None,
    pose_model_path: str | None = None,
    display: bool = False,
    save_video_dir: str | None = None,
) -> None:
    """
    Process all videos and write metrics to output_file (CSV).

    If input_csv is provided, only process those videos.
    Otherwise, crawl video_dir and skip videos already in output_file.
    """
    analysis_kwargs = dict(
        hand_model_path=hand_model_path,
        pose_model_path=pose_model_path,
        display=display,
        save_video_dir=save_video_dir,
    )

    if input_csv:
        new_metrics, success_count, failure_count = _process_from_csv(
            input_csv, video_dir, **analysis_kwargs
        )
        existing_rows: list[list] = []
    else:
        ignore_set, existing_rows = _read_existing_output(output_file)
        new_metrics, success_count, failure_count = _crawl_video_dir(
            video_dir, ignore_set, **analysis_kwargs
        )

    # Convert new metric dicts to rows, filling any missing columns with ""
    new_rows = [
        [str(m.get(col, "")) for col in _CSV_HEADERS]
        for m in new_metrics
    ]

    all_rows = existing_rows + new_rows
    
    # Sort rows by named columns to avoid depending on positional column order.
    # This is more robust if _CSV_HEADERS changes (e.g., columns re-ordered or added).
    sort_columns = ("activity", "hand", "landmark", "participant", "date")
    sort_indices = [_CSV_HEADERS.index(col) for col in sort_columns]
    all_rows.sort(key=lambda r: tuple(r[i] for i in sort_indices))

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(_CSV_HEADERS)
        writer.writerows(all_rows)

    log.info(
        "Wrote %d total rows (%d new) to %s",
        len(all_rows), len(new_rows), output_file,
    )
    
    # Log processing summary
    total_videos = success_count + failure_count
    if total_videos > 0:
        log.info(
            "Processing complete: %d videos succeeded, %d videos failed (total: %d)",
            success_count, failure_count, total_videos
        )
        if failure_count > 0:
            log.warning(
                "Review log for details on the %d failed video(s).",
                failure_count
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="self_care_selfies",
        description=(
            "Analyse Self Care Selfies videos and compute motion metrics "
            "using MediaPipe landmark detection."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Setup (conda — recommended):
  conda create -n selfie_video python=3.13
  conda activate selfie_video
  pip install -r requirements.txt

Setup (venv alternative):
  python3.13 -m venv .venv && source .venv/bin/activate
  pip install -r requirements.txt

Note: mediapipe is NOT available via conda/conda-forge; you must install it with
pip (as shown above), even when working inside a conda environment.

Examples:
  # crawl 'videos/' directory, write to output.csv
  python self_care_selfies.py

  # specify paths explicitly
  python self_care_selfies.py --video-dir videos --output results.csv

  # use a manifest CSV to process a specific subset
  python self_care_selfies.py --video-dir videos --input-csv schedule.csv

  # show live video overlay (local/dev use only — not for headless servers)
  python self_care_selfies.py --display

  # save annotated videos (landmarks drawn) to a directory
  python self_care_selfies.py --save-video annotated/

  # use locally cached model files
  python self_care_selfies.py --hand-model ./models/hand_landmarker.task
""",
    )
    parser.add_argument(
        "--video-dir", default="videos",
        help="Root directory containing participant/date/activity videos. Default: 'videos'",
    )
    parser.add_argument(
        "--output", default="output.csv",
        help="Path to output CSV file. Default: 'output.csv'",
    )
    parser.add_argument(
        "--input-csv", default=None,
        help=(
            "Optional manifest CSV specifying which videos to process. "
            "Columns: participant, date, activity, extension, start_sec, end_sec"
        ),
    )
    parser.add_argument(
        "--hand-model", default=None,
        help="Path to HandLandmarker .task model file. Auto-downloaded if omitted.",
    )
    parser.add_argument(
        "--pose-model", default=None,
        help="Path to PoseLandmarker .task model file. Auto-downloaded if omitted.",
    )
    parser.add_argument(
        "--display", action="store_true", default=False,
        help="Show annotated video in a window (requires a display; not for servers).",
    )
    parser.add_argument(
        "--save-video", default=None, metavar="DIR",
        help=(
            "Save annotated videos (landmarks overlaid) to this directory. "
            "Output path: <DIR>/<participant>/<date>/<activity>_annotated.mp4. "
            "Works on headless servers (no display required)."
        ),
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity. Default: INFO",
    )
    return parser


def main() -> None:
    setup_logging()  # Configure logging for CLI use
    
    parser = build_parser()
    args = parser.parse_args()

    logging.getLogger().setLevel(args.log_level)

    if not os.path.isdir(args.video_dir):
        log.error("Video directory does not exist: %s", args.video_dir)
        sys.exit(1)

    process_all_videos(
        video_dir=args.video_dir,
        output_file=args.output,
        input_csv=args.input_csv or None,
        hand_model_path=args.hand_model,
        pose_model_path=args.pose_model,
        display=args.display,
        save_video_dir=args.save_video,
    )


if __name__ == "__main__":
    main()
