# Self Care Selfies

Video analysis pipeline for the **Selfie Video App** project. Analyzes short videos of patients performing everyday tasks and computes motion metrics that correlate with clinical measures of neurological disease status (e.g., MS progression).

---

## Overview

Patients upload videos of themselves performing specific tasks. This pipeline processes each video using [MediaPipe](https://ai.google.dev/edge/mediapipe/) landmark detection and outputs a CSV of motion metrics per landmark per video. Metrics use MediaPipe normalised coordinates (device/resolution-independent) and include `fps` and `num_frames` so downstream code can convert to real-world units.

**Supported activity types:**

| File prefix | Task | Detector | Key landmarks |
|---|---|---|---|
| `Gait` | Walking | PoseLandmarker | Left & right foot index |
| `Talk` | Facial expression while reading/speaking | PoseLandmarker | Eyes, mouth corners |
| `Button` | Buttoning a button | HandLandmarker | Wrist, index fingertip |
| `Eat` | Eating | HandLandmarker | Wrist, index fingertip |
| `Brush` | Brushing teeth | HandLandmarker | Wrist, index fingertip |

Append `L` or `R` to hand activity names to specify which hand (e.g. `BrushL`, `EatR`).

---

## Requirements

- **Python 3.13** recommended (3.11+ required)
- **mediapipe ≥ 0.10.30** — first release with Python 3.13 support (universal `py3-none` wheels, Jan 2026)
- **mediapipe is not on conda-forge** — must be installed via `pip`
- On headless/server environments use `opencv-python-headless` instead of `opencv-python`

---

## Installation

### Recommended: conda + pip

```bash
# 1. Create a dedicated conda environment with Python 3.13
conda create -n selfie_video python=3.13
conda activate selfie_video

# 2. Install Python dependencies via pip
pip install -r requirements.txt
```

> **Note:** Activate the environment before every session: `conda activate selfie_video`

### Alternative: venv

```bash
python3.13 -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Headless / server install

```bash
pip install "mediapipe>=0.10.30" "opencv-python-headless>=4.8.0"
```

---

## Verify the installation

Run these checks after installing — before processing any real videos:

```bash
# 1. Confirm mediapipe version (must be 0.10.30 or later)
python -c "import mediapipe; print('mediapipe', mediapipe.__version__)"

# 2. Confirm opencv is available
python -c "import cv2; print('opencv', cv2.__version__)"

# 3. Confirm the script loads without errors
python self_care_selfies.py --help
```

Expected `--help` output:
```
usage: self_care_selfies [-h] [--video-dir VIDEO_DIR] [--output OUTPUT]
                         [--input-csv INPUT_CSV] [--hand-model HAND_MODEL]
                         [--pose-model POSE_MODEL] [--display] [--save-video DIR]
                         [--log-level {DEBUG,INFO,WARNING,ERROR}]
...
```

### MediaPipe models

Model `.task` files are downloaded automatically to `~/.cache/self_care_selfies/models/` on first run (requires internet). To pre-download or use local copies:

```bash
# Trigger download by running help (safe, no video needed)
python self_care_selfies.py --help

# Or specify local paths at runtime
python self_care_selfies.py --hand-model /path/to/hand_landmarker.task \
                             --pose-model /path/to/pose_landmarker_lite.task
```

---

## Video directory structure

Organise videos under a root directory as `<root>/<participant>/<date>/<activity>.<ext>`:

```
videos/
  patient1/
    01-01-2023/
      BrushL.mp4
      BrushR.mp4
      Talk.mp4
    01-08-2023/
      BrushL.mp4
      BrushR.mp4
      Talk.mp4
  patient2/
    01-01-2023/
      EatL.mov
      Button.mov
      Gait.mov
```

Supported extensions: `.mp4`, `.mov`, `.avi`, `.mkv`

---

## CLI usage

```bash
# Process all videos in 'videos/', write to 'output.csv' (default paths)
python self_care_selfies.py

# Specify paths explicitly
python self_care_selfies.py --video-dir /data/videos --output results.csv

# Process only videos listed in a manifest CSV
python self_care_selfies.py --video-dir videos --input-csv schedule.csv

# Show annotated video overlay while processing (local/dev use only)
python self_care_selfies.py --display

# Save annotated videos (landmarks overlaid) to a directory — works headless
python self_care_selfies.py --save-video annotated/

# Verbose logging for debugging
python self_care_selfies.py --log-level DEBUG
```

**`--save-video DIR`** — Save each processed video with landmark skeleton overlaid as an MP4 file. Output path structure mirrors the input:
```
<DIR>/<participant>/<date>/<activity>_annotated.mp4
```
This works on headless servers — no display is needed. Useful for QA, debugging detection, and creating labelled datasets.

**Incremental processing:** if `output.csv` already exists, videos already present in it are skipped automatically. Re-run the script as new videos arrive.

### Manifest CSV format (`--input-csv`)

To process a specific subset of videos or set custom time windows:

```csv
participant,date,activity,extension,start_sec,end_sec
patient1,01-01-2023,BrushL,mp4,0,30
patient1,01-01-2023,Gait,mp4,5,25
```

---

## Output CSV columns

Each row is one landmark from one video:

| Column | Description |
|---|---|
| `activity` | Video filename stem (e.g. `BrushL`) |
| `hand` | `Left`, `Right`, or `Pose` |
| `landmark` | Landmark name (e.g. `index_finger_tip`, `left_foot_index`) |
| `participant` | Patient/participant ID (from folder name) |
| `date` | Date string (from folder name) |
| `displacement` | Straight-line distance: first → last position (normalised coords) |
| `total_travel` | Cumulative path length across all frames (normalised coords) |
| `average_velocity` | `total_travel / (num_frames − 1)` — mean per-frame displacement |
| `peak_velocity` | Maximum displacement between any two consecutive frames |
| `normed_velocity` | `average_velocity / peak_velocity` ∈ [0, 1] |
| `velocity_peaks` | Direction-reversal count in velocity signal / `num_frames` |
| `fps` | Video frame rate (multiply velocities by `fps` to get units/second) |
| `num_frames` | Number of frames analysed |

**Coordinate system:** x, y ∈ [0, 1] (normalised). Multiply by frame width/height for pixels. Multiply velocity by `fps` for displacement/second.

---

## Importing into a backend pipeline

Core functions are importable without any CLI dependency, suitable for integration into a production video processing backend:

```python
from self_care_selfies import analyze_video_file

result = analyze_video_file(
    video_path="videos/patient1/01-01-2023/BrushL.mp4",
    activity="BrushL",
    participant="patient1",
    activity_date="01-01-2023",
    # Optional: point to cached model files on your server
    hand_model_path="/models/hand_landmarker.task",
    # display=False is the default — safe for headless servers
    # Optional: save annotated video with landmarks drawn
    save_video_dir="annotated/",
)

print(f"Analysed {result.num_frames} frames at {result.fps:.1f} fps")
for m in result.metrics:
    print(m)
```

---

## Files

| File | Description |
|---|---|
| `self_care_selfies.py` | Updated pipeline (MediaPipe Tasks API, Python 3.13) |
| `requirements.txt` | Python dependencies |
| `2022_self_care_selfies.py` | Original 2022 script (archived — uses deprecated API) |
| `output.csv` | Example output (empty header template) |
