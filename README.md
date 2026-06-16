# Real-Time Pose Estimation

Real-time human pose estimation built with [MediaPipe](https://developers.google.com/mediapipe) and OpenCV, with a live rep counter as a worked example of turning raw body landmarks into a usable signal.

The webcam feed is processed frame-by-frame to detect 33 body landmarks, which are then used to calculate joint angles (e.g. shoulder-elbow-wrist) in real time. The angle signal drives a simple state machine that counts exercise reps and displays them as an on-screen overlay.

## Demo

Run it and you'll see:
- Skeleton overlay tracking 33 body landmarks live
- Real-time joint angle displayed at the elbow
- A rep counter + stage ("up"/"down") overlay in the corner

## Project Structure

```
pose-estimation/
├── notebooks/
│   └── pose_estimation_exploration.ipynb   # step-by-step prototyping notebook
├── src/
│   ├── angle_utils.py        # joint angle math, landmark/pixel helpers
│   ├── rep_counter.py        # stateful rep-counting logic
│   └── pose_estimation.py    # main CLI app (capture loop + overlay)
├── tests/
│   └── test_angle_and_counter.py   # unit tests for the pure-logic pieces
├── requirements.txt
└── README.md
```

The notebook was where I worked through the pipeline interactively (detection → landmark extraction → angle math → counter logic). Once the logic was working, I pulled the angle math and rep-counting state machine out into standalone modules in `src/` so they're testable independently of the webcam/OpenCV loop, and wrapped the whole thing in a small CLI.

## How It Works

1. **Capture & detect** — each webcam frame is converted to RGB and passed through MediaPipe's `Pose` model, which returns 33 normalized landmark coordinates (shoulders, elbows, wrists, hips, knees, ankles, etc.) plus a per-landmark visibility score.
2. **Angle calculation** — for the curl counter, three landmarks (shoulder, elbow, wrist) are used to compute the angle at the elbow using `atan2` on the two vectors elbow→shoulder and elbow→wrist (`src/angle_utils.py`).
3. **Rep counting state machine** — `RepCounter` (`src/rep_counter.py`) tracks two thresholds: a "down" angle (arm extended, e.g. >160°) and an "up" angle (arm curled, e.g. <30°). A rep is only counted on the down→up transition, which avoids double-counting on small angle jitter.
4. **Overlay rendering** — OpenCV draws the skeleton, the live angle value at the elbow, and a status box showing rep count + current stage.

## Getting Started

### Requirements
- Python 3.9–3.11 (MediaPipe doesn't yet support all versions of 3.12+)
- A webcam

### Installation

```bash
git clone https://github.com/<your-username>/pose-estimation.git
cd pose-estimation
pip install -r requirements.txt
```

### Run it

```bash
python src/pose_estimation.py
```

Press **q** to quit the video window.

### CLI options

```bash
python src/pose_estimation.py --camera 1 --down-angle 165 --up-angle 25 --side RIGHT
```

| Flag | Default | Description |
|---|---|---|
| `--camera` | `0` | Webcam index to use |
| `--down-angle` | `160` | Angle threshold (degrees) for the "arm extended" stage |
| `--up-angle` | `30` | Angle threshold (degrees) for the "arm curled" stage |
| `--side` | `LEFT` | Which arm to track (`LEFT` or `RIGHT`) |

## Running Tests

The angle math and rep-counter state machine are covered by unit tests that don't require a webcam:

```bash
pip install pytest
python -m pytest tests/ -v
```

## Possible Extensions

- Track additional exercises (squats via hip-knee-ankle angle, shoulder press, etc.)
- Smooth the angle signal (e.g. moving average) to reduce jitter from noisy landmark detections
- Swap the OpenCV window for a simple Streamlit dashboard
- Export rep history/session stats to CSV

## Acknowledgments

Built on top of Google's [MediaPipe](https://github.com/google-ai-edge/mediapipe) Pose Landmarker model.

## License

[MIT](LICENSE)
