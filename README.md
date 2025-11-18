
# AI Virtual Personal Fitness Coach

Lightweight Streamlit app that uses MediaPipe + a small rep-counting heuristic to provide a real-time virtual fitness coach: live camera feed, joint-angle based repetition counting, feedback, and workout logging to CSV.

**Key features**
- **Real-time pose estimation**: uses `app.pose_estimator.PoseEstimator` (MediaPipe) to compute joint angles.
- **Rep counting**: rule-based `RepCounter` per exercise (see `app.rep_counter`) for counting repetitions.
- **Live feedback**: text feedback based on joint angles and per-frame annotations rendered on the video feed.
- **Persistent logs**: workouts saved to `data/workout_logs.csv`. Transformer writes `data/transformer_log.csv` for diagnostics.

**Quick links**
- Main app: `app.py`
- Rep counter config: `app/rep_counter.py`
- Pose estimation: `app/pose_estimator.py`
- Workout logging: `app/workout_logger.py`
- Data files: `data/workout_logs.csv`, `data/transformer_log.csv`

**Requirements**
- Python 3.8+ (tested in a conda env named `fitness` in this repo)
- See `requirements.txt` for packages. Typical packages include `streamlit`, `streamlit-webrtc`, `opencv-python`, `mediapipe`, `pandas`, and `numpy`.

**Quick Start (Windows / PowerShell)**
1. (Optional) create and activate a conda environment:

```powershell
conda create -n fitness python=3.10 -y
conda activate fitness
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Run the app:

```powershell
streamlit run app.py
```

4. Open the Local URL shown by Streamlit (e.g. `http://localhost:8501` or the port printed in the terminal).

**Usage notes**
- Select an exercise from the sidebar (there are several preconfigured exercises in `app.rep_counter.DEFAULT_EXERCISES`).
- Click the red `START` button to enable the camera feed and begin tracking.
- The UI shows `Total Reps`, `Workout Time`, and a textual feedback box. Click `Save Workout Log` to persist a row to `data/workout_logs.csv`.
- The app writes `data/transformer_log.csv` from the worker thread every frame for debugging (timestamp, reps, stage, feedback).

**File layout (important files)**
- `app.py` — Streamlit front-end + webrtc glue + session handling.
- `app/pose_estimator.py` — MediaPipe wrapper and joint-angle calculation.
- `app/rep_counter.py` — `ExerciseConfig` and `RepCounter` logic. Add exercises to `DEFAULT_EXERCISES`.
- `app/workout_logger.py` — `WorkoutLogger` that appends to `data/workout_logs.csv`.
- `data/` — CSV logs: `workout_logs.csv` (saved workouts) and `transformer_log.csv` (transformer debug log).

**Adding or tuning exercises**
- To add an exercise, edit `app/rep_counter.py` and add an `ExerciseConfig` to `DEFAULT_EXERCISES`. Use a joint name that exists in `PoseEstimator.joint_definitions` (e.g. `Left Elbow`, `Right Knee`).
- Thresholds (`down_threshold`, `up_threshold`) and `min_hold_time` control rep detection. Tune these for camera angle and exercise form.

**Troubleshooting**
- If streamlit prints warnings like `Please replace use_container_width with width` the app already uses `width='stretch'` in the latest code; ignore remaining older messages.
- Warnings from TensorFlow/MediaPipe and protobuf such as `SymbolDatabase.GetPrototype() is deprecated` are informational and come from third-party libs — they do not indicate a functional error in this app.
- If reps are undercounted at save time: use `data/transformer_log.csv` to verify what the transformer actually observed. Example PowerShell commands:

```powershell
Get-Content data\transformer_log.csv -Tail 100
Get-Content data\workout_logs.csv -Tail 20
```

**Development notes**
- The app uses a worker thread (via `streamlit-webrtc`) to perform per-frame pose estimation. The worker pushes small state messages into a thread-safe `queue.Queue`. The main thread aggregates queue messages and (in recent fixes) also consults the transformer's on-disk log to avoid losing peak counts.
- If you change `PoseTransformer` or `RepCounter`, verify behavior by watching `data/transformer_log.csv` and the UI simultaneously.

**Possible improvements**
- Add a per-exercise calibration flow to auto-tune thresholds.
- Persist logs in a lightweight database (SQLite) instead of CSV for easier querying.
- Add unit tests around `RepCounter.update` for edge cases.

**Contributing**
- Fork, create a branch, and send a PR. Keep changes small and focused.

