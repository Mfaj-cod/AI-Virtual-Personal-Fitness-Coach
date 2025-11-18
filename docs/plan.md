# AI Virtual Personal Fitness Coach Plan

## Overview
- Real-time pose detection using MediaPipe Pose via webcam feed.
- Streamlit app for UI: video stream, rep counter, timer, workout selection, feedback, and logs.
- Modular Python package structure with utilities, models, and services.

## Components
1. src/pose/pose_detector.py
   - Initialize MediaPipe Pose.
   - Process frames, return landmarks and metrics (joint angles, visibility).

2. src/pose/angle_utils.py
   - Compute joint angles (elbow, knee, shoulder) using vector math.

3. src/workouts/workout_tracker.py
   - Define classes per exercise with thresholds and state machine for rep counting.
   - Track timers, session metadata, and aggregate statistics.

4. src/logging/log_manager.py
   - Persist workout sessions (CSV/JSON) with timestamps, reps, duration.

5. app.py
   - Streamlit entry point.
   - Capture webcam frames (via streamlit-webrtc), integrate pose detector.
   - Display live video with overlay, metrics, rep count, session summary.

6. data/ directory
   - Store saved workout logs.

7. tests/
   - Basic unit tests for angle calculations and workout state machine.

## Key Features
- Exercise selection (e.g., squats, push-ups, bicep curls) with tailored joint tracking.
- Visual feedback: highlight correct form, warnings when angles out of range.
- Rep counter using up/down thresholds and debouncing.
- Session timer and log export.

## Implementation Steps
1. Implement pose detection wrapper and angle utilities.
2. Build workout state machine for counting reps.
3. Create logging and session management utilities.
4. Develop Streamlit UI integrating webcam, metrics, rep counter, and controls.
5. Add persistence for workout logs.
6. Write tests and usage instructions.

## Dependencies
- Python 3.10+
- streamlit, streamlit-webrtc
- opencv-python
- mediapipe
- numpy, pandas

