from __future__ import annotations

import queue
import time
import os
from datetime import datetime
from typing import Optional

import cv2
import streamlit as st
from av import VideoFrame
from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, WebRtcMode, webrtc_streamer

from app.pose_estimator import JointAngle, PoseEstimator
from app.rep_counter import DEFAULT_EXERCISES, ExerciseConfig, RepCounter
from app.workout_logger import WorkoutEntry, WorkoutLogger

RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


def format_seconds(seconds: float) -> str:
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


class PoseTransformer(VideoProcessorBase):
    def __init__(self, exercise_config: ExerciseConfig, state_queue: Optional[queue.Queue] = None) -> None:
        self.pose_estimator = PoseEstimator()
        self.rep_counter = RepCounter(exercise_config)
        self.exercise_config = exercise_config
        self.reps = 0
        self.stage = "up"
        self.latest_feedback: str = "Ready"
        self._last_joint_angle: Optional[float] = None
        self.frame_joint_angles = {}
        self.state_queue = state_queue

    def update_exercise(self, config: ExerciseConfig) -> None:
        self.exercise_config = config
        self.rep_counter = RepCounter(config)
        self.reps = 0
        self.stage = "up"
        self.latest_feedback = "Ready"
        self._last_joint_angle = None
        # Send reset state to queue (include timestamp)
        if self.state_queue is not None:
            try:
                self.state_queue.put_nowait({
                    "reps": 0,
                    "stage": "up",
                    "feedback": "Ready",
                    "ts": time.time(),
                })
            except queue.Full:
                pass

    def recv(self, frame: VideoFrame) -> VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        annotated, joint_angles = self.pose_estimator.process_frame(img)
        self.frame_joint_angles = joint_angles
        timestamp = time.time()
        self.reps = self.rep_counter.update(joint_angles, timestamp)
        self.stage = self.rep_counter.stage
        joint_angle = joint_angles.get(self.exercise_config.target_joint)

        self.latest_feedback = self._generate_feedback(joint_angle)

        # Send state update to main thread via queue
        if self.state_queue is not None:
            try:
                self.state_queue.put_nowait({
                    "reps": self.reps,
                    "stage": self.stage,
                    "feedback": self.latest_feedback,
                    "ts": time.time(),
                })
            except queue.Full:
                pass  # Queue is full, skip this update

        try:
            log_path = "data/transformer_log.csv"
            # Ensure data directory exists
            import os
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"{time.time()},{self.reps},{self.stage},{self.latest_feedback}\n")
        except Exception:
            # Never raise from the worker thread on logging failures
            pass

        cv2.putText(
            annotated,
            f"Reps: {self.reps}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            f"Stage: {self.stage}",
            (10, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        if joint_angle:
            cv2.putText(
                annotated,
                f"{joint_angle.name}: {joint_angle.value:.1f}",
                (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 215, 0),
                2,
                cv2.LINE_AA,
            )

        output_frame = VideoFrame.from_ndarray(annotated, format="bgr24")
        output_frame.pts = frame.pts
        output_frame.time_base = frame.time_base
        return output_frame

    def _generate_feedback(self, joint_angle: Optional[JointAngle]) -> str:
        if not joint_angle:
            return "Move into frame"
        self._last_joint_angle = joint_angle.value
        if joint_angle.value <= self.exercise_config.down_threshold:
            return "Great depth! Press up."
        if joint_angle.value >= self.exercise_config.up_threshold:
            return "Extend fully to lockout."
        return "Maintain controlled motion."


def init_session_state() -> None:
    st.session_state.setdefault("exercise_name", "Push-Up")
    st.session_state.setdefault("current_reps", 0)
    st.session_state.setdefault("start_time", None)
    st.session_state.setdefault("workout_duration", 0.0)
    st.session_state.setdefault("logger", WorkoutLogger())
    st.session_state.setdefault("transformer_feedback", "Ready")
    # Create a queue for thread-safe state updates
    if "state_queue" not in st.session_state:
        st.session_state.state_queue = queue.Queue(maxsize=10)


def main() -> None:
    st.set_page_config(
        page_title="AI Virtual Personal Fitness Coach",
        layout="wide",
    )
    st.title("AI Virtual Personal Fitness Coach")
    st.caption("Real-time pose tracking, rep counting, and workout logging.")

    init_session_state()

    exercise_name = st.sidebar.selectbox(
        "Select Exercise",
        list(DEFAULT_EXERCISES.keys()),
        index=list(DEFAULT_EXERCISES.keys()).index(st.session_state.exercise_name),
    )

    if exercise_name != st.session_state.exercise_name:
        st.session_state.exercise_name = exercise_name
        st.session_state.current_reps = 0
        st.session_state.workout_duration = 0.0

    exercise_config = DEFAULT_EXERCISES[exercise_name]

    st.sidebar.markdown("### Session Controls")
    reset_clicked = st.sidebar.button("Reset Reps")

    # Ensure state_queue exists before creating the transformer
    state_queue = st.session_state.get("state_queue", queue.Queue(maxsize=10))
    if "state_queue" not in st.session_state:
        st.session_state.state_queue = state_queue

    col_video, col_metrics = st.columns([2, 1])

    with col_video:
        st.subheader("Live Camera Feed")
        # Capture state_queue in closure to avoid accessing session_state in worker thread
        webrtc_ctx = webrtc_streamer(
            key="fitness-coach",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIG,
            video_processor_factory=lambda sq=state_queue: PoseTransformer(exercise_config, sq),
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

    metrics_placeholder = col_metrics.empty()
    feedback_placeholder = col_metrics.empty()
    timer_placeholder = col_metrics.empty()

    if webrtc_ctx.state.playing:
        if reset_clicked and webrtc_ctx.video_processor:
            webrtc_ctx.video_processor.update_exercise(exercise_config)
            st.session_state.current_reps = 0
            st.session_state.workout_duration = 0.0
            st.session_state.start_time = time.time()
            # Clear the queue
            while not state_queue.empty():
                try:
                    state_queue.get_nowait()
                except queue.Empty:
                    break
        if st.session_state.start_time is None:
            st.session_state.start_time = time.time()

        elapsed = time.time() - st.session_state.start_time
        total_duration = st.session_state.workout_duration + elapsed
        timer_placeholder.metric("Workout Time", format_seconds(total_duration))

        # Aggregate max reps seen in queue to avoid missing a peak value
        # Aggregate queue updates: prefer the record with highest reps, tie-break by newest timestamp
        best_state = None
        best_reps = st.session_state.current_reps
        best_ts = 0.0
        while not state_queue.empty():
            try:
                s = state_queue.get_nowait()
            except queue.Empty:
                break
            try:
                s_reps = int(s.get("reps", best_reps))
            except Exception:
                s_reps = best_reps
            s_ts = float(s.get("ts", 0.0)) if s.get("ts") is not None else 0.0
            # prefer higher reps; if equal, prefer newer timestamp
            if s_reps > best_reps or (s_reps == best_reps and s_ts > best_ts):
                best_reps = s_reps
                best_ts = s_ts
                best_state = s

        if best_state is not None:
            st.session_state.current_reps = best_reps
            st.session_state.transformer_feedback = best_state.get('feedback', 'Ready')

        # Also try to read directly from transformer as fallback (treat as another candidate)
        processor: PoseTransformer = webrtc_ctx.video_processor
        if processor:
            try:
                processor_reps = int(processor.reps)
                # treat processor read as if it were a state with current timestamp
                if processor_reps > st.session_state.current_reps:
                    st.session_state.current_reps = processor_reps
            except (AttributeError, RuntimeError, ValueError):
                pass
        
        metrics_placeholder.metric("Total Reps", st.session_state.current_reps)
        feedback_placeholder.success(st.session_state.transformer_feedback)
    else:
        if st.session_state.start_time is not None:
            st.session_state.workout_duration += time.time() - st.session_state.start_time
            st.session_state.start_time = None

        if reset_clicked:
            st.session_state.current_reps = 0
            st.session_state.workout_duration = 0.0
            st.session_state.start_time = None
            if webrtc_ctx.video_processor:
                webrtc_ctx.video_processor.update_exercise(exercise_config)

        timer_placeholder.metric("Workout Time", format_seconds(st.session_state.workout_duration))
        metrics_placeholder.metric("Total Reps", st.session_state.current_reps)
        feedback_placeholder.info("Press Start to begin tracking.")

    st.divider()
    st.subheader("Workout Summary")

    col_summary, col_log = st.columns([1, 1])
    with col_summary:
        st.metric("Exercise", st.session_state.exercise_name)
        st.metric("Repetitions", st.session_state.current_reps)
        
        # Calculate current total duration (including elapsed time if video is playing)
        if webrtc_ctx.state.playing and st.session_state.start_time is not None:
            current_duration = st.session_state.workout_duration + (time.time() - st.session_state.start_time)
        else:
            current_duration = st.session_state.workout_duration
        st.metric("Duration", format_seconds(current_duration))

        notes = st.text_area("Notes", placeholder="How did this session feel?")
        # Allow saving even with 0 reps (user might want to log an attempt)
        save_disabled = False

        if st.button("Save Workout Log", disabled=save_disabled, key="save_workout_btn"):
            # Calculate final duration including any elapsed time
            if webrtc_ctx.state.playing and st.session_state.start_time is not None:
                final_duration = st.session_state.workout_duration + (time.time() - st.session_state.start_time)
                # Accumulate the elapsed time into workout_duration
                st.session_state.workout_duration = final_duration
                st.session_state.start_time = time.time()  # Reset start time for next session
            else:
                final_duration = st.session_state.workout_duration
            
            # Use the transformer's on-disk log as the single source of truth
            # for saved reps. We scan recent entries (last 30s) and take the
            # maximum recorded reps. If no recent log entries exist, fall
            # back to session_state.current_reps.
            final_reps = st.session_state.current_reps
            try:
                log_path = "data/transformer_log.csv"
                if os.path.exists(log_path):
                    cutoff = time.time() - 30.0  # consider last 30 seconds
                    max_logged_reps = -1
                    with open(log_path, "r", encoding="utf-8") as lf:
                        for line in lf:
                            parts = line.strip().split(",")
                            if len(parts) < 2:
                                continue
                            try:
                                ts = float(parts[0])
                                reps_logged = int(parts[1])
                            except Exception:
                                continue
                            if ts >= cutoff and reps_logged > max_logged_reps:
                                max_logged_reps = reps_logged
                    if max_logged_reps >= 0:
                        final_reps = max_logged_reps
            except Exception:
                # If anything goes wrong reading the log, keep session value
                pass

            # DEBUG: show sources used to compute final_reps
            try:
                # show queue contents summary
                q = st.session_state.get("state_queue")
                q_latest = None
                if q is not None:
                    # peek remaining items (non-destructive not available; just show that queue is non-empty)
                    q_size = q.qsize()
                    q_latest = None
                    while not q.empty():
                        try:
                            q_latest = q.get_nowait()
                        except queue.Empty:
                            break
                else:
                    q_size = 0
                st.info(f"DEBUG: session_state.current_reps={st.session_state.get('current_reps')} | queue_size={q_size} | queue_latest={q_latest} | transformer_reps={(webrtc_ctx.video_processor.reps if webrtc_ctx.video_processor else 'N/A')} | final_reps={final_reps}")
            except Exception as _dbg:
                st.warning(f"DEBUG: failed to collect debug info: {_dbg}")
            
            
            try:
                entry = WorkoutEntry(
                    timestamp=datetime.utcnow(),
                    exercise=st.session_state.exercise_name,
                    reps=final_reps,
                    duration_sec=final_duration,
                    notes=notes.strip(),
                )
                st.session_state.logger.append(entry)
                st.success(f"Workout saved! ({final_reps} reps, {format_seconds(final_duration)})")
                st.session_state.current_reps = 0
                st.session_state.workout_duration = 0.0
                st.session_state.start_time = None
                # Force refresh by using a unique key
                if 'history_key' not in st.session_state:
                    st.session_state.history_key = 0
                st.session_state.history_key += 1
                st.rerun()
            except Exception as e:
                st.error(f"Error saving workout: {str(e)}")
                st.exception(e)

    with col_log:
        st.markdown("#### Recent Workout History")
        # Use key to force refresh
        history_key = st.session_state.get('history_key', 0)
        history_df = st.session_state.logger.load_history()
        if history_df.empty:
            st.info("No workout history yet. Complete a workout and save it to see it here.")
        else:
            st.dataframe(history_df, width="stretch", height=300, key=f"history_{history_key}")


if __name__ == "__main__":
    main()

