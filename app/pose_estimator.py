from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

PoseLandmark = mp.solutions.pose.PoseLandmark


@dataclass(frozen=True)
class JointDefinition:
    """Describe a single joint used for angle calculation."""

    name: str
    points: Tuple[PoseLandmark, PoseLandmark, PoseLandmark] # pyright: ignore[reportInvalidTypeForm]
    angle_range: Tuple[float, float]


@dataclass
class JointAngle:
    """Hold the computed angle for a joint."""

    name: str
    value: float
    is_within_range: bool


class PoseEstimator:
    """Detect pose landmarks and compute joint angles."""

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        joints: Optional[Iterable[JointDefinition]] = None,
    ) -> None:
        self._pose = mp.solutions.pose.Pose(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._drawing_utils = mp.solutions.drawing_utils
        self._drawing_styles = mp.solutions.drawing_styles
        self.joint_definitions: List[JointDefinition] = (
            list(joints)
            if joints
            else [
                JointDefinition(
                    "Left Elbow",
                    (
                        PoseLandmark.LEFT_SHOULDER,
                        PoseLandmark.LEFT_ELBOW,
                        PoseLandmark.LEFT_WRIST,
                    ),
                    (40.0, 165.0),
                ),
                JointDefinition(
                    "Right Elbow",
                    (
                        PoseLandmark.RIGHT_SHOULDER,
                        PoseLandmark.RIGHT_ELBOW,
                        PoseLandmark.RIGHT_WRIST,
                    ),
                    (40.0, 165.0),
                ),
                JointDefinition(
                    "Left Knee",
                    (
                        PoseLandmark.LEFT_HIP,
                        PoseLandmark.LEFT_KNEE,
                        PoseLandmark.LEFT_ANKLE,
                    ),
                    (50.0, 170.0),
                ),
                JointDefinition(
                    "Right Knee",
                    (
                        PoseLandmark.RIGHT_HIP,
                        PoseLandmark.RIGHT_KNEE,
                        PoseLandmark.RIGHT_ANKLE,
                    ),
                    (50.0, 170.0),
                ),
            ]
        )

    def process_frame(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, Dict[str, JointAngle]]:
        """Run pose detection, draw annotations, and return joint angles."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._pose.process(rgb)
        annotated_frame = frame_bgr.copy()
        joint_angles: Dict[str, JointAngle] = {}

        if results.pose_landmarks:
            self._drawing_utils.draw_landmarks(
                annotated_frame,
                results.pose_landmarks,
                mp.solutions.pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self._drawing_styles.get_default_pose_landmarks_style(),
            )
            landmarks = self._landmarks_to_dict(results)
            for joint_def in self.joint_definitions:
                try:
                    angle = self._calculate_joint_angle(landmarks, joint_def.points)
                    is_within_range = joint_def.angle_range[0] <= angle <= joint_def.angle_range[1]
                    joint_angles[joint_def.name] = JointAngle(
                        name=joint_def.name,
                        value=angle,
                        is_within_range=is_within_range,
                    )
                except KeyError:
                    continue

        return annotated_frame, joint_angles

    def _landmarks_to_dict(self, results: mp.framework.formats.landmark_pb2.NormalizedLandmarkList) -> Dict[str, Tuple[float, float, float]]:
        """Convert MediaPipe landmarks to a dictionary keyed by landmark name."""
        landmark_dict: Dict[str, Tuple[float, float, float]] = {}
        for landmark_enum, landmark in enumerate(results.pose_landmarks.landmark):
            landmark_name = PoseLandmark(landmark_enum).name
            landmark_dict[landmark_name] = (landmark.x, landmark.y, landmark.z)
        return landmark_dict

    @staticmethod
    def _calculate_joint_angle(
        landmarks: Dict[str, Tuple[float, float, float]],
        points: Tuple[PoseLandmark, PoseLandmark, PoseLandmark], # pyright: ignore[reportInvalidTypeForm]
    ) -> float:
        """Calculate the angle (in degrees) for the 3-point joint definition."""
        a = np.array(landmarks[points[0].name][:2])
        b = np.array(landmarks[points[1].name][:2])
        c = np.array(landmarks[points[2].name][:2])

        ba = a - b
        bc = c - b

        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-7)
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        angle = np.degrees(np.arccos(cosine_angle))
        return float(angle)

    def close(self) -> None:
        """Release Mediapipe resources."""
        self._pose.close()

    def __del__(self) -> None:
        self.close()

