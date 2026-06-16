"""
pose_estimation.py

Real-time human pose estimation using MediaPipe and OpenCV, with a live
bicep-curl rep counter as a worked example of turning raw landmarks into
a useful signal.

Usage:
    python src/pose_estimation.py
    python src/pose_estimation.py --camera 1 --down-angle 165 --up-angle 25

Press 'q' to quit the video window.
"""

import argparse

import cv2
import mediapipe as mp

from angle_utils import calculate_angle, get_landmark_xy, to_pixel_coords
from rep_counter import RepCounter

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def draw_rep_counter_overlay(image, stage, counter):
    """Draw the rep/stage status box in the top-left corner of the frame."""
    cv2.rectangle(image, (0, 0), (225, 73), (245, 117, 16), -1)

    cv2.putText(image, "REPS", (15, 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(image, str(counter), (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.putText(image, "STAGE", (65, 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(image, stage if stage else "-", (60, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)


def run(camera_index=0, down_angle=160, up_angle=30, side="LEFT"):
    """
    Run the live pose-estimation + curl-counter loop.

    Args:
        camera_index (int): index of the webcam to use.
        down_angle (float): angle threshold (degrees) for "arm extended".
        up_angle (float): angle threshold (degrees) for "arm curled".
        side (str): "LEFT" or "RIGHT" arm to track.
    """
    shoulder_lm = getattr(mp_pose.PoseLandmark, f"{side}_SHOULDER")
    elbow_lm = getattr(mp_pose.PoseLandmark, f"{side}_ELBOW")
    wrist_lm = getattr(mp_pose.PoseLandmark, f"{side}_WRIST")

    rep_counter = RepCounter(down_threshold=down_angle, up_threshold=up_angle)

    cap = cv2.VideoCapture(camera_index)

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_h, frame_w = frame.shape[:2]

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                landmarks = results.pose_landmarks.landmark

                shoulder = get_landmark_xy(landmarks, shoulder_lm)
                elbow = get_landmark_xy(landmarks, elbow_lm)
                wrist = get_landmark_xy(landmarks, wrist_lm)

                angle = calculate_angle(shoulder, elbow, wrist)
                stage, counter = rep_counter.update(angle)

                cv2.putText(
                    image, f"{angle:.1f}",
                    to_pixel_coords(elbow, frame_w, frame_h),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                )

                draw_rep_counter_overlay(image, stage, counter)

            except AttributeError:
                # No pose detected in this frame; skip gracefully.
                pass

            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )

            cv2.imshow("Pose Estimation", image)

            if cv2.waitKey(10) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description="Real-time pose estimation rep counter.")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default: 0)")
    parser.add_argument("--down-angle", type=float, default=160, help="Angle for arm-extended stage")
    parser.add_argument("--up-angle", type=float, default=30, help="Angle for arm-curled stage")
    parser.add_argument("--side", choices=["LEFT", "RIGHT"], default="LEFT", help="Which arm to track")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        camera_index=args.camera,
        down_angle=args.down_angle,
        up_angle=args.up_angle,
        side=args.side,
    )
