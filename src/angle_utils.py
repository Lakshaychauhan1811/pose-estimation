"""
angle_utils.py

Helper functions for converting MediaPipe pose landmarks into joint angles.
Keeping this logic separate from the capture loop makes it reusable across
different exercise counters (curls, squats, etc.) and easy to unit test.
"""

import numpy as np


def calculate_angle(a, b, c):
    """
    Calculate the angle (in degrees) at point `b`, formed by the line
    segments a-b and b-c.

    Args:
        a, b, c: Iterable of (x, y) coordinates (normalized 0-1 or pixel
                  space both work, since we only care about the angle).

    Returns:
        float: angle in degrees, in the range [0, 180].
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


def get_landmark_xy(landmarks, landmark_enum):
    """
    Pull the (x, y) normalized coordinates for a given landmark out of
    MediaPipe's landmark list.

    Args:
        landmarks: results.pose_landmarks.landmark from MediaPipe
        landmark_enum: e.g. mp_pose.PoseLandmark.LEFT_SHOULDER

    Returns:
        list[float, float]: [x, y]
    """
    point = landmarks[landmark_enum.value]
    return [point.x, point.y]


def to_pixel_coords(point, frame_width, frame_height):
    """
    Convert a normalized (x, y) coordinate to pixel coordinates for the
    given frame dimensions. Useful for placing text/overlays with cv2.

    Args:
        point: [x, y] normalized coordinates
        frame_width: width of the video frame in pixels
        frame_height: height of the video frame in pixels

    Returns:
        tuple[int, int]: pixel coordinates
    """
    return tuple(np.multiply(point, [frame_width, frame_height]).astype(int))
