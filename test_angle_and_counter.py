"""
Unit tests for the pure-logic pieces (angle math, rep counting) that
don't depend on a webcam or MediaPipe model.

Run with:
    python -m pytest tests/
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from angle_utils import calculate_angle, to_pixel_coords  # noqa: E402
from rep_counter import RepCounter  # noqa: E402


def test_calculate_angle_straight_line():
    # a-b-c forming a straight line should be ~180 degrees
    a, b, c = (0, 0), (1, 0), (2, 0)
    angle = calculate_angle(a, b, c)
    assert math.isclose(angle, 180.0, abs_tol=1e-6)


def test_calculate_angle_right_angle():
    a, b, c = (0, 1), (0, 0), (1, 0)
    angle = calculate_angle(a, b, c)
    assert math.isclose(angle, 90.0, abs_tol=1e-6)


def test_to_pixel_coords():
    point = [0.5, 0.5]
    pixels = to_pixel_coords(point, 640, 480)
    assert pixels == (320, 240)


def test_rep_counter_full_cycle():
    counter = RepCounter(down_threshold=160, up_threshold=30)

    # Arm extended -> stage goes to "down", no rep yet
    stage, count = counter.update(170)
    assert stage == "down"
    assert count == 0

    # Arm curled after being down -> rep counted, stage "up"
    stage, count = counter.update(20)
    assert stage == "up"
    assert count == 1

    # Staying curled shouldn't double-count
    stage, count = counter.update(15)
    assert stage == "up"
    assert count == 1


def test_rep_counter_requires_down_before_up():
    counter = RepCounter()
    # Starting curled with no prior "down" stage should not count a rep
    stage, count = counter.update(20)
    assert count == 0


def test_rep_counter_reset():
    counter = RepCounter()
    counter.update(170)
    counter.update(20)
    assert counter.counter == 1
    counter.reset()
    assert counter.counter == 0
    assert counter.stage is None
