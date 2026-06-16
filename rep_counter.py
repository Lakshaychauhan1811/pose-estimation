"""
rep_counter.py

A small stateful class that tracks exercise stage ("up"/"down") and rep
count based on a joint angle. Pulled out of the main loop so the counting
logic can be tested independently of OpenCV/MediaPipe.
"""


class RepCounter:
    """
    Tracks repetitions of a single-joint exercise (e.g. bicep curl) based
    on threshold angles for the "down" and "up" positions.
    """

    def __init__(self, down_threshold=160, up_threshold=30):
        self.down_threshold = down_threshold
        self.up_threshold = up_threshold
        self.stage = None
        self.counter = 0

    def update(self, angle):
        """
        Feed in the current joint angle and update stage/counter state.

        Args:
            angle (float): current joint angle in degrees.

        Returns:
            tuple[str, int]: (stage, counter) after this update.
        """
        if angle > self.down_threshold:
            self.stage = "down"

        if angle < self.up_threshold and self.stage == "down":
            self.stage = "up"
            self.counter += 1

        return self.stage, self.counter

    def reset(self):
        """Reset the counter and stage back to their initial state."""
        self.stage = None
        self.counter = 0
