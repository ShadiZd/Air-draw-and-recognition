"""
draw_signrec.py

Small extension on top of SimpleSignRec (simple_signrec.py) that adds
LEFT / RIGHT hand awareness. Everything about how gestures are encoded,
saved and matched is identical to the original sign-language project:

    - a gesture is a normalized (63,) landmark vector (see
      SimpleSignRec.normalize_landmarks)
    - known gestures are loaded from .npy files named <label>_<index>.npy
    - a query vector is matched to the closest known vector (nearest
      neighbour) if the distance is below `distance_threshold`

The only new piece of logic here is telling apart the left hand from the
right hand, because in this project each hand has a different job:

    - LEFT hand  -> makes a gesture (Draw / Erase / Clear / Change) that
                    controls the drawing mode
    - RIGHT hand -> its index-fingertip position is used as the "pen"

IMPORTANT trick used everywhere in this file: we flip the camera frame
BEFORE running it through MediaPipe (instead of only flipping for
display, like collect_signs.py / tmain.py do). MediaPipe's handedness
label assumes a normal (non-mirrored) camera image, so if you flip the
frame first (mirror / "selfie view") the "Left" / "Right" labels it
returns line up correctly with the hand the user actually sees on
screen. If you flip only for display, the labels come out swapped.
"""

import cv2
import numpy as np

from simple_signrec import SimpleSignRec


class DrawSignRec(SimpleSignRec):
    """SimpleSignRec + left/right hand bookkeeping."""

    def __init__(self, distance_threshold=0.15, max_num_hands=2):
        super().__init__(distance_threshold=distance_threshold)
        # Recreate the MediaPipe Hands object in case we want a different
        # max_num_hands than the parent's default (parent already made one
        # with max_num_hands=2, so this is usually a no-op, but keep it
        # explicit / future-proof).
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )

    def process(self, mirrored_frame):
        """
        Run MediaPipe on an already-mirrored (cv2.flip(frame, 1)) BGR frame.

        Returns a list of dicts, one per detected hand:
            {
              "label": "Left" | "Right",   # from the user's own point of view
              "landmarks": <mediapipe landmark list>,
              "box": [x1, y1, x2, y2],
            }
        """
        h, w, _ = mirrored_frame.shape
        rgb = cv2.cvtColor(mirrored_frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        hands_info = []
        if results.multi_hand_landmarks and results.multi_handedness:
            for landmarks, handedness in zip(
                results.multi_hand_landmarks, results.multi_handedness
            ):
                label = handedness.classification[0].label  # "Left" / "Right"
                xs = [lm.x * w for lm in landmarks.landmark]
                ys = [lm.y * h for lm in landmarks.landmark]
                box = [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]
                hands_info.append(
                    {"label": label, "landmarks": landmarks, "box": box}
                )
        return hands_info

    def classify_gesture(self, hand_landmarks):
        """
        Classify a single hand's landmarks against the loaded encodings.
        Returns (label, distance) or (None, None) if nothing matched
        closely enough / nothing is loaded.
        """
        if not self.known_encodings:
            return None, None

        query = self.normalize_landmarks(hand_landmarks)
        distances = [np.linalg.norm(query - known) for known in self.known_encodings]
        best_idx = int(np.argmin(distances))
        best_dist = distances[best_idx]
        if best_dist < self.distance_threshold:
            return self.known_labels[best_idx], best_dist
        return None, best_dist
