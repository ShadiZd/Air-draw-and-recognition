"""
draw_classifier.py

Lightweight inference wrapper around the CNN trained by
train_draw_classifier.py. Import this into air_draw.py to recognize
whatever the user has drawn on the canvas and show the predicted label
in the HUD, next to the mode/color text.

Usage inside air_draw.py:

    from draw_classifier import DrawClassifier

    classifier = DrawClassifier()   # loads models/draw_classifier.keras
    ...
    # each frame (or throttled, see note below):
    label, confidence = classifier.predict(canvas)
    hud_text = classifier.format_hud_text(label, confidence)
    cv2.putText(display, hud_text, (x, y), ...)

Design notes:
    - Fails soft: if the model/labels can't be loaded, or a frame is
      blank/invalid, predict() returns (None, 0.0) instead of raising,
      so it never crashes the drawing app.
    - Cheap to call, but since a full CNN forward pass every single
      frame is wasteful, air_draw.py should throttle calls (e.g. run
      recognition every N frames, or only when the canvas has enough
      ink and hasn't changed for a few frames). A small helper
      `ShouldPredictThrottle` is included for convenience.
"""

import os
import sys
import json
import time

import cv2
import numpy as np


def resource_dir():
    """Base directory for bundled data (works both frozen and unfrozen)."""
    if getattr(sys, "frozen", False):
        return getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


if getattr(sys, "frozen", False):
    MODEL_DIR = os.path.join(resource_dir(), "models")
else:
    MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "draw_classifier.tflite")
LABELS_PATH = os.path.join(MODEL_DIR, "draw_labels.json")

IMG_SIZE = 64
MIN_INK_PIXELS = 200          # ignore near-empty canvases
CONFIDENCE_THRESHOLD = 0.55   # below this, report "Unknown"


class DrawClassifier:
    def __init__(self, model_path=MODEL_PATH, labels_path=LABELS_PATH,
                 img_size=IMG_SIZE, confidence_threshold=CONFIDENCE_THRESHOLD,
                 debug=True):
        self.img_size = img_size
        self.confidence_threshold = confidence_threshold
        self.debug = debug  # prints raw top prediction to terminal
        self.model = None
        self.labels = []
        self._load_error = None

        try:
            self._load(model_path, labels_path)
        except Exception as e:
            # Keep the app usable even if the model isn't trained yet.
            self._load_error = str(e)
            print(f"[DrawClassifier] Could not load model: {e}")
            print("[DrawClassifier] Recognition disabled until you train "
                  "the model with train_draw_classifier.py")

    def _load(self, model_path, labels_path):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.isfile(labels_path):
            raise FileNotFoundError(f"Labels file not found: {labels_path}")

        # Use a lightweight TFLite interpreter rather than full tensorflow.
        # This deliberately avoids pulling tensorflow (and its modern,
        # incompatible protobuf requirement) into the same environment as
        # mediapipe, which needs an old protobuf. Train with train_draw_
        # classifier.py in a separate venv that has full tensorflow; only
        # the exported .tflite file + one of the runtimes below is needed
        # here. Tried in order of preference:
        try:
            from ai_edge_litert.interpreter import Interpreter  # newest
        except ImportError:
            try:
                from tflite_runtime.interpreter import Interpreter  # older
            except ImportError:
                # Last resort: full tensorflow's bundled interpreter
                # (only works if this env also has tensorflow installed).
                from tensorflow.lite.python.interpreter import Interpreter

        self.interpreter = Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self._input_details = self.interpreter.get_input_details()
        self._output_details = self.interpreter.get_output_details()
        self.model = self.interpreter  # marks is_ready True once set

        with open(labels_path, "r") as f:
            self.labels = json.load(f)

    @property
    def is_ready(self):
        return self.model is not None and len(self.labels) > 0

    def _preprocess(self, canvas_bgr):
        # Turn every painted pixel into pure white ink, regardless of pen
        # color, so the model always sees the same white strokes it was
        # trained on. (e.g. red text-to-gray is ~29, nearly black, which
        # would otherwise be misclassified.)
        gray = (canvas_bgr.any(axis=2) * 255).astype(np.uint8)

        ink_pixels = int(np.count_nonzero(gray))
        if ink_pixels < MIN_INK_PIXELS:
            if self.debug:
                print(f"[DrawClassifier] skipped: only {ink_pixels} ink "
                      f"pixels (need >= {MIN_INK_PIXELS})")
            return None  # not enough drawn yet

        # crop to the bounding box of the ink so the drawing fills the
        # frame the way training samples likely do, then pad to square.
        ys, xs = np.nonzero(gray)
        y1, y2 = ys.min(), ys.max()
        x1, x2 = xs.min(), xs.max()
        pad = 15
        y1 = max(0, y1 - pad)
        x1 = max(0, x1 - pad)
        y2 = min(gray.shape[0], y2 + pad)
        x2 = min(gray.shape[1], x2 + pad)
        crop = gray[y1:y2, x1:x2]

        h, w = crop.shape
        side = max(h, w)
        square = np.zeros((side, side), dtype=np.uint8)
        y_off = (side - h) // 2
        x_off = (side - w) // 2
        square[y_off:y_off + h, x_off:x_off + w] = crop

        resized = cv2.resize(square, (self.img_size, self.img_size),
                              interpolation=cv2.INTER_AREA)
        norm = resized.astype(np.float32) / 255.0
        return norm[np.newaxis, ..., np.newaxis]  # (1, H, W, 1)

    def predict(self, canvas_bgr):
        """
        Returns (label:str|None, confidence:float in [0,1]).
        label is None if the model isn't loaded, the canvas is (nearly)
        blank, or confidence is below the threshold.
        """
        if not self.is_ready:
            return None, 0.0

        try:
            batch = self._preprocess(canvas_bgr)
            if batch is None:
                return None, 0.0

            in_detail = self._input_details[0]
            out_detail = self._output_details[0]

            batch = batch.astype(in_detail["dtype"])
            self.interpreter.set_tensor(in_detail["index"], batch)
            self.interpreter.invoke()
            probs = self.interpreter.get_tensor(out_detail["index"])[0]

            best_idx = int(np.argmax(probs))
            confidence = float(probs[best_idx])

            if self.debug:
                top3_idx = np.argsort(probs)[::-1][:3]
                top3 = ", ".join(
                    f"{self.labels[i]}={probs[i]*100:.1f}%" for i in top3_idx
                )
                print(f"[DrawClassifier] top guesses: {top3}  "
                      f"(threshold: {self.confidence_threshold*100:.0f}%)")

            if confidence < self.confidence_threshold:
                return None, confidence

            return self.labels[best_idx], confidence
        except Exception as e:
            print(f"[DrawClassifier] predict() error: {e}")
            return None, 0.0

    def format_hud_text(self, label, confidence):
        if label is None:
            return "Recognized: -"
        return f"Recognized: {label} ({confidence * 100:.0f}%)"


class PredictionThrottle:
    """
    Small helper so air_draw.py doesn't run a CNN forward pass every
    single frame. Call should_predict() each loop iteration; it returns
    True only every `interval_sec` seconds.
    """

    def __init__(self, interval_sec=0.75):
        self.interval_sec = interval_sec
        self._last_time = 0.0

    def should_predict(self):
        now = time.time()
        if now - self._last_time >= self.interval_sec:
            self._last_time = now
            return True
        return False