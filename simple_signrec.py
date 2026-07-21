import cv2
import os
import glob
import numpy as np
import mediapipe as mp
from gtts import gTTS
import pygame
import tempfile
import threading


class SimpleSignRec:
    """
    Hand sign recognizer, structured similarly to SimpleFacerec:
      - load_encoding_signs(path)  -> loads known gesture "encodings" from disk
      - detect_known_signs(frame)  -> returns hand boxes + predicted labels

    Instead of face photos, "encodings" here are normalized hand landmark
    vectors saved as .npy files (one or more per gesture, see collect_signs.py).
    File naming convention: <label>_<index>.npy  e.g. hello_0.npy, hello_1.npy
    """

    def __init__(self, distance_threshold=0.15):
        self.known_encodings = []   # list of normalized landmark vectors (63,)
        self.known_labels = []      # matching label for each encoding
        self.distance_threshold = distance_threshold  # lower = stricter match

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        self.mp_draw = mp.solutions.drawing_utils
        pygame.mixer.init()
        pygame.mixer.music.set_volume(1)
        self.language = 'ar'
        self.last_word = None
        self.is_speaking = False
        self.lock = threading.Lock()

    # ---------- Loading known gestures ----------

    def load_encoding_signs(self, encodings_path):
        """
        Load pre-recorded landmark encodings from a folder of .npy files.
        Each file should contain a single normalized (21, 3) or (63,) array.
        """
        files = glob.glob(os.path.join(encodings_path, "*.npy"))
        print("{} sign encodings found.".format(len(files)))

        for file_path in files:
            basename = os.path.basename(file_path)
            filename, _ = os.path.splitext(basename)
            # label_0.npy -> label
            label = filename.rsplit("_", 1)[0] if "_" in filename else filename

            vec = np.load(file_path)
            vec = vec.flatten()
            self.known_encodings.append(vec)
            self.known_labels.append(label)

        print(len(set(self.known_labels)),"Sign encodings loaded:", sorted(set(self.known_labels)))

    # ---------- Landmark normalization ----------

    @staticmethod
    def normalize_landmarks(hand_landmarks):
        """
        Convert MediaPipe landmarks into a translation/scale-invariant vector.
        Anchors on the wrist (landmark 0) and scales by the hand size,
        so it works regardless of hand position/distance from camera.
        """
        pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
        wrist = pts[0]
        pts = pts - wrist  # translate so wrist is origin

        scale = np.linalg.norm(pts, axis=1).max()
        if scale > 0:
            pts = pts / scale  # scale-invariant

        return pts.flatten()  # shape (63,)

    # ---------- Detection / classification ----------

    def detect_known_signs(self, frame):
        """
        Detect hands in frame, classify each against known encodings.
        Returns:
            hand_boxes: list of [x1, y1, x2, y2] in original frame coords
            hand_names: list of predicted labels (or "Unknown")
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        hand_boxes = []
        hand_names = []

        if results.multi_hand_landmarks:
            h, w, _ = frame.shape
            for hand_landmarks in results.multi_hand_landmarks:
                # Bounding box from landmark pixel coords
                xs = [lm.x * w for lm in hand_landmarks.landmark]
                ys = [lm.y * h for lm in hand_landmarks.landmark]
                x1, x2 = int(min(xs)), int(max(xs))
                y1, y2 = int(min(ys)), int(max(ys))

                name = ""
                if self.known_encodings:
                    query = self.normalize_landmarks(hand_landmarks)
                    distances = [
                        np.linalg.norm(query - known)
                        for known in self.known_encodings
                    ]
                    best_idx = int(np.argmin(distances))
                    if distances[best_idx] < self.distance_threshold:
                        name = self.known_labels[best_idx]

                hand_boxes.append([x1, y1, x2, y2])
                hand_names.append(name)

        return hand_boxes, hand_names

    def draw_landmarks(self, frame, results_landmarks):
        """Optional: draw MediaPipe hand skeleton on frame for debugging."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )
        return frame
    
    def _speak(self, word):
        try:
            tts = gTTS(text=word, lang=self.language)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
                filename = f.name
            tts.save(filename)
            pygame.mixer.music.load(filename)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.wait(20)
            pygame.mixer.music.unload()
            os.remove(filename)
        finally:
            with self.lock:
                self.is_speaking = False
    def speak(self, word):
        word = word.strip()
        if not word:
            return
        with self.lock:
            # Ignore if it's the same as the previous spoken word
            if word == self.last_word:
                return
            # Ignore if another word is currently being spoken
            if self.is_speaking:
                return
            self.last_word = word
            self.is_speaking = True
        threading.Thread(
            target=self._speak,
            args=(word,),
            daemon=True
        ).start()