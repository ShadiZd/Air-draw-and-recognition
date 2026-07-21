"""
air_draw.py

Full-screen "air drawing" app, built on top of the same sign-recognition
approach as the sign-language project (encoded landmark vectors matched
by nearest neighbour, see simple_signrec.py / draw_signrec.py).

How it works:
    - Run collect_draw_signs.py first to record your LEFT-hand gestures:
      Draw, Erase, Clear, Change (see that file's docstring).
    - Then run this file.

Roles of each hand:
    LEFT hand  -> makes a gesture that sets the current "tool":
                    Draw    - right hand's fingertip draws a line
                    Erase   - right hand's fingertip erases nearby ink
                    Clear   - wipes the whole canvas (triggers once per
                              time you make the gesture, not continuously)
                    Change  - cycles to the next pen color (also
                              triggers once per gesture, not continuously)
                  If the left hand shows no recognized gesture (or is not
                  in frame), the pen is simply lifted: nothing is drawn.

    RIGHT hand -> its index-fingertip (landmark #8) position is the
                  "pen tip" used for drawing / erasing.

Controls:
    ESC    - quit
    c      - clear canvas manually (keyboard shortcut, in case gestures
             aren't set up yet)
    q      - also quits
    SPACE  - freeze/unfreeze the camera feed. While frozen, the webcam
             device is released so other programs can use it; the last
             frame stays on screen (with a "FROZEN" banner) until you
             press SPACE again to resume.
"""

import os
import cv2
import numpy as np

from draw_signrec import DrawSignRec
from draw_classifier import DrawClassifier, PredictionThrottle

ENCODINGS_DIR = "encodings_draw"
WINDOW_NAME = "Air Draw"

# Pen colors to cycle through with the "Change" gesture (BGR for OpenCV)
COLORS = [
    (255, 255, 255), # white
    (0, 0, 255),     # red
    (0, 255, 0),     # green
    (255, 0, 0),     # blue
    (0, 255, 255),   # yellow
    (255, 0, 255),   # magenta
]

INDEX_FINGERTIP = 8
DRAW_THICKNESS = 6
ERASE_RADIUS = 35


def get_point(landmarks, idx, w, h):
    lm = landmarks.landmark[idx]
    return int(lm.x * w), int(lm.y * h)


def main():
    if not os.path.isdir(ENCODINGS_DIR) or not os.listdir(ENCODINGS_DIR):
        print(f"No gesture encodings found in '{ENCODINGS_DIR}/'.")
        print("Run collect_draw_signs.py first to record Draw/Erase/Clear/Change.")
        return

    dsr = DrawSignRec(distance_threshold=0.28, max_num_hands=2)
    dsr.load_encoding_signs(ENCODINGS_DIR)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open the webcam.")
        return

    ret, frame = cap.read()
    if not ret:
        print("Could not read from the webcam.")
        return
    h, w = frame.shape[:2]

    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1600, 900)

    color_idx = 0
    prev_draw_point = None
    prev_mode = None  # used to edge-trigger Clear / Change

    frozen = False
    frozen_frame = None  # last composed frame shown while frozen

    # ---- drawing recognition (CNN) ----
    classifier = DrawClassifier()
    predict_throttle = PredictionThrottle(interval_sec=0.75)
    last_prediction_text = classifier.format_hud_text(None, 0.0)

    while True:
        if not frozen:
            ret, frame = cap.read()
            if not ret:
                break

            # Mirror BEFORE processing (see draw_signrec.py docstring) so
            # MediaPipe's Left/Right labels match what the user sees.
            frame = cv2.flip(frame, 1)
            hands_info = dsr.process(frame)

            left_hand = next((hh for hh in hands_info if hh["label"] == "Left"), None)
            right_hand = next((hh for hh in hands_info if hh["label"] == "Right"), None)

            # ---- figure out the current mode from the LEFT hand gesture ----
            mode = None
            if left_hand is not None:
                mode, _dist = dsr.classify_gesture(left_hand["landmarks"])

            just_entered_mode = (mode is not None and mode != prev_mode)

            # ---- act on the mode ----
            if mode == "Draw" and right_hand is not None:
                pt = get_point(right_hand["landmarks"], INDEX_FINGERTIP, w, h)
                if prev_draw_point is not None:
                    cv2.line(canvas, prev_draw_point, pt, COLORS[color_idx], DRAW_THICKNESS)
                prev_draw_point = pt
            else:
                prev_draw_point = None  # lift the pen when not actively drawing

            if mode == "Erase" and right_hand is not None:
                pt = get_point(right_hand["landmarks"], INDEX_FINGERTIP, w, h)
                cv2.circle(canvas, pt, ERASE_RADIUS, (0, 0, 0), -1)

            if mode == "Clear" and just_entered_mode:
                canvas[:] = 0
                last_prediction_text = classifier.format_hud_text(None, 0.0)

            if mode == "Change" and just_entered_mode:
                color_idx = (color_idx + 1) % len(COLORS)

            prev_mode = mode

            # ---- recognize what's been drawn (throttled CNN inference) ----
            if classifier.is_ready and predict_throttle.should_predict():
                label, confidence = classifier.predict(canvas)
                last_prediction_text = classifier.format_hud_text(label, confidence)

            # ---- compose the final frame: camera + canvas overlay ----
            ink_mask = canvas.any(axis=2)
            display = frame.copy()
            display[ink_mask] = canvas[ink_mask]

            # draw hand skeletons + boxes for feedback
            for hh in hands_info:
                box_color = (0, 255, 0) if hh["label"] == "Left" else (255, 120, 0)
                dsr.mp_draw.draw_landmarks(
                    display, hh["landmarks"], dsr.mp_hands.HAND_CONNECTIONS
                )
                x1, y1, x2, y2 = hh["box"]
                cv2.rectangle(display, (x1, y1), (x2, y2), box_color, 2)
                cv2.putText(display, hh["label"], (x1, max(0, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

            if right_hand is not None:
                pt = get_point(right_hand["landmarks"], INDEX_FINGERTIP, w, h)
                cv2.circle(display, pt, 8, COLORS[color_idx], -1)

            hud = (f"Mode: {mode or '-'}   Color: {color_idx + 1}/{len(COLORS)}   "
                   f"{last_prediction_text}")
            cv2.rectangle(display, (0, 0), (w, 40), (0, 0, 0), -1)
            cv2.putText(display, hud, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (255, 255, 255), 2)
            cv2.putText(display, "ESC/q: quit   c: clear   SPACE: freeze",
                        (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

            frozen_frame = display  # keep a copy in case we freeze next
        else:
            # Frozen: don't touch the webcam at all, just re-show the last
            # composed frame so another program can grab the device.
            display = frozen_frame.copy()
            banner_y = 40
            cv2.rectangle(display, (0, banner_y), (w, banner_y + 40), (0, 0, 150), -1)
            cv2.putText(display, "FROZEN - camera released, press SPACE to resume",
                        (10, banner_y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2)

        cv2.imshow(WINDOW_NAME, display)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break
        elif key == ord('c'):
            canvas[:] = 0
            last_prediction_text = classifier.format_hud_text(None, 0.0)
        elif key == ord(' '):
            frozen = not frozen
            if frozen:
                # Release the device so other camera programs can use it.
                cap.release()
            else:
                # Reacquire the webcam before resuming capture.
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    print("Could not reopen the webcam.")
                    break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()