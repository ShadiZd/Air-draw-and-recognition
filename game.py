"""
game.py

"Air Draw Challenge" - a small single-player game built on top of the
air-drawing + CNN-recognition machinery (see new_air_draw.py and
draw_classifier.py).

How to play:
    - Run collect_draw_signs.py first so the LEFT-hand gestures
      (Draw / Erase / Clear / Change) are available.
    - Run this file and press any key on the start screen.
    - A 3-2-1-GO countdown runs, then five drawing levels appear one by
      one, picked randomly from: cake, circle, flower, face, star, clock.
    - Draw the requested object with your RIGHT index fingertip while
      holding the Draw gesture. When the CNN recognizes it confidently
      enough, the level clears and the next one starts.
    - After all five levels you get your average accuracy and the total
      time spent drawing.

Controls:
    ESC / q  - quit
    c        - clear the canvas
    n        - skip the current level (counts the best accuracy seen so far)
    r        - play again (on the results screen)
"""

import os
import sys
import time
import random

import cv2
import numpy as np

from draw_signrec import DrawSignRec
from draw_classifier import DrawClassifier, PredictionThrottle, resource_dir

if getattr(sys, "frozen", False):
    ENCODINGS_DIR = os.path.join(resource_dir(), "encodings_draw")
else:
    ENCODINGS_DIR = "encodings_draw"
WINDOW_NAME = "Air Draw Challenge"

COLORS = [
    (255, 255, 255),
    (0, 0, 255),
    (0, 255, 0),
    (255, 0, 0),
    (0, 255, 255),
    (255, 0, 255),
]

CANDIDATES = ["cake", "circle", "flower", "face", "star", "clock"]
ACCEPTED = {c: {c} for c in CANDIDATES}
ACCEPTED["face"] = {"face", "smiley face"}

LEVELS_COUNT = 5
INDEX_FINGERTIP = 8
DRAW_THICKNESS = 6
ERASE_RADIUS = 35

COUNTDOWN_STEP_SEC = 0.9
GO_SEC = 0.7
INTRO_SEC = 1.6
DONE_SEC = 1.8
PREDICT_INTERVAL = 0.5
STREAK_TO_COMPLETE = 2
COMPLETE_CONFIDENCE = 0.55


def get_point(landmarks, idx, w, h):
    lm = landmarks.landmark[idx]
    return int(lm.x * w), int(lm.y * h)


def put_center(display, text, y, scale, color, thickness=2):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x = (display.shape[1] - tw) // 2
    cv2.putText(display, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness)


def fmt_time(sec):
    return f"{sec:.1f}s"


def dim(display, alpha=0.7):
    overlay = display.copy()
    cv2.rectangle(overlay, (0, 0), (display.shape[1], display.shape[0]),
                  (0, 0, 0), -1)
    return cv2.addWeighted(overlay, alpha, display, 1 - alpha, 0)


def list_cameras(max_index=6):
    """Return the indices that actually yield a frame, probing in order."""
    found = []
    for idx in range(max_index):
        cap = cv2.VideoCapture(idx)
        try:
            if cap.isOpened() and cap.read()[0]:
                found.append(idx)
        finally:
            cap.release()
    return found


def open_camera(preferred=None, max_index=6):
    """
    Open a webcam, tolerating the messier setups found on other PCs.

    Tries, in order:
      1. the index the user asked for (CLI arg / AIRDRAW_CAMERA env var),
      2. indices 0..max_index, probing each one for a real frame.

    Falls back to the DirectShow backend, which some Windows drivers refuse
    to serve under the default Media Foundation backend.

    Returns an open, frame-yielding capture, or None.
    """
    order = []
    if preferred is not None:
        order.append(int(preferred))
    order += [i for i in range(max_index) if i not in order]

    api_prefs = [cv2.CAP_ANY, cv2.CAP_DSHOW]

    for api in api_prefs:
        for idx in order:
            cap = cv2.VideoCapture(idx, api)
            if not cap.isOpened():
                cap.release()
                continue
            ret = False
            for _ in range(30):          # give the driver a moment to warm up
                ret, _ = cap.read()
                if not ret:
                    break
            if ret:
                name = ""
                try:
                    name = cap.get(cv2.CAP_PROP_FRAME_WIDTH), \
                        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), \
                        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                except Exception:
                    pass
                print(f"Camera opened: index {idx} (backend api={api}) {name}")
                return cap
            cap.release()

    return None


def main():
    if not os.path.isdir(ENCODINGS_DIR) or not os.listdir(ENCODINGS_DIR):
        print(f"No gesture encodings found in '{ENCODINGS_DIR}/'.")
        print("Run collect_draw_signs.py first to record Draw/Erase/Clear/Change.")
        return

    dsr = DrawSignRec(distance_threshold=0.28, max_num_hands=2)
    dsr.load_encoding_signs(ENCODINGS_DIR)

    classifier = DrawClassifier(debug=False)
    if not classifier.is_ready:
        print("Draw classifier is not ready. Train the model with "
              "train_draw_classifier.py first.")
        return

    available = [c for c in CANDIDATES if c in classifier.labels]
    if not available:
        print("None of the game targets are in the model labels.")
        return
    print("Game targets:", available)

    throttle = PredictionThrottle(interval_sec=PREDICT_INTERVAL)

    preferred = os.environ.get("AIRDRAW_CAMERA")
    if len(sys.argv) > 1:
        preferred = sys.argv[1]

    cap = open_camera(preferred)
    if cap is None:
        print("Could not open any webcam.")
        print(f"Cameras detected: {list_cameras()}")
        print("If your webcam is listed, pass its index as the first "
              "argument, or set AIRDRAW_CAMERA=<index>.")
        print("Also check Windows Settings > Privacy > Camera: "
              "'Camera access' must be on for desktop apps.")
        input("Press Enter to exit...")
        return

    ret, frame = cap.read()
    if not ret:
        print("Could not read from the webcam.")
        input("Press Enter to exit...")
        return
    h, w = frame.shape[:2]

    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    prev_draw_point = None

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1600, 900)

    color_idx = 0
    prev_mode = None

    state = "start"
    state_start = time.time()
    targets = []
    accuracies = []
    times = []
    level_idx = 0
    best_conf = 0.0
    streak = 0
    level_start = None
    level_time = 0.0
    last_text = classifier.format_hud_text(None, 0.0)

    def reset_level():
        nonlocal best_conf, streak, level_time, level_start, last_text, prev_draw_point
        canvas[:] = 0
        prev_draw_point = None
        best_conf = 0.0
        streak = 0
        level_time = 0.0
        level_start = None
        last_text = classifier.format_hud_text(None, 0.0)

    def begin_countdown():
        nonlocal state, state_start, targets, accuracies, times, level_idx
        targets = random.sample(available, min(LEVELS_COUNT, len(available)))
        accuracies = []
        times = []
        level_idx = 0
        state = "countdown"
        state_start = time.time()

    def begin_intro():
        nonlocal state, state_start
        reset_level()
        state = "intro"
        state_start = time.time()

    def begin_playing():
        nonlocal state, level_start
        state = "playing"
        level_start = time.time()

    def finish_level():
        nonlocal state, state_start
        accuracies.append(best_conf)
        times.append(level_time)
        state = "done"
        state_start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        hands_info = dsr.process(frame)

        mode = None
        left_hand = next((hh for hh in hands_info if hh["label"] == "Left"), None)
        right_hand = next((hh for hh in hands_info if hh["label"] == "Right"), None)
        if left_hand is not None:
            mode, _dist = dsr.classify_gesture(left_hand["landmarks"])

        just_entered_mode = (mode is not None and mode != prev_mode)
        now = time.time()

        if state == "countdown":
            elapsed = now - state_start
            if elapsed >= 3 * COUNTDOWN_STEP_SEC + GO_SEC:
                begin_intro()
        elif state == "intro":
            if now - state_start >= INTRO_SEC:
                begin_playing()
        elif state == "playing":
            if mode == "Draw" and right_hand is not None:
                pt = get_point(right_hand["landmarks"], INDEX_FINGERTIP, w, h)
                if prev_draw_point is not None:
                    cv2.line(canvas, prev_draw_point, pt, COLORS[color_idx],
                             DRAW_THICKNESS)
                prev_draw_point = pt
            else:
                prev_draw_point = None

            if mode == "Erase" and right_hand is not None:
                pt = get_point(right_hand["landmarks"], INDEX_FINGERTIP, w, h)
                cv2.circle(canvas, pt, ERASE_RADIUS, (0, 0, 0), -1)

            if mode == "Clear" and just_entered_mode:
                canvas[:] = 0
                last_text = classifier.format_hud_text(None, 0.0)
            if mode == "Change" and just_entered_mode:
                color_idx = (color_idx + 1) % len(COLORS)

            level_time = time.time() - level_start

            if classifier.is_ready and throttle.should_predict():
                label, conf = classifier.predict(canvas)
                last_text = classifier.format_hud_text(label, conf)
                if label in ACCEPTED[targets[level_idx]]:
                    best_conf = max(best_conf, conf)
                    if conf >= COMPLETE_CONFIDENCE:
                        streak += 1
                        if streak >= STREAK_TO_COMPLETE:
                            finish_level()
                    else:
                        streak = 0
                else:
                    streak = 0
        elif state == "done":
            if now - state_start >= DONE_SEC:
                if level_idx + 1 < len(targets):
                    level_idx += 1
                    begin_intro()
                else:
                    state = "results"
                    state_start = now

        prev_mode = mode

        ink_mask = canvas.any(axis=2)
        display = frame.copy()
        display[ink_mask] = canvas[ink_mask]

        for hh in hands_info:
            box_color = (0, 255, 0) if hh["label"] == "Left" else (255, 120, 0)
            dsr.mp_draw.draw_landmarks(
                display, hh["landmarks"], dsr.mp_hands.HAND_CONNECTIONS
            )
            x1, y1, x2, y2 = hh["box"]
            cv2.rectangle(display, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(display, hh["label"], (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

        if right_hand is not None and state == "playing":
            pt = get_point(right_hand["landmarks"], INDEX_FINGERTIP, w, h)
            cv2.circle(display, pt, 8, COLORS[color_idx], -1)

        cv2.rectangle(display, (0, 0), (w, 40), (0, 0, 0), -1)
        if state in ("intro", "playing", "done"):
            hud = (f"Level {level_idx + 1}/{len(targets)}   "
                   f"Target: {targets[level_idx]}")
            if state == "playing":
                hud += f"   Time: {fmt_time(level_time)}"
            cv2.putText(display, hud, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 255, 255), 2)
        cv2.putText(display, "ESC/q: quit   c: clear   n: skip",
                    (w - 420, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (200, 200, 200), 2)

        if state in ("playing", "done"):
            cv2.putText(display, last_text, (10, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        elif state == "results":
            cv2.putText(display, "r: play again   ESC/q: quit",
                        (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (200, 200, 200), 2)

        if state == "start":
            display = dim(display, 0.85)
            put_center(display, "AIR DRAW CHALLENGE", h // 2 - 90, 2.0,
                       (0, 255, 255), 4)
            put_center(display, "Draw with your right index finger, "
                                "control tools with your left hand",
                       h // 2 - 20, 0.7, (255, 255, 255))
            put_center(display, f"5 levels: {', '.join(available)}",
                       h // 2 + 30, 0.6, (200, 200, 200))
            put_center(display, "Press any key to start", h // 2 + 90, 1.0,
                       (0, 255, 0), 2)
        elif state == "countdown":
            elapsed = now - state_start
            if elapsed < COUNTDOWN_STEP_SEC:
                count_text = "3"
            elif elapsed < 2 * COUNTDOWN_STEP_SEC:
                count_text = "2"
            elif elapsed < 3 * COUNTDOWN_STEP_SEC:
                count_text = "1"
            elif elapsed < 3 * COUNTDOWN_STEP_SEC + GO_SEC:
                count_text = "GO!"
            else:
                count_text = ""
            display = dim(display, 0.7)
            if count_text:
                color = (0, 255, 0) if count_text == "GO!" else (0, 255, 255)
                put_center(display, count_text, h // 2 + 40, 3.5, color, 6)
        elif state == "intro":
            display = dim(display, 0.6)
            put_center(display, f"LEVEL {level_idx + 1}/{len(targets)}",
                       h // 2 - 70, 1.2, (0, 255, 255), 3)
            put_center(display, f"DRAW A {targets[level_idx].upper()}",
                       h // 2 + 10, 1.7, (0, 255, 0), 4)
            put_center(display, "Hold the Draw gesture and trace it in the air",
                       h // 2 + 90, 0.7, (200, 200, 200))
        elif state == "done":
            display = dim(display, 0.6)
            result = accuracies[-1]
            put_center(display, "CORRECT!", h // 2 - 60, 1.5, (0, 255, 0), 4)
            put_center(display,
                       f"{targets[level_idx].upper()}   {result * 100:.0f}%   "
                       f"{fmt_time(times[-1])}",
                       h // 2 + 20, 1.0, (255, 255, 255), 2)
        elif state == "results":
            display = dim(display, 0.85)
            avg = (sum(accuracies) / len(accuracies)) if accuracies else 0.0
            total = sum(times)
            put_center(display, "GAME OVER", h // 2 - 240, 1.8, (0, 0, 255), 5)
            put_center(display, f"Average accuracy: {avg * 100:.1f}%",
                       h // 2 - 160, 1.3, (0, 255, 0), 3)
            put_center(display, f"Total time: {fmt_time(total)}",
                       h // 2 - 105, 1.3, (255, 255, 0), 3)
            for i, (tgt, acc, tsec) in enumerate(zip(targets, accuracies, times)):
                put_center(display,
                           f"{i + 1}) {tgt}: {acc * 100:.0f}%  "
                           f"({fmt_time(tsec)})",
                           h // 2 + 10 + i * 40, 0.7, (255, 255, 255))

        cv2.imshow(WINDOW_NAME, display)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break
        if state == "start" and key != 255:
            begin_countdown()
        elif state == "playing":
            if key == ord('c'):
                canvas[:] = 0
                last_text = classifier.format_hud_text(None, 0.0)
            elif key == ord('n'):
                finish_level()
        elif state == "results":
            if key == ord('r'):
                begin_countdown()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()