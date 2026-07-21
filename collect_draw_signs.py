"""
Run this first, once per LEFT-hand gesture you want the air-drawing app
to recognize.

This is the same idea as collect_signs.py from the sign-language project,
but restricted to the LEFT hand only, since in the air-drawing app the
left hand is the one making control gestures (Draw / Erase / Clear /
Change) while the right hand is the "pen".

Usage:
    python collect_draw_signs.py

Suggested labels (type these exactly when prompted so air_draw.py
recognizes them out of the box):
    Draw    - left hand shape that means "start drawing"
    Erase   - left hand shape that means "erase"
    Clear   - left hand shape that means "clear the whole canvas"
    Change  - left hand shape that means "change color"

Controls while recording a gesture:
    - Hold your LEFT hand in frame making the gesture
    - Press SPACE to capture a sample (capture ~15-20 samples per
      gesture, varying hand angle/position slightly for robustness)
    - Press ESC to finish that gesture and move to the next, or quit

Samples are saved as .npy files into the "encodings_draw" folder as:
    <label>_<index>.npy
which DrawSignRec / SimpleSignRec's load_encoding_signs() reads
automatically.
"""

import cv2
import os
import numpy as np
from draw_signrec import DrawSignRec

OUTPUT_DIR = "encodings_draw"
os.makedirs(OUTPUT_DIR, exist_ok=True)

dsr = DrawSignRec()
cap = cv2.VideoCapture(0)

while True:
    label = input("Enter LEFT-hand gesture label (Draw/Erase/Clear/Change, "
                   "or leave empty to quit): ").strip()
    if not label:
        break

    existing = [f for f in os.listdir(OUTPUT_DIR) if f.startswith(label + "_")]
    sample_idx = len(existing)

    print(f"Recording '{label}' (LEFT hand only). "
          f"SPACE = capture, ESC = done with this gesture.")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        # Mirror BEFORE processing so MediaPipe's Left/Right labels match
        # what the user sees on screen (see draw_signrec.py docstring).
        frame = cv2.flip(frame, 1)

        hands_info = dsr.process(frame)
        left_hand = next((h for h in hands_info if h["label"] == "Left"), None)

        display = frame.copy()
        for h in hands_info:
            color = (0, 255, 0) if h["label"] == "Left" else (255, 120, 0)
            dsr.mp_draw.draw_landmarks(
                display, h["landmarks"], dsr.mp_hands.HAND_CONNECTIONS
            )
            x1, y1, x2, y2 = h["box"]
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            cv2.putText(display, h["label"], (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        status = "LEFT HAND FOUND" if left_hand else "Show your LEFT hand"
        cv2.putText(display, f"Label: {label}  Samples: {sample_idx}  ({status})",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("Collect Draw Signs (LEFT hand)", display)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == 32 and left_hand is not None:  # SPACE
            vec = dsr.normalize_landmarks(left_hand["landmarks"])
            save_path = os.path.join(OUTPUT_DIR, f"{label}_{sample_idx}.npy")
            np.save(save_path, vec)
            print("Saved:", save_path)
            sample_idx += 1

cap.release()
cv2.destroyAllWindows()
print("Done collecting. Encodings saved in:", OUTPUT_DIR)
