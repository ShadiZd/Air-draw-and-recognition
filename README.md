# Air Draw

Full-screen "air drawing" app controlled by hand gestures (via MediaPipe),
with a CNN that recognizes what you've drawn and shows it live on screen.

- **LEFT hand** — makes a gesture that sets the tool: `Draw`, `Erase`,
  `Clear`, `Change` (color).
- **RIGHT hand** — its index fingertip is the "pen."
- The drawn canvas is continuously fed to a CNN, which shows its best
  guess (`Recognized: <label> (xx%)`) in the HUD next to the mode and
  color indicator.

---

## Project structure

```
Air_Drawing/
├── .venv/                  # runtime environment: run air_draw.py here
├── train_venv/              # training environment: run training scripts here
├── dataset/                 # QuickDraw .npy files (see below)
├── encodings_draw/          # recorded LEFT-hand gesture encodings (Draw/Erase/Clear/Change)
├── models/                  # training output (draw_classifier.keras/.tflite, draw_labels.json)
├── air_draw.py               # main app — run this
├── collect_draw_signs.py     # records the LEFT-hand gestures into encodings_draw/
├── draw_signrec.py           # left/right hand tracking + gesture matching (MediaPipe)
├── simple_signrec.py         # base landmark-encoding / nearest-neighbour matching
├── draw_classifier.py        # loads the trained TFLite model, used by air_draw.py
├── train_draw_classifier.py  # trains the CNN on dataset/ → models/
└── export_tflite.py          # re-exports an existing .keras model to .tflite (no retraining)
```

### `dataset/`
Download category files from Google's QuickDraw dataset (numpy bitmap
format) here:
https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/numpy_bitmap

Each downloaded file is named `full_numpy_bitmap_<category>.npy` and
contains thousands of flattened 28×28 drawings for that category — drop
them straight into `dataset/` as-is, no need to rename or sort into
subfolders. The label used for training is the `<category>` part of the
filename.

---

## Why two virtual environments?

`tensorflow` (needed for training) and `mediapipe` (needed for hand
tracking) require conflicting versions of `protobuf`. Rather than fight
that conflict, training and running are split into two separate venvs:

- **`train_venv`** — has full TensorFlow. Only used to run
  `train_draw_classifier.py` / `export_tflite.py`.
- **`.venv`** — has MediaPipe + a lightweight TFLite interpreter
  (no TensorFlow at all). Used to run `air_draw.py` day-to-day.

They never need to be active at the same time.

### `train_venv` — libraries

```
pip install tensorflow numpy opencv-python scikit-learn
```

If you hit a `protobuf`-related import error in this venv, force an
upgrade (this venv has no MediaPipe to conflict with):
```
pip install -U protobuf
```

### `.venv` — libraries

```
pip install mediapipe==0.10.14 protobuf==4.25.3 ai-edge-litert numpy opencv-python
```

`ai-edge-litert` is the actively-maintained TFLite runtime — it runs the
exported model without needing TensorFlow or a conflicting `protobuf`.

---

## How to train the model (summary)

1. Download some QuickDraw category files into `dataset/` (see above).
2. Activate `train_venv`.
3. Run:
   ```
   python train_draw_classifier.py
   ```
   This will:
   - Load every `.npy` file in `dataset/`, using the filename as the
     label (e.g. `full_numpy_bitmap_cake.npy` → label `cake`).
   - Split each file's stacked drawings into individual 28×28 samples
     (capped at `MAX_SAMPLES_PER_CLASS`, default 1500, per label — raise
     this for more accuracy at the cost of longer training).
   - Train a small CNN (Conv/BatchNorm/Pool ×3 → GlobalAvgPool → Dense).
   - Save `models/draw_classifier.keras`, `models/draw_labels.json`, and
     `models/draw_classifier.tflite`.
4. If the `.tflite` export step ever fails on its own (rare, but can
   happen with some TensorFlow builds) without needing to retrain:
   ```
   python export_tflite.py
   ```
5. Copy/keep the `models/` folder where `.venv`'s `air_draw.py` can find
   it (same project folder).

---

## How to run the app

1. Make sure `encodings_draw/` exists — if not, activate `.venv` and run
   `collect_draw_signs.py` first to record your LEFT-hand gestures
   (Draw / Erase / Clear / Change).
2. Make sure `models/draw_classifier.tflite` and `models/draw_labels.json`
   exist (see training above).
3. Activate `.venv`.
4. Run:
   ```
   python air_draw.py
   ```

**Controls:**
- `ESC` / `q` — quit
- `c` — clear canvas manually
- `SPACE` — freeze/unfreeze the camera feed (releases the webcam while
  frozen so other programs can use it)

**HUD:** top bar shows current `Mode`, `Color`, and
`Recognized: <label> (xx%)` — the CNN's live guess at what's been drawn.
Recognition is throttled (~once every 0.75s) rather than running every
frame, and only kicks in once there's enough ink on the canvas.

---

## Notes / troubleshooting

- If `Recognized:` never changes from `-`, check the terminal — with
  `debug=True` (the default in `draw_classifier.py`), it prints the raw
  top-3 guesses and their confidence every prediction, even when below
  the display threshold (default 55%, set via `CONFIDENCE_THRESHOLD` in
  `draw_classifier.py`).
- More samples per class and more distinct classes generally improve
  accuracy — `MAX_SAMPLES_PER_CLASS` in `train_draw_classifier.py` is
  the easiest first knob to raise.
