# 🎨 Air Draw — Hand Gesture Drawing & AI Recognition Game

**Air Draw** is a computer vision and machine learning project that lets users draw in the air using their hands and a webcam. It combines MediaPipe hand tracking with a Convolutional Neural Network (CNN) trained to recognize drawings, turning a simple drawing application into an interactive AI-powered experience.

The project includes a drawing application, an enhanced drawing implementation, and a five-round recognition game that challenges users to draw randomly selected objects as quickly and accurately as possible.

LinkedIn post: https://lnkd.in/p/en7kubUj

## ✨ Key Features

### 🖐️ Air Drawing
- Draw on a virtual canvas using hand movements.
- Track hand landmarks in real time with MediaPipe.
- Use hand gestures to control drawing tools.
- Change colors, erase strokes, and clear the canvas.
- Freeze or resume the camera feed using the `SPACE` key in the supported implementation.

### 🧠 CNN Drawing Recognition
- Recognize sketches using a trained Convolutional Neural Network.
- Train the classifier on Google's Quick, Draw! dataset.
- Display predicted drawing labels and confidence scores.
- Export the trained model to TensorFlow Lite for lightweight inference.

### 🎮 AI Drawing Challenge
- Play a five-round drawing challenge.
- Receive a randomly selected drawing target from the trained classes.
- Draw the target in the air while the CNN evaluates your sketch.
- Complete each round when the target is recognized.
- View your overall completion time and recognition accuracy at the end.

## 🛠️ Technologies

| Technology | Purpose |
|---|---|
| Python | Application logic |
| OpenCV | Webcam access and image processing |
| MediaPipe | Real-time hand tracking |
| NumPy | Numerical computation and dataset processing |
| TensorFlow / Keras | CNN training |
| TensorFlow Lite / LiteRT | Lightweight model inference |

## 📂 Project Structure

```text
Air_Drawing/
├── models/
│   ├── draw_classifier.keras
│   ├── draw_classifier.tflite
│   └── draw_labels.json
├── dataset/                  # QuickDraw training data
├── encodings_draw/           # Recorded hand gesture encodings
├── air_draw.py               # Main air drawing application
├── new_air_draw.py           # Enhanced air drawing implementation
├── game.py                   # Five-round AI drawing challenge
├── collect_draw_signs.py     # Collect hand gesture encodings
├── draw_signrec.py           # Hand gesture recognition
├── simple_signrec.py         # Basic gesture matching
├── draw_classifier.py        # CNN classifier and predictions
├── train_draw_classifier.py  # Train the CNN model
├── export_tflite.py          # Export the trained model
└── .gitignore
```

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/ShadiZd/Air-draw-and-recognition.git
cd Air-draw-and-recognition
```

### 2. Set up the runtime environment

Use a compatible Python version and create a virtual environment:

```bash
python -m venv .venv
```

Activate it in Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the runtime dependencies:

```bash
pip install mediapipe==0.10.14 protobuf==4.25.3 ai-edge-litert numpy opencv-python
```

> The project separates runtime and training environments to help avoid dependency conflicts between MediaPipe and TensorFlow.

### 3. Prepare gesture encodings

The drawing application uses recorded hand gestures to control drawing tools. If the required encodings are missing, run:

```bash
python collect_draw_signs.py
```

Follow the instructions to record the required gestures.

### 4. Run the drawing application

For the standard drawing application:

```bash
python air_draw.py
```

For the enhanced implementation:

```bash
python new_air_draw.py
```

For the AI drawing challenge:

```bash
python game.py
```

Ensure that your webcam, gesture encodings, and required model files are available before launching the applications.

## 🧠 Training the CNN

Air Draw includes a CNN training pipeline based on Google's Quick, Draw! dataset.

### 1. Download the dataset

Get the NumPy bitmap files from the [Quick, Draw! dataset](https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/numpy_bitmap).

Place the selected files inside `dataset/`:

```text
dataset/
├── full_numpy_bitmap_apple.npy
├── full_numpy_bitmap_banana.npy
└── full_numpy_bitmap_cat.npy
```

Each filename determines its class label. For example, `full_numpy_bitmap_apple.npy` represents the `apple` class.

**Note:** The dataset files are not included in the repository by default because they can be very large.

### 2. Create the training environment

```bash
python -m venv train_venv
```

Activate it:

```powershell
.\train_venv\Scripts\Activate.ps1
```

Install the training dependencies:

```bash
pip install tensorflow numpy opencv-python scikit-learn
```

### 3. Train the model

```bash
python train_draw_classifier.py
```

The training script processes the available dataset classes, trains the CNN, and saves model artifacts in `models/`.

Expected outputs include:

- `draw_classifier.keras` — trained Keras model.
- `draw_labels.json` — class label information.
- `draw_classifier.tflite` — lightweight inference model.

To export an existing Keras model without retraining:

```bash
python export_tflite.py
```

## ⌨️ Controls

The available controls depend on the implementation you launch.

- `ESC` or `Q` — Quit the application.
- `C` — Clear the canvas.
- `SPACE` — Freeze or resume the camera feed in the supported implementation.

Hand gestures control the drawing tools. Follow the application's instructions for the available gestures and their mappings.

## 🔧 Troubleshooting

**The webcam does not open**
- Check that the camera is connected and available.
- Close other applications using the webcam.
- Verify that OpenCV and MediaPipe are installed correctly.

**Hand gestures are not recognized**
- Ensure that the required gesture encodings exist.
- Run `collect_draw_signs.py` to record them if necessary.
- Keep your hands visible and well positioned in the camera view.

**The CNN does not recognize drawings**
- Confirm that the model files exist in `models/`.
- Ensure that the class labels match the trained model.
- Verify that the classifier's expected input format matches the processed drawing.
- Retrain the model if the available classes or training data have changed.

## 🔮 Future Improvements

- Improve recognition accuracy and robustness.
- Add more drawing categories and gesture controls.
- Introduce difficulty levels and more game modes.
- Track high scores and completion times across sessions.
- Optimize the drawing and prediction pipeline for real-time performance.

## 🤝 Contributions

Suggestions, bug reports, and contributions are welcome!

1. Fork the repository.
2. Create a feature branch.
3. Implement and test your changes.
4. Submit a pull request describing your improvements.

## 📄 License

A license has not yet been specified. Add a `LICENSE` file if you intend to define how others may use, modify, and distribute the project.

---

**Developed by Shadi Zyoud**

*Combining computer vision, hand gestures, and machine learning to make drawing interactive.*
