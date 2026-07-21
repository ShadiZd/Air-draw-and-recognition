"""
export_tflite.py

Standalone helper: converts an already-trained models/draw_classifier.keras
into models/draw_classifier.tflite, without retraining. Useful if training
succeeded but the TFLite export step failed/crashed.

Run (inside train_venv, the environment that has full tensorflow):
    python export_tflite.py
"""

import os
from train_draw_classifier import MODEL_PATH, TFLITE_MODEL_PATH, export_model_for_tflite


def main():
    if not os.path.isfile(MODEL_PATH):
        print(f"'{MODEL_PATH}' not found. Run train_draw_classifier.py first.")
        return

    from tensorflow import keras
    print(f"Loading '{MODEL_PATH}' ...")
    model = keras.models.load_model(MODEL_PATH)

    print("Converting to TFLite ...")
    export_model_for_tflite(model, TFLITE_MODEL_PATH)
    print(f"Saved TFLite model to '{TFLITE_MODEL_PATH}'")


if __name__ == "__main__":
    main()
