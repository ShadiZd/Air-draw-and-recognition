"""
train_draw_classifier.py

Trains a CNN on the labelled .npy drawing dataset and saves:
    models/draw_classifier.keras
    models/draw_labels.json

Expected dataset layout:
    dataset/
        <label_1>/
            file1.npy
            file2.npy
            ...
        <label_2>/
            ...

Each .npy file can be EITHER:
    - a single 2D image array (H, W)                -> one sample
    - a single 3D image array (H, W, C)              -> one sample
    - a stacked array of many samples (N, H, W[, C])  -> many samples
This is auto-detected per file, so you don't need every file to be the
same shape going in (they will all be resized to a common size before
training).

Run:
    python train_draw_classifier.py

Requires (see requirements at bottom of file / README):
    tensorflow, numpy, opencv-python, scikit-learn
"""

import os
import re
import json
import numpy as np
import cv2

DATASET_DIR = "dataset"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "draw_classifier.keras")
TFLITE_MODEL_PATH = os.path.join(MODEL_DIR, "draw_classifier.tflite")
LABELS_PATH = os.path.join(MODEL_DIR, "draw_labels.json")

IMG_SIZE = 64          # every sample gets resized to IMG_SIZE x IMG_SIZE
EPOCHS = 30
BATCH_SIZE = 32
VAL_SPLIT = 0.15
MAX_SAMPLES_PER_CLASS = 1500  # cap per class; QuickDraw files can have 100k+ rows


def _to_image(arr, img_size=IMG_SIZE):
    """
    Normalize an arbitrary array (image or flat landmark vector) into a
    single-channel img_size x img_size float32 image in [0, 1].
    """
    arr = np.asarray(arr, dtype=np.float32)

    if arr.ndim == 1:
        # Flat vector (e.g. landmark encoding) -> render as a square
        # "heatmap-ish" image so a CNN can still be used on it.
        side = int(np.ceil(np.sqrt(arr.size)))
        padded = np.zeros(side * side, dtype=np.float32)
        padded[: arr.size] = arr
        img = padded.reshape(side, side)
    elif arr.ndim == 2:
        img = arr
    elif arr.ndim == 3:
        # (H, W, C) -> collapse to grayscale
        img = arr.mean(axis=2)
    else:
        raise ValueError(f"Unsupported array shape for a single sample: {arr.shape}")

    # normalize to 0..1
    mn, mx = float(img.min()), float(img.max())
    if mx - mn > 1e-8:
        img = (img - mn) / (mx - mn)
    else:
        img = np.zeros_like(img)

    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
    return img.astype(np.float32)


def _load_one_file(path, max_samples=MAX_SAMPLES_PER_CLASS):
    """
    Returns a list of samples (each a raw array, before resizing) found in
    a single .npy file. Handles:
      - a single flat vector (1D)                          -> 1 sample
      - a single 2D image (H, W)                            -> 1 sample
      - a QuickDraw-style (N, D) array where D is a perfect
        square (e.g. 784 = 28*28) and N is large            -> N samples,
        one flattened image per row (subsampled to max_samples)
      - a stacked (N, H, W) or (N, H, W, C) array            -> N samples
      - a single (H, W, C) image                             -> 1 sample
    """
    data = np.load(path, allow_pickle=True)
    data = np.asarray(data)

    if data.ndim == 1:
        return [data]

    if data.ndim == 2:
        n, d = data.shape
        root = int(round(np.sqrt(d)))
        if root * root == d and n > 1:
            # QuickDraw-style: many flattened images stacked as rows.
            if n > max_samples:
                idx = np.random.choice(n, size=max_samples, replace=False)
                idx.sort()
            else:
                idx = np.arange(n)
            return [data[i] for i in idx]
        # Otherwise treat the whole 2D array as a single image.
        return [data]

    if data.ndim == 3:
        # (N, H, W) stacked samples, unless it's actually a single (H, W, C)
        # image with a small channel count and a small first dimension.
        if data.shape[-1] <= 4 and data.shape[0] <= 8:
            return [data]
        n = data.shape[0]
        if n > max_samples:
            idx = np.random.choice(n, size=max_samples, replace=False)
            idx.sort()
        else:
            idx = np.arange(n)
        return [data[i] for i in idx]

    if data.ndim == 4:
        n = data.shape[0]
        if n > max_samples:
            idx = np.random.choice(n, size=max_samples, replace=False)
            idx.sort()
        else:
            idx = np.arange(n)
        return [data[i] for i in idx]

    raise ValueError(f"Unsupported .npy shape {data.shape} in {path}")


_FNAME_RE = re.compile(r"^(?P<label>.+)_(?P<index>\d+)\.npy$", re.IGNORECASE)


def _label_from_filename(fname):
    """
    Parses '<label>_<index>.npy' -> '<label>'. Falls back to the filename
    without extension (minus a trailing _<number> if present) if the
    strict pattern doesn't match. Also strips the common QuickDraw
    'full_numpy_bitmap_' prefix so labels come out as e.g. 'cake' instead
    of 'full_numpy_bitmap_cake'.
    """
    m = _FNAME_RE.match(fname)
    if m:
        label = m.group("label")
    else:
        stem = os.path.splitext(fname)[0]
        label = re.sub(r"_\d+$", "", stem)

    prefix = "full_numpy_bitmap_"
    if label.startswith(prefix):
        label = label[len(prefix):]
    return label


def load_dataset(dataset_dir=DATASET_DIR, img_size=IMG_SIZE):
    """
    Loads a FLAT dataset directory of files named '<label>_<index>.npy',
    e.g. 'dataset/circle_0.npy', 'dataset/circle_1.npy', 'dataset/square_0.npy'.
    (Subfolders per label are also supported as a bonus: if a subfolder
    is found, every .npy inside it is loaded under that subfolder's name
    as the label, regardless of the file's own name.)
    """
    X, y = [], []
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"Dataset directory '{dataset_dir}' not found.")

    entries = sorted(os.listdir(dataset_dir))
    subdirs = [d for d in entries if os.path.isdir(os.path.join(dataset_dir, d))]
    flat_files = [f for f in entries if f.lower().endswith(".npy")]

    per_label_count = {}

    def _add(fpath, label):
        try:
            for raw_sample in _load_one_file(fpath, max_samples=MAX_SAMPLES_PER_CLASS):
                img = _to_image(raw_sample, img_size)
                X.append(img)
                y.append(label)
                per_label_count[label] = per_label_count.get(label, 0) + 1
        except Exception as e:
            print(f"  [skipping] {fpath}: {e}")

    if subdirs:
        for label in subdirs:
            folder = os.path.join(dataset_dir, label)
            for fname in os.listdir(folder):
                if fname.lower().endswith(".npy"):
                    _add(os.path.join(folder, fname), label)

    if flat_files:
        for fname in flat_files:
            label = _label_from_filename(fname)
            _add(os.path.join(dataset_dir, fname), label)

    if not subdirs and not flat_files:
        raise RuntimeError(
            f"No '.npy' files or label subfolders found inside '{dataset_dir}'."
        )

    if not X:
        raise RuntimeError("No samples were loaded. Check dataset folder contents.")

    for label, count in sorted(per_label_count.items()):
        print(f"Loaded {count} samples for label '{label}'")

    X = np.stack(X, axis=0)[..., np.newaxis]  # (N, H, W, 1)
    labels = sorted(per_label_count.keys())
    label_to_idx = {lbl: i for i, lbl in enumerate(labels)}
    y_idx = np.array([label_to_idx[lbl] for lbl in y], dtype=np.int64)

    return X, y_idx, labels


def export_model_for_tflite(model, tflite_path):
    """
    Exports a Keras model to TFLite via an intermediate SavedModel
    directory, which is the more crash-resistant path through TF's
    converter compared to converting straight from the in-memory model.
    """
    import shutil
    import tempfile
    from tensorflow import lite

    tmp_dir = tempfile.mkdtemp(prefix="draw_savedmodel_")
    try:
        model.export(tmp_dir)  # writes a SavedModel to tmp_dir
        converter = lite.TFLiteConverter.from_saved_model(tmp_dir)
        tflite_model = converter.convert()
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def build_model(input_shape, num_classes):
    from tensorflow import keras
    from tensorflow.keras import layers

    model = keras.Sequential([
        keras.Input(shape=input_shape),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),

        layers.Dense(128, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    from tensorflow import keras
    from sklearn.model_selection import train_test_split

    print(f"Loading dataset from '{DATASET_DIR}' ...")
    X, y, labels = load_dataset()
    print(f"Total samples: {len(X)}  |  classes: {labels}")

    if len(labels) < 2:
        raise RuntimeError("Need at least 2 classes to train a classifier.")

    stratify = y if len(X) >= len(labels) * 2 else None
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=VAL_SPLIT, random_state=42, stratify=stratify
    )

    model = build_model(X_train.shape[1:], len(labels))
    model.summary()

    os.makedirs(MODEL_DIR, exist_ok=True)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=6, restore_best_weights=True
        ),
        keras.callbacks.ModelCheckpoint(
            MODEL_PATH, monitor="val_accuracy", save_best_only=True
        ),
    ]

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=2,
    )

    model.save(MODEL_PATH)
    with open(LABELS_PATH, "w") as f:
        json.dump(labels, f, indent=2)

    print(f"\nSaved model to '{MODEL_PATH}'")
    print(f"Saved labels to '{LABELS_PATH}'")

    # Also export a TFLite version. This is what draw_classifier.py uses
    # at inference time, since it lets the air_draw.py runtime environment
    # avoid installing full tensorflow (which conflicts with mediapipe's
    # protobuf requirement) and just use a lightweight tflite runtime.
    #
    # Converting from a SavedModel directory (rather than directly from
    # the in-memory Keras object) is the more robust path through TF's
    # converter and avoids a known class of silent/native crashes with
    # from_keras_model() on some TensorFlow builds.
    try:
        export_model_for_tflite(model, TFLITE_MODEL_PATH)
        print(f"Saved TFLite model to '{TFLITE_MODEL_PATH}'")
    except Exception:
        import traceback
        print("\n[WARNING] TFLite export failed. The .keras model above is "
              "still valid and usable; only the TFLite export failed.")
        print("Full error below:\n")
        traceback.print_exc()


if __name__ == "__main__":
    main()