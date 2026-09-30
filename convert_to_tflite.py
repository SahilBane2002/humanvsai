"""Convert best_model.keras to best_model.tflite and check that both give the same answers.
Run once, inside your local virtual environment:  python convert_to_tflite.py"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"                     # hide TensorFlow's info messages

import csv
import json
import shutil

import numpy as np
import tensorflow as tf
import keras
from PIL import Image

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    Interpreter = tf.lite.Interpreter


def is_augmentation(layer):
    """True for a random augmentation layer (RandomFlip, RandomRotation, ...) or a block made only of them."""
    if layer.__class__.__name__.startswith("Random"):
        return True
    sub = getattr(layer, "layers", None)
    return bool(sub) and all(l.__class__.__name__.startswith("Random") for l in sub)


# ---- Test images, preprocessed exactly like the app does ----
cfg = json.load(open("model_config.json"))
rows = list(csv.DictReader(open("test_images/labels.csv")))
x = np.stack([np.array(Image.open(f"test_images/{r['filename']}").convert("RGB"), dtype=np.float32)
              for r in rows]) / 255.0 * cfg["input_scale"]
labels = np.array([int(r["label"]) for r in rows])

# ---- 1. Rebuild the model without its augmentation block ----
# Augmentation only matters during training, and LiteRT can't run its random-number operations.
# The rebuilt model reuses the same trained layers (same weights).
model = keras.models.load_model("best_model.keras")
inputs = keras.Input(shape=model.input_shape[1:])
h = inputs
kept, skipped = [], []
for layer in model.layers:
    if isinstance(layer, keras.layers.InputLayer):
        continue
    if is_augmentation(layer):
        skipped.append(layer.name)
        continue
    h = layer(h)
    kept.append(layer.name)
clean = keras.Model(inputs, h, name="inference_model")
print("Removed:", skipped)
print("Kept:   ", kept)

keras_probs = model.predict(x, verbose=0).ravel()
clean_probs = clean.predict(x, verbose=0).ravel()
print(f"Original vs. rebuilt model, largest difference: {np.max(np.abs(keras_probs - clean_probs)):.6f}   (should be 0)")

# ---- 2. Rebuilt model -> TensorFlow SavedModel -> LiteRT (.tflite) ----
shutil.rmtree("best_model_savedmodel", ignore_errors=True)   # clear the failed attempt
clean.export("best_model_savedmodel", verbose=False)
converter = tf.lite.TFLiteConverter.from_saved_model("best_model_savedmodel")
with open("best_model.tflite", "wb") as f:
    f.write(converter.convert())
print(f"Saved best_model.tflite ({os.path.getsize('best_model.tflite') / 1e6:.1f} MB)")

# ---- 3. Check: LiteRT must agree with the original Keras model ----
interp = Interpreter(model_path="best_model.tflite")
interp.allocate_tensors()
inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
lite_probs = []
for img in x:
    interp.set_tensor(inp["index"], img[None].astype(np.float32))
    interp.invoke()
    lite_probs.append(float(interp.get_tensor(out["index"]).ravel()[0]))
lite_probs = np.array(lite_probs)

print(f"Original vs. LiteRT, largest difference: {np.max(np.abs(keras_probs - lite_probs)):.6f}")
print(f"Keras accuracy:  {np.mean((keras_probs > 0.5) == labels):.3f}")
print(f"LiteRT accuracy: {np.mean((lite_probs > 0.5) == labels):.3f}")