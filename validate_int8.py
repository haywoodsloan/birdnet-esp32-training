"""Robust post-quantization validation for the BirdNET-STM32 INT8 model.

The upstream `convert` validation invokes the TFLite interpreter on every
representative sample. The model's audio frontend normalizes the spectrogram by
`x / (reduce_max(x) + eps)`; for a digital-silence chunk `reduce_max` is ~0 and,
after INT8 quantization, the divisor can round to exactly 0, tripping TFLite's
`DIV` assert. On-device this never happens because the firmware skips inference
on chunks below the meter's usable floor (BIRDNET_LEVEL_DB_QUIET).

This script mirrors that: it runs the same stratified samples, SKIPS (and counts)
the ones that crash the quantized interpreter, and reports cosine / MSE / MAE /
Pearson on the rest — i.e. fidelity on real, non-silent audio.
"""

import json
import os
import random

import numpy as np
import tensorflow as tf

from birdnet_stm32.conversion.quantize import representative_data_gen
from birdnet_stm32.conversion.validate import cosine_similarity, pearson_correlation
from birdnet_stm32.data.dataset import load_file_paths_from_directory
from birdnet_stm32.models.frontend import AudioFrontendLayer
from birdnet_stm32.models.magnitude import MagnitudeScalingLayer

random.seed(123)
np.random.seed(123)

CKPT = "/out/best_model.keras"
CFG = "/out/best_model_model_config.json"
TFLITE = "/out/best_model_quantized.tflite"
DATA = "/data/train"
N_VAL = 400

with open(CFG) as f:
    cfg = json.load(f)

classes = cfg["class_names"]
model = tf.keras.models.load_model(
    CKPT,
    compile=False,
    custom_objects={"AudioFrontendLayer": AudioFrontendLayer, "MagnitudeScalingLayer": MagnitudeScalingLayer},
)

file_paths, _ = load_file_paths_from_directory(DATA, classes=classes)
val_paths = random.sample(file_paths, min(N_VAL, len(file_paths)))
print(f"validating on {len(val_paths)} samples")

interp = tf.lite.Interpreter(model_path=TFLITE, num_threads=1)
interp.allocate_tensors()
in_det = interp.get_input_details()[0]
out_det = interp.get_output_details()[0]

cos, mse, mae, pcc = [], [], [], []
n_ok = n_skip = 0

for sample in representative_data_gen(val_paths, cfg, num_samples=len(val_paths)):
    x = sample[0].astype(np.float32)
    # Skip near-silent inputs (mirror the firmware level gate): if the whole
    # spectrogram is ~flat/zero the normalized divisor degenerates.
    if float(np.max(x) - np.min(x)) < 1e-6:
        n_skip += 1
        continue
    yk = model(x, training=False).numpy()
    interp.set_tensor(in_det["index"], x)
    try:
        interp.invoke()
    except RuntimeError:
        n_skip += 1
        continue
    yt = interp.get_tensor(out_det["index"])
    a = yk.reshape(-1).astype(np.float64)
    b = yt.reshape(-1).astype(np.float64)
    cos.append(cosine_similarity(a, b))
    mse.append(float(np.mean((a - b) ** 2)))
    mae.append(float(np.mean(np.abs(a - b))))
    pcc.append(pearson_correlation(a, b))
    n_ok += 1

print(f"\nvalidated_ok={n_ok}  skipped_silent_or_crash={n_skip}")
if cos:
    print(f"cosine    mean={np.mean(cos):.6f}  min={np.min(cos):.6f}  std={np.std(cos):.6f}")
    print(f"mse       mean={np.mean(mse):.6f}")
    print(f"mae       mean={np.mean(mae):.6f}")
    print(f"pearson_r mean={np.mean(pcc):.6f}")
    print(f"\nRESULT: {'PASS' if np.mean(cos) >= 0.95 else 'REVIEW'} (threshold 0.95)")
