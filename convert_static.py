"""Re-convert best_model.keras to INT8 TFLite with a STATIC batch=1 input.

Why: the hybrid frontend builds channel-pad tensors with tf.zeros([B,1,T,pad])
where B = tf.shape(y)[0] (dynamic batch). With a dynamic batch that lowers to
SHAPE -> PACK -> FILL ops. TFLite-Micro's FILL kernel only supports a CONSTANT
dims tensor, so AllocateTensors() fails on-device ("Non-constant dims tensor is
not supported", "Node FILL failed to prepare").

Fix: rebuild the functional model with a FIXED batch dimension of 1, then run
the normal from_keras_model PTQ path. With a fully-static input shape the TFLite
converter constant-folds tf.shape(y)[0] into a constant, so tf.zeros(...) becomes
a constant tensor and the Shape/Pack/Fill chain is eliminated. (Wrapping the
model in a tf.function + from_concrete_functions instead leaves the weights as
resource variables -> READ_VARIABLE fails during calibration, so we avoid that.)

Mirrors birdnet_stm32 conversion/quantize.py PTQ settings exactly.
"""
import os
import random
from collections import defaultdict

import numpy as np
import tensorflow as tf

from birdnet_stm32.conversion.quantize import representative_data_gen
from birdnet_stm32.conversion.validate import validate_models
from birdnet_stm32.data.dataset import load_file_paths_from_directory
from birdnet_stm32.models.frontend import AudioFrontendLayer
from birdnet_stm32.models.magnitude import MagnitudeScalingLayer
from birdnet_stm32.training.config import ModelConfig

random.seed(42)
np.random.seed(42)

CKPT = "/out/best_model.keras"
CFGP = "/out/best_model_model_config.json"
OUTP = "/out/best_model_static.tflite"
DATA = "/data/train"
NUM_SAMPLES = 1024
VAL_SAMPLES = 256

cfg = ModelConfig.load(CFGP).to_dict()
model = tf.keras.models.load_model(
    CKPT, compile=False,
    custom_objects={"AudioFrontendLayer": AudioFrontendLayer, "MagnitudeScalingLayer": MagnitudeScalingLayer},
)
print("loaded", CKPT, "input_shape", model.input_shape)

fft_bins = int(cfg["fft_length"]) // 2 + 1
spec_width = int(cfg["spec_width"])

# Rebuild the SAME graph but with a fixed batch of 1, so every tf.shape()[0]
# becomes a compile-time constant and the FILL/SHAPE/PACK pad chain folds away.
fixed_in = tf.keras.Input(batch_shape=(1, fft_bins, spec_width, 1), name="input")
fixed_out = model(fixed_in, training=False)
fixed_model = tf.keras.Model(fixed_in, fixed_out, name="fixed_batch")
print("fixed model input_shape", fixed_model.input_shape)

# Stratified representative dataset (same approach as cli/convert.py).
file_paths, _classes = load_file_paths_from_directory(DATA)
class_files = defaultdict(list)
for p in file_paths:
    class_files[os.path.basename(os.path.dirname(p))].append(p)
per_class = max(1, NUM_SAMPLES // max(len(class_files), 1))
strat = []
for _cls, paths in class_files.items():
    strat.extend(random.sample(paths, min(per_class, len(paths))))
random.shuffle(strat)
strat = strat[:NUM_SAMPLES]
print(f"rep dataset: {len(strat)} stratified samples from {len(class_files)} classes")

val_subset = random.sample(file_paths, min(VAL_SAMPLES, len(file_paths)))


def rep_data_gen():
    return representative_data_gen(strat, cfg, num_samples=len(strat))


def rep_data_gen_val():
    return representative_data_gen(val_subset, cfg, num_samples=len(val_subset))


converter = tf.lite.TFLiteConverter.from_keras_model(fixed_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.inference_input_type = tf.float32
converter.inference_output_type = tf.float32
converter.representative_dataset = rep_data_gen
converter._experimental_new_quantizer = True
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

tflite_model = converter.convert()

# Verify float32 I/O (same guard as the package).
interp = tf.lite.Interpreter(model_content=tflite_model)
interp.allocate_tensors()
in_dt = interp.get_input_details()[0]["dtype"]
out_dt = interp.get_output_details()[0]["dtype"]
print("I/O dtypes:", in_dt, out_dt, "in shape", interp.get_input_details()[0]["shape"],
      "out shape", interp.get_output_details()[0]["shape"])
assert in_dt == np.float32 and out_dt == np.float32, "non-float32 I/O!"

os.makedirs(os.path.dirname(OUTP), exist_ok=True)
with open(OUTP, "wb") as f:
    f.write(tflite_model)
print("WROTE", OUTP, len(tflite_model), "bytes")

# List ops to confirm Fill/Shape/Pack are gone.
from tensorflow.lite.python import schema_py_generated as s  # noqa: E402

m2 = s.ModelT.InitFromObj(s.Model.GetRootAsModel(bytearray(tflite_model), 0))
names = {v: k for k, v in vars(s.BuiltinOperator).items() if isinstance(v, int)}
ops = sorted({names[max(o.builtinCode, o.deprecatedBuiltinCode)] for o in m2.operatorCodes})
print("OPS:", ops)
print("HAS_FILL:", "FILL" in ops, " HAS_SHAPE:", "SHAPE" in ops, " HAS_PACK:", "PACK" in ops)

# Fidelity check against the ORIGINAL (dynamic-batch) Keras model.
val = validate_models(model, OUTP, rep_data_gen_val)
print("cosine_mean:", val.get("cosine_mean"), " mae_mean:", val.get("mae_mean"))
