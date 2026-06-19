#!/usr/bin/env python3
"""Generate BirdNET soft targets for knowledge distillation -- DIRECT TFLITE.

BirdNET's high-level predict() uses a multiprocessing producer/consumer pipeline
that DEADLOCKS in our container (fork-after-TF-threads). We bypass it entirely:
load BirdNET's underlying TFLite model and run inference ourselves, one process,
no forking.

For each training clip we:
  - decode + resample to 48 kHz mono (BirdNET's rate),
  - split into non-overlapping 3 s windows (144000 samples, pad the last),
  - run each window through the TFLite model -> 6522 logits -> sigmoid,
  - take the MAX confidence per species across windows,
  - keep only our 74 target species (mapped by "Scientific_Common" prefix),
then BLEND with the iNat hard label:
    target = clip( (1-alpha)*onehot + alpha*birdnet_probs , 0, 1 )   (true class floored)

WHY: iNatSounds clips are weakly labeled -- a clip tagged species X often
contains a louder non-target bird, so the small student learns the loud/dominant
species and quiet species (dove, goldfinch) collapse to 0%. BirdNET hears the
target even when it isn't loudest, so its soft targets restore probability mass
on the correct species. Cross-entropy against these soft targets transfers that
knowledge with no change to the loss or model -- only the per-file label the
data loader emits (see patch_distill.py / BIRDNET_SOFT_TARGETS).

Output: <out>/soft_targets.npz {paths[str], targets[f32,N,74], classes[str,74]}.
The student keeps a sigmoid head, so independent per-class confidences + BCE.

Usage (in the training container):
    pip install birdnet            # provides the model download + species list
    python make_soft_targets.py --data /data/train --out /data --alpha 0.5 --threads 16
"""
from __future__ import annotations

import argparse
import glob
import os
import time

import numpy as np

BN_SR = 48000
BN_WIN = 144000  # 3 s * 48 kHz


def load_class_order(data_dir: str) -> list[str]:
    """Training class order = sorted Genus_species dir names, excluding noise."""
    noise = {"noise", "silence", "background", "other"}
    dirs = [d for d in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, d)) and d.lower() not in noise]
    return sorted(dirs)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="/data/train")
    ap.add_argument("--out", default="/data")
    ap.add_argument("--alpha", type=float, default=0.5, help="BirdNET weight in blend")
    ap.add_argument("--min-teacher", type=float, default=0.05,
                    help="if BirdNET max target conf < this, fall back to pure one-hot")
    ap.add_argument("--threads", type=int, default=16, help="TFLite interpreter threads")
    ap.add_argument("--max-windows", type=int, default=10, help="cap windows scored per file")
    ap.add_argument("--progress-every", type=int, default=500)
    args = ap.parse_args()

    import tensorflow as tf
    import soundfile as sf
    import birdnet
    from birdnet.utils.local_data import get_model_path

    # Species list (order matters) + ensure the model is downloaded.
    teacher = birdnet.load("acoustic", "2.4", "tf")
    bn_species = list(teacher.species_list)            # "Scientific_Common"
    del teacher                                        # use the raw tflite below
    model_path = str(get_model_path("acoustic", "2.4", "tf", "fp32"))

    classes = load_class_order(args.data)
    n = len(classes)
    cls_to_idx = {c: i for i, c in enumerate(classes)}

    # Map BirdNET output columns -> our class index (by scientific name).
    sci_to_cls = {c.replace("_", " ").lower(): c for c in classes}
    col_to_clsidx: dict[int, int] = {}
    matched = set()
    for col, label in enumerate(bn_species):
        sci = label.split("_", 1)[0].strip().lower()
        c = sci_to_cls.get(sci)
        if c is not None:
            col_to_clsidx[col] = cls_to_idx[c]
            matched.add(c)
    print(f"{n} classes; matched {len(matched)}/{n} to BirdNET columns; model={model_path}",
          flush=True)
    bn_cols = np.array(sorted(col_to_clsidx.keys()), dtype=np.int64)
    bn_col_cls = np.array([col_to_clsidx[c] for c in bn_cols], dtype=np.int64)

    itp = tf.lite.Interpreter(model_path=model_path, num_threads=args.threads)
    itp.allocate_tensors()
    in_idx = itp.get_input_details()[0]["index"]
    out_idx = itp.get_output_details()[0]["index"]

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    def teacher_vec(path: str) -> np.ndarray:
        """Max BirdNET confidence per target class over the file's 3 s windows."""
        vec = np.zeros(n, dtype=np.float32)
        try:
            audio, sr = sf.read(path, dtype="float32", always_2d=False)
        except Exception:
            return vec
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != BN_SR and audio.size:
            new_len = int(round(audio.size * BN_SR / sr))
            if new_len > 0:
                audio = np.interp(
                    np.linspace(0, audio.size - 1, new_len, dtype=np.float64),
                    np.arange(audio.size), audio).astype(np.float32)
        if audio.size == 0:
            return vec
        n_win = max(1, min(args.max_windows, (audio.size + BN_WIN - 1) // BN_WIN))
        for w in range(n_win):
            seg = audio[w * BN_WIN:(w + 1) * BN_WIN]
            if seg.size < BN_WIN:
                seg = np.pad(seg, (0, BN_WIN - seg.size))
            itp.set_tensor(in_idx, seg.reshape(1, BN_WIN).astype(np.float32))
            itp.invoke()
            logits = itp.get_tensor(out_idx)[0]
            probs = sigmoid(logits[bn_cols])
            # NOTE: fancy-indexed assignment with out= writes to a temp copy, not
            # vec. Use np.maximum.at for a correct in-place scatter-max.
            np.maximum.at(vec, bn_col_cls, probs)
        return vec

    paths = sorted(glob.glob(os.path.join(args.data, "*", "*.wav")))
    print(f"scoring {len(paths)} files with direct TFLite (threads={args.threads}) ...",
          flush=True)

    targets = np.zeros((len(paths), n), dtype=np.float32)
    # Raw BirdNET (teacher) max-confidence per target class per file, BEFORE the
    # one-hot blend/floor. Used by clean_labels.py (option B) to drop files where
    # the teacher does not actually hear the labeled species.
    raw = np.zeros((len(paths), n), dtype=np.float32)
    n_fallback = agree = scored = 0
    t0 = time.time()
    for i, p in enumerate(paths):
        label_str = os.path.basename(os.path.dirname(p))
        onehot = np.zeros(n, dtype=np.float32)
        if label_str in cls_to_idx:
            onehot[cls_to_idx[label_str]] = 1.0

        bn = teacher_vec(p)
        raw[i] = bn
        if bn.max() < args.min_teacher:
            targets[i] = onehot
            n_fallback += 1
        else:
            scored += 1
            if label_str in cls_to_idx and int(np.argmax(bn)) == cls_to_idx[label_str]:
                agree += 1
            t = np.clip((1.0 - args.alpha) * onehot + args.alpha * bn, 0.0, 1.0)
            if label_str in cls_to_idx:
                j = cls_to_idx[label_str]
                t[j] = max(t[j], 1.0 - args.alpha)
            targets[i] = t

        if (i + 1) % args.progress_every == 0:
            rate = (i + 1) / (time.time() - t0)
            eta = (len(paths) - i - 1) / max(rate, 1e-6)
            print(f"  [{i+1}/{len(paths)}] {rate:.1f} files/s, eta {eta/60:.0f} min, "
                  f"fallback {n_fallback}", flush=True)

    out_path = os.path.join(args.out, "soft_targets.npz")
    np.savez_compressed(out_path, paths=np.array(paths), targets=targets,
                        classes=np.array(classes), raw_teacher=raw)
    print(f"\nsaved {out_path}: {len(paths)} files x {n} classes in "
          f"{(time.time()-t0)/60:.0f} min", flush=True)
    print(f"  fallback-to-onehot (BirdNET silent): {n_fallback}", flush=True)
    if scored:
        print(f"  BirdNET argmax == iNat label: {agree}/{scored} "
              f"({100.0*agree/scored:.1f}%)  <- low = weak labels confirmed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
