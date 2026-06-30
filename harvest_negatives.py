#!/usr/bin/env python3
"""Harvest CLEAN device-domain background negatives from field recordings.

Runs the full BirdNET acoustic model over a folder of on-device WAV clips and
copies the ones BirdNET hears NO real (in-season) bird in into an output dir as
training negatives. Clips where BirdNET DOES hear a confident in-season species
are EXCLUDED (so we never teach the model that a real bird is "background") and
listed for review.

These outdoor clips are exactly the device-domain noise (engine / ambient /
wind) that the on-device 74-class model false-positives on, so they make ideal
additions to birdnet-stm32's `background` negative class -- drop them in ./rec
and sort_device_negs.py windows them into /data/train/background.

Mirrors make_soft_targets.py's direct-TFLite path (the high-level predict()
forks after TF threads and deadlocks in-container).

Usage (in the training container):
    pip install birdnet soundfile
    python harvest_negatives.py --clips /out/fieldall --out /rec --prefix field_ \
        --lat 39.7042 --lon -86.3994 --week 26 --neg-thresh 0.25
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil

import numpy as np

BN_SR = 48000
BN_WIN = 144000  # 3 s * 48 kHz


def common(label: str) -> str:
    return label.split("_", 1)[1] if "_" in label else label


def sci(label: str) -> str:
    return label.split("_", 1)[0].strip().lower()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", default="/out/fieldall")
    ap.add_argument("--out", default="/rec")
    ap.add_argument("--prefix", default="field_")
    ap.add_argument("--lat", type=float, default=39.7042)
    ap.add_argument("--lon", type=float, default=-86.3994)
    ap.add_argument("--week", type=int, default=26)
    ap.add_argument("--neg-thresh", type=float, default=0.25,
                    help="exclude a clip if its max IN-SEASON BirdNET conf >= this")
    ap.add_argument("--sf-thresh", type=float, default=0.03,
                    help="min eBird occurrence to count a species 'in season'")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--progress-every", type=int, default=100)
    args = ap.parse_args()

    import tensorflow as tf
    import soundfile as sf
    import birdnet
    from birdnet.utils.local_data import get_model_path

    teacher = birdnet.load("acoustic", "2.4", "tf")
    species = list(teacher.species_list)
    del teacher
    model_path = str(get_model_path("acoustic", "2.4", "tf", "fp32"))

    geo = birdnet.load("geo", "2.4", "tf")
    gr = geo.predict(args.lat, args.lon, week=args.week, min_confidence=0.0)
    occ = {s: float(p) for s, p in zip(gr.species_list, gr.species_probs)}
    del geo
    allowed = np.array([occ.get(s, 0.0) >= args.sf_thresh for s in species], dtype=bool)
    print(f"in-season filter: {int(allowed.sum())}/{len(species)} species "
          f"(occ>={args.sf_thresh}) at lat={args.lat} lon={args.lon} week={args.week}",
          flush=True)

    itp = tf.lite.Interpreter(model_path=model_path, num_threads=args.threads)
    itp.allocate_tensors()
    in_idx = itp.get_input_details()[0]["index"]
    out_idx = itp.get_output_details()[0]["index"]

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    def max_inseason(path):
        """(species, conf) of the loudest in-season BirdNET hit in the clip."""
        audio, sr = sf.read(path, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != BN_SR and audio.size:
            new_len = int(round(audio.size * BN_SR / sr))
            audio = np.interp(
                np.linspace(0, audio.size - 1, new_len, dtype=np.float64),
                np.arange(audio.size), audio).astype(np.float32)
        if audio.size == 0:
            return None, -1.0
        seg = audio[:BN_WIN]
        if seg.size < BN_WIN:
            seg = np.pad(seg, (0, BN_WIN - seg.size))
        itp.set_tensor(in_idx, seg.reshape(1, BN_WIN).astype(np.float32))
        itp.invoke()
        v = sigmoid(itp.get_tensor(out_idx)[0])
        v[~allowed] = 0.0
        i = int(np.argmax(v))
        return common(species[i]), float(v[i])

    os.makedirs(args.out, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(args.clips, "*.wav")))
    print(f"scanning {len(paths)} clips (neg-thresh {args.neg_thresh}) ...\n", flush=True)

    kept = excluded = 0
    excl = []
    for n, p in enumerate(paths):
        sp, c = max_inseason(p)
        if c is None or c < 0:
            continue
        if c >= args.neg_thresh:
            excluded += 1
            excl.append((os.path.basename(p), sp, c))
        else:
            shutil.copyfile(p, os.path.join(args.out, args.prefix + os.path.basename(p)))
            kept += 1
        if (n + 1) % args.progress_every == 0:
            print(f"  [{n+1}/{len(paths)}] kept {kept}, excluded {excluded}", flush=True)

    print(f"\nHARVEST DONE: {kept} clean negatives -> {args.out} (prefix '{args.prefix}')",
          flush=True)
    print(f"excluded {excluded} clips with a real in-season bird (>= {args.neg_thresh}):",
          flush=True)
    for name, sp, c in sorted(excl, key=lambda t: -t[2])[:50]:
        print(f"  {name}: {sp}={c:.2f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
