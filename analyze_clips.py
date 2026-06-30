#!/usr/bin/env python3
"""Cross-check on-device BirdNET detections against the REAL BirdNET teacher.

The ESP32 firmware runs a tiny distilled 74-class model. A field run outdoors
collapsed onto out-of-season winter species (Red-breasted Nuthatch). This script
runs the same audio clips through the full BirdNET acoustic model (6522 species)
to see what BirdNET itself hears -- the ground truth the student was distilled
from.

For each WAV: decode + resample to 48 kHz mono, split into 3 s windows, run
BirdNET's acoustic TFLite directly (sigmoid head), take the per-species MAX over
windows. We print:
  - BirdNET's global top-k species,
  - the top-k after a location+week occurrence filter (eBird meta-model),
  - the explicit confidence BirdNET gives the device's claimed attractors.

Bypasses birdnet's high-level predict() (it forks after TF threads and deadlocks
in-container); mirrors make_soft_targets.py's direct-TFLite path.

Usage (in the training container):
    pip install birdnet soundfile
    python analyze_clips.py --clips /out/fieldclips --lat 39.7042 --lon -86.3994 --week 26
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np

BN_SR = 48000
BN_WIN = 144000  # 3 s * 48 kHz

# What the on-device 74-class model claimed for each curated clip (from
# birdnet_log.csv), so the comparison is inline and self-documenting.
DEVICE = {
    "clip000000.wav": "Veery 0.09 (essentially noise)",
    "clip000011.wav": "Red-breasted Nuthatch 0.66",
    "clip000012.wav": "Red-breasted Nuthatch 0.61",
    "clip000024.wav": "Red-breasted Nuthatch 0.92",
    "clip000176.wav": "Red-breasted Nuthatch 0.92",
    "clip000208.wav": "Red-breasted Nuthatch 0.86 (loud -37 dB)",
    "clip000209.wav": "Red-breasted Nuthatch 0.89 (loud -39 dB)",
    "clip000299.wav": "Hairy Woodpecker 0.61",
    "clip000355.wav": "Red-breasted Nuthatch 0.92",
    "clip000447.wav": "Red-breasted Nuthatch 0.94",
    "clip000514.wav": "Great Horned Owl 0.53",
    "clip000516.wav": "Red-breasted Nuthatch 0.85",
    "clip000533.wav": "Red-breasted Nuthatch 0.87",
    "clip000591.wav": "Red-breasted Nuthatch 0.85",
    "clip000632.wav": "Red-breasted Nuthatch 0.93",
    "clip000639.wav": "Great Horned Owl 0.66",
    "clip000667.wav": "Veery 0.32 (loud -47 dB)",
}


def common(label: str) -> str:
    """'Genus species_Common Name' -> 'Common Name'."""
    return label.split("_", 1)[1] if "_" in label else label


def sci(label: str) -> str:
    return label.split("_", 1)[0].strip().lower()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", default="/out/fieldclips")
    ap.add_argument("--lat", type=float, default=39.7042)   # Plainfield IN
    ap.add_argument("--lon", type=float, default=-86.3994)
    ap.add_argument("--week", type=int, default=26)         # late June
    ap.add_argument("--topk", type=int, default=6)
    ap.add_argument("--sf-thresh", type=float, default=0.03,
                    help="min eBird occurrence to count a species 'in season'")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--max-windows", type=int, default=2)
    args = ap.parse_args()

    import tensorflow as tf
    import soundfile as sf
    import birdnet
    from birdnet.utils.local_data import get_model_path

    teacher = birdnet.load("acoustic", "2.4", "tf")
    species = list(teacher.species_list)          # 'Scientific_Common'
    del teacher
    model_path = str(get_model_path("acoustic", "2.4", "tf", "fp32"))
    print(f"BirdNET acoustic model: {len(species)} species; {model_path}", flush=True)

    # Location + week occurrence filter (same eBird meta-model the prior uses).
    occ = {}
    try:
        geo = birdnet.load("geo", "2.4", "tf")
        gr = geo.predict(args.lat, args.lon, week=args.week, min_confidence=0.0)
        occ = {s: float(p) for s, p in zip(gr.species_list, gr.species_probs)}
        del geo
    except Exception as e:  # noqa: BLE001
        print(f"WARN: geo filter unavailable ({e}); showing raw BirdNET only", flush=True)
    allowed = np.array([occ.get(s, 0.0) >= args.sf_thresh for s in species], dtype=bool)
    if occ:
        print(f"location/week filter: {int(allowed.sum())}/{len(species)} species "
              f"in season (occ>={args.sf_thresh}) at lat={args.lat} lon={args.lon} "
              f"week={args.week}\n", flush=True)

    itp = tf.lite.Interpreter(model_path=model_path, num_threads=args.threads)
    itp.allocate_tensors()
    in_idx = itp.get_input_details()[0]["index"]
    out_idx = itp.get_output_details()[0]["index"]

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    def idx_of(scientific: str) -> int:
        for i, s in enumerate(species):
            if sci(s) == scientific:
                return i
        return -1

    i_rbnu = idx_of("sitta canadensis")        # Red-breasted Nuthatch
    i_wtsp = idx_of("zonotrichia albicollis")  # White-throated Sparrow

    def scores_for(path: str):
        audio, sr = sf.read(path, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != BN_SR and audio.size:
            new_len = int(round(audio.size * BN_SR / sr))
            audio = np.interp(
                np.linspace(0, audio.size - 1, new_len, dtype=np.float64),
                np.arange(audio.size), audio).astype(np.float32)
        if audio.size == 0:
            return None
        n_win = max(1, min(args.max_windows, (audio.size + BN_WIN - 1) // BN_WIN))
        vec = None
        for w in range(n_win):
            seg = audio[w * BN_WIN:(w + 1) * BN_WIN]
            if seg.size < BN_WIN:
                seg = np.pad(seg, (0, BN_WIN - seg.size))
            itp.set_tensor(in_idx, seg.reshape(1, BN_WIN).astype(np.float32))
            itp.invoke()
            p = sigmoid(itp.get_tensor(out_idx)[0])
            vec = p if vec is None else np.maximum(vec, p)
        return vec

    def fmt(idxs, v) -> str:
        return ", ".join(f"{common(species[i])}={v[i]:.2f}" for i in idxs if v[i] > 0)

    paths = sorted(glob.glob(os.path.join(args.clips, "*.wav")))
    print(f"analyzing {len(paths)} clips\n" + "=" * 72, flush=True)
    for p in paths:
        name = os.path.basename(p)
        v = scores_for(p)
        print(f"\n{name}   [device said: {DEVICE.get(name, '?')}]", flush=True)
        if v is None:
            print("  (empty audio)", flush=True)
            continue
        top = np.argsort(v)[::-1][:args.topk]
        print(f"  BirdNET top:   {fmt(top, v)}", flush=True)
        if occ:
            vf = v.copy()
            vf[~allowed] = 0.0
            topf = np.argsort(vf)[::-1][:args.topk]
            shown = fmt(topf, vf)
            print(f"  in-season top: {shown if shown else '(nothing in season above noise)'}",
                  flush=True)
        if i_rbnu >= 0:
            print(f"  -> Red-breasted Nuthatch BirdNET={v[i_rbnu]:.2f} "
                  f"(eBird occ {occ.get(species[i_rbnu], 0.0):.3f})", flush=True)
        if i_wtsp >= 0:
            print(f"  -> White-throated Sparrow BirdNET={v[i_wtsp]:.2f} "
                  f"(eBird occ {occ.get(species[i_wtsp], 0.0):.3f})", flush=True)
    print("\n" + "=" * 72, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
