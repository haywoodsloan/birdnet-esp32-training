"""Ingest device-recorded WAV negatives into the training `background` class.

The model defaulted to a bird (class 0) on this device's own room tone / mic
self-noise because its only negatives were ESC-50 clips that don't match the
device domain. This tool turns the on-device recordings (tools record mode ->
/sdcard/rec/devNNN.wav, copied to ./rec) into background training examples:

  1. Reads each 24 kHz mono device WAV.
  2. Applies the SAME preprocessing the firmware does at inference -- per-chunk
     DC removal + a 200 Hz 2nd-order Butterworth high-pass -- so the negatives
     match what the model actually sees on-device. (The training spectrogram
     pipeline does NOT high-pass, so we must bake it in here.)
  3. Slices each clip into overlapping CHUNK_S-second windows (HOP_S hop) to
     multiply ~11 min of audio into several hundred examples.
  4. Splits by SOURCE CLIP (all windows of one clip go to the same split) into
     train/test background, so no window leaks across the split.

Writes into the mounted data volume: /data/train/background + /data/test/background.
Existing background files (ESC-50) are kept; device clips are added alongside.
"""
import glob
import os
import random
import wave

import numpy as np
from scipy.signal import butter, lfilter

REC = os.environ.get("REC_DIR", "/rec")
TRAIN = os.environ.get("OUT_TRAIN", "/data/train/background")
TEST = os.environ.get("OUT_TEST", "/data/test/background")

SR = 24000
CHUNK_S = 3.0          # model analysis window
HOP_S = 1.5            # 50% overlap -> ~5-6 windows per 10 s clip
HPF_HZ = 200           # must match firmware BIRDNET_HPF_HZ
TEST_FRAC = 0.15
SEED = 42
MIN_RMS = 0.0008       # drop dead-silent windows (uninformative)

random.seed(SEED)


def read_wav(path):
    with wave.open(path, "rb") as w:
        assert w.getnchannels() == 1 and w.getsampwidth() == 2, path
        sr = w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return sr, x.astype(np.float32) / 32768.0


def preprocess(x):
    """DC removal + 200 Hz Butterworth high-pass (mirror of the firmware)."""
    x = x - np.mean(x)
    b, a = butter(2, HPF_HZ / (SR / 2.0), btype="highpass")
    return lfilter(b, a, x).astype(np.float32)


def write_wav(path, x):
    xi = np.clip(np.rint(x * 32768.0), -32768, 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(xi.tobytes())


def main():
    os.makedirs(TRAIN, exist_ok=True)
    os.makedirs(TEST, exist_ok=True)

    clips = sorted(glob.glob(os.path.join(REC, "*.wav")))
    if not clips:
        raise SystemExit(f"no WAVs in {REC}")

    chunk_n = int(CHUNK_S * SR)
    hop_n = int(HOP_S * SR)

    # Decide the split per SOURCE clip up front (no window leakage).
    shuffled = clips[:]
    random.shuffle(shuffled)
    n_test = int(len(shuffled) * TEST_FRAC)
    test_clips = set(shuffled[:n_test])

    n_train = n_test_w = n_skip = 0
    for ci, path in enumerate(sorted(clips)):
        sr, x = read_wav(path)
        if sr != SR:
            print(f"  skip {os.path.basename(path)}: {sr} Hz != {SR}")
            continue
        x = preprocess(x)
        outdir = TEST if path in test_clips else TRAIN
        base = os.path.splitext(os.path.basename(path))[0]

        wi = 0
        for start in range(0, max(1, len(x) - chunk_n + 1), hop_n):
            win = x[start:start + chunk_n]
            if len(win) < chunk_n:
                win = np.pad(win, (0, chunk_n - len(win)))
            if float(np.sqrt(np.mean(win**2))) < MIN_RMS:
                n_skip += 1
                continue
            write_wav(os.path.join(outdir, f"{base}_w{wi:02d}.wav"), win)
            wi += 1
            if outdir is TRAIN:
                n_train += 1
            else:
                n_test_w += 1

    print(f"source clips: {len(clips)} ({n_test} -> test, {len(clips) - n_test} -> train)")
    print(f"windows written: train={n_train}  test={n_test_w}  skipped_silent={n_skip}")
    print(f"train/background total: {len(os.listdir(TRAIN))}")
    print(f"test/background total:  {len(os.listdir(TEST))}")


if __name__ == "__main__":
    main()
