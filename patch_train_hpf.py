#!/usr/bin/env python3
"""Patch the training spectrogram pipeline to match the device audio domain (A).

WHY (option A -- device-domain match): the ESP32 firmware transforms every audio
chunk before the STFT in two ways the training pipeline does NOT, creating a
systematic train->field gap:

  1. DC removal + a 2nd-order Butterworth high-pass at BIRDNET_HPF_HZ (200 Hz)
     -- see stft_frontend.c build_padded. Training never high-passes, so the
     student learned full-spectrum spectrograms (incl. <200 Hz) while the device
     feeds high-passed ones, which also shifts the global min-max normalization.
  2. The mic adds room tone + self-noise (an additive floor) that clean iNat/XC
     recordings lack, so distant/quiet birds look different in the field.

This patch inserts BOTH into ``get_spectrogram_from_audio`` (the single point
where raw audio becomes a spectrogram), so training spectrograms look like what
the device produces:

  - ``_mix_device_noise`` : with prob BIRDNET_TRAIN_NOISE_PROB, add a random
      segment of a recorded device clip (BIRDNET_TRAIN_NOISE_DIR) at a random SNR
      in [5, 25] dB. Mirrors the mic floor.
  - ``_train_highpass``   : subtract the mean + 2nd-order Butterworth high-pass
      at BIRDNET_TRAIN_HPF_HZ. Mirrors firmware build_padded and
      sort_device_negs.preprocess exactly.

Both are env-gated and DEFAULT OFF (HPF "0", noise prob "0"), so the patched file
is a no-op for any run that doesn't opt in; only run10 sets the envs. Noise is
mixed BEFORE the high-pass, matching the device (mic captures bird + floor
together, then high-passes the sum). The noise pool is lazily loaded once per
worker process and cached.

Applied at container start (idempotent), the same way patch_distill.py patches
the generator. Run inside the training container:
    python /tmp/patch_train_hpf.py
then launch training with e.g. BIRDNET_TRAIN_HPF_HZ=200 BIRDNET_TRAIN_NOISE_PROB=0.3
BIRDNET_TRAIN_NOISE_DIR=/rec.
"""
from __future__ import annotations

import ast
import io
import sys
import tokenize

TARGET = "/workspace/birdnet-stm32/birdnet_stm32/audio/spectrogram.py"

IMPORT_ANCHOR = "import librosa\nimport numpy as np\n"
IMPORT_BLOCK = (
    "import librosa\n"
    "import numpy as np\n"
    "import os as _os\n"
    "import glob as _glob\n"
    "import wave as _wave\n"
    "from scipy.signal import butter as _butter, lfilter as _lfilter\n"
    "\n"
    "_TRAIN_HPF_CACHE = {}\n"
    "_TRAIN_NOISE_POOL = None  # lazily loaded once per worker process\n"
    "\n"
    "\n"
    "def _load_noise_pool(sample_rate):\n"
    "    \"\"\"Load device-noise WAVs from BIRDNET_TRAIN_NOISE_DIR as mono float32\n"
    "    arrays at sample_rate. Cached per process. Empty list if unset/missing.\"\"\"\n"
    "    global _TRAIN_NOISE_POOL\n"
    "    if _TRAIN_NOISE_POOL is not None:\n"
    "        return _TRAIN_NOISE_POOL\n"
    "    d = _os.environ.get('BIRDNET_TRAIN_NOISE_DIR', '')\n"
    "    pool = []\n"
    "    if d and _os.path.isdir(d):\n"
    "        for p in sorted(_glob.glob(_os.path.join(d, '*.wav'))):\n"
    "            try:\n"
    "                with _wave.open(p, 'rb') as w:\n"
    "                    if w.getnchannels() != 1 or w.getsampwidth() != 2:\n"
    "                        continue\n"
    "                    sr = w.getframerate()\n"
    "                    x = (np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)\n"
    "                         .astype(np.float32) / 32768.0)\n"
    "            except Exception:\n"
    "                continue\n"
    "            if sr != sample_rate and x.size:\n"
    "                new_len = int(round(x.size * sample_rate / sr))\n"
    "                if new_len > 0:\n"
    "                    x = np.interp(np.linspace(0, x.size - 1, new_len),\n"
    "                                  np.arange(x.size), x).astype(np.float32)\n"
    "            if x.size:\n"
    "                pool.append(x)\n"
    "    _TRAIN_NOISE_POOL = pool\n"
    "    return pool\n"
    "\n"
    "\n"
    "def _mix_device_noise(audio, sample_rate):\n"
    "    \"\"\"With prob BIRDNET_TRAIN_NOISE_PROB, add a random device-noise segment\n"
    "    at a random SNR in [5, 25] dB. No-op unless prob>0 and a pool exists.\"\"\"\n"
    "    try:\n"
    "        prob = float(_os.environ.get('BIRDNET_TRAIN_NOISE_PROB', '0') or '0')\n"
    "    except ValueError:\n"
    "        prob = 0.0\n"
    "    if prob <= 0.0 or audio is None or getattr(audio, 'size', 0) == 0:\n"
    "        return audio\n"
    "    if np.random.rand() > prob:\n"
    "        return audio\n"
    "    pool = _load_noise_pool(sample_rate)\n"
    "    if not pool:\n"
    "        return audio\n"
    "    noise = pool[np.random.randint(len(pool))]\n"
    "    if noise.size < audio.size:\n"
    "        noise = np.tile(noise, int(np.ceil(audio.size / noise.size)))\n"
    "    start = np.random.randint(0, max(1, noise.size - audio.size + 1))\n"
    "    seg = noise[start:start + audio.size]\n"
    "    a_rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2))) + 1e-9\n"
    "    n_rms = float(np.sqrt(np.mean(seg.astype(np.float64) ** 2))) + 1e-9\n"
    "    snr_db = np.random.uniform(5.0, 25.0)\n"
    "    target = a_rms / (10.0 ** (snr_db / 20.0))\n"
    "    return (audio + seg * (target / n_rms)).astype(np.float32)\n"
    "\n"
    "\n"
    "def _train_highpass(audio, sample_rate):\n"
    "    \"\"\"DC-remove + 2nd-order Butterworth high-pass, matching the device.\n"
    "    No-op unless BIRDNET_TRAIN_HPF_HZ > 0. Mirrors firmware build_padded and\n"
    "    sort_device_negs.preprocess so training audio looks like field audio.\"\"\"\n"
    "    try:\n"
    "        hz = float(_os.environ.get('BIRDNET_TRAIN_HPF_HZ', '0') or '0')\n"
    "    except ValueError:\n"
    "        hz = 0.0\n"
    "    if hz <= 0.0 or audio is None or getattr(audio, 'size', 0) == 0:\n"
    "        return audio\n"
    "    audio = audio - np.mean(audio)\n"
    "    key = (round(hz, 3), int(sample_rate))\n"
    "    ba = _TRAIN_HPF_CACHE.get(key)\n"
    "    if ba is None:\n"
    "        wn = min(0.99, max(1e-4, hz / (sample_rate / 2.0)))\n"
    "        ba = _butter(2, wn, btype='high')\n"
    "        _TRAIN_HPF_CACHE[key] = ba\n"
    "    b, a = ba\n"
    "    return _lfilter(b, a, audio).astype(np.float32)\n"
    "\n"
    "\n"
    "def _train_device_match(audio, sample_rate):\n"
    "    \"\"\"Mix device noise (before) then high-pass (after), matching the mic\n"
    "    chain: capture bird + floor together, then DC+HPF the sum.\"\"\"\n"
    "    audio = _mix_device_noise(audio, sample_rate)\n"
    "    return _train_highpass(audio, sample_rate)\n"
)

# Insert the HPF call as the first statement in get_spectrogram_from_audio,
# right before hop_length is computed (audio length is unchanged by the filter).
BODY_ANCHOR = (
    "    hop_length = (len(audio) // spec_width) if spec_width > 0 else n_fft // 2\n"
)
BODY_BLOCK = (
    "    audio = _train_device_match(audio, sample_rate)\n"
    "    hop_length = (len(audio) // spec_width) if spec_width > 0 else n_fft // 2\n"
)


def main() -> int:
    with open(TARGET, "r", encoding="utf-8") as f:
        src = f.read()

    if "_train_device_match" in src:
        print("patch_train_hpf: already patched, skipping")
        return 0

    if IMPORT_ANCHOR not in src:
        print("patch_train_hpf: ERROR import anchor not found", file=sys.stderr)
        return 1
    if BODY_ANCHOR not in src:
        print("patch_train_hpf: ERROR body anchor not found", file=sys.stderr)
        return 1

    src = src.replace(IMPORT_ANCHOR, IMPORT_BLOCK, 1)
    src = src.replace(BODY_ANCHOR, BODY_BLOCK, 1)

    # Validate it still parses before writing back.
    ast.parse(src)
    # Also confirm it tokenizes cleanly (catches stray indentation).
    list(tokenize.generate_tokens(io.StringIO(src).readline))

    with open(TARGET, "w", encoding="utf-8") as f:
        f.write(src)
    print("patch_train_hpf: applied device-match hook (HPF + noise; envs default off)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
