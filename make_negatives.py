"""Build a `background` negative class for birdnet-stm32 from ESC-50.

Reads the ESC-50 zip in place (no full extraction), keeps only clearly
non-bird environmental/urban categories, caps the total to stay balanced
against the ~128-per-species bird classes, and copies WAVs into
``data/train/background`` and ``data/test/background``.
"""

import csv
import io
import os
import random
import zipfile

# Paths are overridable via environment variables. Defaults target the Linux
# training container (prepare_data.sh sets these explicitly); they no longer
# point at any host folder.
ZIP = os.environ.get("ESC50_ZIP", "/inat/esc50.zip")
OUT_TRAIN = os.environ.get("OUT_TRAIN", "/data/train/background")
OUT_TEST = os.environ.get("OUT_TEST", "/data/test/background")

# Deployment-relevant, clearly non-bird ESC-50 categories for a yard/park
# bird detector. Animal vocalizations and bird-like sounds are excluded.
KEEP = {
    # nature / weather / water
    "rain",
    "wind",
    "thunderstorm",
    "water_drops",
    "pouring_water",
    "sea_waves",
    "crackling_fire",
    # distant urban / mechanical
    "engine",
    "train",
    "airplane",
    "car_horn",
    "siren",
    "church_bells",
    # human non-vocal
    "footsteps",
}

TOTAL_CAP = 350  # keep negatives ~comparable to a few bird classes combined
TEST_FRAC = 0.15
SEED = 42

os.makedirs(OUT_TRAIN, exist_ok=True)
os.makedirs(OUT_TEST, exist_ok=True)

z = zipfile.ZipFile(ZIP)
meta_name = next(n for n in z.namelist() if n.endswith("meta/esc50.csv"))
rows = list(csv.DictReader(io.TextIOWrapper(z.open(meta_name), encoding="utf-8")))

keep_rows = [r for r in rows if r["category"] in KEEP]
random.seed(SEED)
random.shuffle(keep_rows)
keep_rows = keep_rows[:TOTAL_CAP]

n_test = int(len(keep_rows) * TEST_FRAC)
test_rows = keep_rows[:n_test]
train_rows = keep_rows[n_test:]

audio_members = {os.path.basename(n): n for n in z.namelist() if n.endswith(".wav")}


def copy_rows(rs, outdir):
    n = 0
    for r in rs:
        member = audio_members.get(r["filename"])
        if not member:
            continue
        with open(os.path.join(outdir, r["filename"]), "wb") as f:
            f.write(z.read(member))
        n += 1
    return n


n_train = copy_rows(train_rows, OUT_TRAIN)
n_test_copied = copy_rows(test_rows, OUT_TEST)

cats = sorted({r["category"] for r in keep_rows})
print(f"categories kept ({len(cats)}): {cats}")
print(f"train background: {n_train}")
print(f"test  background: {n_test_copied}")
