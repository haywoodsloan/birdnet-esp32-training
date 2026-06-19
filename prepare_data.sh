#!/usr/bin/env bash
#
# Idempotent dataset preparation for the BirdNET-STM32 GPU container.
#
# Produces ${DATA_DIR}/{train,test}/<Scientific_name>/*.wav plus a `background`
# negative class, ready for `birdnet_stm32 train`.
#
# It is cheap to re-run: if the sorted dataset already exists it exits
# immediately. If raw iNatSounds recordings are already present (e.g. a host
# `inatsounds/extracted` mounted at ${INAT_DIR}), it skips the 81 GB download
# and only sorts. Otherwise it streams the tarball and extracts just the 74
# species directories listed in extract_dirs.txt (the full archive is never
# written to disk).
set -euo pipefail

DATA_DIR=${DATA_DIR:-/data}
INAT_DIR=${INAT_DIR:-/inat}
PREP=${PREP:-/workspace/prep}

INAT_TRAIN_URL=${INAT_TRAIN_URL:-https://ml-inat-competition-datasets.s3.amazonaws.com/sounds/2024/train.tar.gz}
INAT_JSON_URL=${INAT_JSON_URL:-https://ml-inat-competition-datasets.s3.amazonaws.com/sounds/2024/train.json.tar.gz}
ESC50_URL=${ESC50_URL:-https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip}

MAX_PER_SPECIES=${MAX_PER_SPECIES:-150}
TEST_FRAC=${TEST_FRAC:-0.15}

mkdir -p "${DATA_DIR}" "${INAT_DIR}"

# --- 0. Already prepared? -----------------------------------------------------
# Guard the count: under `set -euo pipefail` a `find` on a not-yet-created
# train dir exits non-zero and would abort the whole script, so only count
# when the directory exists (a fresh data volume has no train dir yet).
if [ -d "${DATA_DIR}/train" ]; then
    species_dirs=$(find "${DATA_DIR}/train" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
else
    species_dirs=0
fi
if [ "${species_dirs}" -ge 75 ]; then
    echo "[prep] dataset already prepared (${species_dirs} class dirs in ${DATA_DIR}/train) -> skip"
    exit 0
fi

# --- 1. Annotations (train.json) ---------------------------------------------
if [ ! -f "${INAT_DIR}/train.json" ]; then
    echo "[prep] downloading annotations -> ${INAT_DIR}/train.json"
    curl -fL "${INAT_JSON_URL}" -o "${INAT_DIR}/train.json.tar.gz"
    tar -xzf "${INAT_DIR}/train.json.tar.gz" -C "${INAT_DIR}"
    rm -f "${INAT_DIR}/train.json.tar.gz"
fi

# --- 2. Raw recordings (targeted 74-species extraction) ----------------------
# Normalize the directory list to LF first: it is authored on Windows, and a
# trailing CR makes tar search for "<dir>\r" and match nothing ("Not found in
# archive"). Strip CR at runtime so the extraction is robust regardless of the
# file's line endings.
EXTRACT_LIST=/tmp/extract_dirs.lf.txt
tr -d '\r' < "${PREP}/extract_dirs.txt" > "${EXTRACT_LIST}"

if [ ! -d "${INAT_DIR}/extracted/train" ] || [ -z "$(ls -A "${INAT_DIR}/extracted/train" 2>/dev/null)" ]; then
    TARBALL="${INAT_DIR}/train.tar.gz"
    echo "[prep] downloading iNatSounds train.tar.gz (~81 GB) -> ${TARBALL} (resumable)..."
    # Download to disk (not a pipe) so the expensive transfer can resume with
    # -C - after any interruption, and so extraction can be re-run for free.
    curl -fL -C - "${INAT_TRAIN_URL}" -o "${TARBALL}"
    echo "[prep] extracting 74 species directories from the local tarball..."
    mkdir -p "${INAT_DIR}/extracted"
    tar --use-compress-program=unpigz -x \
        -f "${TARBALL}" \
        -C "${INAT_DIR}/extracted" \
        -T "${EXTRACT_LIST}"
    echo "[prep] extraction complete; delete ${TARBALL} to reclaim ~81 GB once satisfied"
else
    echo "[prep] reusing existing recordings at ${INAT_DIR}/extracted (no download)"
fi

# --- 3. Sort into birdnet-stm32 data layout ----------------------------------
echo "[prep] sorting species into ${DATA_DIR}..."
python "${PREP}/sort_inat.py" \
    --json "${INAT_DIR}/train.json" \
    --species "${PREP}/species_plainfield.txt" \
    --recordings "${INAT_DIR}/extracted" \
    --out "${DATA_DIR}" \
    --max-per-species "${MAX_PER_SPECIES}" \
    --test-frac "${TEST_FRAC}"

# --- 4. Background negatives (ESC-50) ----------------------------------------
if [ ! -f "${INAT_DIR}/esc50.zip" ]; then
    echo "[prep] downloading ESC-50 for negatives..."
    curl -fL "${ESC50_URL}" -o "${INAT_DIR}/esc50.zip"
fi
echo "[prep] building background negative class..."
ESC50_ZIP="${INAT_DIR}/esc50.zip" \
OUT_TRAIN="${DATA_DIR}/train/background" \
OUT_TEST="${DATA_DIR}/test/background" \
    python "${PREP}/make_negatives.py"

# --- summary ------------------------------------------------------------------
train_dirs=$(find "${DATA_DIR}/train" -mindepth 1 -maxdepth 1 -type d | wc -l)
train_files=$(find "${DATA_DIR}/train" -type f -name '*.wav' | wc -l)
test_files=$(find "${DATA_DIR}/test" -type f -name '*.wav' | wc -l)
echo "[prep] DONE: ${train_dirs} class dirs, ${train_files} train + ${test_files} test clips in ${DATA_DIR}"
