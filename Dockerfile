# BirdNET-STM32 GPU training image.
#
# Strategy: a slim Python base plus the project's own `tensorflow[and-cuda]`
# pin (declared in birdnet-stm32/pyproject.toml). The CUDA/cuDNN runtime is
# delivered entirely through the NVIDIA pip wheels, so the only host
# requirement is the GPU driver, exposed by `--gpus all` via the
# nvidia-container-toolkit (bundled with Docker Desktop's WSL2 engine).
#
# This avoids mixing a system CUDA toolkit with the pip CUDA wheels, which is
# the usual source of "wrong cuDNN version" breakage. The RTX 50-series
# (Blackwell, sm_120) is driven by the recent host driver; modern TF wheels
# ship sm_120 kernels (older ones JIT from PTX on first launch).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive \
    TF_CPP_MIN_LOG_LEVEL=1

# System libraries:
#   git/curl/ca-certificates -> clone repo + download datasets
#   libsndfile1              -> soundfile/librosa audio decode
#   ffmpeg                   -> mp3/m4a fallback decode
#   libgomp1                 -> TensorFlow OpenMP runtime
#   pigz                     -> parallel gzip for fast tar extraction
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates libsndfile1 ffmpeg libgomp1 pigz \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# --- Clone the training repo (single source of truth; no host copy needed) ---
ARG BIRDNET_REPO=https://github.com/birdnet-team/birdnet-stm32.git
ARG BIRDNET_REF=master
RUN git clone --depth 1 --branch "${BIRDNET_REF}" "${BIRDNET_REPO}" birdnet-stm32

# --- Install the project + GPU TensorFlow (pulls tensorflow[and-cuda]) ---
RUN python -m pip install --upgrade pip \
    && python -m pip install -e "./birdnet-stm32[dev]"

# --- Fix the data-loader deadlock ------------------------------------------
# Upstream sets multiprocessing.Pool(maxtasksperchild=100), recycling each
# worker ~6x/epoch. On Linux that forks a fresh worker from a parent that has
# already spawned TensorFlow's background threads -> the classic "fork in a
# multithreaded process" race. It deadlocks on a glibc/malloc lock held across
# the fork: step counter freezes, ALL workers sit in S (sleep), the main
# process burns 0 CPU, GPU goes idle, container stays "running".
#
# PROVEN intermittent and length-dependent: an unpatched build ran 50 epochs
# clean on an idle machine but deadlocked at epoch 26 of a 300-epoch run on the
# same idle box. More epochs = more recycle-forks = higher cumulative odds of
# hitting the race, so long runs reliably hang.
#
# maxtasksperchild=None makes workers fork ONCE at pool creation and live for
# the whole run, eliminating the recurring fork. Zero effect on results
# (unpatched 50ep AUC 0.811 == patched 0.813). The install is editable (-e), so
# patching the source here takes effect at runtime. Fail the build if the line
# is missing so an upstream change can't silently drop the fix.
RUN F=birdnet-stm32/birdnet_stm32/data/generator.py \
    && grep -q "maxtasksperchild=100," "$F" \
    && sed -i 's/maxtasksperchild=100,/maxtasksperchild=None,/' "$F" \
    && grep -q "maxtasksperchild=None," "$F" \
    && echo "deadlock fix applied: maxtasksperchild=None"

# --- Add top-1 / top-3 accuracy metrics ------------------------------------
# Upstream only tracks ROC-AUC. We also want plain classification accuracy
# (the headline number the project targets). CategoricalAccuracy compares
# argmax(y_true) vs argmax(y_pred), so it reports top-1 even with the
# multi-label sigmoid head + mixup/label-smoothing (validation has mixup off,
# so val_top1_acc is clean single-label top-1 accuracy). Editable install, so
# the patch takes effect at runtime; fail the build if the anchor is missing.
RUN F=birdnet-stm32/birdnet_stm32/training/trainer.py \
    && grep -q 'name="roc_auc")]' "$F" \
    && sed -i 's/name="roc_auc")]/name="roc_auc"), tf.keras.metrics.CategoricalAccuracy(name="top1_acc"), tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_acc")]/' "$F" \
    && grep -q "top1_acc" "$F" \
    && echo "accuracy metrics added: top1_acc, top3_acc"

# --- Fix early stopping: monitor val_top1_acc, run the FULL schedule --------
# Two upstream defaults fight a long, distilled run:
#   1. monitor=val_loss bottoms out in a few epochs under label smoothing +
#      mixup + class weights, so it never reflects real convergence.
#   2. ROC-AUC (our previous monitor) is DEGENERATE under knowledge
#      distillation: Keras' AUC metric expects binary {0,1} labels, but the
#      BirdNET-blended soft targets are continuous in [0,1]. val_roc_auc then
#      sits ~0.41 (below random) and never "improves", so EarlyStopping killed
#      a 500-epoch run at epoch 36 (patience 30) AND ModelCheckpoint saved the
#      "best" epoch by a meaningless number.
# CategoricalAccuracy (top1_acc) compares argmax(y_true) vs argmax(y_pred); the
# soft target's true class is floored to (1-alpha) so it stays the argmax,
# making val_top1_acc the correct thing to track with or without distillation.
# Monitor val_top1_acc (mode max) for both EarlyStopping and ModelCheckpoint,
# and set patience to 600 (> any epoch count we run) so the schedule ALWAYS
# completes -- the run can no longer be cut short. restore_best_weights still
# brings back the highest-val_top1_acc epoch at the end. Fail the build if the
# anchors are missing so an upstream change can't silently revert this.
RUN F=birdnet-stm32/birdnet_stm32/training/trainer.py \
    && grep -q 'monitor="val_loss"' "$F" \
    && sed -i 's/monitor="val_loss"/monitor="val_top1_acc"/g; s/mode="min"/mode="max"/g; s/patience: int = 10,/patience: int = 600,/' "$F" \
    && grep -q 'monitor="val_top1_acc"' "$F" \
    && ! grep -q 'monitor="val_loss"' "$F" \
    && echo "early-stopping fix applied: monitor=val_top1_acc mode=max patience=600"

# --- Data-prep tooling + GPU entrypoint (small files from the build context) ---
COPY sort_inat.py make_negatives.py species_plainfield.txt extract_dirs.txt prepare_data.sh patch_distill.py patch_train_hpf.py /workspace/prep/
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
# Normalize line endings (the build context is authored on Windows) and make
# the scripts executable. extract_dirs.txt MUST be LF: tar -T treats a trailing
# CR as part of the member name ("<dir>\r"), which matches nothing.
RUN sed -i 's/\r$//' /workspace/prep/prepare_data.sh /usr/local/bin/entrypoint.sh /workspace/prep/extract_dirs.txt \
    && chmod +x /workspace/prep/prepare_data.sh /usr/local/bin/entrypoint.sh

# --- BirdNET soft-target distillation hook (no-op unless BIRDNET_SOFT_TARGETS set) ---
# Patches data/generator.py so the loader can swap a file's one-hot label for a
# precomputed BirdNET-blended soft target. Gated on the env var, so this does
# not affect normal training. Build fails if the patch anchors are missing.
RUN sed -i 's/\r$//' /workspace/prep/patch_distill.py \
    && python /workspace/prep/patch_distill.py /workspace/birdnet-stm32/birdnet_stm32/data/generator.py \
    && python -c "import ast; ast.parse(open('/workspace/birdnet-stm32/birdnet_stm32/data/generator.py').read())" \
    && echo "distillation hook installed"

# --- Device-domain match hook (no-op unless BIRDNET_TRAIN_HPF_HZ / NOISE set) ---
# Patches audio/spectrogram.py so training audio can be DC-removed + high-passed
# and mixed with recorded device noise, matching what the ESP32 firmware feeds
# the model at inference. Gated on env vars, so it does not affect normal
# training. Build fails if the patch anchors are missing.
RUN sed -i 's/\r$//' /workspace/prep/patch_train_hpf.py \
    && python /workspace/prep/patch_train_hpf.py \
    && python -c "import ast; ast.parse(open('/workspace/birdnet-stm32/birdnet_stm32/audio/spectrogram.py').read())" \
    && echo "device-match hook installed"

# Persist the Blackwell (sm_120) PTX->SASS JIT cache so the one-time
# compilation only happens on the very first GPU run, not every container.
ENV DATA_DIR=/data \
    INAT_DIR=/inat \
    CUDA_CACHE_PATH=/cuda_cache \
    CUDA_CACHE_MAXSIZE=2147483648

WORKDIR /workspace/birdnet-stm32

# Entrypoint fixes up LD_LIBRARY_PATH for the pip CUDA wheels, then runs CMD.
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash"]
