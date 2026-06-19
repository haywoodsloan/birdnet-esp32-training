#!/usr/bin/env bash
#
# Container entrypoint: make the pip-installed CUDA wheels loadable.
#
# `pip install tensorflow[and-cuda]` ships cuDNN/cuBLAS/cuFFT/... as separate
# `nvidia-*` wheels under site-packages/nvidia/*/lib, but does NOT add those
# directories to the dynamic loader path. Without this, TensorFlow prints
# "Cannot dlopen some GPU libraries ... Skipping registering GPU devices" and
# silently falls back to CPU. We compute the list once at container start and
# prepend it to LD_LIBRARY_PATH, then exec the requested command.
set -e

if command -v python >/dev/null 2>&1; then
    nvlibs=$(python - <<'PY'
import os
try:
    import nvidia
    base = os.path.dirname(nvidia.__file__)
    dirs = [os.path.join(base, p, "lib") for p in sorted(os.listdir(base))
            if os.path.isdir(os.path.join(base, p, "lib"))]
    print(":".join(dirs))
except Exception:
    print("")
PY
)
    if [ -n "${nvlibs}" ]; then
        export LD_LIBRARY_PATH="${nvlibs}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    fi
fi

exec "$@"
