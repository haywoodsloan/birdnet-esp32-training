#!/usr/bin/env python3
"""Patch birdnet-stm32's data/generator.py to support BirdNET soft-target
distillation, gated on the BIRDNET_SOFT_TARGETS env var (no-op when unset).

Adds three things to generator.py:
  1. a module-level loader that reads soft_targets.npz and remaps its columns to
     the training class order (order-robust);
  2. a "soft_targets" entry in the picklable worker_cfg (built once, shipped to
     workers); and
  3. a substitution at the sample-append site so a file's one-hot label is
     replaced by its precomputed soft target when one exists.

Idempotent and self-checking: each anchor must be present exactly once or the
script aborts (so an upstream change can't silently drop the patch).
"""
import sys

GEN = sys.argv[1] if len(sys.argv) > 1 else \
    "/workspace/birdnet-stm32/birdnet_stm32/data/generator.py"

with open(GEN, "r", encoding="utf-8") as f:
    src = f.read()

if "_bn_load_soft_targets" in src:
    print("generator.py already patched for distillation; skipping")
    sys.exit(0)

LOADER = '''def _bn_load_soft_targets(classes):
    """Load soft_targets.npz (BIRDNET_SOFT_TARGETS) as {path: float32[C]},
    remapped to the given training class order. Returns None when unset/missing,
    so normal (non-distillation) training is completely unaffected."""
    import os as _os
    p = _os.environ.get("BIRDNET_SOFT_TARGETS")
    if not p or not _os.path.isfile(p):
        return None
    d = np.load(p, allow_pickle=True)
    src_classes = [str(c) for c in d["classes"]]
    col = {c: i for i, c in enumerate(src_classes)}
    if any(c not in col for c in classes):
        print("[distill] soft_targets classes do not cover training classes; disabling")
        return None
    idx = [col[c] for c in classes]
    targets = d["targets"][:, idx].astype("float32")
    paths = [str(x) for x in d["paths"]]
    print(f"[distill] loaded {len(paths)} soft targets x {len(classes)} classes")
    return {paths[i]: targets[i] for i in range(len(paths))}


def _process_file(path: str):'''

anchors = [
    # (description, find, replace)
    ("loader fn", "def _process_file(path: str):", LOADER),
    ("cfg entry",
     '        "num_classes": num_classes,\n        "max_chunks_per_file": max_chunks_per_file,',
     '        "num_classes": num_classes,\n        "soft_targets": _bn_load_soft_targets(classes),\n'
     '        "max_chunks_per_file": max_chunks_per_file,'),
    ("append substitution",
     "        sample = np.expand_dims(sample, axis=-1).astype(np.float32)\n"
     "        results.append((sample, label))",
     "        sample = np.expand_dims(sample, axis=-1).astype(np.float32)\n"
     "        _st = cfg.get(\"soft_targets\")\n"
     "        _lab = _st.get(path, label) if _st else label\n"
     "        results.append((sample, _lab))"),
]

for desc, find, repl in anchors:
    count = src.count(find)
    if count != 1:
        print(f"ERROR: anchor '{desc}' found {count} times (expected 1); aborting")
        sys.exit(1)
    src = src.replace(find, repl)

with open(GEN, "w", encoding="utf-8") as f:
    f.write(src)
print("generator.py patched for distillation (gated on BIRDNET_SOFT_TARGETS)")
