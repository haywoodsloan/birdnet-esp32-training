#!/usr/bin/env python3
"""Option B -- BirdNET label cleaning for the weakly-labeled iNat training set.

iNatSounds clips are labeled by their TARGET species but often contain a louder
non-target bird, so the student can learn the wrong (dominant) species. The
BirdNET teacher (raw per-class confidences saved by make_soft_targets.py as
``raw_teacher``) tells us when the labeled species is NOT actually what's audible.

This QUARANTINES (moves, never deletes -- fully reversible) only the clearly
mislabeled files: the teacher barely hears the labeled species AND is confident
about a DIFFERENT target species. Quiet-but-correct clips (teacher unsure about
everything) are kept. Xeno-canto (xc*) and device (dev*) files are skipped --
they are clean single-species cuts that don't need cleaning. A per-class cap
prevents gutting any class.

Default is a DRY RUN (reports what would move). Pass --apply to actually move.

Usage (in the training container):
    python clean_labels.py                 # dry run
    python clean_labels.py --apply         # quarantine -> /data/quarantine
    # to undo: move files back from /data/quarantine/<class>/ to /data/train/<class>/
"""
from __future__ import annotations

import argparse
import os
import shutil

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", default="/data/soft_targets.npz")
    ap.add_argument("--data", default="/data/train")
    ap.add_argument("--quarantine", default="/data/quarantine")
    ap.add_argument("--label-conf-max", type=float, default=0.10,
                    help="drop only if teacher conf on the LABELED class < this")
    ap.add_argument("--other-conf-min", type=float, default=0.50,
                    help="AND teacher conf on some OTHER target class >= this")
    ap.add_argument("--max-drop-frac", type=float, default=0.15,
                    help="never quarantine more than this fraction of a class")
    ap.add_argument("--apply", action="store_true",
                    help="actually move files (default: dry run)")
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    if "raw_teacher" not in d:
        print("ERROR: npz has no 'raw_teacher' -- re-run make_soft_targets.py "
              "(the version that saves raw_teacher) first.")
        return 1
    classes = [str(c) for c in d["classes"]]
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    paths = [str(p) for p in d["paths"]]
    raw = d["raw_teacher"].astype(np.float32)

    # Candidate flags per file.
    flagged: dict[str, list[str]] = {}   # class -> [paths]
    per_class_total: dict[str, int] = {}
    for p, r in zip(paths, raw):
        cls = os.path.basename(os.path.dirname(p))
        if cls not in cls_to_idx:
            continue
        base = os.path.basename(p)
        if base.startswith("xc") or base.startswith("dev"):
            continue  # clean sources; never quarantine
        per_class_total[cls] = per_class_total.get(cls, 0) + 1
        c = cls_to_idx[cls]
        label_conf = float(r[c])
        other = r.copy()
        other[c] = -1.0
        best_other = float(other.max())
        if label_conf < args.label_conf_max and best_other >= args.other_conf_min:
            flagged.setdefault(cls, []).append(p)

    # Apply per-class cap (drop the most-confidently-wrong first).
    to_move: list[str] = []
    report = []
    for cls in sorted(per_class_total):
        cand = flagged.get(cls, [])
        cap = int(per_class_total[cls] * args.max_drop_frac)
        keep_n = min(len(cand), cap)
        # sort candidates by best_other desc (most confident mislabel first)
        def best_other_of(p):
            i = paths.index(p)
            r = raw[i].copy()
            r[cls_to_idx[cls]] = -1.0
            return float(r.max())
        cand_sorted = sorted(cand, key=best_other_of, reverse=True)[:keep_n]
        to_move.extend(cand_sorted)
        if cand:
            report.append((cls, len(cand), keep_n, per_class_total[cls]))

    report.sort(key=lambda x: x[2], reverse=True)
    print(f"iNat files considered: {sum(per_class_total.values())}")
    print(f"flagged mislabeled (label_conf<{args.label_conf_max} & "
          f"other>={args.other_conf_min}): {sum(len(v) for v in flagged.values())}")
    print(f"to quarantine after {args.max_drop_frac:.0%} per-class cap: {len(to_move)}")
    print("\nclass                         flagged capped /total")
    for cls, nflag, ncap, ntot in report[:25]:
        print(f"  {cls:28s} {nflag:5d} {ncap:5d} /{ntot}")

    if not args.apply:
        print("\nDRY RUN -- pass --apply to move these to", args.quarantine)
        return 0

    moved = 0
    for p in to_move:
        cls = os.path.basename(os.path.dirname(p))
        dst_dir = os.path.join(args.quarantine, cls)
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, os.path.basename(p))
        if os.path.isfile(p):
            shutil.move(p, dst)
            moved += 1
    print(f"\nquarantined {moved} files -> {args.quarantine}")
    print("undo: move files back from quarantine/<class>/ to train/<class>/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
