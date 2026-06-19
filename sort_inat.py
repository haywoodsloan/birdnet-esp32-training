#!/usr/bin/env python3
"""Sort iNatSounds 2024 recordings into birdnet-stm32's data/train layout.

Given the COCO-style train.json annotations and a species list (scientific
names), this:
  1. Resolves each requested species to its iNatSounds category + audio_dir_name.
  2. (validate) reports recordings found per species, and any species missing
     from the dataset -- run this BEFORE downloading the 81 GB recordings.
  3. (copy) once the recordings tarball is extracted, copies each species'
     WAVs into <out>/train/<Scientific_name>/ (and a held-out slice into
     <out>/test/<Scientific_name>/), capping per species with --max-per-species.

Species list format: one scientific name per line ("Genus species"); '#'
comments and blank lines ignored (trailing "# common name" also ok).

Usage:
    # 1) validate the list against the dataset (no recordings needed):
    python sort_inat.py --json train.json --species species_plainfield.txt --validate

    # 2) after extracting train.tar.gz to <recordings_dir>/train/<cat>/...:
    python sort_inat.py --json train.json --species species_plainfield.txt \
        --recordings <recordings_dir> --out ../birdnet-data \
        --max-per-species 150 --test-frac 0.15
"""
import argparse
import collections
import json
import os
import random
import shutil
import sys


def load_species(path):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # strip trailing "# common name" comment
            name = line.split("#", 1)[0].strip()
            if name:
                out.append(name)
    return out


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--species", required=True)
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--recordings", help="dir containing extracted train/<cat>/*.wav")
    ap.add_argument("--out", default="birdnet-data")
    ap.add_argument("--max-per-species", type=int, default=150)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv[1:])

    with open(args.json, "r", encoding="utf-8") as f:
        data = json.load(f)

    # category_id -> recording count
    counts = collections.Counter(a["category_id"] for a in data["annotations"])
    # audio_id -> file_name
    audio_file = {a["id"]: a["file_name"] for a in data["audio"]}
    # category_id -> [file_name, ...]
    cat_files = collections.defaultdict(list)
    for a in data["annotations"]:
        cat_files[a["category_id"]].append(audio_file.get(a["audio_id"]))

    # Index categories by "Genus species" (lowercased) for matching.
    by_sci = {}
    for c in data["categories"]:
        sci = f"{c.get('genus','')} {c.get('specific_epithet','')}".strip().lower()
        if sci:
            by_sci[sci] = c

    wanted = load_species(args.species)
    found, missing = [], []
    for sci in wanted:
        c = by_sci.get(sci.lower())
        if c is None:
            missing.append(sci)
        else:
            found.append((sci, c, counts.get(c["id"], 0)))

    print(f"requested {len(wanted)} species: {len(found)} found, {len(missing)} missing")
    total = 0
    print(f"\n{'recs':>6}  scientific                     common")
    for sci, c, n in sorted(found, key=lambda t: -t[2]):
        total += min(n, args.max_per_species)
        print(f"{n:>6}  {sci:<30} {c.get('common_name','')}")
    if missing:
        print("\nMISSING (not in iNatSounds train set, fix the list):")
        for m in missing:
            print(f"  - {m}")
    print(f"\nwith --max-per-species {args.max_per_species}: ~{total} train clips "
          f"across {len(found)} classes")

    if args.validate or not args.recordings:
        if not args.recordings:
            print("\n(validate only -- pass --recordings <dir> to copy files)")
        return 0 if not missing else 1

    # --- copy mode ---
    rng = random.Random(args.seed)
    rec_root = os.path.join(args.recordings, "train")
    n_train = n_test = 0
    for sci, c, _ in found:
        adir = c.get("audio_dir_name") or str(c["id"])
        src_dir = os.path.join(rec_root, adir)
        files = [fn for fn in cat_files[c["id"]] if fn]
        rng.shuffle(files)
        files = files[: args.max_per_species]
        n_test_take = int(len(files) * args.test_frac)
        safe = sci.replace(" ", "_")
        for i, fn in enumerate(files):
            base = os.path.basename(fn)
            src = os.path.join(src_dir, base)
            if not os.path.isfile(src):
                # some datasets nest file_name with the category dir already
                src = os.path.join(rec_root, fn)
            if not os.path.isfile(src):
                continue
            split = "test" if i < n_test_take else "train"
            dst_dir = os.path.join(args.out, split, safe)
            os.makedirs(dst_dir, exist_ok=True)
            shutil.copy2(src, os.path.join(dst_dir, base))
            if split == "test":
                n_test += 1
            else:
                n_train += 1
    print(f"\ncopied {n_train} train + {n_test} test clips -> {args.out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
