#!/usr/bin/env python3
"""Download Xeno-canto recordings for the Plainfield species and sort them into
the birdnet-stm32 data layout, to augment the iNatSounds training set.

WHY: iNatSounds clips are weakly labeled (a clip tagged species X often contains
other, louder birds), which makes quiet species lose to dominant ones. Adding
Xeno-canto recordings -- many of which are cleaner, closer single-species cuts,
especially for the low-count species -- dilutes that label noise.

Xeno-canto API v3 (the v2 API is retired) requires a free API key from
https://xeno-canto.org/account. Provide it via --key or the XC_API_KEY env var.

The API endpoint is reachable programmatically (the bot wall is only on the HTML
site). Field names in v3 can differ from v2, so run `--probe` first to dump one
raw record and confirm the schema before a full download.

Outputs (matching sort_inat.py / the data volume layout):
    <out>/train/<Genus_species>/xc<id>.wav      (24 kHz mono)
    <out>/test/<Genus_species>/xc<id>.wav
    <out>/XENOCANTO_ATTRIBUTION.tsv             (id, species, recordist, license)

Existing files are never overwritten, so this is safe to re-run / resume, and it
merges with the iNat data already present.

Usage (in the training container, where ffmpeg + requests are available):
    python fetch_xenocanto.py --probe --key $XC_API_KEY
    python fetch_xenocanto.py --key $XC_API_KEY --max-per-species 300 --min-quality B
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.parse
import urllib.request

API = "https://xeno-canto.org/api/3/recordings"
USER_AGENT = "esp32-s3-eye-birdnet/1.0 (personal hobby model training)"


def load_species(path: str) -> list[str]:
    """Read 'Genus species' per line; ignore '#' comments and trailing comments."""
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            name = line.split("#", 1)[0].strip()
            if name:
                out.append(name)
    return out


def api_get(params: dict, retries: int = 3, timeout: int = 60) -> dict:
    url = API + "?" + urllib.parse.urlencode(params)
    last = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            last = exc
            if attempt < retries:
                time.sleep(2.0 + 2 * attempt)
    raise last  # type: ignore[misc]


# Quality grades, best first. Xeno-canto grades recordings A (best) .. E.
_GRADE_ORDER = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "": 9, "no score": 9}


def _rec_field(rec: dict, *names: str, default: str = "") -> str:
    """Return the first present field among `names` (v3 schema is defensive)."""
    for n in names:
        if n in rec and rec[n] not in (None, ""):
            return str(rec[n])
    return default


def _download_url(rec: dict) -> str | None:
    """Resolve a recording's audio download URL across schema variants."""
    url = _rec_field(rec, "file", "download", default="")
    if not url:
        rid = _rec_field(rec, "id")
        if rid:
            url = f"https://xeno-canto.org/{rid}/download"
    if url.startswith("//"):
        url = "https:" + url
    return url or None


def fetch_species_records(sci: str, key: str, min_quality: str, delay: float) -> list[dict]:
    """Fetch all (paged) records for a species at >= min_quality, best grade first."""
    gen, _, sp = sci.partition(" ")
    # Xeno-canto API v3 ONLY accepts tag-based queries (a plain "Genus species"
    # string returns HTTP 400). Use gen:/sp: tags restricted to birds.
    query = f"gen:{gen} sp:{sp} grp:birds"
    page = 1
    records: list[dict] = []
    while True:
        data = api_get({"query": query, "key": key, "per_page": 500, "page": page})
        recs = data.get("recordings") or []
        records.extend(recs)
        num_pages = int(data.get("numPages", 1) or 1)
        if page >= num_pages or not recs:
            break
        page += 1
        time.sleep(delay)

    max_grade = _GRADE_ORDER.get(min_quality.upper(), 1)
    kept = [r for r in records if _GRADE_ORDER.get(_rec_field(r, "q", "quality").upper(), 9) <= max_grade]
    kept.sort(key=lambda r: _GRADE_ORDER.get(_rec_field(r, "q", "quality").upper(), 9))
    return kept


def transcode_to_wav(src_bytes: bytes, dst_path: str, sample_rate: int) -> bool:
    """Pipe downloaded audio through ffmpeg -> mono WAV at sample_rate. ffmpeg
    decodes mp3/wav/flac transparently, so we don't care about the source codec."""
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-i", "pipe:0", "-ac", "1", "-ar", str(sample_rate), dst_path],
        input=src_bytes, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )
    if proc.returncode != 0 or not os.path.isfile(dst_path) or os.path.getsize(dst_path) < 1024:
        if os.path.isfile(dst_path):
            os.remove(dst_path)
        return False
    return True


def http_bytes(url: str, timeout: int = 120) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--species", default="/workspace/prep/species_plainfield.txt")
    ap.add_argument("--out", default="/data")
    ap.add_argument("--key", default=os.environ.get("XC_API_KEY", ""))
    ap.add_argument("--max-per-species", type=int, default=300)
    ap.add_argument("--min-quality", default="B", help="lowest acceptable grade (A best)")
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--sample-rate", type=int, default=24000)
    ap.add_argument("--min-len", type=int, default=3, help="min recording seconds")
    ap.add_argument("--max-len", type=int, default=120, help="max recording seconds")
    ap.add_argument("--delay", type=float, default=1.0, help="seconds between requests")
    ap.add_argument("--only", default=None, help="only this scientific name")
    ap.add_argument("--probe", action="store_true", help="dump one raw record and exit")
    args = ap.parse_args()

    if not args.key:
        print("ERROR: no API key. Pass --key or set XC_API_KEY "
              "(get one at https://xeno-canto.org/account).", file=sys.stderr)
        return 2

    if args.probe:
        data = api_get({"query": "gen:Cardinalis sp:cardinalis grp:birds", "key": args.key, "per_page": 1, "page": 1})
        top = {k: v for k, v in data.items() if k != "recordings"}
        print("=== response top-level keys ===")
        print(json.dumps(top, indent=2)[:1500])
        recs = data.get("recordings") or []
        print("=== first recording (raw) ===")
        print(json.dumps(recs[0], indent=2) if recs else "NO RECORDINGS RETURNED")
        return 0

    species = load_species(args.species)
    if args.only:
        species = [s for s in species if args.only.lower() in s.lower()]

    def parse_len_s(rec: dict) -> int:
        v = _rec_field(rec, "length", "len")
        if ":" in v:
            try:
                m, s = v.split(":")[-2:]
                return int(m) * 60 + int(s)
            except ValueError:
                return 0
        try:
            return int(float(v))
        except ValueError:
            return 0

    att_path = os.path.join(args.out, "XENOCANTO_ATTRIBUTION.tsv")
    os.makedirs(args.out, exist_ok=True)
    new_att = not os.path.exists(att_path)
    att = open(att_path, "a", encoding="utf-8")
    if new_att:
        att.write("xc_id\tspecies\trecordist\tlicense\tquality\n")

    tot_train = tot_test = 0
    for idx, sci in enumerate(species, 1):
        safe = sci.replace(" ", "_")
        tdir = os.path.join(args.out, "train", safe)
        vdir = os.path.join(args.out, "test", safe)
        have = len(os.listdir(tdir)) if os.path.isdir(tdir) else 0

        try:
            recs = fetch_species_records(sci, args.key, args.min_quality, args.delay)
        except Exception as exc:  # noqa: BLE001
            print(f"[{idx}/{len(species)}] {sci}: API error {exc!r}; skipping")
            continue

        recs = [r for r in recs if args.min_len <= parse_len_s(r) <= args.max_len]
        recs = recs[: args.max_per_species]
        n_test_take = int(len(recs) * args.test_frac)
        added_t = added_v = 0

        for i, rec in enumerate(recs):
            rid = _rec_field(rec, "id")
            if not rid:
                continue
            split_dir = vdir if i < n_test_take else tdir
            dst = os.path.join(split_dir, f"xc{rid}.wav")
            if os.path.isfile(dst):
                continue
            url = _download_url(rec)
            if not url:
                continue
            try:
                raw = http_bytes(url)
                if not transcode_to_wav(raw, dst, args.sample_rate):
                    print(f"    xc{rid}: transcode failed")
                    continue
            except Exception as exc:  # noqa: BLE001
                print(f"    xc{rid}: download failed {exc!r}")
                time.sleep(args.delay)
                continue
            att.write(f"{rid}\t{sci}\t{_rec_field(rec, 'rec', 'recordist')}\t"
                      f"{_rec_field(rec, 'lic', 'license')}\t{_rec_field(rec, 'q', 'quality')}\n")
            if split_dir is vdir:
                added_v += 1
            else:
                added_t += 1
            time.sleep(args.delay)

        tot_train += added_t
        tot_test += added_v
        print(f"[{idx}/{len(species)}] {sci}: had {have}, +{added_t} train +{added_v} test "
              f"(from {len(recs)} XC recs >= {args.min_quality})")
        att.flush()

    att.close()
    print(f"\nDONE: added {tot_train} train + {tot_test} test Xeno-canto clips to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
