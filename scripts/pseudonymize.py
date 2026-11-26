#!/usr/bin/env python3
"""Build the pseudonymized data released with this repository.

1. Message tables: reads the raw Telegram exports (not distributed), keeps
   only the fields the analyses need (date, sender_id, topic_name) and
   replaces every Telegram sender_id with a random integer pseudonym.
2. Result tables (optional): replaces the ``user_id`` column of the CNHPP
   result CSVs with the same pseudonyms.

The raw-id -> pseudonym tables are written to data/private/ and must never be
committed. Re-running the script reuses them, so pseudonyms stay stable.

Usage
-----
    python scripts/pseudonymize.py --raw-dir data/raw
    python scripts/pseudonymize.py --raw-dir data/raw \
        --results-in <dir with raw-id result CSVs> --results-out results/cnhpp
"""
import argparse
import secrets
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_FILES = {"dana": "voluntariosdanavalencia_old.json", "vagarenfe": "vagarenfe.json"}
KEEP = ["date", "sender_id", "topic_name"]
FIRST_PSEUDONYM = 100_000


def mapping_path(community, private_dir):
    return Path(private_dir) / f"id_mapping_{community}.csv"


def load_or_extend_mapping(community, raw_ids, private_dir):
    """Return {raw_id: pseudonym}, creating or extending the private table."""
    path = mapping_path(community, private_dir)
    mapping = {}
    if path.exists():
        m = pd.read_csv(path)
        mapping = dict(zip(m["raw_id"].astype("int64"), m["pseudo_id"].astype("int64")))
    new = sorted(set(int(x) for x in raw_ids) - set(mapping))
    if new:
        rng = np.random.default_rng(secrets.randbits(64))
        start = max(mapping.values(), default=FIRST_PSEUDONYM - 1) + 1
        for k, idx in enumerate(rng.permutation(len(new))):
            mapping[new[idx]] = start + k
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"raw_id": list(mapping), "pseudo_id": list(mapping.values())}).to_csv(path, index=False)
    return mapping


def pseudonymize_messages(community, raw_dir, out_dir, private_dir):
    df = pd.read_json(Path(raw_dir) / RAW_FILES[community])[KEEP].copy()
    df["date"] = pd.to_datetime(df["date"], utc=True)
    ids = pd.to_numeric(df["sender_id"], errors="coerce")
    mapping = load_or_extend_mapping(community, ids.dropna().astype("int64").unique(), private_dir)
    df["sender_id"] = ids.map(lambda x: mapping[int(x)] if pd.notna(x) else pd.NA).astype("Int64")
    out = Path(out_dir) / f"{community}_messages.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)  # original export order is preserved on purpose
    return out, len(df), len(mapping)


def community_of(filename):
    return "vagarenfe" if "vagarenfe" in filename else "dana"


def pseudonymize_results(results_in, results_out, private_dir):
    written = []
    for src in sorted(Path(results_in).glob("*.csv")):
        community = community_of(src.name)
        m = pd.read_csv(mapping_path(community, private_dir))
        mapping = dict(zip(m["raw_id"].astype("int64"), m["pseudo_id"].astype("int64")))
        df = pd.read_csv(src)
        raw = df["user_id"].astype("int64")
        missing = set(raw) - set(mapping)
        if missing:
            raise KeyError(f"{src.name}: {len(missing)} user_id values have no pseudonym")
        df["user_id"] = raw.map(mapping)
        dst = Path(results_out) / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(dst, index=False)
        written.append(dst)
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", default=ROOT / "data" / "raw")
    ap.add_argument("--out-dir", default=ROOT / "data" / "processed")
    ap.add_argument("--private-dir", default=ROOT / "data" / "private")
    ap.add_argument("--results-in", default=None)
    ap.add_argument("--results-out", default=ROOT / "results" / "cnhpp")
    a = ap.parse_args()
    for community in RAW_FILES:
        out, n, u = pseudonymize_messages(community, a.raw_dir, a.out_dir, a.private_dir)
        print(f"{community}: {n} messages, {u} pseudonymous senders -> {out}")
    if a.results_in:
        for p in pseudonymize_results(a.results_in, a.results_out, a.private_dir):
            print(f"results -> {p}")


if __name__ == "__main__":
    main()
