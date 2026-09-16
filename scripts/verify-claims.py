#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Re-derive the registered detection-evidence claims from the committed
corpora, and fail loudly if they no longer reproduce.

Why this exists
---------------
The compliance-critical detection numbers (TPR/FPR/mean z at 200/256/512
token truncations — EXPERIMENTS.md "Scheme comparison v2", facts.md
D1/D8/A7 context) were produced on the cluster once. This script makes
that evidence *concretely verifiable* by anyone with a checkout and the
detection key: detection is deterministic given (corpus, tokenizer, key),
so the registered table must reproduce bit-for-bit on a CPU laptop.

    REGISTERED CLAIM  ->  committed corpus + committed detector code
                      ->  this script  ->  PASS/FAIL against the table

What it checks
--------------
1. Corpus integrity: sha256 of the four committed corpora matches the
   values pinned below (detects silent corpus edits).
2. Claim reproduction: runs benchmarks/compare_schemes.py over those
   corpora (or verifies an existing --json output) and compares every
   (corpus, truncation) cell — n exactly, rate exactly at 3 decimals,
   mean z within 0.001 — against the registered table transcribed
   verbatim from EXPERIMENTS.md "Scheme comparison v2" (2026-08-08).

Requirements: local Python with torch+transformers (detector math only,
no GPU, no vLLM) and the watermark key:
    set -a && . cluster/watermark-key.env && set +a
The key is loaded via vllm_watermark.keys.load_key() and never printed.

Usage
-----
    python3 scripts/verify-claims.py                # recompute (~minutes, CPU)
    python3 scripts/verify-claims.py --json PATH    # verify an existing
                                                    # compare_schemes JSON
    python3 scripts/verify-claims.py --demo         # score one watermarked +
                                                    # one human sample (~secs)

Exit code 0 = every check passed; 1 = any mismatch; 2 = setup error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Registered corpora, pinned by content hash (sha256, computed 2026-08-11
# against the committed files that produced the registered table).
CORPORA = {
    "kgw": ("benchmarks/data/corpus_kgw512_fixed.jsonl",
            "e65bef5b8c3eca9e36434150a238205c95ca65db51fb6d67a3ef66e6861a7c37"),
    "synthid": ("benchmarks/data/corpus_synthid512.jsonl",
                "54788b69476e5176707e18c4b27dc0c33544fdc2777dd1073ca0fbcd0bb45b69"),
    "unwatermarked": ("benchmarks/data/corpus_wm_off.jsonl",
                      "107ff38689f9904c877fc3eb01df56a946f6cb780a1a74ef34f2eeba7a61232e"),
    "human": ("benchmarks/data/human_corpus_512.jsonl",
              "fbd9e945129d3ec2c504914d571a855d1cde98384c2acc1ce9599b1fc5d769b6"),
}

MODEL_TOKENIZER = "Qwen/Qwen2.5-0.5B-Instruct"
KEY_ID = "poc-2026-08"

# Transcribed verbatim from EXPERIMENTS.md
# "2026-08-08 — Scheme comparison v2" (the registered evidence table).
# {corpus_label: {truncation: (n, mean_z, rate)}}; n=0 rows carry None stats.
REGISTERED = {
    "kgw": {
        200: (120, 9.084, 0.992),
        256: (116, 10.261, 1.000),
        512: (76, 14.317, 1.000),
    },
    "synthid": {
        200: (120, 13.744, 1.000),
        256: (117, 15.651, 1.000),
        512: (96, 22.886, 1.000),
    },
    "unwatermarked (kgw det)": {
        200: (119, -0.148, 0.000),
        256: (115, -0.068, 0.000),
        512: (0, None, None),
    },
    "unwatermarked (synthid det)": {
        200: (119, -0.088, 0.000),
        256: (115, -0.060, 0.000),
        512: (0, None, None),
    },
    "human (kgw det)": {
        200: (150, 0.027, 0.000),
        256: (150, 0.084, 0.000),
        512: (150, 0.099, 0.000),
    },
    "human (synthid det)": {
        200: (150, -0.087, 0.000),
        256: (150, -0.070, 0.000),
        512: (150, -0.098, 0.000),
    },
}

MEAN_Z_TOL = 0.001


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_corpora() -> list[tuple[str, bool, str]]:
    rows = []
    for name, (rel, expected) in CORPORA.items():
        path = REPO / rel
        if not path.exists():
            rows.append((rel, False, "missing"))
            continue
        actual = sha256_file(path)
        rows.append((rel, actual == expected, actual))
    return rows


def recompute(out_stem: Path) -> Path:
    cmd = [
        sys.executable, str(REPO / "benchmarks" / "compare_schemes.py"),
        "--kgw-corpus", str(REPO / CORPORA["kgw"][0]),
        "--synthid-corpus", str(REPO / CORPORA["synthid"][0]),
        "--unwatermarked-corpus", str(REPO / CORPORA["unwatermarked"][0]),
        "--human-corpus", str(REPO / CORPORA["human"][0]),
        "--model-tokenizer", MODEL_TOKENIZER,
        "--key-id", KEY_ID,
        "--out", str(out_stem.with_suffix(".md")),
    ]
    print("Recomputing (CPU-bound, several minutes)...")
    subprocess.run(cmd, check=True)
    return out_stem.with_suffix(".json")


def compare(json_path: Path) -> list[tuple[str, int, bool, str]]:
    with open(json_path, encoding="utf-8") as f:
        payload = json.load(f)
    results = payload["results"]
    rows = []
    for label, per_len in REGISTERED.items():
        got_all = results.get(label)
        for length, (exp_n, exp_z, exp_rate) in per_len.items():
            got = (got_all or {}).get(str(length)) or (got_all or {}).get(length)
            if got is None:
                rows.append((label, length, False, "row absent from results"))
                continue
            problems = []
            if got["n"] != exp_n:
                problems.append(f"n {got['n']} != {exp_n}")
            if exp_n == 0:
                pass  # n/a cells: nothing further to compare
            else:
                if got["mean_z"] is None or abs(got["mean_z"] - exp_z) > MEAN_Z_TOL:
                    problems.append(f"mean_z {got['mean_z']} != {exp_z} (tol {MEAN_Z_TOL})")
                if got["rate"] is None or round(got["rate"], 3) != exp_rate:
                    problems.append(f"rate {got['rate']} != {exp_rate}")
            ok = not problems
            rows.append((label, length, ok, "; ".join(problems) if problems else
                         f"n={got['n']} mean_z={got['mean_z'] if got['mean_z'] is None else round(got['mean_z'], 3)} rate={got['rate']}"))
    return rows


def demo() -> int:
    """Score one committed watermarked sample and one human sample live.

    Content-safe: prints token counts, hashes, and z-scores — never the text.
    """
    sys.path.insert(0, str(REPO / "src"))
    from transformers import AutoConfig, AutoTokenizer  # noqa: PLC0415

    from vllm_watermark.keys import load_key  # noqa: PLC0415
    from vllm_watermark.kgw.core import KGWConfig  # noqa: PLC0415
    from vllm_watermark.kgw.detector import (  # noqa: PLC0415
        DEFAULT_Z_THRESHOLD,
        score_token_ids,
    )

    key = load_key(key_id=KEY_ID)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_TOKENIZER)
    vocab_size = AutoConfig.from_pretrained(MODEL_TOKENIZER).vocab_size
    cfg = KGWConfig(vocab_size=vocab_size, hash_key=key.hash_key, gamma=0.25)

    print(f"KGW detector demo — threshold z >= {DEFAULT_Z_THRESHOLD}, key_id={KEY_ID}")
    print(f"{'sample':<44} {'tokens':>6} {'z':>8}  verdict")
    for label, rel in (("watermarked (corpus_kgw512_fixed row 0)", CORPORA["kgw"][0]),
                       ("human (human_corpus_512 row 0)", CORPORA["human"][0])):
        with open(REPO / rel, encoding="utf-8") as f:
            text = json.loads(f.readline())["text"]
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        result = score_token_ids(ids, cfg, ignore_repeated_ngrams=True)
        digest = hashlib.sha256(text.encode()).hexdigest()[:12]
        verdict = "WATERMARK DETECTED" if result.prediction else "not detected"
        print(f"{label:<44} {len(ids):>6} {result.z_score:>8.3f}  {verdict}   [text elided sha256:{digest}]")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", default=None,
                        help="existing compare_schemes.py JSON output to verify instead of recomputing")
    parser.add_argument("--demo", action="store_true",
                        help="score one watermarked + one human sample and exit")
    args = parser.parse_args()

    if not (os.environ.get("WATERMARK_KEYS") or os.environ.get("WATERMARK_KEY")):
        print("error: watermark key not configured. Run:\n"
              "  set -a && . cluster/watermark-key.env && set +a", file=sys.stderr)
        return 2

    if args.demo:
        return demo()

    print("== 1. Corpus integrity (sha256 vs pinned values) ==")
    corpus_rows = check_corpora()
    corpora_ok = all(ok for _, ok, _ in corpus_rows)
    for rel, ok, detail in corpus_rows:
        print(f"  [{'PASS' if ok else 'FAIL'}] {rel}" + ("" if ok else f"  ({detail})"))
    if not corpora_ok:
        print("Corpus files changed — registered claims no longer apply to these files.")
        return 1

    if args.json:
        json_path = Path(args.json)
    else:
        out_stem = Path(tempfile.mkdtemp(prefix="verify-claims-")) / "scheme_reverify"
        json_path = recompute(out_stem)

    print("\n== 2. Registered-claim reproduction (vs EXPERIMENTS.md scheme comparison v2) ==")
    rows = compare(json_path)
    all_ok = True
    for label, length, ok, detail in rows:
        all_ok &= ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {label:<28} @{length:<4} {detail}")

    n_pass = sum(1 for _, _, ok, _ in rows if ok)
    print(f"\n{'ALL CHECKS PASSED' if all_ok else 'MISMATCHES FOUND'} "
          f"({n_pass}/{len(rows)} cells; source table: EXPERIMENTS.md 'Scheme comparison v2', 2026-08-08)")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
