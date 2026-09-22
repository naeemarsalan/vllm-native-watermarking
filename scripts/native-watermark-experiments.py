#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Measurement harness for vLLM's native (Gumbel-max) text watermark.

Answers the three due-diligence questions that source reading cannot:

  length-sweep   How short can a generation be and still be detected?
                 Generates completions at a range of target lengths and
                 temperatures and reports the true-positive rate at the
                 detector's threshold, plus the p-value distribution.

  robustness     Does the watermark survive editing, truncation and
                 paraphrase? Takes watermarked completions and applies
                 deterministic edits (truncation, word substitution,
                 word deletion) and a model-generated paraphrase, then
                 re-detects each and reports the degradation curve.

  modes          Is every generation mode actually watermarked? Exercises
                 plain sampling, greedy, top_k=1, structured JSON output,
                 tool calling, streaming, n>1 and the per-request opt-out,
                 and reports the detector's verdict for each. This is the
                 empirical check on silent bypasses.

All three talk to a running vLLM OpenAI-compatible server and the reference
detector over HTTP. Standard library only.

Every subcommand writes a JSON file with the full raw data (prompts,
completions, detector responses) so results can be re-analysed without
re-running the GPU work.

Usage:
    python3 scripts/native-watermark-experiments.py length-sweep \\
        --vllm-url https://... --detector-url https://... \\
        --cacert cluster/router-ca.crt --out results.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
import ssl
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_PROMPTS = REPO / "benchmarks" / "prompts.txt"
SSL_CONTEXT: ssl.SSLContext | None = None


# --------------------------------------------------------------------------- http


def _post(url: str, payload: dict, timeout: float = 600.0) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout, context=SSL_CONTEXT) as resp:
        return json.load(resp)


def _get(url: str, timeout: float = 30.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout, context=SSL_CONTEXT) as resp:
        return json.load(resp)


def generate(
    vllm: str,
    model: str,
    prompt: str,
    *,
    max_tokens: int,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    watermarking: bool | None = None,
    extra: dict | None = None,
) -> dict:
    """One chat completion. Returns {"text", "completion_tokens", "finish_reason"}."""
    payload: dict = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }
    if temperature is not None:
        payload["temperature"] = temperature
    if top_p is not None:
        payload["top_p"] = top_p
    if top_k is not None:
        payload["top_k"] = top_k
    if watermarking is not None:
        payload["watermarking"] = watermarking
    if extra:
        payload.update(extra)
    out = _post(f"{vllm}/v1/chat/completions", payload)
    choice = out["choices"][0]
    msg = choice.get("message") or {}
    text = msg.get("content") or ""
    if not text and msg.get("tool_calls"):
        # Tool calls put the model's sampled tokens in the arguments string.
        text = " ".join(
            tc.get("function", {}).get("arguments", "") for tc in msg["tool_calls"]
        )
    return {
        "text": text,
        "completion_tokens": (out.get("usage") or {}).get("completion_tokens"),
        "finish_reason": choice.get("finish_reason"),
        "raw_choice_keys": sorted(choice.keys()),
    }


def detect(det: str, text: str) -> dict:
    return _post(f"{det}/detect", {"text": text}, timeout=180.0)


# --------------------------------------------------------------------------- helpers


def load_prompts(path: Path, n: int) -> list[str]:
    lines = [p.strip() for p in path.read_text().splitlines() if p.strip()]
    if not lines:
        raise SystemExit(f"no prompts in {path}")
    out = []
    while len(out) < n:
        out.extend(lines)
    return out[:n]


def summarise(rows: list[dict], threshold: float = 0.01) -> dict:
    """Detection summary over rows that each carry a `detection` dict."""
    dets = [r["detection"] for r in rows if r.get("detection")]
    if not dets:
        return {"n": 0}
    flagged = sum(1 for d in dets if d["is_watermarked"])
    pvals = sorted(d["p_value"] for d in dets)
    scored = [d["num_scored_tokens"] for d in dets]
    return {
        "n": len(dets),
        "flagged": flagged,
        "rate": round(flagged / len(dets), 4),
        "p_median": pvals[len(pvals) // 2],
        "p_min": pvals[0],
        "p_max": pvals[-1],
        "scored_tokens_median": sorted(scored)[len(scored) // 2],
        "threshold": threshold,
    }


def emit(out_path: Path, payload: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=1))
    print(f"\nwrote {out_path}")


# --------------------------------------------------------------------------- length sweep


def cmd_length_sweep(args: argparse.Namespace) -> int:
    lengths = [int(x) for x in args.lengths.split(",")]
    temps = [float(x) for x in args.temperatures.split(",")]
    prompts = load_prompts(args.prompts, args.n)
    records: list[dict] = []

    print(f"length sweep: lengths={lengths} temperatures={temps} n={args.n} per cell")
    for temp in temps:
        for length in lengths:
            t0 = time.time()
            cell: list[dict] = []
            for i, prompt in enumerate(prompts):
                # Ask for a length-appropriate answer so the model does not simply
                # stop early: short cells get an explicit brevity instruction.
                steer = (
                    "Answer in one or two sentences. " if length <= 100
                    else "Answer in a short paragraph. " if length <= 200
                    else ""
                )
                for arm, wm in (("watermarked", True), ("opt-out", False)):
                    g = generate(
                        args.vllm_url, args.model, steer + prompt,
                        max_tokens=length, temperature=temp,
                        top_p=args.top_p, watermarking=wm,
                    )
                    if not g["text"].strip():
                        continue
                    d = detect(args.detector_url, g["text"])
                    cell.append({
                        "arm": arm, "prompt_index": i, "target_length": length,
                        "temperature": temp, "completion_tokens": g["completion_tokens"],
                        "finish_reason": g["finish_reason"],
                        "text": g["text"], "detection": d,
                    })
            wm_rows = [r for r in cell if r["arm"] == "watermarked"]
            off_rows = [r for r in cell if r["arm"] == "opt-out"]
            s_on, s_off = summarise(wm_rows), summarise(off_rows)
            print(
                f"  T={temp} len<={length:<4} "
                f"TPR {s_on.get('rate', 0):.3f} ({s_on.get('flagged',0)}/{s_on.get('n',0)}) "
                f"median p={s_on.get('p_median', float('nan')):.2e}  "
                f"| opt-out FPR {s_off.get('rate', 0):.3f}  "
                f"median scored tokens {s_on.get('scored_tokens_median','?')}  "
                f"[{time.time()-t0:.0f}s]"
            )
            records.extend(cell)

    emit(args.out, {
        "experiment": "length-sweep", "model": args.model,
        "lengths": lengths, "temperatures": temps, "n_per_cell": args.n,
        "top_p": args.top_p, "records": records,
    })
    return 0


# --------------------------------------------------------------------------- robustness

_WORD = re.compile(r"\S+")


def _edit_truncate(text: str, frac: float, rng: random.Random) -> str:
    words = text.split()
    keep = max(1, int(len(words) * frac))
    return " ".join(words[:keep])


def _edit_substitute(text: str, frac: float, rng: random.Random) -> str:
    """Replace a fraction of words with other words drawn from the same text."""
    words = text.split()
    if len(words) < 4:
        return text
    pool = [w for w in words if len(w) > 3] or words
    k = max(1, int(len(words) * frac))
    idx = rng.sample(range(len(words)), min(k, len(words)))
    out = list(words)
    for i in idx:
        out[i] = rng.choice(pool)
    return " ".join(out)


def _edit_delete(text: str, frac: float, rng: random.Random) -> str:
    words = text.split()
    if len(words) < 4:
        return text
    k = max(1, int(len(words) * frac))
    idx = set(rng.sample(range(len(words)), min(k, len(words) - 1)))
    return " ".join(w for i, w in enumerate(words) if i not in idx)


def cmd_robustness(args: argparse.Namespace) -> int:
    rng = random.Random(args.seed)
    prompts = load_prompts(args.prompts, args.n)
    fracs = [float(x) for x in args.fractions.split(",")]
    records: list[dict] = []

    print(f"robustness: n={args.n} prompts, edit fractions={fracs}, "
          f"paraphrase={'on' if not args.no_paraphrase else 'off'}")
    for i, prompt in enumerate(prompts):
        g = generate(
            args.vllm_url, args.model, prompt,
            max_tokens=args.max_tokens, temperature=args.temperature,
            top_p=args.top_p, watermarking=True,
        )
        original = g["text"]
        if not original.strip():
            continue
        base_det = detect(args.detector_url, original)
        records.append({
            "prompt_index": i, "transform": "none", "fraction": 0.0,
            "text": original, "detection": base_det,
            "completion_tokens": g["completion_tokens"],
        })

        for frac in fracs:
            for name, fn in (
                ("truncate", _edit_truncate),
                ("substitute", _edit_substitute),
                ("delete", _edit_delete),
            ):
                edited = fn(original, frac if name != "truncate" else 1.0 - frac, rng)
                if not edited.strip():
                    continue
                records.append({
                    "prompt_index": i, "transform": name, "fraction": frac,
                    "text": edited, "detection": detect(args.detector_url, edited),
                })

        if not args.no_paraphrase:
            # Paraphrase with the SAME model but watermarking off, so the
            # paraphrase itself carries no mark. This is the realistic attack:
            # an adversary rewrites the text with any model they control.
            para = generate(
                args.vllm_url, args.model,
                "Rewrite the following text completely in your own words. "
                "Keep all the information and roughly the same length, but change "
                "the wording and sentence structure throughout. Output only the "
                "rewritten text.\n\n" + original,
                max_tokens=args.max_tokens + 128,
                temperature=args.temperature, top_p=args.top_p,
                watermarking=False,
            )
            if para["text"].strip():
                records.append({
                    "prompt_index": i, "transform": "paraphrase", "fraction": 1.0,
                    "text": para["text"], "detection": detect(args.detector_url, para["text"]),
                    "completion_tokens": para["completion_tokens"],
                })
        print(f"  [{i+1}/{len(prompts)}] base p={base_det['p_value']:.2e} "
              f"({'detected' if base_det['is_watermarked'] else 'MISSED'})")

    print("\n| transform | fraction | n | detected | rate | median p |")
    print("|---|---:|---:|---:|---:|---:|")
    keys = sorted({(r["transform"], r["fraction"]) for r in records},
                  key=lambda k: (k[0], k[1]))
    for tname, frac in keys:
        rows = [r for r in records if r["transform"] == tname and r["fraction"] == frac]
        s = summarise(rows)
        print(f"| {tname} | {frac:.2f} | {s['n']} | {s['flagged']} | "
              f"{s['rate']:.3f} | {s['p_median']:.2e} |")

    emit(args.out, {
        "experiment": "robustness", "model": args.model, "seed": args.seed,
        "fractions": fracs, "temperature": args.temperature,
        "max_tokens": args.max_tokens, "records": records,
    })
    return 0


# --------------------------------------------------------------------------- modes

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "write_summary",
        "description": "Store a long written summary of a topic.",
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "A detailed multi-paragraph summary, at least 200 words.",
                }
            },
            "required": ["summary"],
        },
    },
}

JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "explanation": {"type": "string"},
        "key_points": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["explanation", "key_points"],
}


def cmd_modes(args: argparse.Namespace) -> int:
    prompt = ("Explain how tectonic plates move and why their motion causes "
              "earthquakes along certain faults.")
    T, P = args.temperature, args.top_p
    modes: list[tuple[str, str, dict]] = [
        ("baseline sampling", "watermarked default path",
         dict(temperature=T, top_p=P)),
        ("per-request opt-out", 'watermarking: false',
         dict(temperature=T, top_p=P, watermarking=False)),
        ("greedy temperature=0", "expected: NOT watermarked",
         dict(temperature=0.0, top_p=P)),
        ("top_k=1", "expected: NOT watermarked (degenerate sampling)",
         dict(temperature=T, top_p=P, top_k=1)),
        ("low temperature 0.2", "expected: weak signal",
         dict(temperature=0.2, top_p=P)),
        ("structured JSON output", "guided decoding narrows the sampler",
         dict(temperature=T, top_p=P,
              extra={"response_format": {
                  "type": "json_schema",
                  "json_schema": {"name": "explanation", "schema": JSON_SCHEMA},
              }})),
        ("tool call arguments", "sampled inside a function-call string",
         dict(temperature=T, top_p=P,
              extra={"tools": [WEATHER_TOOL],
                     "tool_choice": {"type": "function",
                                     "function": {"name": "write_summary"}}})),
        ("n=2 (second choice)", "multiple completions per request",
         dict(temperature=T, top_p=P, extra={"n": 2})),
        ("beam search over HTTP", "claimed: NOT rejected and NOT watermarked",
         dict(temperature=T, top_p=P,
              extra={"use_beam_search": True, "best_of": 4})),
        ("min_p 0.9 (near-degenerate)", "very few candidate tokens survive",
         dict(temperature=T, top_p=P, extra={"min_p": 0.9})),
        ("logit_bias forcing", "raw-logit manipulation before the draw",
         dict(temperature=T, top_p=P,
              extra={"logit_bias": {"785": 100.0, "1782": 100.0}})),
    ]

    records = []
    print(f"generation-mode matrix (model {args.model}, max_tokens {args.max_tokens})\n")
    print("| mode | tokens | scored | p-value | detector verdict | note |")
    print("|---|---:|---:|---:|---|---|")
    for name, note, kw in modes:
        extra = kw.pop("extra", None)
        try:
            g = generate(args.vllm_url, args.model, prompt,
                         max_tokens=args.max_tokens, extra=extra, **kw)
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")[:160].replace("\n", " ")
            print(f"| {name} | - | - | - | REQUEST REJECTED | HTTP {e.code}: {body} |")
            records.append({"mode": name, "error": f"HTTP {e.code}", "body": body})
            continue
        text = g["text"]
        if not text.strip():
            print(f"| {name} | 0 | - | - | EMPTY RESPONSE | {note} |")
            records.append({"mode": name, "empty": True})
            continue
        d = detect(args.detector_url, text)
        verdict = "WATERMARKED" if d["is_watermarked"] else "not detected"
        print(f"| {name} | {g['completion_tokens']} | {d['num_scored_tokens']} | "
              f"{d['p_value']:.2e} | {verdict} | {note} |")
        records.append({"mode": name, "note": note, "text": text,
                        "completion_tokens": g["completion_tokens"], "detection": d})

    # Streaming is a separate path: assemble the deltas and detect the result.
    try:
        payload = {"model": args.model,
                   "messages": [{"role": "user", "content": prompt}],
                   "max_tokens": args.max_tokens, "temperature": T, "top_p": P,
                   "stream": True}
        req = urllib.request.Request(
            f"{args.vllm_url}/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        chunks = []
        with urllib.request.urlopen(req, timeout=600, context=SSL_CONTEXT) as resp:
            for raw in resp:
                line = raw.decode().strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                obj = json.loads(line[6:])
                delta = (obj["choices"][0].get("delta") or {}).get("content")
                if delta:
                    chunks.append(delta)
        text = "".join(chunks)
        if text.strip():
            d = detect(args.detector_url, text)
            print(f"| streaming | ~{len(chunks)} deltas | {d['num_scored_tokens']} | "
                  f"{d['p_value']:.2e} | "
                  f"{'WATERMARKED' if d['is_watermarked'] else 'not detected'} | "
                  f"reassembled from SSE deltas |")
            records.append({"mode": "streaming", "text": text, "detection": d})
    except Exception as e:  # noqa: BLE001 - report, do not abort the matrix
        print(f"| streaming | - | - | - | ERROR | {type(e).__name__}: {e} |")
        records.append({"mode": "streaming", "error": str(e)})

    emit(args.out, {"experiment": "modes", "model": args.model,
                    "temperature": T, "top_p": P,
                    "max_tokens": args.max_tokens, "records": records})
    return 0


# --------------------------------------------------------------------------- cli


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vllm-url", required=True)
    ap.add_argument("--detector-url", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--cacert", type=Path, default=None)
    ap.add_argument("--insecure", action="store_true")
    ap.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--out", type=Path, required=True)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("length-sweep", help="detection power vs output length")
    s.add_argument("--lengths", default="50,100,150,200,300,512")
    s.add_argument("--temperatures", default="1.0,0.7")
    s.add_argument("--n", type=int, default=15)
    s.set_defaults(func=cmd_length_sweep)

    s = sub.add_parser("robustness", help="survival under edits and paraphrase")
    s.add_argument("--n", type=int, default=20)
    s.add_argument("--max-tokens", type=int, default=400)
    s.add_argument("--temperature", type=float, default=1.0)
    s.add_argument("--fractions", default="0.05,0.10,0.25,0.50")
    s.add_argument("--seed", type=int, default=20260921)
    s.add_argument("--no-paraphrase", action="store_true")
    s.set_defaults(func=cmd_robustness)

    s = sub.add_parser("modes", help="which generation modes are watermarked")
    s.add_argument("--max-tokens", type=int, default=400)
    s.add_argument("--temperature", type=float, default=1.0)
    s.set_defaults(func=cmd_modes)

    args = ap.parse_args()

    global SSL_CONTEXT
    if args.cacert:
        SSL_CONTEXT = ssl.create_default_context(cafile=str(args.cacert))
    elif args.insecure:
        SSL_CONTEXT = ssl.create_default_context()
        SSL_CONTEXT.check_hostname = False
        SSL_CONTEXT.verify_mode = ssl.CERT_NONE

    args.vllm_url = args.vllm_url.rstrip("/")
    args.detector_url = args.detector_url.rstrip("/")
    if not args.model:
        args.model = _get(f"{args.vllm_url}/v1/models")["data"][0]["id"]
        print(f"model (auto): {args.model}")
    return args.func(args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except urllib.error.HTTPError as e:
        sys.stderr.write(f"HTTP {e.code} from {e.url}: "
                         f"{e.read().decode(errors='replace')[:500]}\n")
        sys.exit(1)
