#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""End-to-end demo for vLLM's native (Gumbel-max) text watermarking.

For each prompt it asks the vLLM server for two completions -- one with the
engine's watermark (the default) and one with the per-request opt-out
(`"watermarking": false`) -- sends both texts to the detector, and prints the
detector's verdicts side by side. Optionally it also scores human-written text.

Only the Python standard library is used, so it runs from any machine that can
reach the two HTTP endpoints (Routes or `oc port-forward`).

Usage (see docs/guide-native-watermarking.md):

    python3 scripts/native-watermark-demo.py \
        --vllm-url https://vllm-watermark-watermark-demo.apps.<cluster-domain> \
        --detector-url https://watermark-detector-watermark-demo.apps.<cluster-domain> \
        --n 10 --max-tokens 300 --temperature 1.0 --top-p 1.0 \
        --cacert cluster/router-ca.crt
"""

from __future__ import annotations

import argparse
import json
import ssl
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_PROMPTS = Path(__file__).resolve().parent.parent / "benchmarks" / "prompts.txt"

# TLS context for the OpenShift Routes. A fresh cluster serves a self-signed
# ingress certificate: pass --cacert <router CA> (preferred) or --insecure.
SSL_CONTEXT: ssl.SSLContext | None = None


def http_json(url: str, payload: dict, timeout: float = 600.0) -> dict:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout, context=SSL_CONTEXT) as resp:
        return json.load(resp)


def http_get(url: str, timeout: float = 30.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout, context=SSL_CONTEXT) as resp:
        return json.load(resp)


def generate(
    base: str,
    model: str,
    prompt: str,
    *,
    watermarking: bool,
    max_tokens: int,
    temperature: float | None,
    top_p: float | None,
    seed: int | None,
) -> tuple[str, int]:
    payload: dict = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        # The request field that switches the engine's watermark off. It defaults
        # to true whenever the engine was started with --watermark-config; a
        # production gateway must strip or deny it for untrusted callers.
        "watermarking": watermarking,
    }
    if temperature is not None:
        payload["temperature"] = temperature
    if top_p is not None:
        payload["top_p"] = top_p
    if seed is not None:
        payload["seed"] = seed
    out = http_json(f"{base}/v1/chat/completions", payload)
    choice = out["choices"][0]
    return choice["message"]["content"], out["usage"]["completion_tokens"]


def detect(base: str, text: str) -> dict:
    return http_json(f"{base}/detect", {"text": text}, timeout=120.0)


def fmt_row(label: str, tokens: int, res: dict) -> str:
    verdict = "WATERMARKED" if res["is_watermarked"] else "not detected"
    return (
        f"  {label:<14} tokens={tokens:>4}  scored={res['num_scored_tokens']:>4}  "
        f"score={res['score']:>9.2f}  p={res['p_value']:.3e}  -> {verdict}"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vllm-url", required=True, help="base URL of the vLLM OpenAI-compatible server")
    ap.add_argument("--detector-url", required=True, help="base URL of the watermark detector")
    ap.add_argument("--model", default=None, help="model name; default: first model served")
    ap.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    ap.add_argument("--n", type=int, default=10, help="number of prompts to use")
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="sampling temperature; 0 disables the watermark entirely (pass -1 to use the model's default)")
    ap.add_argument("--top-p", type=float, default=1.0, help="pass -1 to use the model's default")
    ap.add_argument("--seed", type=int, default=None,
                    help="OpenAI-style request seed; steers only positions sampled without the watermark "
                         "(opt-out requests, and repeated-context positions of watermarked answers); it never "
                         "changes a watermarked token, which depends only on the key and the preceding context")
    ap.add_argument("--human-text", type=Path, default=None,
                    help="optional file with human-written text to score as a negative control")
    ap.add_argument("--human-jsonl", type=Path, default=None,
                    help="optional JSONL ({\"text\": ...} per line) of human-written passages; "
                         "the first --human-n are scored to estimate the false-positive rate")
    ap.add_argument("--human-n", type=int, default=50)
    ap.add_argument("--json-out", type=Path, default=None, help="write all requests/responses/verdicts here")
    ap.add_argument("--cacert", type=Path, default=None,
                    help="CA certificate that signed the Routes (e.g. the OpenShift ingress router CA)")
    ap.add_argument("--insecure", action="store_true", help="skip TLS verification (quick tests only)")
    args = ap.parse_args()
    # Fail fast on missing inputs: the human-corpus check runs after the GPU loop, and a
    # FileNotFoundError there would discard the completed generation/detection records.
    for opt, path in (("--prompts", args.prompts), ("--human-jsonl", args.human_jsonl), ("--human-text", args.human_text)):
        if path is not None and not path.is_file():
            hint = " (generate it with benchmarks/fetch_human_corpus.py or omit the flag)" if opt == "--human-jsonl" else ""
            ap.error(f"{opt} {path}: file not found{hint}")

    global SSL_CONTEXT
    if args.cacert:
        SSL_CONTEXT = ssl.create_default_context(cafile=str(args.cacert))
    elif args.insecure:
        SSL_CONTEXT = ssl.create_default_context()
        SSL_CONTEXT.check_hostname = False
        SSL_CONTEXT.verify_mode = ssl.CERT_NONE

    vllm = args.vllm_url.rstrip("/")
    det = args.detector_url.rstrip("/")
    temperature = None if args.temperature < 0 else args.temperature
    top_p = None if args.top_p < 0 else args.top_p

    models = http_get(f"{vllm}/v1/models")["data"]
    model = args.model or models[0]["id"]
    print(f"vLLM: {vllm}  model: {model}")
    print(f"detector: {det}")
    print(f"sampling: temperature={temperature if temperature is not None else 'model default'} "
          f"top_p={top_p if top_p is not None else 'model default'} max_tokens={args.max_tokens}\n")

    prompts = [p.strip() for p in args.prompts.read_text().splitlines() if p.strip()][: args.n]
    records = []
    tp = fp = 0
    t0 = time.time()
    for i, prompt in enumerate(prompts, 1):
        print(f"[{i}/{len(prompts)}] {prompt[:90]}")
        row = {"prompt": prompt}
        for label, wm in (("watermarked", True), ("opt-out", False)):
            text, ntok = generate(
                vllm, model, prompt, watermarking=wm, max_tokens=args.max_tokens,
                temperature=temperature, top_p=top_p, seed=args.seed,
            )
            res = detect(det, text)
            print(fmt_row(label, ntok, res))
            row[label] = {"text": text, "completion_tokens": ntok, "detection": res}
            if wm and res["is_watermarked"]:
                tp += 1
            if not wm and res["is_watermarked"]:
                fp += 1
        records.append(row)

    print()
    n = len(prompts)
    print(f"summary over {n} prompts: watermarked detected {tp}/{n} (TPR {tp / n:.3f}); "
          f"opt-out flagged {fp}/{n} (FPR {fp / n:.3f}); elapsed {time.time() - t0:.0f}s")

    human = None
    if args.human_text:
        text = args.human_text.read_text()
        human = detect(det, text)
        print("\nhuman-written control:")
        print(fmt_row("human", -1, human).replace("tokens=  -1  ", ""))

    human_batch = None
    if args.human_jsonl:
        texts = [json.loads(line)["text"] for line in args.human_jsonl.read_text().splitlines() if line.strip()]
        texts = texts[: args.human_n]
        flagged = 0
        pvals = []
        for text in texts:
            res = detect(det, text)
            pvals.append(res["p_value"])
            flagged += int(res["is_watermarked"])
        human_batch = {"n": len(texts), "flagged": flagged, "p_values": pvals}
        print(f"\nhuman-written passages ({args.human_jsonl.name}): flagged {flagged}/{len(texts)} "
              f"(FPR {flagged / max(1, len(texts)):.3f}); min p={min(pvals):.3e} median p={sorted(pvals)[len(pvals)//2]:.3e}")

    if args.json_out:
        args.json_out.write_text(json.dumps(
            {"model": model, "temperature": temperature, "top_p": top_p,
             "max_tokens": args.max_tokens, "records": records, "human": human,
             "human_batch": human_batch}, indent=1))
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except urllib.error.HTTPError as e:
        sys.stderr.write(f"HTTP {e.code} from {e.url}: {e.read().decode(errors='replace')[:500]}\n")
        sys.exit(1)
