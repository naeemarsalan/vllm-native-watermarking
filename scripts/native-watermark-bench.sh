#!/usr/bin/env bash
# Paired serving-overhead benchmark for vLLM's NATIVE Gumbel-max watermark.
#
# Why this exists: as of 2026-09-21 the only published serving numbers for the
# native watermark are upstream H100 runs on Qwen3.5-2B/27B (RFC #53916, PRs
# #56233 and #56122). Nothing has been measured on any other GPU, model or
# workload. This script produces that measurement for YOUR hardware and model.
#
# What it measures. One engine, started with --watermark-config, serving two
# interleaved conditions:
#   on  = watermarked (the engine default)
#   off = the per-request opt-out, "watermarking": false
# Because both conditions hit the same warm engine, the difference isolates the
# watermark sampling path from every other cost. Trials alternate on/off so that
# drift (thermal, neighbours, cache state) hits both arms equally.
#
# What it does NOT measure. The cost of enabling the feature at all: an engine
# started WITHOUT --watermark-config also skips the watermark warmup and keeps
# Model Runner V2 selection free of the watermark override. To measure that,
# run this script against a second deployment that omits --watermark-config and
# compare the `off` arms. Pass --cross-engine-note to record that you did.
#
# GPU utilization is sampled with nvidia-smi inside the serving pod, if
# --gpu-pod is given. Without it, throughput and latency are still measured.
#
# Usage:
#   scripts/native-watermark-bench.sh \
#       --base-url https://vllm-watermark-watermark-demo.apps.<domain>/v1 \
#       --model Qwen/Qwen2.5-1.5B-Instruct \
#       --cacert cluster/router-ca.crt \
#       --trials 3 --n 200 --concurrency 8 --max-tokens 300 \
#       --gpu-pod deploy/vllm-watermark --namespace watermark-demo \
#       --out-dir benchmarks/results/native-$(date -u +%Y%m%dT%H%M%SZ)
#
# Output: one JSON per arm per trial, a gpu-samples CSV if --gpu-pod was given,
# and summary.txt with the paired deltas. Feed the directory to
# benchmarks/analyze_serving_profile.py for the full statistical treatment.
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
base_url=""; model=""; cacert=""; gpu_pod=""; namespace="watermark-demo"
trials=3; n=200; concurrency=8; max_tokens=300; temperature=1.0; top_p=1.0
out_dir="${repo_dir}/benchmarks/results/native-bench"
cross_engine_note=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-url) base_url=$2; shift 2;;
    --model) model=$2; shift 2;;
    --cacert) cacert=$2; shift 2;;
    --gpu-pod) gpu_pod=$2; shift 2;;
    --namespace) namespace=$2; shift 2;;
    --trials) trials=$2; shift 2;;
    --n) n=$2; shift 2;;
    --concurrency) concurrency=$2; shift 2;;
    --max-tokens) max_tokens=$2; shift 2;;
    --temperature) temperature=$2; shift 2;;
    --top-p) top_p=$2; shift 2;;
    --out-dir) out_dir=$2; shift 2;;
    --cross-engine-note) cross_engine_note=$2; shift 2;;
    -h|--help) sed -n '2,40p' "${BASH_SOURCE[0]}"; exit 0;;
    *) echo "unknown argument: $1" >&2; exit 2;;
  esac
done

[[ -n "$base_url" && -n "$model" ]] || { echo "--base-url and --model are required" >&2; exit 2; }
[[ "$trials" =~ ^[0-9]+$ && "$trials" -ge 1 ]] || { echo "--trials must be a positive integer" >&2; exit 2; }

mkdir -p "$out_dir"
export OPENAI_BASE_URL="$base_url"
[[ -n "$cacert" ]] && export REQUESTS_CA_BUNDLE="$(cd "$(dirname "$cacert")" && pwd)/$(basename "$cacert")"

echo "# native watermark paired serving benchmark" | tee "$out_dir/summary.txt"
{
  echo "date_utc: $(date -u +%FT%TZ)"
  echo "base_url: $base_url"
  echo "model: $model"
  echo "trials: $trials  n: $n  concurrency: $concurrency  max_tokens: $max_tokens"
  echo "temperature: $temperature  top_p: $top_p"
  [[ -n "$cross_engine_note" ]] && echo "cross_engine_note: $cross_engine_note"
} | tee -a "$out_dir/summary.txt"

# Sample GPU utilization for the duration, if we can reach the serving pod.
gpu_sampler_pid=""
if [[ -n "$gpu_pod" ]]; then
  echo "timestamp,utilization_gpu_pct,utilization_mem_pct,memory_used_mib,power_w" > "$out_dir/gpu-samples.csv"
  (
    while true; do
      oc -n "$namespace" exec "$gpu_pod" -- nvidia-smi \
        --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw \
        --format=csv,noheader,nounits 2>/dev/null >> "$out_dir/gpu-samples.csv" || true
      sleep 5
    done
  ) &
  gpu_sampler_pid=$!
  trap '[[ -n "$gpu_sampler_pid" ]] && kill "$gpu_sampler_pid" 2>/dev/null || true' EXIT
  echo "sampling GPU every 5s from pod $gpu_pod" | tee -a "$out_dir/summary.txt"
fi

run_arm() {
  local arm=$1 trial=$2 extra=$3
  local out="$out_dir/${arm}-trial${trial}.json"
  echo "  [trial $trial] arm=$arm"
  python3 "${repo_dir}/benchmarks/bench_serving.py" \
    --model "$model" \
    --prompts-file "${repo_dir}/benchmarks/prompts.txt" \
    --n "$n" --max-tokens "$max_tokens" \
    --temperature "$temperature" --concurrency "$concurrency" \
    --extra-body "$extra" \
    --condition "$arm" --trial "$trial" \
    --out "$out" >/dev/null
}

for t in $(seq 1 "$trials"); do
  # Alternate the order each trial so neither arm is systematically first.
  if (( t % 2 == 1 )); then
    run_arm on  "$t" "{\"top_p\": $top_p}"
    run_arm off "$t" "{\"top_p\": $top_p, \"watermarking\": false}"
  else
    run_arm off "$t" "{\"top_p\": $top_p, \"watermarking\": false}"
    run_arm on  "$t" "{\"top_p\": $top_p}"
  fi
done

[[ -n "$gpu_sampler_pid" ]] && { kill "$gpu_sampler_pid" 2>/dev/null || true; gpu_sampler_pid=""; }

python3 - "$out_dir" <<'PY' | tee -a "$out_dir/summary.txt"
import json, statistics, sys, glob, os
d = sys.argv[1]
def load(arm):
    rows = []
    for p in sorted(glob.glob(os.path.join(d, f"{arm}-trial*.json"))):
        with open(p) as fh:
            rows.append(json.load(fh))
    return rows
on, off = load("on"), load("off")
if not on or not off:
    print("\nno paired results found"); raise SystemExit(0)

def pick(r, *path):
    cur = r
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur

def series(rows, *path):
    vals = [pick(r, *path) for r in rows]
    return [v for v in vals if isinstance(v, (int, float))]

print("\n## paired results (same engine, per-request opt-out as the baseline)\n")
print("| metric | watermark on | watermark off (opt-out) | delta |")
print("|---|---:|---:|---:|")
METRICS = [
    ("output tokens/s", ("summary", "output_tokens_per_s"), True),
    ("wall time (s)", ("summary", "wall_time_s"), False),
    ("latency mean (s)", ("summary", "latency_s", "mean"), False),
    ("latency p50 (s)", ("summary", "latency_s", "p50"), False),
    ("latency p95 (s)", ("summary", "latency_s", "p95"), False),
    ("latency p99 (s)", ("summary", "latency_s", "p99"), False),
]
for label, path, higher_better in METRICS:
    a, b = series(on, *path), series(off, *path)
    if not a or not b:
        continue
    ma, mb = statistics.median(a), statistics.median(b)
    delta = "n/a" if mb == 0 else f"{(ma - mb) / mb * 100:+.2f}%"
    print(f"| {label} | {ma:.3f} | {mb:.3f} | {delta} |")

failed = sum((pick(r, "summary", "requests_failed") or 0) for r in on + off)
print(f"\ntrials per arm: on={len(on)} off={len(off)} (medians across trials); failed requests: {failed}")
print("delta is watermark-on relative to the opt-out arm; negative on tokens/s means the")
print("watermark cost throughput, negative on latency means it was faster (i.e. noise).")

g = os.path.join(d, "gpu-samples.csv")
if os.path.exists(g):
    rows = [l.strip().split(",") for l in open(g).read().splitlines()[1:] if l.strip()]
    util = [float(r[1]) for r in rows if len(r) > 1 and r[1].strip().replace('.','',1).isdigit()]
    mem = [float(r[3]) for r in rows if len(r) > 3 and r[3].strip().replace('.','',1).isdigit()]
    if util:
        print(f"\nGPU utilization over the whole run: mean {statistics.mean(util):.1f}%, "
              f"max {max(util):.0f}%, samples {len(util)}")
    if mem:
        print(f"GPU memory used: mean {statistics.mean(mem):.0f} MiB, max {max(mem):.0f} MiB")
    print("note: samples span both arms; use the per-arm timestamps in the JSON to split them.")
PY

echo
echo "results in $out_dir"
