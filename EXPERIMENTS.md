# EXPERIMENTS.md — append-only run log

Every entry: date, environment, command(s), raw output (condensed where huge, never
altered), and what it proves. Verification tags per [`docs/facts.md`](docs/facts.md).
Secrets are redacted as `[REDACTED]`. Newest entries at the bottom.

---

## 2026-08-08 — Phase 0 infrastructure bring-up (ocp-ai cluster)

**Environment:** OpenShift 4.20 `ocp-ai` (api.ocp-ai.<redacted-sandbox-domain>), local
workstation `oc` with `KUBECONFIG=cluster/auth/kubeconfig`.

### GPU node scale-up

```
$ ./scripts/scale-gpu.sh 1
machineset.machine.openshift.io/ocp-ai-p9j4n-gpu-us-east-1a scaled
```

NFD + NVIDIA GPU operators were already installed (from `scripts/install-gpu-operators.sh`
during provisioning):

```
$ oc get csv -n openshift-nfd
nfd.4.20.0-202607290013   Node Feature Discovery Operator   4.20.0-202607290013   Succeeded
$ oc get csv -n nvidia-gpu-operator
gpu-operator-certified.v26.3.3   NVIDIA GPU Operator   26.3.3   Succeeded
```

### Bug found + fixed: MachineSet never labels the *node* (EXECUTED)

The g5.xlarge machine reached `Running` in ~5 min and the node `ip-10-0-5-194.ec2.internal`
joined and went `Ready`, but a 30-minute wait for a node with label
`node-role.kubernetes.io/gpu` timed out. Root cause: `scripts/create-gpu-machineset.sh`
set the label only on `.spec.template.metadata.labels` (the **Machine** object), not on
`.spec.template.spec.metadata.labels` (propagated to the **Node**). Confirmed:

```
$ oc -n openshift-machine-api get machineset ocp-ai-p9j4n-gpu-us-east-1a -o jsonpath='{.spec.template.spec.metadata}'
{}
```

Fix applied (all three, same session):

```
$ oc label node ip-10-0-5-194.ec2.internal node-role.kubernetes.io/gpu="" --overwrite
node/ip-10-0-5-194.ec2.internal labeled
$ oc -n openshift-machine-api patch machineset ocp-ai-p9j4n-gpu-us-east-1a --type merge \
    -p '{"spec":{"template":{"spec":{"metadata":{"labels":{"node-role.kubernetes.io/gpu":""}}}}}}'
machineset.machine.openshift.io/ocp-ai-p9j4n-gpu-us-east-1a patched
```

plus the same line added to `scripts/create-gpu-machineset.sh` (this commit).

### GPU node verified operational (EXECUTED)

```
$ oc get node ip-10-0-5-194.ec2.internal ... (condensed)
instance-type: g5.xlarge
allocatable nvidia.com/gpu: 1
taints: None
nvidia.com/gpu.product: NVIDIA-A10G, nvidia.com/gpu.memory: 23028 (MiB)
nvidia.com/cuda.driver-version.full: 580.126.20, cuda.runtime-version.full: 13.0
```

GPU operator stack all Running on the node; `nvidia-cuda-validator` pod **Completed**
(driver validation passed).

### Serving image pinned (OFFICIAL-SRC)

Target: vLLM v0.18.0 to match RHOAI 3.4 (facts C1). Docker Hub queried 2026-08-08:

```
$ curl -s https://hub.docker.com/v2/repositories/vllm/vllm-openai/tags/v0.18.0
name=v0.18.0
manifest-list digest: sha256:c32358ebfc115d56ade2acfdbcd00df5b115417dbd6006547c88f07e2b39de06
amd64 image digest:   sha256:96c7e88811a07030f27bc44cd71b9007258a15f130cfec2bb4ab057512238b05
```

Namespace `watermark` created; image pre-pull pod (pinned by manifest-list digest)
scheduled on the GPU node to overlap the multi-GB pull with remaining setup.

## 2026-08-08 — Phase 0 baseline serving + benchmark (EXECUTED)

**Environment:** pod `vllm-baseline` on `ip-10-0-5-194.ec2.internal` (g5.xlarge / A10G
24GB), image `vllm/vllm-openai@sha256:c32358eb…` (tag v0.18.0), engine log confirms
`Initializing a V1 LLM engine (v0.18.0)`, model `Qwen/Qwen2.5-0.5B-Instruct`
(`--max-model-len 4096 --gpu-memory-utilization 0.90`), dtype bfloat16.
Manifests: `deploy/phase0/`.

### Friction found + fixed: `VLLM_PORT` service-link collision (EXECUTED)

First start crashed at engine-core init:

```
ValueError: VLLM_PORT 'tcp://172.30.40.120:8000' appears to be a URI. This may be
caused by a Kubernetes service discovery issue,check the warning in:
https://docs.vllm.ai/en/stable/serving/env_vars.html
```

Cause: our Service is named `vllm`, so Kubernetes legacy service links inject
`VLLM_PORT=tcp://<clusterIP>:8000` into same-namespace pods, and vLLM reads
`VLLM_PORT` as its own config. Fix: `enableServiceLinks: false` on all phase0 pods
(committed). Also pre-empted: `HOME`/`XDG_CACHE_HOME` pointed at the emptyDir because
restricted-v2 SCC runs the container as an arbitrary UID with an unwritable default
`$HOME` (vLLM writes torch/triton caches under `$HOME`).

After the fix the pod went Ready in ~75s from container start (readiness probe
`/health`); `GET /v1/models` from the in-cluster bench pod returned the served model
(raw JSON reproduced in the raw-evidence addendum at the end of this file).

### Baseline benchmark (EXECUTED)

Command (in-cluster `bench` pod, python:3.12-slim + requests):

```
OPENAI_BASE_URL=http://vllm:8000/v1 python3 bench_serving.py \
  --model Qwen/Qwen2.5-0.5B-Instruct --prompts-file prompts.txt \
  --n 100 --max-tokens 256 --temperature 0.7 --concurrency 4 \
  --out baseline_results.json
```

Raw summary output:

```
=== Summary ===
requests: 100 ok / 0 failed / 100 total
wall time: 28.18s
total completion tokens: 25481
aggregate output tokens/sec: 904.35
latency (s): mean=1.124 p50=1.117 p95=1.126 p99=1.430 min=1.005 max=1.441
```

Full per-request data: `benchmarks/results/phase0_baseline_results.json`.

**Phase 0 acceptance: MET.** OpenAI-compatible endpoint answers; baseline
tokens/sec + p50/p95 recorded. Comparison figures for Phase 1 overhead:
**904.35 tok/s aggregate, p50 1.117s, p95 1.126s** (concurrency 4, 256 max tokens,
temperature 0.7, prompts `benchmarks/prompts.txt`).

## 2026-08-08 — vllm_watermark package: local test suite (EXECUTED)

**Environment:** local workstation, Python 3.14.4, torch 2.9.1+cu128 (CPU),
transformers 4.57.6.

```
$ /usr/bin/python3 -m pytest tests/ -q
34 passed in 329.54s (0:05:29)
```

Covers (see tests/test_kgw_equivalence.py, tests/test_processor_static.py):
- Greenlist equivalence vs transformers' WatermarkLogitsProcessor (exact set
  equality, 200 random prev_tokens × vocab 50257 and 151936, default hash key).
- Detector z-score equivalence vs transformers' WatermarkDetector scoring logic
  (rtol 1e-6, 50 random sequences, both ignore_repeated_ngrams modes).
- Keyed generation↔detection self-consistency with a 64-bit derived hash key:
  simulated 256-step generation z > 4; random sequences z < 1.5.
- V1 processor batch bookkeeping (add/remove/move incl. swap) against a stub of the
  fetched v0.18.0 interface; apply() row-selection math on CPU tensors.

**Upstream bug found by execution (STATIC consequence for others):** in
transformers 4.57.6, `WatermarkDetector`'s `ignore_repeated_ngrams=True` is a no-op:
`_score_ngrams_in_passage` builds `collections.Counter` over `torch.Tensor` rows,
whose hash is identity-based, so identical ngrams are never merged. Our port
implements the documented semantics (value-based dedup) — see
src/vllm_watermark/kgw/detector.py DEVIATION 1.

**Pipeline smoke (EXECUTED, local, no vLLM):** simulated KGW generation (delta-biased
sampling over random logits, Qwen2.5 tokenizer decode) through
benchmarks/analyze_detection.py: watermarked n=25 mean z=27.74 TPR=1.0;
unwatermarked n=25 mean z=0.56 FPR=0.0; human corpus n=150 mean z=0.19 FPR=0.0
(report: benchmarks/data/smoke_report.md — data files gitignored). This validates the
detector/analysis tooling only — NOT `vllm serve` (that is Phase 1's job).

## 2026-08-08 — End-of-session GPU scale-down (EXECUTED)

**Environment:** OpenShift 4.20 `ocp-ai`, local `oc` with
`KUBECONFIG=cluster/auth/kubeconfig`.

The live MachineSet check showed one desired/current/ready GPU node. Per the repository's
billable-resource rule, it was scaled down before the session ended:

```
$ ./scripts/scale-gpu.sh 0
machineset.machine.openshift.io/ocp-ai-p9j4n-gpu-us-east-1a scaled

$ oc -n openshift-machine-api get machineset ocp-ai-p9j4n-gpu-us-east-1a \
    -o custom-columns=NAME:.metadata.name,DESIRED:.spec.replicas,CURRENT:.status.replicas,READY:.status.readyReplicas \
    --no-headers
ocp-ai-p9j4n-gpu-us-east-1a   0     0     <none>
```

This proves the billable GPU MachineSet had reached desired/current replica count zero at
the end of the 2026-08-08 work. It does not add any watermarking or serving result.

## 2026-08-08 — Evidence qualification for the local pipeline smoke

The earlier “Pipeline smoke (EXECUTED, local, no vLLM)” paragraph records only a prose
summary. Its command, raw output, and referenced gitignored report were not preserved in
this repository. Under the repository's verification rules, those numerical rates do
**not** qualify as `EXECUTED` publication evidence. Fact B22 is therefore `OPEN` until a
fresh run records its environment, command, and raw output in this append-only log. This
qualification does not alter the separately preserved 34-test suite result.

## 2026-08-08 — Phase 1: KGW watermark end-to-end through `vllm serve` (EXECUTED — closes D1)

**Environment:** pod `vllm-watermark` (manifest `deploy/phase0/vllm-watermark-pod.yaml`),
image `vllm/vllm-openai@sha256:c32358eb…` (v0.18.0), engine `v0.18.0`, model
`Qwen/Qwen2.5-0.5B-Instruct` (vocab_size 151936), A10G. Plugin injected as a wheel
(pip install --no-deps --target; PYTHONPATH), loaded via
`--logits-processors vllm_watermark.kgw.processor:KGWLogitsProcessor` (log-verified in
`non-default args`). Key: k8s Secret → `WATERMARK_KEY` env, key_id `poc-2026-08`
(64-bit hash key derived via sha256; secret material in gitignored `cluster/`, never
committed or logged). KGW params: gamma 0.25, delta 2.0, lefthash, context_width 1.
`VLLM_WATERMARK_DEFAULT=on`.

### Per-request control + validation (EXECUTED)

Through `/v1/completions` from the in-cluster bench pod:
- default (no vllm_xargs) → watermarked; `{"watermark":"off"}` → not; `{"watermark":"on","watermark_key_id":"poc-2026-08"}` → watermarked.
- `{"watermark":"banana"}` → HTTP 400 `watermark must be 'on'/'off' (or a boolean), got 'banana'`
- `{"watermark_key_idz":"x"}` → HTTP 400 `Unknown watermark_* extra_args key(s) ['watermark_key_idz']…`
- `{"watermark_key_id":"no-such-key"}` → HTTP 400 `…not found among configured keys: ['poc-2026-08']`
(validate_params → ValueError → BadRequestError chain works exactly as the STATIC
api-notes predicted.)

### Detection statistics (EXECUTED)

Corpora generated through the server (120 wm-on / 120 wm-off at temp 0.7, 40 wm-on at
temp 0.0, 256 max tokens, prompts `benchmarks/prompts.txt`), plus 150 human Gutenberg
chunks. Scored locally (CPU detector, `ignore_repeated_ngrams=True`, z≥4.0):

| corpus | n | mean z | median | min | max | TPR@z≥4 | FPR@z≥4 |
|---|---|---|---|---|---|---|---|
| watermarked (T=0.7) | 120 | 21.35 | 21.67 | 15.01 | 25.17 | **1.000** | — |
| unwatermarked (T=0.7) | 120 | −0.08 | −0.19 | −1.68 | 2.58 | — | **0.000** |
| watermarked (T=0.0) | 40 | 20.60 | 21.05 | 13.09 | 23.34 | **1.000** | — |
| human (Gutenberg) | 150 | 0.03 | −0.04 | −2.69 | 3.13 | — | **0.000** |

Full report: `benchmarks/data/phase1_report.md` (data files gitignored; regenerate via
benchmarks/gen_corpus.py + analyze_detection.py). Detector CLI spot-checks: watermarked
sample z=25.40 (p=1.2e-142), human sample z=−1.30 — JSON outputs captured above in
session; commands: `python -m vllm_watermark.cli detect --model-tokenizer
Qwen/Qwen2.5-0.5B-Instruct --key-id poc-2026-08 --file <sample> --json`.

**Negative-test surprises, recorded honestly:**
- **Temperature 0 did NOT degrade KGW at delta 2.0 on this model** (mean z 20.6 vs
  21.3 at T=0.7). Greedy argmax still flips to green wherever the model's logit margin
  < delta. The literature-derived expectation (facts B18) holds for entropy-carried
  schemes/low delta; for KGW additive bias on this small model it does not. Quality
  impact of delta under greedy decoding remains unmeasured (Phase 5).
- **Structured output composes and retains signal:** 8 guided_json completions
  (temp 0.7) all detected — mean z 14.5, min 11.2 (vs ~21 free-form). These outputs
  contained prose-heavy JSON fields; stricter low-entropy schemas will do worse (B9
  ordering unchanged: grammar mask applies before our processor).

### Spec-decode incompatibility (EXECUTED — B7 upgraded)

Pod with `--speculative-config '{"method":"ngram","num_speculative_tokens":3,
"prompt_lookup_max":4}'` + `--logits-processors …` fails at engine start:

```
(EngineCore pid=70) ERROR …     raise ValueError(STR_SPEC_DEC_REJECTS_LOGITSPROCS)
(EngineCore pid=70) ValueError: Custom logits processors are not supported when speculative decoding is enabled.
```

Note: raised before the plugin module is even imported (the plugin wheel was not
installed in that pod) — the rejection precedes processor loading.

### Overhead (EXECUTED — D2 partially closed: one config measured)

Same protocol as Phase 0 (100 req, 256 tok, T=0.7, concurrency 4, in-cluster):

| configuration | agg. output tok/s | p50 | p95 | vs baseline |
|---|---|---|---|---|
| Phase 0 baseline (no plugin) | 904.35 | 1.117s | 1.126s | — |
| plugin loaded, watermark off | 914.62 | 1.114s | 1.119s | **~0% (noise)** |
| watermark on, no cache | 280.57 | 3.640s | 3.694s | **3.22× slower** |
| watermark on, LRU cache 1024 | 444.83 | 2.275s | 2.626s | **2.03× slower** |

Active-path cost is CPU `torch.randperm(151936)` (~7 ms/call measured locally,
`benchmarks/bench_greenlist.py`) once per watermarked row per decode step. The LRU
memo (pure memoization keyed `(hash_key, prev_token)` — bit-identical outputs, unit
test `test_greenlist_cache_identical_and_lru`) recovers ~59%. An 8192-entry cache
measured within noise of 1024 on partial data while the pod died mid-run (below) —
default stays 1024. Remaining overhead is a Phase 5 optimization target (candidates:
thread-pool across rows, int32 storage, pinned-memory async copies).

### Operational friction found by execution (all fixed in manifests)

1. **Liveness-probe kill under load:** with the 1s default probe timeout, sustained
   watermark-on load starved `/health` (CPU-bound greenlist work) → kubelet killed the
   pod mid-benchmark (exit 137, explicit `Killing` event). Fix: probe
   `timeoutSeconds` 5 (readiness) / 10 (liveness), liveness `initialDelaySeconds` 300.
2. **External GPU scale-down automation:** at ~03:00Z the sandbox scaled the GPU
   MachineSet to 0 (no ClusterAutoscaler/MachineAutoscaler exists; no in-cluster
   cronjob; machine-api events show drain+delete). The node — and the pod on it —
   vanished mid-session. Assume the g5 node can disappear at any hour boundary;
   re-scale with `./scripts/scale-gpu.sh 1` and re-inject. (Suspected trigger: sandbox
   cost automation, possibly keyed on the `openshift-ai-node=gpu` AWS tag.)

**Phase 1 acceptance: MET.** Clean statistical separation end-to-end through
`vllm serve` (TPR 1.000, FPR 0.000, human max z 3.13 < 4); overhead quantified
(table above); per-request control validated; negative tests executed verbatim.

## 2026-08-08 — CORRECTION: Phase 1 ran two KGW processor instances (effective delta ~4.0)

While wiring SynthID, we found by reading v0.18.0 source — and then REPRODUCED IN THE
SERVING POD — that `_load_custom_logitsprocs()` concatenates auto-loaded entry-point
plugins with `--logits-processors` FQCN classes **with no deduplication**:

```
>>> _load_custom_logitsprocs(["vllm_watermark.kgw.processor:KGWLogitsProcessor"])
['KGWLogitsProcessor', 'SynthIDLogitsProcessor', 'KGWLogitsProcessor']   # KGW twice
>>> _load_custom_logitsprocs([])
['KGWLogitsProcessor', 'SynthIDLogitsProcessor']                          # fix
```

Because the Phase 1 pod both installed the wheel (entry point) and passed the flag,
**every Phase 1 "watermark on" measurement ran two stacked KGW instances — effective
delta ≈ 4.0, and two apply() passes of overhead.** What this invalidates and what it
does not:

- Still valid: end-to-end mechanism proof (D1), per-request control + HTTP-400
  validation, plugin-off ≈ zero overhead, spec-decode error, all FPR numbers
  (controls had no active processor), the wheel-injection deploy flow.
- Superseded: watermarked z-distributions (were δ≈4), watermark-on overhead
  (was 2 passes), and the "temp-0 not degraded" B18 note (was at δ≈4).

Fix: manifests pass NO `--logits-processors` flag (entry points only — exactly one
instance of each processor; verified in-pod, output above). Corrected single-instance
measurements below. `docs/api-notes-vllm-v0.18.0.md` §8 documents the trap.

## 2026-08-08 — Phase 1 corrected + Phase 2 SynthID through `vllm serve` (closes D8)

**Environment:** same pod design (new GPU node `ip-10-0-0-85` after the sandbox's
external scale-down — see below), image v0.18.0 by digest, wheel with both processors
(entry points kgw + synthid), key `poc-2026-08` from Secret, VLLM_WATERMARK_DEFAULT=on,
scheme via `vllm_xargs watermark_scheme` (default kgw). SynthID: depth 30, ngram_len 5,
sampling table 2^16 seed 0, layer keys derived from the secret digest with canonical
label `vllm_watermark.synthid.core.SYNTHID_KEY_LABEL`.

### SynthID device-independence + GPU hot path (EXECUTED in pod, A10G)

- `g_values` GPU vs CPU: **0 mismatching trials / 20** (full vocab 151936, depth 30) —
  integer LCG + bit-exact moved table; CPU detection of GPU-generated text is sound.
- `process_scores_row`: **2.57 ms/call on A10G vs 290.98 ms/call CPU** (measured
  locally, clean run) — the device-native path is what makes SynthID servable
  (~113×). CPU cost is architectural: depth×vocab sequential reduction passes.

### Detection statistics (EXECUTED; 512-token corpora, T=0.7, n=120 each; controls:
120 unwatermarked + 150 human)

KGW single-instance δ=2.0 (report `benchmarks/data/phase2_kgw_fixed_report.md`):

| corpus | n | mean z | min z | TPR@z≥4 | FPR@z≥4 |
|---|---|---|---|---|---|
| KGW δ2 T=0.7 512tok | 120 | 13.41 | 5.72 | **1.000** | — |
| KGW δ2 T=0.0 256tok | 40 | 9.91 | 6.47 | **1.000** | — |
| unwatermarked | 120 | −0.08 | — | — | **0.000** |
| human | 150 | 0.03 (max 3.13) | — | — | **0.000** |

Corrected temp-0 note: at matched length (z scales ~√T; 13.41 at 512 ⇒ ~9.5 expected
at 256), temp-0's mean z 9.91 shows **no additional degradation** for KGW δ2 on this
model — the B18 literature expectation still did not materialize at the honest delta.
Quality cost of greedy+bias remains unmeasured (Phase 5).

SynthID (untrained scorers; reports `benchmarks/data/phase2_synthid_report_*.md`):

| scorer | watermarked mean z (min) | TPR@z≥4 | control FPR |
|---|---|---|---|
| mean | 22.51 (12.24) | **1.000** | **0.000** (n=270) |
| weighted mean | 23.61 (14.11) | **1.000** | **0.000** (n=270) |

**D8 decision data:** the untrained weighted-mean scorer already achieves perfect
separation at 512 tokens on this model; Bayesian-detector training (~10k matched
examples, generable with our harness in <1h GPU) is NOT required for PoC-grade
reliability. (200/256-token truncation table appended below when the comparison run
completes.)

### Overhead (EXECUTED; 100 req, 256 tok, T=0.7, conc 4, in-cluster)

| configuration | agg. tok/s | p50 | vs 904.35 baseline |
|---|---|---|---|
| both processors loaded, watermark off | 913.78 | 1.115s | ~0% |
| KGW on, single instance, LRU 1024 | **643.38** | 1.576s | **1.41×** |
| SynthID on, GPU path, depth 30 | **287.82** | 3.563s | **3.17×** |
| (superseded: KGW double-instance) | 444.83 | 2.275s | 2.03× |

### Operational note

The sandbox's external automation scaled the GPU MachineSet to 0 again during this
phase (~hourly pattern holds). The replacement node received the
`node-role.kubernetes.io/gpu` label automatically — the MachineSet fix from Phase 0
verified in production. GPU scaled to 0 at phase end (all remaining work is CPU-side).

**Phase 2 acceptance: MET** (SynthID generation+detection through `vllm serve` with
quantified reliability; comparison table + 200-token row pending the local scoring
run, appended below).

## 2026-08-08 — Phase 3: detector service + FMS GuardrailsOrchestrator end-to-end (closes D5's executable half)

**Context shift recorded first:** facts C11 (added to the register the same day by separate maintainer commits, OFFICIAL-SRC)
says RHOAI 3.4 labels FMS Guardrails **legacy** and points to NeMo Guardrails. We
validated the FMS/TrustyAI detector contract anyway (it is the documented, shipped,
executable detector interface today and what Phase 3 targets), and separately
assessed the NeMo fit (`docs/api-notes-nemo-guardrails.md`): NeMo has NO first-class
external-detector interface; the pattern is a custom `@action` POSTing to any URL —
our direct endpoint fits; no watermark rail exists in the NVIDIA-NeMo org; RHOAI's
`NemoGuardrails` CR mounts arbitrary ConfigMaps (custom actions permitted at CR
schema level; live-CR verification needs RHOAI — Phase 4).

**Build (EXECUTED):** detector image built on-cluster (BuildConfig docker-strategy,
binary source from a **staged context containing only detector/ + dist/** — never the
repo root, which holds gitignored live credentials; see deploy/phase3/README.md).
Pushed: `image-registry…/watermark/detector@sha256:6250a08b…`. Deployments:
`detector` (scheme env kgw) + `detector-synthid` (scheme env synthid) — same image.
Orchestrator: `quay.io/opendatahub/ta-guardrails-orchestrator:odh-3.4.2.git`
(self-reports **fms-guardrails-orchestr8 0.16.0** on :8034/health).

**Findings by execution:**
1. This orchestrator image **silently ignores** the 0.18.3+ `path_prefix` detector
   config field (rollout succeeds, routing unchanged) — server-side scheme authority
   therefore uses two Deployments with per-Deployment `WATERMARK_DETECTOR_SCHEME`.
2. First wiring attempt failed cross-scheme because both Services selected
   `app.kubernetes.io/component: detector` (label collision → round-robin across
   schemes). Fixed with distinct component labels; endpoints verified disjoint.
3. Orchestrator forwards client `detector_params` verbatim (source-predicted,
   confirmed live: passing `{"scheme": "synthid"}` reroutes scoring).
4. Health endpoints live on the dedicated :8034 port (`/health`, `/info`), not :8033.

**Acceptance test (EXECUTED, all through `POST orchestrator:8033/api/v2/text/detection/content`,
no client scheme params — server-side authority only):**

```
[PASS] kgw     -> watermark-kgw     : detected (kgw-watermark, score 1.0)
[PASS] synthid -> watermark-synthid : detected (synthid-watermark, score 1.0)
[PASS] kgw     -> watermark-synthid : not detected      [PASS] synthid -> watermark-kgw: not detected
[PASS] clean   -> both              : not detected      [PASS] human   -> both         : not detected
both-detectors-one-request: exactly the correct detector fires with detector_id attribution
```

**Signed results (EXECUTED):** Ed25519 key from Secret; `/v1/watermark/detect`
returns detached JWS (`alg=EdDSA`, `kid=poc-signing-2026-08`, `b64=false`); verified
locally against the public key; tampered payload correctly rejected. Key material in
gitignored `cluster/`, never logged.

**Retention posture (EXECUTED, scoped precisely):** three distinct 28-char
substrings of a submitted sample occur 0 times in the last-600-line log windows
of ALL THREE services — detector, detector-synthid, AND orchestrator (counted
python-side from `oc logs deploy/<name> --tail=600` captures so the substrings
never enter shell output; an earlier single-service 24-char/400-line count also
returned 0) — evidence of no *plaintext logging in those windows*, not a proof
of absolute zero retention. The stronger claim rests on
design + code evidence: the service writes only stdout logs (no PVC, no writable
data volume besides the tokenizer cache emptyDir), the logging paths emit
sha256-prefix + verdict only (detector/app.py, STATIC), and a caplog unit test
asserts no content reaches the log records (part of the 30/30 service suite).

**Phase 3 acceptance: MET** — the orchestrator routes detection to our detector and
returns correct verdicts for known-watermarked (both schemes) and known-clean text,
end-to-end on the cluster. Remaining non-engineering item: RHOAI's FMS-legacy/NeMo
transition (C11) means the *long-term* guardrails surface needs a NeMo custom-action
integration decision — recorded in facts D5, not improvised here.

## 2026-08-08 — Phase 2 completion: KGW-vs-SynthID detectability by length (EXECUTED)

Truncation-based scoring of the 512-token corpora (each scheme scored by its own
detector; controls scored per scheme in the per-scheme reports above — FPR 0.000
everywhere). Full table: `benchmarks/data/phase2_scheme_comparison.md` (data dir
gitignored; regenerate with benchmarks/compare_schemes.py).

| tokens | KGW δ2 mean z | KGW TPR | SynthID mean z | SynthID TPR | control FPR |
|---|---|---|---|---|---|
| **200** (Code threshold) | 9.08 | **0.992** (119/120) | 13.74 | **1.000** | 0.000 |
| 256 | 10.26 | 1.000 | 15.65 | 1.000 | 0.000 |
| 512 | 14.32 | 1.000 | 22.89 | 1.000 | (n/a — controls shorter) |

Reading: SynthID (depth 30) out-detects KGW (δ2) at every length on this model, with
comfortable margin at the Code-relevant 200-token threshold. KGW's single sub-threshold
sample at exactly 200 tokens (z<4, TPR 0.992) shows its δ2 margin is thinner there —
raising δ or preferring SynthID are both available levers. Combined with zero measured
FPR (270 controls) and the overhead table above, this completes the Phase 2
acceptance packet: **Phase 2 acceptance: MET.**

## Session close-out 2026-08-08

- GPU MachineSet at 0 replicas (verified; billable node released).
- Still running on (always-on) worker nodes: detector, detector-synthid,
  orchestrator, bench pods — the validated Phase 3 stack; teardown commands in
  deploy/phase3/README.md.
- Out of scope / blocked (unchanged, per AGENTS.md §6): D6 support-policy carve-out
  (product management), D7 grace-period scope (counsel), C11 NeMo live-CR validation
  (needs RHOAI install — Phase 4).

## 2026-08-08 — Adversarial-audit resolution + raw-evidence addendum

An adversarial audit (a 6-agent verification workflow plus independent review findings) challenged the completion
claims. Every finding below gets an explicit disposition; fixes were applied in this
commit and re-tested. This addendum also supplies the raw command+output evidence
whose absence the audit correctly flagged — the earlier prose entries ("captured in
session") did not meet this repo's EXECUTED bar on their own.

### Corrections to earlier entries (append-only; earlier text stands as history)

1. **SynthID overhead multiplier**: the Phase 2 table printed **3.17×**, which mixed
   baselines (913.78/287.82). Against the table's own stated 904.35 baseline it is
   **3.14×** (904.35/287.82=3.142). KGW's 1.41× was computed correctly.
2. **Orchestrator image attribution**: the Phase 3 entry named
   `quay.io/opendatahub/ta-guardrails-orchestrator:odh-3.4.2.git` — that was the
   research candidate from the contract survey, NOT what deploy/phase3/orchestrator.yaml
   pins and what actually ran. Deployed (verified live against the Deployment spec):
   `quay.io/trustyai/ta-guardrails-orchestrator@sha256:f3952b13…`, self-reporting
   `{"fms-guardrails-orchestr8":"0.16.0"}` — which also corrects the manifest header's
   own 0.17.0 release-date inference. All conclusions stand (0.16.0 likewise predates
   the 0.18.3 `path_prefix` field; its silent-ignore was verified live either way).
3. **Structured-output magnitude is δ4-era**: the guided_json test (8/8 detected,
   mean z 14.5) ran in the double-instance window. Composition (grammar + watermark
   coexist; output detected) is delta-independent and stands; the z **magnitude** at
   true δ2 is unmeasured (needs the GPU node scaled up again; recorded as pending work). facts.md B9
   updated accordingly.
4. **D8 time estimate**: "<1h A10G" was uncited; replaced in facts.md with arithmetic
   from measured throughput (10k×256 tok at 287.82 tok/s ≈ 2.5 h + controls ≈ 0.8 h).
5. **"512-token corpora (n=120)" phrasing**: the Phase 2 entry's tables label the
   corpora "512tok / n=120". Precisely: n=120 completions REQUESTED with
   max_tokens=512 and scored over full completions; only 76 (KGW) / 96 (SynthID)
   reach ≥512 tokenizer tokens (the comparison table's 512-truncation row uses
   exactly those subsets — its n columns are the honest lengths-reached counts).
6. **Phase 2/3 reproduction commands** — RECONSTRUCTED COMMAND SUMMARY, not raw
   execution transcripts: these are the commands executed during the 2026-08-08 work, re-assembled
   for readability (generation ran inside the bench pod via `oc exec … sh -c`
   batteries whose per-run stdout summaries are quoted in the Phase-2 sections
   above; analysis ran locally with `PYTHONPATH=src` and `WATERMARK_KEYS` set).
   Every command below is complete and literal — env prefixes included, no
   elided arguments:

   ```
   # corpora (inside the bench pod; scripts previously oc cp'd to /tmp)
   OPENAI_BASE_URL=http://vllm:8000/v1 python3 gen_corpus.py --model Qwen/Qwen2.5-0.5B-Instruct \
     --prompts-file prompts.txt --n 120 --max-tokens 512 --temperature 0.7 --watermark on \
     --key-id poc-2026-08 --scheme kgw --out corpus_kgw512_fixed.jsonl
   OPENAI_BASE_URL=http://vllm:8000/v1 python3 gen_corpus.py --model Qwen/Qwen2.5-0.5B-Instruct \
     --prompts-file prompts.txt --n 120 --max-tokens 512 --temperature 0.7 --watermark on \
     --key-id poc-2026-08 --scheme synthid --out corpus_synthid512.jsonl
   OPENAI_BASE_URL=http://vllm:8000/v1 python3 gen_corpus.py --model Qwen/Qwen2.5-0.5B-Instruct \
     --prompts-file prompts.txt --n 40 --max-tokens 256 --temperature 0.0 --watermark on \
     --key-id poc-2026-08 --scheme kgw --out corpus_kgw_temp0_fixed.jsonl
   # control corpus (pre-existing input to the analyses above — generated during
   # Phase 1, before the --scheme flag existed; flag-free form still valid):
   OPENAI_BASE_URL=http://vllm:8000/v1 python3 gen_corpus.py --model Qwen/Qwen2.5-0.5B-Instruct \
     --prompts-file prompts.txt --n 120 --max-tokens 256 --temperature 0.7 --watermark off \
     --out corpus_wm_off.jsonl
   # human corpora (local)
   PYTHONPATH=src /usr/bin/python3 benchmarks/fetch_human_corpus.py --chunk-tokens 512 \
     --n 150 --out benchmarks/data/human_corpus_512.jsonl
   # serving benchmarks (inside the bench pod; one command per configuration)
   OPENAI_BASE_URL=http://vllm:8000/v1 python3 bench_serving.py --model Qwen/Qwen2.5-0.5B-Instruct \
     --prompts-file prompts.txt --n 100 --max-tokens 256 --temperature 0.7 --concurrency 4 \
     --extra-body '{"vllm_xargs": {"watermark": "on", "watermark_key_id": "poc-2026-08", "watermark_scheme": "kgw"}}' \
     --out bench_kgw_fixed.json
   OPENAI_BASE_URL=http://vllm:8000/v1 python3 bench_serving.py --model Qwen/Qwen2.5-0.5B-Instruct \
     --prompts-file prompts.txt --n 100 --max-tokens 256 --temperature 0.7 --concurrency 4 \
     --extra-body '{"vllm_xargs": {"watermark": "on", "watermark_key_id": "poc-2026-08", "watermark_scheme": "synthid"}}' \
     --out bench_synthid.json
   OPENAI_BASE_URL=http://vllm:8000/v1 python3 bench_serving.py --model Qwen/Qwen2.5-0.5B-Instruct \
     --prompts-file prompts.txt --n 100 --max-tokens 256 --temperature 0.7 --concurrency 4 \
     --extra-body '{"vllm_xargs": {"watermark": "off"}}' --out bench_off2.json
   # analysis (local)
   PYTHONPATH=src python3 benchmarks/analyze_detection.py \
     --corpus benchmarks/data/corpus_kgw512_fixed.jsonl --corpus benchmarks/data/corpus_kgw_temp0_fixed.jsonl \
     --corpus benchmarks/data/corpus_wm_off.jsonl --corpus benchmarks/data/human_corpus.jsonl \
     --model-tokenizer Qwen/Qwen2.5-0.5B-Instruct --key-id poc-2026-08 --out benchmarks/data/phase2_kgw_fixed_report
   PYTHONPATH=src python3 benchmarks/analyze_detection.py --scheme synthid \
     --corpus benchmarks/data/corpus_synthid512.jsonl --corpus benchmarks/data/corpus_wm_off.jsonl \
     --corpus benchmarks/data/human_corpus.jsonl --model-tokenizer Qwen/Qwen2.5-0.5B-Instruct \
     --key-id poc-2026-08 --out benchmarks/data/phase2_synthid_report
   PYTHONPATH=src python3 benchmarks/compare_schemes.py \
     --kgw-corpus benchmarks/data/corpus_kgw512_fixed.jsonl --synthid-corpus benchmarks/data/corpus_synthid512.jsonl \
     --unwatermarked-corpus benchmarks/data/corpus_wm_off.jsonl --human-corpus benchmarks/data/human_corpus_512.jsonl \
     --model-tokenizer Qwen/Qwen2.5-0.5B-Instruct --key-id poc-2026-08 \
     --out benchmarks/data/phase2_scheme_comparison_v2.md
   ```

### Raw evidence: vLLM plugin double-load (independent re-execution)

Fresh CPU-only pod (image v0.18.0 by digest, wheel installed, no GPU — proves the
loading semantics are GPU-independent):

```
$ oc -n watermark exec loadercheck -- sh -c 'set -e
pip install -q --no-deps --target /tmp/site /tmp/vllm_watermark-0.1.0.dev0-py3-none-any.whl
PYTHONPATH=/tmp/site python3 -c "
from vllm.v1.sample.logits_processor import _load_custom_logitsprocs
old = _load_custom_logitsprocs([\"vllm_watermark.kgw.processor:KGWLogitsProcessor\"])
new = _load_custom_logitsprocs([])
print(\"flag+entrypoints:\", [c.__name__ for c in old])
print(\"entrypoints only:\", [c.__name__ for c in new])"'
flag+entrypoints: ['KGWLogitsProcessor', 'SynthIDLogitsProcessor', 'KGWLogitsProcessor']
entrypoints only: ['KGWLogitsProcessor', 'SynthIDLogitsProcessor']
```
(pod: image `vllm/vllm-openai@sha256:c32358eb…`, CPU-only, wheel `oc cp`'d in with
its canonical filename — pip rejects renamed wheels, a failure this capture hit and
fixed on the way.)

### Raw evidence: Phase 3 verdict matrix, signing, retention, health (fresh re-run)

Produced by the COMMITTED script `benchmarks/phase3_raw_capture.py` (run per its
header: `oc cp` script + samples into the bench pod, `oc -n watermark exec bench --
python3 /tmp/raw_capture.py`). The `$ POST …` lines inside the transcript are the
script's own request labels, not shell commands — the executable command is the one
`oc exec` line above. Only echoed text content is elided, marked with its sha256
prefix:

```
### Raw orchestrator verdict matrix (server-side scheme authority; no client scheme params)

$ POST /api/v2/text/detection/content  content=<kgw sample 1> detectors=watermark-kgw
HTTP 200 -> {"detections": [{"detection": "kgw-watermark", "detection_type": "watermark", "detector_id": "watermark-kgw", "end": 1113, "metadata": {"detector_version": "vllm-watermark-detector/0.1.0.dev0", "gamma": 0.25, "key_id": "poc-2026-08", "num_green": 85, "num_tokens_scored": 188, "p_value": 7.750825489048927e-11, "scheme": "kgw", "z_score": 6.400354600105544}, "score": 0.9999999999224918, "start": 0, "text": "[TEXT ELIDED, sha256:45fa45bf2b4f324c]"}]}

$ POST /api/v2/text/detection/content  content=<kgw sample 1> detectors=watermark-synthid
HTTP 200 -> {"detections": []}

$ POST /api/v2/text/detection/content  content=<synthid sample 1> detectors=watermark-kgw
HTTP 200 -> {"detections": []}

$ POST /api/v2/text/detection/content  content=<synthid sample 1> detectors=watermark-synthid
HTTP 200 -> {"detections": [{"detection": "synthid-watermark", "detection_type": "watermark", "detector_id": "watermark-synthid", "end": 1527, "metadata": {"depth": 30, "detector_version": "vllm-watermark-detector/0.1.0.dev0", "key_id": "poc-2026-08", "mean_g": 0.5726467331118494, "num_tokens_scored": 301, "p_value": 8.620629879501494e-51, "scheme": "synthid", "score": 0.5875031677758222, "scorer": "weighted_mean", "z_score": 14.943229604963577}, "score": 1.0, "start": 0, "text": "[TEXT ELIDED, sha256:7fc176c179a6ff71]"}]}

$ POST /api/v2/text/detection/content  content=<clean sample 1> detectors=watermark-kgw
HTTP 200 -> {"detections": []}

$ POST /api/v2/text/detection/content  content=<clean sample 1> detectors=watermark-synthid
HTTP 200 -> {"detections": []}

$ POST /api/v2/text/detection/content  content=<human sample 1> detectors=watermark-kgw
HTTP 200 -> {"detections": []}

$ POST /api/v2/text/detection/content  content=<human sample 1> detectors=watermark-synthid
HTTP 200 -> {"detections": []}

### Raw signed direct-endpoint response (kgw sample 1)
HTTP 200 -> {"detector_version": "vllm-watermark-detector/0.1.0.dev0", "key_id": "poc-2026-08", "model_tokenizer": "Qwen/Qwen2.5-0.5B-Instruct", "num_tokens_scored": 188, "p_value": 7.750825489048927e-11, "scheme": "kgw", "scheme_details": {"gamma": 0.25, "num_green": 85}, "score": 0.9999999999224918, "signature": "eyJhbGciOiJFZERTQSIsImI2NCI6ZmFsc2UsImNyaXQiOlsiYjY0Il0sImtpZCI6InBvYy1zaWduaW5nLTIwMjYtMDgiLCJ0eXAiOiJKT1NFIn0..XL4-t0GMqbBIc-czGWrBuSnh4-rBgLB2Bpo-kRTHY8gt3l-6wm3VNK9DrcWw-lFkZNM3QrxSRZsKLWVTF893Dw", "signing": "enabled", "verdict": true, "z_score": 6.400354600105544}

### Orchestrator health (dedicated port 8034)
GET :8034/health -> 200 {"fms-guardrails-orchestr8":"0.16.0"}
GET :8034/info -> 200 {"services":{"watermark-synthid":{"status":"HEALTHY"},"watermark-kgw":{"status":"HEALTHY"}}}
### Zero-retention raw check (distinctive 24-char substring of the submitted kgw sample, counted in detector logs — value itself never printed)
$ oc logs deploy/detector --tail=400 | grep -cF "<24-char sample substring>"
0

### Fresh JWS verification (local, public key from gitignored cluster/)
decode_complete OK: header = {"alg": "EdDSA", "b64": false, "crit": ["b64"], "kid": "poc-signing-2026-08", "typ": "JOSE"}
tampered payload -> InvalidSignatureError (correctly rejected)

### GPU MachineSet raw
$ oc -n openshift-machine-api get machineset ocp-ai-p9j4n-gpu-us-east-1a
NAME                          DESIRED   CURRENT   READY   AVAILABLE   AGE
ocp-ai-p9j4n-gpu-us-east-1a   0         0                             10h
```

### Phase 2 reference-validation evidence (audit: "identical seeds" criterion)

The acceptance criterion "validate the logits-processor formulation against the
reference implementation's outputs on identical seeds" is satisfied by
tests/test_synthid_equivalence.py, executed locally:

```
$ PYTHONPATH=src /usr/bin/python3 -m pytest tests/test_synthid_equivalence.py -q --durations=10
19 passed in 662.90s (0:11:02)
```

Specifically: `test_process_scores_row_matches_reference_call[50257]` and `[1000]`
drive transformers' own stateful `SynthIDTextWatermarkLogitsProcessor.__call__` and
ours on identical configs/inputs (max abs diff 0.0 over the trial set; development
probes additionally verified `_compute_keys`-fold equivalence and characterized the
reference's first-`ngram_len`-calls zero-context quirk — see the test file's
docstrings). KGW equivalence: tests/test_kgw_equivalence.py (`6 passed in 405.03s`),
exact greenlist set equality vs transformers at vocab 50257 and 151936.

### Phase 2 quality spot-check (exploratory fluency proxy — labeled precisely)

Committed script `benchmarks/quality_spotcheck.py` (Qwen2.5-0.5B, float32 CPU;
first 256 completion tokens; two estimators — UNCONDITIONAL completion PPL and
PROMPT-CONDITIONED completion PPL with prompt labels masked to −100). Selection is
a PAIRED-PROMPT selection: the first-15 prompts of the three corpora are
byte-identical (verified by SHA-256 over the prompt lists — `545179464ce6c394`
for all three; the script self-reports this check on every run). The reported
statistics are nonetheless UNPAIRED per-corpus aggregate means — no per-prompt
differencing — so between-corpus comparisons carry more variance than a paired
estimator would. An earlier single-estimator run (unconditional-only) reported
4.97/7.69/2.92; the recorded run below supersedes it.

```
$ PYTHONPATH=src /usr/bin/python3 benchmarks/quality_spotcheck.py \
    --corpus kgw_wm_d2=benchmarks/data/corpus_kgw512_fixed.jsonl \
    --corpus synthid_wm=benchmarks/data/corpus_synthid512.jsonl \
    --corpus unwatermarked=benchmarks/data/corpus_wm_off.jsonl
kgw_wm_d2      n=15 PPL uncond mean=   4.70  prompt-conditioned mean=   3.74  distinct-1=0.389 distinct-2=0.822
synthid_wm     n=15 PPL uncond mean=   7.15  prompt-conditioned mean=   5.70  distinct-1=0.346 distinct-2=0.799
unwatermarked  n=15 PPL uncond mean=   2.89  prompt-conditioned mean=   2.32  distinct-1=0.432 distinct-2=0.864
```

Reading, precisely scoped: this is an EXPLORATORY fluency proxy under one small
model at small n — not a human quality rating and not claimed as a bound on one;
PPL under the generating model partly re-measures the watermark's own
distribution shift by construction. Observed: prompt-conditioned completion PPL
rises 2.32 → 3.74 (KGW δ2) and → 5.70 (SynthID depth-30); lexical diversity dips
modestly (distinct-2 0.864 → 0.822/0.799 — NOTE: in the recorded run distinct-n
was computed over FULL completions while PPL used the first 256 tokens; the
committed script now computes both over the identical first-256-token slice for
future runs). Because the recorded PPL output above predates the script's
pairing-check print, the pairing verification was executed SEPARATELY against
the current source:

```
$ PYTHONPATH=src /usr/bin/python3 benchmarks/quality_spotcheck.py \
    --corpus kgw_wm_d2=benchmarks/data/corpus_kgw512_fixed.jsonl \
    --corpus synthid_wm=benchmarks/data/corpus_synthid512.jsonl \
    --corpus unwatermarked=benchmarks/data/corpus_wm_off.jsonl \
    --verify-pairing-only
prompt-set sha256/16 per corpus: {'kgw_wm_d2': '545179464ce6c394', 'synthid_wm': '545179464ce6c394', 'unwatermarked': '545179464ce6c394'} -> PAIRED (byte-identical prompts)
PAIRED
``` Human-rated evaluation remains Phase 5
work; no external study is cited as covering this model/configuration.

### Phase 1 z-score distribution (audit: distributions, not just summary stats)

Corrected single-instance KGW δ2 corpus: n=120 completions generated with
`max_tokens=512` at T=0.7 and scored over each FULL completion (variable length —
only 76/120 reach ≥512 tokenizer tokens; the rest stopped earlier), 20-bin
histogram from `benchmarks/data/phase2_kgw_fixed_report.md` (regenerate with the
full analyze_detection.py invocation from the reproduction-commands list above —
the same four-corpus command that produced the report):

```
[   5.72,    6.47)  ####                                      2
[   6.47,    7.22)                                            0
[   7.22,    7.97)                                            0
[   7.97,    8.72)  ########                                  4
[   8.72,    9.47)  ######                                    3
[   9.47,   10.22)  ##########                                5
[  10.22,   10.97)  #############                             7
[  10.97,   11.72)  #############                             7
[  11.72,   12.48)  #####################                     11
[  12.48,   13.23)  #############################             15
[  13.23,   13.98)  ########################################  21
[  13.98,   14.73)  ###############                           8
[  14.73,   15.48)  ###################                       10
[  15.48,   16.23)  #######################                   12
[  16.23,   16.98)  ###########                               6
[  16.98,   17.73)  ##########                                5
[  17.73,   18.48)  ##                                        1
[  18.48,   19.24)  ####                                      2
[  19.24,   19.99)                                            0
[  19.99,   20.74)  ##                                        1
```

### Phase 1 "~256 tokens" single-instance dataset (audit item: explicit result)

The corrected primary corpus is 512-token; the protocol says ~256. Disposition: for
an autoregressive decode-time watermark, the first 256 tokens of a 512-token
generation are distributionally identical to a native 256-token generation (per-step
bias, no lookahead), so the 256-token truncation row of the corrected corpus is a
valid ~256-token single-instance result: **n=116 (≥100), TPR 1.000, mean z 10.26;
control FPR 0.000 at the same 256-token row on n=115 unwatermarked + n=149 human =
264 controls** (the 269-control figure belongs to the 200-token row: 119+150). The single-instance T=0 dataset
(n=40, REQUESTED max_tokens=256 — 38/40 reached 256 completion tokens, min 192,
mixed length/stop finish reasons; mean z 9.91, TPR 1.000) corroborates at
approximately-256 native generation length.

### Phase ordering + same-commit discipline (audit item)

- **Workflow violation, recorded:** docs/implementation.md says work phases in
  order and meet acceptance before moving on. In fact, commit 59b03ee declared
  "Phase 2 acceptance: MET" while the KGW-vs-SynthID comparison table was still
  computing (its completion landed in the NEXT commit, 0701fc9), and Phase 3 was
  executed and committed in 59b03ee itself — i.e. before Phase 2's acceptance
  evidence was complete. The independently produced Phase 2 and Phase 3 results
  remain valid (each carries its own evidence), but the phase-ordering discipline
  was breached and this line is the record of it.
- facts.md tag upgrades landed in the same commits as their EXPERIMENTS.md evidence
  with one exception the audit found: fact C9 (Phase 0 baseline) was added one
  commit after its evidence (76c196e → 81e11a7). Recorded as a process violation;
  not retroactively fixable without history rewrite (declined — see below).

### Repo-policy dispositions (maintainer-level decisions, recorded not improvised)

- **Commit-trailer attribution vs AGENTS.md §4 — exact state**: origin/main
  contains 4 pushed commits, all carrying the AI attribution trailer (ef3b0e4,
  76c196e — the Phase 0 pair; 5db603d, eb4c76a — separate same-day maintainer
  commits). Additionally origin/main..HEAD holds 4 LOCAL, UNPUSHED commits
  (81e11a7, 3ae2f35, 59b03ee, 0701fc9), also with trailers — these are NOT pushed.
  Recorded disposition: branch history is preserved as-is (no rewrite/rebase);
  new commits omit the trailer; curing the already-pushed trailers would require
  a maintainer-authorized history rewrite + force-push.
- **Sandbox base domain in pushed history**: commit 5db603d redacted it from the
  working tree, but it exists in the 76c196e blob (pushed). Current tracked files
  verified clean (`git grep sandbox…` = 0 hits). Cure requires history rewrite +
  force-push (a maintainer decision) or accepting sandbox credential rotation.
- **Stale committed docs/technical.md + docs/openshift-ai.md** (still describe D1 as
  open): corrected versions exist as pending uncommitted working-tree edits,
  which this work deliberately does not touch. Cure: committing those pending
  corrections (a maintainer action).

### Combined test-suite invocation (audit: reproducible single command)

`tests/` and `detector/tests/` initially could not be collected together (two
`conftest.py` files + `from conftest import …` module imports → collection errors,
reproduced before fixing). Fixed via `--import-mode=importlib` in pyproject
`[tool.pytest.ini_options]` (with `testpaths`) and canonical-path imports in the two
test files. Reproducible invocation and result:

```
$ /usr/bin/python3 -m pytest -q
154 passed, 192 warnings in 1746.78s (0:29:06)
```
(192 warnings are third-party FastAPI/Starlette deprecations under Python 3.14.)

### NeMo Guardrails forward-path validation (upstream library — precisely scoped)

EXECUTED on-cluster (namespace watermark, one ConfigMap + one temporary pod, GPU at
0 throughout): a custom output-rail action POSTing the bot message to the detector
service blocked a known-KGW sample and passed a human sample, consistently across
three angles — direct detector call (`verdict=true, z=12.817, p=6.57e-38` vs
`verdict=false, z=0.713`), the library Python API (`check_async` →
`BLOCKED`/`PASSED`), and `nemoguardrails server` `POST /v1/checks`
(`{"status":"blocked","rail":"watermark check"}` / `{"status":"passed","rail":null}`),
nemoguardrails==0.23.0. Config + full header documentation:
`deploy/phase3/nemo-guardrails-poc.yaml`.

**Scope limits (explicit):** this validates the UPSTREAM library path only. It does
NOT prove the RHOAI-managed `NemoGuardrails` CR path (no RHOAI/operator/CRD was
installed) and does not close that part of D5. Version skew risk recorded: RHOAI's
shipped NeMo version is unverified vs the pip-installed 0.23.0, and real behavioral
differences were found between 0.23.0 and the develop-HEAD sources the earlier
api-notes cited.

**Findings of independent value:** NeMo's own defaults conflict with a
zero-retention posture (422 validation errors echo the full request body; the
server's internal event log records full message content) — a gap to close before
adopting this path for compliance use.

**Incident record:** during initial validation, a debugging step briefly echoed
non-sensitive, repo-resident benchmark text (a Gutenberg excerpt and one generated
sample) into command output, contrary to the content-handling rule for this work;
corrected immediately, the pod was deleted, nothing persisted on-cluster. A
follow-up hardening pass (fail-closed failure policy, pinned installs, env-sourced
key id, neutral header wording) was dispatched and its results are recorded below
when complete.

### Namespace-state re-verification (fourth audit round)

An independent check initially reported namespace `watermark` empty of
workloads. Direct re-verification could not reproduce that, and a follow-up
JSON-output query on the intended kubeconfig subsequently CONFIRMED the five
pods and three Available Deployments — the earlier empty result was a
display/query anomaly, not evidence of a different cluster. The block below is
an EXECUTED SUMMARY (condensed from raw output; server URL redacted per repo
convention). Exact commands: `oc get ns watermark`; `oc -n watermark get
pods,deploy,svc,cm,secrets`; `oc -n openshift-machine-api get machineset
ocp-ai-p9j4n-gpu-us-east-1a`; the health/detection lines ran via `oc -n
watermark exec bench -- python3 -c '<urllib POST>'` with the request bodies
shown:

```
### Namespace inventory re-verification, 2026-08-08T06:19:20Z
watermark   Active   4h36m
pods: bench, detector-1-build(Completed), detector-…, detector-synthid-…, orchestrator-…  (all Running)
deploy: detector 1/1, detector-synthid 1/1, orchestrator 1/1
svc: detector, detector-synthid, orchestrator, vllm
cm: orchestrator-config, nemo-watermark-config, detector-1-* (build), …
secret: watermark-key, detector-signing-key
machineset ocp-ai-p9j4n-gpu-us-east-1a: 0 replicas
### Health + one KGW + one SynthID orchestrator request, 2026-08-08T06:19:24Z
GET :8034/health -> {"fms-guardrails-orchestr8":"0.16.0"}
GET :8034/info   -> {"services":{"watermark-synthid":{"status":"HEALTHY"},"watermark-kgw":{"status":"HEALTHY"}}}
kgw sample2 -> watermark-kgw: HTTP 200 [{"detection": "kgw-watermark", "detector_id": "watermark-kgw", "score": 1.0}]
synthid sample2 -> watermark-synthid: HTTP 200 [{"detection": "synthid-watermark", "detector_id": "watermark-synthid", "score": 1.0}]
```

The recent event log shows only this work's own deployment rollouts (the
SIGNING_KEY_ID Secret-sourcing change) — no mass deletion. The contrary observation
remains unexplained from here (possible different kubeconfig/context or transient
view); all live-state claims in this log are therefore tied to explicit timestamps,
and the stack is reproducible from deploy/phase3/ regardless.


### Correction to "Session close-out" (append-only)

The earlier close-out entry listed detector, detector-synthid, orchestrator, and
bench as "still running". That statement was true when written but is superseded
as a live claim: pods have since been REPLACED by rollouts (Secret-sourced
SIGNING_KEY_ID), transient single-purpose pods existed briefly and were deleted
(`loadercheck` — the CPU loader-repro pod above — and `nemo-test`), and the
namespace-state re-verification section above (timestamped 06:19Z) is the current
evidence of record. The GPU MachineSet has remained at 0 replicas since the
close-out. Live-state statements in this file are valid only at their recorded
timestamps; the deploy/phase3 manifests are the reproducible source of truth.

## 2026-08-08 — NeMo PoC hardening evidence (full transcript, fresh-pod pass)

The section below is the verbatim command+output transcript produced in ONE
coherent fresh-pod sequence (content redacted only via [TEXT ELIDED sha256:…]
markers), covering: constrained install + zero-diff freeze check; fail-closed
on malformed-200 responses (missing verdict, non-boolean verdict); the
adversarial log-tunneling case (detector returns verdict=true plus 5 poisoned
string fields — all marker counts 0 in 135 lines of process output; the single
emitted log line carries request-local scheme/key_id and <invalid:str> markers
only); missing-vs-empty bot_message semantics; real-detector happy-path
regression; pod cleanup; GPU MachineSet at 0. The committed actions.py copy
was extracted from the final ConfigMap and py_compile-verified.

# NeMo Guardrails PoC — full evidence transcript

Date: 2026-08-08. Repo: `/home/anaeem/vllm-watermark`. Cluster: `ocp-ai`,
`KUBECONFIG=cluster/auth/kubeconfig` (never printed). Namespace: `watermark`.
Cluster objects touched in this pass: ConfigMap `nemo-watermark-config`
(updated in place) and Pod `nemo-test` (created fresh, deleted at end).
No RHOAI, operator, or CRD installed. GPU MachineSet
`ocp-ai-p9j4n-gpu-us-east-1a` untouched throughout (confirmed 0/0 before
and after, item 9 below).

This is the raw, content-redacted evidence record backing
`deploy/phase3/nemo-guardrails-poc.yaml`'s header claims, per AGENTS.md's
verification discipline ("Works" means EXECUTED — command + raw output).
All nine items below were run in one coherent sequence, in the order
listed, against a single fresh `nemo-test` Pod (items 1–7), then cleanup
(items 8–9). Submitted/returned TEXT content is never printed anywhere in
this transcript; where a command's semantics involve real corpus text, the
text is redacted as `[TEXT ELIDED sha256:<prefix>]` and the actual script
reads the text from a file at runtime rather than inlining it on any
command line.

Local corpus samples used (read locally, `oc cp`'d into the pod as files,
byte counts confirmed to match after copy, sha256 prefixes computed
locally for redaction markers only):
- `benchmarks/data/corpus_kgw512_fixed.jsonl` record[0] `text` field →
  `/tmp/kgw_sample_0.txt`, 2350 bytes, `sha256:64b551969b24`
- `benchmarks/data/human_corpus.jsonl` record[0] `text` field →
  `/tmp/human_sample_0.txt`, 1107 bytes, `sha256:d93033d39053`

---

## (1) Fresh pod creation

Command:
```
export KUBECONFIG=/home/anaeem/vllm-watermark/cluster/auth/kubeconfig
oc apply -f pod.yaml   # pod.yaml == the Pod spec below, name nemo-test
oc -n watermark wait --for=condition=Ready pod/nemo-test --timeout=90s
```

`pod.yaml` (matches the Pod spec applied; identical shape used across all
passes in this validation):
```yaml
apiVersion: v1
kind: Pod
metadata:
  name: nemo-test
  namespace: watermark
  labels:
    app: nemo-test
spec:
  restartPolicy: Never
  enableServiceLinks: false
  containers:
    - name: nemo-test
      image: python:3.12-slim
      command: ["sleep", "infinity"]
      resources:
        requests:
          cpu: 500m
          memory: 1Gi
      env:
        - name: VLLM_WATERMARK_SCHEME
          value: "kgw"
        - name: VLLM_WATERMARK_KEY_ID
          value: "poc-2026-08"
        - name: OPENAI_API_KEY
          value: "unused-model-credential-placeholder"  # never called; value chosen to not resemble a secret
      volumeMounts:
        - name: nemo-config
          mountPath: /app/config/watermark-poc
  volumes:
    - name: nemo-config
      configMap:
        name: nemo-watermark-config
```

Raw output:
```
Warning: would violate PodSecurity "restricted:latest": allowPrivilegeEscalation != false (container "nemo-test" must set securityContext.allowPrivilegeEscalation=false), unrestricted capabilities (container "nemo-test" must set securityContext.capabilities.drop=["ALL"]), runAsNonRoot != true (pod or container "nemo-test" must set securityContext.runAsNonRoot=true), seccompProfile (pod or container "nemo-test" must set securityContext.seccompProfile.type to "RuntimeDefault" or "Localhost")
pod/nemo-test created
pod/nemo-test condition met
```
(The PodSecurity line is an advisory warning only — the Pod was created
and became Ready. The ConfigMap `nemo-watermark-config` mounted by this
Pod already existed in the namespace with the current `actions.py`
content, applied immediately before this step from
`deploy/phase3/nemo-guardrails-poc.yaml`.)

---

## (2) Pinned install + pip freeze diff (zero differences)

Commands:
```
oc -n watermark cp deploy/phase3/nemo-poc-constraints.txt nemo-test:/tmp/nemo-poc-constraints.txt

oc -n watermark exec nemo-test -- python3 -m pip install --quiet --no-input \
  --root-user-action=ignore -c /tmp/nemo-poc-constraints.txt \
  'nemoguardrails[server]==0.23.0' httpx

oc -n watermark exec nemo-test -- python3 -m pip freeze | sort > /tmp/pass4_freeze.txt
diff /tmp/pass4_freeze.txt deploy/phase3/nemo-poc-constraints.txt
echo "diff_exit_code=$?"
```

Raw output:
```
=== install ===

[notice] A new release of pip is available: 25.0.1 -> 26.2.1
[notice] To update, run: pip install --upgrade pip
exit_code=0
=== freeze + diff ===
diff_exit_code=0
78
```
(`diff_exit_code=0` means zero differences — all 78 pinned packages,
including every transitive dependency, matched exactly between the
constraints file and this fresh Pod's actual resolved install.)

---

## (3) Malformed-200 missing-verdict → BLOCKED

An in-pod mock HTTP server (Python `http.server`, listening only inside
`nemo-test` on `localhost:9010` — the real `detector`/`detector-synthid`
Deployments were never touched or restarted) returns HTTP 200 with a
well-formed JSON object that has no `"verdict"` key at all:
```python
# mock_missing_verdict.py, served on 0.0.0.0:9010
body = {"scheme": "kgw", "key_id": "poc-2026-08"}  # no "verdict" key
```

Command:
```
oc -n watermark exec nemo-test -- sh -c '
  python3 /tmp/mock_missing_verdict.py > /tmp/mock1.log 2>&1 &
  sleep 1
  VLLM_WATERMARK_DETECTOR_URL=http://localhost:9010/v1/watermark/detect \
    python3 /tmp/test_check_generic.py 2>&1 | grep -E "^WARNING:actions|^\{"
'
```
(`test_check_generic.py` builds an `LLMRails` from `/app/config/watermark-poc`
and calls `check_async` with a synthetic, non-corpus dummy assistant
message — content is irrelevant since the mock ignores the request body
entirely.)

Raw output:
```
WARNING:actions.py:watermark_check: FAILING CLOSED (blocking response) reason=detector response missing 'verdict' field policy=closed
{"status": "RailStatus.BLOCKED", "rail": "watermark check", "blocked": true}
```

---

## (4) Malformed-200 non-boolean verdict → BLOCKED

Same mock pattern, `localhost:9011`, returning `{"verdict": "true"}` (a
JSON string, not a boolean):
```python
body = {"scheme": "kgw", "key_id": "poc-2026-08", "verdict": "true"}
```

Command:
```
oc -n watermark exec nemo-test -- sh -c '
  python3 /tmp/mock_bad_type_verdict.py > /tmp/mock2.log 2>&1 &
  sleep 1
  VLLM_WATERMARK_DETECTOR_URL=http://localhost:9011/v1/watermark/detect \
    python3 /tmp/test_check_generic.py 2>&1 | grep -E "^WARNING:actions|^\{"
'
```

Raw output:
```
WARNING:actions.py:watermark_check: FAILING CLOSED (blocking response) reason=detector response 'verdict' field is not boolean: type=str policy=closed
{"status": "RailStatus.BLOCKED", "rail": "watermark check", "blocked": true}
```

---

## (5) Adversarial marker-fields case (log-tunneling defect fix) → all marker counts 0

Same mock pattern, `localhost:9012`, `verdict: true` (a genuine bool, so
the response otherwise validates) but every OTHER field the action reads
is a distinctive marker string:
```python
PAYLOAD = {
    "verdict": True,
    "z_score": "MARKER_A_7f3d9c2e1b8a4f6d",
    "p_value": "MARKER_D_1d3b8e4a9e2a5c7f",
    "scheme": "MARKER_B_9e2a5c7f1d3b8e4a",
    "key_id": "MARKER_E_5c7f1d3b8e4a9e2a",
    "extra": "MARKER_C_3b8e4a9e2a5c7f1d",
}
```

Commands (run the rail, capture the ENTIRE process stdout/stderr to a
file, then count marker occurrences across that whole file — not just
this action's own log line, so this also covers any NeMo-internal
echoing):
```
oc -n watermark exec nemo-test -- sh -c '
  python3 /tmp/mock_adversarial.py > /tmp/mock3.log 2>&1 &
  sleep 1
  VLLM_WATERMARK_DETECTOR_URL=http://localhost:9012/v1/watermark/detect \
    python3 /tmp/test_check_generic.py > /tmp/adversarial_test_output.log 2>&1
  echo "capture_exit_code=$?"
  wc -l /tmp/adversarial_test_output.log
'

# count_markers.py: reads /tmp/adversarial_test_output.log, counts each of
# the 5 marker literals, prints ONLY the counts (must all be 0) and any
# log line matching "watermark_check verdict=" that does NOT itself
# contain any marker literal (defensive double-check before printing).
oc -n watermark exec nemo-test -- python3 /tmp/count_markers.py

# separately, confirm the rail's own decision (safe JSON line only):
oc -n watermark exec nemo-test -- sh -c 'grep -E "^\{" /tmp/adversarial_test_output.log'
```

Raw output:
```
=== run ===
capture_exit_code=0
135 /tmp/adversarial_test_output.log
=== count markers (python-side counting; only counts + sanitized line printed) ===
{"marker_counts": {"MARKER_A_ZSCORE": 0, "MARKER_B_SCHEME": 0, "MARKER_C_EXTRA": 0, "MARKER_D_PVALUE": 0, "MARKER_E_KEYID": 0}}
{"sanitized_log_lines": ["INFO:actions.py:watermark_check verdict=True z_score=<invalid:str> p_value=<invalid:str> scheme=kgw key_id=poc-2026-08"], "unsafe_lines_withheld": 0}
=== rail decision ===
{"status": "RailStatus.BLOCKED", "rail": "watermark check", "blocked": true}
```
All 5 marker counts are 0 (across the entire 135-line captured process
output, not just the action's own log line). The single log line the
action emitted logs the REQUEST-LOCAL `scheme=kgw key_id=poc-2026-08`
(the values this Pod's env actually configured), not the mock's
`MARKER_B_.../MARKER_E_...` echoes, and `<invalid:str>` markers in place
of the mock's `MARKER_A_.../MARKER_D_...` strings for `z_score`/`p_value`
— exactly the fix under test. `unsafe_lines_withheld: 0` confirms the
defensive per-line marker check never had to withhold anything (the
aggregate count and the per-line check agree).

---

## (6) bot_message missing-key → BLOCKED; empty-string → ALLOWED

Direct unit-level calls to the `watermark_check` action (imported from the
mounted ConfigMap path), no detector HTTP call reached in either case
since both short-circuit before that point:

```python
# test_botmessage2.py
r1 = await actions.watermark_check(context={"some_other_key": "x"})   # no "bot_message" key
r2 = await actions.watermark_check(context={"bot_message": ""})        # empty string
```

Command:
```
oc -n watermark exec nemo-test -- python3 /tmp/test_botmessage2.py 2>&1 \
  | grep -E "^WARNING:actions|^DEBUG:actions|^\{"
```

Raw output:
```
WARNING:actions:watermark_check: FAILING CLOSED (blocking response) reason=bot_message missing from context (integration failure) policy=closed
{"case": "bot_message_key_missing", "blocked": true}
DEBUG:actions:watermark_check: bot_message is empty -- nothing to scan, allowing.
{"case": "bot_message_empty_string", "blocked": false}
```

---

## (7) Real-detector happy-path regression (KGW → blocked, human → passed via /v1/checks)

Commands:
```
oc -n watermark exec nemo-test -- sh -c \
  'nohup nemoguardrails server --config /app/config --port 8080 --disable-chat-ui > /tmp/server.log 2>&1 & sleep 6; echo LAUNCHED'
oc -n watermark exec nemo-test -- sh -c 'grep -E "Uvicorn running|ERROR" /tmp/server.log | tail -5'

# http_checks5.py POSTs to POST /v1/checks; body constructed as:
#   {"model": "watermark-poc",
#    "messages": [{"role": "user", "content": "Please write a short paragraph."},
#                 {"role": "assistant", "content": "[TEXT ELIDED sha256:64b551969b24]"}],  # kgw sample
#                  -- or --                       {"content": "[TEXT ELIDED sha256:d93033d39053]"}]  # human sample
#    "guardrails": {"config_id": "watermark-poc"}}
# (the script reads the actual text from /tmp/kgw_sample_0.txt / /tmp/human_sample_0.txt
# at runtime -- text is never inlined on any command line or printed; the
# script's own output extracts only http_status/status/rail fields, never
# the response "content" field, on every status-code path.)
oc -n watermark exec nemo-test -- python3 /tmp/http_checks5.py
```

Raw output:
```
LAUNCHED
INFO:     Uvicorn running on http://0.0.0.0:8080 (Press CTRL+C to quit)
{"label": "kgw_watermarked", "http_status": 200, "status": "blocked", "rail": "watermark check"}
{"label": "human", "http_status": 200, "status": "passed", "rail": null}
```

Direct-detector ground truth for these same two samples (recorded in an
earlier pass, unchanged — `POST /v1/watermark/detect`, scheme=kgw,
key_id=poc-2026-08):
- kgw sample (`sha256:64b551969b24`): `verdict=true, z_score=12.817, p_value=6.57e-38, num_tokens_scored=400`
- human sample (`sha256:d93033d39053`): `verdict=false, z_score=0.713, p_value=0.238, num_tokens_scored=237`

---

## (8) Pod deletion + NotFound confirmation

Commands:
```
oc -n watermark delete pod nemo-test --wait=true
oc -n watermark get pod nemo-test
```

Raw output:
```
pod "nemo-test" deleted
Error from server (NotFound): pods "nemo-test" not found
```

---

## (9) GPU MachineSet confirmed at 0

Command:
```
oc -n openshift-machine-api get machineset ocp-ai-p9j4n-gpu-us-east-1a
```

Raw output:
```
NAME                          DESIRED   CURRENT   READY   AVAILABLE   AGE
ocp-ai-p9j4n-gpu-us-east-1a   0         0                             11h
```

---

## Post-cleanup state

After item 8, the `watermark` namespace's Pod list contained none of
this validation's objects (only the pre-existing `bench`,
`detector-1-build`, `detector-*`, `detector-synthid-*`, `orchestrator-*`
workloads unrelated to this validation). The ConfigMap
`nemo-watermark-config` was left in place, re-applied from
`deploy/phase3/nemo-guardrails-poc.yaml` immediately before this pass so
that the live object and the committed file are byte-identical (diffed
and confirmed as part of this same session, outside the 9 numbered items
above since it precedes item 1).


### Documentation-boundary corrections (final pass)

1. **Correction to the committed Phase 3 acceptance paragraph** (append-only): it
   called the RHOAI-managed NeMo transition a "non-engineering item". That is
   wrong — the RHOAI-managed `NemoGuardrails` CR integration (operator mount
   behavior, shipped-version verification, retention-gap mitigation) is an
   ENGINEERING integration item deferred to Phase 4; only the C11 lifecycle/
   support-posture question itself is non-engineering. facts.md D5 carries the
   precise open list.
2. **Scope of the NeMo install-reproducibility claim**: the constraints file pins
   all 78 resolved Python packages and was verified zero-drift in a fresh pod —
   that claim covers VERSION-RESOLVED PYTHON DEPENDENCIES under the recorded
   image/platform (python:3.12-slim, CPython 3.12, linux/amd64) only. The base
   image tag is mutable and no artifact hashes are pinned, so this is NOT a
   bit-reproducible container/supply-chain claim. The temporary test pod also
   ran with PodSecurity "restricted" WARNINGS (no securityContext hardening) —
   acceptable for a deleted single-purpose test pod, but any reusable deployment
   of this rail must add a restricted-profile securityContext and pin the base
   image by digest. Recorded as a limitation; no rerun claimed or needed.
3. **Evidence vs reusable artifacts**: the `/tmp/...` helper commands in the NeMo
   transcript above are EXACT HISTORICAL EXECUTION EVIDENCE from the deleted test
   pod — they are not a committed replay harness. The reusable, committed
   artifacts are `deploy/phase3/nemo-guardrails-poc.yaml` (the ConfigMap with the
   rail config and hardened actions.py) and `deploy/phase3/nemo-poc-constraints.txt`.


## 2026-08-08 — Scheme comparison v2 (per-scheme control FPR; supersedes the v1 table's control rows)

Produced by the committed, six-row-model benchmarks/compare_schemes.py (exact
invocation in the reproduction-commands list above, with
`--human-corpus benchmarks/data/human_corpus_512.jsonl`). What v2 adds over the
v1 table quoted earlier: (a) every control corpus is scored by BOTH detectors —
SynthID's FPR is now independently supported instead of implied; (b) a 512-token
human control corpus covers the 512 row (the 256-token unwatermarked corpus
still cannot — its n=0 row is explicit); (c) the config header carries the
resolved numeric depth. TPR rows are unchanged from v1 (same corpora/detectors).

# KGW vs SynthID scheme comparison

model_tokenizer=`Qwen/Qwen2.5-0.5B-Instruct` vocab_size=`151936` key_id=`poc-2026-08` gamma=`0.25` synthid_depth=`30` synthid_ngram_len=`5` truncation_lengths=`[200, 256, 512]`

## Truncation length 200 tokens

| corpus | n | mean z | rate | metric |
|---|---|---|---|---|
| kgw | 120 | 9.084 | 0.992 | TPR |
| synthid | 120 | 13.744 | 1.000 | TPR |
| unwatermarked (kgw det) | 119 | -0.148 | 0.000 | FPR |
| unwatermarked (synthid det) | 119 | -0.088 | 0.000 | FPR |
| human (kgw det) | 150 | 0.027 | 0.000 | FPR |
| human (synthid det) | 150 | -0.087 | 0.000 | FPR |

## Truncation length 256 tokens

| corpus | n | mean z | rate | metric |
|---|---|---|---|---|
| kgw | 116 | 10.261 | 1.000 | TPR |
| synthid | 117 | 15.651 | 1.000 | TPR |
| unwatermarked (kgw det) | 115 | -0.068 | 0.000 | FPR |
| unwatermarked (synthid det) | 115 | -0.060 | 0.000 | FPR |
| human (kgw det) | 150 | 0.084 | 0.000 | FPR |
| human (synthid det) | 150 | -0.070 | 0.000 | FPR |

## Truncation length 512 tokens

| corpus | n | mean z | rate | metric |
|---|---|---|---|---|
| kgw | 76 | 14.317 | 1.000 | TPR |
| synthid | 96 | 22.886 | 1.000 | TPR |
| unwatermarked (kgw det) | 0 | n/a | n/a | FPR |
| unwatermarked (synthid det) | 0 | n/a | n/a | FPR |
| human (kgw det) | 150 | 0.099 | 0.000 | FPR |
| human (synthid det) | 150 | -0.098 | 0.000 | FPR |

## Notes

- `kgw`/`synthid` rows: TPR at that scheme's own detection threshold, scored with that scheme's own detector.
- `unwatermarked (...)` / `human (...)` rows: FPR of the named detector on that control corpus -- one row per (corpus, detector) pair, so each scheme's FPR is independently supported.
- n varies by truncation length: a row is scored at length L only if it has >= L scored tokens (shorter completions are excluded, counted in n_skipped_too_short in the JSON). Control corpora generated/chunked at ~256 tokens therefore have n=0 at the 512 row unless a 512-token control corpus is supplied.
- `not present` cells mean that --*-corpus flag was omitted, not a scoring failure.

## 2026-08-08 — Independent post-push review correction

This append-only correction supersedes the audit addendum's unqualified
"phrase sweep clean" statement. That sweep did not catch repository-internal
instruction references (`CLAUDE.md`, `Task A2/B2/B3/C3`, and workflow narration).
Explicit `CLAUDE.md` references were removed from the active source, test, benchmark,
and deployment files touched by this correction, but older append-only evidence and
several research/API-note files still retain historical task/workflow language. The
AGENTS.md content-policy criterion is therefore **not fully closed**; the standalone
review artifact records it as an open repository-hygiene finding rather than claiming
a clean sweep.

The review also found and corrected stale runbook statements without changing runtime
behavior: Phase 0/3 README files now say the cluster paths were executed; Phase 3 health
expects the measured orchestrator 0.16.0; SynthID uses empty client params with the
twin-Service server-side routing; combined-detector attribution is recorded as executed;
teardown includes `detector-synthid`; the YAML count is five; and the obsolete detector
container sketch now points to the executed Dockerfile/build path. The Phase 1 Pod header
now agrees with its actual entry-point-only loading, avoiding the previously measured
double-load trap.

### Focused verification after the correction

```
$ /usr/bin/python3 -m pytest -q detector/tests tests/test_processor_static.py tests/test_synthid_processor_static.py
........................................................ [ 55%]
.........................................................                [100%]
129 passed, 192 warnings in 125.36s (0:02:05)
```

```
$ python3 -m py_compile src/vllm_watermark/__init__.py src/vllm_watermark/keys.py src/vllm_watermark/kgw/processor.py src/vllm_watermark/synthid/core.py src/vllm_watermark/synthid/processor.py detector/app.py benchmarks/gen_corpus.py benchmarks/analyze_detection.py
# no output; exit 0

$ python3 - <<'PY'
from pathlib import Path
import yaml
for path in sorted(Path('deploy/phase3').glob('*.yaml')):
    docs = list(yaml.safe_load_all(path.read_text()))
    print(f'{path}: {len(docs)} document(s) OK')
PY
deploy/phase3/detector-build.yaml: 2 document(s) OK
deploy/phase3/detector-deploy.yaml: 2 document(s) OK
deploy/phase3/detector-synthid-deploy.yaml: 2 document(s) OK
deploy/phase3/nemo-guardrails-poc.yaml: 1 document(s) OK
deploy/phase3/orchestrator.yaml: 3 document(s) OK
```

Server-side dry-run accepted every Phase 0/3 object. It also produced a real
PodSecurity warning for the bare Phase 1 vLLM Pod; the warning is preserved here as
an open Phase 4 hardening item rather than hidden:

```
$ KUBECONFIG=cluster/auth/kubeconfig oc apply --dry-run=server \
    -f deploy/phase0/vllm-watermark-pod.yaml \
    -f deploy/phase3/detector-build.yaml \
    -f deploy/phase3/detector-deploy.yaml \
    -f deploy/phase3/detector-synthid-deploy.yaml \
    -f deploy/phase3/orchestrator.yaml \
    -f deploy/phase3/nemo-guardrails-poc.yaml -o name
pod/vllm-watermark
Warning: would violate PodSecurity "restricted:latest": allowPrivilegeEscalation != false (container "vllm" must set securityContext.allowPrivilegeEscalation=false), unrestricted capabilities (container "vllm" must set securityContext.capabilities.drop=["ALL"]), runAsNonRoot != true (pod or container "vllm" must set securityContext.runAsNonRoot=true), seccompProfile (pod or container "vllm" must set securityContext.seccompProfile.type to "RuntimeDefault" or "Localhost")
imagestream.image.openshift.io/detector
buildconfig.build.openshift.io/detector
service/detector
deployment.apps/detector
service/detector-synthid
deployment.apps/detector-synthid
configmap/orchestrator-config
service/orchestrator
deployment.apps/orchestrator
configmap/nemo-watermark-config
```

### Live closing state

```
$ KUBECONFIG=cluster/auth/kubeconfig oc -n watermark get deploy -o custom-columns='NAME:.metadata.name,READY:.status.readyReplicas,CURRENT:.status.replicas'
NAME               READY   CURRENT
detector           1       1
detector-synthid   1       1
orchestrator       1       1

$ KUBECONFIG=cluster/auth/kubeconfig oc -n watermark get pods -o custom-columns='NAME:.metadata.name,READY:.status.containerStatuses[*].ready,PHASE:.status.phase'
NAME                                READY   PHASE
bench                               true    Running
detector-1-build                    false   Succeeded
detector-676674bdd4-s45sr           true    Running
detector-synthid-779c8c4f6-6xm6r   true    Running
orchestrator-6d66dbb944-p4k7t      true    Running

$ KUBECONFIG=cluster/auth/kubeconfig oc -n watermark exec bench -- python3 -c 'import urllib.request; print(urllib.request.urlopen("http://orchestrator:8034/health",timeout=10).read().decode()); print(urllib.request.urlopen("http://orchestrator:8034/info",timeout=10).read().decode())'
{"fms-guardrails-orchestr8":"0.16.0"}
{"services":{"watermark-synthid":{"status":"HEALTHY"},"watermark-kgw":{"status":"HEALTHY"}}}

$ KUBECONFIG=cluster/auth/kubeconfig oc -n openshift-machine-api get machineset ocp-ai-p9j4n-gpu-us-east-1a -o custom-columns='NAME:.metadata.name,DESIRED:.spec.replicas,CURRENT:.status.replicas,READY:.status.readyReplicas'
NAME                          DESIRED   CURRENT   READY
ocp-ai-p9j4n-gpu-us-east-1a   0         0         <none>
```

### Newly found detector configuration-validation gap

This is an executed negative probe, not a fix claim. `load_settings()` accepts values
that should make startup/readiness fail. `WATERMARK_Z_THRESHOLD=nan` is the most severe:
comparisons against NaN are false, so it can silently suppress positive verdicts. The
other invalid values fail later while constructing per-request detector configs.

```
$ PYTHONPATH=src /usr/bin/python3 - <<'PY'
from detector.app import load_settings
for name, env in [
    ('nan-threshold', {'WATERMARK_Z_THRESHOLD':'nan'}),
    ('zero-depth', {'VLLM_WATERMARK_SYNTHID_KEY_DEPTH':'0'}),
    ('zero-ngram', {'VLLM_WATERMARK_SYNTHID_NGRAM_LEN':'0'}),
    ('bad-gamma', {'VLLM_WATERMARK_GAMMA':'2'}),
]:
    s = load_settings(env)
    print(name, 'ACCEPTED', repr(s.z_threshold), s.synthid_key_depth,
          s.synthid_ngram_len, s.kgw_gamma)
PY
nan-threshold ACCEPTED nan 30 5 0.25
zero-depth ACCEPTED 4.0 0 5 0.25
zero-ngram ACCEPTED 4.0 30 0 0.25
bad-gamma ACCEPTED 4.0 30 5 2.0
```

## 2026-08-08 — Phase 4 RHOAI evidence index (reconstructed; not an `EXECUTED` transcript)

**Evidence qualification (`OPEN`; source: task-supplied execution summary, 2026-08-08):**
this entry registers results supplied after the work session. Exact commands and raw
terminal output were not supplied to the documentation lane. To avoid inventing either,
the blocks below reproduce only the supplied, secret-free result fields; they are **not
raw output**. Consequently, this entry does not by itself upgrade any fact to
`EXECUTED` under this repository's command-plus-raw-output rule. A future rerun or
recovered transcript may make the narrowly scoped upgrades described below.

### Supplied environment and build result

**Claim (`OPEN`; source: reconstructed evidence index):** the supplied session reports
OpenShift `4.20.27`; RHOAI operator `rhods-operator` `3.4.2` on `stable-3.4` with its
CSV `Succeeded`; and a `DSC` `Ready` for KServe and TrustyAI.

**Supplied result fields (not raw output):**

```
OpenShift: 4.20.27
rhods operator: 3.4.2, channel stable-3.4, CSV Succeeded
DSC: Ready; KServe and TrustyAI enabled
```

**Claim (`OPEN`; source: reconstructed evidence index):** custom image build
`watermark-vllm-2` reportedly succeeded, including package and entry-point smoke;
reported custom-image digest
`sha256:571746d756a6d8671660b98da1f2738616f662822630979e1848b6b1b9ab9683`; reported
pinned base digest
`sha256:5800e12b2a465f15961fcf34b645d79ed4f91ec9161eab22b1205d12682183c8`.
Reported serving version: `vLLM 0.18.0+rhaiv.11`.

### Actual predictor-Service scope

**Claim (`OPEN`; source: reconstructed evidence index):** the supplied session reports
requests to the actual KServe predictor **Service** (not an external KServe gateway).
At 256 tokens, it reports these direct-path detector results:

| scheme/control | z | p | tokens scored |
|---|---:|---:|---:|
| KGW on / true | 8.26558 | 6.95e-17 | 247 |
| KGW off / false | 0.14665 | 0.4417 | 248 |
| SynthID on / true | 17.61158 | 1.00e-69 | 251 |
| SynthID off / false | -0.28217 | 0.61109 | 225 |

This is a reported direct predictor-Service result only. It does **not** establish that
an external KServe gateway or Istio pass-through preserves `vllm_xargs`/`extra_body`;
that question remains `OPEN` (C8). Response bodies, request bodies, keys, and commands
were not supplied and are intentionally absent.

### RHOAI-managed NeMo scope

**Claim (`OPEN`; source: reconstructed evidence index):** the supplied session reports
the RHOAI-managed `NemoGuardrails` CR/action path using server image digest
`sha256:22125dbbd05d1cfa7af931fdeca9da72c6d267a05c33a1f757daf3095ddcca7c`, package
`nemoguardrails==0.21.0`, and Python `3.12.13`. It further reports: an unauthenticated
request returned `401`; KGW-on `/checks` was blocked while the off control passed;
`/chat` returned `200` with a fixed refusal; and a detector outage caused `/checks`
to return `200` with a blocked result before the deployment was restored. These are
reported action outcomes, not an unredacted response-body record.

### Finite retention-marker scan

**Claim (`OPEN`; source: reconstructed evidence index):** after `config.py`, the
supplied retention-marker scan used marker hash
`c15f1804dbddaa2f6b91290107e22da20b01fb2733bd0c7ceacaeb144a7ebec9` and reported zero
matches in its relevant marker, event, and action-log scans. It notes one possible
harmless verbose startup line. This finite scan cannot establish platform-wide content
retention, non-retention, or supportability; those remain `OPEN`.

**Supplied result fields (not raw output):**

```
marker hash: c15f1804dbddaa2f6b91290107e22da20b01fb2733bd0c7ceacaeb144a7ebec9
relevant marker matches: 0
relevant event matches: 0
relevant action-log matches: 0
possible harmless startup verbose line: 1
```

### End-of-session GPU state

**Claim (`OPEN`; source: reconstructed evidence index):** the supplied session reports
the GPU MachineSet was scaled down and last observed at desired/current `0/0` at
`2026-08-08T19:58:27Z`. The scale command and its raw output were not supplied, so this
is not an `EXECUTED` upgrade from this entry.

### Still open

**Claim (`OPEN`; source: facts D9/D10 and reconstructed evidence index):** D9 still
needs the fail-fast configuration fix, image rebuild, and a preserved live rerun. D10
remains unimplemented and unexecuted; neither direct predictor-Service results nor the
reported managed-NeMo checks constitute its fixed-frequency, observability, failure,
latency, retry, and backpressure acceptance evidence.

## 2026-08-08 — Phase 4 RHOAI exact transcript recovered (EXECUTED; redacted)

**Evidence qualification (`EXECUTED`; source: recovered local session transcript
and the raw command/output below):** this is the command/output recovery that the preceding
reconstructed index lacked. Reproduced commands and raw output are verbatim except for
explicit `[REDACTED …]` substitutions for route hosts, pod suffixes, and request-body
scripts; a marked condensed block contains only the load-bearing fields from a larger
raw result. No bearer-token value, request body, generated response text, or key
material is recorded. The recovery upgrades only the scopes stated here.

### Operator, DSC, and API surfaces

**Claim (`EXECUTED`; source: recovered local session transcript and raw output
below):** the RHOAI operator CSV was `Succeeded` at version
3.4.2. The submitted DSC was accepted by server-side dry run and created; its recorded
status was `Ready`, including `KserveReady=True` and `TrustyAIReady=True`. The cluster
advertised `InferenceService`, `ServingRuntime`, and `NemoGuardrails` APIs.

```
$ KUBECONFIG=cluster/auth/kubeconfig oc get csv rhods-operator.3.4.2 \
    -n redhat-ods-operator \
    -o custom-columns='NAME:.metadata.name,PHASE:.status.phase,VERSION:.spec.version,REASON:.status.reason'
NAME                   PHASE       VERSION   REASON
rhods-operator.3.4.2   Succeeded   3.4.2     InstallSucceeded

$ KUBECONFIG=cluster/auth/kubeconfig oc apply --dry-run=server \
    -f deploy/phase4/10-datasciencecluster-minimal.yaml
datasciencecluster.datasciencecluster.opendatahub.io/default-dsc created (server dry run)

$ KUBECONFIG=cluster/auth/kubeconfig oc apply -f deploy/phase4/10-datasciencecluster-minimal.yaml
datasciencecluster.datasciencecluster.opendatahub.io/default-dsc created

$ KUBECONFIG=cluster/auth/kubeconfig oc get datasciencecluster default-dsc -o json | \
    jq '{phase:.status.phase,conditions:[.status.conditions[]? | {type,status,reason,message}],installedComponents:.status.installedComponents}'
[raw output condensed to the load-bearing fields; other component conditions not reproduced]
"phase": "Ready"
{"type":"Ready","status":"True","reason":null,"message":null}
{"type":"ComponentsReady","status":"True","reason":null,"message":null}
{"type":"KserveReady","status":"True","reason":null,"message":null}
{"type":"TrustyAIReady","status":"True","reason":null,"message":null}

$ KUBECONFIG=cluster/auth/kubeconfig oc api-resources | rg 'ServingRuntime|InferenceService|NemoGuardrails|TrustyAI|Predictor'
inferenceservices       isvc     serving.kserve.io/v1beta1       true   InferenceService
servingruntimes                  serving.kserve.io/v1alpha1       true   ServingRuntime
nemoguardrails                   trustyai.opendatahub.io/v1alpha1  true   NemoGuardrails
trustyaiservices                 trustyai.opendatahub.io/v1        true   TrustyAIService
```

### Custom image build

**Claim (`EXECUTED`; source: recovered local session transcript and raw output
below):** the local package
wheel built; the OpenShift custom-image build installed it and successfully imported
the package with both `kgw` and `synthid` entry points; build `watermark-vllm-2`
completed with the recorded output digest. The base image digest was pinned.

```
$ ./deploy/phase0/build-wheel.sh
Building wheels for collected packages: vllm-watermark
  Created wheel for vllm-watermark: filename=vllm_watermark-0.1.0.dev0-py3-none-any.whl size=60916 sha256=2f385fc4e3766173f769ad06a94d10bb198b357e344743f99be12df3afacac05
Successfully built vllm-watermark
Wheel built: dist/vllm_watermark-0.1.0.dev0-py3-none-any.whl

$ [recovered build-output block; originating shell invocation was outside the recovered command set]
STEP 1/7: FROM registry.redhat.io/rhaii/vllm-cuda-rhel9@sha256:5800e12b2a465f15961fcf34b645d79ed4f91ec9161eab22b1205d12682183c8
STEP 4/7: RUN python -m pip install --no-cache-dir --no-deps /tmp/vllm_watermark-*.whl [command continued]
Successfully installed vllm-watermark-0.1.0.dev0
vllm-watermark 0.1.0.dev0 ['kgw', 'synthid']

$ KUBECONFIG=cluster/auth/kubeconfig oc get build watermark-vllm-2 -n watermark \
    -o custom-columns='PHASE:.status.phase,OUTPUT:.status.outputDockerImageReference,DIGEST:.status.output.to.imageDigest'
PHASE      OUTPUT                                                                            DIGEST
Complete   image-registry.openshift-image-registry.svc:5000/watermark/watermark-vllm:0.1.0   sha256:571746d756a6d8671660b98da1f2738616f662822630979e1848b6b1b9ab9683
```

### KServe ServingRuntime / InferenceService / predictor Service

**Claim (`EXECUTED`; source: recovered local session transcript and raw output
below):** the actual RHOAI
KServe `InferenceService` `watermark-vllm` became Ready in Standard mode using runtime
`watermark-vllm`; its internal predictor Service answered `/health` and `/v1/models`.
Four 256-token direct-predictor cases showed positive KGW/SynthID results and clean
off-controls. This is internal predictor-Service evidence, **not** external
KServe-gateway/Istio pass-through evidence.

```
$ KUBECONFIG=cluster/auth/kubeconfig oc wait --for=condition=Ready \
    inferenceservice/watermark-vllm -n watermark --timeout=45s && \
  KUBECONFIG=cluster/auth/kubeconfig oc get inferenceservice watermark-vllm -n watermark \
    -o custom-columns='READY:.status.conditions[?(@.type=="Ready")].status,MODE:.status.deploymentMode,RUNTIME:.status.servingRuntimeName,URL:.status.url'
inferenceservice.serving.kserve.io/watermark-vllm condition met
READY   MODE       RUNTIME          URL
True    Standard   watermark-vllm   http://watermark-vllm-predictor.watermark.svc.cluster.local

$ KUBECONFIG=cluster/auth/kubeconfig oc exec -n watermark bench -- python3 -c 'import urllib.request; base="http://watermark-vllm-predictor:8080"; print("health", urllib.request.urlopen(base+"/health", timeout=10).status); print("models_status", urllib.request.urlopen(base+"/v1/models", timeout=10).status)'
health 200
models_status 200

$ KUBECONFIG=cluster/auth/kubeconfig oc exec -i -n watermark bench -- python3 - <<'PY'
[REDACTED REQUEST-BODY SCRIPT: four 256-token direct predictor requests and detector checks; output emits hashes/metrics only]
PY
{"case":"kgw-on","completion_tokens":256,"content_bytes":1478,"content_sha256":"a8f017d22d10047510f3a2bcd5e6c24df29979c0ed2913f7bcc46c4b6a2ab486","detector_http":200,"detector_ms_client":1265.64,"finish_reason":"length","generation_http":200,"generation_ms":1445.3,"key_id":"poc-2026-08","num_tokens_scored":247,"p_value":6.950769705311311e-17,"scheme":"kgw","verdict":true,"z_score":8.265581531669756}
{"case":"kgw-off","completion_tokens":256,"content_bytes":1390,"content_sha256":"f9145db58ef17f982d441ac6b155c0be8f0397f48fbc14b9ebda8f80bbd47627","detector_http":200,"detector_ms_client":1100.78,"finish_reason":"length","generation_http":200,"generation_ms":922.83,"key_id":"poc-2026-08","num_tokens_scored":248,"p_value":0.44170528162986755,"scheme":"kgw","verdict":false,"z_score":0.1466471150213533}
{"case":"synthid-on","completion_tokens":256,"content_bytes":1527,"content_sha256":"9c21d6d854145d26abc6545bd61efbdf94a2f8847a7ffa6d8c782cb0c2478206","detector_http":200,"detector_ms_client":29.85,"finish_reason":"length","generation_http":200,"generation_ms":1924.72,"key_id":"poc-2026-08","num_tokens_scored":251,"p_value":1.0037381122735403e-69,"scheme":"synthid","verdict":true,"z_score":17.611584050973715}
{"case":"synthid-off","completion_tokens":256,"content_bytes":1220,"content_sha256":"a58531fe66421b1fc05379e5c95c5a10d17878f7b8f56c2a8d3001bd668380f6","detector_http":200,"detector_ms_client":30.27,"finish_reason":"length","generation_http":200,"generation_ms":916.16,"key_id":"poc-2026-08","num_tokens_scored":225,"p_value":0.611091738577513,"scheme":"synthid","verdict":false,"z_score":-0.28216561486796143}
```

### Managed NeMo CR/action and controlled outage

**Claim (`EXECUTED`; source: recovered local session transcript and raw output
below):** the
RHOAI-managed `NemoGuardrails` CR reached phase `Ready`, using server image digest
`sha256:22125dbbd05d1cfa7af931fdeca9da72c6d267a05c33a1f757daf3095ddcca7c`. Its guarded
checks blocked a KGW-on sample and passed the off control. With the detector scaled to
zero ready endpoints, the configured closed policy returned HTTP 200 with a blocked
result, then the detector deployment was restored. These are the former managed-action
semantics; the current D10 correlation/broker design is separately `STATIC` and
unexecuted.

```
$ KUBECONFIG=cluster/auth/kubeconfig oc get nemoguardrails nemo-watermark -n watermark -o json | jq '{generation:.metadata.generation,status:.status}'
{
  "generation": 1,
  "status": {
    "conditions": [
      {"message":"Deployment is ready","reason":"DeploymentReady","status":"True","type":"DeploymentReady"},
      {"message":"Route is ready","reason":"RouteReady","status":"True","type":"RouteReady"},
      {"message":"Reconcile completed successfully","reason":"ReconcileComplete","status":"True","type":"ReconcileComplete"}
    ],
    "phase": "Ready"
  }
}

$ KUBECONFIG=cluster/auth/kubeconfig oc get deployment,pod,service,route -n watermark | rg 'nemo-watermark|NAME'
deployment.apps/nemo-watermark   0/1   1   0   [startup observation]
pod/nemo-watermark-7bd4b4ccc7-[REDACTED POD SUFFIX]   0/2   Init:0/1
service/nemo-watermark   ClusterIP   [REDACTED CLUSTER IP]   <none>   443/TCP
route.route.openshift.io/nemo-watermark   [REDACTED ROUTE HOST]
registry.redhat.io/rhoai/odh-trustyai-nemo-guardrails-server-rhel9@sha256:22125dbbd05d1cfa7af931fdeca9da72c6d267a05c33a1f757daf3095ddcca7c

$ guardrails_route="https://$(KUBECONFIG=cluster/auth/kubeconfig oc get route nemo-watermark -n watermark -o jsonpath='{.spec.host}')"
$ guardrails_token=$(KUBECONFIG=cluster/auth/kubeconfig oc create token nemo-watermark-serviceaccount -n watermark --duration=10m)
$ curl --fail-with-body --silent --show-error --insecure -H "Authorization: Bearer $guardrails_token" "$guardrails_route/openapi.json" | jq '{title:.info.title,version:.info.version,paths:(.paths|keys)}'
$ unset guardrails_token guardrails_route
[temporary bearer token value and route host were never printed]
{
  "title": "Guardrails Server API",
  "version": "0.1.0",
  "paths": ["/","/v1/challenges","/v1/chat/completions","/v1/guardrail/checks","/v1/models","/v1/rails/configs"]
}

$ KUBECONFIG=cluster/auth/kubeconfig python3 - <<'PY'
[REDACTED REQUEST-BODY SCRIPT: obtains temporary bearer token, sends generated hashes only to /v1/guardrail/checks, and emits redacted result metadata]
PY
{"case":"kgw-on","completion_tokens":256,"content_bytes":1457,"content_redacted":true,"content_sha256":"1d5eb16d95ea221a84c8aa630131dd55db90bf797804f8094211f5fa402bcaf1","generation_http":200,"guardrails_data_keys":["log"],"guardrails_http":200,"guardrails_ms":2411.82,"guardrails_status":"blocked","message_results":[{"index":0,"rails":{"watermark check":"blocked"},"role":"assistant"}],"rails_status":{"watermark check":"blocked"}}
{"case":"kgw-off","completion_tokens":256,"content_bytes":1336,"content_redacted":true,"content_sha256":"8fbdfe9ee560ac39826ef4fd865a21e770776d78edd087c44aeaea1bb4fd490b","generation_http":200,"guardrails_data_keys":["log"],"guardrails_http":200,"guardrails_ms":1270.19,"guardrails_status":"success","message_results":[{"index":0,"rails":{"watermark check":"success"},"role":"assistant"}],"rails_status":{"watermark check":"success"}}

$ [controlled detector-outage command; temporary bearer token generated and restored deployment, value never printed; request-body script redacted]
detector_replicas_before=1
deployment.apps/detector scaled
pod/detector-[REDACTED POD SUFFIX] condition met
detector_ready_endpoints=0
{"configured_failure_policy":"closed","content_redacted":true,"detector_state":"zero-ready-endpoints","elapsed_ms":325.64,"http":200,"rails_status":{"watermark check":"blocked"},"status":"blocked"}
Waiting for deployment "detector" rollout to finish: 0 of 1 updated replicas are available...
deployment "detector" successfully rolled out
DESIRED   READY   AVAILABLE
1         1       1
```

### Finite retention scan and billable-resource shutdown

**Claim (`EXECUTED`; source: recovered local session transcript and raw output
below):** the stated post-
`config.py` scan found zero matches for its one marker in the sampled managed-NeMo,
detector, event, and action-log surfaces. This is a finite, time-bounded scan only; it
does not prove platform-wide non-retention or supportability, which remain `OPEN`.

```
$ [redacted retention-probe request-body script, followed by bounded log scans]
{"content_bytes":8921,"content_redacted":true,"content_sha256":"c15f1804dbddaa2f6b91290107e22da20b01fb2733bd0c7ceacaeb144a7ebec9","http":200,"rails_status":{"watermark check":"success"},"status":"success"}
RETENTION_PROBE_20260808_7b2e15 matches=0
Event ContextUpdate matches=0
StartInternalSystemAction matches=0
Executing action watermark_check matches=0
detector-676674bdd4-[REDACTED POD SUFFIX] marker_matches=0
detector-synthid-779c8c4f6-[REDACTED POD SUFFIX] marker_matches=0

$ date -u '+%Y-%m-%dT%H:%M:%SZ' && ./scripts/scale-gpu.sh 0 && \
  KUBECONFIG=cluster/auth/kubeconfig oc get machineset ocp-ai-p9j4n-gpu-us-east-1a \
    -n openshift-machine-api \
    -o custom-columns='NAME:.metadata.name,DESIRED:.spec.replicas,CURRENT:.status.replicas,READY:.status.readyReplicas'
2026-08-08T19:58:27Z
machineset.machine.openshift.io/ocp-ai-p9j4n-gpu-us-east-1a scaled
NAME                          DESIRED   CURRENT   READY
ocp-ai-p9j4n-gpu-us-east-1a   0         0         <none>
```

### Remaining boundaries

**Claim (`OPEN`; source: D9/D10 acceptance criteria):** D9's code fix, fixed-image
rebuild, and live detector-matrix rerun remain open. D10 remains open: this recovered
evidence does not establish fixed-frequency selection, end-to-end correlation metadata,
the current broker action, delivery-mode policy, retry/backpressure, metrics, or the
full observability acceptance matrix. C8's external gateway/Istio pass-through is also
unexecuted.

## 2026-08-08 — Phase 5 gateway base-image digest resolution (EXECUTED)

**Claim (`EXECUTED`; source: raw command/output below):** the official Docker registry
resolved the linux/amd64 `python:3.12-slim` reference to
`sha256:229a2c5bfa27522db7815ea81f9bed70af17ccb9de9fc7ad142b1877b5830d36`, and a
second inspection resolved that immutable reference for linux/amd64. This verifies only
the base-image reference used by `deploy/phase5/Containerfile`; the gateway image has
not yet been built, scanned, deployed, or executed.

```text
$ skopeo inspect --override-os linux --override-arch amd64 --format '{{.Digest}} {{.Name}} {{.Architecture}} {{.Os}}' docker://docker.io/library/python:3.12-slim
sha256:229a2c5bfa27522db7815ea81f9bed70af17ccb9de9fc7ad142b1877b5830d36 docker.io/library/python amd64 linux

$ skopeo inspect --override-os linux --override-arch amd64 --format '{{.Digest}} {{.Architecture}} {{.Os}}' docker://docker.io/library/python@sha256:229a2c5bfa27522db7815ea81f9bed70af17ccb9de9fc7ad142b1877b5830d36
sha256:229a2c5bfa27522db7815ea81f9bed70af17ccb9de9fc7ad142b1877b5830d36 amd64 linux
```

## 2026-08-09 — Phase 4 current managed path and D10 continuous validation (EXECUTED; redacted)

**Evidence qualification (`EXECUTED`; source: recovered local session transcript
and the raw command/output below):**
the current metadata-only gateway → RHOAI-managed NeMo action → authenticated
broker → detector path and the one-in-`N` synchronous validation contract ran on
the recorded OpenShift AI environment. Commands and content-safe output are
preserved below. No generated response text, prompt, bearer-token value,
watermark key, signing key, model token, route host, cluster IP, or pod suffix is
recorded.

The harness emitted 40 hash-only selected-record rows in addition to its aggregate
object. The aggregate fields below are verbatim; only the two repetitive
`record_evidence` arrays are structurally condensed to their exact row counts.
The retained projection also records the four exact scheme/verdict/action counts
derived from each array. The full content-safe stdout remains in the cited local
session transcript. This scoped evidence does not establish external
KServe/Istio pass-through, product supportability, platform-wide non-retention,
multi-replica/global sampling, streaming/asynchronous behavior, or production
network/authentication policy; those remain `OPEN`.

This later append-only entry supersedes only the earlier time-bounded statement in
“Phase 5 gateway base-image digest resolution” that the gateway image had not yet
been built or deployed; the earlier registry-resolution output remains valid.

### Immutable build and deployed runtime

**Claim (`EXECUTED`; source: raw build/deployment output below):** OpenShift build
`watermark-validation-gateway-5` completed and pushed the gateway image at digest
`sha256:9443345c272c55a6dcea41bae84f8e04bab5d0f27e0934df2e4b4ff11b37fed6`.
The deployed gateway used that immutable digest with one replica, `Recreate`, a
bound `ReadWriteOnce` PVC, an arbitrary admitted UID, a read-only root filesystem,
a mode-`600` SQLite database, and the mounted OpenShift service CA. The current
managed-NeMo image was
`sha256:22125dbbd05d1cfa7af931fdeca9da72c6d267a05c33a1f757daf3095ddcca7c`
with `nemoguardrails==0.21.0`. The vLLM and detector images used in the matrix were
respectively
`sha256:f8294ee0459869e9659b1178ed91f57a1b52a52c6a5f5f819ca651646b317e4c`
and
`sha256:e13e51ca3ec5f578b78aa5882431789d9cdc891cb4ae239909cbdd7c56bf3520`.

The binary build context was a fresh `/tmp/vllm-watermark-gateway-build.*`
directory containing only the reviewed `Containerfile`, entrypoint,
`validation/requirements.txt`, and four runtime modules. The repository root and
its ignored credential paths were not submitted.

```text
$ KUBECONFIG=cluster/auth/kubeconfig oc start-build \
    watermark-validation-gateway -n watermark \
    --from-dir="$build_context" --follow --wait
[intermediate package-download and layer-copy output omitted]
Successfully installed annotated-types-0.8.0 anyio-4.14.2 certifi-2026.7.22 click-8.4.2 fastapi-0.115.12 h11-0.16.0 httpcore-1.0.9 httpx-0.28.1 idna-3.18 pydantic-2.13.4 pydantic-core-2.46.4 starlette-0.46.2 typing-extensions-4.16.0 typing-inspection-0.4.2 uvicorn-0.34.3
STEP 8/14: RUN chmod 0555 /usr/local/bin/validation-gateway-entrypoint     && find /opt/app-root/src/validation -type d -exec chmod 0555 {} +     && find /opt/app-root/src/validation -type f -exec chmod 0444 {} +     && python -c 'import fastapi, httpx, uvicorn; import validation.gateway, validation.http_clients, validation.main'
STEP 12/14: CMD ["python","-m","validation.main"]
Successfully pushed image-registry.openshift-image-registry.svc:5000/watermark/watermark-validation-gateway@sha256:9443345c272c55a6dcea41bae84f8e04bab5d0f27e0934df2e4b4ff11b37fed6
Push successful
GATEWAY_IMAGE_REFERENCE=image-registry.openshift-image-registry.svc:5000/watermark/watermark-validation-gateway@sha256:9443345c272c55a6dcea41bae84f8e04bab5d0f27e0934df2e4b4ff11b37fed6

$ [content-safe build, image, deployment, PVC, pod-runtime, and NeMo status queries]
--- build ---
watermark-validation-gateway-5   Complete   2026-08-08T23:35:53Z   2026-08-08T23:36:15Z   image-registry.openshift-image-registry.svc:5000/watermark/watermark-validation-gateway:0.1.0
--- image ---
image-registry.openshift-image-registry.svc:5000/watermark/watermark-validation-gateway@sha256:9443345c272c55a6dcea41bae84f8e04bab5d0f27e0934df2e4b4ff11b37fed6    sha256:9443345c272c55a6dcea41bae84f8e04bab5d0f27e0934df2e4b4ff11b37fed6
--- deployment ---
Recreate    1    1    image-registry.openshift-image-registry.svc:5000/watermark/watermark-validation-gateway@sha256:9443345c272c55a6dcea41bae84f8e04bab5d0f27e0934df2e4b4ff11b37fed6    true    false
--- pvc ---
watermark-validation-gateway-state   Bound   ReadWriteOnce   2Gi
--- pod-runtime ---
uid=1000750000
ca_bytes=1212
db_mode=600
root_write=denied
--- nemo ---
2    Ready    null    null
```

### Strict startup configuration probes

**Claim (`EXECUTED`; source: the raw gateway and detector probe/recovery output
below):** the gateway refused an invalid sampler
value before readiness and recovered with test controls off. The detector image
refused the non-finite threshold `NaN` before readiness and recovered at the
expected immutable image. The harness separately accepted `1` and `5` and
rejected `0`, `-1`, `1.5`, empty, and nonnumeric sampler values.

The first detector cleanup attempted to re-apply the declarative manifest.
Because `oc apply` does not remove an environment item introduced imperatively
by `oc set env`, that cleanup timed out at `ready=0`; an explicit
`WATERMARK_Z_THRESHOLD-` removal immediately restored `ready=1`. The corrected,
bounded rerun below is the authoritative paired negative/recovery result.

```text
$ [bounded gateway startup probe with VALIDATION_SAMPLE_EVERY=0; Secret values never printed]
GATEWAY_INVALID_STARTUP invalid_ready=0 waiting_reason=CrashLoopBackOff restart_count=1 config_error_matches=1
INVALID_CONFIG_CLEANUP ready=1 controls=off

$ KUBECONFIG=cluster/auth/kubeconfig oc -n watermark set env \
    deployment/detector WATERMARK_Z_THRESHOLD=NaN
$ [bounded readiness/log observation]
$ KUBECONFIG=cluster/auth/kubeconfig oc -n watermark set env \
    deployment/detector WATERMARK_Z_THRESHOLD-
$ KUBECONFIG=cluster/auth/kubeconfig oc -n watermark rollout status \
    deployment/detector --timeout=5m
DETECTOR_INVALID_STARTUP invalid_ready=0 waiting_reason=Unknown restart_count=1 config_error_matches=4
DETECTOR_INVALID_CLEANUP ready=1 image_digest_match=true
```

### Actual RHOAI endpoint and continuous-validation matrix

**Claim (`EXECUTED`; source: the raw precheck, endpoint, and harness output
below):** the precheck found the RHOAI CSV/DSC and
managed-NeMo resource ready, the current action and deployed runtime-source
hashes equal to local reviewed sources, both detector deployments ready, and no
GPU node. The bounded runner scaled one GPU node, waited for the actual KServe
`InferenceService` and predictor rollout, verified `/health` and `/v1/models`,
and checked the expected vLLM image and installed SynthID source before running
the gateway harness.

```text
$ [bounded acceptance runner; temporary credentials loaded into environment variables and never printed]
PRECHECK gpu=0/0/0 controls=off gateway_ready=1 detector_ready=1 synthid_ready=1 action_hash_match=true runtime_source_match=true csv=Succeeded dsc_ready=True nemo=Ready
GPU_READY desired=1 current=1 ready=1
inferenceservice.serving.kserve.io/watermark-vllm condition met
deployment "watermark-vllm-predictor" successfully rolled out
RHOAI_ENDPOINT isvc_ready=true predictor_health=200 models=200 image_digest_match=true pod_image_id_present=true synthid_source_match=true
ACCEPTANCE_START gateway_health=200 gateway_ready=200 controls=on secret_values_redacted=true

$ python benchmarks/continuous_validation.py \
    --gateway-url http://127.0.0.1:18080 \
    --model watermark-vllm \
    --key-id poc-2026-08 \
    --admin-token-env D10_ADMIN_TOKEN \
    --secret-marker-env D10_ADMIN_TOKEN \
    --secret-marker-env D10_BROKER_TOKEN \
    --secret-marker-env D10_NEMO_TOKEN \
    --secret-marker-env D10_WATERMARK_KEY \
    --secret-marker-env D10_SIGNING_KEY \
    --secret-marker-env D10_MODEL_TOKEN \
    --max-tokens 256 --temperature 0.7 \
    --timeout-seconds 120 --queue-pending-check-seconds 0.5 \
    --positive-action block --clean-action pass \
    --expected-mode synchronous --expected-failure-policy closed
{
  "configuration": {
    "0": "rejected",
    "1": "accepted",
    "5": "accepted",
    "empty": "rejected",
    "fraction": "rejected",
    "negative": "rejected",
    "nonnumeric": "rejected"
  },
  "content_logged": false,
  "contract": "phase5-v1",
  "faults": {
    "malformed_success": {"attempts": 1, "retries": 0, "terminal_state": "malformed_response"},
    "retry_exhausted": {"attempts": 3, "retries": 2, "terminal_state": "retry_exhausted"},
    "retry_then_success": {"attempts": 2, "retries": 1, "terminal_state": "success"}
  },
  "latency_semantics": {
    "client_delivery": "request_start_to_gateway_delivery",
    "generation_completion": "request_start_to_upstream_completion",
    "validation": "validation_attempt_window",
    "validation_lag": "validation_queue_wait_to_attempt_start"
  },
  "n1": {
    "client_delivery_latency": {"count": 20, "p50_seconds": 0.898447910505638, "p95_seconds": 1.61472814619483, "p99_seconds": 1.9217282948474161},
    "counters": {"cancelled": 0, "clean": 10, "completed": 20, "detector_attempts": 20, "dropped": 0, "errors": 0, "failed": 0, "guardrails_attempts": 20, "queue_overflow": 0, "retries": 0, "selected": 20, "started": 20, "terminal": 20, "unsampled": 0, "watermarked": 10},
    "generation_completion_latency": {"count": 20, "p50_seconds": 0.4827059195013135, "p95_seconds": 1.0066926561528817, "p99_seconds": 1.0252088120230474},
    "latency_samples": {"client_delivery": 20, "generation_completion": 20, "validation": 20, "validation_lag": 20},
    "queue_depth": 0,
    "record_evidence": {"rows": 20, "projection": "structurally condensed; all per-row assertions passed in the harness"},
    "responses": 20,
    "selected": 20,
    "terminal": 20,
    "validation_lag": {"count": 20, "p50_seconds": 0.000042261504859197885, "p95_seconds": 0.00005339034469216131, "p99_seconds": 0.00005477887039887719},
    "validation_latency": {"count": 20, "p50_seconds": 0.15892895850265631, "p95_seconds": 0.7875182849616978, "p99_seconds": 0.9465901297942034}
  },
  "n5": {
    "client_delivery_latency": {"count": 100, "p50_seconds": 0.4950446209986694, "p95_seconds": 1.523351200445904, "p99_seconds": 1.8363652998648479},
    "counters": {"cancelled": 0, "clean": 10, "completed": 100, "detector_attempts": 20, "dropped": 0, "errors": 0, "failed": 0, "guardrails_attempts": 20, "queue_overflow": 0, "retries": 0, "selected": 20, "started": 100, "terminal": 20, "unsampled": 80, "watermarked": 10},
    "generation_completion_latency": {"count": 100, "p50_seconds": 0.4773876684921561, "p95_seconds": 0.9267352951959765, "p99_seconds": 1.0038456328251084},
    "latency_samples": {"client_delivery": 100, "generation_completion": 100, "validation": 20, "validation_lag": 20},
    "queue_depth": 0,
    "record_evidence": {"rows": 20, "projection": "structurally condensed; all per-row assertions passed in the harness"},
    "responses": 100,
    "selected": 20,
    "terminal": 20,
    "validation_lag": {"count": 20, "p50_seconds": 0.00004648699541576207, "p95_seconds": 0.00006120179459685461, "p99_seconds": 0.00006964995787711813},
    "validation_latency": {"count": 20, "p50_seconds": 0.3162148160045035, "p95_seconds": 0.9328927019509139, "p99_seconds": 1.018073516388249}
  },
  "observability": {"marker_count": 6, "required_metric_count": 10},
  "passed": true,
  "policy_semantics": {
    "gateway_positive_delivery": "flag",
    "managed_guardrails_positive_action": "block",
    "mode": "synchronous",
    "validation_failure": "closed"
  },
  "queue": {"overflow_policy": "non_blocking", "peak_depth": 2, "queue_depth": 0, "queue_overflow": 1, "terminal_records": 3, "validated_records": 2},
  "unsampled_baseline": {
    "client_delivery_latency": {"count": 4, "p50_seconds": 0.4997960870023235, "p95_seconds": 0.8247467453416903, "p99_seconds": 0.8637641562602949},
    "counters": {"cancelled": 0, "clean": 0, "completed": 4, "detector_attempts": 0, "dropped": 0, "errors": 0, "failed": 0, "guardrails_attempts": 0, "queue_overflow": 0, "retries": 0, "selected": 0, "started": 4, "terminal": 0, "unsampled": 4, "watermarked": 0},
    "generation_completion_latency": {"count": 4, "p50_seconds": 0.4966351850016508, "p95_seconds": 0.8214526599957026, "p99_seconds": 0.8604468423940126},
    "latency_samples": {"client_delivery": 4, "generation_completion": 4, "validation": 0, "validation_lag": 0},
    "queue_depth": 0,
    "responses": 4,
    "sample_every": 5,
    "selected": 0
  }
}
```

The structurally condensed selected-record arrays had these exact counts in both
the `N=1` and `N=5` runs:

```json
[
  {"scheme":"kgw","verdict":false,"guardrails_action":"pass","count":5},
  {"scheme":"kgw","verdict":true,"guardrails_action":"block","count":5},
  {"scheme":"synthid","verdict":false,"guardrails_action":"pass","count":5},
  {"scheme":"synthid","verdict":true,"guardrails_action":"block","count":5}
]
```

Each of the 20 selected rows in each run asserted the response ID and SHA-256
digest, scheme, non-secret key ID, verdict, managed/action outcome, attempts and
timings, and exact equality of validation, detector-call, and guardrails-action
IDs. The pure unsampled baseline used `N=5` ordinals 1–4 and made zero detector
or guardrails calls. The queue case paused a capacity-two consumer, retained
synchronous request semantics, admitted two requests, rejected the third under
the configured overflow policy, terminalized all three records, and returned to
depth zero.

### Finite retention scan, real detector outage, and recovery

**Claim (`EXECUTED`; source: the raw recovery command/output below):** a separate
hash-only probe verified exact
content-digest and response/validation-ID correlation. A finite scan then found
zero matches for that response marker and each of six actual current secret
values across eight named surfaces. This is a finite bounded scan only, not a
platform-wide non-retention claim. Scaling the real detector to zero caused the
selected request to exhaust three attempts/two retries and return content-free
HTTP 503 under the configured closed policy; the detector was restored.

```text
RECOVERY_PRECHECK gpu=1/1/1 controls=off detector_ready=1
inferenceservice.serving.kserve.io/watermark-vllm condition met
deployment "watermark-vllm-predictor" successfully rolled out
RECOVERY_RHOAI_READY isvc=true
{"recovery_retention_probe":{"attempts_to_echo":1,"http":200,"content_bytes":83,"content_sha256":"d99ae2304fb4330153c17ae85f645025dcf3836be4f41882940ea7f57430ed3c","digest_matches":true,"selected":true,"verdict":false,"managed_action":"success","guardrails_action":"pass","ids_correlated":true,"response_id_correlated":true,"plaintext_fields":false}}
{"surface":"gateway-logs","response_marker_matches":0,"secret_matches":0}
{"surface":"nemo-logs","response_marker_matches":0,"secret_matches":0}
{"surface":"predictor-logs","response_marker_matches":0,"secret_matches":0}
{"surface":"detector-logs","response_marker_matches":0,"secret_matches":0}
{"surface":"detector-synthid-logs","response_marker_matches":0,"secret_matches":0}
{"surface":"kubernetes-events","response_marker_matches":0,"secret_matches":0}
{"surface":"gateway-metrics","response_marker_matches":0,"secret_matches":0}
{"surface":"gateway-redacted-events","response_marker_matches":0,"secret_matches":0}
RECOVERY_RETENTION_SCAN complete=true surfaces=8
{"recovery_real_detector_outage":{"attempts":3,"delivery_outcome":"fail_closed","detector_attempts":3,"guardrails_attempts":3,"http":503,"plaintext_fields":false,"queue_depth":0,"retries":2,"selected":1,"terminal":1,"terminal_state":"retry_exhausted"}}
RECOVERY_DETECTOR_RESTORED ready=1
RECOVERY_FINAL_CLEANUP controls=off detector_ready=1 gpu_desired=0 gpu_current=0 gpu_ready=0 run_finished=1 cleanup_ok=1
```

A first post-run scan used the wrong detector label selector and stopped before
covering every intended surface; none of that partial scan is counted above. Its
exit cleanup nevertheless returned controls to `off`, restored the detector, and
scaled the GPU MachineSet to zero:

```text
FINAL_CLEANUP controls=off detector_ready=1 gpu_desired=0 gpu_current=0 gpu_ready=0 run_finished=0 cleanup_ok=1
```

### Execution incidents and resource safety

**Claim (`EXECUTED`; source: quota failure output and recovery transcript):** a
postflight retry requested a new `g5.xlarge` while the prior instance teardown
still consumed the account's four-vCPU bucket. AWS rejected the replacement; no
instance was created. A later retry succeeded after quota release. One automatic
continuation was terminated before its shell `EXIT` trap after that successful
provisioning; the next turn detected GPU `1/1/1`, took explicit cleanup ownership,
completed the retention/outage probes above, and returned the GPU to `0/0/0`.

```text
{
  "phase": "Failed",
  "errorReason": "InvalidConfiguration",
  "errorMessage": "error launching instance: You have requested more vCPU capacity than your current vCPU limit of 4 allows for the instance bucket that the specified instance type belongs to. Please visit http://aws.amazon.com/contact-us/ec2-request to request an adjustment to this limit.",
  "conditions": [
    {"type":"InstanceExists","status":"False","reason":"InstanceNotCreated","message":"Instance has not been created"}
  ]
}
POST_FINAL_CLEANUP controls=off detector_ready=1 gpu_desired=0 gpu_current=0 gpu_ready=0 run_finished=0 cleanup_ok=1
RECOVERY_PRECHECK gpu=1/1/1 controls=off detector_ready=1
RECOVERY_FINAL_CLEANUP controls=off detector_ready=1 gpu_desired=0 gpu_current=0 gpu_ready=0 run_finished=1 cleanup_ok=1
```

During an earlier diagnostic traceback one now-retired admin credential value was
printed to the private session console. The entire gateway-auth Secret was rotated
immediately, and the old credential no longer authenticated. No value is reproduced
here. The acceptance run and six-secret scan used only the rotated credentials.

### Local verification at the executed revision

**Claim (`EXECUTED`; source: raw local test output):** the final source revision
used for deployment passed the complete local suite. Focused gateway/action and
detector suites also passed; warnings were third-party Python 3.14/FastAPI/uvicorn
deprecations, not test failures.

```text
$ pytest -q validation/tests tests/test_phase4_nemo_actions.py
106 passed, 73 warnings in 3.49s
{"content_logged": false, "fault_cases": 3, "n1": {"responses": 20, "selected": 20}, "n5": {"responses": 100, "selected": 20}, "queue_overflow_checked": true, "self_test": "passed"}

$ pytest -q detector/tests/test_service.py
58 passed, 276 warnings in 124.10s (0:02:04)

$ pytest -q
213 passed, 276 warnings in 395.48s (0:06:35)
```

### Scoped conclusion and remaining boundaries

**Claim (`EXECUTED`; source: the command/output record above):** D10 acceptance
items 1–8 are executed for the current single-replica, synchronous, non-streaming,
fail-closed deployment: strict configuration, exact `N=1` and `N=5` selection,
both KGW and SynthID positive/clean outcomes through managed NeMo and the broker,
latency distributions, bounded retry terminal states, bounded backpressure,
hash-only correlation/metrics, finite secret-marker scans, and exact counter
reconciliation. The gateway positive-result delivery policy was `flag`; the
managed action's positive result was `blocked`. These are distinct layers.

**Claim (`OPEN`; source: D4/D6/D10 and the bounded scope above):** key lifecycle
and application scoping; product supportability; external KServe gateway/Istio
pass-through; caller authentication/authorization, Route, NetworkPolicy, and mTLS;
multi-replica/global ordinal and restart/rollout semantics; streaming and
asynchronous modes; HA/PDB behavior; and platform-wide retention remain outside
this executed PoC scope. The single-replica `Recreate`/RWO deployment is not HA.

### Independent final postflight

**Claim (`EXECUTED`; source: raw command/output below):** after documentation
reconciliation, the four local managed-action ConfigMap data values still matched
the live cluster byte for byte; the four deployed gateway runtime modules still
matched local source; test controls were off; the gateway, both detector deployments,
and managed-NeMo resource were ready; and the billable GPU MachineSet remained at
`0/0/0`.

```text
$ [hash-only local-versus-cluster ConfigMap/runtime comparison plus bounded readiness query]
ACTION_DATA key=actions.py hash_match=true sha256=9294ebab8d27ee542d66e8dbb9093b39033e45c244570610451761fd7d39c6bd
ACTION_DATA key=config.py hash_match=true sha256=e7a36150275601fb99df32e480c705e5e8d8426f8d94223fd5e0e1429b7246ee
ACTION_DATA key=config.yaml hash_match=true sha256=da052dc808a83dd39484f40503466e6b8d2ab90ed29a96d3880142694b7a2e00
ACTION_DATA key=rails.co hash_match=true sha256=53b4b1dc3919a220e890f0896f3cb72643aa02f4ee1792d203064e0650219bdb
FINAL_POSTFLIGHT controls=off gateway_ready=1 detector_ready=1 synthid_ready=1 nemo=Ready action_data_match=true runtime_source_match=true gpu=0/0/0
```

## 2026-08-09 — Post-execution adversarial review follow-ups (STATIC/OPEN)

**Component-counter qualification (`STATIC`/`OPEN`; source: the executed fixed-run
and real detector-outage records above, plus the validation acceptance criteria in
[`docs/implementation.md`](docs/implementation.md#continuous-validation-acceptance))**:
component attempt counters are evidenced for the successful managed path and the
real detector-outage path. Their semantics for pre-action transport/schema failures
and for injected-fault paths remain `OPEN`; broader counter-contract regressions are
not implied by the executed aggregate counters.

**Managed-action response-read hardening (`STATIC`/`OPEN`; source: the embedded
`_broker_validate` implementation in
[`deploy/phase4/30-nemo-watermark-config.yaml`](deploy/phase4/30-nemo-watermark-config.yaml))**:
the managed-NeMo action reads the broker response with an unbounded `response.read()`.
A byte-bounded response read and its production validation remain `OPEN` hardening
work. The literal ConfigMap action data was not changed in this review.

**Regression coverage (`OPEN`; source: the executed fixed-run/outage scope above and
the remaining acceptance criteria in
[`docs/implementation.md`](docs/implementation.md#continuous-validation-acceptance))**:
broader concrete transport, metadata-mismatch/schema, and cancellation regressions
remain to be executed.

**Evidence boundary (`STATIC`; source: the executed fixed-run and real detector-outage
records above):** these follow-ups do not invalidate the executed fixed-run or real
detector-outage evidence; they qualify its untested hardening boundaries.

## 2026-08-09 — Detector upper-bound rebuild and startup recovery (EXECUTED; redacted)

### Why this follow-up was required

**Correction (`STATIC`/`EXECUTED`; source: independent source review and the commands
below):** detector image
`sha256:e13e51ca3ec5f578b78aa5882431789d9cdc891cb4ae239909cbdd7c56bf3520`
rejected non-finite and out-of-domain settings, but several numeric settings still
had no explicit upper bound. The earlier D9 wording "finite and bounded" was
therefore too broad. The current revision adds service-side maxima, validates the
tokenizer-derived vocabulary fallback, rejects an empty effective KGW green list,
and expands startup and secret-redaction tests. These maxima are implementation
safety policy, not algorithmic or vendor-defined limits.

The current source accepts the following inclusive service maxima: vocabulary
`2**20`, z threshold `100`, KGW delta `100`, SynthID n-gram length `1024`,
context-history size `2**16`, key depth `256`, sampling-table size `2**24`, and
PyTorch seed range `[-2**63, 2**64-1]`; KGW gamma remains strictly between zero and
one (`STATIC`; `detector/app.py`). The recorded deployment defaults remain inside
those limits.

### Local startup and regression verification

**Claim (`EXECUTED`; source: raw local output):** every lower/non-finite invalid
case in the focused matrix failed through the actual FastAPI lifespan, each exact
maximum reached readiness, and each overflow failed before readiness. The full
detector service suite, SynthID completion-boundary suite, and Phase 4/D10 gateway
suite passed. Warnings were third-party Python 3.14/FastAPI/uvicorn deprecations.

```text
$ python3 -m pytest -q detector/tests/test_service.py::TestStartupConfigurationValidation
81 passed, 192 warnings in 14.23s

$ python3 -m pytest -q detector/tests/test_service.py
129 passed, 486 warnings in 110.55s (0:01:50)

$ python3 -m pytest -q tests/test_synthid_processor_static.py
52 passed in 4.52s

$ python3 -m pytest -q validation/tests tests/test_phase4_nemo_actions.py
106 passed, 73 warnings in 3.72s

$ python3 benchmarks/continuous_validation.py --self-test
{"content_logged": false, "fault_cases": 3, "n1": {"responses": 20, "selected": 20}, "n5": {"responses": 100, "selected": 20}, "queue_overflow_checked": true, "self_test": "passed"}
```

### Curated build, immutable image, and source identity

**Claim (`EXECUTED`; source: build 3 output and hash-only comparison below):** the
binary build context contained only the detector Dockerfile, application, pinned
requirements, and freshly built wheel. It did not contain the repository root or
ignored credential paths. OpenShift build `detector-3` pushed immutable detector
image
`sha256:ca26d69aea3a17be1e89ed678ea566c47b396664254ee4be9de720374d2d53f1`.
Both detector deployments rolled to that digest. The deployed application and
installed key-loader bytes matched local source exactly.

```text
$ ./deploy/phase0/build-wheel.sh
Created wheel for vllm-watermark: filename=vllm_watermark-0.1.0.dev0-py3-none-any.whl size=60918 sha256=fe2dca7d9d0129018137c7605c40f4276886448e8164ebe91021f870df489265
Successfully built vllm-watermark

DETECTOR_BUILD_CONTEXT_FILES
./detector/Dockerfile
./detector/app.py
./detector/requirements.txt
./dist/vllm_watermark-0.1.0.dev0-py3-none-any.whl

$ KUBECONFIG=cluster/auth/kubeconfig oc -n watermark start-build detector --from-dir=<curated-context> --follow --wait
build.build.openshift.io/detector-3 started
[intermediate dependency installation and layer-copy output omitted]
Successfully pushed image-registry.openshift-image-registry.svc:5000/watermark/detector@sha256:ca26d69aea3a17be1e89ed678ea566c47b396664254ee4be9de720374d2d53f1
Push successful

$ KUBECONFIG=cluster/auth/kubeconfig oc -n watermark rollout status deployment/detector --timeout=8m
deployment "detector" successfully rolled out
$ KUBECONFIG=cluster/auth/kubeconfig oc -n watermark rollout status deployment/detector-synthid --timeout=8m
deployment "detector-synthid" successfully rolled out
detector           1  1  image-registry.openshift-image-registry.svc:5000/watermark/detector@sha256:ca26d69aea3a17be1e89ed678ea566c47b396664254ee4be9de720374d2d53f1
detector-synthid   1  1  image-registry.openshift-image-registry.svc:5000/watermark/detector@sha256:ca26d69aea3a17be1e89ed678ea566c47b396664254ee4be9de720374d2d53f1

DETECTOR_RUNTIME_SOURCE_MATCH=true app_sha=f3266b6abec65ee4d09e7b8714916870cfbdae9996a9fe873041817005d19d14 keys_sha=0218eb715da864b956bf5f52017fb33171190aaf5009fb6aae5173aa921255e5
```

### Built-image bounds and live startup failure/recovery

**Claim (`EXECUTED`; source: in-container and deployment output below):** the built
image accepted all nine exact maxima and rejected all nine overflow values. A real
Deployment rollout with `WATERMARK_VOCAB_SIZE=1048577` reached zero ready replicas
and logged the expected bound failure; restoring the manifest value `151936`
returned the same immutable image to readiness. Both scheme requests then returned
coherent digest and response/validation-ID correlation fields without printing
submitted text or key material.

The first observation wrapper used arithmetic expansion where command substitution
was required and exited after initiating the invalid rollout. Its `EXIT` trap
restored `WATERMARK_VOCAB_SIZE=151936` and waited for readiness; a direct query
confirmed `ready=1` at the expected digest before the authoritative rerun below
(`EXECUTED`; shell error and recovery query). No result from that failed wrapper is
counted.

```text
BUILT_IMAGE_BOUND_PROBE {"failures": [], "maxima_accepted": 9, "overflow_rejected": 9, "settings": 9}

DETECTOR_UPPER_BOUND_STARTUP invalid_ready=0 restart_count=1 config_error_matches=1
Waiting for deployment "detector" rollout to finish: 0 of 1 updated replicas are available...
deployment "detector" successfully rolled out
DETECTOR_UPPER_BOUND_RECOVERY ready=1 image=image-registry.openshift-image-registry.svc:5000/watermark/detector@sha256:ca26d69aea3a17be1e89ed678ea566c47b396664254ee4be9de720374d2d53f1

DETECTOR_POSTBUILD_API {"checks": [{"digest_match": true, "http_ok": true, "response_id_match": true, "scheme": "kgw", "validation_id_match": true, "verdict_boolean": true}, {"digest_match": true, "http_ok": true, "response_id_match": true, "scheme": "synthid", "validation_id_match": true, "verdict_boolean": true}], "default_key_available": true, "ready": true}
```

### Scoped result and remaining hardening

**Claim (`EXECUTED`/`STATIC`; source: commands above and `detector/app.py`):** D9 is
closed for individual detector environment-value bounds, tokenizer-fallback bounds,
effective KGW green-list validation, default-key readiness, and the tested key-error
redaction surfaces. The earlier D10 generated-response matrix remains valid evidence
for its recorded detector digest; this follow-up did not repeat that GPU matrix and
does not claim that it did.

**Claim (`OPEN`; source: independent review of `kgw/detector.py`,
`synthid/core.py`, and the generation processors):** the direct detector API still
has no request token/body limit, and KGW's per-request distinct-predecessor cache can
grow with a very long request. The maximum SynthID table can allocate 128 MiB on
first use, and maximum n-gram/history combinations can be expensive on very long
inputs. Generation-side KGW/SynthID processors do not yet share all detector-side
upper-bound validation. These are production hardening boundaries, not claims
covered by the scoped D9 startup result.

<a id="final-detector-reconciliation-2026-08-09"></a>

## 2026-08-09 — Final detector image and source reconciliation (EXECUTED; redacted)

**Environment:** local Python 3.14 workstation plus namespace `watermark` on the
recorded OpenShift 4.20 / RHOAI 3.4.2 cluster. Cluster commands used
`KUBECONFIG=cluster/auth/kubeconfig`. No prompt, generated response, key, token,
Secret value, route host, cluster IP, or pod suffix is reproduced below. This
append-only entry supersedes the preceding detector digest only for the current
deployed detector image; it does not rewrite or claim a rerun of the earlier D10
GPU generated-response matrix.

### Immutable build 4 and exact source identity

**Claim (`EXECUTED`; source: curated-context build output, Build status, rollout
state, and hash-only comparison below):** after the detector module documentation
was reconciled with the already executed managed-NeMo path, the same four-file
curated binary context was rebuilt. OpenShift build `detector-4` completed and
pushed immutable image
`sha256:9cd8a696785d350cf14e2769e7f566e1546642249854fba7e7f76ac29c3fbcf1`.
Both detector deployments used that digest and were ready. The deployed
application and key-loader bytes matched the final local source exactly.

```text
DETECTOR_BUILD_4_CONTEXT file_count=4
./detector/Dockerfile
./detector/app.py
./detector/requirements.txt
./dist/vllm_watermark-0.1.0.dev0-py3-none-any.whl

$ sha256sum dist/vllm_watermark-0.1.0.dev0-py3-none-any.whl
fe2dca7d9d0129018137c7605c40f4276886448e8164ebe91021f870df489265  dist/vllm_watermark-0.1.0.dev0-py3-none-any.whl

build.build.openshift.io/detector-4 started
[intermediate dependency installation and layer-copy output omitted]
Successfully pushed image-registry.openshift-image-registry.svc:5000/watermark/detector@sha256:9cd8a696785d350cf14e2769e7f566e1546642249854fba7e7f76ac29c3fbcf1
Push successful

$ KUBECONFIG=cluster/auth/kubeconfig oc -n watermark get build detector-4 -o custom-columns=<bounded-fields>
NAME         PHASE      OUTPUT                                                                       DIGEST
detector-4   Complete   image-registry.openshift-image-registry.svc:5000/watermark/detector:latest   sha256:9cd8a696785d350cf14e2769e7f566e1546642249854fba7e7f76ac29c3fbcf1

NAME               READY   AVAILABLE   IMAGE
detector           1       1           image-registry.openshift-image-registry.svc:5000/watermark/detector@sha256:9cd8a696785d350cf14e2769e7f566e1546642249854fba7e7f76ac29c3fbcf1
detector-synthid   1       1           image-registry.openshift-image-registry.svc:5000/watermark/detector@sha256:9cd8a696785d350cf14e2769e7f566e1546642249854fba7e7f76ac29c3fbcf1

deployment=detector
app_sha256=2559f23eadd869cf8cf01aa0fcee01e9b645fd1198c0898a8e29cd7f152c43eb
keys_sha256=0218eb715da864b956bf5f52017fb33171190aaf5009fb6aae5173aa921255e5
deployment=detector-synthid
app_sha256=2559f23eadd869cf8cf01aa0fcee01e9b645fd1198c0898a8e29cd7f152c43eb
keys_sha256=0218eb715da864b956bf5f52017fb33171190aaf5009fb6aae5173aa921255e5
```

### Final built-image, startup, and API probes

**Claim (`EXECUTED`; source: in-container settings probe, controlled Deployment
mutation/recovery, and direct API output below):** build 4 again accepted all nine
exact maxima and rejected all nine overflows. With
`WATERMARK_VOCAB_SIZE=1048577`, the real detector Deployment reached zero ready
replicas, restarted, and emitted the expected bounded configuration error. Restoring
the manifest value `151936` returned the same immutable digest to readiness. Both
scheme services then returned coherent hash and correlation fields.

The first combined observation wrapper restored `WATERMARK_VOCAB_SIZE=151936` but
exited while its replacement pod was still starting. An explicit rollout wait
returned the restored Deployment to readiness before the authoritative split probe
below. No partial result from that wrapper is counted.

```text
BUILT_IMAGE_BOUND_PROBE {"failures": [], "maxima_accepted": 9, "overflow_rejected": 9, "settings": 9}

INVALID_STATE_OBSERVED ready=0 restart_count=1
DETECTOR_UPPER_BOUND_STARTUP invalid_ready=0 restart_count=1 config_error_matches=2
Waiting for deployment "detector" rollout to finish: 0 of 1 updated replicas are available...
deployment "detector" successfully rolled out
DETECTOR_UPPER_BOUND_RECOVERY ready=1 vocab=151936 image=image-registry.openshift-image-registry.svc:5000/watermark/detector@sha256:9cd8a696785d350cf14e2769e7f566e1546642249854fba7e7f76ac29c3fbcf1

DETECTOR_POSTBUILD_API {"checks": [{"digest_match": true, "http_ok": true, "response_id_match": true, "scheme": "kgw", "validation_id_match": true, "verdict_boolean": true}, {"digest_match": true, "http_ok": true, "response_id_match": true, "scheme": "synthid", "validation_id_match": true, "verdict_boolean": true}], "default_key_available": true, "ready": true}
```

### Exact-revision repository and manifest verification

**Claim (`EXECUTED`; source: raw local and server-side verification summaries
below):** the final code revision passed the complete 284-test local suite and
compiled. All 17 Phase 3–5 YAML files parsed as 31 documents and passed the
repository's relaxed lint invocation. The detector, Phase 4, and Phase 5
Kustomize resources passed server-side dry-run. All 406 checked repository-local
Markdown links resolved. `git diff --check` passed.

The first Phase 5 directory dry-run used `-f` and correctly rejected
`kustomization.yaml` as though it were an API resource. The authoritative command
used `-k deploy/phase5` and passed; neither dry-run mutated the cluster.

```text
$ python3 -m pytest -q
284 passed, 486 warnings in 491.45s (0:08:11)

$ python3 -m compileall -q src detector validation benchmarks
[no output; exit 0]

YAML_PARSE files=17 documents=31 result=pass
YAMLLINT_RELAXED result=pass
SERVER_DRY_RUN path=deploy/phase3/detector-deploy.yaml result=pass
SERVER_DRY_RUN path=deploy/phase3/detector-synthid-deploy.yaml result=pass
SERVER_DRY_RUN path=deploy/phase3/nemo-guardrails-poc.yaml result=pass
SERVER_DRY_RUN path=deploy/phase4 result=pass
SERVER_DRY_RUN path=deploy/phase5 mode=kustomize result=pass
MARKDOWN_LOCAL_LINKS files=29 links=406 broken=0
GIT_DIFF_CHECK result=pass
```

### Final managed-path and resource-safety postflight

**Claim (`EXECUTED`; source: hash-only comparison and bounded cluster status
below):** the four managed-action data values and four gateway runtime modules
still matched local source; test controls were off; the validation gateway, both
detectors, and managed-NeMo resource were ready; and the billable GPU MachineSet
was `0/0/0`.

```text
ACTION_DATA key=actions.py hash_match=true sha256=9294ebab8d27ee542d66e8dbb9093b39033e45c244570610451761fd7d39c6bd
ACTION_DATA key=config.py hash_match=true sha256=e7a36150275601fb99df32e480c705e5e8d8426f8d94223fd5e0e1429b7246ee
ACTION_DATA key=config.yaml hash_match=true sha256=da052dc808a83dd39484f40503466e6b8d2ab90ed29a96d3880142694b7a2e00
ACTION_DATA key=rails.co hash_match=true sha256=53b4b1dc3919a220e890f0896f3cb72643aa02f4ee1792d203064e0650219bdb
FINAL_POSTFLIGHT controls=off gateway_ready=1 detector_ready=1 synthid_ready=1 nemo=Ready action_data_match=true runtime_source_match=true gpu=0/0/0
```

**Scope (`EXECUTED`/`OPEN`; source: this entry and the preceding D10 record):**
this closes reconciliation of the current detector image, source, startup bounds,
and service smoke behavior. The generated-response D10 acceptance matrix remains
the executed evidence for its recorded vLLM, managed-NeMo, gateway, and preceding
detector digests; build 4 was not presented as a GPU matrix rerun. External
KServe/Istio pass-through, product supportability, caller/network production
controls, multi-replica/streaming/asynchronous behavior, platform-wide retention,
request/cache bounds, expensive maximum configurations, and generation-processor
bound parity remain explicitly `OPEN`.

<a id="latency-semantics-correction-2026-08-09"></a>

## 2026-08-09 — Continuous-validation latency semantics correction (STATIC/OPEN)

**Correction (`STATIC`; source: `validation/gateway.py::_deliver_response` and
the current harness report schema):** the executed D10 report retained the field
and metric name `client_delivery`, and its historical output described the interval
as `request_start_to_gateway_delivery`. Source review shows that the timer stops
when the gateway response mapping is constructed, before FastAPI JSON serialization,
socket transmission, or client receipt. The recorded quantiles therefore measure
request start to **gateway response ready**, not client-observed delivery. Their
numeric values and sample-count reconciliation remain the recorded executed output;
only the interpretation is narrowed. The current harness labels that interval
`request_start_to_gateway_response_ready`.

**Remaining boundary (`OPEN`; source: the same review):** client-observed end-to-end
latency, including serialization and network delivery, was not measured by this
acceptance run.

<a id="current-detector-reconciliation-2026-08-09"></a>

## 2026-08-09 — Blank-vocabulary correction and current detector image (EXECUTED; redacted)

**Environment:** the same local workstation and recorded OpenShift 4.20 / RHOAI
3.4.2 cluster as the preceding reconciliation. No request content, generated
response, key, token, Secret value, route host, cluster IP, or pod suffix is
reproduced. This is an append-only follow-up to the independent final-diff review;
the earlier image records remain valid for their recorded revisions.

### Review finding and exact-code verification

**Claim (`STATIC`/`EXECUTED`; source: `detector/app.py`, regression tests, and raw
test output below):** final review found that an explicitly present empty or
whitespace-only `WATERMARK_VOCAB_SIZE` was treated as absent, allowing tokenizer
fallback instead of fail-fast startup. The parser now distinguishes absence from
an explicit blank. Both blank forms fail through `load_settings()` and real FastAPI
lifespan, while the absent-variable fallback remains covered. Independent re-review
found no remaining blocker. The detector suite and complete repository suite passed
at the corrected code revision.

```text
$ python3 -m pytest -q detector/tests/test_service.py
133 passed, 498 warnings in 111.71s (0:01:51)

$ python3 -m pytest -q
288 passed, 498 warnings in 436.54s (0:07:16)

$ python3 -m compileall -q src detector validation benchmarks
[no output; exit 0]

$ python3 benchmarks/continuous_validation.py --self-test
{"content_logged": false, "fault_cases": 3, "n1": {"responses": 20, "selected": 20}, "n5": {"responses": 100, "selected": 20}, "queue_overflow_checked": true, "self_test": "passed"}

$ git diff --check
[no output; exit 0]
```

### Curated build 5, immutable digest, and source identity

**Claim (`EXECUTED`; source: build, rollout, and hash-only output below):** the
corrected application was submitted in an exact four-file context. OpenShift build
`detector-5` completed and pushed immutable image
`sha256:2bb606b8e08b039b28f4b22356840f4cb0855d05fa688da4f4b70d26d8d0ea0f`.
Both detector deployments rolled to that digest. Their application and installed
key-loader hashes matched local source.

```text
DETECTOR_BUILD_5_CONTEXT file_count=4
detector/Dockerfile
detector/app.py
detector/requirements.txt
dist/vllm_watermark-0.1.0.dev0-py3-none-any.whl

build.build.openshift.io/detector-5 started
[intermediate dependency installation and layer-copy output omitted]
Successfully pushed image-registry.openshift-image-registry.svc:5000/watermark/detector@sha256:2bb606b8e08b039b28f4b22356840f4cb0855d05fa688da4f4b70d26d8d0ea0f
Push successful

NAME         PHASE      OUTPUT                                                                       DIGEST
detector-5   Complete   image-registry.openshift-image-registry.svc:5000/watermark/detector:latest   sha256:2bb606b8e08b039b28f4b22356840f4cb0855d05fa688da4f4b70d26d8d0ea0f
NAME               READY   AVAILABLE   IMAGE
detector           1       1           image-registry.openshift-image-registry.svc:5000/watermark/detector@sha256:2bb606b8e08b039b28f4b22356840f4cb0855d05fa688da4f4b70d26d8d0ea0f
detector-synthid   1       1           image-registry.openshift-image-registry.svc:5000/watermark/detector@sha256:2bb606b8e08b039b28f4b22356840f4cb0855d05fa688da4f4b70d26d8d0ea0f

local_app_sha256=a4ef7a931bdc810fc02df727a79a02cc993f88ba33d6eadd4ebdc5e8113287d6
local_keys_sha256=0218eb715da864b956bf5f52017fb33171190aaf5009fb6aae5173aa921255e5
deployment=detector
app_sha256=a4ef7a931bdc810fc02df727a79a02cc993f88ba33d6eadd4ebdc5e8113287d6
keys_sha256=0218eb715da864b956bf5f52017fb33171190aaf5009fb6aae5173aa921255e5
deployment=detector-synthid
app_sha256=a4ef7a931bdc810fc02df727a79a02cc993f88ba33d6eadd4ebdc5e8113287d6
keys_sha256=0218eb715da864b956bf5f52017fb33171190aaf5009fb6aae5173aa921255e5
```

### Built-image blank/bound probes and live recovery

**Claim (`EXECUTED`; source: in-image parser, controlled Deployment, and API
output below):** build 5 rejected both explicit blank forms as well as all nine
overflows, while accepting all nine exact maxima. An actual blank-valued Deployment
reached zero ready replicas and emitted the expected configuration error. Restoring
`151936` returned the same immutable image to readiness. Both scheme APIs retained
coherent digest/ID fields and boolean verdicts.

```text
BUILT_IMAGE_BOUND_PROBE {"explicit_blank_rejected": 2, "failures": [], "maxima_accepted": 9, "overflow_rejected": 9, "settings": 9}
DETECTOR_BLANK_VOCAB_STARTUP explicit_blank=1 invalid_ready=0 restart_count=1
DETECTOR_BLANK_VOCAB_ERROR invalid_ready=0 restart_count=1 config_error_matches=2
Waiting for deployment "detector" rollout to finish: 0 of 1 updated replicas are available...
deployment "detector" successfully rolled out
DETECTOR_BLANK_VOCAB_RECOVERY ready=1 vocab=151936 image=image-registry.openshift-image-registry.svc:5000/watermark/detector@sha256:2bb606b8e08b039b28f4b22356840f4cb0855d05fa688da4f4b70d26d8d0ea0f

DETECTOR_POSTBUILD_API {"checks": [{"digest_match": true, "http_ok": true, "response_id_match": true, "scheme": "kgw", "validation_id_match": true, "verdict_boolean": true}, {"digest_match": true, "http_ok": true, "response_id_match": true, "scheme": "synthid", "validation_id_match": true, "verdict_boolean": true}], "default_key_available": true, "ready": true}
```

### Deployment and resource postflight

**Claim (`EXECUTED`; source: server dry-run, live diff, and bounded status below):**
the Phase 3 detector, Phase 4, and Phase 5 resources passed server-side dry-run;
Phase 4 and Phase 5 had no live diff. Test controls were off, gateway/action runtime
hashes still matched local source, the managed services were ready, both detector
images matched, and the GPU MachineSet remained `0/0/0`.

```text
SERVER_DRY_RUN path=deploy/phase3/detector-deploy.yaml result=pass
SERVER_DRY_RUN path=deploy/phase3/detector-synthid-deploy.yaml result=pass
SERVER_DRY_RUN path=deploy/phase3/nemo-guardrails-poc.yaml result=pass
SERVER_DRY_RUN path=deploy/phase4 result=pass
SERVER_DRY_RUN path=deploy/phase5 mode=kustomize result=pass
OC_DIFF phase4_exit=0 phase4_lines=1 phase5_exit=0 phase5_lines=1
FINAL_POSTFLIGHT controls=off gateway_ready=1 detector_ready=1 synthid_ready=1 nemo=Ready action_data_match=true runtime_source_match=true detector_images_match=true gpu=0/0/0
```

**Scope (`EXECUTED`/`OPEN`; source: this record and D10):** build 5 is the current
detector image and closes the explicit-blank startup gap. The earlier generated-
response D10 matrix remains evidence for its recorded detector digest and was not
represented as a GPU rerun. Its production boundaries and the detector request/
cache/generator-bound hardening items remain `OPEN`.

<a id="current-build5-d10-rerun-2026-08-09"></a>

## 2026-08-09 — Current build-5 D10 matrix rerun (EXECUTED; redacted)

**Evidence qualification (`EXECUTED`; source: exact command and complete content-safe
stdout below):** this append-only follow-up removes the revision boundary recorded
above. The fixed D10 matrix ran again through the ready RHOAI
ServingRuntime/InferenceService internal predictor, gateway, RHOAI-managed NeMo
action, authenticated broker, and both detector services after detector build 5
became current. Unlike the preceding matrix record, both `record_evidence` arrays
are reproduced in full. They contain only identifiers, SHA-256 content digests,
non-secret scheme/key IDs, verdicts, action outcomes, attempt counts, and timings;
the report contains no plaintext-response field. The mode-`600` report was scanned
against the six live secret values supplied to the harness and contained zero
matches. Prompt text, response text, token values, key material, route hosts, cluster
IPs, pod suffixes, and credential-source output are not reproduced.

### Ready stack and current immutable revisions

**Claim (`EXECUTED`; source: bounded readiness, image, and API output below):**
the GPU MachineSet started from zero without the prior quota-overlap failure and
reached `1/1/1`. The RHOAI `InferenceService` reached `Ready=True` in Standard
mode; its predictor was `1/1`, answered health/model probes, and used the recorded
vLLM digest. The gateway, both build-5 detectors, and managed NeMo were ready. Test
controls were still off at preflight.

```text
$ ./scripts/scale-gpu.sh 1
machineset.machine.openshift.io/ocp-ai-p9j4n-gpu-us-east-1a scaled
{"gpu_sets":1,"desired":1,"current":1,"ready":0}
GPU_WAIT attempt=21 {"desired":1,"current":1,"ready":1}

$ KUBECONFIG=cluster/auth/kubeconfig oc wait --for=condition=Ready \
    inferenceservice/watermark-vllm -n watermark --timeout=20m
inferenceservice.serving.kserve.io/watermark-vllm condition met
deployment "watermark-vllm-predictor" successfully rolled out
{"ready":"True","mode":"Standard","runtime":"watermark-vllm"}
{"desired":1,"ready":1,"available":1,"image_digest":"sha256:f8294ee0459869e9659b1178ed91f57a1b52a52c6a5f5f819ca651646b317e4c"}

{
  "preflight_pass": true,
  "deployments": [
    {
      "name": "detector",
      "ready": 1,
      "desired": 1,
      "controls": "n/a",
      "image": "image-registry.openshift-image-registry.svc:5000/watermark/detector@sha256:2bb606b8e08b039b28f4b22356840f4cb0855d05fa688da4f4b70d26d8d0ea0f"
    },
    {
      "name": "detector-synthid",
      "ready": 1,
      "desired": 1,
      "controls": "n/a",
      "image": "image-registry.openshift-image-registry.svc:5000/watermark/detector@sha256:2bb606b8e08b039b28f4b22356840f4cb0855d05fa688da4f4b70d26d8d0ea0f"
    },
    {
      "name": "watermark-validation-gateway",
      "ready": 1,
      "desired": 1,
      "controls": "off",
      "image": "image-registry.openshift-image-registry.svc:5000/watermark/watermark-validation-gateway@sha256:9443345c272c55a6dcea41bae84f8e04bab5d0f27e0934df2e4b4ff11b37fed6"
    }
  ]
}
{
  "name": "nemo-watermark",
  "phase": "Ready"
}
{"health": 200, "models": 200}
```

The managed-NeMo server image remained
`sha256:22125dbbd05d1cfa7af931fdeca9da72c6d267a05c33a1f757daf3095ddcca7c`
(`EXECUTED`; bounded live image query following the run). The current detector
application/key-loader source hashes and gateway/action runtime source identity were
already reconciled in the immediately preceding build-5 record; no source changed
between that check and this run.

### Exact harness invocation and full hash-only report

Existing Secret values were loaded into the named process environment variables
without shell tracing or output, and the key ID was treated as non-secret. The
gateway Service was port-forwarded only after the acceptance-control rollout. The
harness held generated text only long enough to recompute its SHA-256 digest.

```text
D10_STAGE secret_markers_loaded=true
D10_STAGE controls=on gateway_rollout=ready
D10_STAGE port_forward=ready

$ python3 benchmarks/continuous_validation.py \
    --gateway-url http://127.0.0.1:18080 \
    --model watermark-vllm \
    --key-id "$D10_KEY_ID" \
    --admin-token-env D10_ADMIN_TOKEN \
    --secret-marker-env D10_ADMIN_TOKEN \
    --secret-marker-env D10_BROKER_TOKEN \
    --secret-marker-env D10_NEMO_TOKEN \
    --secret-marker-env D10_WATERMARK_KEY \
    --secret-marker-env D10_SIGNING_KEY \
    --secret-marker-env D10_MODEL_TOKEN \
    --max-tokens 256 --temperature 0.7 \
    --timeout-seconds 120 --queue-pending-check-seconds 0.5 \
    --positive-action block --clean-action pass \
    --expected-mode synchronous --expected-failure-policy closed
```

Complete stdout follows. Its SHA-256 as written to the mode-`600` capture was
`3a67502f9d3f99f82dfb4c4c99618e3cc95517e5fe3e1d7d1a39525a41241e40`.

```json
{
  "configuration": {
    "0": "rejected",
    "1": "accepted",
    "5": "accepted",
    "empty": "rejected",
    "fraction": "rejected",
    "negative": "rejected",
    "nonnumeric": "rejected"
  },
  "content_logged": false,
  "contract": "phase5-v1",
  "faults": {
    "malformed_success": {
      "attempts": 1,
      "retries": 0,
      "terminal_state": "malformed_response"
    },
    "retry_exhausted": {
      "attempts": 3,
      "retries": 2,
      "terminal_state": "retry_exhausted"
    },
    "retry_then_success": {
      "attempts": 2,
      "retries": 1,
      "terminal_state": "success"
    }
  },
  "latency_semantics": {
    "client_delivery": "request_start_to_gateway_response_ready",
    "generation_completion": "request_start_to_upstream_completion",
    "validation": "validation_attempt_window",
    "validation_lag": "validation_queue_wait_to_attempt_start"
  },
  "n1": {
    "client_delivery_latency": {
      "count": 20,
      "p50_seconds": 0.8418383164971601,
      "p95_seconds": 1.550921352223668,
      "p99_seconds": 1.8908106376402425
    },
    "counters": {
      "cancelled": 0,
      "clean": 10,
      "completed": 20,
      "detector_attempts": 20,
      "dropped": 0,
      "errors": 0,
      "failed": 0,
      "guardrails_attempts": 20,
      "queue_overflow": 0,
      "retries": 0,
      "selected": 20,
      "started": 20,
      "terminal": 20,
      "unsampled": 0,
      "watermarked": 10
    },
    "generation_completion_latency": {
      "count": 20,
      "p50_seconds": 0.4957721539976774,
      "p95_seconds": 1.0462472340484965,
      "p99_seconds": 1.0582058044042788
    },
    "latency_samples": {
      "client_delivery": 20,
      "generation_completion": 20,
      "validation": 20,
      "validation_lag": 20
    },
    "queue_depth": 0,
    "record_evidence": [
      {
        "attempts": 1,
        "content_digest": "26d3464eb84a331826923d4a1fc347d6bc0d26d8de57b3a81491016798ade995",
        "delivery_outcome": "delivered",
        "detector_call_id": "88f85368-7cce-418b-adf8-93ede7845ab4",
        "guardrails_action": "block",
        "guardrails_action_id": "88f85368-7cce-418b-adf8-93ede7845ab4",
        "ids_correlated": true,
        "key_id": "poc-2026-08",
        "managed_action": "blocked",
        "response_id": "chatcmpl-9c3585701b45e320",
        "scheme": "kgw",
        "timing": {
          "validation_lag_seconds": 4.9520982429385185e-05,
          "validation_latency_seconds": 0.534812249999959
        },
        "validation_id": "88f85368-7cce-418b-adf8-93ede7845ab4",
        "verdict": true
      },
      {
        "attempts": 1,
        "content_digest": "b7cb7bb51069ac8733a13a71b54f6fa119bd15953b1ca784c7aa4a85232149de",
        "delivery_outcome": "delivered",
        "detector_call_id": "fd4d2d36-755a-42dd-bf17-66b134df3cdd",
        "guardrails_action": "block",
        "guardrails_action_id": "fd4d2d36-755a-42dd-bf17-66b134df3cdd",
        "ids_correlated": true,
        "key_id": "poc-2026-08",
        "managed_action": "blocked",
        "response_id": "chatcmpl-bc5c4f892af738b2",
        "scheme": "kgw",
        "timing": {
          "validation_lag_seconds": 5.05319912917912e-05,
          "validation_latency_seconds": 0.8262455940130167
        },
        "validation_id": "fd4d2d36-755a-42dd-bf17-66b134df3cdd",
        "verdict": true
      },
      {
        "attempts": 1,
        "content_digest": "70e44c079d3a668c5cc58539259fb919a05d0d9b792f226ca424bea8e3e2cca9",
        "delivery_outcome": "delivered",
        "detector_call_id": "95bae2ea-03ad-45c5-81c0-143f140f2b9d",
        "guardrails_action": "block",
        "guardrails_action_id": "95bae2ea-03ad-45c5-81c0-143f140f2b9d",
        "ids_correlated": true,
        "key_id": "poc-2026-08",
        "managed_action": "blocked",
        "response_id": "chatcmpl-bcf4570be875929a",
        "scheme": "kgw",
        "timing": {
          "validation_lag_seconds": 3.535498399287462e-05,
          "validation_latency_seconds": 0.7550994410121348
        },
        "validation_id": "95bae2ea-03ad-45c5-81c0-143f140f2b9d",
        "verdict": true
      },
      {
        "attempts": 1,
        "content_digest": "96ea28755677cba49d989b1c6c3992d6595139143e182082152792f424b2c255",
        "delivery_outcome": "delivered",
        "detector_call_id": "24ca9626-3e51-4d75-bfbf-0823ce7a610e",
        "guardrails_action": "block",
        "guardrails_action_id": "24ca9626-3e51-4d75-bfbf-0823ce7a610e",
        "ids_correlated": true,
        "key_id": "poc-2026-08",
        "managed_action": "blocked",
        "response_id": "chatcmpl-b195a01cd753b508",
        "scheme": "kgw",
        "timing": {
          "validation_lag_seconds": 3.1898001907393336e-05,
          "validation_latency_seconds": 0.46676337299868464
        },
        "validation_id": "24ca9626-3e51-4d75-bfbf-0823ce7a610e",
        "verdict": true
      },
      {
        "attempts": 1,
        "content_digest": "588935709ab71beaa9c36ed206ff684bdad0afcc7c07fc0f5021af2104ece97e",
        "delivery_outcome": "delivered",
        "detector_call_id": "328b0237-41a7-4e45-a606-660e8b99d916",
        "guardrails_action": "block",
        "guardrails_action_id": "328b0237-41a7-4e45-a606-660e8b99d916",
        "ids_correlated": true,
        "key_id": "poc-2026-08",
        "managed_action": "blocked",
        "response_id": "chatcmpl-9bff550753d37126",
        "scheme": "kgw",
        "timing": {
          "validation_lag_seconds": 3.19909886457026e-05,
          "validation_latency_seconds": 0.9209215279843193
        },
        "validation_id": "328b0237-41a7-4e45-a606-660e8b99d916",
        "verdict": true
      },
      {
        "attempts": 1,
        "content_digest": "6aa5127c60b8437ac6115b9e24fa9c1f025ddc524fc14c9d54ef974852658086",
        "delivery_outcome": "delivered",
        "detector_call_id": "f869f340-7d43-4577-b01b-2bc80eda8ead",
        "guardrails_action": "block",
        "guardrails_action_id": "f869f340-7d43-4577-b01b-2bc80eda8ead",
        "ids_correlated": true,
        "key_id": "poc-2026-08",
        "managed_action": "blocked",
        "response_id": "chatcmpl-9280a0e0fc3d7a88",
        "scheme": "synthid",
        "timing": {
          "validation_lag_seconds": 3.79109987989068e-05,
          "validation_latency_seconds": 0.04516677398351021
        },
        "validation_id": "f869f340-7d43-4577-b01b-2bc80eda8ead",
        "verdict": true
      },
      {
        "attempts": 1,
        "content_digest": "614c8d731c9b52a2ac4f20f6ca28956cf1a87873f570212b6f6d984a4d80ed78",
        "delivery_outcome": "delivered",
        "detector_call_id": "5b77a682-c714-4597-889c-9f019aa2276d",
        "guardrails_action": "block",
        "guardrails_action_id": "5b77a682-c714-4597-889c-9f019aa2276d",
        "ids_correlated": true,
        "key_id": "poc-2026-08",
        "managed_action": "blocked",
        "response_id": "chatcmpl-9be4195b90d143af",
        "scheme": "synthid",
        "timing": {
          "validation_lag_seconds": 3.17390076816082e-05,
          "validation_latency_seconds": 0.03465224301908165
        },
        "validation_id": "5b77a682-c714-4597-889c-9f019aa2276d",
        "verdict": true
      },
      {
        "attempts": 1,
        "content_digest": "754340269111828d8a74d8ef4b9475804e3376f5e856a2e8683893baec6494d4",
        "delivery_outcome": "delivered",
        "detector_call_id": "92a6622d-2f10-4977-a117-41a5eebe0256",
        "guardrails_action": "block",
        "guardrails_action_id": "92a6622d-2f10-4977-a117-41a5eebe0256",
        "ids_correlated": true,
        "key_id": "poc-2026-08",
        "managed_action": "blocked",
        "response_id": "chatcmpl-9b82df907caf1e91",
        "scheme": "synthid",
        "timing": {
          "validation_lag_seconds": 3.6793004255741835e-05,
          "validation_latency_seconds": 0.043212895980104804
        },
        "validation_id": "92a6622d-2f10-4977-a117-41a5eebe0256",
        "verdict": true
      },
      {
        "attempts": 1,
        "content_digest": "428f325bc87d38fd9489fa71410f5cae1754c2b4d414ddc36188ebc9b942ca3e",
        "delivery_outcome": "delivered",
        "detector_call_id": "7fe16e46-402d-44a3-bee0-2e844af589c3",
        "guardrails_action": "block",
        "guardrails_action_id": "7fe16e46-402d-44a3-bee0-2e844af589c3",
        "ids_correlated": true,
        "key_id": "poc-2026-08",
        "managed_action": "blocked",
        "response_id": "chatcmpl-996e6a3fb4a4993a",
        "scheme": "synthid",
        "timing": {
          "validation_lag_seconds": 3.501400351524353e-05,
          "validation_latency_seconds": 0.03553824100526981
        },
        "validation_id": "7fe16e46-402d-44a3-bee0-2e844af589c3",
        "verdict": true
      },
      {
        "attempts": 1,
        "content_digest": "97bb97d996ce857c079d0e83d65a49d1fc69d06467a6992820eac718b6302285",
        "delivery_outcome": "delivered",
        "detector_call_id": "d04bb8d1-c77c-4eaf-aa11-0c355322c059",
        "guardrails_action": "block",
        "guardrails_action_id": "d04bb8d1-c77c-4eaf-aa11-0c355322c059",
        "ids_correlated": true,
        "key_id": "poc-2026-08",
        "managed_action": "blocked",
        "response_id": "chatcmpl-9541e3192ef3a570",
        "scheme": "synthid",
        "timing": {
          "validation_lag_seconds": 3.7321995478123426e-05,
          "validation_latency_seconds": 0.050677432998782024
        },
        "validation_id": "d04bb8d1-c77c-4eaf-aa11-0c355322c059",
        "verdict": true
      },
      {
        "attempts": 1,
        "content_digest": "70a252d2cc0d126367fb30cfb502cc8761539e408a96453335c2d8e647b72a8f",
        "delivery_outcome": "delivered",
        "detector_call_id": "cec73604-3280-447a-9368-eb90fa69b361",
        "guardrails_action": "pass",
        "guardrails_action_id": "cec73604-3280-447a-9368-eb90fa69b361",
        "ids_correlated": true,
        "key_id": "poc-2026-08",
        "managed_action": "success",
        "response_id": "chatcmpl-91d8401ad604b2fe",
        "scheme": "kgw",
        "timing": {
          "validation_lag_seconds": 4.3334002839401364e-05,
          "validation_latency_seconds": 0.7678233329788782
        },
        "validation_id": "cec73604-3280-447a-9368-eb90fa69b361",
        "verdict": false
      },
      {
        "attempts": 1,
        "content_digest": "ca9f392f17b098ec0e43e13ad9756989d95c10ff25fbbd071fc55c31d2570fc5",
        "delivery_outcome": "delivered",
        "detector_call_id": "f82a27e5-6c62-4f1e-a62c-3a2ec968d16d",
        "guardrails_action": "pass",
        "guardrails_action_id": "f82a27e5-6c62-4f1e-a62c-3a2ec968d16d",
        "ids_correlated": true,
        "key_id": "poc-2026-08",
        "managed_action": "success",
        "response_id": "chatcmpl-a24c8b183f5f96ad",
        "scheme": "kgw",
        "timing": {
          "validation_lag_seconds": 3.838699194602668e-05,
          "validation_latency_seconds": 0.2333381750213448
        },
        "validation_id": "f82a27e5-6c62-4f1e-a62c-3a2ec968d16d",
        "verdict": false
      },
      {
        "attempts": 1,
        "content_digest": "550b36e7d71f6d4c7bd63e078ab01a3f5c5565b6fe7de73b9186d4850f902236",
        "delivery_outcome": "delivered",
        "detector_call_id": "806a7d8d-8a57-48bf-95a2-ec63204db019",
        "guardrails_action": "pass",
        "guardrails_action_id": "806a7d8d-8a57-48bf-95a2-ec63204db019",
        "ids_correlated": true,
        "key_id": "poc-2026-08",
        "managed_action": "success",
        "response_id": "chatcmpl-971ca541b1c8dd70",
        "scheme": "kgw",
        "timing": {
          "validation_lag_seconds": 3.248898428864777e-05,
          "validation_latency_seconds": 0.44858754600863904
        },
        "validation_id": "806a7d8d-8a57-48bf-95a2-ec63204db019",
        "verdict": false
      },
      {
        "attempts": 1,
        "content_digest": "1a72e9352f42383b0ec4f55a37402a1a3ffe02a35fbfdbc80e64e7c64c6ca634",
        "delivery_outcome": "delivered",
        "detector_call_id": "9bb0a78b-bae1-4b24-9340-f39e4ae0796f",
        "guardrails_action": "pass",
        "guardrails_action_id": "9bb0a78b-bae1-4b24-9340-f39e4ae0796f",
        "ids_correlated": true,
        "key_id": "poc-2026-08",
        "managed_action": "success",
        "response_id": "chatcmpl-9814a25e9829c7b8",
        "scheme": "kgw",
        "timing": {
          "validation_lag_seconds": 3.615900641307235e-05,
          "validation_latency_seconds": 0.696785019012168
        },
        "validation_id": "9bb0a78b-bae1-4b24-9340-f39e4ae0796f",
        "verdict": false
      },
      {
        "attempts": 1,
        "content_digest": "bce07f766131bcb131873c1a4c7a850b4d72d882a083cb09e7c42c8c85c015dd",
        "delivery_outcome": "delivered",
        "detector_call_id": "f8252879-c283-41fd-92ff-2bfeab16d703",
        "guardrails_action": "pass",
        "guardrails_action_id": "f8252879-c283-41fd-92ff-2bfeab16d703",
        "ids_correlated": true,
        "key_id": "poc-2026-08",
        "managed_action": "success",
        "response_id": "chatcmpl-a8280622261fcfed",
        "scheme": "kgw",
        "timing": {
          "validation_lag_seconds": 3.09899915009737e-05,
          "validation_latency_seconds": 0.45176462599192746
        },
        "validation_id": "f8252879-c283-41fd-92ff-2bfeab16d703",
        "verdict": false
      },
      {
        "attempts": 1,
        "content_digest": "db77dd35fada3dbeeb06b53e15306afba5bfacfb8401a614d31663b8328fcf82",
        "delivery_outcome": "delivered",
        "detector_call_id": "d56b0c43-f71d-4ca2-83fe-c6afce8ae828",
        "guardrails_action": "pass",
        "guardrails_action_id": "d56b0c43-f71d-4ca2-83fe-c6afce8ae828",
        "ids_correlated": true,
        "key_id": "poc-2026-08",
        "managed_action": "success",
        "response_id": "chatcmpl-b682ba5e66facd57",
        "scheme": "synthid",
        "timing": {
          "validation_lag_seconds": 3.1256990041583776e-05,
          "validation_latency_seconds": 0.044221074000233784
        },
        "validation_id": "d56b0c43-f71d-4ca2-83fe-c6afce8ae828",
        "verdict": false
      },
      {
        "attempts": 1,
        "content_digest": "7abe7012c0be9190f93f8c237c94880b9108e5b077e493b54eb27cc619d4b272",
        "delivery_outcome": "delivered",
        "detector_call_id": "8df941cf-6b10-41ac-b392-5b7efa956009",
        "guardrails_action": "pass",
        "guardrails_action_id": "8df941cf-6b10-41ac-b392-5b7efa956009",
        "ids_correlated": true,
        "key_id": "poc-2026-08",
        "managed_action": "success",
        "response_id": "chatcmpl-ae8300860b1ea82d",
        "scheme": "synthid",
        "timing": {
          "validation_lag_seconds": 3.307100269012153e-05,
          "validation_latency_seconds": 0.040018836996750906
        },
        "validation_id": "8df941cf-6b10-41ac-b392-5b7efa956009",
        "verdict": false
      },
      {
        "attempts": 1,
        "content_digest": "8de4026f28fbd21f5a248b16c1e031a2b39415e22de917fa55dd7c2a5275bea4",
        "delivery_outcome": "delivered",
        "detector_call_id": "4fa6eccd-08cd-4bfe-accf-60f20bedad6e",
        "guardrails_action": "pass",
        "guardrails_action_id": "4fa6eccd-08cd-4bfe-accf-60f20bedad6e",
        "ids_correlated": true,
        "key_id": "poc-2026-08",
        "managed_action": "success",
        "response_id": "chatcmpl-b3748a02146ec163",
        "scheme": "synthid",
        "timing": {
          "validation_lag_seconds": 3.0722992960363626e-05,
          "validation_latency_seconds": 0.04291650198865682
        },
        "validation_id": "4fa6eccd-08cd-4bfe-accf-60f20bedad6e",
        "verdict": false
      },
      {
        "attempts": 1,
        "content_digest": "d22fa0c53b3ca49b3edb51980554bbc29c0fd752e8250e7ba44cd1bd12a549d9",
        "delivery_outcome": "delivered",
        "detector_call_id": "75e44813-c1e1-4cd3-8655-97c3790392fd",
        "guardrails_action": "pass",
        "guardrails_action_id": "75e44813-c1e1-4cd3-8655-97c3790392fd",
        "ids_correlated": true,
        "key_id": "poc-2026-08",
        "managed_action": "success",
        "response_id": "chatcmpl-bd3eab7344ba2966",
        "scheme": "synthid",
        "timing": {
          "validation_lag_seconds": 3.749801544472575e-05,
          "validation_latency_seconds": 0.04323223198298365
        },
        "validation_id": "75e44813-c1e1-4cd3-8655-97c3790392fd",
        "verdict": false
      },
      {
        "attempts": 1,
        "content_digest": "f59b177b550cfd6eb2504cbf3aff8314a223e98f287c58717c86d3c763235fe5",
        "delivery_outcome": "delivered",
        "detector_call_id": "fb0519df-4d31-4157-88d2-23769ddac10e",
        "guardrails_action": "pass",
        "guardrails_action_id": "fb0519df-4d31-4157-88d2-23769ddac10e",
        "ids_correlated": true,
        "key_id": "poc-2026-08",
        "managed_action": "success",
        "response_id": "chatcmpl-8976bad572b274f5",
        "scheme": "synthid",
        "timing": {
          "validation_lag_seconds": 3.60650010406971e-05,
          "validation_latency_seconds": 0.041313228983199224
        },
        "validation_id": "fb0519df-4d31-4157-88d2-23769ddac10e",
        "verdict": false
      }
    ],
    "responses": 20,
    "selected": 20,
    "terminal": 20,
    "validation_lag": {
      "count": 20,
      "p50_seconds": 3.570999251678586e-05,
      "p95_seconds": 4.957153287250549e-05,
      "p99_seconds": 5.033989960793406e-05
    },
    "validation_latency": {
      "count": 20,
      "p50_seconds": 0.1420078040100634,
      "p95_seconds": 0.8309793907115819,
      "p99_seconds": 0.9029331005297716
    }
  },
  "n5": {
    "client_delivery_latency": {
      "count": 100,
      "p50_seconds": 0.49164154499885626,
      "p95_seconds": 1.5549047835142116,
      "p99_seconds": 1.8172679362515927
    },
    "counters": {
      "cancelled": 0,
      "clean": 10,
      "completed": 100,
      "detector_attempts": 20,
      "dropped": 0,
      "errors": 0,
      "failed": 0,
      "guardrails_attempts": 20,
      "queue_overflow": 0,
      "retries": 0,
      "selected": 20,
      "started": 100,
      "terminal": 20,
      "unsampled": 80,
      "watermarked": 10
    },
    "generation_completion_latency": {
      "count": 100,
      "p50_seconds": 0.4738889250147622,
      "p95_seconds": 0.9219275674127857,
      "p99_seconds": 1.0144956239516645
    },
    "latency_samples": {
      "client_delivery": 100,
      "generation_completion": 100,
      "validation": 20,
      "validation_lag": 20
    },
    "queue_depth": 0,
    "record_evidence": [
      {
        "attempts": 1,
        "content_digest": "e95086243fa92d7a707964369e34854b71a99b1009773e884f3c79f17895a6d1",
        "delivery_outcome": "delivered",
        "detector_call_id": "63cfbc42-4bcf-45c2-a2e9-0a233f8987ad",
        "guardrails_action": "block",
        "guardrails_action_id": "63cfbc42-4bcf-45c2-a2e9-0a233f8987ad",
        "ids_correlated": true,
        "key_id": "poc-2026-08",
        "managed_action": "blocked",
        "response_id": "chatcmpl-93eb7cc7b0a1e33d",
        "scheme": "kgw",
        "timing": {
          "validation_lag_seconds": 3.561301855370402e-05,
          "validation_latency_seconds": 0.5506727789761499
        },
        "validation_id": "63cfbc42-4bcf-45c2-a2e9-0a233f8987ad",
        "verdict": true
      },
      {
        "attempts": 1,
        "content_digest": "0476b8dc4428248f4c4c68d008f67e2185353b6abba8bbdf4cd1471d3c2548ed",
        "delivery_outcome": "delivered",
        "detector_call_id": "43cc5249-3d73-4d5f-b69d-a44467734640",
        "guardrails_action": "block",
        "guardrails_action_id": "43cc5249-3d73-4d5f-b69d-a44467734640",
        "ids_correlated": true,
        "key_id": "poc-2026-08",
        "managed_action": "blocked",
        "response_id": "chatcmpl-8cff6d440924bf62",
        "scheme": "kgw",
        "timing": {
          "validation_lag_seconds": 3.578100586310029e-05,
          "validation_latency_seconds": 0.7996050500078127
        },
        "validation_id": "43cc5249-3d73-4d5f-b69d-a44467734640",
        "verdict": true
      },
      {
        "attempts": 1,
        "content_digest": "e542ab3dea3cce3abb609e39fbcaf50a685a04c2549aea35919c3f3879c1f146",
        "delivery_outcome": "delivered",
        "detector_call_id": "bdf08aa3-617b-4e54-8ac8-a45d025ffee6",
        "guardrails_action": "block",
        "guardrails_action_id": "bdf08aa3-617b-4e54-8ac8-a45d025ffee6",
        "ids_correlated": true,
        "key_id": "poc-2026-08",
        "managed_action": "blocked",
        "response_id": "chatcmpl-a7c5aa320de93854",
        "scheme": "kgw",
        "timing": {
          "validation_lag_seconds": 3.4886994399130344e-05,
          "validation_latency_seconds": 0.8459617439948488
        },
        "validation_id": "bdf08aa3-617b-4e54-8ac8-a45d025ffee6",
        "verdict": true
      },
      {
        "attempts": 1,
        "content_digest": "f3cc24c726b6023d3befa59cd312076f14e9e642227462d28d3408bfee46dafd",
        "delivery_outcome": "delivered",
        "detector_call_id": "b7d8b85b-f2cd-4267-9fd7-31b2fffe8e26",
        "guardrails_action": "block",
        "guardrails_action_id": "b7d8b85b-f2cd-4267-9fd7-31b2fffe8e26",
        "ids_correlated": true,
        "key_id": "poc-2026-08",
        "managed_action": "blocked",
        "response_id": "chatcmpl-bd672c9a2063ec86",
        "scheme": "kgw",
        "timing": {
          "validation_lag_seconds": 3.4577009500935674e-05,
          "validation_latency_seconds": 0.9275513339962345
        },
        "validation_id": "b7d8b85b-f2cd-4267-9fd7-31b2fffe8e26",
        "verdict": true
      },
      {
        "attempts": 1,
        "content_digest": "b66f6859f106f0fde1ad8563c4dad88e15b23b1f47f0c7f564a903015b74a37d",
        "delivery_outcome": "delivered",
        "detector_call_id": "f52fe000-e032-45af-bf75-761f80b5fc17",
        "guardrails_action": "block",
        "guardrails_action_id": "f52fe000-e032-45af-bf75-761f80b5fc17",
        "ids_correlated": true,
        "key_id": "poc-2026-08",
        "managed_action": "blocked",
        "response_id": "chatcmpl-97eeb9e733021e2e",
        "scheme": "kgw",
        "timing": {
          "validation_lag_seconds": 3.588400431908667e-05,
          "validation_latency_seconds": 0.9430541369947605
        },
        "validation_id": "f52fe000-e032-45af-bf75-761f80b5fc17",
        "verdict": true
      },
      {
        "attempts": 1,
        "content_digest": "3dc7365f6068899261f5e16f1c312e8086f4915a9824e7987dab88c80832546e",
        "delivery_outcome": "delivered",
        "detector_call_id": "4fc41a15-39f1-4445-986e-71291c92fc08",
        "guardrails_action": "block",
        "guardrails_action_id": "4fc41a15-39f1-4445-986e-71291c92fc08",
        "ids_correlated": true,
        "key_id": "poc-2026-08",
        "managed_action": "blocked",
        "response_id": "chatcmpl-91eb10fb21031e6f",
        "scheme": "synthid",
        "timing": {
          "validation_lag_seconds": 3.64459992852062e-05,
          "validation_latency_seconds": 0.04605561701464467
        },
        "validation_id": "4fc41a15-39f1-4445-986e-71291c92fc08",
        "verdict": true
      },
      {
        "attempts": 1,
        "content_digest": "1729a4a87988576104c5d570200bd1bb2e514af85dd626ed316f022d7181609f",
        "delivery_outcome": "delivered",
        "detector_call_id": "8b8a05b1-a001-422e-a902-0d5df01ab70d",
        "guardrails_action": "block",
        "guardrails_action_id": "8b8a05b1-a001-422e-a902-0d5df01ab70d",
        "ids_correlated": true,
        "key_id": "poc-2026-08",
        "managed_action": "blocked",
        "response_id": "chatcmpl-8fbae62f5c7a847f",
        "scheme": "synthid",
        "timing": {
          "validation_lag_seconds": 3.55039956048131e-05,
          "validation_latency_seconds": 0.06159901001956314
        },
        "validation_id": "8b8a05b1-a001-422e-a902-0d5df01ab70d",
        "verdict": true
      },
      {
        "attempts": 1,
        "content_digest": "5b698665b970af241caa53378495b912d785be3d58f37985fdee867b4753ab06",
        "delivery_outcome": "delivered",
        "detector_call_id": "0ed59b1a-ebf9-4ef6-a4fc-a82e5abe3a1c",
        "guardrails_action": "block",
        "guardrails_action_id": "0ed59b1a-ebf9-4ef6-a4fc-a82e5abe3a1c",
        "ids_correlated": true,
        "key_id": "poc-2026-08",
        "managed_action": "blocked",
        "response_id": "chatcmpl-b1cf074457aa9448",
        "scheme": "synthid",
        "timing": {
          "validation_lag_seconds": 3.56219825334847e-05,
          "validation_latency_seconds": 0.057481905998429283
        },
        "validation_id": "0ed59b1a-ebf9-4ef6-a4fc-a82e5abe3a1c",
        "verdict": true
      },
      {
        "attempts": 1,
        "content_digest": "614c8d731c9b52a2ac4f20f6ca28956cf1a87873f570212b6f6d984a4d80ed78",
        "delivery_outcome": "delivered",
        "detector_call_id": "bd4ee186-3869-40a0-9d11-9e9016eac366",
        "guardrails_action": "block",
        "guardrails_action_id": "bd4ee186-3869-40a0-9d11-9e9016eac366",
        "ids_correlated": true,
        "key_id": "poc-2026-08",
        "managed_action": "blocked",
        "response_id": "chatcmpl-871d57aabee7bcbc",
        "scheme": "synthid",
        "timing": {
          "validation_lag_seconds": 3.448600182309747e-05,
          "validation_latency_seconds": 0.035405653005000204
        },
        "validation_id": "bd4ee186-3869-40a0-9d11-9e9016eac366",
        "verdict": true
      },
      {
        "attempts": 1,
        "content_digest": "614c8d731c9b52a2ac4f20f6ca28956cf1a87873f570212b6f6d984a4d80ed78",
        "delivery_outcome": "delivered",
        "detector_call_id": "7c1dc6bc-1783-4c18-9676-893aa86ed5e0",
        "guardrails_action": "block",
        "guardrails_action_id": "7c1dc6bc-1783-4c18-9676-893aa86ed5e0",
        "ids_correlated": true,
        "key_id": "poc-2026-08",
        "managed_action": "blocked",
        "response_id": "chatcmpl-9951b87557e36353",
        "scheme": "synthid",
        "timing": {
          "validation_lag_seconds": 3.822901635430753e-05,
          "validation_latency_seconds": 0.03722644998924807
        },
        "validation_id": "7c1dc6bc-1783-4c18-9676-893aa86ed5e0",
        "verdict": true
      },
      {
        "attempts": 1,
        "content_digest": "ce162f8d4e11b56532a77bcb93e8c071cac6a03b05ee667c3c7bf8ad53a37205",
        "delivery_outcome": "delivered",
        "detector_call_id": "bf116516-2fad-4d99-b215-2f15fc240929",
        "guardrails_action": "pass",
        "guardrails_action_id": "bf116516-2fad-4d99-b215-2f15fc240929",
        "ids_correlated": true,
        "key_id": "poc-2026-08",
        "managed_action": "success",
        "response_id": "chatcmpl-89cc4a1a1fa06f27",
        "scheme": "kgw",
        "timing": {
          "validation_lag_seconds": 5.000398959964514e-05,
          "validation_latency_seconds": 0.5309865710150916
        },
        "validation_id": "bf116516-2fad-4d99-b215-2f15fc240929",
        "verdict": false
      },
      {
        "attempts": 1,
        "content_digest": "0c6c85a4bd814ea9a46695c8b9839c0c0ae2abaece6a4f4b0b09dffef49031b9",
        "delivery_outcome": "delivered",
        "detector_call_id": "b0f11d77-fa45-4e4b-baf5-e43099e751d2",
        "guardrails_action": "pass",
        "guardrails_action_id": "b0f11d77-fa45-4e4b-baf5-e43099e751d2",
        "ids_correlated": true,
        "key_id": "poc-2026-08",
        "managed_action": "success",
        "response_id": "chatcmpl-befe245aa216ca7b",
        "scheme": "kgw",
        "timing": {
          "validation_lag_seconds": 3.7040008464828134e-05,
          "validation_latency_seconds": 0.5792410520080011
        },
        "validation_id": "b0f11d77-fa45-4e4b-baf5-e43099e751d2",
        "verdict": false
      },
      {
        "attempts": 1,
        "content_digest": "d9ea2aeced18f3eafe9e559e019c40a8f91b14a52f67fa86e7b5170bb3c94912",
        "delivery_outcome": "delivered",
        "detector_call_id": "a8497a03-0120-4c14-8136-2951d12455a3",
        "guardrails_action": "pass",
        "guardrails_action_id": "a8497a03-0120-4c14-8136-2951d12455a3",
        "ids_correlated": true,
        "key_id": "poc-2026-08",
        "managed_action": "success",
        "response_id": "chatcmpl-95ed5bfcbbd1c406",
        "scheme": "kgw",
        "timing": {
          "validation_lag_seconds": 3.4594006137922406e-05,
          "validation_latency_seconds": 0.8531688159855548
        },
        "validation_id": "a8497a03-0120-4c14-8136-2951d12455a3",
        "verdict": false
      },
      {
        "attempts": 1,
        "content_digest": "23445d3259bef664f64438319253112ace0c72b3ce05acf3df0c83518df5f760",
        "delivery_outcome": "delivered",
        "detector_call_id": "6348ccd1-15ff-40d3-95fc-436d529d457e",
        "guardrails_action": "pass",
        "guardrails_action_id": "6348ccd1-15ff-40d3-95fc-436d529d457e",
        "ids_correlated": true,
        "key_id": "poc-2026-08",
        "managed_action": "success",
        "response_id": "chatcmpl-82bf58fa70530604",
        "scheme": "kgw",
        "timing": {
          "validation_lag_seconds": 3.955498686991632e-05,
          "validation_latency_seconds": 0.4884521919884719
        },
        "validation_id": "6348ccd1-15ff-40d3-95fc-436d529d457e",
        "verdict": false
      },
      {
        "attempts": 1,
        "content_digest": "74e19126bc9beb59430cddb6378e916cbffbaa7ff42223994bf75c497cfd6097",
        "delivery_outcome": "delivered",
        "detector_call_id": "0c49b557-586d-4edd-9412-31250a354c5b",
        "guardrails_action": "pass",
        "guardrails_action_id": "0c49b557-586d-4edd-9412-31250a354c5b",
        "ids_correlated": true,
        "key_id": "poc-2026-08",
        "managed_action": "success",
        "response_id": "chatcmpl-a57c5a78ab4ecb32",
        "scheme": "kgw",
        "timing": {
          "validation_lag_seconds": 4.0565995732322335e-05,
          "validation_latency_seconds": 0.6360854629892856
        },
        "validation_id": "0c49b557-586d-4edd-9412-31250a354c5b",
        "verdict": false
      },
      {
        "attempts": 1,
        "content_digest": "451e0961f4ccd69558f825f9486e099cfb2dff412e17d43d9338d79bce7cd943",
        "delivery_outcome": "delivered",
        "detector_call_id": "93cfe86b-bf41-4e19-8acf-ec9320e3ebbf",
        "guardrails_action": "pass",
        "guardrails_action_id": "93cfe86b-bf41-4e19-8acf-ec9320e3ebbf",
        "ids_correlated": true,
        "key_id": "poc-2026-08",
        "managed_action": "success",
        "response_id": "chatcmpl-91577a814c003580",
        "scheme": "synthid",
        "timing": {
          "validation_lag_seconds": 3.572201239876449e-05,
          "validation_latency_seconds": 0.041132370999548584
        },
        "validation_id": "93cfe86b-bf41-4e19-8acf-ec9320e3ebbf",
        "verdict": false
      },
      {
        "attempts": 1,
        "content_digest": "1311a913bddd5112b23d7b7a91e7e064361cb1ec9753c79afecbd0f7ae85ba54",
        "delivery_outcome": "delivered",
        "detector_call_id": "b1f74beb-add1-476b-a38e-9d9027a703ab",
        "guardrails_action": "pass",
        "guardrails_action_id": "b1f74beb-add1-476b-a38e-9d9027a703ab",
        "ids_correlated": true,
        "key_id": "poc-2026-08",
        "managed_action": "success",
        "response_id": "chatcmpl-bcef0394c3c18684",
        "scheme": "synthid",
        "timing": {
          "validation_lag_seconds": 4.1888997657224536e-05,
          "validation_latency_seconds": 0.047180964989820495
        },
        "validation_id": "b1f74beb-add1-476b-a38e-9d9027a703ab",
        "verdict": false
      },
      {
        "attempts": 1,
        "content_digest": "bf6a556e310382886814e27a5e83a29e4c985182ca5dfde387af521b28673534",
        "delivery_outcome": "delivered",
        "detector_call_id": "193ab198-a7de-4ede-8f39-c80470769802",
        "guardrails_action": "pass",
        "guardrails_action_id": "193ab198-a7de-4ede-8f39-c80470769802",
        "ids_correlated": true,
        "key_id": "poc-2026-08",
        "managed_action": "success",
        "response_id": "chatcmpl-ab9def8f0ef19470",
        "scheme": "synthid",
        "timing": {
          "validation_lag_seconds": 3.7927995435893536e-05,
          "validation_latency_seconds": 0.04385608300799504
        },
        "validation_id": "193ab198-a7de-4ede-8f39-c80470769802",
        "verdict": false
      },
      {
        "attempts": 1,
        "content_digest": "9cb083be210fe7e506f318b87f1f2868207fe1dc07618900682901a62375400e",
        "delivery_outcome": "delivered",
        "detector_call_id": "efd3552c-2ad6-4d23-b677-6625822c487a",
        "guardrails_action": "pass",
        "guardrails_action_id": "efd3552c-2ad6-4d23-b677-6625822c487a",
        "ids_correlated": true,
        "key_id": "poc-2026-08",
        "managed_action": "success",
        "response_id": "chatcmpl-bc0c3e6c9727c538",
        "scheme": "synthid",
        "timing": {
          "validation_lag_seconds": 4.190197796560824e-05,
          "validation_latency_seconds": 0.04210138201597147
        },
        "validation_id": "efd3552c-2ad6-4d23-b677-6625822c487a",
        "verdict": false
      },
      {
        "attempts": 1,
        "content_digest": "a26835a32bbae0220dba71ab0ee1f88e90d592e02cbfe7c7a90177fcabe55058",
        "delivery_outcome": "delivered",
        "detector_call_id": "d4f11032-81f0-4d4e-a7cd-fc9d504a9068",
        "guardrails_action": "pass",
        "guardrails_action_id": "d4f11032-81f0-4d4e-a7cd-fc9d504a9068",
        "ids_correlated": true,
        "key_id": "poc-2026-08",
        "managed_action": "success",
        "response_id": "chatcmpl-9d9946d48b853902",
        "scheme": "synthid",
        "timing": {
          "validation_lag_seconds": 3.906100755557418e-05,
          "validation_latency_seconds": 0.0443791389989201
        },
        "validation_id": "d4f11032-81f0-4d4e-a7cd-fc9d504a9068",
        "verdict": false
      }
    ],
    "responses": 100,
    "selected": 20,
    "terminal": 20,
    "validation_lag": {
      "count": 20,
      "p50_seconds": 3.6165001802146435e-05,
      "p95_seconds": 4.230707854731009e-05,
      "p99_seconds": 4.846460738917811e-05
    },
    "validation_latency": {
      "count": 20,
      "p50_seconds": 0.2750256010040175,
      "p95_seconds": 0.9283264741461608,
      "p99_seconds": 0.9401086044250405
    }
  },
  "observability": {
    "marker_count": 6,
    "required_metric_count": 10
  },
  "passed": true,
  "policy_semantics": {
    "gateway_positive_delivery": "flag",
    "managed_guardrails_positive_action": "block",
    "mode": "synchronous",
    "validation_failure": "closed"
  },
  "queue": {
    "overflow_policy": "non_blocking",
    "peak_depth": 2,
    "queue_depth": 0,
    "queue_overflow": 1,
    "terminal_records": 3,
    "validated_records": 2
  },
  "unsampled_baseline": {
    "client_delivery_latency": {
      "count": 4,
      "p50_seconds": 0.49661912299052346,
      "p95_seconds": 0.8561615062993951,
      "p99_seconds": 0.9001360316597857
    },
    "counters": {
      "cancelled": 0,
      "clean": 0,
      "completed": 4,
      "detector_attempts": 0,
      "dropped": 0,
      "errors": 0,
      "failed": 0,
      "guardrails_attempts": 0,
      "queue_overflow": 0,
      "retries": 0,
      "selected": 0,
      "started": 4,
      "terminal": 0,
      "unsampled": 4,
      "watermarked": 0
    },
    "generation_completion_latency": {
      "count": 4,
      "p50_seconds": 0.49345759249990806,
      "p95_seconds": 0.8530325526677189,
      "p99_seconds": 0.8969949113499025
    },
    "latency_samples": {
      "client_delivery": 4,
      "generation_completion": 4,
      "validation": 0,
      "validation_lag": 0
    },
    "queue_depth": 0,
    "responses": 4,
    "sample_every": 5,
    "selected": 0
  }
}
```

### Cleanup and evidence boundary

**Claim (`EXECUTED`; source: trap-owned cleanup and postflight below):** the full
report passed, both per-run arrays contained 20 records, the capture contained zero
secret markers, the base deployment restored test controls to off, the gateway
returned ready, and the GPU MachineSet returned to `0/0/0`.

```text
D10_HARNESS result=passed secret_leaks=0
D10_CLEANUP controls=off gateway_ready=1 gpu_zero=1 restore_apply_rc=0 restore_rollout_rc=0 scale_down_rc=0
CURRENT_D10_POSTFLIGHT {"detector_image_ids":["sha256:2bb606b8e08b039b28f4b22356840f4cb0855d05fa688da4f4b70d26d8d0ea0f"],"detector_synthid_image_ids":["sha256:2bb606b8e08b039b28f4b22356840f4cb0855d05fa688da4f4b70d26d8d0ea0f"],"gateway_image_ids":["sha256:9443345c272c55a6dcea41bae84f8e04bab5d0f27e0934df2e4b4ff11b37fed6"],"nemo_server_image_ids":["sha256:22125dbbd05d1cfa7af931fdeca9da72c6d267a05c33a1f757daf3095ddcca7c"],"services_ready":true,"gpu":"0/0/0"}
```

**Scope (`EXECUTED`/`OPEN`; source: this report and the preceding D10 record):**
this rerun establishes the configuration, unsampled baseline, fixed `N=1`/`N=5`
matrix, both scheme controls, injected retries/malformed response, capacity-two
overflow, metrics schema, gateway redacted-event/metric secret scans, exact
correlation, counters, and gateway-response-ready latency against the current
build-5 detector. The earlier real detector-outage and eight-surface finite scan
remain valid for their recorded detector revision and were not repeated here.
External pass-through and the production boundaries listed under D10 remain
`OPEN`.

<a id="current-build5-d10-mode-evidence-2026-08-09"></a>

## 2026-08-09 — Mode-complete current build-5 D10 rerun (EXECUTED; redacted)

**Evidence correction (`STATIC`/`EXECUTED`; source: harness projection review,
regression test, and complete live stdout below):** independent completion review
found that the preceding full report's harness had validated every selected record's
`mode` field internally but omitted that field from its emitted
`record_evidence` projection. This was an evidence-format gap against D10
acceptance item 7, not a runtime validation failure. The projection now retains
`mode`, focused and complete local suites passed, and the full live matrix reran.
Every one of the 40 selected-response records below literally carries
`"mode": "synchronous"`.

### Exact-code verification

```text
$ python3 -m pytest -q validation/tests/test_harness_integration.py
8 passed, 18 warnings in 1.98s

$ python3 -m pytest -q
288 passed, 498 warnings in 510.69s (0:08:30)

$ python3 -m compileall -q src detector validation benchmarks
[no output; exit 0]

$ python3 benchmarks/continuous_validation.py --self-test
{"content_logged": false, "fault_cases": 3, "n1": {"responses": 20, "selected": 20}, "n5": {"responses": 100, "selected": 20}, "queue_overflow_checked": true, "self_test": "passed"}

$ git diff --check
[no output; exit 0]
```

Only the local acceptance projection and its regression assertion changed. The
deployed gateway, vLLM, managed-NeMo, and build-5 detector runtime bytes therefore
required no rebuild for this evidence-only rerun.

### Live preflight and exact invocation

**Claim (`EXECUTED`; source: bounded query below):** the same current immutable
stack was ready with controls off, the predictor health/model probes returned 200,
and the GPU MachineSet reached `1/1/1` without a quota error.

```text
D10_MODE_PREFLIGHT {"deployments":[{"name":"detector","desired":1,"ready":1,"controls":"n/a","image_digest":"sha256:2bb606b8e08b039b28f4b22356840f4cb0855d05fa688da4f4b70d26d8d0ea0f"},{"name":"detector-synthid","desired":1,"ready":1,"controls":"n/a","image_digest":"sha256:2bb606b8e08b039b28f4b22356840f4cb0855d05fa688da4f4b70d26d8d0ea0f"},{"name":"watermark-validation-gateway","desired":1,"ready":1,"controls":"off","image_digest":"sha256:9443345c272c55a6dcea41bae84f8e04bab5d0f27e0934df2e4b4ff11b37fed6"},{"name":"watermark-vllm-predictor","desired":1,"ready":1,"controls":"n/a","image_digest":"sha256:f8294ee0459869e9659b1178ed91f57a1b52a52c6a5f5f819ca651646b317e4c"}],"nemo":{"phase":"Ready"},"probe":{"health":200,"models":200},"gpu":{"desired":1,"current":1,"ready":1},"passed":true}

D10_MODE_STAGE secret_markers_loaded=true
D10_MODE_STAGE controls=on gateway_rollout=ready
D10_MODE_STAGE port_forward=ready

$ python3 benchmarks/continuous_validation.py \
    --gateway-url http://127.0.0.1:18080 \
    --model watermark-vllm \
    --key-id "$D10_KEY_ID" \
    --admin-token-env D10_ADMIN_TOKEN \
    --secret-marker-env D10_ADMIN_TOKEN \
    --secret-marker-env D10_BROKER_TOKEN \
    --secret-marker-env D10_NEMO_TOKEN \
    --secret-marker-env D10_WATERMARK_KEY \
    --secret-marker-env D10_SIGNING_KEY \
    --secret-marker-env D10_MODEL_TOKEN \
    --max-tokens 256 --temperature 0.7 \
    --timeout-seconds 120 --queue-pending-check-seconds 0.5 \
    --positive-action block --clean-action pass \
    --expected-mode synchronous --expected-failure-policy closed
```

The complete mode-`600`, content-safe stdout follows. Its SHA-256 including the
final newline is
`5baf26508ba75362ba0bf6d39086fceb8f628e0080ce807bd80fe8d2daa9452c`.
The capture was scanned against all six live secret values and contained zero
matches; it has no plaintext-response field.

```json
{
  "configuration": {
    "0": "rejected",
    "1": "accepted",
    "5": "accepted",
    "empty": "rejected",
    "fraction": "rejected",
    "negative": "rejected",
    "nonnumeric": "rejected"
  },
  "content_logged": false,
  "contract": "phase5-v1",
  "faults": {
    "malformed_success": {
      "attempts": 1,
      "retries": 0,
      "terminal_state": "malformed_response"
    },
    "retry_exhausted": {
      "attempts": 3,
      "retries": 2,
      "terminal_state": "retry_exhausted"
    },
    "retry_then_success": {
      "attempts": 2,
      "retries": 1,
      "terminal_state": "success"
    }
  },
  "latency_semantics": {
    "client_delivery": "request_start_to_gateway_response_ready",
    "generation_completion": "request_start_to_upstream_completion",
    "validation": "validation_attempt_window",
    "validation_lag": "validation_queue_wait_to_attempt_start"
  },
  "n1": {
    "client_delivery_latency": {
      "count": 20,
      "p50_seconds": 0.8669895674829604,
      "p95_seconds": 1.6276022721169288,
      "p99_seconds": 1.988435605617123
    },
    "counters": {
      "cancelled": 0,
      "clean": 10,
      "completed": 20,
      "detector_attempts": 20,
      "dropped": 0,
      "errors": 0,
      "failed": 0,
      "guardrails_attempts": 20,
      "queue_overflow": 0,
      "retries": 0,
      "selected": 20,
      "started": 20,
      "terminal": 20,
      "unsampled": 0,
      "watermarked": 10
    },
    "generation_completion_latency": {
      "count": 20,
      "p50_seconds": 0.5027875705127371,
      "p95_seconds": 1.0556942602488566,
      "p99_seconds": 1.4528903584621724
    },
    "latency_samples": {
      "client_delivery": 20,
      "generation_completion": 20,
      "validation": 20,
      "validation_lag": 20
    },
    "queue_depth": 0,
    "record_evidence": [
      {
        "attempts": 1,
        "content_digest": "26d3464eb84a331826923d4a1fc347d6bc0d26d8de57b3a81491016798ade995",
        "delivery_outcome": "delivered",
        "detector_call_id": "54849bce-ed06-4b16-b54e-98e01bb8ddf9",
        "guardrails_action": "block",
        "guardrails_action_id": "54849bce-ed06-4b16-b54e-98e01bb8ddf9",
        "ids_correlated": true,
        "key_id": "poc-2026-08",
        "managed_action": "blocked",
        "mode": "synchronous",
        "response_id": "chatcmpl-9d8ec40eb4191ca4",
        "scheme": "kgw",
        "timing": {
          "validation_lag_seconds": 4.633798380382359e-05,
          "validation_latency_seconds": 0.46456600399687886
        },
        "validation_id": "54849bce-ed06-4b16-b54e-98e01bb8ddf9",
        "verdict": true
      },
      {
        "attempts": 1,
        "content_digest": "b7cb7bb51069ac8733a13a71b54f6fa119bd15953b1ca784c7aa4a85232149de",
        "delivery_outcome": "delivered",
        "detector_call_id": "5e17d58d-d78c-4d7b-abb8-a6d30c0d14fa",
        "guardrails_action": "block",
        "guardrails_action_id": "5e17d58d-d78c-4d7b-abb8-a6d30c0d14fa",
        "ids_correlated": true,
        "key_id": "poc-2026-08",
        "managed_action": "blocked",
        "mode": "synchronous",
        "response_id": "chatcmpl-86500e8edea84e63",
        "scheme": "kgw",
        "timing": {
          "validation_lag_seconds": 4.001901834271848e-05,
          "validation_latency_seconds": 0.5242233559838496
        },
        "validation_id": "5e17d58d-d78c-4d7b-abb8-a6d30c0d14fa",
        "verdict": true
      },
      {
        "attempts": 1,
        "content_digest": "70e44c079d3a668c5cc58539259fb919a05d0d9b792f226ca424bea8e3e2cca9",
        "delivery_outcome": "delivered",
        "detector_call_id": "f399c8e5-6f8d-4c78-8dca-c371a0b2b658",
        "guardrails_action": "block",
        "guardrails_action_id": "f399c8e5-6f8d-4c78-8dca-c371a0b2b658",
        "ids_correlated": true,
        "key_id": "poc-2026-08",
        "managed_action": "blocked",
        "mode": "synchronous",
        "response_id": "chatcmpl-a69a09ca4e40de8b",
        "scheme": "kgw",
        "timing": {
          "validation_lag_seconds": 3.041198942810297e-05,
          "validation_latency_seconds": 0.7280435290012974
        },
        "validation_id": "f399c8e5-6f8d-4c78-8dca-c371a0b2b658",
        "verdict": true
      },
      {
        "attempts": 1,
        "content_digest": "96ea28755677cba49d989b1c6c3992d6595139143e182082152792f424b2c255",
        "delivery_outcome": "delivered",
        "detector_call_id": "dc96276b-0ee9-4b6b-b269-765274692c3c",
        "guardrails_action": "block",
        "guardrails_action_id": "dc96276b-0ee9-4b6b-b269-765274692c3c",
        "ids_correlated": true,
        "key_id": "poc-2026-08",
        "managed_action": "blocked",
        "mode": "synchronous",
        "response_id": "chatcmpl-9ccfba7ab68cdb0a",
        "scheme": "kgw",
        "timing": {
          "validation_lag_seconds": 3.2728014048188925e-05,
          "validation_latency_seconds": 0.4392082240083255
        },
        "validation_id": "dc96276b-0ee9-4b6b-b269-765274692c3c",
        "verdict": true
      },
      {
        "attempts": 1,
        "content_digest": "588935709ab71beaa9c36ed206ff684bdad0afcc7c07fc0f5021af2104ece97e",
        "delivery_outcome": "delivered",
        "detector_call_id": "867b56ae-3328-430f-aa3f-02593640d0e1",
        "guardrails_action": "block",
        "guardrails_action_id": "867b56ae-3328-430f-aa3f-02593640d0e1",
        "ids_correlated": true,
        "key_id": "poc-2026-08",
        "managed_action": "blocked",
        "mode": "synchronous",
        "response_id": "chatcmpl-883c8bad819dfb40",
        "scheme": "kgw",
        "timing": {
          "validation_lag_seconds": 4.218801041133702e-05,
          "validation_latency_seconds": 1.0671527659869753
        },
        "validation_id": "867b56ae-3328-430f-aa3f-02593640d0e1",
        "verdict": true
      },
      {
        "attempts": 1,
        "content_digest": "6aa5127c60b8437ac6115b9e24fa9c1f025ddc524fc14c9d54ef974852658086",
        "delivery_outcome": "delivered",
        "detector_call_id": "365de86d-073d-46c9-a468-b2dedab48b42",
        "guardrails_action": "block",
        "guardrails_action_id": "365de86d-073d-46c9-a468-b2dedab48b42",
        "ids_correlated": true,
        "key_id": "poc-2026-08",
        "managed_action": "blocked",
        "mode": "synchronous",
        "response_id": "chatcmpl-b29546032f7cfee7",
        "scheme": "synthid",
        "timing": {
          "validation_lag_seconds": 3.239599755033851e-05,
          "validation_latency_seconds": 0.04282429098384455
        },
        "validation_id": "365de86d-073d-46c9-a468-b2dedab48b42",
        "verdict": true
      },
      {
        "attempts": 1,
        "content_digest": "614c8d731c9b52a2ac4f20f6ca28956cf1a87873f570212b6f6d984a4d80ed78",
        "delivery_outcome": "delivered",
        "detector_call_id": "c948d18c-d303-4d65-abee-84108b0682be",
        "guardrails_action": "block",
        "guardrails_action_id": "c948d18c-d303-4d65-abee-84108b0682be",
        "ids_correlated": true,
        "key_id": "poc-2026-08",
        "managed_action": "blocked",
        "mode": "synchronous",
        "response_id": "chatcmpl-92371039a5e8d6e5",
        "scheme": "synthid",
        "timing": {
          "validation_lag_seconds": 3.1207018764689565e-05,
          "validation_latency_seconds": 0.03589801097405143
        },
        "validation_id": "c948d18c-d303-4d65-abee-84108b0682be",
        "verdict": true
      },
      {
        "attempts": 1,
        "content_digest": "754340269111828d8a74d8ef4b9475804e3376f5e856a2e8683893baec6494d4",
        "delivery_outcome": "delivered",
        "detector_call_id": "ec8b160e-1ee6-40e9-9d02-e4fc01e3635e",
        "guardrails_action": "block",
        "guardrails_action_id": "ec8b160e-1ee6-40e9-9d02-e4fc01e3635e",
        "ids_correlated": true,
        "key_id": "poc-2026-08",
        "managed_action": "blocked",
        "mode": "synchronous",
        "response_id": "chatcmpl-8c1a7b5675b8fe80",
        "scheme": "synthid",
        "timing": {
          "validation_lag_seconds": 3.548999666236341e-05,
          "validation_latency_seconds": 0.04157481400761753
        },
        "validation_id": "ec8b160e-1ee6-40e9-9d02-e4fc01e3635e",
        "verdict": true
      },
      {
        "attempts": 1,
        "content_digest": "428f325bc87d38fd9489fa71410f5cae1754c2b4d414ddc36188ebc9b942ca3e",
        "delivery_outcome": "delivered",
        "detector_call_id": "a3cd184b-a36b-40b0-b7c4-95475de19c6d",
        "guardrails_action": "block",
        "guardrails_action_id": "a3cd184b-a36b-40b0-b7c4-95475de19c6d",
        "ids_correlated": true,
        "key_id": "poc-2026-08",
        "managed_action": "blocked",
        "mode": "synchronous",
        "response_id": "chatcmpl-b193708bc5a5d871",
        "scheme": "synthid",
        "timing": {
          "validation_lag_seconds": 2.9937014915049076e-05,
          "validation_latency_seconds": 0.03556919898255728
        },
        "validation_id": "a3cd184b-a36b-40b0-b7c4-95475de19c6d",
        "verdict": true
      },
      {
        "attempts": 1,
        "content_digest": "97bb97d996ce857c079d0e83d65a49d1fc69d06467a6992820eac718b6302285",
        "delivery_outcome": "delivered",
        "detector_call_id": "ba06c7ef-4197-4b39-ba27-565ffb6c0b1a",
        "guardrails_action": "block",
        "guardrails_action_id": "ba06c7ef-4197-4b39-ba27-565ffb6c0b1a",
        "ids_correlated": true,
        "key_id": "poc-2026-08",
        "managed_action": "blocked",
        "mode": "synchronous",
        "response_id": "chatcmpl-b0cadf92e998c9fd",
        "scheme": "synthid",
        "timing": {
          "validation_lag_seconds": 3.2397976610809565e-05,
          "validation_latency_seconds": 0.045650493004359305
        },
        "validation_id": "ba06c7ef-4197-4b39-ba27-565ffb6c0b1a",
        "verdict": true
      },
      {
        "attempts": 1,
        "content_digest": "70a252d2cc0d126367fb30cfb502cc8761539e408a96453335c2d8e647b72a8f",
        "delivery_outcome": "delivered",
        "detector_call_id": "3a7f3895-cacf-49b6-8346-107fbc969c56",
        "guardrails_action": "pass",
        "guardrails_action_id": "3a7f3895-cacf-49b6-8346-107fbc969c56",
        "ids_correlated": true,
        "key_id": "poc-2026-08",
        "managed_action": "success",
        "mode": "synchronous",
        "response_id": "chatcmpl-a60771784a0bcfbe",
        "scheme": "kgw",
        "timing": {
          "validation_lag_seconds": 3.3370975870639086e-05,
          "validation_latency_seconds": 0.684702777012717
        },
        "validation_id": "3a7f3895-cacf-49b6-8346-107fbc969c56",
        "verdict": false
      },
      {
        "attempts": 1,
        "content_digest": "ca9f392f17b098ec0e43e13ad9756989d95c10ff25fbbd071fc55c31d2570fc5",
        "delivery_outcome": "delivered",
        "detector_call_id": "931ff696-776e-4246-8a2b-81953ba5cba5",
        "guardrails_action": "pass",
        "guardrails_action_id": "931ff696-776e-4246-8a2b-81953ba5cba5",
        "ids_correlated": true,
        "key_id": "poc-2026-08",
        "managed_action": "success",
        "mode": "synchronous",
        "response_id": "chatcmpl-b57aaf6b54912994",
        "scheme": "kgw",
        "timing": {
          "validation_lag_seconds": 4.102601087652147e-05,
          "validation_latency_seconds": 0.22617158998036757
        },
        "validation_id": "931ff696-776e-4246-8a2b-81953ba5cba5",
        "verdict": false
      },
      {
        "attempts": 1,
        "content_digest": "550b36e7d71f6d4c7bd63e078ab01a3f5c5565b6fe7de73b9186d4850f902236",
        "delivery_outcome": "delivered",
        "detector_call_id": "30e30a88-def5-44cc-a97e-5d70379150bb",
        "guardrails_action": "pass",
        "guardrails_action_id": "30e30a88-def5-44cc-a97e-5d70379150bb",
        "ids_correlated": true,
        "key_id": "poc-2026-08",
        "managed_action": "success",
        "mode": "synchronous",
        "response_id": "chatcmpl-b645661f8c4e6934",
        "scheme": "kgw",
        "timing": {
          "validation_lag_seconds": 3.340400871820748e-05,
          "validation_latency_seconds": 0.4615720169967972
        },
        "validation_id": "30e30a88-def5-44cc-a97e-5d70379150bb",
        "verdict": false
      },
      {
        "attempts": 1,
        "content_digest": "1a72e9352f42383b0ec4f55a37402a1a3ffe02a35fbfdbc80e64e7c64c6ca634",
        "delivery_outcome": "delivered",
        "detector_call_id": "697eb668-74f5-47b8-99f5-367f3e0b25d1",
        "guardrails_action": "pass",
        "guardrails_action_id": "697eb668-74f5-47b8-99f5-367f3e0b25d1",
        "ids_correlated": true,
        "key_id": "poc-2026-08",
        "managed_action": "success",
        "mode": "synchronous",
        "response_id": "chatcmpl-b62e75b4a0837f4f",
        "scheme": "kgw",
        "timing": {
          "validation_lag_seconds": 3.411501529626548e-05,
          "validation_latency_seconds": 0.6748240909946617
        },
        "validation_id": "697eb668-74f5-47b8-99f5-367f3e0b25d1",
        "verdict": false
      },
      {
        "attempts": 1,
        "content_digest": "bce07f766131bcb131873c1a4c7a850b4d72d882a083cb09e7c42c8c85c015dd",
        "delivery_outcome": "delivered",
        "detector_call_id": "1b8d4646-cc3c-4e34-ab36-85f9f8f0f7be",
        "guardrails_action": "pass",
        "guardrails_action_id": "1b8d4646-cc3c-4e34-ab36-85f9f8f0f7be",
        "ids_correlated": true,
        "key_id": "poc-2026-08",
        "managed_action": "success",
        "mode": "synchronous",
        "response_id": "chatcmpl-89ad2731c48530a2",
        "scheme": "kgw",
        "timing": {
          "validation_lag_seconds": 3.233598545193672e-05,
          "validation_latency_seconds": 0.48437985102646053
        },
        "validation_id": "1b8d4646-cc3c-4e34-ab36-85f9f8f0f7be",
        "verdict": false
      },
      {
        "attempts": 1,
        "content_digest": "db77dd35fada3dbeeb06b53e15306afba5bfacfb8401a614d31663b8328fcf82",
        "delivery_outcome": "delivered",
        "detector_call_id": "64e5403c-eac1-43c9-a561-4fcb210c07fc",
        "guardrails_action": "pass",
        "guardrails_action_id": "64e5403c-eac1-43c9-a561-4fcb210c07fc",
        "ids_correlated": true,
        "key_id": "poc-2026-08",
        "managed_action": "success",
        "mode": "synchronous",
        "response_id": "chatcmpl-a5fc532eaadf161d",
        "scheme": "synthid",
        "timing": {
          "validation_lag_seconds": 3.175801248289645e-05,
          "validation_latency_seconds": 0.05027908898773603
        },
        "validation_id": "64e5403c-eac1-43c9-a561-4fcb210c07fc",
        "verdict": false
      },
      {
        "attempts": 1,
        "content_digest": "7abe7012c0be9190f93f8c237c94880b9108e5b077e493b54eb27cc619d4b272",
        "delivery_outcome": "delivered",
        "detector_call_id": "b35f0f52-a664-48c5-bd88-3b56b653e328",
        "guardrails_action": "pass",
        "guardrails_action_id": "b35f0f52-a664-48c5-bd88-3b56b653e328",
        "ids_correlated": true,
        "key_id": "poc-2026-08",
        "managed_action": "success",
        "mode": "synchronous",
        "response_id": "chatcmpl-b93d618de02c238e",
        "scheme": "synthid",
        "timing": {
          "validation_lag_seconds": 3.19909886457026e-05,
          "validation_latency_seconds": 0.041069525002967566
        },
        "validation_id": "b35f0f52-a664-48c5-bd88-3b56b653e328",
        "verdict": false
      },
      {
        "attempts": 1,
        "content_digest": "8de4026f28fbd21f5a248b16c1e031a2b39415e22de917fa55dd7c2a5275bea4",
        "delivery_outcome": "delivered",
        "detector_call_id": "f0f796ee-f4e2-4cac-a57f-724ee5de5c9b",
        "guardrails_action": "pass",
        "guardrails_action_id": "f0f796ee-f4e2-4cac-a57f-724ee5de5c9b",
        "ids_correlated": true,
        "key_id": "poc-2026-08",
        "managed_action": "success",
        "mode": "synchronous",
        "response_id": "chatcmpl-af10ea9aa0b6e57e",
        "scheme": "synthid",
        "timing": {
          "validation_lag_seconds": 2.99840176012367e-05,
          "validation_latency_seconds": 0.04315023199887946
        },
        "validation_id": "f0f796ee-f4e2-4cac-a57f-724ee5de5c9b",
        "verdict": false
      },
      {
        "attempts": 1,
        "content_digest": "d22fa0c53b3ca49b3edb51980554bbc29c0fd752e8250e7ba44cd1bd12a549d9",
        "delivery_outcome": "delivered",
        "detector_call_id": "c0c6924c-4cd5-4eb0-a5cc-c8038284ac5f",
        "guardrails_action": "pass",
        "guardrails_action_id": "c0c6924c-4cd5-4eb0-a5cc-c8038284ac5f",
        "ids_correlated": true,
        "key_id": "poc-2026-08",
        "managed_action": "success",
        "mode": "synchronous",
        "response_id": "chatcmpl-9583732dbe66ef8c",
        "scheme": "synthid",
        "timing": {
          "validation_lag_seconds": 3.336300142109394e-05,
          "validation_latency_seconds": 0.07652998901903629
        },
        "validation_id": "c0c6924c-4cd5-4eb0-a5cc-c8038284ac5f",
        "verdict": false
      },
      {
        "attempts": 1,
        "content_digest": "f59b177b550cfd6eb2504cbf3aff8314a223e98f287c58717c86d3c763235fe5",
        "delivery_outcome": "delivered",
        "detector_call_id": "2e74304e-f217-4eef-b541-6b7bb82643ff",
        "guardrails_action": "pass",
        "guardrails_action_id": "2e74304e-f217-4eef-b541-6b7bb82643ff",
        "ids_correlated": true,
        "key_id": "poc-2026-08",
        "managed_action": "success",
        "mode": "synchronous",
        "response_id": "chatcmpl-bcaefc655d9e1c99",
        "scheme": "synthid",
        "timing": {
          "validation_lag_seconds": 3.3182004699483514e-05,
          "validation_latency_seconds": 0.048016933986218646
        },
        "validation_id": "2e74304e-f217-4eef-b541-6b7bb82643ff",
        "verdict": false
      }
    ],
    "responses": 20,
    "selected": 20,
    "terminal": 20,
    "validation_lag": {
      "count": 20,
      "p50_seconds": 3.295500937383622e-05,
      "p95_seconds": 4.239550908096135e-05,
      "p99_seconds": 4.5549488859251135e-05
    },
    "validation_latency": {
      "count": 20,
      "p50_seconds": 0.15135078949970193,
      "p95_seconds": 0.7449989908505815,
      "p99_seconds": 1.0027220109596962
    }
  },
  "n5": {
    "client_delivery_latency": {
      "count": 100,
      "p50_seconds": 0.49329461000161245,
      "p95_seconds": 1.4369361560937246,
      "p99_seconds": 1.811031836946205
    },
    "counters": {
      "cancelled": 0,
      "clean": 10,
      "completed": 100,
      "detector_attempts": 20,
      "dropped": 0,
      "errors": 0,
      "failed": 0,
      "guardrails_attempts": 20,
      "queue_overflow": 0,
      "retries": 0,
      "selected": 20,
      "started": 100,
      "terminal": 20,
      "unsampled": 80,
      "watermarked": 10
    },
    "generation_completion_latency": {
      "count": 100,
      "p50_seconds": 0.4761550800030818,
      "p95_seconds": 0.9260812611901201,
      "p99_seconds": 1.0070910610587485
    },
    "latency_samples": {
      "client_delivery": 100,
      "generation_completion": 100,
      "validation": 20,
      "validation_lag": 20
    },
    "queue_depth": 0,
    "record_evidence": [
      {
        "attempts": 1,
        "content_digest": "e95086243fa92d7a707964369e34854b71a99b1009773e884f3c79f17895a6d1",
        "delivery_outcome": "delivered",
        "detector_call_id": "701280f3-82d5-44b4-837f-c8cd71e451ef",
        "guardrails_action": "block",
        "guardrails_action_id": "701280f3-82d5-44b4-837f-c8cd71e451ef",
        "ids_correlated": true,
        "key_id": "poc-2026-08",
        "managed_action": "blocked",
        "mode": "synchronous",
        "response_id": "chatcmpl-aed04f41429b6f5d",
        "scheme": "kgw",
        "timing": {
          "validation_lag_seconds": 3.2387993996962905e-05,
          "validation_latency_seconds": 0.5294381369894836
        },
        "validation_id": "701280f3-82d5-44b4-837f-c8cd71e451ef",
        "verdict": true
      },
      {
        "attempts": 1,
        "content_digest": "0476b8dc4428248f4c4c68d008f67e2185353b6abba8bbdf4cd1471d3c2548ed",
        "delivery_outcome": "delivered",
        "detector_call_id": "95762259-589a-482c-9b20-05ceee2fd285",
        "guardrails_action": "block",
        "guardrails_action_id": "95762259-589a-482c-9b20-05ceee2fd285",
        "ids_correlated": true,
        "key_id": "poc-2026-08",
        "managed_action": "blocked",
        "mode": "synchronous",
        "response_id": "chatcmpl-964178d6faf0c0e6",
        "scheme": "kgw",
        "timing": {
          "validation_lag_seconds": 3.0759983928874135e-05,
          "validation_latency_seconds": 0.7255623530072626
        },
        "validation_id": "95762259-589a-482c-9b20-05ceee2fd285",
        "verdict": true
      },
      {
        "attempts": 1,
        "content_digest": "e542ab3dea3cce3abb609e39fbcaf50a685a04c2549aea35919c3f3879c1f146",
        "delivery_outcome": "delivered",
        "detector_call_id": "77417d2a-3b7d-47f5-a8cd-c09767040a79",
        "guardrails_action": "block",
        "guardrails_action_id": "77417d2a-3b7d-47f5-a8cd-c09767040a79",
        "ids_correlated": true,
        "key_id": "poc-2026-08",
        "managed_action": "blocked",
        "mode": "synchronous",
        "response_id": "chatcmpl-8558fa0cc9b4a97b",
        "scheme": "kgw",
        "timing": {
          "validation_lag_seconds": 3.489799564704299e-05,
          "validation_latency_seconds": 0.8864564090035856
        },
        "validation_id": "77417d2a-3b7d-47f5-a8cd-c09767040a79",
        "verdict": true
      },
      {
        "attempts": 1,
        "content_digest": "f3cc24c726b6023d3befa59cd312076f14e9e642227462d28d3408bfee46dafd",
        "delivery_outcome": "delivered",
        "detector_call_id": "978c3423-1b96-41f9-8cd3-0bde8c6a2bf1",
        "guardrails_action": "block",
        "guardrails_action_id": "978c3423-1b96-41f9-8cd3-0bde8c6a2bf1",
        "ids_correlated": true,
        "key_id": "poc-2026-08",
        "managed_action": "blocked",
        "mode": "synchronous",
        "response_id": "chatcmpl-a47776652665f6fc",
        "scheme": "kgw",
        "timing": {
          "validation_lag_seconds": 3.1060975743457675e-05,
          "validation_latency_seconds": 0.8038727060193196
        },
        "validation_id": "978c3423-1b96-41f9-8cd3-0bde8c6a2bf1",
        "verdict": true
      },
      {
        "attempts": 1,
        "content_digest": "b66f6859f106f0fde1ad8563c4dad88e15b23b1f47f0c7f564a903015b74a37d",
        "delivery_outcome": "delivered",
        "detector_call_id": "7b4e1a98-3ac8-440d-ac3c-ca68c3058256",
        "guardrails_action": "block",
        "guardrails_action_id": "7b4e1a98-3ac8-440d-ac3c-ca68c3058256",
        "ids_correlated": true,
        "key_id": "poc-2026-08",
        "managed_action": "blocked",
        "mode": "synchronous",
        "response_id": "chatcmpl-b4718507d46c61ef",
        "scheme": "kgw",
        "timing": {
          "validation_lag_seconds": 3.348197788000107e-05,
          "validation_latency_seconds": 1.0672340610180981
        },
        "validation_id": "7b4e1a98-3ac8-440d-ac3c-ca68c3058256",
        "verdict": true
      },
      {
        "attempts": 1,
        "content_digest": "3dc7365f6068899261f5e16f1c312e8086f4915a9824e7987dab88c80832546e",
        "delivery_outcome": "delivered",
        "detector_call_id": "4afdffc1-a396-40de-b3f5-ac7039ac113c",
        "guardrails_action": "block",
        "guardrails_action_id": "4afdffc1-a396-40de-b3f5-ac7039ac113c",
        "ids_correlated": true,
        "key_id": "poc-2026-08",
        "managed_action": "blocked",
        "mode": "synchronous",
        "response_id": "chatcmpl-96df61b488507548",
        "scheme": "synthid",
        "timing": {
          "validation_lag_seconds": 3.219398786313832e-05,
          "validation_latency_seconds": 0.046428149013081565
        },
        "validation_id": "4afdffc1-a396-40de-b3f5-ac7039ac113c",
        "verdict": true
      },
      {
        "attempts": 1,
        "content_digest": "1729a4a87988576104c5d570200bd1bb2e514af85dd626ed316f022d7181609f",
        "delivery_outcome": "delivered",
        "detector_call_id": "e7cf73c5-4ed7-4134-b21e-33654a91d47c",
        "guardrails_action": "block",
        "guardrails_action_id": "e7cf73c5-4ed7-4134-b21e-33654a91d47c",
        "ids_correlated": true,
        "key_id": "poc-2026-08",
        "managed_action": "blocked",
        "mode": "synchronous",
        "response_id": "chatcmpl-93785db3700dd1ac",
        "scheme": "synthid",
        "timing": {
          "validation_lag_seconds": 3.053699037991464e-05,
          "validation_latency_seconds": 0.056498174992157146
        },
        "validation_id": "e7cf73c5-4ed7-4134-b21e-33654a91d47c",
        "verdict": true
      },
      {
        "attempts": 1,
        "content_digest": "5b698665b970af241caa53378495b912d785be3d58f37985fdee867b4753ab06",
        "delivery_outcome": "delivered",
        "detector_call_id": "b96ca381-dc1d-464d-afc6-d7c90be0f01f",
        "guardrails_action": "block",
        "guardrails_action_id": "b96ca381-dc1d-464d-afc6-d7c90be0f01f",
        "ids_correlated": true,
        "key_id": "poc-2026-08",
        "managed_action": "blocked",
        "mode": "synchronous",
        "response_id": "chatcmpl-8f3e09f4a3576688",
        "scheme": "synthid",
        "timing": {
          "validation_lag_seconds": 3.25550208799541e-05,
          "validation_latency_seconds": 0.06374847798724659
        },
        "validation_id": "b96ca381-dc1d-464d-afc6-d7c90be0f01f",
        "verdict": true
      },
      {
        "attempts": 1,
        "content_digest": "614c8d731c9b52a2ac4f20f6ca28956cf1a87873f570212b6f6d984a4d80ed78",
        "delivery_outcome": "delivered",
        "detector_call_id": "1dd17656-f875-4fe2-bddf-03e276456064",
        "guardrails_action": "block",
        "guardrails_action_id": "1dd17656-f875-4fe2-bddf-03e276456064",
        "ids_correlated": true,
        "key_id": "poc-2026-08",
        "managed_action": "blocked",
        "mode": "synchronous",
        "response_id": "chatcmpl-b3bc35d51d612365",
        "scheme": "synthid",
        "timing": {
          "validation_lag_seconds": 3.436399856582284e-05,
          "validation_latency_seconds": 0.03591733000939712
        },
        "validation_id": "1dd17656-f875-4fe2-bddf-03e276456064",
        "verdict": true
      },
      {
        "attempts": 1,
        "content_digest": "614c8d731c9b52a2ac4f20f6ca28956cf1a87873f570212b6f6d984a4d80ed78",
        "delivery_outcome": "delivered",
        "detector_call_id": "c8e73849-b757-49c7-9e6e-b125a05214ab",
        "guardrails_action": "block",
        "guardrails_action_id": "c8e73849-b757-49c7-9e6e-b125a05214ab",
        "ids_correlated": true,
        "key_id": "poc-2026-08",
        "managed_action": "blocked",
        "mode": "synchronous",
        "response_id": "chatcmpl-966289ac6bf03f39",
        "scheme": "synthid",
        "timing": {
          "validation_lag_seconds": 4.105299012735486e-05,
          "validation_latency_seconds": 0.03455893599311821
        },
        "validation_id": "c8e73849-b757-49c7-9e6e-b125a05214ab",
        "verdict": true
      },
      {
        "attempts": 1,
        "content_digest": "ce162f8d4e11b56532a77bcb93e8c071cac6a03b05ee667c3c7bf8ad53a37205",
        "delivery_outcome": "delivered",
        "detector_call_id": "634deb4b-01fb-4a03-83bf-1c25448d7bbf",
        "guardrails_action": "pass",
        "guardrails_action_id": "634deb4b-01fb-4a03-83bf-1c25448d7bbf",
        "ids_correlated": true,
        "key_id": "poc-2026-08",
        "managed_action": "success",
        "mode": "synchronous",
        "response_id": "chatcmpl-bc3d1002be82da35",
        "scheme": "kgw",
        "timing": {
          "validation_lag_seconds": 3.2881012884899974e-05,
          "validation_latency_seconds": 0.4475540579878725
        },
        "validation_id": "634deb4b-01fb-4a03-83bf-1c25448d7bbf",
        "verdict": false
      },
      {
        "attempts": 1,
        "content_digest": "0c6c85a4bd814ea9a46695c8b9839c0c0ae2abaece6a4f4b0b09dffef49031b9",
        "delivery_outcome": "delivered",
        "detector_call_id": "477fcbd9-1a75-466c-9f74-9331a88891bb",
        "guardrails_action": "pass",
        "guardrails_action_id": "477fcbd9-1a75-466c-9f74-9331a88891bb",
        "ids_correlated": true,
        "key_id": "poc-2026-08",
        "managed_action": "success",
        "mode": "synchronous",
        "response_id": "chatcmpl-8c005cd87a71824b",
        "scheme": "kgw",
        "timing": {
          "validation_lag_seconds": 3.491601091809571e-05,
          "validation_latency_seconds": 0.5640277679776773
        },
        "validation_id": "477fcbd9-1a75-466c-9f74-9331a88891bb",
        "verdict": false
      },
      {
        "attempts": 1,
        "content_digest": "d9ea2aeced18f3eafe9e559e019c40a8f91b14a52f67fa86e7b5170bb3c94912",
        "delivery_outcome": "delivered",
        "detector_call_id": "ad261a60-928e-4fe0-875d-60c792044de0",
        "guardrails_action": "pass",
        "guardrails_action_id": "ad261a60-928e-4fe0-875d-60c792044de0",
        "ids_correlated": true,
        "key_id": "poc-2026-08",
        "managed_action": "success",
        "mode": "synchronous",
        "response_id": "chatcmpl-8338d09af427fdd9",
        "scheme": "kgw",
        "timing": {
          "validation_lag_seconds": 3.134901635348797e-05,
          "validation_latency_seconds": 0.7855058259738144
        },
        "validation_id": "ad261a60-928e-4fe0-875d-60c792044de0",
        "verdict": false
      },
      {
        "attempts": 1,
        "content_digest": "23445d3259bef664f64438319253112ace0c72b3ce05acf3df0c83518df5f760",
        "delivery_outcome": "delivered",
        "detector_call_id": "dbd56d58-bb86-48cc-8091-9fe49db9a967",
        "guardrails_action": "pass",
        "guardrails_action_id": "dbd56d58-bb86-48cc-8091-9fe49db9a967",
        "ids_correlated": true,
        "key_id": "poc-2026-08",
        "managed_action": "success",
        "mode": "synchronous",
        "response_id": "chatcmpl-8ca19f357c8b534d",
        "scheme": "kgw",
        "timing": {
          "validation_lag_seconds": 3.689402365125716e-05,
          "validation_latency_seconds": 0.47166678198846057
        },
        "validation_id": "dbd56d58-bb86-48cc-8091-9fe49db9a967",
        "verdict": false
      },
      {
        "attempts": 1,
        "content_digest": "74e19126bc9beb59430cddb6378e916cbffbaa7ff42223994bf75c497cfd6097",
        "delivery_outcome": "delivered",
        "detector_call_id": "e8a556d9-39a3-423e-8789-6e12307db6f8",
        "guardrails_action": "pass",
        "guardrails_action_id": "e8a556d9-39a3-423e-8789-6e12307db6f8",
        "ids_correlated": true,
        "key_id": "poc-2026-08",
        "managed_action": "success",
        "mode": "synchronous",
        "response_id": "chatcmpl-b513e49df1de810f",
        "scheme": "kgw",
        "timing": {
          "validation_lag_seconds": 3.242099774070084e-05,
          "validation_latency_seconds": 0.5764373360143509
        },
        "validation_id": "e8a556d9-39a3-423e-8789-6e12307db6f8",
        "verdict": false
      },
      {
        "attempts": 1,
        "content_digest": "451e0961f4ccd69558f825f9486e099cfb2dff412e17d43d9338d79bce7cd943",
        "delivery_outcome": "delivered",
        "detector_call_id": "96065703-b4e7-48f8-ba1f-b11f921a6eef",
        "guardrails_action": "pass",
        "guardrails_action_id": "96065703-b4e7-48f8-ba1f-b11f921a6eef",
        "ids_correlated": true,
        "key_id": "poc-2026-08",
        "managed_action": "success",
        "mode": "synchronous",
        "response_id": "chatcmpl-bc45b471038ec474",
        "scheme": "synthid",
        "timing": {
          "validation_lag_seconds": 4.3742009438574314e-05,
          "validation_latency_seconds": 0.04197403401485644
        },
        "validation_id": "96065703-b4e7-48f8-ba1f-b11f921a6eef",
        "verdict": false
      },
      {
        "attempts": 1,
        "content_digest": "1311a913bddd5112b23d7b7a91e7e064361cb1ec9753c79afecbd0f7ae85ba54",
        "delivery_outcome": "delivered",
        "detector_call_id": "c153ea05-3678-4841-9308-71fc659199ab",
        "guardrails_action": "pass",
        "guardrails_action_id": "c153ea05-3678-4841-9308-71fc659199ab",
        "ids_correlated": true,
        "key_id": "poc-2026-08",
        "managed_action": "success",
        "mode": "synchronous",
        "response_id": "chatcmpl-b04799346145410f",
        "scheme": "synthid",
        "timing": {
          "validation_lag_seconds": 3.8614001823589206e-05,
          "validation_latency_seconds": 0.04495404500630684
        },
        "validation_id": "c153ea05-3678-4841-9308-71fc659199ab",
        "verdict": false
      },
      {
        "attempts": 1,
        "content_digest": "bf6a556e310382886814e27a5e83a29e4c985182ca5dfde387af521b28673534",
        "delivery_outcome": "delivered",
        "detector_call_id": "ce148dcd-fcb2-4fb2-8305-798f4f298359",
        "guardrails_action": "pass",
        "guardrails_action_id": "ce148dcd-fcb2-4fb2-8305-798f4f298359",
        "ids_correlated": true,
        "key_id": "poc-2026-08",
        "managed_action": "success",
        "mode": "synchronous",
        "response_id": "chatcmpl-abef96511cbefca5",
        "scheme": "synthid",
        "timing": {
          "validation_lag_seconds": 3.302100230939686e-05,
          "validation_latency_seconds": 0.04250578899518587
        },
        "validation_id": "ce148dcd-fcb2-4fb2-8305-798f4f298359",
        "verdict": false
      },
      {
        "attempts": 1,
        "content_digest": "9cb083be210fe7e506f318b87f1f2868207fe1dc07618900682901a62375400e",
        "delivery_outcome": "delivered",
        "detector_call_id": "950a25d8-9a70-4080-9180-1df8326eb1ce",
        "guardrails_action": "pass",
        "guardrails_action_id": "950a25d8-9a70-4080-9180-1df8326eb1ce",
        "ids_correlated": true,
        "key_id": "poc-2026-08",
        "managed_action": "success",
        "mode": "synchronous",
        "response_id": "chatcmpl-81a7ee44bd1b24e0",
        "scheme": "synthid",
        "timing": {
          "validation_lag_seconds": 3.344498691149056e-05,
          "validation_latency_seconds": 0.04378015402471647
        },
        "validation_id": "950a25d8-9a70-4080-9180-1df8326eb1ce",
        "verdict": false
      },
      {
        "attempts": 1,
        "content_digest": "a26835a32bbae0220dba71ab0ee1f88e90d592e02cbfe7c7a90177fcabe55058",
        "delivery_outcome": "delivered",
        "detector_call_id": "0d29d6ad-edd2-4e3c-9d7e-b7ac9034ab65",
        "guardrails_action": "pass",
        "guardrails_action_id": "0d29d6ad-edd2-4e3c-9d7e-b7ac9034ab65",
        "ids_correlated": true,
        "key_id": "poc-2026-08",
        "managed_action": "success",
        "mode": "synchronous",
        "response_id": "chatcmpl-ab00b4ba159f7a31",
        "scheme": "synthid",
        "timing": {
          "validation_lag_seconds": 3.23719868902117e-05,
          "validation_latency_seconds": 0.04229474702151492
        },
        "validation_id": "0d29d6ad-edd2-4e3c-9d7e-b7ac9034ab65",
        "verdict": false
      }
    ],
    "responses": 100,
    "selected": 20,
    "terminal": 20,
    "validation_lag": {
      "count": 20,
      "p50_seconds": 3.295100759714842e-05,
      "p95_seconds": 4.118744109291583e-05,
      "p99_seconds": 4.3231095769442615e-05
    },
    "validation_latency": {
      "count": 20,
      "p50_seconds": 0.25565126798755955,
      "p95_seconds": 0.8954952916043113,
      "p99_seconds": 1.0328863071353405
    }
  },
  "observability": {
    "marker_count": 6,
    "required_metric_count": 10
  },
  "passed": true,
  "policy_semantics": {
    "gateway_positive_delivery": "flag",
    "managed_guardrails_positive_action": "block",
    "mode": "synchronous",
    "validation_failure": "closed"
  },
  "queue": {
    "overflow_policy": "non_blocking",
    "peak_depth": 2,
    "queue_depth": 0,
    "queue_overflow": 1,
    "terminal_records": 3,
    "validated_records": 2
  },
  "unsampled_baseline": {
    "client_delivery_latency": {
      "count": 4,
      "p50_seconds": 0.4976473429996986,
      "p95_seconds": 1.5045385892342895,
      "p99_seconds": 1.639969601031917
    },
    "counters": {
      "cancelled": 0,
      "clean": 0,
      "completed": 4,
      "detector_attempts": 0,
      "dropped": 0,
      "errors": 0,
      "failed": 0,
      "guardrails_attempts": 0,
      "queue_overflow": 0,
      "retries": 0,
      "selected": 0,
      "started": 4,
      "terminal": 0,
      "unsampled": 4,
      "watermarked": 0
    },
    "generation_completion_latency": {
      "count": 4,
      "p50_seconds": 0.4945689529849915,
      "p95_seconds": 1.5013386756865659,
      "p99_seconds": 1.636763617526449
    },
    "latency_samples": {
      "client_delivery": 4,
      "generation_completion": 4,
      "validation": 0,
      "validation_lag": 0
    },
    "queue_depth": 0,
    "responses": 4,
    "sample_every": 5,
    "selected": 0
  }
}
```

### Cleanup and supersession boundary

```text
D10_MODE_HARNESS result=passed secret_leaks=0
D10_MODE_CLEANUP controls=off gateway_ready=1 gpu_zero=1 restore_apply_rc=0 restore_rollout_rc=0 scale_down_rc=0
D10_MODE_POSTFLIGHT {"detector_image_ids":["sha256:2bb606b8e08b039b28f4b22356840f4cb0855d05fa688da4f4b70d26d8d0ea0f"],"detector_synthid_image_ids":["sha256:2bb606b8e08b039b28f4b22356840f4cb0855d05fa688da4f4b70d26d8d0ea0f"],"gateway_image_ids":["sha256:9443345c272c55a6dcea41bae84f8e04bab5d0f27e0934df2e4b4ff11b37fed6"],"nemo_server_image_ids":["sha256:22125dbbd05d1cfa7af931fdeca9da72c6d267a05c33a1f757daf3095ddcca7c"],"services_ready":true,"controls":"off","gateway_ready":1,"gpu":"0/0/0"}
```

**Claim (`EXECUTED`; source: the report and cleanup above):** the rerun again
selected `20/20` at `N=1` and `20/100` at `N=5`; both arrays contain exactly
20 mode-bearing hash-only records; faults, overflow, metrics, correlation, counters,
and corrected gateway-response-ready latency semantics passed; six report markers
had zero matches; controls returned off; and the GPU returned to `0/0/0`.

**Scope (`EXECUTED`/`OPEN`; source: this record and the preceding D10 records):**
this section supersedes only the missing-`mode` projection in the immediately
preceding current-build report. The earlier actual detector-outage and eight-surface
finite scan remain evidence for their recorded detector revision and were not
repeated. All explicitly listed D10 production boundaries remain `OPEN`.

## 2026-08-10 — fuzz/stress harness hardening and resource-budget evidence

**Scope:** local CPU execution only. No model server, generated text, deployment
key, or GPU was used. The smoke matrices below are harness checks, not estimates of
cluster failure rates, detector population rates, or production performance.

**Environment (`EXECUTED`; raw output):**

```text
python3 -c 'import platform, torch, transformers; print("python=" + platform.python_version()); print("torch=" + torch.__version__); print("transformers=" + transformers.__version__)'
python=3.14.4
torch=2.9.1+cu128
transformers=4.57.6
```

The first diagnostic full-suite run exposed seven test-fixture failures: scalar
maxima had been combined with default values that correctly violated the new
cross-parameter budgets. Its raw summary was
`7 failed, 487 passed, 561 warnings in 504.07s (0:08:24)`. The fixtures were
changed to exercise each scalar maximum inside a compatible cross-budget; no
runtime ceiling was weakened. Focused and final raw results:

```text
python3 -m pytest -q tests/test_synthid_processor_static.py::test_init_accepts_compatible_synthid_generation_boundaries detector/tests/test_service.py::TestHealthReady::test_vocab_size_fallback_maximum_is_accepted detector/tests/test_service.py::TestStartupConfigurationValidation::test_numeric_setting_maximum_is_accepted_by_load_settings detector/tests/test_service.py::TestStartupConfigurationValidation::test_numeric_setting_maximum_reaches_ready_lifespan
....................                                                     [100%]
20 passed, 66 warnings in 10.17s

python3 -m pytest -q --disable-warnings
........................................................................ [ 14%]
........................................................................ [ 29%]
........................................................................ [ 43%]
........................................................................ [ 58%]
........................................................................ [ 72%]
........................................................................ [ 87%]
..............................................................           [100%]
494 passed, 561 warnings in 391.84s (0:06:31)
```

The warnings are third-party Python 3.14 deprecations from FastAPI and its
runtime dependencies; the unsuppressed diagnostic run identified no repository
warning category.

**Bounded harness smokes (`EXECUTED`; exact commands and raw compact output):**

```text
set -o pipefail; PYTHONPATH=src python3 benchmarks/fuzz_watermark.py --seed 123 --kgw-equivalence-cases 3 --kgw-invariant-cases 5 --synthid-equivalence-cases 3 --detector-cases 10 --profile-iterations 1 --profile-warmup 0 --kgw-profile-vocab 32 --synthid-process-profile 32:2 --synthid-detect-profile 32:8:2 | jq -c '{status,aggregate}'
{"status":"passed","aggregate":{"elapsed_seconds":0.11704231599287596,"expected_errors":{"kgw_short_input":1,"synthid_short_input":1},"failure_rate":0.0,"failure_rate_wilson_95":[0.0,0.154639018924847],"failures":0,"latency_note":"omitted at aggregate level; campaign reservoirs are not count-weighted","throughput_cases_per_second":179.4222869041502,"total_cases":21}}

set -o pipefail; PYTHONPATH=src python3 benchmarks/stress_detection.py --output - --compact --max-cells 4 --repeats 1 --lengths 1,8 --vocab-sizes 32 --patterns uniform_random --timeout-seconds 30 | jq -c '{matrix_cells,summary}'
{"matrix_cells":4,"summary":{"attempt_timeouts":0,"attempts_completed":4,"attempts_requested":4,"attempts_started":4,"cells":4,"cells_with_unexpected_outcomes":0,"contract_successes":4,"expected_too_short_errors":2,"incomplete_attempts":0,"nonfinite_results":0,"peak_rss_bytes_max":682188800,"protocol_errors":0,"rates_wilson95":{"cell_timeout":{"count":0,"denominator":4,"lower":0.0,"rate":0.0,"upper":0.4898908364545973},"contract_success":{"count":4,"denominator":4,"lower":0.5101091635454027,"rate":1.0,"upper":1.0},"expected_too_short":{"count":2,"denominator":4,"lower":0.15003898915214947,"rate":0.5,"upper":0.8499610108478506},"nonfinite":{"count":0,"denominator":4,"lower":0.0,"rate":0.0,"upper":0.4898908364545973},"scoring_success":{"count":2,"denominator":4,"lower":0.15003898915214947,"rate":0.5,"upper":0.8499610108478506},"technical_failure_attempt":{"count":0,"denominator":4,"lower":0.0,"rate":0.0,"upper":0.4898908364545973},"timeout":{"count":0,"denominator":4,"lower":0.0,"rate":0.0,"upper":0.4898908364545973}},"scoring_successes":2,"technical_failure_attempts":0,"technical_failures":0,"timeouts":0,"unexpected_outcomes":0,"unexpected_successes":0,"worker_failures":0}}

set -o pipefail; python3 benchmarks/stress_gateway.py --requests 20 --concurrency 1,4 --sample-every 1,5,257 | jq -c '{passed,aggregate}'
{"passed":true,"aggregate":{"attempts":120,"invariant_failures":0,"invariants_passed":true,"peak_rss_bytes":50397184,"request_failure_rate":{"events":0,"rate":0.0,"total":120,"wilson_95":[0.0,0.031019166418703486]},"request_failures":0,"wall_seconds":0.11156903099617921}}
```

**Static checks and billable-resource cleanup (`EXECUTED`; raw output):**

```text
python3 -m compileall -q benchmarks src tests validation detector && git diff --check && echo 'compileall=passed diff_check=passed'
compileall=passed diff_check=passed

yamllint deploy/phase4/20-watermark-vllm-servingruntime.yaml && echo 'yamllint=passed'
yamllint=passed

python3 scripts/check-doc-links.py --external README.md EXPERIMENTS.md docs/*.md docs/*.html
clean: 15 documents, 333 local references, 46 external URLs (network checks requested)

KUBECONFIG=cluster/auth/kubeconfig oc -n openshift-machine-api get machineset -o json | jq -r '[.items[] | select(.spec.template.metadata.labels["node-role.kubernetes.io/gpu"] == "")] as $gpu | if ($gpu | length) == 1 then ($gpu[0] | "gpu_machineset=1 replicas=\(.spec.replicas // 0)/\(.status.replicas // 0)/\(.status.readyReplicas // 0)/\(.status.availableReplicas // 0)") else error("expected exactly one GPU MachineSet") end'
gpu_machineset=1 replicas=0/0/0/0
```

**Resource guards (`STATIC`, source: current code; regression behavior covered by
the passing suite above):** direct detector inputs are capped at 32 texts,
1,048,576 characters per text, 4,194,304 aggregate characters, 131,072 tokens per
text, and 262,144 aggregate tokens before scoring. KGW green-list allocations are
capped at 4 MiB; generation cache configuration is capped at 128 entries and a
64 MiB cross-budget; detector cache capacity derives from a conservative 64 MiB
budget. SynthID sampling tables are capped at 16 MiB each with four CPU entries
and two entries per device; three simultaneous vocab-by-depth int64 matrices are
capped at 192 MiB; context history is capped at 16,384 tokens per row. The
151,936-vocabulary/depth-30 deployment configuration remains accepted. These are
deployment resource constraints, not algorithmic compatibility or production
capacity claims. The changed image has not been rebuilt or rerun on OpenShift;
that deployment validation remains `OPEN`.

## 2026-08-11 — Local re-verification harness: registered detection claims reproduce from committed corpora (EXECUTED)

Adds `scripts/verify-claims.py` and `docs/compliance-map.md`. Purpose: make the
registered detection evidence *independently re-derivable* — detection is a pure
function of (text, tokenizer, key), the corpora behind the registered scheme
comparison v2 table are committed, so every registered TPR/FPR/mean-z cell must
reproduce bit-for-bit on local CPU with the detection key. It does.

Environment: local workstation, Python 3.14.4, torch 2.9.1+cu128 (CPU),
transformers 4.57.6 (same as B21). Key sourced from gitignored
`cluster/watermark-key.env` (never printed). Tokenizer from local HF cache.

### Recompute (exact registered invocation, fresh run)

```
$ set -a && . cluster/watermark-key.env && set +a && python3 benchmarks/compare_schemes.py \
    --kgw-corpus benchmarks/data/corpus_kgw512_fixed.jsonl \
    --synthid-corpus benchmarks/data/corpus_synthid512.jsonl \
    --unwatermarked-corpus benchmarks/data/corpus_wm_off.jsonl \
    --human-corpus benchmarks/data/human_corpus_512.jsonl \
    --model-tokenizer Qwen/Qwen2.5-0.5B-Instruct \
    --key-id poc-2026-08 \
    --out <scratch>/scheme_reverify.md
vllm_watermark import: sys.path fallback -> /home/anaeem/vllm-watermark/src (...)
Scoring kgw corpus <- benchmarks/data/corpus_kgw512_fixed.jsonl (KGW detector)
Scoring synthid corpus <- benchmarks/data/corpus_synthid512.jsonl (SynthID mean detector)
Scoring unwatermarked corpus <- benchmarks/data/corpus_wm_off.jsonl (FPR, kgw det)
Scoring unwatermarked corpus <- benchmarks/data/corpus_wm_off.jsonl (FPR, synthid det)
Scoring human corpus <- benchmarks/data/human_corpus_512.jsonl (FPR, kgw det)
Scoring human corpus <- benchmarks/data/human_corpus_512.jsonl (FPR, synthid det)
wrote <scratch>/scheme_reverify.md
wrote <scratch>/scheme_reverify.json
```

(~25 min wall clock, CPU-bound in KGW greenlist derivation.)

### Assertion against the registered table

```
$ set -a && . cluster/watermark-key.env && set +a && python3 scripts/verify-claims.py --json <scratch>/scheme_reverify.json
== 1. Corpus integrity (sha256 vs pinned values) ==
  [PASS] benchmarks/data/corpus_kgw512_fixed.jsonl
  [PASS] benchmarks/data/corpus_synthid512.jsonl
  [PASS] benchmarks/data/corpus_wm_off.jsonl
  [PASS] benchmarks/data/human_corpus_512.jsonl

== 2. Registered-claim reproduction (vs EXPERIMENTS.md scheme comparison v2) ==
  [PASS] kgw                          @200  n=120 mean_z=9.084 rate=0.9916666666666667
  [PASS] kgw                          @256  n=116 mean_z=10.261 rate=1.0
  [PASS] kgw                          @512  n=76 mean_z=14.317 rate=1.0
  [PASS] synthid                      @200  n=120 mean_z=13.744 rate=1.0
  [PASS] synthid                      @256  n=117 mean_z=15.651 rate=1.0
  [PASS] synthid                      @512  n=96 mean_z=22.886 rate=1.0
  [PASS] unwatermarked (kgw det)      @200  n=119 mean_z=-0.148 rate=0.0
  [PASS] unwatermarked (kgw det)      @256  n=115 mean_z=-0.068 rate=0.0
  [PASS] unwatermarked (kgw det)      @512  n=0 mean_z=None rate=None
  [PASS] unwatermarked (synthid det)  @200  n=119 mean_z=-0.088 rate=0.0
  [PASS] unwatermarked (synthid det)  @256  n=115 mean_z=-0.06 rate=0.0
  [PASS] unwatermarked (synthid det)  @512  n=0 mean_z=None rate=None
  [PASS] human (kgw det)              @200  n=150 mean_z=0.027 rate=0.0
  [PASS] human (kgw det)              @256  n=150 mean_z=0.084 rate=0.0
  [PASS] human (kgw det)              @512  n=150 mean_z=0.099 rate=0.0
  [PASS] human (synthid det)          @200  n=150 mean_z=-0.087 rate=0.0
  [PASS] human (synthid det)          @256  n=150 mean_z=-0.07 rate=0.0
  [PASS] human (synthid det)          @512  n=150 mean_z=-0.098 rate=0.0

ALL CHECKS PASSED (18/18 cells; source table: EXPERIMENTS.md 'Scheme comparison v2', 2026-08-08)
```

Corpus pins (sha256, also embedded in the script):
```
e65bef5b8c3eca9e36434150a238205c95ca65db51fb6d67a3ef66e6861a7c37  benchmarks/data/corpus_kgw512_fixed.jsonl
54788b69476e5176707e18c4b27dc0c33544fdc2777dd1073ca0fbcd0bb45b69  benchmarks/data/corpus_synthid512.jsonl
107ff38689f9904c877fc3eb01df56a946f6cb780a1a74ef34f2eeba7a61232e  benchmarks/data/corpus_wm_off.jsonl
fbd9e945129d3ec2c504914d571a855d1cde98384c2acc1ce9599b1fc5d769b6  benchmarks/data/human_corpus_512.jsonl
```

### Demo mode (stakeholder-facing, seconds)

```
$ set -a && . cluster/watermark-key.env && set +a && python3 scripts/verify-claims.py --demo
KGW detector demo — threshold z >= 4.0, key_id=poc-2026-08
sample                                       tokens        z  verdict
watermarked (corpus_kgw512_fixed row 0)         512   12.817  WATERMARK DETECTED   [text elided sha256:64b551969b24]
human (human_corpus_512 row 0)                  512   -0.935  not detected   [text elided sha256:510da11ac1b0]
```

Note: the elided watermarked-sample digest `64b551969b24` is the same digest
recorded in the NeMo PoC evidence transcript above (`http_checks5.py`, kgw
sample) — the demo demonstrably runs on the registered evidence corpus.

Scope: this reproduces *detection scoring* from committed corpora. It does not
re-execute generation (`vllm serve`), which remains cluster evidence (D1/D8).
Third-party reproduction requires the detection key by construction (keyed
watermarking); detector-access conditions remain A14 (`OPEN`, counsel).


## 2026-09-16 — Native Gumbel-max watermarking end-to-end on a fresh cluster (EXECUTED; redacted)

**Purpose.** First execution of vLLM's native text watermarking (upstream PR #54053 merged
2026-09-10 + #56122 merged 2026-09-12; not in any released vLLM as of this date) on a freshly
provisioned OpenShift cluster, as the evidence base for
[`docs/guide-native-watermarking.md`](docs/guide-native-watermarking.md). Everything below ran
from this workstation against the cluster; secrets (AWS credentials, kubeadmin password, the
watermark key) are redacted as `<redacted>`, the sandbox base domain as
`<redacted-sandbox-domain>` and Route hosts as `<route>`. Assets: `deploy/native/*.yaml`,
`scripts/native-watermark-up.sh`, `scripts/native-watermark-demo.py`.

**Environment.** New AWS sandbox (base domain `<redacted-sandbox-domain>`); cluster `ocp-ai`
created with `openshift-install create cluster --dir cluster` (installer 4.20.27, region
us-east-1, 3× m6i.xlarge control plane, 2× m6i.xlarge workers; `Install complete` after
`Time elapsed: 45m43s`). GPU node added with `GPU_REPLICAS=1 ./scripts/create-gpu-machineset.sh`
(g5.xlarge, A10G); NFD + NVIDIA GPU Operator via `./scripts/install-gpu-operators.sh`.
Image: `vllm/vllm-openai@sha256:20a52b807cef7a3be15d2174d293b5642d437615b20b5930d78a927e709e6bb6`
(tag `nightly-cd10ed6f9f6b37a8ace9cf380007e66fe12ec0c3`, CUDA 13.0.2; amd64 image-manifest digest
`sha256:0a329f66a92e19ad8c8e9b9a17bda6ecf70b1a8dcfd9c735360a251ce870d0d8`, the pod's Image ID).
Model: `Qwen/Qwen2.5-1.5B-Instruct`. Key: 64-bit integer generated with `secrets.randbits(64)`
into gitignored `cluster/gumbel-key.env`, delivered as Secret `watermark-key`.

### Cluster facts (raw)

```
$ oc version
Server Version: 4.20.27
Kubernetes Version: v1.33.12
$ oc get clusterversion
NAME      VERSION   AVAILABLE   PROGRESSING   SINCE   STATUS
version   4.20.27   True        False         6m58s   Cluster version is 4.20.27
$ oc get nodes -L node.kubernetes.io/instance-type,nvidia.com/gpu.product,nvidia.com/cuda.driver-version.full,nvidia.com/cuda.runtime-version.full
NAME                           STATUS   ROLES                  AGE     VERSION    INSTANCE-TYPE   GPU.PRODUCT   CUDA.DRIVER-VERSION.FULL   CUDA.RUNTIME-VERSION.FULL
ip-10-0-150-204.ec2.internal   Ready    worker                 26m     v1.33.12   m6i.xlarge                                               
ip-10-0-21-8.ec2.internal      Ready    control-plane,master   34m     v1.33.12   m6i.xlarge                                               
ip-10-0-26-129.ec2.internal    Ready    gpu,worker             7m26s   v1.33.12   g5.xlarge       NVIDIA-A10G   595.91.07                  13.2
ip-10-0-55-253.ec2.internal    Ready    control-plane,master   33m     v1.33.12   m6i.xlarge                                               
ip-10-0-88-133.ec2.internal    Ready    control-plane,master   33m     v1.33.12   m6i.xlarge                                               
ip-10-0-98-150.ec2.internal    Ready    worker                 26m     v1.33.12   m6i.xlarge                                               
$ oc get csv -A | grep -E 'nfd|gpu'
nvidia-gpu-operator                    gpu-operator-certified.v26.7.0   NVIDIA GPU Operator               26.7.0                gpu-operator-certified.v26.3.3   Succeeded
openshift-nfd                          nfd.4.20.0-202609091330          Node Feature Discovery Operator   4.20.0-202609091330                                    Succeeded
$ oc -n nvidia-gpu-operator get pods
NAME                                           READY   STATUS      RESTARTS      AGE
gpu-feature-discovery-scjzh                    1/1     Running     0             6m30s
gpu-operator-7c7f969b75-wcrgb                  1/1     Running     0             10m
nvidia-container-toolkit-daemonset-rwrm9       1/1     Running     0             6m33s
nvidia-cuda-validator-gw4nf                    0/1     Completed   0             82s
nvidia-dcgm-exporter-9t9cf                     0/1     Running     2 (38s ago)   6m31s
nvidia-dcgm-sjfkb                              1/1     Running     0             6m31s
nvidia-device-plugin-daemonset-2mzck           1/1     Running     0             6m32s
nvidia-driver-daemonset-9.6.20260623-0-mz4mv   2/2     Running     0             6m33s
nvidia-node-status-exporter-szqn5              1/1     Running     0             6m29s
nvidia-operator-validator-8hb84                1/1     Running     0             6m32s
$ oc -n openshift-machine-api get machineset
NAME                             DESIRED   CURRENT   READY   AVAILABLE   AGE
ocp-ai-wg9fl-gpu-us-east-1a      1         1         1       1           11m
ocp-ai-wg9fl-worker-us-east-1a   0         0                             35m
ocp-ai-wg9fl-worker-us-east-1b   0         0                             35m
ocp-ai-wg9fl-worker-us-east-1c   0         0                             35m
ocp-ai-wg9fl-worker-us-east-1d   1         1         1       1           35m
ocp-ai-wg9fl-worker-us-east-1f   1         1         1       1           35m
```

### Deploy (raw)

The first `oc apply` of the manifests happened while the GPU node was still booting (to
overlap image pulls); the script run below is the recorded, idempotent path. Its rollout wait
for vLLM hit the Deployment's default 600 s progress deadline because the first start pulls
the image (8.7 GB compressed on the wire, 21.8 GB unpacked on the node; measured below) and
downloads the model; `progressDeadlineSeconds: 1800` was added to
`deploy/native/10-vllm.yaml` afterwards and re-applied (no pod restart, spec-level field).
The script's `set -e` made it exit 1 at that point, before the Route URLs were printed; the
`oc rollout status` / `oc wait` fallback added to the script afterwards has not been run on a
fresh start (see "Startup timing" below for the read-only checks that were run).

```
$ ./scripts/native-watermark-up.sh
namespace/watermark-demo unchanged
secret/watermark-key configured
deployment.apps/vllm-watermark configured
service/vllm-watermark unchanged
route.route.openshift.io/vllm-watermark unchanged
configmap/watermark-detector-src unchanged
deployment.apps/watermark-detector configured
service/watermark-detector unchanged
route.route.openshift.io/watermark-detector unchanged
Waiting for the detector (CPU) and vLLM (GPU; first start downloads the model) ...
deployment "watermark-detector" successfully rolled out
Waiting for deployment "vllm-watermark" rollout to finish: 0 of 1 updated replicas are available...
error: deployment "vllm-watermark" exceeded its progress deadline
exit=1
```

vLLM engine startup lines (`oc -n watermark-demo logs deploy/vllm-watermark`, filtered):

```
(APIServer pid=1) INFO 09-16 05:20:50 [api_utils.py:286] non-default args: {'model_tag': 'Qwen/Qwen2.5-1.5B-Instruct', 'host': '0.0.0.0', 'model': 'Qwen/Qwen2.5-1.5B-Instruct', 'max_model_len': 4096, 'gpu_memory_utilization': 0.85, 'watermark_config': '***'}
(EngineCore pid=75) INFO 09-16 05:21:38 [core.py:123] Initializing a V1 LLM engine (v0.29.1rc1.dev128+gcd10ed6f9) with config: model='Qwen/Qwen2.5-1.5B-Instruct', speculative_config=None, tokenizer='Qwen/Qwen2.5-1.5B-Instruct', sk
(EngineCore pid=75) INFO 09-16 05:21:42 [gpu_worker.py:441] Using V2 Model Runner
(EngineCore pid=75) INFO 09-16 05:22:12 [default_loader.py:430] Loading weights took 5.48 seconds
(EngineCore pid=75) INFO 09-16 05:22:46 [model_runner.py:1069] Graph capturing finished in 4 secs, took 0.30 GiB
(EngineCore pid=75) INFO 09-16 05:22:49 [watermark_sample_warmup.py:130] Warming up watermark sampler kernel (vocab=151936, keys=1, dtypes=['torch.bfloat16', 'torch.float32'], skip_mask=[True]).
(EngineCore pid=75) INFO 09-16 05:23:13 [model_runner.py:1069] Graph capturing finished in 5 secs, took 0.20 GiB
(APIServer pid=1) WARNING 09-16 05:23:17 [model.py:1772] Default vLLM sampling parameters have been overridden by the model's `generation_config.json`: `{'repetition_penalty': 1.1, 'temperature': 0.7, 'top_k': 20, 'top_p': 0.8}`. If this is not intended, please relaunch vLLM instance with `--generation-config vllm`.
(APIServer pid=1) INFO:     Waiting for application startup.
(APIServer pid=1) INFO:     Application startup complete.
```

Pulled image: `docker.io/vllm/vllm-openai@sha256:0a329f66a92e19ad8c8e9b9a17bda6ecf70b1a8dcfd9c735360a251ce870d0d8`.
Detector pod logs ended with `Uvicorn running on http://0.0.0.0:8080`; its startup probe failed
once (`connection refused`) while the vLLM import completed, then passed.

### Startup timing, image size, wait semantics (raw; captured 2026-09-16 09:07–09:12Z from the same pod, restarts=0)

Measured post hoc from the kubelet-recorded pod status and timestamped logs of the pod that
served every run above. The namespace's Events had already been garbage-collected
(`oc -n watermark-demo get events` → 0 items), so the kubelet `Pulled` event with the exact
pull duration could not be recorded; the pull is bounded below by PodScheduled → container start.

```
$ oc -n watermark-demo get pod -l app=vllm-watermark -o jsonpath='{.items[0].metadata.name}'
vllm-watermark-7dc9fb8f79-rzgxw
$ oc -n watermark-demo get pod vllm-watermark-7dc9fb8f79-rzgxw -o jsonpath='{"created="}{.metadata.creationTimestamp}{"\n"}{range .status.conditions[*]}{.type}{"="}{.status}{" @"}{.lastTransitionTime}{"\n"}{end}{"containerStarted="}{.status.containerStatuses[0].state.running.startedAt}{"\nrestarts="}{.status.containerStatuses[0].restartCount}{"\nimageID="}{.status.containerStatuses[0].imageID}{"\nnode="}{.spec.nodeName}{"\n"}'
created=2026-09-16T05:06:02Z
PodReadyToStartContainers=True @2026-09-16T05:20:02Z
Initialized=True @2026-09-16T05:15:13Z
Ready=True @2026-09-16T05:23:24Z
ContainersReady=True @2026-09-16T05:23:24Z
PodScheduled=True @2026-09-16T05:15:13Z
containerStarted=2026-09-16T05:20:01Z
restarts=0
imageID=docker.io/vllm/vllm-openai@sha256:0a329f66a92e19ad8c8e9b9a17bda6ecf70b1a8dcfd9c735360a251ce870d0d8
node=ip-10-0-26-129.ec2.internal
$ oc -n watermark-demo get deploy vllm-watermark -o jsonpath='{"progressDeadlineSeconds="}{.spec.progressDeadlineSeconds}{"\n"}{range .status.conditions[*]}{.type}{"="}{.status}{" reason="}{.reason}{" @"}{.lastTransitionTime}{"\n"}{end}'
progressDeadlineSeconds=1800
Available=True reason=MinimumReplicasAvailable @2026-09-16T05:23:24Z
Progressing=True reason=NewReplicaSetAvailable @2026-09-16T05:23:24Z
$ oc -n watermark-demo logs deploy/vllm-watermark --timestamps | grep -i 'application startup'
2026-09-16T05:23:17.406020278Z (APIServer pid=1) INFO:     Waiting for application startup.
2026-09-16T05:23:17.718321517Z (APIServer pid=1) INFO:     Application startup complete.
```

Derived: pod created → scheduled (waiting for the GPU node) 9 min 11 s; scheduled → container
started (image pull and sandbox) 4 min 48 s; container started → `Application startup complete`
3 min 17 s; → pod Ready 3 min 23 s (10 s readiness probe period); created → Ready 17 min 22 s.
The 600 s default progress deadline (clock from Deployment creation) therefore fired while the
pod was still Pending.

Image size (the manifest pins the manifest-list digest; the node stores the amd64 manifest):

```
$ skopeo inspect --raw docker://vllm/vllm-openai@sha256:20a52b807cef7a3be15d2174d293b5642d437615b20b5930d78a927e709e6bb6 | jq -r '.manifests[]|"\(.platform.os)/\(.platform.architecture) \(.digest)"'
linux/arm64 sha256:c24efc18b169ee882463b01e20d17581f5ea3d953050eb9fb6a041711da0b88e
linux/amd64 sha256:0a329f66a92e19ad8c8e9b9a17bda6ecf70b1a8dcfd9c735360a251ce870d0d8
$ skopeo inspect --raw docker://vllm/vllm-openai@sha256:0a329f66a92e19ad8c8e9b9a17bda6ecf70b1a8dcfd9c735360a251ce870d0d8 | jq '{layers:(.layers|length), compressed_bytes:([.layers[].size]|add)}'
{"layers": 37, "compressed_bytes": 8708612941}
$ oc get node ip-10-0-26-129.ec2.internal -o json | python3 -c 'import json,sys; n=json.load(sys.stdin); [print(i["sizeBytes"], [x for x in i["names"] if "vllm-openai" in x]) for i in n["status"]["images"] if any("vllm-openai" in x for x in i["names"])]'
21776624330 ['docker.io/vllm/vllm-openai@sha256:0a329f66a92e19ad8c8e9b9a17bda6ecf70b1a8dcfd9c735360a251ce870d0d8', 'docker.io/vllm/vllm-openai@sha256:20a52b807cef7a3be15d2174d293b5642d437615b20b5930d78a927e709e6bb6']
```

So 8.71 GB compressed over the wire, 21.78 GB unpacked on the node.

CUDA toolkit of the image, from the image config (captured 2026-09-16 11:29Z; the source of
the "CUDA 13.0.2" figure in the Environment paragraph and fact E3, and of the image's declared
driver requirement that the guide's section 2 cites; the long `NVIDIA_REQUIRE_CUDA` brand list
is truncated after its first clauses):

```
$ skopeo inspect --config docker://vllm/vllm-openai@sha256:0a329f66a92e19ad8c8e9b9a17bda6ecf70b1a8dcfd9c735360a251ce870d0d8 | jq -r '.config.Env[]' | grep -E '^(CUDA_VERSION|NV_CUDA_CUDART_VERSION|NVIDIA_REQUIRE_CUDA)='
NVIDIA_REQUIRE_CUDA=cuda>=13.0 brand=unknown,driver>=535,driver<536 brand=grid,driver>=535,driver<536 brand=tesla,driver>=535,driver<536 …
NV_CUDA_CUDART_VERSION=13.0.96-1
CUDA_VERSION=13.0.2
```

Wait semantics (read-only, against the now-Ready Deployment; the guide's statement that
`oc rollout status` fails immediately while the Progressing reason is ProgressDeadlineExceeded
is STATIC from kubectl `pkg/polymorphichelpers/rollout_status.go` L75-79 and kubernetes
`pkg/controller/deployment/util/deployment_util.go` `DeploymentTimedOut`, release-1.33, not
re-executed here):

```
$ oc version --client | head -1
Client Version: 4.19.0-202511102034.p2.gb61226d.assembly.stream.el9-b61226d
$ oc -n watermark-demo wait --for=condition=Available deployment/vllm-watermark --timeout=10s; echo "exit=$?"
deployment.apps/vllm-watermark condition met
exit=0
$ oc -n watermark-demo rollout status deployment/vllm-watermark --timeout=10s; echo "exit=$?"
deployment "vllm-watermark" successfully rolled out
exit=0
$ oc -n watermark-demo wait --for=condition=Bogus deployment/vllm-watermark --timeout=3s; echo "exit=$?"   # 2026-09-16 11:27Z: message form of the script's fallback wait when it expires (a condition the Deployment does not carry, so it must time out)
error: timed out waiting for the condition on deployments/vllm-watermark
exit=1
$ oc diff -f deploy/native/10-vllm.yaml >/dev/null; echo "exit=$?"; oc diff -f deploy/native/20-detector.yaml >/dev/null; echo "exit=$?"
exit=0
exit=0
```

(`oc diff` exit 0 = the live objects match the manifests; a re-run of the script would change
nothing.) Local check backing the guide's KUBECONFIG note (`~/.kube/config` exists on this
workstation):

```
$ KUBECONFIG=/tmp/claude-1000/does-not-exist/kubeconfig oc get pods; echo "exit=$?"
error: Missing or incomplete configuration info.  Please point to an existing, complete config file:
exit=1
$ KUBECONFIG=/tmp/claude-1000/does-not-exist/kubeconfig bash scripts/native-watermark-up.sh; echo "exit=$?"   # 2026-09-16 11:29Z: the deploy script pre-checks the file and never reaches oc
KUBECONFIG=/tmp/claude-1000/does-not-exist/kubeconfig not found; export KUBECONFIG=<your kubeconfig> (oc does not fall back to ~/.kube/config)
exit=2
```

ConfigMap detector script vs the upstream example at the pinned commit:

```
$ python3 - <<'EOF'
import yaml,subprocess,base64
cm=list(yaml.safe_load_all(open('deploy/native/20-detector.yaml')))[0]
local=cm['data']['watermark_detection_server.py']
up=base64.b64decode(subprocess.run(['gh','api','repos/vllm-project/vllm/contents/examples/basic/online_serving/watermark_detection_server.py?ref=cd10ed6f9f6b37a8ace9cf380007e66fe12ec0c3','--jq','.content'],capture_output=True,text=True).stdout).decode()
print('identical:', local==up, 'local_lines:', local.count('\n'), 'upstream_lines:', up.count('\n'))
EOF
identical: True local_lines: 70 upstream_lines: 70
```

The same comparison with `diff` and `cmp` (2026-09-16 11:29Z; the guide's "`diff` and `cmp` …
no differences" wording refers to this run), plus the live ConfigMap's copy:

```
$ python3 -c 'import yaml; cm=list(yaml.safe_load_all(open("deploy/native/20-detector.yaml")))[0]; open("local_det.py","w").write(cm["data"]["watermark_detection_server.py"])'
$ gh api "repos/vllm-project/vllm/contents/examples/basic/online_serving/watermark_detection_server.py?ref=cd10ed6f9f6b37a8ace9cf380007e66fe12ec0c3" --jq .content | base64 -d > up_det.py
$ diff local_det.py up_det.py; echo "exit=$?"
exit=0
$ cmp local_det.py up_det.py; echo "exit=$?"
exit=0
$ wc -l local_det.py up_det.py
  70 local_det.py
  70 up_det.py
 140 total
$ sha256sum local_det.py up_det.py
c81cc8e72555aee527dd40a4ec88db48ef547d40bf29c4d7ee3c0a6e386f338e  local_det.py
c81cc8e72555aee527dd40a4ec88db48ef547d40bf29c4d7ee3c0a6e386f338e  up_det.py
$ oc -n watermark-demo get configmap watermark-detector-src -o jsonpath='{.data.watermark_detection_server\.py}' | sha256sum
c81cc8e72555aee527dd40a4ec88db48ef547d40bf29c4d7ee3c0a6e386f338e  -
```

Checks on the script edits made during this verification pass (deploy script: `KUBECONFIG`
existence check, `oc rollout status` → `oc wait --for=condition=Available` fallback with
diagnostics, vLLM timeout 35m; demo script: fail-fast on missing `--prompts`/`--human-jsonl`/
`--human-text`). Not a fresh-start run of the deploy script:

```
$ bash -n scripts/native-watermark-up.sh && echo "bash -n ok"
bash -n ok
$ python3 -m py_compile scripts/native-watermark-demo.py && echo "py_compile ok"
py_compile ok
$ python3 scripts/native-watermark-demo.py --vllm-url x --detector-url y --human-jsonl /nonexistent.jsonl 2>&1 | tail -1
native-watermark-demo.py: error: --human-jsonl /nonexistent.jsonl: file not found (generate it with benchmarks/fetch_human_corpus.py or omit the flag)
```

Further script edits after the review of the guide (2026-09-16 11:29–11:41Z): the deploy script
now runs under `set -Eeuo pipefail` and requires the key file to have exactly one non-blank,
non-comment line, matching `^WATERMARK_KEY=[0-9]+$`; the demo script's `--seed` help says it
steers only positions sampled without the watermark. Why `-E`: without errtrace bash does not
run the ERR trap for a command that fails inside a function, so `on_fail` (the pod list and log
tail) would never print when `wait_for` fails. Local repro, GNU bash 5.3.0:

```
$ cat trap.sh
set -euo pipefail
wait_for() { false || false; }
on_fail() { echo "on_fail ran"; }
trap on_fail ERR
wait_for
echo "not reached"
$ bash trap.sh; echo "exit=$?"
exit=1
$ sed 's/set -euo pipefail/set -Eeuo pipefail/' trap.sh > trap_E.sh; bash trap_E.sh; echo "exit=$?"
on_fail ran
exit=1
```

Key-file check: the `if` block now in `scripts/native-watermark-up.sh` (extracted with `sed`,
wrapped in `set -euo pipefail; key_file="$1"` … `echo accept`) against nine synthetic files; the
previous check, `grep -qE '^WATERMARK_KEY=[0-9]+$'`, accepted `extra`, `dup` and `comment` too.
`grep` resolves to GNU grep 3.12 in a non-interactive bash on this workstation; the results were
the same under ugrep 7.8.4. The real `cluster/gumbel-key.env` is accepted (`accept`, exit 0):

```
$ printf 'WATERMARK_KEY=123\n' > one.env; printf 'WATERMARK_KEY=123' > nonewline.env; printf 'WATERMARK_KEY=123\n\n' > trailingblank.env; printf '# comment\n\nWATERMARK_KEY=123\n' > comment.env; printf 'WATERMARK_KEY=123\nEXTRA=oops\n' > extra.env; printf 'WATERMARK_KEY=1\nWATERMARK_KEY=2\n' > dup.env; printf 'FOO=bar\n' > nokey.env; : > empty.env; printf 'WATERMARK_KEY=abc\n' > nonint.env
$ sed -n '/^# oc --from-env-file/,/^fi/p' scripts/native-watermark-up.sh
# oc --from-env-file skips blank and #-comment lines; every other line must be the single key line.
if [[ $(grep -cvE '^[[:space:]]*(#|$)' "$key_file") -ne 1 ]] \
   || ! grep -qE '^WATERMARK_KEY=[0-9]+$' "$key_file"; then
  echo "$key_file must contain exactly one non-comment line: WATERMARK_KEY=<non-negative integer>" >&2
  exit 2
fi
$ for f in one nonewline trailingblank comment extra dup nokey empty nonint; do printf '%-14s ' $f; bash keycheck_real.sh $f.env 2>&1 | tr -d '\n'; echo "  exit=${PIPESTATUS[0]}"; done
one            accept  exit=0
nonewline      accept  exit=0
trailingblank  accept  exit=0
comment        accept  exit=0
extra          extra.env must contain exactly one non-comment line: WATERMARK_KEY=<non-negative integer>  exit=2
dup            dup.env must contain exactly one non-comment line: WATERMARK_KEY=<non-negative integer>  exit=2
nokey          nokey.env must contain exactly one non-comment line: WATERMARK_KEY=<non-negative integer>  exit=2
empty          empty.env must contain exactly one non-comment line: WATERMARK_KEY=<non-negative integer>  exit=2
nonint         nonint.env must contain exactly one non-comment line: WATERMARK_KEY=<non-negative integer>  exit=2
$ for f in one nonewline trailingblank comment extra dup nokey empty nonint; do printf '%-14s ' $f; if grep -qE '^WATERMARK_KEY=[0-9]+$' $f.env; then echo PASSES; else echo rejected; fi; done   # the previous check
one            PASSES
nonewline      PASSES
trailingblank  PASSES
comment        PASSES
extra          PASSES
dup            PASSES
nokey          rejected
empty          rejected
nonint         rejected
$ bash -n scripts/native-watermark-up.sh && echo "bash -n ok"
bash -n ok
$ python3 -m py_compile scripts/native-watermark-demo.py && echo "py_compile ok"
py_compile ok
$ oc diff -f deploy/native/10-vllm.yaml >/dev/null; echo "exit=$?"; oc diff -f deploy/native/20-detector.yaml >/dev/null; echo "exit=$?"   # 11:41Z, after adding comments to both manifests: still no drift
exit=0
exit=0
```

### Route TLS (raw)

A fresh cluster's Routes use the self-signed ingress CA, so `curl` and Python fail
verification until that CA is trusted. Extracting it:

```
$ oc -n openshift-ingress-operator get secret router-ca -o jsonpath='{.data.tls\.crt}' | base64 -d > cluster/router-ca.crt
$ openssl x509 -in cluster/router-ca.crt -noout -subject -issuer -dates
subject=CN=ingress-operator@1789533950
issuer=CN=ingress-operator@1789533950
notBefore=Sep 16 04:45:49 2026 GMT
notAfter=Sep 15 04:45:50 2028 GMT
$ curl -sS --cacert cluster/router-ca.crt -w '\nhttp=%{http_code}\n' $VLLM/v1/models
{"object":"list","data":[{"id":"Qwen/Qwen2.5-1.5B-Instruct","object":"model","created":1789549679,"owned_by":"vllm","root":"Qwen/Qwen2.5-1.5B-Instruct","parent":null,"max_model_len":4096,"permission":[{"id":"modelperm-afb3d832013d592d","object":"model_permission","created":1789549679,"allow_create_engine":false,"allow_sampling":true,"allow_logprobs":true,"allow_search_indices":false,"allow_view":true,"allow_fine_tuning":false,"organization":"*","group":null,"is_blocking":false}]}]}
http=200
$ curl -sS --cacert cluster/router-ca.crt $VLLM/v1/models | python3 -c 'import json,sys; print([m["id"] for m in json.load(sys.stdin)["data"]])'
['Qwen/Qwen2.5-1.5B-Instruct']
```

Correction before commit: the `/v1/models` line first written here was a hand-written paraphrase
(`{"object": "list", "models": [...]}`), not command output; the response above was re-captured
2026-09-16 09:07Z from the same pod. `created` and `permission[].id` are regenerated on every
request, so those two values differ between captures.

### 4.1 hand steps (raw)

Detector smoke with a human sentence, then one watermarked completion and one opt-out
completion for the same prompt, each scored by the detector:

```
$ curl -sS --cacert cluster/router-ca.crt -w ' http=%{http_code}\n' $DET/detect -H 'Content-Type: application/json' -d '{"text":"It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife."}'
{"score":19.542122906190592,"p_value":0.9070642274102403,"num_scored_tokens":26,"is_watermarked":false} http=200

### 4.1 generate (watermarked, default)
$ curl -s --cacert cluster/router-ca.crt $VLLM/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"Qwen/Qwen2.5-1.5B-Instruct","messages":[{"role":"user","content":"Explain how vaccines train the immune system."}],"max_tokens":300,"temperature":1.0,"top_p":1.0}' | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"])' > wm.txt
usage: {"prompt_tokens": 38, "total_tokens": 338, "completion_tokens": 300, "prompt_tokens_details": null, "completion_tokens_details": null}
$ wc -w wm.txt; head -c 500 wm.txt
228 wm.txt
Vaccines work by teaching the immune system how to recognize and attack specific viruses or bacteria without actually causing disease. Here's a simplified explanation of how this happens:

1. **Antigen Exposure**: The vaccine contains small pieces of the virus (antigens) that make up its outer coat. When an individual is vaccinated, these antigens enter their body.

2. **Recognition**: Because the antigens resemble parts of the real virus inside host cells, they trigger an immediate response fro
...
$ python3 -c 'import json; print(json.dumps({"text": open("wm.txt").read()}))' | curl -s --cacert cluster/router-ca.crt $DET/detect -H 'Content-Type: application/json' -d @-
{"score":617.807335901951,"p_value":2.704772159904535e-46,"num_scored_tokens":300,"is_watermarked":true}

### 4.1 generate (opt-out: watermarking=false)
$ curl -s --cacert cluster/router-ca.crt $VLLM/v1/chat/completions -H 'Content-Type: application/json' -d '{... same ..., "watermarking": false}' | python3 -c '...' > plain.txt
usage: {"prompt_tokens": 38, "total_tokens": 338, "completion_tokens": 300, "prompt_tokens_details": null, "completion_tokens_details": null}
$ wc -w plain.txt; head -c 300 plain.txt
230 plain.txt
Vaccines work to train the body's immune system in several ways so that if it encounters a real threat later, it can recognize and defend against it more effectively.

1. **Antigen Presentation**: When a vaccine is administered, weakened forms or whole pathogens (viruses, bacteria, etc.) of the dise
...
$ python3 -c 'import json; print(json.dumps({"text": open("plain.txt").read()}))' | curl -s --cacert cluster/router-ca.crt $DET/detect -H 'Content-Type: application/json' -d @-
{"score":317.3888859925337,"p_value":0.17178151881976203,"num_scored_tokens":301,"is_watermarked":false}
```

### 4.2 / 4.3 demo-script runs (raw)

```
### run A: T=1.0 top_p=1.0, n=10, human n=50
$ python3 scripts/native-watermark-demo.py --vllm-url $VLLM --detector-url $DET --cacert cluster/router-ca.crt --n 10 --max-tokens 300 --temperature 1.0 --top-p 1.0 --human-jsonl benchmarks/data/human_corpus.jsonl --human-n 50 --json-out native-demo-t1.json
vLLM: <route>  model: Qwen/Qwen2.5-1.5B-Instruct
detector: <route>
sampling: temperature=1.0 top_p=1.0 max_tokens=300

[1/10] Explain how vaccines train the immune system to recognize pathogens it has never encounter
  watermarked    tokens= 300  scored= 300  score=   622.52  p=2.342e-47  -> WATERMARKED
  opt-out        tokens= 300  scored= 300  score=   320.48  p=1.198e-01  -> not detected
[2/10] Explain why the night sky is dark even though the universe contains an enormous number of 
  watermarked    tokens= 300  scored= 300  score=   740.07  p=5.242e-76  -> WATERMARKED
  opt-out        tokens= 272  scored= 271  score=   250.95  p=8.904e-01  -> not detected
[3/10] Explain how compound interest can turn small, regular savings into significant wealth over
  watermarked    tokens= 300  scored= 298  score=   446.12  p=3.553e-14  -> WATERMARKED
  opt-out        tokens= 300  scored= 284  score=   309.29  p=6.974e-02  -> not detected
[4/10] Explain the difference between weather and climate, and why one unusually cold winter does
  watermarked    tokens= 300  scored= 300  score=   726.35  p=1.791e-72  -> WATERMARKED
  opt-out        tokens= 293  scored= 292  score=   288.15  p=5.820e-01  -> not detected
[5/10] Explain how a criminal trial in a common-law system moves from arrest to verdict.
  watermarked    tokens= 300  scored= 300  score=   628.54  p=1.003e-48  -> WATERMARKED
  opt-out        tokens= 300  scored= 300  score=   295.98  p=5.847e-01  -> not detected
[6/10] Explain why some metals conduct electricity so much better than others.
  watermarked    tokens= 172  scored= 170  score=   354.48  p=3.804e-28  -> WATERMARKED
  opt-out        tokens= 282  scored= 281  score=   302.62  p=1.007e-01  -> not detected
[7/10] Explain how search engines decide which web pages to show first for a given query.
  watermarked    tokens= 300  scored= 300  score=   631.73  p=1.861e-49  -> WATERMARKED
  opt-out        tokens= 290  scored= 289  score=   308.91  p=1.221e-01  -> not detected
[8/10] Explain why yeast makes bread dough rise and how that process changes the texture of the f
  watermarked    tokens= 300  scored= 300  score=   665.92  p=1.762e-57  -> WATERMARKED
  opt-out        tokens= 300  scored= 300  score=   287.96  p=7.534e-01  -> not detected
[9/10] Explain how tectonic plates move and why their motion causes earthquakes along certain fau
  watermarked    tokens= 300  scored= 299  score=   662.83  p=4.360e-57  -> WATERMARKED
  opt-out        tokens= 300  scored= 299  score=   299.67  p=4.769e-01  -> not detected
[10/10] Explain how astronomers can detect a black hole even though no light escapes it.
  watermarked    tokens= 296  scored= 295  score=   670.77  p=2.026e-60  -> WATERMARKED
  opt-out        tokens= 300  scored= 299  score=   308.90  p=2.791e-01  -> not detected

summary over 10 prompts: watermarked detected 10/10 (TPR 1.000); opt-out flagged 0/10 (FPR 0.000); elapsed 59s

human-written passages (human_corpus.jsonl): flagged 1/50 (FPR 0.020); min p=4.062e-03 median p=5.211e-01

wrote /tmp/claude-1000/-home-anaeem-vllm-watermark/3f94f42f-6d28-4614-b036-7e6a0f444d4b/scratchpad/native-demo-t1.json
exit=0

### run B: model defaults (temperature -1, top_p -1), n=10
$ python3 scripts/native-watermark-demo.py --vllm-url $VLLM --detector-url $DET --cacert cluster/router-ca.crt --n 10 --max-tokens 300 --temperature -1 --top-p -1 --json-out native-demo-default.json
vLLM: <route>  model: Qwen/Qwen2.5-1.5B-Instruct
detector: <route>
sampling: temperature=model default top_p=model default max_tokens=300

[1/10] Explain how vaccines train the immune system to recognize pathogens it has never encounter
  watermarked    tokens= 300  scored= 294  score=   355.20  p=3.797e-04  -> WATERMARKED
  opt-out        tokens= 300  scored= 300  score=   273.53  p=9.402e-01  -> not detected
[2/10] Explain why the night sky is dark even though the universe contains an enormous number of 
  watermarked    tokens= 300  scored= 300  score=   386.74  p=1.995e-06  -> WATERMARKED
  opt-out        tokens= 200  scored= 199  score=   189.20  p=7.526e-01  -> not detected
[3/10] Explain how compound interest can turn small, regular savings into significant wealth over
  watermarked    tokens= 300  scored= 296  score=   332.49  p=1.976e-02  -> not detected
  opt-out        tokens= 300  scored= 287  score=   283.84  p=5.664e-01  -> not detected
[4/10] Explain the difference between weather and climate, and why one unusually cold winter does
  watermarked    tokens= 269  scored= 268  score=   330.59  p=1.715e-04  -> WATERMARKED
  opt-out        tokens= 300  scored= 300  score=   305.97  p=3.588e-01  -> not detected
[5/10] Explain how a criminal trial in a common-law system moves from arrest to verdict.
  watermarked    tokens= 300  scored= 300  score=   387.93  p=1.505e-06  -> WATERMARKED
  opt-out        tokens= 300  scored= 300  score=   330.65  p=4.168e-02  -> not detected
[6/10] Explain why some metals conduct electricity so much better than others.
  watermarked    tokens= 300  scored= 297  score=   414.10  p=5.913e-10  -> WATERMARKED
  opt-out        tokens= 300  scored= 297  score=   288.12  p=6.918e-01  -> not detected
[7/10] Explain how search engines decide which web pages to show first for a given query.
  watermarked    tokens= 300  scored= 300  score=   419.13  p=3.819e-10  -> WATERMARKED
  opt-out        tokens= 300  scored= 293  score=   309.82  p=1.626e-01  -> not detected
[8/10] Explain why yeast makes bread dough rise and how that process changes the texture of the f
  watermarked    tokens= 300  scored= 298  score=   416.77  p=3.821e-10  -> WATERMARKED
  opt-out        tokens= 300  scored= 299  score=   314.04  p=1.908e-01  -> not detected
[9/10] Explain how tectonic plates move and why their motion causes earthquakes along certain fau
  watermarked    tokens= 300  scored= 299  score=   397.77  p=9.916e-08  -> WATERMARKED
  opt-out        tokens= 282  scored= 276  score=   263.85  p=7.648e-01  -> not detected
[10/10] Explain how astronomers can detect a black hole even though no light escapes it.
  watermarked    tokens= 300  scored= 300  score=   406.89  p=1.205e-08  -> WATERMARKED
  opt-out        tokens= 300  scored= 299  score=   303.91  p=3.814e-01  -> not detected

summary over 10 prompts: watermarked detected 9/10 (TPR 0.900); opt-out flagged 0/10 (FPR 0.000); elapsed 59s

wrote /tmp/claude-1000/-home-anaeem-vllm-watermark/3f94f42f-6d28-4614-b036-7e6a0f444d4b/scratchpad/native-demo-default.json
exit=0

### run C: greedy (temperature 0), n=5
$ python3 scripts/native-watermark-demo.py --vllm-url $VLLM --detector-url $DET --cacert cluster/router-ca.crt --n 5 --max-tokens 300 --temperature 0 --top-p 1.0 --json-out native-demo-t0.json
vLLM: <route>  model: Qwen/Qwen2.5-1.5B-Instruct
detector: <route>
sampling: temperature=0.0 top_p=1.0 max_tokens=300

[1/5] Explain how vaccines train the immune system to recognize pathogens it has never encounter
  watermarked    tokens= 267  scored= 261  score=   261.40  p=4.820e-01  -> not detected
  opt-out        tokens= 267  scored= 261  score=   261.40  p=4.820e-01  -> not detected
[2/5] Explain why the night sky is dark even though the universe contains an enormous number of 
  watermarked    tokens= 300  scored= 299  score=   309.99  p=2.587e-01  -> not detected
  opt-out        tokens= 300  scored= 299  score=   309.99  p=2.587e-01  -> not detected
[3/5] Explain how compound interest can turn small, regular savings into significant wealth over
  watermarked    tokens= 300  scored= 271  score=   287.21  p=1.621e-01  -> not detected
  opt-out        tokens= 300  scored= 271  score=   287.21  p=1.621e-01  -> not detected
[4/5] Explain the difference between weather and climate, and why one unusually cold winter does
  watermarked    tokens= 300  scored= 299  score=   305.42  p=3.490e-01  -> not detected
  opt-out        tokens= 300  scored= 299  score=   305.42  p=3.490e-01  -> not detected
[5/5] Explain how a criminal trial in a common-law system moves from arrest to verdict.
  watermarked    tokens= 275  scored= 271  score=   307.21  p=1.664e-02  -> not detected
  opt-out        tokens= 275  scored= 271  score=   307.21  p=1.664e-02  -> not detected

summary over 5 prompts: watermarked detected 0/5 (TPR 0.000); opt-out flagged 0/5 (FPR 0.000); elapsed 29s

wrote /tmp/claude-1000/-home-anaeem-vllm-watermark/3f94f42f-6d28-4614-b036-7e6a0f444d4b/scratchpad/native-demo-t0.json
exit=0
```

Engine warning emitted during run C (`oc -n watermark-demo logs deploy/vllm-watermark | grep -i greedy`):

```
(EngineCore pid=75) WARNING 09-16 05:27:32 [gpu_sampler.py:45] Watermarking is enabled, but greedy decoding (temperature=0) cannot be watermarked. This request will use ordinary greedy sampling.
```

Run C's watermarked and opt-out texts are byte-identical for all 5 prompts (greedy decoding is
deterministic and bypasses the watermark, so the opt-out field changes nothing). Command run in
the scratchpad directory the `wrote` lines name (the JSON files are not committed, per AGENTS.md
§6, so this is reproducible only after re-running run C); run A's file is the control:

```
$ python3 -c 'import json; r=json.load(open("native-demo-t0.json"))["records"]; print(len(r), sum(x["watermarked"]["text"]==x["opt-out"]["text"] for x in r))'
5 5
$ python3 -c 'import json; r=json.load(open("native-demo-t1.json"))["records"]; print(len(r), sum(x["watermarked"]["text"]==x["opt-out"]["text"] for x in r))'
10 0
```

### Key exposure check (raw)

Correction before commit: this block originally recorded `watermark_config in /server_info:
(not present)` and `digit-run that could be the key present? False`, computed by a
`python3 -c` filter over the response body with `curl -s` (no status code). The body was a
404 (`/server_info` is a development endpoint mounted only when `VLLM_SERVER_DEV_MODE=1`;
`vllm/entrypoints/launchers/api_server/routers.py:34-37` and `vllm/envs.py:167` @cd10ed6f), so
both lines carried no information about key exposure and are replaced by the raw captures
below (2026-09-16 09:07–09:12Z, same pod). `/version` is the positive control.

```
$ curl -sS --cacert cluster/router-ca.crt -w '\nhttp=%{http_code}\n' $VLLM/server_info
{"detail":"Not Found"}
http=404
$ curl -sS --cacert cluster/router-ca.crt -w '\nhttp=%{http_code}\n' "$VLLM/server_info?config_format=json"
{"detail":"Not Found"}
http=404
$ curl -sS --cacert cluster/router-ca.crt -w '\nhttp=%{http_code}\n' $VLLM/version
{"version":"0.29.1rc1.dev128+gcd10ed6f9"}
http=200
$ oc -n watermark-demo logs deploy/vllm-watermark | grep -o 'Route: [^,]*' | sort -u | tr '\n' ' '
Route: /detokenize Route: /docs Route: /docs/oauth2-redirect Route: /generative_scoring Route: /health Route: /invocations Route: /is_scaling_elastic_ep Route: /load Route: /metrics Route: /openapi.json Route: /ping Route: /redoc Route: /scale_elastic_ep Route: /tokenize Route: /v1/chat/completions Route: /v1/chat/completions/batch Route: /v1/completions Route: /v1/messages Route: /v1/messages/count_tokens Route: /v1/models Route: /v1/responses Route: /v1/responses/{response_id} Route: /v1/responses/{response_id}/cancel Route: /version
$ oc -n watermark-demo logs deploy/vllm-watermark | grep -c 'Development endpoints'
0
$ oc -n watermark-demo get deploy vllm-watermark -o jsonpath='{range .spec.template.spec.containers[0].env[*]}{.name}{" "}{end}'
MODEL WATERMARK_KEY HOME HF_HOME
$ oc -n watermark-demo logs deploy/vllm-watermark --timestamps | grep 'non-default args'
2026-09-16T05:20:50.794839664Z (APIServer pid=1) INFO 09-16 05:20:50 [api_utils.py:286] non-default args: {'model_tag': 'Qwen/Qwen2.5-1.5B-Instruct', 'host': '0.0.0.0', 'model': 'Qwen/Qwen2.5-1.5B-Instruct', 'max_model_len': 4096, 'gpu_memory_utilization': 0.85, 'watermark_config': '***'}
$ oc -n watermark-demo logs deploy/vllm-watermark | grep 'core.py:123' | grep -o 'watermark[a-z_]*' | sort | uniq -c; echo "(line length: $(oc -n watermark-demo logs deploy/vllm-watermark | grep 'core.py:123' | wc -c))"
(line length: 4116)
```

(The `grep -c server_info` count over the full log is non-zero only because of the reviewers'
own `"GET /server_info HTTP/1.1" 404 Not Found` access-log lines.) Redaction sources at
cd10ed6f (STATIC): `vllm/entrypoints/serve/utils/api_utils.py:271`
`_SENSITIVE_ARG_FIELDS = frozenset({"api_key", "hf_token", "watermark_config"})`;
`vllm/config/watermarking.py:31` `key: int = Field(ge=0, repr=False, exclude=True)`;
`VllmConfig.__str__` (`vllm/config/vllm.py:2597`) contains no `watermark` reference, so the text
form of `/server_info` would omit the block entirely even with dev mode on. What the JSON form
would show with dev mode on was not executed.

```
$ oc -n watermark-demo get deploy vllm-watermark -o jsonpath='{.spec.template.spec.containers[0].args}'
["exec vllm serve \"$MODEL\" --host 0.0.0.0 --port 8000 --max-model-len 4096 --gpu-memory-utilization 0.85 --watermark-config \"{\\\"algorithm\\\":\\\"gumbel\\\",\\\"key\\\":${WATERMARK_KEY}}\""]
$ oc -n watermark-demo exec deploy/vllm-watermark -- sh -c 'ps -o args= -p 1 | sed -E "s/key[^0-9]*[0-9]+/key<redacted>/"'
/usr/bin/python3 /usr/local/bin/vllm serve Qwen/Qwen2.5-1.5B-Instruct --host 0.0.0.0 --port 8000 --max-model-len 4096 --gpu-memory-utilization 0.85 --watermark-config {"algorithm":"gumbel","key":<reda
```

### generation_config defaults and per-field merge (raw; 2026-09-16 09:10Z)

```
$ curl -sSL https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/raw/main/generation_config.json
{
  "bos_token_id": 151643,
  "pad_token_id": 151643,
  "do_sample": true,
  "eos_token_id": [
    151645,
    151643
  ],
  "repetition_penalty": 1.1,
  "temperature": 0.7,
  "top_p": 0.8,
  "top_k": 20,
  "transformers_version": "4.37.0"
}
$ oc -n watermark-demo logs deploy/vllm-watermark --timestamps | grep -i overridden
2026-09-16T05:23:17.266792473Z (APIServer pid=1) WARNING 09-16 05:23:17 [model.py:1772] Default vLLM sampling parameters have been overridden by the model's `generation_config.json`: `{'repetition_penalty': 1.1, 'temperature': 0.7, 'top_k': 20, 'top_p': 0.8}`. If this is not intended, please relaunch vLLM instance with `--generation-config vllm`.
$ curl -sSL https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/raw/main/generation_config.json   # 2026-09-16 11:29Z; the guide's "the 7B variant ships repetition_penalty 1.05"
{
  "bos_token_id": 151643,
  "pad_token_id": 151643,
  "do_sample": true,
  "eos_token_id": [
    151645,
    151643
  ],
  "repetition_penalty": 1.05,
  "temperature": 0.7,
  "top_p": 0.8,
  "top_k": 20,
  "transformers_version": "4.37.0"
}
```

Seeded opt-out requests (`temperature 1.0, top_p 1.0, seed 7, max_tokens 80, watermarking false`,
prompt "Explain how vaccines train the immune system.") differing only in whether `top_k` /
`repetition_penalty` are passed; `gen '<extra fields>'` = the §4.1 curl with those fields appended:

```
$ gen '' > gc_omit.txt; gen ',"top_k":20,"repetition_penalty":1.1' > gc_explicit.txt; gen ',"top_k":0,"repetition_penalty":1.0' > gc_neutral.txt; gen '' > gc_omit2.txt
$ cmp -s gc_omit.txt gc_explicit.txt && echo identical || echo differ
identical
$ cmp -s gc_omit.txt gc_neutral.txt && echo identical || echo differ
differ
$ cmp -s gc_omit.txt gc_omit2.txt && echo identical || echo differ
identical
$ sha256sum gc_*.txt
9755230999197c1ed5451c069674da4be6dd902b4bea51d396d838c339d1f3d7  gc_explicit.txt
9755230999197c1ed5451c069674da4be6dd902b4bea51d396d838c339d1f3d7  gc_omit.txt
9755230999197c1ed5451c069674da4be6dd902b4bea51d396d838c339d1f3d7  gc_omit2.txt
f6515da93d546602ce16ad2d6763304402d4b80f2cfc3c3e06e9494164901ffe  gc_neutral.txt
```

So a request that sets only temperature/top_p still runs with the model's `top_k 20` and
`repetition_penalty 1.1`; runs A, B, C and the §4.1 hand requests all did. STATIC source:
`vllm/entrypoints/openai/chat_completion/protocol.py` `to_sampling_params` @cd10ed6f falls back
to `default_sampling_params` per field (repetition_penalty L679, temperature L684, top_p L688,
top_k L692).

### Detector re-tokenization vs `completion_tokens` (raw; 2026-09-16 09:11Z)

Why `num_scored_tokens` can differ from `usage.completion_tokens`. (a) A completion that stops
naturally: `completion_tokens` counts the `<|im_end|>` stop token (id 151645), which is not in the
returned text.

```
$ curl -s --cacert cluster/router-ca.crt $VLLM/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"Qwen/Qwen2.5-1.5B-Instruct","messages":[{"role":"user","content":"Reply with exactly one short sentence: what colour is the sky on a clear day?"}],"max_tokens":300,"temperature":1.0,"top_p":1.0,"seed":11,"return_token_ids":true}'
finish_reason: stop  completion_tokens: 19
token_ids len: 19  last3: [34615, 13, 151645]
text: "The sky is typically blue on a clear day due to scattering of sunlight by air molecules."
$ curl -s --cacert cluster/router-ca.crt $VLLM/tokenize -d '{"model":"Qwen/Qwen2.5-1.5B-Instruct","prompt":<that text>,"add_special_tokens":false}'
count: 18  last3: [3720, 34615, 13]
```

A 300-token completion cut at `max_tokens` (seed 11, prompt "Explain why the sky is blue.")
re-tokenized to exactly 300 (`return_token_ids` len 300, `/tokenize` count 300, same last ids
`[11, 12299, 1526]`). (b) The §4.1 pipelines write the completion with `print(...)`, which appends
a newline; on a completion cut mid-sentence that newline is one extra token. Re-scoring a
300-token completion saved that way (`wm.txt`, a reviewer's re-run of the §4.1 request, 234 words):

```
$ python3 -c 'import json,sys; print(json.dumps({"text": sys.stdin.read()}))' < wm.txt | curl -s --cacert cluster/router-ca.crt $DET/detect -H 'Content-Type: application/json' -d @-
{"score":618.1141308873476,"p_value":4.769335228500949e-46,"num_scored_tokens":301,"is_watermarked":true}
$ python3 -c 'import sys; sys.stdout.write(open("wm.txt").read().rstrip("\n"))' | python3 -c 'import json,sys; print(json.dumps({"text": sys.stdin.read()}))' | curl -s --cacert cluster/router-ca.crt $DET/detect -H 'Content-Type: application/json' -d @-
{"score":614.650557485355,"p_value":1.3802714481342609e-45,"num_scored_tokens":300,"is_watermarked":true}
```

(On a 259-token opt-out completion that ended in a full stop the same test gave 259 both ways: the
newline merged into the final token.) The original §4.1 `plain.txt` (301 scored for 300 generated)
was not re-scored because the file had been overwritten; the mechanism above is the most likely
cause.

Re-run of the §4.1 request with the full response saved first (2026-09-16 11:30–11:32Z), so the
generator's `finish_reason` and `usage` back the "300-token completion" above: the `wm.txt` scored
above is byte-identical to what it returned (`sha256sum wm2.txt wm.txt` → the same
`6a9c4cfd…`), `/tokenize` gives 301 as written and 300 stripped (the extra id 198 is `"\n"`), and
the detector gives the same 618.114 / 614.651 as above. Part (c): the detokenize/tokenize pairs
behind the guide's "either direction" statement.

```
$ curl -s --cacert cluster/router-ca.crt $VLLM/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"Qwen/Qwen2.5-1.5B-Instruct","messages":[{"role":"user","content":"Explain how vaccines train the immune system."}],"max_tokens":300,"temperature":1.0,"top_p":1.0}' > wm_1.json
$ python3 -c 'import json; print(json.load(open("wm_1.json"))["choices"][0]["message"]["content"])' > wm2.txt   # the guide's print(...) pipeline, applied to the saved response
$ python3 -c 'import json,hashlib; d=json.load(open("wm_1.json")); c=d["choices"][0]; print(c["finish_reason"], json.dumps(d["usage"])); print("sha256(content)=", hashlib.sha256(c["message"]["content"].encode()).hexdigest())'
length {"prompt_tokens": 38, "total_tokens": 338, "completion_tokens": 300, "prompt_tokens_details": null, "completion_tokens_details": null}
sha256(content)= 5b8a9a9418901e8e111c9facf9a36360a8ab14e8a908d482e696787c5e2b021c
$ wc -w wm2.txt; sha256sum wm2.txt
234 wm2.txt
6a9c4cfdddcc756b2d97e78fa7ab61e8c108523d93da43b55871523ac2b41cbf  wm2.txt
$ for f in wm_1 wm_2 wm_3 wm_seed1_a wm_seed1_b wm_seed2 wm_seed3 plain_1 plain_2 plain_3 plain_seed1_1 plain_seed1_2; do python3 -c 'import json,hashlib,sys; d=json.load(open(sys.argv[1])); print(hashlib.sha256(d["choices"][0]["message"]["content"].encode()).hexdigest(), sys.argv[1])' $f.json; done   # the responses of the "Fixed-key determinism" block below
5b8a9a9418901e8e111c9facf9a36360a8ab14e8a908d482e696787c5e2b021c wm_1.json
5b8a9a9418901e8e111c9facf9a36360a8ab14e8a908d482e696787c5e2b021c wm_2.json
5b8a9a9418901e8e111c9facf9a36360a8ab14e8a908d482e696787c5e2b021c wm_3.json
5b8a9a9418901e8e111c9facf9a36360a8ab14e8a908d482e696787c5e2b021c wm_seed1_a.json
5b8a9a9418901e8e111c9facf9a36360a8ab14e8a908d482e696787c5e2b021c wm_seed1_b.json
5b8a9a9418901e8e111c9facf9a36360a8ab14e8a908d482e696787c5e2b021c wm_seed2.json
5b8a9a9418901e8e111c9facf9a36360a8ab14e8a908d482e696787c5e2b021c wm_seed3.json
36c213b33d5998b804beb40e7532b3faa5cd14276c758b0abe576f6454f5cbf7 plain_1.json
f6dbfd416c4ad770041f4dc901dc41419010576e2523ba4930ee7e48ff47d4a0 plain_2.json
2577bdc7adf46675909b7fad4b2504bb311b52a8b5d5de99291cc66413f335e8 plain_3.json
3655747be8ac98106ed290f3951d2171349f866c762378f06d7e51aa1389e65d plain_seed1_1.json
3655747be8ac98106ed290f3951d2171349f866c762378f06d7e51aa1389e65d plain_seed1_2.json
$ python3 -c 'import json,sys; print(json.dumps({"model":"Qwen/Qwen2.5-1.5B-Instruct","prompt":open("wm2.txt").read(),"add_special_tokens":False}))' | curl -s --cacert $CA $VLLM/tokenize -H "Content-Type: application/json" -d @- | python3 -c 'import json,sys; d=json.load(sys.stdin); print("as-written count:", d["count"], "last3:", d["tokens"][-3:])'
as-written count: 301 last3: [69458, 3842, 198]
$ python3 -c 'import json,sys; print(json.dumps({"model":"Qwen/Qwen2.5-1.5B-Instruct","prompt":open("wm2.txt").read().rstrip("\n"),"add_special_tokens":False}))' | curl -s --cacert $CA $VLLM/tokenize -H "Content-Type: application/json" -d @- | python3 -c 'import json,sys; d=json.load(sys.stdin); print("stripped count:", d["count"], "last3:", d["tokens"][-3:])'
stripped count: 300 last3: [279, 69458, 3842]
$ python3 -c 'import json,sys; print(json.dumps({"text": sys.stdin.read()}))' < wm2.txt | curl -s --cacert $CA $DET/detect -H "Content-Type: application/json" -d @-
{"score":618.1141308873476,"p_value":4.769335228500949e-46,"num_scored_tokens":301,"is_watermarked":true}
$ python3 -c 'import sys; sys.stdout.write(open("wm2.txt").read().rstrip("\n"))' | python3 -c 'import json,sys; print(json.dumps({"text": sys.stdin.read()}))' | curl -s --cacert $CA $DET/detect -H "Content-Type: application/json" -d @-
{"score":614.650557485355,"p_value":1.3802714481342609e-45,"num_scored_tokens":300,"is_watermarked":true}
$ # (c) a non-canonical token pair re-encodes to fewer or more tokens: /detokenize the pair, /tokenize the text
$ curl -s --cacert $CA $VLLM/detokenize -d '{"model":"Qwen/Qwen2.5-1.5B-Instruct","tokens":[279, 265]}'
{"prompt":" there"}
$ curl -s --cacert $CA $VLLM/tokenize -d '{"model":"Qwen/Qwen2.5-1.5B-Instruct","prompt":" there","add_special_tokens":false}'
{"count":1,"max_model_len":4096,"tokens":[1052],"token_strs":null}
$ curl -s --cacert $CA $VLLM/detokenize -d '{"model":"Qwen/Qwen2.5-1.5B-Instruct","tokens":[45421, 2912]}'
{"prompt":"Liopt"}
$ curl -s --cacert $CA $VLLM/tokenize -d '{"model":"Qwen/Qwen2.5-1.5B-Instruct","prompt":"Liopt","add_special_tokens":false}'
{"count":3,"max_model_len":4096,"tokens":[43,815,417],"token_strs":null}
$ for t in 279 265 45421 2912 1052 43 815 417 198; do printf '[%s] ' $t; curl -s --cacert cluster/router-ca.crt $VLLM/detokenize -H 'Content-Type: application/json' -d '{"model":"Qwen/Qwen2.5-1.5B-Instruct","tokens":['$t']}'; echo; done
[279] {"prompt":" the"}
[265] {"prompt":"re"}
[45421] {"prompt":"Li"}
[2912] {"prompt":"opt"}
[1052] {"prompt":" there"}
[43] {"prompt":"L"}
[815] {"prompt":"io"}
[417] {"prompt":"pt"}
[198] {"prompt":"\n"}
$ sha256sum wm2.txt wm.txt
6a9c4cfdddcc756b2d97e78fa7ab61e8c108523d93da43b55871523ac2b41cbf  wm2.txt
6a9c4cfdddcc756b2d97e78fa7ab61e8c108523d93da43b55871523ac2b41cbf  wm.txt
```

### Nightly ancestry (GitHub compare API; 2026-09-16 09:10Z)

```
$ for p in 54053 56122; do gh api repos/vllm-project/vllm/pulls/$p --jq '"\(.number) merged_at=\(.merged_at) merge_commit=\(.merge_commit_sha)"'; done
54053 merged_at=2026-09-10T04:00:15Z merge_commit=ea40bb9e905f8d552281dd1ec074f91865a0a242
56122 merged_at=2026-09-12T18:42:02Z merge_commit=7ee8a6dd013819838da8012ca549d724bee7c6c6
$ gh api repos/vllm-project/vllm/compare/ea40bb9e905f8d552281dd1ec074f91865a0a242...7ee8a6dd013819838da8012ca549d724bee7c6c6 --jq '{status,ahead_by,behind_by}'
{"ahead_by":141,"behind_by":0,"status":"ahead"}
$ gh api repos/vllm-project/vllm/git/ref/tags/v0.29.1rc0 --jq .object.sha
7ee8a6dd013819838da8012ca549d724bee7c6c6
$ gh api repos/vllm-project/vllm/compare/7ee8a6dd013819838da8012ca549d724bee7c6c6...v0.29.0 --jq '{status,ahead_by,behind_by}'
{"ahead_by":14,"behind_by":596,"status":"diverged"}
$ # Docker Hub daily nightly-<sha> tags with a September last_updated, and each one compared with both merge commits
$ # (re-captured 2026-09-16 11:28Z with the exact commands; two hand-formatted lists stood here before and are replaced by this raw output)
$ curl -s 'https://hub.docker.com/v2/repositories/vllm/vllm-openai/tags?page_size=100&name=nightly-' | jq -r '.count, .next'
51
null
$ curl -s 'https://hub.docker.com/v2/repositories/vllm/vllm-openai/tags?page_size=100&name=nightly-' | jq -r '.results[]|select(.name|test("^nightly-[0-9a-f]{40}$"))|select(.last_updated|startswith("2026-09"))|"\(.last_updated) \(.name)"' | sort
2026-09-05T06:16:31.617403Z nightly-e962733e08d10f7ca65dac4df99e116460b8b174
2026-09-06T06:17:03.716835Z nightly-1970f3ed4be7fa8620e4ddc4a12c36a8384cfc27
2026-09-07T06:16:01.102138Z nightly-d9105ea8001e0a6d77a96327d17515bb5791fb36
2026-09-08T06:16:53.323512Z nightly-9ea8f3ffc354901b740f0b31988900897b7221d7
2026-09-09T06:16:33.643746Z nightly-385dce36bcee42309924a5ece951a96db3dce7f2
2026-09-10T06:17:13.892169Z nightly-2a02f6efe319c885e3ccbcecde402e0028f9ec1e
2026-09-11T06:18:48.451548Z nightly-e7edf17cea217e52701f913cd8491fcacf2d9490
2026-09-12T06:16:44.776152Z nightly-eed1f3d0c6043bd494424a22443ee198dd56f657
2026-09-13T06:15:10.189688Z nightly-2671fedfc7ae604761990603fc736c0c4f21de57
2026-09-14T06:15:25.054268Z nightly-dc36fcce902a63eab06c1b93a5c4a5ee178a0c56
2026-09-15T06:18:57.060717Z nightly-cd10ed6f9f6b37a8ace9cf380007e66fe12ec0c3
2026-09-16T06:20:11.695298Z nightly-af1c01499b289be555c475669ba50a88e96d846e

$ for sha in $(awk '{sub("nightly-","",$2); print $2}' <that list>); do for m in ea40bb9e905f8d552281dd1ec074f91865a0a242 7ee8a6dd013819838da8012ca549d724bee7c6c6; do printf "%s...%s " "${m:0:8}" "${sha:0:8}"; gh api "repos/vllm-project/vllm/compare/$m...$sha" --jq '"\(.status) ahead=\(.ahead_by) behind=\(.behind_by)"'; done; done
ea40bb9e...e962733e behind ahead=0 behind=190
7ee8a6dd...e962733e behind ahead=0 behind=331
ea40bb9e...1970f3ed behind ahead=0 behind=176
7ee8a6dd...1970f3ed behind ahead=0 behind=317
ea40bb9e...d9105ea8 behind ahead=0 behind=155
7ee8a6dd...d9105ea8 behind ahead=0 behind=296
ea40bb9e...9ea8f3ff behind ahead=0 behind=111
7ee8a6dd...9ea8f3ff behind ahead=0 behind=252
ea40bb9e...385dce36 behind ahead=0 behind=47
7ee8a6dd...385dce36 behind ahead=0 behind=188
ea40bb9e...2a02f6ef ahead ahead=1 behind=0
7ee8a6dd...2a02f6ef behind ahead=0 behind=140
ea40bb9e...e7edf17c ahead ahead=54 behind=0
7ee8a6dd...e7edf17c behind ahead=0 behind=87
ea40bb9e...eed1f3d0 ahead ahead=116 behind=0
7ee8a6dd...eed1f3d0 behind ahead=0 behind=25
ea40bb9e...2671fedf ahead ahead=150 behind=0
7ee8a6dd...2671fedf ahead ahead=9 behind=0
ea40bb9e...dc36fcce ahead ahead=188 behind=0
7ee8a6dd...dc36fcce ahead ahead=47 behind=0
ea40bb9e...cd10ed6f ahead ahead=269 behind=0
7ee8a6dd...cd10ed6f ahead ahead=128 behind=0
ea40bb9e...af1c0149 ahead ahead=328 behind=0
7ee8a6dd...af1c0149 ahead ahead=187 behind=0
```

Partition: nightlies up to 2026-09-09 contain neither PR (09-05 through 09-09 are `behind` both
merges); 2026-09-10 to 09-12 contain only #54053; 2026-09-13 onward contain both (the 09-16
nightly `af1c0149…`, pushed 06:20:11Z on the day of this run, is `ahead` of both). A compare
against `7ee8a6dd` alone suffices because it descends from `ea40bb9e`.

### Fixed-key determinism: repeated requests, `seed`, opt-out and `n` (raw; 2026-09-16 11:30–11:34Z)

Sequential requests, one at a time, with no other client of ours running (one vLLM pod). `gen <extra-json>
<file>` is the §4.1 curl (`max_tokens 300, temperature 1.0, top_p 1.0`) with the extra fields
appended and the full JSON response saved; `summ` prints `finish_reason`, `usage.completion_tokens`,
the word count, the first 16 hex digits of sha256(content) and the first 60 characters. Full hashes
of every response are in the re-tokenization block above.

```
# gen <extra-json> <file> = the §4.1 curl with the extra fields appended, full JSON response saved; summ = finish_reason, usage.completion_tokens, word count, sha256(content)[:16], first 60 chars
$ gen "" wm_1.json; summ wm_1.json
length completion_tokens=300 words=234 sha256=5b8a9a9418901e8e head='Vaccines work by teaching the immune system how to recognize'
$ gen "" wm_2.json; summ wm_2.json
length completion_tokens=300 words=234 sha256=5b8a9a9418901e8e head='Vaccines work by teaching the immune system how to recognize'
$ gen "" wm_3.json; summ wm_3.json
length completion_tokens=300 words=234 sha256=5b8a9a9418901e8e head='Vaccines work by teaching the immune system how to recognize'
$ gen ',"seed":1' wm_seed1_a.json; summ
length completion_tokens=300 words=234 sha256=5b8a9a9418901e8e head='Vaccines work by teaching the immune system how to recognize'
$ gen ',"seed":1' wm_seed1_b.json; summ
length completion_tokens=300 words=234 sha256=5b8a9a9418901e8e head='Vaccines work by teaching the immune system how to recognize'
$ gen ',"seed":2' wm_seed2.json; summ
length completion_tokens=300 words=234 sha256=5b8a9a9418901e8e head='Vaccines work by teaching the immune system how to recognize'
$ gen ',"seed":3' wm_seed3.json; summ
length completion_tokens=300 words=234 sha256=5b8a9a9418901e8e head='Vaccines work by teaching the immune system how to recognize'
$ gen ',"watermarking":false' plain_1.json; summ
length completion_tokens=300 words=227 sha256=36c213b33d5998b8 head='Vaccines work by exposing the immune system to parts of a vi'
$ gen ',"watermarking":false' plain_2.json; summ
length completion_tokens=300 words=254 sha256=f6dbfd416c4ad770 head='Vaccines work by training the immune system to recognize and'
$ gen ',"watermarking":false' plain_3.json; summ
length completion_tokens=300 words=181 sha256=2577bdc7adf46675 head='Vaccines stimulate an immune response to create immunity wit'
$ gen ',"watermarking":false,"seed":1' plain_seed1_1.json; summ
length completion_tokens=300 words=241 sha256=3655747be8ac9810 head='Vaccines work by exposing the immune system to harmless part'
$ gen ',"watermarking":false,"seed":1' plain_seed1_2.json; summ
length completion_tokens=300 words=241 sha256=3655747be8ac9810 head='Vaccines work by exposing the immune system to harmless part'
```

Result: 7 of 7 watermarked completions byte-identical (`5b8a9a94…`), including `seed` 1, 1, 2 and 3;
the three unseeded opt-out completions all differ; the two opt-out completions with `seed` 1 are
identical (`3655747b…`). The completion is the 234-word `wm.txt` scored in the re-tokenization block,
not the 228-word one captured for §4.1 earlier in this section (score 617.807), so the fixed key gives
near-determinism, not a guarantee; what changed between the two captures was not isolated. Second
prompt, same shape (short `stop` completions):

```
# second prompt: "Write two sentences about the ocean." (same curl shape as 4.1; sha256 of content)
$ gen ""
stop completion_tokens=50 sha256=5fbe32bc8d926da6 head='The vastness of the ocean is truly breathtaking - '
$ gen ""
stop completion_tokens=50 sha256=5fbe32bc8d926da6 head='The vastness of the ocean is truly breathtaking - '
$ gen ""
stop completion_tokens=50 sha256=5fbe32bc8d926da6 head='The vastness of the ocean is truly breathtaking - '
$ gen ',"seed":1'
stop completion_tokens=50 sha256=5fbe32bc8d926da6 head='The vastness of the ocean is truly breathtaking - '
$ gen ',"seed":1'
stop completion_tokens=50 sha256=5fbe32bc8d926da6 head='The vastness of the ocean is truly breathtaking - '
$ gen ',"seed":2'
stop completion_tokens=50 sha256=5fbe32bc8d926da6 head='The vastness of the ocean is truly breathtaking - '
$ gen ',"watermarking":false'
stop completion_tokens=76 sha256=1a9e7e402907c6e3 head='The vastness of the ocean, covering approximately '
$ gen ',"watermarking":false'
stop completion_tokens=53 sha256=af29a07648f75333 head='1. The vast expanse of the ocean is home to millio'
$ gen ',"watermarking":false'
stop completion_tokens=77 sha256=6cd72afc2c919048 head='1. The blue expanse of the sea is both tranquil an'
$ gen ',"watermarking":false,"seed":1'
stop completion_tokens=44 sha256=772a1245aeadd1c1 head="The ocean is vast and powerful, shaping the earth'"
$ gen ',"watermarking":false,"seed":1'
stop completion_tokens=44 sha256=772a1245aeadd1c1 head="The ocean is vast and powerful, shaping the earth'"
```

6 of 6 watermarked completions identical (`5fbe32bc…`), with and without `seed`; unseeded opt-out
completions differ, `seed 1` opt-out completions match. Finally `n: 2` on the §4.1 prompt: two
different completions, neither equal to the single-request `5b8a9a94…` text (the two sequences share
a batch; vLLM is not batch-invariant by default, `VLLM_BATCH_INVARIANT` in `vllm/envs.py` L92
@cd10ed6f, STATIC; whether that is the cause was not tested, OPEN):

```
$ curl -s --cacert $CA $VLLM/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"Qwen/Qwen2.5-1.5B-Instruct","messages":[{"role":"user","content":"Explain how vaccines train the immune system."}],"max_tokens":300,"temperature":1.0,"top_p":1.0,"n":2}' | python3 -c 'import json,sys,hashlib; d=json.load(sys.stdin); [print(i, c["finish_reason"], "sha256="+hashlib.sha256(c["message"]["content"].encode()).hexdigest()[:16], repr(c["message"]["content"][:50])) for i,c in enumerate(d["choices"])]; print("usage:", json.dumps(d["usage"]))'
0 length sha256=5262933374e0575a 'Vaccines work by teaching the immune system how to'
1 length sha256=c77c5d12a84643ad 'Vaccines work by teaching the immune system how to'
usage: {"prompt_tokens": 38, "total_tokens": 638, "completion_tokens": 600, "prompt_tokens_details": null, "completion_tokens_details": null}
```

STATIC sources for why `seed` does not reach watermarked positions (all @cd10ed6f): the keyed draw is
`philox_gumbel_sample(logits, contexts, self.prf.key)` (`vllm/v1/watermarking/gumbel.py` L64-67);
the mixed kernel `vllm/v1/worker/gpu/sample/watermark.py` L364-371 loads `seeds_ptr` only inside its
`if tl.load(skip_mask_ptr + row)` branch; `vllm/v1/watermarking/gpu_sampler.py` L91-104 builds that
`skip_mask` from non-watermarking requests, `temperature == 0` and repeated contexts. RFC #53916
"Note on diversity": "a _catastrophic_ loss of diversity (determinism, given a fixed key and input).
The proposed Gumbel-max algorithm, given its simplicity, does exhibit this behaviour."

### Port-forward alternative (raw; 2026-09-16 11:36Z)

The guide's section 4.0 alternative to the Routes, with and without `-n` (the installer-written
kubeconfig has no current project; local ports 18000/18080 used so as not to collide with anything):

```
$ timeout 10 oc port-forward svc/vllm-watermark 18000:8000; echo "exit=$?"
Error from server (NotFound): services "vllm-watermark" not found
exit=1
$ (timeout 20 oc -n watermark-demo port-forward svc/vllm-watermark 18000:8000 >/dev/null &); sleep 4; curl -s http://localhost:18000/v1/models | python3 -c ...
['Qwen/Qwen2.5-1.5B-Instruct']
$ (timeout 20 oc -n watermark-demo port-forward svc/watermark-detector 18080:8080 >/dev/null &); sleep 4; curl -s http://localhost:18080/detect -H "Content-Type: application/json" -d '{"text":"It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife."}'
{"score":19.542122906190592,"p_value":0.9070642274102403,"num_scored_tokens":26,"is_watermarked":false}
```

### Human corpus provenance (raw; local workstation, 2026-09-16 11:29Z)

`benchmarks/data/human_corpus.jsonl` (gitignored) is the file run A scored. Its only recorded
generation before today was the `--chunk-tokens 512` variant (2026-08-08, line 555); the file's
mtime is 2026-08-08 03:06 and `benchmarks/fetch_human_corpus.py` is unchanged since ef3b0e4. The
2026-08-08 record[0] prefix at line 941 (`sha256:d93033d39053`, 1107 bytes) does not match the
current file's record[0] `text` (`74f6a8b5cd92`, 1107 bytes) under sha256 of the UTF-8 text, so the
full-file hash below is the authoritative identity. Regeneration with the script defaults
(`Qwen/Qwen2.5-0.5B-Instruct` tokenizer, n=150, seed=42, 256 tokens) and with the served model's
tokenizer both reproduce it byte-for-byte; the two models' tokenizer files at HF `main` are
sha256-identical, which is why. Environment: local workstation, Python 3.14.4, transformers 4.57.6,
tokenizers 0.22.2, requests 2.32.5; HF `main` revisions at the time are listed last.

```
$ python3 benchmarks/fetch_human_corpus.py --out hc_default.jsonl; echo "exit=$?"
Token indices sequence length is longer than the specified maximum sequence length for this model (173297 > 131072). Running this sequence through the model will result in indexing errors
Loading tokenizer 'Qwen/Qwen2.5-0.5B-Instruct' ...
Fetching Gutenberg #1342: 'Pride and Prejudice' ...
  676 candidate 256-token windows: kept 651, dropped 25
Fetching Gutenberg #84: 'Frankenstein; or, The Modern Prometheus' ...
  383 candidate 256-token windows: kept 382, dropped 1
Fetching Gutenberg #11: "Alice's Adventures in Wonderland" ...
  145 candidate 256-token windows: kept 139, dropped 6
Fetching Gutenberg #2701: 'Moby-Dick; or, The Whale' ...
  1207 candidate 256-token windows: kept 1195, dropped 12

=== Summary ===
total filtered chunks available across all books: 2367
sampled (seed=42): 150
tokens per chunk: min=256 max=256 (all == --chunk-tokens by construction)
chunks per source:
    11  gutenberg:11:Alice's Adventures in Wonderland
    45  gutenberg:1342:Pride and Prejudice
    66  gutenberg:2701:Moby-Dick; or, The Whale
    28  gutenberg:84:Frankenstein; or, The Modern Prometheus
wrote hc_default.jsonl

real	0m25.859s
user	0m17.755s
sys	0m2.185s
exit=0
$ python3 benchmarks/fetch_human_corpus.py --model-tokenizer Qwen/Qwen2.5-1.5B-Instruct --out hc_15b.jsonl; echo "exit=$?"
Token indices sequence length is longer than the specified maximum sequence length for this model (173297 > 131072). Running this sequence through the model will result in indexing errors
Loading tokenizer 'Qwen/Qwen2.5-1.5B-Instruct' ...
Fetching Gutenberg #1342: 'Pride and Prejudice' ...
  676 candidate 256-token windows: kept 651, dropped 25
Fetching Gutenberg #84: 'Frankenstein; or, The Modern Prometheus' ...
  383 candidate 256-token windows: kept 382, dropped 1
Fetching Gutenberg #11: "Alice's Adventures in Wonderland" ...
  145 candidate 256-token windows: kept 139, dropped 6
Fetching Gutenberg #2701: 'Moby-Dick; or, The Whale' ...
  1207 candidate 256-token windows: kept 1195, dropped 12

=== Summary ===
total filtered chunks available across all books: 2367
sampled (seed=42): 150
tokens per chunk: min=256 max=256 (all == --chunk-tokens by construction)
chunks per source:
    11  gutenberg:11:Alice's Adventures in Wonderland
    45  gutenberg:1342:Pride and Prejudice
    66  gutenberg:2701:Moby-Dick; or, The Whale
    28  gutenberg:84:Frankenstein; or, The Modern Prometheus
wrote hc_15b.jsonl
exit=0
$ sha256sum benchmarks/data/human_corpus.jsonl hc_default.jsonl hc_15b.jsonl; wc -l benchmarks/data/human_corpus.jsonl hc_default.jsonl hc_15b.jsonl
8601024c868eb704bfe5f2b3e7c536ae4eb1a7f3970bcf524619d24039a04195  benchmarks/data/human_corpus.jsonl
8601024c868eb704bfe5f2b3e7c536ae4eb1a7f3970bcf524619d24039a04195  hc_default.jsonl
8601024c868eb704bfe5f2b3e7c536ae4eb1a7f3970bcf524619d24039a04195  hc_15b.jsonl
   150 benchmarks/data/human_corpus.jsonl
   150 hc_default.jsonl
   150 hc_15b.jsonl
   450 total
$ cmp benchmarks/data/human_corpus.jsonl hc_default.jsonl && cmp benchmarks/data/human_corpus.jsonl hc_15b.jsonl && echo IDENTICAL
IDENTICAL
$ ls -l --time-style=full-iso benchmarks/data/human_corpus.jsonl
-rw-r--r--. 1 anaeem anaeem 183550 2026-08-08 03:06:21.675947196 +0100 benchmarks/data/human_corpus.jsonl
$ python3 -c "import json,hashlib; t=json.loads(open(\"benchmarks/data/human_corpus.jsonl\").readline())[\"text\"]; print(len(t.encode()), hashlib.sha256(t.encode()).hexdigest()[:12])"
1107 74f6a8b5cd92
$ for m in Qwen2.5-0.5B-Instruct Qwen2.5-1.5B-Instruct; do for f in tokenizer.json vocab.json merges.txt tokenizer_config.json; do printf "%s %s " $m $f; curl -sSL https://huggingface.co/Qwen/$m/resolve/main/$f | sha256sum | cut -c1-64; done; done
Qwen2.5-0.5B-Instruct tokenizer.json c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539
Qwen2.5-0.5B-Instruct vocab.json ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910
Qwen2.5-0.5B-Instruct merges.txt 599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3
Qwen2.5-0.5B-Instruct tokenizer_config.json 5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583
Qwen2.5-1.5B-Instruct tokenizer.json c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539
Qwen2.5-1.5B-Instruct vocab.json ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910
Qwen2.5-1.5B-Instruct merges.txt 599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3
Qwen2.5-1.5B-Instruct tokenizer_config.json 5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583
$ for m in Qwen2.5-0.5B-Instruct Qwen2.5-1.5B-Instruct; do printf "%s main sha=" $m; curl -sSL https://huggingface.co/api/models/Qwen/$m/revision/main | jq -r .sha; done
Qwen2.5-0.5B-Instruct main sha=7ae557604adf67be50417f59c2c2f167def9a775
Qwen2.5-1.5B-Instruct main sha=989aa7980e4cf806f80c7fef2b1adb7bc71aa306
```

### Detector pod placement and GPU MachineSet state (raw; 2026-09-16 11:27Z)

The detector Deployment has no nodeSelector or affinity; it ran on a CPU worker because it was
created (05:06:03Z) before the GPU node existed (05:08:40Z), not because the manifest requires it.
The GPU node carries no taint (the `nvidia.com/gpu` toleration in `10-vllm.yaml` is inert here).
The GPU MachineSet was created at replicas=1 and never scaled (generation 1; its Machine has the
same creation timestamp), so `scripts/scale-gpu.sh` did not run on this cluster.

```
$ oc -n watermark-demo get pods -o wide
NAME                                 READY   STATUS    RESTARTS   AGE     IP            NODE                           NOMINATED NODE   READINESS GATES
vllm-watermark-7dc9fb8f79-rzgxw      1/1     Running   0          6h21m   10.129.2.17   ip-10-0-26-129.ec2.internal    <none>           <none>
watermark-detector-d4b8c9766-sfdzp   1/1     Running   0          6h21m   10.131.0.24   ip-10-0-150-204.ec2.internal   <none>           <none>
$ oc -n watermark-demo get pod -l app=watermark-detector -o jsonpath=...
name=watermark-detector-d4b8c9766-sfdzp created=2026-09-16T05:06:03Z node=ip-10-0-150-204.ec2.internal restarts=0
$ oc -n watermark-demo get deploy watermark-detector -o jsonpath=nodeSelector/affinity
nodeSelector= affinity= tolerations=
$ oc get nodes -L node.kubernetes.io/instance-type -o custom-columns
NAME                           ROLES    TYPE         CREATED                TAINTS
ip-10-0-150-204.ec2.internal   <none>   m6i.xlarge   2026-09-16T04:49:09Z   <none>
ip-10-0-21-8.ec2.internal      <none>   m6i.xlarge   2026-09-16T04:41:42Z   [map[effect:NoSchedule key:node-role.kubernetes.io/master]]
ip-10-0-26-129.ec2.internal             g5.xlarge    2026-09-16T05:08:40Z   <none>
ip-10-0-55-253.ec2.internal    <none>   m6i.xlarge   2026-09-16T04:43:04Z   [map[effect:NoSchedule key:node-role.kubernetes.io/master]]
ip-10-0-88-133.ec2.internal    <none>   m6i.xlarge   2026-09-16T04:42:44Z   [map[effect:NoSchedule key:node-role.kubernetes.io/master]]
ip-10-0-98-150.ec2.internal    <none>   m6i.xlarge   2026-09-16T04:49:22Z   <none>
$ oc -n openshift-machine-api get machineset ocp-ai-wg9fl-gpu-us-east-1a -o custom-columns
GEN   REPLICAS   CREATED
1     1          2026-09-16T05:04:19Z
$ oc -n openshift-machine-api get machine -l machine.openshift.io/cluster-api-machineset=ocp-ai-wg9fl-gpu-us-east-1a -o custom-columns
NAME                                CREATED                PHASE
ocp-ai-wg9fl-gpu-us-east-1a-p4ptc   2026-09-16T05:04:19Z   Running
```

### Results summary

| Run | Sampling | Watermarked detected (p ≤ 0.01) | Opt-out flagged | Human passages flagged |
|---|---|---|---|---|
| A | temperature 1.0, top_p 1.0, ≤300 tokens, n=10 (top_k 20 and repetition_penalty 1.1 from `generation_config.json` also in effect; the script does not set them) | 10/10 (TPR 1.000); p from 3.553e-14 to 5.242e-76 | 0/10 | 1/50 (min p 4.062e-03, median p 5.211e-01) |
| B | model defaults (temperature 0.7, top_p 0.8; top_k 20 and repetition_penalty 1.1 from `generation_config.json`, as in A), n=10 | 9/10 (TPR 0.900); miss had p = 1.976e-02; detected p from 3.797e-04 to 3.819e-10 | 0/10 | — |
| C | temperature 0 (greedy), n=5 | 0/5 (TPR 0.000); outputs identical to opt-out | 0/5 | — |

Hand step: watermarked completion `p = 2.70e-46` over 300 scored tokens; opt-out
`p = 0.172`; 26-token human sentence `p = 0.907`.

**What this does and does not show.** It shows the native feature generating and detecting
on OpenShift with the shipped detector, the opt-out field working, the greedy bypass, and the
entropy dependence via temperature/top_p (run B vs A; `top_k 20` and `repetition_penalty 1.1` from
`generation_config.json` were in force for every run, so run A was not unconstrained sampling).
The human-corpus rate (1/50 at threshold 0.01) is a small
sample and consistent with the documented calibration caveat; a production deployment must
measure its own false-positive rate with its key. Not measured here: serving overhead vs. a
watermark-off baseline, robustness to edits/paraphrase, speculative decoding, structured
output, multi-replica behaviour. GPU node left running for the verification pass; scale to 0
with `./scripts/scale-gpu.sh 0` when done.


## 2026-09-21 — Native watermarking measured on an L40S: coverage, detection power, robustness, serving cost (EXECUTED; redacted)

**Purpose.** Answer five due-diligence questions with measurement rather than
source reading: coverage across generation modes, detection power at short
lengths, survival under editing and paraphrase, and serving impact. Supersedes
the "not measured" entries of the 2026-09-16 run for those areas.

**Environment.** Fresh OpenShift 4.20.27 cluster on AWS; GPU node `g6e.2xlarge`
(NVIDIA L40S 46068 MiB, driver 595.91.07) added via
`GPU_INSTANCE_TYPE=g6e.2xlarge GPU_REPLICAS=1 ./scripts/create-gpu-machineset.sh`.
Model `Qwen/Qwen2.5-7B-Instruct`, `--max-model-len 8192`,
`--gpu-memory-utilization 0.90`. Image
`vllm/vllm-openai@sha256:20a52b807cef7a3be15d2174d293b5642d437615b20b5930d78a927e709e6bb6`
(nightly `cd10ed6f…`), pulled amd64 layer
`sha256:0a329f66a92e19ad8c8e9b9a17bda6ecf70b1a8dcfd9c735360a251ce870d0d8`.
Detector: the upstream reference server from the same image with the **matching**
7B tokenizer (`deploy/native/20-detector-7b.yaml`); a mismatched tokenizer
silently destroys the signal rather than failing. Harness:
`scripts/native-watermark-experiments.py`, `scripts/native-watermark-bench.sh`.

```
=== engine version + runner ===
(EngineCore pid=91) INFO 09-21 04:12:08 [core.py:123] Initializing a V1 LLM engine (v0.29.1rc1.dev128+gcd10ed6f9) with config: model='Qwen/Qwen2.5-7B-Instruct', speculative_config=None, tokenizer='Qwe
(EngineCore pid=91) INFO 09-21 04:12:11 [gpu_worker.py:441] Using V2 Model Runner
(EngineCore pid=91) INFO 09-21 04:13:55 [watermark_sample_warmup.py:130] Warming up watermark sampler kernel (vocab=152064, keys=1, dtypes=['torch.float32', 'torch.bfloat16'], skip_mask=[True]).
=== node ===
ip-10-0-8-39.ec2.internal   Ready   gpu,worker   56m   v1.33.12   g6e.2xlarge   NVIDIA-L40S   595.91.07
=== image ===
docker.io/vllm/vllm-openai@sha256:0a329f66a92e19ad8c8e9b9a17bda6ecf70b1a8dcfd9c735360a251ce870d0d8
=== beam-search raw evidence ===
HTTP 200
finish_reason: length
first 150 chars: Tectonic plates move due to the forces generated by the convection currents in the Earth's mantle. The Earth's lithosphere, which includes the crust a
```

### 1. Generation-mode coverage (raw)

```
model (auto): Qwen/Qwen2.5-7B-Instruct
generation-mode matrix (model Qwen/Qwen2.5-7B-Instruct, max_tokens 400)

| mode | tokens | scored | p-value | detector verdict | note |
|---|---:|---:|---:|---|---|
| baseline sampling | 400 | 389 | 1.01e-22 | WATERMARKED | watermarked default path |
| per-request opt-out | 400 | 394 | 5.92e-02 | not detected | watermarking: false |
| greedy temperature=0 | 400 | 391 | 4.30e-01 | not detected | expected: NOT watermarked |
| top_k=1 | 400 | 391 | 4.30e-01 | not detected | expected: NOT watermarked (degenerate sampling) |
| low temperature 0.2 | 400 | 393 | 3.47e-01 | not detected | expected: weak signal |
| structured JSON output | 181 | 179 | 3.82e-07 | WATERMARKED | guided decoding narrows the sampler |
| tool call arguments | - | - | - | REQUEST REJECTED | HTTP 400: {"error":{"message":"tool_choice=function \"write_summary\" requires --tool-call-parser to be set","type":"BadRequestError","param":null,"code":400}} |
| n=2 (second choice) | 800 | 392 | 3.60e-18 | WATERMARKED | multiple completions per request |
| beam search over HTTP | 400 | 388 | 9.09e-01 | not detected | claimed: NOT rejected and NOT watermarked |
| min_p 0.9 (near-degenerate) | 400 | 396 | 8.09e-01 | not detected | very few candidate tokens survive |
| logit_bias forcing | 400 | 16 | 1.63e-01 | not detected | raw-logit manipulation before the draw |
| streaming | ~400 deltas | 391 | 9.35e-23 | WATERMARKED | reassembled from SSE deltas |

wrote benchmarks/results/lab-2026-09-21/modes.json
```

**Confirmed silent bypasses.** Six modes returned ordinary HTTP 200 responses of
normal-looking text that the detector did not flag. The most consequential is
**beam search over the OpenAI-compatible HTTP API**: `use_beam_search: true` was
neither rejected nor watermarked (p = 0.909; a second targeted request returned
HTTP 200 with `finish_reason: length` and ordinary prose). The offline Python
`LLM.beam_search()` path raises `ValueError` when watermarking is configured, but
the online path does not reach that check. Greedy and `top_k=1` produced the
identical p-value (4.30e-01), the expected signature of the same deterministic
path. `temperature 0.2` was **not** detected over 400 tokens, so the entropy
dependence bites well above greedy. `min_p 0.9` was not detected. Aggressive
`logit_bias` was not detected and scored only 16 of 400 tokens, because the
output became repetitive and context deduplication discarded the remainder.

**Watermarked as expected:** baseline sampling (p 1.01e-22), streaming
reassembled from SSE deltas (9.35e-23), `n=2` (3.60e-18), and structured JSON
output under a schema (3.82e-07) — guided decoding did **not** defeat it here,
though over fewer tokens and at a weaker margin.

**Not tested:** tool-call arguments. The request was rejected with HTTP 400
because this deployment did not set `--tool-call-parser`; that is a deployment
configuration gap, not a watermark property, and the mode remains `OPEN`.

### 2. Detection power vs. length (288 records, n=12 per cell, bucketed by
tokens the detector actually scored rather than requested `max_tokens`, because
brevity steering made short cells stop early)

| temperature | scored tokens | n | detected | TPR | median p |
|---|---|---:|---:|---:|---:|
| 1.0 | 0-59 | 20 | 3 | 0.150 | 9.13e-02 |
| 1.0 | 60-99 | 4 | 2 | 0.500 | 1.10e-02 |
| 1.0 | 100-159 | 24 | 24 | 1.000 | 1.38e-05 |
| 1.0 | 250-399 | 12 | 12 | 1.000 | 6.04e-19 |
| 1.0 | 400+ | 12 | 12 | 1.000 | 2.53e-29 |
| 0.7 | 0-59 | 22 | 1 | 0.045 | 1.83e-01 |
| 0.7 | 60-99 | 2 | 1 | 0.500 | 6.12e-02 |
| 0.7 | 100-159 | 22 | 19 | 0.864 | 7.58e-05 |
| 0.7 | 160-249 | 2 | 2 | 1.000 | 1.07e-04 |
| 0.7 | 250-399 | 12 | 12 | 1.000 | 7.06e-10 |
| 0.7 | 400+ | 12 | 12 | 1.000 | 8.95e-14 |

Across the whole sweep the opt-out arm was flagged 1/144 times
(FPR 0.0069).

**Reading.** Below roughly 60 scored tokens the watermark is usually missed
(TPR 0.15 at T=1.0, 0.045 at T=0.7). Between 100 and 160 scored tokens it becomes
reliable at T=1.0 (24/24) but not yet at T=0.7 (19/22). From about 250 scored
tokens it was detected in every case at both temperatures. This is the empirical
basis for treating short outputs as unverifiable, and it is consistent with the
Code of Practice's 200-token threshold rather than in tension with it.

### 3. Robustness to editing and paraphrase (n=10 completions of 400 tokens,
T=1.0, deterministic edits with seed 20260921)

| transform | fraction | n | detected | TPR | median p |
|---|---:|---:|---:|---:|---:|
| delete | 0.05 | 10 | 10 | 1.000 | 1.56e-19 |
| delete | 0.10 | 10 | 10 | 1.000 | 1.98e-12 |
| delete | 0.25 | 10 | 8 | 0.800 | 1.24e-03 |
| none | 0.00 | 10 | 10 | 1.000 | 1.03e-25 |
| paraphrase | 1.00 | 10 | 1 | 0.100 | 7.25e-01 |
| substitute | 0.05 | 10 | 10 | 1.000 | 5.14e-13 |
| substitute | 0.10 | 10 | 10 | 1.000 | 1.55e-10 |
| substitute | 0.25 | 10 | 8 | 0.800 | 2.68e-04 |
| truncate | 0.05 | 10 | 10 | 1.000 | 3.62e-21 |
| truncate | 0.10 | 10 | 10 | 1.000 | 8.38e-19 |
| truncate | 0.25 | 10 | 10 | 1.000 | 1.72e-15 |

**Reading.** The mark survives moderate editing: deleting or substituting 5-10%
of words left detection at 10/10, and 25% still detected 8/10. Truncating up to
25% of the text left 10/10, because the surviving prefix still carries enough
scored tokens. **Paraphrase defeats it**: rewriting each completion with the same
model (watermarking off, so the rewrite carries no mark of its own) dropped
detection to 1/10 with median p = 0.725. This is the single most important
robustness result and it matches the direction of the published literature for
other schemes, now measured for this implementation.

### 4. Serving impact, paired on one warm engine (baseline = per-request opt-out,
3 trials per arm, 50 requests each, concurrency 8, 300 max tokens, T=1.0)

```
# native watermark paired serving benchmark
date_utc: 2026-09-21T04:46:42Z
base_url: https://vllm-watermark-watermark-demo.apps.<cluster-domain>/v1
model: Qwen/Qwen2.5-7B-Instruct
trials: 3  n: 50  concurrency: 8  max_tokens: 300
temperature: 1.0  top_p: 1.0

## paired serving results (same engine; baseline = per-request opt-out)

| metric | watermark on | opt-out | delta |
|---|---:|---:|---:|
| output tokens/s | 327.698 | 328.386 | -0.21% |
| wall time (s) | 45.636 | 45.581 | +0.12% |
| latency mean (s) | 6.534 | 6.500 | +0.51% |
| latency p50 (s) | 6.551 | 6.511 | +0.62% |
| latency p95 (s) | 6.584 | 6.551 | +0.50% |
| latency p99 (s) | 6.598 | 6.563 | +0.53% |

per-arm output tok/s  on=[328.0, 326.3, 327.7]
per-arm output tok/s off=[328.4, 328.1, 328.6]
trials per arm: on=3 off=3; failed requests: 0

GPU during the run: utilization mean 100.0% max 100% (54 samples)
GPU memory: mean 40363 MiB, max 40363 MiB of 46068 MiB
GPU power: mean 317 W, max 338 W
```

**Reading.** The watermark sampling path cost about 0.2% of output throughput and
about 0.5% of mean latency on this hardware and model, with zero failed requests
and per-trial throughput spreads smaller than the difference between arms. The
GPU was saturated (100% utilization) throughout, so this is a compute-bound
measurement rather than an idle-GPU artifact. **Boundaries:** this measures the
sampling path within one engine configured for watermarking. It does **not**
measure the cost of enabling the feature at all (an engine started without
`--watermark-config` also skips the watermark warmup, observed here at roughly
100 seconds of additional startup before ready). Single replica, no tensor
parallelism, no speculative decoding, one model, one GPU, one workload shape.

**Not measured here:** time-to-first-token, streaming latency, multi-replica or
tensor-parallel behaviour, speculative decoding throughput, `deduplicate_contexts=all`
cost, and any model larger than 7B.


## 2026-09-21 — Dual-key Gumbel-max: detection power vs single key, and the alpha pairing (EXECUTED; redacted)

**Why.** Upstream publishes dual-key detectability only at temperature 1.0, on
Qwen3.5-2B/27B, and only under speculative decoding (PR #56122). Nothing is
published for plain dual-key at the lower temperatures used in real serving, and
the single-key result recorded earlier today showed detection falling sharply at
temperature 0.7. This run measures dual-key on the same cluster, GPU, model and
prompts so the two are directly comparable.

**Setup.** Same L40S / `Qwen/Qwen2.5-7B-Instruct` deployment as the earlier run,
with `--watermark-config '{"algorithm":"dual_key_gumbel","key":<redacted>}'`
(generation `alpha` left at the upstream default 0.1, so roughly one token in ten
is drawn from the key-B stream). Detection used a purpose-written server
(`deploy/native/detector_server_dual.py`) because the upstream example wires up
only the single-key detector; three instances were deployed at key-B weights
0.1, 0.2 and 0.5, each confirming its configuration through a `/config` endpoint.

### Dual-key vs single-key at equal length and temperature

| temperature | scored tokens | single-key detected | dual-key detected | single median p | dual median p | p ratio |
|---|---|---:|---:|---:|---:|---:|
| 0.7 | ~296 | 12/12 | 11/12 | 7.06e-10 | 1.87e-06 | 2644x |
| 0.7 | ~500 | 12/12 | 12/12 | 8.95e-14 | 3.33e-12 | 37x |
| 1.0 | ~296 | 12/12 | 13/13 | 6.04e-19 | 6.35e-15 | 10509x |
| 1.0 | ~500 | 12/12 | 11/11 | 2.53e-29 | 3.03e-27 | 120x |

**Reading.** Dual-key carries a systematically weaker signal than single-key at
the same length and temperature. Where true-positive rate saturates the effect is
visible in the p-value: at ~500 scored tokens the dual-key median p is roughly two
orders of magnitude larger, and at ~296 tokens three to four orders larger. At
temperature 0.7 that difference costs real detections: 11/12 for dual-key against
12/12 for single-key at ~296 scored tokens, with the median p moving from 7.06e-10
to 1.87e-06, much closer to the 0.01 threshold. Dual-key at temperature 0.7 and
~133 scored tokens detected 5/12.

This is the expected direction. Splitting generation across two key streams and
recombining them with weighted early fusion spreads the evidence across two
statistics instead of concentrating it in one, so the per-token evidence is lower
even though the scheme remains non-distortionary. The practical consequence is
that dual-key needs **more tokens, or more entropy, than single-key** to reach the
same confidence, and the margin it loses is exactly the margin that matters near
the threshold.

### Generation/detection alpha pairing (36 identical dual-key completions, all
generated at alpha 0.1, re-scored by each detector — a paired comparison on the
same texts)

| detector | scored tokens | n | detected | TPR | median p |
|---|---|---:|---:|---:|---:|
| alpha 0.1 (matches generation) | ~133 | 12 | 5 | 0.417 | 3.57e-02 |
| alpha 0.1 (matches generation) | ~296 | 12 | 11 | 0.917 | 1.87e-06 |
| alpha 0.1 (matches generation) | ~496 | 12 | 12 | 1.000 | 3.33e-12 |
| alpha 0.2 (upstream detector default) | ~133 | 12 | 5 | 0.417 | 3.24e-02 |
| alpha 0.2 (upstream detector default) | ~296 | 12 | 11 | 0.917 | 1.10e-06 |
| alpha 0.2 (upstream detector default) | ~496 | 12 | 12 | 1.000 | 1.08e-11 |
| alpha 0.5 | ~133 | 12 | 2 | 0.167 | 7.13e-02 |
| alpha 0.5 | ~296 | 12 | 12 | 1.000 | 3.84e-04 |
| alpha 0.5 | ~496 | 12 | 12 | 1.000 | 3.62e-08 |
| single-key detector (key A only) | ~133 | 12 | 1 | 0.083 | 6.66e-01 |
| single-key detector (key A only) | ~296 | 12 | 0 | 0.000 | 6.94e-01 |
| single-key detector (key A only) | ~496 | 12 | 0 | 0.000 | 5.62e-01 |

**Three findings.**

1. **The default alpha asymmetry is benign at these values.** Upstream's generator
   defaults to `alpha=0.1` while `DualKeyGumbelWatermarkDetector` defaults to
   `alpha=0.2`, so a deployer accepting both defaults detects with twice the key-B
   weight actually used. Measured on identical texts, this changed nothing: TPR was
   identical in every band (0.417 / 0.917 / 1.000) and median p-values moved by
   well under an order of magnitude. The asymmetry is worth documenting but is not
   a practical hazard at the defaults.
2. **A badly chosen alpha does cost detection.** At `alpha=0.5` the short band fell
   from 5/12 to 2/12. Weighting key B far above its true share dilutes the dominant
   key-A evidence.
3. **Detecting dual-key text with the single-key detector fails silently and
   completely** — 1/12, 0/12, 0/12, with median p near 0.6 even at ~496 scored
   tokens. This is not a subtle degradation: dual-key derives its two keys from the
   configured key by SHA-256 domain separation, so the single-key detector is
   scoring with an unrelated key and sees noise. It returns a normal, confident
   "not watermarked" verdict with no error. **Operationally this is the most
   dangerous configuration mistake available in this feature**, because it is
   indistinguishable from "this text was never watermarked".

**Boundaries.** One model (7B), one GPU (L40S), one prompt set, n=12 per cell,
generation alpha 0.1 only, no speculative decoding (the setting upstream's own
dual-key numbers were taken under). Speculative decoding is where dual-key earns
its keep, and its interaction with detection power at low temperature remains
unmeasured here.
