# Implementation plan

Phased, each with acceptance criteria. Work phases **in order** — each closes a gap listed in [`facts.md`](facts.md) §D. Log every run (command, environment, raw output) in `EXPERIMENTS.md` at repo root (create on first run). Update `facts.md` verification tags in the same commit as the evidence.

**Algorithm sources (license-clean; `OFFICIAL-SRC`):** `transformers` KGW +
SynthID-Text implementations and `google-deepmind/synthid-text` (Apache-2.0),
MarkLLM for reference (Apache-2.0). **Never copy from
`eth-sri/unified-watermarking` (no license).** Sources: [facts B11, B13, B15,
and B16](facts.md) and the per-file attribution headers in
[`src/vllm_watermark/`](../src/vllm_watermark/).

**Environment (`EXECUTED` cluster lifecycle / `STATIC` local constraint):**
OpenShift 4.20 cluster `ocp-ai` ([cluster record](cluster.md)); the GPU node is
billable, so use `./scripts/scale-gpu.sh 1` only for execution and
`./scripts/scale-gpu.sh 0` when done. The local workstation cannot run vLLM
(Python 3.14, no GPU), so vLLM execution happens in cluster containers/pods
([current GPU lifecycle evidence](../EXPERIMENTS.md#2026-08-09--phase-4-current-managed-path-and-d10-continuous-validation-executed-redacted)).

<a id="delivery-target"></a>
## Delivery target — RHOAI serving and continuous validation

**Status: the single-replica synchronous PoC target is met (`EXECUTED`, facts
C8/D5/D9/D10).** The RHOAI ServingRuntime/InferenceService/internal-predictor path,
the current metadata-only managed-NeMo/broker/detector path, and the exact `N=1` and
`N=5` acceptance matrix executed on 2026-08-09 and reran through the current
detector image ([current build-5 matrix](../EXPERIMENTS.md#current-build5-d10-mode-evidence-2026-08-09)).
The destination is watermark-enabled vLLM running through an actual OpenShift AI
ServingRuntime/InferenceService path. Selected generated responses must be
validated by the TrustyAI-compatible detector through the current RHOAI-managed
guardrails path. That current internal path is now executed; external KServe/Istio
pass-through, product supportability, multi-replica/streaming behavior, production
network policy, and platform-wide retention remain `OPEN` (facts C8/D5/D6/D10).
Standalone FMS is executed legacy evidence; the separate executed
upstream NeMo 0.23.0 PoC does not establish the managed RHOAI path
(`EXECUTED` / `OFFICIAL-SRC` / `OPEN`; facts C11/D5/D6).
“TrustyAI-compatible” describes the detector service's executed legacy-FMS API
surface; it does not mean that managed NeMo natively speaks that contract. The
executed D10 configuration uses an authenticated internal broker adapter: managed
NeMo carries bounded correlation metadata only, while the gateway-held pending
response is the detector's exact-content authority (`EXECUTED`; D5/D10 and the
[current build-5 matrix](../EXPERIMENTS.md#current-build5-d10-mode-evidence-2026-08-09)).

### Executed sampling contract

`validation_sample_every=N` is a strict positive-integer setting (`EXECUTED`, D10;
[configuration and fixed-run evidence](../EXPERIMENTS.md#current-build5-d10-mode-evidence-2026-08-09)).
For this contract, one response means one successfully completed generated-text
result. The executed gateway is non-streaming and accepts one choice. A non-streaming
result counts once when complete. Requests cancelled or failed before a generated-
text result are tracked separately and do not consume an ordinal. At one
defined validation sampler, eligible response ordinals start at 1 and an ordinal is
selected when it is divisible by `N`. Thus `N=1` selects every completed
generated-text response, while `N=5` selects responses 5, 10, 15, and so on. The
fixed runs produced exactly 20/20 selected responses at `N=1` and 20/100 at `N=5`;
retries did not increase the unique selected-response count (`EXECUTED`; D10 and the
[current build-5 matrix](../EXPERIMENTS.md#current-build5-d10-mode-evidence-2026-08-09)).
Streaming, asynchronous delivery, and ordinal coordination across replicas,
restarts, and rollouts remain `OPEN` (D10).

### Validation modes and failure policy

The synchronous row is executed for the current one-replica, non-streaming gateway;
the asynchronous alternative remains design analysis only (`EXECUTED` / `OPEN`;
D10 and the [current build-5 matrix](../EXPERIMENTS.md#current-build5-d10-mode-evidence-2026-08-09)):

| Mode | Response flow | Latency, availability, and failure consequence |
|---|---|---|
| Synchronous blocking (`EXECUTED`) | Hold each selected generated response until detector and managed-guardrails validation reaches a terminal result. Unsampled responses bypass this watermark-validation step under the configured sampling rule. | The executed policy failed closed on validation error, returned a content-free 503 after the real detector outage, and measured selected-response validation plus gateway response-ready latency. The retained `client_delivery` field name does not include FastAPI serialization or network/client receipt (`STATIC`; [`_deliver_response`](../validation/gateway.py)). Positive detector results were flagged for gateway delivery while the managed action reported `blocked`; those are distinct layers. |
| Asynchronous mirroring (`OPEN`) | Deliver the vLLM response, then enqueue or mirror a copy for detector and managed-guardrails validation. | Avoids detector latency on the original response, but cannot block a response already delivered and is therefore fail-open for that delivery. Validation lag, bounded queue capacity, retry exhaustion, overload behavior, and any dropped or quarantined item require explicit policy, metrics, and retention review. |

### Cross-cutting requirements

These requirements are met for the executed synchronous single-replica scope, with
the expressly stated broader boundaries still `OPEN` (`EXECUTED` / `OPEN`; D10 and
the [current build-5 matrix](../EXPERIMENTS.md#current-build5-d10-mode-evidence-2026-08-09)):

- Propagate the selected `watermark_scheme` and non-secret
  `watermark_key_id` consistently from generation metadata to the detector and
  guardrails request. Secret key material remains server-side in mounted Secrets;
  it must never enter request bodies, logs, traces, or metric labels. Propagation and
  the finite six-secret scan are `EXECUTED`; key lifecycle/application scoping remain
  `OPEN` under D4.
- Application observability is hash-only for generated content: correlate by a
  request identifier and content hash/digest plus scheme, key ID, verdict, mode,
  attempt count, and timing. Do not log plaintext responses or keys. The
  eight-surface zero-match result is finite and scoped (`EXECUTED`); platform-wide
  retention remains `OPEN` under D5 ([evidence](../EXPERIMENTS.md#2026-08-09--phase-4-current-managed-path-and-d10-continuous-validation-executed-redacted)).
- Bound retry count, elapsed retry time, and concurrency; distinguish retryable
  transport failures from terminal schema/verdict failures; and preserve one
  idempotent validation record per selected response.
- Bound synchronous in-flight work and asynchronous queue depth. When capacity is
  exhausted, apply the configured blocking/fail-open/fail-closed/drop policy and
  increment a reason-labelled metric; do not silently lose validation work.
- Expose at least response, selected, attempt, terminal watermarked/clean/error,
  retry, fail-open delivery, fail-closed block, queue-depth, dropped-item,
  validation-lag, validation-latency, generation-completion latency, and
  gateway response-ready latency metrics (currently exposed under the legacy
  `client_delivery` name; client-observed network delivery is `OPEN`; [latency
  correction](../EXPERIMENTS.md#latency-semantics-correction-2026-08-09)).
  Metric labels must stay bounded and must not contain response text, hashes,
  secret material, or unconstrained request/key identifiers. Track failed and
  cancelled requests separately from completed generated-text responses.

<a id="continuous-validation-acceptance"></a>
### Required acceptance evidence

Items 1–8 below were executed through the actual RHOAI endpoint and their command plus
content-redacted output is preserved in `EXPERIMENTS.md` (`EXECUTED`, single-replica
synchronous scope; D10; [complete current-build report](../EXPERIMENTS.md#current-build5-d10-mode-evidence-2026-08-09)).

For this matrix, an expected watermarked result means a positive matching-scheme
detector verdict plus the configured managed-guardrails block/flag outcome; an
expected clean result means no detection plus the configured pass outcome. Record
the exact version-specific response shapes rather than assuming the legacy FMS or
standalone upstream-NeMo envelope. The exact managed `nemoguardrails==0.21.0` shape
and broker correlation were recorded (`EXECUTED`; D5/D10 and the linked evidence).

1. Validate configuration before serving: accept positive integers including `1`
   and `5`; reject `0`, `-1`, `1.5`, an empty value, and a non-numeric value. An
   invalid setting must fail configuration/startup or readiness rather than silently
   changing the sampling rate.
2. Reset one defined sampler and send a fixed, concurrency-1 20-response run with `N=1`:
   exactly 20 unique responses must be selected and reach terminal validation.
   Pre-register five KGW-enabled, five SynthID-enabled, five KGW-disabled/clean,
   and five SynthID-disabled/clean cases. Every case, including a disabled case,
   carries a pre-registered expected scheme and non-secret key ID so the detector
   evaluates the clean text under the matching configuration. Require 20/20
   expected detector verdicts and matching metadata. Require 20 unique managed-
   guardrails inputs/actions and 20 correlated detector calls/results: the 10
   positive detector results map to 10 configured block/flag actions, and the 10
   clean results map to 10 pass actions. With no injected failure, each component
   has exactly 20 attempts. Require counters of 20 responses, 20 selected, 0
   unsampled, 20 terminal, 10 watermarked, 10 clean, and 0 errors. The disabled
   cases are controlled negative probes, not a proposed production default.
3. Reset the same sampler and send a fixed, concurrency-1 100-response run with `N=5`:
   exactly 20 unique ordinals (5, 10, …, 100) must be selected. Arrange those
   selected positions as five KGW-enabled, five SynthID-enabled, five
   KGW-disabled/clean, and five SynthID-disabled/clean cases; require 20/20 expected
   verdicts with the same explicit scheme/key-ID and detector-to-guardrails mapping
   as item 2. Require exactly 20 unique managed-guardrails inputs/actions and 20
   correlated detector calls/results, with 20 attempts at each component in the
   no-error run. Require counters of 100 responses, 20 selected, 80 unsampled, 20
   terminal, 10 watermarked, 10 clean, and 0 errors. Retry attempts may exceed 20
   only in separate fault runs; unique selected and terminal-result counts remain
   one per selected response.
4. For the chosen delivery mode, record unsampled baseline and `N=1`/`N=5`
   generation-completion latency, gateway response-ready latency for responses
   constructed for delivery, and validation latency/lag at p50, p95, and p99.
   Record client-observed latency separately if it is required. If both modes remain
   candidates, run both; otherwise record why the other was rejected.
5. Run isolated fault cases with `N=1`, `max_attempts=3`, and one selected response
   unless noted: (a) one retryable failure then success must produce 2 attempts, 1
   retry, 1 terminal success, and 1 unique record; (b) connection refusal/timeout
   through all attempts must produce 3 attempts, 2 retries, 1 retry-exhausted
   terminal error, and exactly one fail-open delivery or fail-closed block according
   to the chosen synchronous policy (asynchronous delivery records exactly one
   fail-open outcome); and (c) a malformed terminal response must produce 1
   attempt, 0 retries, 1 terminal error, and exactly one configured failure-policy
   outcome. In every case, selected-response uniqueness remains 1.
6. Set the bounded queue/in-flight capacity to 2, pause its consumer, and submit 3
   selected responses with `N=1`. For a non-blocking overflow policy, require peak
   depth 2, exactly 1 overflow counter, 2 accepted items that later reach detector
   verdicts, and 1 explicit dropped/error/fail-open/fail-closed terminal record with
   no detector verdict invented for the unaccepted item. For a blocking-backpressure
   policy, require the third submission to remain pending while capacity is full,
   zero overflow/drop outcomes, and all 3 items to be accepted and reach detector
   verdicts after the consumer resumes. In either case require exactly 3 unique
   terminal records and no missing or duplicate selected response.
7. For every selected response, require one validation record containing its
   response ID, content digest, expected scheme/key ID, verdict, mode, attempts,
   and timing; recompute the digest and correlate the same response ID across the
   managed-guardrails and detector records. Scan defined logs, traces/events, and
   metrics exposition for distinctive response substrings and secret material and
   require zero matches. Hashes may appear in validation records/logs but not metric
   labels. Inspect the metrics schema for the required series and a bounded label
   allowlist. In the no-error fixed runs, require generation-completion-latency
   sample count equal to completed generated responses,
   gateway-response-ready-latency (`client_delivery`) sample count equal to
   responses constructed for delivery, and
   validation-latency/lag sample count equal to selected responses. Record the
   applicable counts separately in fault and overload runs, where fail-closed blocks
   or drops can make delivered and completed-generated counts differ. Require final
   queue depth 0 after each run.
8. Reconcile the no-error fixed-run metrics: `started = completed = 20`,
   `completed = selected + unsampled = 20 + 0`, and `selected = terminal = 20`
   for `N=1`; then `started = completed = 100`,
   `completed = selected + unsampled = 20 + 80`, and
   `selected = terminal = 20` for `N=5`. Failed/cancelled counters must be 0 in
   these runs; attempt counters must match the per-component counts above.

---

## Phase 0 — Baseline serving

Stand up plain vLLM on the cluster and capture a performance baseline.

**Status: acceptance met.** The bare-OpenShift vLLM 0.18.0 endpoint completed 100/100 benchmark requests and recorded the baseline throughput and latency (`EXECUTED`; [fact C9](facts.md), [raw run](../EXPERIMENTS.md#2026-08-08--phase-0-baseline-serving--benchmark-executed)). This was not an RHOAI ServingRuntime deployment.

- Install OpenShift AI (RHOAI 3.4.x) or, for the first spike, run the vLLM container directly in a pod on the GPU node. Record exact image digest and vLLM version.
- Serve a small open model (e.g. `Qwen/Qwen2.5-0.5B-Instruct` or a Llama 3.x 1B–8B variant that fits a g5.xlarge / A10G 24GB).
- Baseline: tokens/sec and p50/p95 latency for a fixed prompt set (script it; keep the script in `benchmarks/`).

**Accept when:** OpenAI-compatible endpoint answers; baseline numbers recorded in `EXPERIMENTS.md`.

## Phase 1 — KGW watermark logits processor under `vllm serve` (closes D1)

**Status: acceptance met; D1 closed.** Corrected single-instance KGW generation and detection ran through `vllm serve`, with the required controls and overhead measurement (`EXECUTED`; [fact D1](facts.md), [correction and rerun](../EXPERIMENTS.md#2026-08-08--phase-1-corrected--phase-2-synthid-through-vllm-serve-closes-d8)). The earlier double-loaded KGW signal and active-path overhead figures are superseded.

Build `src/vllm_watermark/` as a pip-installable package:

**Implementation status (`STATIC` source / `EXECUTED` serving behavior):** the
components below are implemented in [`src/vllm_watermark/`](../src/vllm_watermark/)
and their KGW behavior is established by [fact D1](facts.md) and the
[corrected run](../EXPERIMENTS.md#2026-08-08--phase-1-corrected--phase-2-synthid-through-vllm-serve-closes-d8).

- A V1 `LogitsProcessor` subclass implementing KGW green-list biasing, ported from `transformers`' implementation (Apache-2.0). `is_argmax_invariant()` → `False`. Keyed hashing seeded from a secret **read from env/mounted Secret — never hardcoded, never logged**.
- Entry points in group `vllm.logits_processors`; an explicit `--logits-processors` FQCN is an alternative loading mode, never an addition to this wheel's entry points (the combined form caused the executed double-load correction in D1).
- Per-request control via `vllm_xargs` (e.g. `watermark: on/off`, `watermark_key_id`) with `validate_params()` rejecting malformed args.
- A detector CLI/module using the same key(s) (port of `transformers` `WatermarkDetector` logic), independent of vLLM.

Test protocol (all generated samples through the **OpenAI-compatible server**, not offline `LLM()`):
- ≥100 watermarked and ≥100 unwatermarked generations at ~256 tokens, temperature 0.7, plus a separately sourced ≥100-sample human-text corpus.
- Report z-score distributions, TPR at the z≥4 threshold, FPR on human corpus.
- Throughput/latency vs Phase 0 baseline (same prompts, same settings).
- Negative tests: temperature 0 (measure rather than assume; the recorded KGW run is an `EXECUTED` exception to the general degradation expectation in fact B18), structured-output request (expect composition per docs/technical.md §1 ordering), spec-decode flag (expect the documented startup error, verbatim).

**Accept when:** threshold-relative watermarked and control distributions are demonstrated end to end through `vllm serve`; overhead is quantified; all results and commands are in `EXPERIMENTS.md`; and D1 is updated to `EXECUTED`.

## Phase 2 — SynthID-Text

**Status: acceptance met; D8 closed.** SynthID generation and untrained detection ran through `vllm serve`; the recorded 200/256/512-token comparison and corrected overhead are `EXECUTED` ([fact D8](facts.md), [scheme comparison v2](../EXPERIMENTS.md#2026-08-08--scheme-comparison-v2-per-scheme-control-fpr-supersedes-the-v1-tables-control-rows)). This one-model proof does not establish general reliability or production readiness.

**Implementation status (`STATIC` source / `EXECUTED` serving behavior):** the
components below are implemented in
[`src/vllm_watermark/synthid/`](../src/vllm_watermark/synthid/) with Apache-2.0
attribution, and their recorded behavior is established by [fact D8](facts.md) and
the linked scheme-comparison evidence.

- Second `LogitsProcessor` implementing SynthID-Text tournament sampling, ported from `transformers`/`google-deepmind/synthid-text` (Apache-2.0). Note: SynthID interacts with the *sampling* step differently than pure logit-bias schemes — validate the logits-processor formulation against the reference implementation's outputs on identical seeds before trusting it.
- Detection: start with the untrained weighted-mean scorer; measure. Then decide whether Bayesian-detector training (~10k matched examples — generate them with the Phase 1 harness) is warranted; if trained, version the detector artifact with the exact generation config it matches.
- Same test protocol as Phase 1; add a KGW-vs-SynthID comparison table (detectability at 200/256/512 tokens, quality spot-check, overhead).

**Accept when:** SynthID generation and detection work through `vllm serve` with a quantified result at the 200-token length named in the Code (`OJ-VERBATIM`, [quotes](quotes.md#cop-measure-1-1)); the comparison table is in `EXPERIMENTS.md`; and D8 is closed. The legal wording is not itself a reliability threshold.

## Phase 3 — Detection service + current guardrails-path confirmation (addresses D5)

- The detector service exposes the historical FMS contract, `POST /api/v1/text/contents`, and a direct endpoint returning `{z_score, p_value, verdict, key_id, detector_version}` with an Ed25519-JWS-signed result. Both were exercised on OpenShift (`EXECUTED`, fact D5 and the [Phase 3 experiment](../EXPERIMENTS.md#2026-08-08--phase-3-detector-service--fms-guardrailsorchestrator-end-to-end-closes-d5s-executable-half)). Signing is an engineering feature here, not a claimed legal requirement (`OPEN`).
- The FMS Guardrails Orchestrator routed KGW and SynthID requests to the detector, and an upstream `nemoguardrails==0.23.0` custom output-rail action called it (`EXECUTED`, fact D5 and the "NeMo PoC hardening evidence" transcript in the [append-only evidence log](../EXPERIMENTS.md)). Malformed-200 fail-closed behavior executed in that earlier upstream-NeMo scope. The current managed-NeMo/broker path also executed a real detector outage: three attempts were exhausted and the synchronous gateway returned a content-free 503 (`EXECUTED`, fact D5 and the [current-path evidence](../EXPERIMENTS.md#2026-08-09--phase-4-current-managed-path-and-d10-continuous-validation-executed-redacted)). Platform-wide outage behavior remains `OPEN`.
- RHOAI 3.4 labels FMS Guardrails legacy and directs users to NeMo Guardrails (`OFFICIAL-SRC`, fact C11). The RHOAI-managed `NemoGuardrails` custom-resource path, shipped image, former fixed-action block/pass behavior, and controlled detector-outage fail-closed result executed; the finite retention scan was also executed, but platform-wide retention and supportability remain `OPEN` (`EXECUTED` / `OPEN`, facts C11/D5/D6; [recovered evidence](../EXPERIMENTS.md#2026-08-08--phase-4-rhoai-exact-transcript-recovered-executed-redacted)).
- The detector is designed to avoid storing submitted content and its application logs contain hashes plus verdict metadata; the executed evidence is scoped to the detector logs inspected in Phase 3 (`STATIC` / `EXECUTED`, fact D5 and the [Phase 3 experiment](../EXPERIMENTS.md#2026-08-08--phase-3-detector-service--fms-guardrailsorchestrator-end-to-end-closes-d5s-executable-half)). End-to-end retention behavior and mitigation remain `OPEN` under D5; this engineering evidence makes no legal conclusion.

**Acceptance status:** the FMS/upstream-NeMo executable half, the former managed
fixed-action path, and the later current metadata-only broker path are met in their
recorded scopes (`EXECUTED`, facts D5/D10; [current-path evidence](../EXPERIMENTS.md#2026-08-09--phase-4-current-managed-path-and-d10-continuous-validation-executed-redacted)).
This does not establish Red Hat supportability, platform-wide retention behavior, or
external gateway pass-through (`OPEN`, facts C8/D5/D6).

## Phase 4 — RHOAI deployment pattern (scopes C8, informs D6)

**Status: acceptance met for the scoped internal RHOAI path (`EXECUTED`; `OPEN`
production boundaries).** RHOAI 3.4.2 operator/DSC state, the current custom vLLM
image, ServingRuntime, InferenceService/internal predictor, and the current
metadata-only managed-NeMo/broker/detector block/pass/fail-closed flow are recorded
as executed. External KServe gateway/Istio pass-through and product supportability
remain open (facts C8/D5/D6/D10; [current-path evidence](../EXPERIMENTS.md#2026-08-09--phase-4-current-managed-path-and-d10-continuous-validation-executed-redacted)).

- Build the custom runtime image: Red Hat vLLM base + `pip install` of the package. The recorded build pinned the base digest and produced the derived digest in `deploy/` (`EXECUTED`; fact C8 and [recovered evidence](../EXPERIMENTS.md#2026-08-08--phase-4-rhoai-exact-transcript-recovered-executed-redacted)).
- Duplicate the vLLM ServingRuntime → custom image; deploy via InferenceService with the package's entry-point loading (do not also pass the same processors through `--logits-processors`); mount the key from a Secret (`STATIC`; facts B4/C3 and the executed double-load correction in D1).
- The internal predictor Service answered health/model probes and four direct 256-token
  `vllm_xargs` cases produced the recorded KGW/SynthID positive and clean-control
  detector results (`EXECUTED`; fact C8 and [recovered evidence](../EXPERIMENTS.md#2026-08-08--phase-4-rhoai-exact-transcript-recovered-executed-redacted)). External
  KServe gateway/Istio pass-through remains unexecuted (`OPEN`; C8).
- The current managed-NeMo action propagated bounded response/validation correlation
  metadata to the authenticated internal broker. The broker validated the exact
  pending gateway response through the detector; KGW/SynthID positive cases mapped
  to managed `blocked`, clean controls to `success`, and a real detector outage
  exhausted three attempts before the gateway failed closed (`EXECUTED`; D5/D10 and
  [current-path evidence](../EXPERIMENTS.md#2026-08-09--phase-4-current-managed-path-and-d10-continuous-validation-executed-redacted)).
- The chosen initial mode is synchronous, non-streaming, single-replica validation;
  one-in-`N`, delivery/failure policy, retry, backpressure, metric, and correlation
  acceptance all executed. Asynchronous, streaming, and multi-replica behavior remain
  `OPEN` (`EXECUTED` / `OPEN`; D10 and the linked evidence).
- Document the supportability posture in the Phase 4 deployment record. D6 requires a
  product-management/support decision and cannot be closed by engineering execution
  (`OPEN`; D6).

**Acceptance status:** the RHOAI generation/internal-predictor and current managed
metadata-only broker block/pass/fail-closed path are preserved (`EXECUTED`, facts
C8/D5/D10 and the linked current-path evidence). External gateway pass-through and
supportability remain `OPEN` (facts C8/D6); they are explicit production boundaries,
not unexecuted parts of the internal Phase 4 acceptance path.

## Phase 5 — Benchmarks, robustness, hardening

**Status: partially complete.** Detector fail-fast hardening (D9) and the defined D10
single-replica synchronous acceptance matrix are complete (`EXECUTED`); D2 remains
partially closed, while D3, D4, the D10 production boundaries, and the broader
robustness/key-management work remain `OPEN` ([facts D2–D4 and D9–D10](facts.md);
[current-path evidence](../EXPERIMENTS.md#2026-08-09--phase-4-current-managed-path-and-d10-continuous-validation-executed-redacted);
[current detector reconciliation](../EXPERIMENTS.md#current-detector-reconciliation-2026-08-09)).

- GPU overhead at realistic batch sizes (closes D2); tensor-parallel correctness if a multi-GPU node is available (D3 — identical key/seed across ranks).
- Robustness: quantify paraphrase and translation attacks on the repository's own outputs and compare them with the cited literature without assuming the published rates transfer (see `technical.md` §3).
- Key management design doc (D4): generation, storage (Vault/Secrets), rotation, per-app `key_id` scoping, compromise runbook.
- Detector fail-fast configuration validation (D9): lower, upper, non-finite,
  tokenizer-fallback, and empty-effective-KGW-green-list cases are rejected through
  lifespan. The current immutable image matched local source, passed the built-image
  explicit-blank and 9/9 maximum/overflow probes, failed a real blank-valued rollout
  before readiness, recovered, and answered both scheme API smoke requests
  (`EXECUTED`; fact D9 and the linked current detector reconciliation). The complete
  D10 fixed matrix subsequently reran through this immutable digest with all 40
  mode-bearing, hash-only selected-response records retained (`EXECUTED`; [current build-5
  matrix](../EXPERIMENTS.md#current-build5-d10-mode-evidence-2026-08-09)).
- Continuous validation (D10): the configured one-in-`N` selector, selected-response
  validation flow, explicit delivery/failure policy, key/scheme propagation,
  hash-only observability, retries, backpressure, metrics, and the
  [fixed-run acceptance matrix](#continuous-validation-acceptance) executed for the
  stated synchronous single-replica scope (`EXECUTED`; fact D10 and the [current
  build-5 matrix](../EXPERIMENTS.md#current-build5-d10-mode-evidence-2026-08-09)).
  Production boundaries listed above remain `OPEN`.
- Post-execution adversarial follow-ups remain `OPEN`: define and exercise component
  attempt-counter semantics for pre-action transport/schema and injected-fault paths;
  add a byte-bounded broker response read to the embedded managed-NeMo action; and add
  concrete transport/mismatch/cancellation regression coverage (`OPEN`; [review
  follow-ups](../EXPERIMENTS.md#2026-08-09--post-execution-adversarial-review-follow-ups-staticopen)).
- Detector/generator resource hardening is implemented in the current local source:
  direct requests, tokenized batches, KGW caches, SynthID tables/matrices/context,
  and generation-side configurations have explicit ceilings (`STATIC`), with local
  regression and fuzz/stress execution recorded in [fact FZ1](facts.md). The changed
  runtime image has not been rebuilt or exercised on the cluster, and production
  workload sizing, GPU profiling, and broader adversarial robustness remain `OPEN`.
- Compliance mapping doc: map the exact quoted Code language and sources in [`docs/quotes.md`](quotes.md) to what this implementation does (`OJ-VERBATIM` for the source text; implementation status tagged separately). The `EXPERIMENTS.md` log may contribute to required documentation only if it meets the applicable requirements; keep it audit-grade. **First version delivered 2026-08-11:** [`docs/compliance-map.md`](compliance-map.md) (obligation → status → evidence → verification command), backed by the executed local re-verification harness `scripts/verify-claims.py` (fact B25). Keep the map current with `facts.md` in the same commit as any status change; independent review of the map is still pending.

**Acceptance status:** the D9 and D10 portions are met with command/raw evidence
(`EXECUTED`; facts D9/D10 and the linked current-path and detector-rebuild records).
Phase 5 as a broader
hardening phase remains incomplete until the remaining overhead/robustness,
tensor-parallel, key-management, and compliance-mapping work is reviewed
(`OPEN`; D2/D3/D4).

---

## Out of scope (engineering cannot close)

- Legal determination of grace-period applicability to internal-only systems (D7 — counsel).
- Red Hat support-policy carve-out and roadmap statements (D6/C6 — product management).
