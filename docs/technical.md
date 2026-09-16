# Technical landscape

Verification tags per [`facts.md`](facts.md). Fetch dates: 2026-08-07/08; vendor-landscape addendum fetched 2026-08-13 (facts B26/B27).

## 1. vLLM's extension point

vLLM's V1 engine exposes a documented plugin API for custom logits processors — batch-level processors that adjust the logits tensor after the model forward pass and before sampling (`OFFICIAL-SRC`, facts B3–B5). Using that hook for decode-time watermark bias without modifying vLLM source is the repository's `STATIC` integration design.

**Interface** (`vllm.v1.sample.logits_processor.LogitsProcessor`):

| Method | Role |
|---|---|
| `validate_params(cls, sampling_params)` | Reject bad per-request args at request time |
| `__init__(self, vllm_config, device, is_pin_memory)` | Engine-init construction |
| `apply(self, logits) -> torch.Tensor` | The hot path — adjust batch logits |
| `is_argmax_invariant(self) -> bool` | Watermarking changes argmax → must return `False` |
| `update_state(self, batch_update)` | Track per-request state across batch Add/Remove/Move |

**Loading:** `--logits-processors <FQCN>` on `vllm serve` / `LLM()`, or a setuptools entry point in group `vllm.logits_processors` (auto-discovered — a pip-installed package is enough). The processor set is immutable after engine init (`OFFICIAL-SRC`, fact B4).

**Per-request control:** `SamplingParams.extra_args` offline; `vllm_xargs` in the REST body (OpenAI SDK: `extra_body={"vllm_xargs": {...}}`) online. This is how per-request key identifiers and enable/disable values reach the processor (`OFFICIAL-SRC`, B4; `EXECUTED`, D1).

**Version facts** `OFFICIAL-SRC`:

| Event | Version | Ref |
|---|---|---|
| Extension point landed | v0.10.1 | PR #19912, design RFC #17799 |
| Docs page added | 2025-09-17 | PR #22919 |
| V0 per-request callable (`SamplingParams.logits_processors`) **removed** | v0.17.0 | PR #34400 |
| Current release (as of 2026-08-07) | v0.26.0 (2026-07-27) | pins torch==2.11.0, transformers>=5.5.3 |

**Constraints:**

- **Speculative decoding: hard-incompatible in the recorded stack.** Engine-config validation raises `"Custom logits processors are not supported when speculative decoding is enabled."` The source restriction is `OFFICIAL-SRC`; the startup rejection was also `EXECUTED` on vLLM 0.18.0 ([fact B7](facts.md), [raw run](../EXPERIMENTS.md#spec-decode-incompatibility-executed--b7-upgraded)). Opt-in support remained an open PR at the recorded 2026-08-08 review.
- **Model Runner V2** doesn't support custom processors yet; silently falls back to the V1 runner (PR #47585 open, active 2026-08-04). Architectural churn is real — re-verify on every vLLM upgrade. `OFFICIAL-SRC`
- **Ordering:** structured-output grammar bitmasks are applied to logits *before* logits processors run (`STATIC`). Guided JSON and KGW produced valid, detectable output in 8/8 requests (`EXECUTED`), but the recorded mean z=14.5 came from the superseded double-instance/delta≈4 window; the true-delta-2 magnitude remains `OPEN` ([fact B9](facts.md)).
- **Tensor-parallel / prefix caching / chunked prefill:** no documentation or executed evidence was registered. `OPEN` — Phase 5 ([facts B10 and D3](facts.md)).

**No upstream content-watermarking work was identified in the recorded 2026-08-07 searches.** The searched issues, pull requests, code, and discussions returned only unrelated hits, and `vllm-project/rfcs` was empty at that time. This is a dated, search-bounded negative (`OFFICIAL-SRC` for searched records / `OPEN` for absence; fact B2).

<a id="what-changes-where"></a>
### What changes in the current design

The model weights, files, and checkpoints stay untouched — no fine-tuning, no conversion. Decode-time watermarking hooks the serving layer after the model computes logits and before the sampler picks a token, so the extension surface is not tied to one model architecture. That is a `STATIC` compatibility inference, not evidence that watermark correctness, quality, and detectability are identical across every model; those require per-model execution. The integration surface is three pieces:

**1. A package available to the runtime.** The watermark logits processor is a pip-installable Python package that must be importable inside the vLLM container (`STATIC`; current package layout). Duplicating a ServingRuntime with a custom image is a documented RHOAI pattern (`OFFICIAL-SRC`, fact C3), and the internal RHOAI ServingRuntime/InferenceService predictor path executed with the package installed in the pinned image (`EXECUTED`; [current Phase 4/D10 evidence](../EXPERIMENTS.md#2026-08-09--phase-4-current-managed-path-and-d10-continuous-validation-executed-redacted)). Supportability and external KServe/Istio pass-through remain `OPEN` (same evidence; facts C4/C8/D6).

**2. vLLM launch config.** This package registers both processors through
`vllm.logits_processors` entry points, so the corrected deployment passes no
`--logits-processors` flag (`STATIC`; `pyproject.toml` and the Phase 4 runtime
manifest):

Keep the normal model/runtime arguments, but do not add a
`--logits-processors` item for this wheel.

Using an FQCN such as `vllm_watermark.kgw.processor:KGWLogitsProcessor` is an alternative for a package without the matching entry point (`OFFICIAL-SRC`, fact B4). Combining that flag with this installed wheel double-loaded KGW in vLLM 0.18.0; the single-instance correction used entry points only (`EXECUTED`; [correction](../EXPERIMENTS.md#2026-08-08--correction-phase-1-ran-two-kgw-processor-instances-effective-delta-40)). The watermark key is mounted from a Secret into an environment variable (`STATIC`; manifests). Speculative decoding cannot be enabled alongside the custom processors in the recorded stack (`EXECUTED`, B7).

**3. Optionally, per-request control from the client.** With the server-wide processor loaded, individual requests can pass parameters through vLLM's `vllm_xargs` extension on its OpenAI-compatible endpoint (`OFFICIAL-SRC`, fact B4):

```python
client.chat.completions.create(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    messages=[...],
    extra_body={"vllm_xargs": {
        "watermark": "on",
        "watermark_scheme": "kgw",
        "watermark_key_id": "app-a",
    }},
)
```

If a request sends nothing, the processor applies the deployment default; the executed spike used `on` (`EXECUTED`, D1). The production default is a deployment-policy and legal-scope decision (`OPEN`). The internal predictor accepted the recorded request controls, while external KServe/Istio gateway pass-through of `vllm_xargs` remains `OPEN` (`EXECUTED` scoped / `OPEN`; [current Phase 4/D10 evidence](../EXPERIMENTS.md#2026-08-09--phase-4-current-managed-path-and-d10-continuous-validation-executed-redacted)).

**Second-order consequences:** the literature reports entropy dependence in general (`CORROBORATED`, B18), but the recorded KGW delta-2/Qwen run was an exception: temperature-0 completions retained mean z=9.91 and TPR 1.000 at the tested lengths (`EXECUTED`, B18); quality remains unmeasured. KGW's bias delta and SynthID's configuration still require quality/detectability evaluation (`OPEN`, Phase 5). The implemented detector runs outside the serving path and shares the configured watermark key (`STATIC`; detector and deployment sources).

<a id="plugin-assessments"></a>
## 2. Plugin assessments (hands-on, 2026-08-07)

The table summarizes the dated source review registered in facts B11–B16.
Cells are `STATIC`/`OFFICIAL-SRC` unless they explicitly say `EXECUTED` or
`OPEN`.

| | `eth-sri/unified-watermarking` | `dapurv5/vLLM-Watermark` | `THU-BPM/MarkLLM` | `transformers` built-ins + `google-deepmind/synthid-text` |
|---|---|---|---|---|
| Integration | Current V1 plugin API (`AdapterLogitsProcessor`); imports/signatures match v0.26.0 (`STATIC`) | Monkey-patches private `Sampler` via `llm.llm_engine.engine_core…model_runner.sampler`; needs `VLLM_ENABLE_V1_MULTIPROCESSING=0` (`STATIC`) | Research toolkit; has a vLLM demo script | HF `generate()` logits processors — not vLLM-wired, but the algorithm reference implementations |
| Server path (`vllm serve`) | Documented with `vllm_xargs`; no preserved execution evidence was found (`OPEN`) | **None found.** Offline `LLM()` only (`STATIC`) | Demo-level (`STATIC`) | n/a |
| Detection | `detect()` per scheme (six in the inspected source); a prior CPU-run summary has no preserved command/raw output, so execution and numbers are `OPEN` (B11) | Nine detector variants (`STATIC`) | Per-algorithm (`STATIC`) | KGW detection and untrained SynthID scoring were `EXECUTED` through the serving and detector paths (D1/D5/D8); the separate historical CPU report is `OPEN` (B14) |
| Maturity | Two commits, no tests/CI found, companion to arXiv 2602.06754 (`STATIC`, dated 2026-08-07) | No CI against real vLLM found; latent `Sampler.forward()` signature drift (`STATIC`) | 1,000+ stars at the dated review, Apache-2.0 | Official Apache-2.0 sources |
| License | **NO LICENSE FILE — do not copy code** (all-rights-reserved by default). Design reference only | Inconsistent: pyproject says MIT, LICENSE file + GitHub API say Apache-2.0 | Apache-2.0 | Apache-2.0 |
| Verdict | Architecture reference only; code reuse is barred by the missing license (`STATIC`, B11) | Rejected for this repository's server design (`STATIC`, B12) | Algorithm reference only (`STATIC`, B16) | Primary Apache-2.0 implementation sources (`OFFICIAL-SRC`, B13/B15) |

**Consequence for implementation:** use the repository's independently implemented V1 processors and port algorithm logic only from Apache-2.0 sources (`STATIC`; source headers and [repository licensing rules](../AGENTS.md#2-licensing-rules)). The unlicensed prior-art repository is architecture reference only; it is not execution evidence (B11).

## 3. Watermarking science — what the implementation must respect

**Schemes that matter here:**

- **KGW / green-list:** a keyed hash partitions the vocabulary per step; bias delta is added to the green set; detection uses a z-test on the green fraction. This repository deliberately measured gamma=0.25, delta=2.0, z≥4.0, and repeated-ngram deduplication (`EXECUTED`, D1); those are not all `transformers` detector defaults. The official implementation is in `transformers` (`OFFICIAL-SRC`, B13).
- **SynthID-Text:** tournament sampling with trained Bayesian and untrained scoring options. It is Apache-2.0 and available in `transformers`; the cited production study reports a 20-million-response Gemini test and thumbs-rate changes of ±0.01–0.02 percentage points (`OFFICIAL-SRC`/`CORROBORATED`, B15). The repository's untrained weighted-mean scorer is `EXECUTED` (D8).
- Gumbel/EXP, Unigram, DiPmark, SWEET, and other research algorithms are implemented in MarkLLM (`OFFICIAL-SRC`, B16); they are not current implementation targets (`STATIC`, package source).

**Production-deployment landscape (updated 2026-08-13):** Anthropic announced that supported Claude models mark output — an imperceptible text watermark plus C2PA signed provenance metadata for generated .svg/.png/.jpg files — across its API, apps, Claude Code, and cloud-partner deployments, and states it has signed the EU AI Act's Article 50(2) Code of Practice (`OFFICIAL-SRC`, B26). Press coverage corroborates worldwide (not EU-only) application and reports roughly 190 Code of Practice signatories (`CORROBORATED`, B27). No algorithm, scheme, or public detector was published at the fetch date, so the mechanism is externally unverifiable and secondary-blog mechanism descriptions must not be cited (`OPEN`, B26/B27). Consequences for this repository's landscape facts: Gemini is no longer the only identified production text-watermark deployment (B15 uniqueness superseded), while B19's dated negative for self-hosted serving stacks stands — hosted frontier providers now ship text marking, but the reviewed self-hosted stacks still expose none, which remains the gap this repository addresses (B19/B20).

**Known limitations and open questions:**

1. **Entropy dependence.** The literature reports degradation in low-entropy settings such as greedy decoding, code, JSON, and structured output (`CORROBORATED`; arXiv 2405.14604 and 2506.06409). The recorded KGW delta-2 temperature-0 result is a model/configuration-specific `EXECUTED` exception (B18), and guided-JSON composition was executed only in the superseded delta≈4 window (B9).
2. **Length dependence.** The reviewed studies report scheme- and configuration-dependent detection at different lengths (`CORROBORATED`, B17/B18). Separately, the voluntary Code's exact 200-token wording is quoted in [`quotes.md`](quotes.md#cop-measure-1-1) (`OJ-VERBATIM`); no equivalence between that legal text and a measured reliability threshold is inferred.
3. **Paraphrase/translation attacks.** One KGW study reports 99.8% → 9.7% TPR@1%FPR after five paraphrase rounds (arXiv 2303.11156); one SynthID study reports F1 1.0 → 0.71 after one Chinese back-translation (arXiv 2508.20228). A retrieval study is arXiv 2303.13408. These are literature results, not measurements of this implementation (`CORROBORATED`, B17).
4. **Outlier study.** The reviewed July 2026 preprint reports pre-attack recall of roughly 17–30% for KGW/Unigram/SynthID and 5.4% FPR for SynthID on human text (`CORROBORATED`, B17). Its small sample and wide confidence intervals limit generalization (`CORROBORATED`; arXiv 2607.16010).
5. **Key management.** Generator and detector share configured secret material (`STATIC`, implementation). Generation, storage, rotation, and per-application scoping remain undesigned (`OPEN`, D4).
6. **Multi-turn/agentic effects** such as watermarked text re-entering context or tool output remain unmeasured here; the recorded literature search found no quantified result (`OPEN`, B18).
7. **Classifier detectors are not marking.** They infer from text and do not embed a mark (`STATIC`, mechanism distinction). Whether such detection contributes to a specific legal obligation remains `OPEN` for counsel.

## 4. Historical local CPU report (`OPEN`)

The read-only report describes a CPU-only transformers 4.57.6 / torch 2.9.1 / Python 3.14.4, gpt2, 60-token run with `WatermarkingConfig(bias=2.5, seeding_scheme="selfhash")` and temperature 0.7. It includes source and console text under [`../research/demo/`](../research/demo/), but the exact invocation/raw output is absent from `EXPERIMENTS.md`; the table is therefore historical reported data, not `EXECUTED` evidence (B14).

| Input | z-score | p-value | Detector verdict |
|---|---|---|---|
| Watermarked generation | **5.37** | 5.4e-9 | AI-generated ✓ |
| Unwatermarked generation | 0.89 | 0.30 | not flagged ✓ |
| Human-written control | −0.54 | 0.58 | not flagged ✓ |

That historical report says SynthID generation ran but detection was not attempted (`OPEN`, B14). It does not describe the current repository status: untrained SynthID detection later ran through `vllm serve` (`EXECUTED`, D8). Its CPU timing is excluded from current evidence under B24; use the corrected A10G figures in D2.

## 5. Citations

- "A Watermark for Large Language Models" — arXiv 2301.10226
- "Scalable watermarking for identifying LLM outputs", Nature 634 (2024) — github.com/google-deepmind/synthid-text
- arXiv 2303.11156 (paraphrase attacks) · arXiv 2508.20228 (SynthID back-translation) · arXiv 2303.13408 (retrieval defense) · arXiv 2405.14604 / 2506.06409 (low-entropy) · arXiv 2607.16010 (forensic-readiness outlier) · arXiv 2405.10051 (MarkLLM) · arXiv 2602.06754 (unified framework, eth-sri)
- vLLM: docs.vllm.ai/en/latest/features/custom_logitsprocs/ · RFC #17799 · PRs #19912, #22919, #34400, #43672, #47585
- Vendor landscape 2026-08-13: Anthropic, "How Claude marks AI-generated content" — support.claude.com/en/articles/16266773 (B26) · Euronews 2026-08-11 · The Next Web 2026-08-12 (B27)
