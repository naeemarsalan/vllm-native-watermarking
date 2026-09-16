# Mark, detect, repeat

**Decode-time text watermarking and continuous validation for vLLM on OpenShift AI.**

## [Open the animated architecture and full article](https://naeemarsalan.github.io/vllm-watermark/)

> [!NOTE]
> **2026-09-16 — the approach has moved upstream.** vLLM now implements text
> watermarking natively (Gumbel-max sampler plus a detector, merged upstream
> 2026-09-10 and 2026-09-12; facts [E1–E9](docs/facts.md)). That supersedes the
> out-of-tree logits-processor plugin this repository originally built.
> **Start here: [Guide: native text watermarking in vLLM, end to end on OpenShift](docs/guide-native-watermarking.md)** —
> executed on a fresh cluster on 2026-09-16, with deployable manifests in
> [`deploy/native/`](deploy/native/) and a standalone client in
> [`scripts/native-watermark-demo.py`](scripts/native-watermark-demo.py).
> The rest of this README describes the original plugin work, kept for its
> regulatory analysis and evidence base.

This repository is an evidence-tracked proof of concept for generating and
detecting KGW and SynthID-Text watermarks through vLLM, deploying the components
on OpenShift AI, and continuously validating selected completed responses
through the current RHOAI-managed NeMo guardrails path (<code>EXECUTED</code>,
scoped; [facts C8, D5 and D10](docs/facts.md);
[current evidence](EXPERIMENTS.md#current-build5-d10-mode-evidence-2026-08-09)).

> [!IMPORTANT]
> This is engineering research—not legal advice, a Red Hat product or support
> statement, or proof of EU AI Act compliance. Applying the quoted provisions
> to a particular deployment remains a question for counsel
> (<code>OPEN</code>; [facts A4, A11 and D7](docs/facts.md)).

## What is changing in the EU

Article 50(2) expressly joins a machine-readable mark with the ability to detect
that mark. The voluntary Code and Commission Guidelines add text-specific
implementation detail. The quotations below are exact excerpts; they do not
replace their full context or deployment-specific legal analysis.

| Date | Registered text | Status and source |
|---|---|---|
| **2 August 2026** | Article 113 says, “It shall apply from 2 August 2026.” | <code>OJ-VERBATIM</code>; [Regulation (EU) 2024/1689](https://eur-lex.europa.eu/eli/reg/2024/1689/oj/eng), Article 113; [fact A2](docs/facts.md) |
| **2 December 2026** | Covered providers “shall take the necessary steps in order to comply with Article 50(2) by 2 December 2026.” | <code>OJ-VERBATIM</code>; [Regulation (EU) 2026/1744](https://eur-lex.europa.eu/eli/reg/2026/1744/oj/eng), Article 111(4); [fact A3](docs/facts.md) |
| **2 February 2027** | “Signatories will implement an interoperability solution for their detection mechanisms by 2 February 2027”. | <code>OJ-VERBATIM</code>; [voluntary Code, Measure 3.4](docs/quotes.md#guidelines-para-70); [fact A10](docs/facts.md) |

> “shall ensure that the outputs of the AI system are marked in a machine-readable
> format and detectable as artificially generated or manipulated.”
>
> — Regulation (EU) 2024/1689, Article 50(2)
> (<code>OJ-VERBATIM</code>; [official text](https://eur-lex.europa.eu/eli/reg/2024/1689/oj/eng);
> [registered full paragraph](docs/quotes.md#art-50-2); [fact A1](docs/facts.md))

> “For free-form text longer than 200 tokens, watermarking still needs to be
> applied, even though it may have lower reliability compared to that of
> watermarking very long text.”
>
> — Voluntary Code of Practice, Sub-measure 1.1.2
> (<code>OJ-VERBATIM</code>; [official Commission PDF](https://ec.europa.eu/newsroom/dae/redirection/document/129555);
> [registered context](docs/quotes.md#cop-measure-1-1); [fact A7](docs/facts.md)).
> The 200-token language comes from the voluntary Code, not Article 50 itself.

> “Fulfilling only one element (e.g. for machine-readable marking of outputs
> without the means for their detection being available) will not suffice to
> comply with that provision.”
>
> — Commission Guidelines on Article 50, paragraph 70
> (<code>OJ-VERBATIM</code>; [official Commission PDF](https://ec.europa.eu/newsroom/dae/redirection/document/131215);
> [registered context](docs/quotes.md#guidelines-para-70); [fact A9](docs/facts.md))

Article 111(4) says “placed on the market,” while Article 111(2) says “placed on
the market or put into service” (<code>OJ-VERBATIM</code>). Applying that
contrast to an internal-only system remains <code>OPEN</code> for counsel
([registered extract](docs/quotes.md#art-111-4); [facts A3–A5](docs/facts.md)).

## What that means for the engineering

- **The mark must travel with free-form text.** A UI badge or response field can
  disappear when text is copied. Decode-time watermarking instead alters keyed
  token-selection statistics, subject to reliability and robustness limits
  (<code>STATIC</code>; [technical design](docs/technical.md);
  [facts B17–B18](docs/facts.md)).
- **Detection is an operating path, not merely an algorithm.** It needs the
  matching scheme, tokenizer, configuration and server-held key, plus
  authentication, correlation, failure policy and observability
  (<code>STATIC</code> design / <code>EXECUTED</code> scoped path;
  [facts D5 and D10](docs/facts.md)).
- **A verdict has narrow meaning.** A positive watermark score does not establish
  truth, safety, copyright ownership or human identity, and a negative score
  does not prove human authorship (<code>STATIC</code>;
  [technical limitations](docs/technical.md)).

## How this solution can help

The implementation adds keyed KGW and SynthID-Text processors to vLLM's V1
decoding path, keeps detection in a separate CPU service, and places a
persistent sampling gateway in front of the internal RHOAI predictor
(<code>EXECUTED</code>, scoped; [facts D1, D5, D8 and D10](docs/facts.md);
[current fixed matrix](EXPERIMENTS.md#current-build5-d10-mode-evidence-2026-08-09)).

- **Decode-time marking:** watermark processors adjust token selection after
  model logits and before sampling; model weights remain unchanged
  (<code>STATIC</code> mechanism / <code>EXECUTED</code> serving;
  [facts B3–B5, D1 and D8](docs/facts.md)).
- **Independent detection:** the TrustyAI-compatible detector scores exact text
  using the matching scheme and a server-held key without requiring the vLLM
  GPU process (<code>STATIC</code> service separation /
  <code>EXECUTED</code> detector path; [fact D5](docs/facts.md)).
- **Continuous validation:** <code>N=1</code> validates every completed response
  and <code>N=5</code> validates ordinals 5, 10, 15, and so on at one persistent
  sampler (<code>EXECUTED</code>; [fact D10](docs/facts.md);
  [mode-complete evidence](EXPERIMENTS.md#current-build5-d10-mode-evidence-2026-08-09)).
- **Auditable evidence:** selected terminal records retain bounded correlation
  data, content digests, verdicts, attempts and timings without persisting prompt
  or response plaintext in application records
  (<code>EXECUTED</code> finite scans / <code>STATIC</code> storage design;
  [facts D5 and D10](docs/facts.md)).

## Architecture

The solid path below executed inside the cluster. The external KServe/Istio
entry, production caller authentication, mTLS, HA and streaming remain
<code>OPEN</code> ([facts C8 and D10](docs/facts.md)).

~~~mermaid
flowchart LR
    caller[Application caller]
    edge[External KServe / Istio<br/>OPEN]

    subgraph OCP[OpenShift AI — executed internal path]
      gateway[Validation gateway<br/>single replica]
      predictor[RHOAI predictor<br/>vLLM 0.18.0]
      secret[Mounted watermark Secret]
      sampler[(SQLite ordinal sampler)]
      pending[Exact selected response<br/>held in memory]
      nemo[Managed NeMo 0.21.0]
      broker[Authenticated metadata broker]
      detector[KGW / SynthID detector]
      evidence[(Hash-only records<br/>bounded metrics)]
    end

    edge -. OPEN .-> gateway
    caller -->|OpenAI-compatible request| gateway
    gateway --> predictor
    secret --> predictor
    predictor -->|completed response| gateway
    gateway --> sampler
    gateway --> pending
    gateway -->|selected response + bounded context| nemo
    nemo -->|IDs, digest, scheme, key ID| broker
    broker --> pending
    pending --> detector
    detector --> broker
    broker --> nemo
    nemo -->|blocked / success| gateway
    gateway --> evidence
    gateway -->|deliver, 403, or content-free 503| caller
~~~

The [animated nine-step version](https://naeemarsalan.github.io/vllm-watermark/)
shows live ordinals, selection, retry and queue state. It distinguishes managed
NeMo's positive action (<code>blocked</code>) from the executed gateway delivery
policy (<code>flag</code>) and includes the separately executed detector-outage
branch (<code>EXECUTED</code>; [fact D10](docs/facts.md);
[outage evidence](EXPERIMENTS.md#2026-08-09--phase-4-current-managed-path-and-d10-continuous-validation-executed-redacted)).

## What the evidence supports

The measurements below are bounded to upstream vLLM 0.18.0,
Qwen2.5-0.5B-Instruct, one NVIDIA A10G, one key/configuration and the recorded
corpora. They are observations, not universal rates
(<code>EXECUTED</code>, scoped; [facts D1, D2, D8 and D10](docs/facts.md)).

### Continuous-validation matrix

| Result | Observed |
|---|---:|
| Completed responses selected at <code>N=1</code> | **20 / 20** |
| Responses selected at <code>N=5</code> | **20 / 100** |
| Selected records carrying <code>mode=synchronous</code> | **40 / 40** |
| Tests at the last cluster-validated revision | **288 passed** |

<code>EXECUTED</code>; [mode-complete fixed matrix](EXPERIMENTS.md#current-build5-d10-mode-evidence-2026-08-09)
and [detector reconciliation](EXPERIMENTS.md#current-detector-reconciliation-2026-08-09).

### Detection at 200 scored tokens

| Scheme | Watermarked detected | Unwatermarked false positives | Human-text false positives |
|---|---:|---:|---:|
| KGW, delta 2.0 | 119 / 120 | 0 / 119 | 0 / 150 |
| SynthID, depth 30 | 120 / 120 | 0 / 119 | 0 / 150 |

<code>EXECUTED</code>; [corrected corpus evidence](EXPERIMENTS.md#2026-08-08--phase-2-completion-kgw-vs-synthid-detectability-by-length-executed).
“Zero observed” is not a universal zero-error claim.

### Matched serving performance

| Configuration | Aggregate output tokens/s | p50 latency |
|---|---:|---:|
| No plugin | 904.35 | 1.117 s |
| Processors loaded, watermark off | 913.78 | 1.115 s |
| KGW on | 643.38 | 1.576 s |
| SynthID on | 287.82 | 3.563 s |

<code>EXECUTED</code>; one small model, concurrency 4 and 256 requested output
tokens on the recorded A10G configuration
([corrected Phase 1/2 evidence](EXPERIMENTS.md#2026-08-08--phase-1-corrected--phase-2-synthid-through-vllm-serve-closes-d8)).
Larger models, realistic batches and optimisation remain <code>OPEN</code>
([facts D2–D4](docs/facts.md)).

### The result that had to be discarded

The first KGW run loaded the processor twice—once through the wheel entry point
and once explicitly—roughly doubling the intended bias. Those figures were
marked superseded, the duplicate loading path was removed, and the corpora and
measurements were rerun (<code>EXECUTED</code>;
[correction record](EXPERIMENTS.md#2026-08-08--correction-phase-1-ran-two-kgw-processor-instances-effective-delta-40)).
This is why the repository treats raw execution evidence as part of the
implementation rather than optional documentation.

## Where the claim stops

Demonstrated in the recorded scope (<code>EXECUTED</code>;
[facts C8, D5, D9 and D10](docs/facts.md)):

- KGW and SynthID-Text generation through <code>vllm serve</code>.
- Independent matching-scheme detection with server-held keys.
- RHOAI ServingRuntime, InferenceService and internal predictor execution.
- Managed-NeMo correlation through an authenticated metadata broker.
- Strict one-in-<code>N</code> synchronous validation, bounded retry and
  fail-closed detector-outage behaviour.
- Hash-only records, bounded metric labels and finite secret/plaintext scans.

Still unresolved for production (<code>OPEN</code>;
[facts D2–D4, D6, D9 and D10](docs/facts.md)):

- External KServe/Istio pass-through and an authenticated public entry point.
- NetworkPolicy, mTLS, HA, PDB and multi-replica/global-ordinal semantics.
- Restarts, rollouts, streaming and asynchronous delivery.
- Platform-wide retention guarantees and a full data-flow threat model.
- Key creation, rotation, application scoping and compromise response.
- Paraphrase, translation, multilingual, code and structured-output robustness.
- Larger-model quality and production-performance evaluation.
- Product supportability and deployment-specific legal conclusions.

## Repository map

| Path | Purpose |
|---|---|
| [Animated article](https://naeemarsalan.github.io/vllm-watermark/) | Responsive article and animated reference architecture |
| [<code>docs/blog.html</code>](docs/blog.html) | Self-contained source for the Pages article |
| [<code>docs/guide-native-watermarking.md</code>](docs/guide-native-watermarking.md) | **Step-by-step guide to vLLM's native watermark** (deploy, generate, detect) |
| [<code>deploy/native/</code>](deploy/native/) | Manifests for the native-watermark demo (vLLM + reference detector) |
| [<code>scripts/native-watermark-demo.py</code>](scripts/native-watermark-demo.py) | Generate-and-detect client (stdlib only) |
| [<code>docs/facts.md</code>](docs/facts.md) | Claim register with verification status and source |
| [<code>docs/quotes.md</code>](docs/quotes.md) | Exact legal quotations with provenance |
| [<code>docs/technical.md</code>](docs/technical.md) | vLLM extension point, watermarking design and limitations |
| [<code>docs/openshift-ai.md</code>](docs/openshift-ai.md) | OpenShift AI and TrustyAI integration facts |
| [<code>docs/implementation.md</code>](docs/implementation.md) | Ordered phases and acceptance criteria |
| [<code>EXPERIMENTS.md</code>](EXPERIMENTS.md) | Append-only commands, environments and raw execution evidence |
| [<code>src/vllm_watermark/</code>](src/vllm_watermark/) | KGW and SynthID-Text vLLM processors |
| [<code>detector/</code>](detector/) | Independent detection service |
| [<code>validation/</code>](validation/) | Continuous-validation gateway and managed-NeMo integration |
| [<code>benchmarks/</code>](benchmarks/) | Serving, analysis, fuzz and bounded stress harnesses |
| [<code>deploy/</code>](deploy/) | Container and OpenShift deployment assets |

## View the article locally

~~~bash
python3 -m http.server 8765 --bind 127.0.0.1 --directory docs
~~~

Then open <http://127.0.0.1:8765/blog.html>.

The article is a single HTML file with inlined CSS, JavaScript and SVG; it makes
no external asset requests (<code>STATIC</code>; [source](docs/blog.html)). Its
player pattern is adapted from the MIT-licensed
[<code>refarch-animator</code>](https://github.com/naeemarsalan/refarch-animator).

## Licensing and safety

- Do not copy from <code>eth-sri/unified-watermarking</code>; no license was
  found in the recorded review (<code>STATIC</code>; [fact B11](docs/facts.md)).
- Algorithm logic in this repository is derived only from the attributed
  Apache-2.0 sources listed in file headers and the fact register
  (<code>STATIC</code> / <code>OFFICIAL-SRC</code>;
  [facts B13, B15 and B16](docs/facts.md)).
- Watermark keys belong in environment variables or mounted Secrets. They must
  never be committed or logged (<code>STATIC</code>; repository security
  contract in [<code>AGENTS.md</code>](AGENTS.md)).
