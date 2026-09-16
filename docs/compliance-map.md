# Compliance map — quoted obligation → implementation evidence → how to verify

This document traces each quoted regulatory obligation (from
[`quotes.md`](quotes.md) — the only place legal text is stated in this
repository) to its implementation status and the concrete evidence behind
that status, with a runnable verification command wherever one exists.
Statuses use the [`facts.md`](facts.md) tag discipline; **this is an
engineering traceability document, not legal advice, and no row here is a
compliance determination** — the counsel-reserved questions are listed
explicitly at the end.

How to read the Status column:

- **Implemented (`EXECUTED`)** — the mechanism ran and its command + raw
  output are preserved in [`EXPERIMENTS.md`](../EXPERIMENTS.md).
- **Partial** — the mechanism is executed but a quoted qualifier of the
  obligation is not yet evidenced.
- **Gap** — no implementation exists yet.
- **Counsel / PM** — not closable by engineering; owner named.

## The map

| # | Quoted obligation (source) | Status | Evidence | Verify it yourself |
|---|---|---|---|---|
| M1 | Machine-readable marking + detectability of outputs — [Art. 50(2)](quotes.md#art-50-2) (fact A1) | **Partial** — mechanism `EXECUTED`; the quoted qualifiers (effective/interoperable/robust/reliable) not fully evidenced (see M6, M7) | D1 (KGW through `vllm serve`, TPR 1.000 / FPR 0.000), D8 (SynthID untrained scorer, TPR 1.000 / FPR 0.000) in [`facts.md`](facts.md); raw runs in [`EXPERIMENTS.md`](../EXPERIMENTS.md#2026-08-08--phase-1-corrected--phase-2-synthid-through-vllm-serve-closes-d8) | `python3 scripts/verify-claims.py` re-derives every registered TPR/FPR/mean-z cell from the committed corpora (needs the detection key; see below) |
| M2 | Watermarking of free-form text longer than 200 tokens; single layer considered sufficient for free-form text — [CoP Measure 1.1](quotes.md#cop-measure-1-1) (facts A7, A8) | **Implemented (`EXECUTED`)** at the 200-token threshold: KGW TPR 0.992 / FPR 0.000, SynthID TPR 1.000 / FPR 0.000 | [Scheme comparison v2](../EXPERIMENTS.md#2026-08-08--scheme-comparison-v2-per-scheme-control-fpr-supersedes-the-v1-tables-control-rows), 200-token rows | Same command as M1 — the 200-token rows are among the asserted cells |
| M3 | Marking without available detection means "will not suffice"; fingerprinting/logging alone insufficient — [Guidelines para 70 / CoP](quotes.md#guidelines-para-70) (fact A9) | **Implemented (`EXECUTED`)** — standalone statistical detector service (TrustyAI/FMS contract `POST /api/v1/text/contents` + direct signed endpoint), correct positive/negative/cross-scheme verdicts | D5 (partially closed; executable half `EXECUTED`) in [`facts.md`](facts.md); [Phase 3 evidence](../EXPERIMENTS.md#2026-08-08--phase-3-detector-service--fms-guardrailsorchestrator-end-to-end-closes-d5s-executable-half) | `python3 scripts/verify-claims.py --demo` scores one committed watermarked sample (detected) and one human sample (not detected) in seconds, locally |
| M4 | Detection-mechanism interoperability solution by 2 Feb 2027 — [Measure 3.4](quotes.md#guidelines-para-70) (fact A10) | **Gap** — no interoperability solution exists or is scheduled in [`implementation.md`](implementation.md); detector recognizes only this repo's two schemes with local keys | — | — (nothing to verify yet) |
| M5 | Detector-access conditions (free-of-charge / professional-setting language) — [Guidelines/CoP extracts](quotes.md#guidelines-para-70) (fact A14) | **Counsel** — application to a particular deployment is reserved (A14 `OPEN`); current exposure is ClusterIP-only by default | — | — |
| M6 | "Effective, interoperable, robust and reliable as far as technically feasible" qualifiers of [Art. 50(2)](quotes.md#art-50-2) | **Partial / Open** — robustness against paraphrase/translation unmeasured for this implementation (B17 `CORROBORATED`/`OPEN`); scale reliability partially measured (D2, one config); tensor-parallel untested (D3 `OPEN`) | B17, B18, D2, D3 in [`facts.md`](facts.md) | Overhead numbers: [`benchmarks/results/`](../benchmarks/results/) JSON captures; robustness: no executed result yet (Phase 5) |
| M7 | Recognised techniques (watermarking named) — [Recital 133](quotes.md#recital-133) and CoP technique naming (fact A12) | **Implemented (`EXECUTED`)** — the two implemented schemes (KGW, SynthID-Text) are the named technique class, ported from Apache-2.0 sources with attribution | [`src/vllm_watermark/`](../src/vllm_watermark/) file headers; B13, B15 in [`facts.md`](facts.md) | `python3 -m pytest -q tests detector/tests` (local suite incl. transformers-equivalence tests, B21/B23) |
| M8 | Deployer disclosure duties for published/interacting text — [Art. 50(4)](quotes.md#art-50-4), [Art. 50(5)](quotes.md#art-50-5) | **Out of repo scope** — end-user-facing product duty of the deployer, not of the marking/detection substrate | — | — |
| M9 | Deadline framework — [Art. 113](quotes.md), [Art. 111(4)](quotes.md#art-111-4) (facts A2–A5) | **Counsel** — grace-period applicability to internal-only deployment is D7/A4 `OPEN` | [`facts.md`](facts.md) A2–A5, D7 | — |
| M10 | Production-deployment boundaries required before any live compliance claim: key lifecycle (D4), modified-image supportability (D6, PM), external gateway pass-through (C8 `OPEN` half), HA/streaming/retention (D10 `OPEN` halves) | **Gap / PM** | D4, D6, C8, D10 in [`facts.md`](facts.md); [`ADVERSARIAL_REVIEW.md`](../ADVERSARIAL_REVIEW.md) | Executed halves: reproduction commands preserved per run in [`EXPERIMENTS.md`](../EXPERIMENTS.md) |

## Concrete verification — three levels

The detection claims are **deterministically re-derivable**: detection is a
pure function of (text, tokenizer, key), and the corpora that produced the
registered numbers are committed under [`benchmarks/data/`](../benchmarks/data/),
pinned by sha256 in the verifier. No GPU or cluster access is needed to
re-check them — only a checkout, local CPU Python (torch + transformers),
and the detection key:

```sh
set -a && . cluster/watermark-key.env && set +a   # key never printed/committed
```

1. **Seconds — live demo.** Score one committed watermarked sample and one
   human sample; watch the z-scores separate (≈12.8 vs ≈-0.9 at threshold 4):

   ```sh
   python3 scripts/verify-claims.py --demo
   ```

2. **Minutes — full claim reproduction.** Recompute every cell of the
   registered scheme-comparison table (6 corpus/detector rows × 3
   truncation lengths) from the committed corpora and diff against the
   values transcribed from `EXPERIMENTS.md`; corpus hashes are checked
   first so silent data edits fail loudly:

   ```sh
   python3 scripts/verify-claims.py
   ```

3. **Cluster — end-to-end regeneration.** Regenerate corpora through
   `vllm serve` with the watermark plugin and re-run levels 1–2. Exact
   commands for every executed run are preserved append-only in
   [`EXPERIMENTS.md`](../EXPERIMENTS.md) (see each run's reproduction
   notes); cluster bring-up is in [`cluster.md`](cluster.md).

Because detection is keyed, third-party verification requires being given
the detection key (or a demo key + freshly generated corpus). That is
inherent to the scheme, and the access-conditions question is M5.

## Showing the evidence (stakeholder path)

- The 30-second version: `verify-claims.py --demo` — two samples, two
  z-scores, one threshold. The elided-sample hashes it prints match the
  hashes recorded in the `EXPERIMENTS.md` evidence transcripts, so the
  demo is visibly running on the registered evidence, not staged inputs.
- The 5-minute version: `verify-claims.py` full run — the registered
  table reproduces cell-for-cell, live.
- The paper trail: [`facts.md`](facts.md) (every claim tagged) →
  [`EXPERIMENTS.md`](../EXPERIMENTS.md) (append-only raw evidence) →
  this map (obligation → evidence).
