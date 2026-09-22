# Open questions and recorded disagreements

Short, dated entries. Each records a question this repository cannot answer from
its own evidence, or a recommendation it declined to follow and why. Verification
tags follow [`facts.md`](facts.md).

## 2026-09-21 — External review of "commit f5b0dd3" does not describe this project

An outside review was received describing a "dispatch loop", an orchestrator
idling at 38-53%, issue counts falling from 21 to 16, issue #18 closed and #19
plus "twelve renderer questions" opened, and "three unblocked orphaned test
ports" left unported. It recommended stopping the orchestrator.

**Declined: none of it is about this repository.** Checked at the time of the
review (`EXECUTED`):

| Claim in the review | Checked | Result |
|---|---|---|
| commit `f5b0dd3` | `git cat-file -t f5b0dd3`, `git log --all` | not a valid object in this repository |
| issues #18, #19, twelve renderer issues | `gh issue list --state all` on both remotes | **zero** issues have ever been opened on either repository |
| issue count 21 -> 16 | same | there is no issue tracker activity to count |
| `docs/QUESTIONS.md` already exists | `ls` | did not exist; this file creates it |
| a "dispatch loop" / orchestrator | `git grep -i dispatch`, `.github/workflows` | no scheduled automation exists; the only matches are an unrelated `--scheme` dispatch in `benchmarks/analyze_detection.py` and a NeMo action dispatch in an API note |
| three orphaned test ports | `git grep -i "orphaned\|test port"` | no such work items exist |

The reviewer was told the commit under review was `f5b0dd3`; the head of this
repository at that time was `ce32015`. The review appears to have been produced
against a different codebase.

**The general pattern it describes is still worth naming, because a version of
it did occur here.** Its core warning is that orchestration activity can be
mistaken for progress, and that the characteristic failure is one that looks
like success. On 2026-09-21 the guide-verification workflow reported rounds 3
and 4 as finding "0 findings, 0 new", which its own loop-until-dry condition
would have read as convergence. Those rounds actually found nothing because all
eight finder agents had failed on exhausted usage credits. The distinction was
caught by reading the failure list rather than the round counters, and the
verification was re-run on cheaper models, which converged for real with one
finding. Recorded here because the near-miss is the honest part: the counter and
the reality agreed only by accident.

Where the review's charge does not hold: over the period in question this
repository shipped an executed end-to-end guide, deployable manifests, a
deploy script, a generate-and-detect client, a paired serving benchmark, a
three-experiment measurement harness, ten registered facts, and recorded
measurements from a real cluster, all committed and pushed. Orchestration was
used to verify that work, not in place of it.
