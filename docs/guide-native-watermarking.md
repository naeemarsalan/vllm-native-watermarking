# Guide: native text watermarking in vLLM, end to end on OpenShift

This guide shows how to turn on vLLM's built-in text watermark, generate text
that carries it, and detect it again. It is written for someone who operates
OpenShift and has used vLLM, but has never touched watermarking. Everything
below that is tagged `EXECUTED` was run on 2026-09-16 on a fresh OpenShift 4.20
cluster on AWS with one NVIDIA A10G node; the exact commands and raw outputs are
preserved in
[`EXPERIMENTS.md`](../EXPERIMENTS.md#2026-09-16--native-gumbel-max-watermarking-end-to-end-on-a-fresh-cluster-executed-redacted).

Verification tags follow [`facts.md`](facts.md): `EXECUTED` means it ran and the
output is recorded; `OFFICIAL-SRC` means it is stated in upstream vLLM docs or
source at the pinned commit; `STATIC` means it was verified by reading source, not
by running it; `CORROBORATED` and `OPEN` are as defined there. The upstream claims
behind this guide are registered as facts E1–E10 in
[`facts.md`](facts.md#e-native-vllm-watermarking-upstream-feature-registered-2026-09-1516).

---

## 1. What a text watermark is (two minutes)

A language model does not pick one "correct" next word. At every step it has a
probability for each candidate token, and normally it rolls dice to choose. A
**sampling watermark** replaces the dice with a coin that only the server can
predict: the randomness is computed from a **secret key** plus the last few
tokens (the *context*, 4 tokens by default). The text still reads exactly as it
would have — the choice at each step is still drawn from the model's own
probabilities — but the sequence of "lucky" choices now follows a pattern. One
consequence: because the coin depends only on the key and the context, repeating a
prompt under one key normally returns the same watermarked answer, even at
temperature 1.0, and the request `seed` does not change it (the first bullet below
and section 5 cover this).

Later, a **detector** that holds the same key and the same tokenizer replays
that coin over any text you give it. Each token gets a score: how lucky was this
choice under the key? Human text, or text from a model without the key, scores
like random noise. Watermarked text scores consistently higher; how much higher
depends on how much randomness the sampler was left with. At temperature 1.0 and
top_p 1.0 (section 4.2) every watermarked answer in this run (`max_tokens 300`; the
shortest was 172 tokens) came back at p ≤ 4e-14, while at the model's shipped defaults
of temperature 0.7 and top_p 0.8 (section 4.3; its top_k 20 and repetition_penalty 1.1
applied to both runs) the same prompts scored between p ≈ 4e-4 and 4e-10 and one of
ten was missed at the 0.01 threshold. The detector reports the evidence as a
**p-value**: the probability that unwatermarked text would score this high by
chance. `p = 1e-9` means "essentially impossible unless watermarked";
`p = 0.4` means "nothing unusual here".

The algorithm vLLM implements is called **Gumbel-max**, due to Scott Aaronson
([talk "Watermarking of Large Language Models", Simons Institute, 17 Aug 2023](https://simons.berkeley.edu/talks/scott-aaronson-ut-austin-openai-2023-08-17);
the [October 2024 slides](https://simons.berkeley.edu/sites/default/files/2024-10/LLM24-2%20Slides%20-%20Scott%20Aaronson.pdf)
are what the upstream
[`docs/features/watermarking.md`](https://github.com/vllm-project/vllm/blob/cd10ed6f9f6b37a8ace9cf380007e66fe12ec0c3/docs/features/watermarking.md)
links; fact E10). Three properties matter in practice; each carries its own tag:

- **The sampling is non-distorting per token.** Averaged over keys, the expected
  next-token distribution is the model's own (`CORROBORATED`: RFC #53916,
  "Non-distortion", and Aaronson's slides, both linked from the upstream doc), and
  with the default `deduplicate_contexts: single_turn` upstream states single-turn
  non-distortion for a whole answer (`OFFICIAL-SRC`, upstream doc). There is no
  "watermark strength" knob: `WatermarkConfig` has no delta/strength field
  (`OFFICIAL-SRC`, `vllm/config/watermarking.py` @cd10ed6f; fact E4). Upstream's RFC
  warns that per-token non-distortion does not guarantee it over a keyed sequence
  and that, for one fixed key, this scheme shows a "catastrophic loss of diversity
  (determinism, given a fixed key and input)" (`CORROBORATED`, RFC #53916 "Note on
  diversity"). That collapse is observable here (`EXECUTED`, EXPERIMENTS.md 2026-09-16
  "Fixed-key determinism"): on a quiet server the section 4.1 request returned
  byte-identical text on 7 of 7 consecutive calls, with no `seed` and with `seed` 1, 1,
  2 and 3, and a second prompt was identical on 6 of 6, while unseeded opt-out repeats
  of both prompts all differed and two opt-out calls with `seed` 1 matched. The request
  `seed` is read only for positions sampled without the watermark (`STATIC`: the keyed
  draw is `philox_gumbel_sample(logits, contexts, key)`, `vllm/v1/watermarking/gumbel.py`
  L64-67 @cd10ed6f, and the kernel in `vllm/v1/worker/gpu/sample/watermark.py` L364-371
  loads `seeds_ptr` only inside its `skip_mask` branch). It is near-determinism, not a
  guarantee: the section 4.1 completion recorded below (228 words, score 617.807) is
  not the text the same request returned on every later call (234 words, 618.114 as
  written), and an `n: 2` request returned two different completions, neither equal to
  the single-request text (`EXECUTED`); vLLM is not batch-invariant by default
  (`VLLM_BATCH_INVARIANT`, `vllm/envs.py` L92 @cd10ed6f, `STATIC`), and whether that
  explains these differences was not determined (`OPEN`). Output quality was not
  measured in this repository (`OPEN`).
- **It only works where the model is actually rolling dice.** Greedy decoding
  (`temperature: 0`) skips the watermark and logs a warning once per worker
  (`OFFICIAL-SRC`, upstream doc; `EXECUTED`, section 4.3). `top_k: 1` is not a
  special case: the request still goes through the watermark sampler, but the top-k
  mask leaves a single candidate, so the output carries no signal and nothing is
  logged (`STATIC`: `vllm/v1/watermarking/gpu_sampler.py` @cd10ed6f warns only on
  `temperature == 0` and applies `apply_top_k_top_p` before the keyed draw; not
  exercised in the recorded runs; fact E6). Low-temperature or truncated sampling
  carries a weaker signal: 9 of 10 answers detected at the model's shipped defaults
  against 10 of 10 at temperature 1.0 (`EXECUTED`, section 4.3; upstream reports
  ≈42 % against ≈84 % for a 2B model on GSM8K, `CORROBORATED`, fact E9). Longer
  outputs accumulate more evidence, because the score is a sum over scored tokens
  (fact E7); length was capped at `max_tokens 300` and not varied in this run.
- **Detection needs the key.** No key, no detection. The detector must run with
  the same tokenizer, PRF (the pseudorandom function that turns the key, the context
  and the candidate token into the coin; `philox` is the only one implemented,
  `WatermarkPRFName = Literal["philox"]` in `vllm/config/watermarking.py` @cd10ed6f;
  `OFFICIAL-SRC`, fact E4), algorithm, algorithm-specific configuration (context
  width) and key as generation (`OFFICIAL-SRC`, upstream doc, "Detection").
  Keeping the key on the serving side is deployment guidance; section 5 covers it.

---

## 2. Prerequisites

| Requirement | What was used here (`EXECUTED`) |
|---|---|
| OpenShift cluster | 4.20.27 on AWS (`openshift-install`), 3 × m6i.xlarge control plane, 2 × m6i.xlarge workers |
| One NVIDIA GPU node | g5.xlarge (A10G 24 GB) added as a MachineSet; NVIDIA GPU Operator 26.7.0 installed driver 595.91.07, which reports CUDA 13.2 (node label `nvidia.com/cuda.runtime-version.full`: the highest CUDA the driver supports). The vLLM image ships CUDA toolkit 13.0.2 and declares `NVIDIA_REQUIRE_CUDA=cuda>=13.0` (image config `CUDA_VERSION` / `NVIDIA_REQUIRE_CUDA` read with `skopeo inspect --config`; `EXECUTED`, EXPERIMENTS.md 2026-09-16; fact E3), so a driver reporting CUDA ≥ 13.0 runs it and the two numbers are not expected to match |
| Node Feature Discovery + NVIDIA GPU Operator | installed by [`scripts/install-gpu-operators.sh`](../scripts/install-gpu-operators.sh) |
| A vLLM build that contains the feature | upstream nightly `vllm/vllm-openai:nightly-cd10ed6f9f6b37a8ace9cf380007e66fe12ec0c3`, pinned by digest in the manifests |
| `oc`, `python3`, `curl`, `jq` on your workstation | any recent version; `python3` needs only the standard library for the demo client; `jq` is used by the three GPU scripts below and by appendix step 2 |
| `gh`, `skopeo` (appendix only) | to check whether another image tag contains the feature and to resolve its digest |

**Why a nightly image.** As of 2026-09-15 the watermark is merged upstream
(PR [#54053](https://github.com/vllm-project/vllm/pull/54053) on 2026-09-10,
follow-up [#56122](https://github.com/vllm-project/vllm/pull/56122) on 2026-09-12)
but is **not in any released vLLM**: v0.29.0 was tagged before the merge. Daily
`nightly-<sha>` images from 2026-09-13 onward (first: `nightly-2671fedf…`) contain
both PRs; the 2026-09-10, -11 and -12 nightlies (`nightly-2a02f6ef…`,
`nightly-e7edf17c…`, `nightly-eed1f3d0…`) contain only #54053, and anything earlier
predates the feature entirely (`OFFICIAL-SRC`: GitHub compare API against the merge
commits `ea40bb9e` and `7ee8a6dd` for each September tag; table in EXPERIMENTS.md
2026-09-16; fact E3). The one used here is from 2026-09-15 and reports itself as
`v0.29.1rc1.dev128+gcd10ed6f9`. When the
next stable release (or a Red Hat AI Inference image that packages it) ships,
replace the image reference in `deploy/native/*.yaml` and nothing else changes.
OpenShift AI is not required for this demo; it is a fine place to run the same
container as a custom serving runtime once a supported build exists.

**GPU prerequisites, if you are starting from a bare cluster** (each step is a
script in this repository). `EXECUTED` for the resulting operator CSVs, validator pod,
MachineSet and node state (EXPERIMENTS.md 2026-09-16), except `scale-gpu.sh`, which did
not run on this cluster: the recorded run created the MachineSet directly at one replica,
and the MachineSet still has generation 1, replicas 1, created 05:04:19Z together with its
Machine (`EXECUTED`). `scale-gpu.sh` sets `spec.replicas` on that same MachineSet, which it
selects by the `node-role.kubernetes.io/gpu` template label `create-gpu-machineset.sh` sets
(`STATIC`, both scripts); its 0 → 1 and 1 → 0 runs are recorded on the previous cluster
(EXPERIMENTS.md 2026-08-08) and the script is unchanged since:

```bash
export KUBECONFIG=<path to your kubeconfig>   # the scripts default to cluster/auth/kubeconfig; see "Before you start" in section 3
./scripts/install-gpu-operators.sh        # NFD + NVIDIA GPU Operator + ClusterPolicy
GPU_REPLICAS=1 ./scripts/create-gpu-machineset.sh   # g5.xlarge MachineSet, node starts now (what the recorded run did)
# or: GPU_REPLICAS=0 ./scripts/create-gpu-machineset.sh now, then ./scripts/scale-gpu.sh 1 when you need the (billable) node
oc get nodes -l node-role.kubernetes.io/gpu   # wait until Ready, then:
oc -n nvidia-gpu-operator get pods | grep validator   # nvidia-cuda-validator Completed
```

---

## 3. Deploy: vLLM with the watermark on, plus the detector

Everything lives in [`deploy/native/`](../deploy/native/):

| File | What it creates |
|---|---|
| `00-namespace.yaml` | namespace `watermark-demo` |
| `10-vllm.yaml` | Deployment + Service + Route for vLLM on the GPU node |
| `20-detector.yaml` | ConfigMap (the detector script), Deployment + Service + Route for the detector (no GPU request, no node selector: any worker) |

**Before you start.** Run everything from the repository root. The guide writes into
`cluster/`, which is gitignored and not part of a clone; it is already there only if
`openshift-install create cluster --dir cluster` wrote it, otherwise `mkdir -p cluster`
first. Point `oc` at your cluster with `export KUBECONFIG=<path to your kubeconfig>`:
every shell script in `scripts/` defaults `KUBECONFIG` to `cluster/auth/kubeconfig` (the
Python scripts never touch the cluster API; the demo client talks only HTTP to the two
Routes). If that file is absent and `KUBECONFIG` is not exported, `native-watermark-up.sh`
stops before its first `oc` call, exit 2, with `KUBECONFIG=<path> not found; export
KUBECONFIG=<your kubeconfig> (oc does not fall back to ~/.kube/config)`; the three GPU
scripts and bare `oc` stop with `Missing or incomplete configuration info` instead of
falling back to `~/.kube/config` (both `EXECUTED`, EXPERIMENTS.md 2026-09-16). The key
file defaults to `cluster/gumbel-key.env`;
`native-watermark-up.sh` honours `WATERMARK_KEY_FILE=<path>` if you keep it elsewhere.

### 3.1 Create the key

The key is one integer between 0 and 2^64-1 (`ge=0` on `WatermarkConfig.key`,
`vllm/config/watermarking.py:31`, and the validator at lines 61-62, `philox keys must
fit in 64 bits`, @cd10ed6f; `STATIC`). Generate it once, keep it out of
git (the `cluster/` directory is gitignored), and treat it like a password:

```bash
mkdir -p cluster   # gitignored; also used for router-ca.crt in section 4
python3 -c 'import secrets; print("WATERMARK_KEY=%d" % secrets.randbits(64))' > cluster/gumbel-key.env
chmod 600 cluster/gumbel-key.env
```

### 3.2 Deploy

```bash
./scripts/native-watermark-up.sh
```

After checking that the key file exists and that its only non-blank, non-comment line is
`WATERMARK_KEY=<integer>` (blank and `#` lines are skipped, as `--from-env-file` skips them;
the check was `EXECUTED` against nine synthetic files, EXPERIMENTS.md 2026-09-16), the
deploy half of that script is these four commands, which you could also run by
hand (the Secret step is written so that a re-run updates the existing Secret instead of
failing with `AlreadyExists`):

```bash
oc apply -f deploy/native/00-namespace.yaml
oc -n watermark-demo create secret generic watermark-key --from-env-file="${WATERMARK_KEY_FILE:-cluster/gumbel-key.env}" \
  --dry-run=client -o yaml | oc apply -f -
oc apply -f deploy/native/10-vllm.yaml
oc apply -f deploy/native/20-detector.yaml
```

The only watermark-specific part of the vLLM pod is one server flag. The pod's
entrypoint reads the key from the Secret and starts (full args line from
`deploy/native/10-vllm.yaml`):

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct --host 0.0.0.0 --port 8000 \
  --max-model-len 4096 --gpu-memory-utilization 0.85 \
  --watermark-config '{"algorithm":"gumbel","key":<WATERMARK_KEY>}'   # the only watermark-specific flag
```

`--max-model-len` and `--gpu-memory-utilization` are ordinary sizing choices for this
model on a 24 GB A10G, not part of the watermark feature.

Once the engine has that flag, **every request is watermarked by default**.
A caller can switch it off for one request with `"watermarking": false` in the
request body; section 5 covers why your gateway must control that field.

The detector pod runs the reference server that ships with vLLM: a byte-identical
copy of `examples/basic/online_serving/watermark_detection_server.py` at the pinned
commit `cd10ed6f` (`diff` and `cmp` of the ConfigMap script against upstream at that ref:
no differences, 70 lines each, one sha256, which the live ConfigMap also carries;
`EXECUTED`, EXPERIMENTS.md 2026-09-16), embedded in the
`watermark-detector-src` ConfigMap in `20-detector.yaml` and mounted at `/app`, so the
demo does not depend on `examples/` being present in the image. It runs on the same
image digest as the generator, so it imports the identical `vllm.v1.watermarking`
package, with the same key and tokenizer, on any worker: it requests no GPU, sets
`CUDA_VISIBLE_DEVICES` empty and carries no node selector or affinity, so the scheduler
places it. In this run it ran on the CPU worker `ip-10-0-150-204` (m6i.xlarge): the pod was
created at 05:06:03Z, before the GPU node existed (node created 05:08:40Z) (`EXECUTED`,
EXPERIMENTS.md 2026-09-16). Add a `nodeAffinity` excluding `node-role.kubernetes.io/gpu`
if it must stay off the GPU node:

```bash
python3 /app/watermark_detection_server.py --tokenizer Qwen/Qwen2.5-1.5B-Instruct \
  --key <WATERMARK_KEY> --context-width 4 --p-value-threshold 0.01 --host 0.0.0.0 --port 8080
```

A first start takes longer than the engine start alone: the GPU node has to schedule the
pod, pull the image and download the model before CUDA graph capture begins. When both
rollouts finish within its timeouts the script prints the two Route URLs (`STATIC`: on the
recorded run the vLLM wait stopped at the Deployment's then-default 600 s progress
deadline and the script exited 1 before printing them; see below).

What the engine logged at startup on this run (`oc -n watermark-demo logs deploy/vllm-watermark`,
key redacted; `EXECUTED`):

```
[core.py:123] Initializing a V1 LLM engine (v0.29.1rc1.dev128+gcd10ed6f9) with config: model='Qwen/Qwen2.5-1.5B-Instruct', ...
[gpu_worker.py:441] Using V2 Model Runner
[watermark_sample_warmup.py:130] Warming up watermark sampler kernel (vocab=151936, keys=1, dtypes=['torch.bfloat16', 'torch.float32'], skip_mask=[True]).
INFO:     Application startup complete.
```

Two lines matter: `Using V2 Model Runner` (the watermark exists only in Model Runner V2, which
is the default since v0.29.0; `OFFICIAL-SRC`: `VllmConfig.use_v2_model_runner` in
`vllm/config/vllm.py` L675-690 @cd10ed6f returns True whenever `watermark_config` is set,
and the v0.29.0 release notes; fact E5) and the `Warming up watermark sampler kernel`
line, which only appears when `--watermark-config` was accepted. On this A10G node the engine logged
`Application startup complete` 3 min 17 s after the container started (05:20:01Z →
05:23:17Z, `oc logs --timestamps`), and the pod went Ready at 05:23:24Z (3 min 23 s;
readiness is probed every 10 s). Before that, the pod waited 9 min for the GPU node
(created 05:06:02Z, scheduled 05:15:13Z) and under 5 min for the image pull (container
started 05:20:01Z) (`EXECUTED`; pod conditions and timestamped log lines in
EXPERIMENTS.md 2026-09-16).

On a slow first start the script's wait can fail, and the script then exits non-zero
without printing the Route URLs (it runs under `set -e`). The sequence is fixed, not a
race: `progressDeadlineSeconds: 1800` counts from the Deployment's last progress event,
i.e. within seconds of `oc apply` and before the 15 min detector wait, while the script's
vLLM `oc rollout status --timeout` is 35 min. So if the pod is not Ready 30 minutes after
`oc apply`, `oc rollout status` prints `error: deployment "vllm-watermark" exceeded its
progress deadline` and the script falls back to `oc wait --for=condition=Available
deployment/vllm-watermark --timeout=35m`. Only if that also expires (about 65 minutes
after `oc apply`) do you see `error: timed out waiting for the condition on
deployments/vllm-watermark` (that message form: `EXECUTED`, `oc wait` against a condition
the live Deployment does not have), after which the script prints `wait failed; current
state:` with the pod list and the last 20 engine log lines and exits 1 (`STATIC`: not
exercised on a fresh start; the ERR trap fires inside the `wait_for` function only because
the script runs under `set -E`, `EXECUTED` on a local bash repro). The recorded run hit the
deadline message under the Kubernetes default of 600 s; the 1800 s deadline has not been
exercised on a fresh start (`STATIC`). A bare `error: timed out waiting for the condition`
from `oc rollout status` itself is only possible on a re-run against an already-completed
Deployment whose pod is no longer Ready (for example after the GPU node was scaled down):
the controller does not re-apply the progress deadline once the Progressing reason is
`NewReplicaSetAvailable`, so `oc rollout status` waits out its own 35 min and returns
client-go's `wait.ErrWaitTimeout` (`STATIC`: kubernetes `deployment_util.go`
`DeploymentTimedOut` L797, release-1.33; client-go `tools/watch/until.go` L88-89,
release-1.32). The pod is usually still waiting
for the GPU node, pulling the image (8.7 GB compressed on the wire, 21.8 GB unpacked on
the node; `EXECUTED`, measured with `skopeo` and from the node's image list) or
downloading the model. Do not simply re-run `oc rollout status`: once the Progressing
condition carries `ProgressDeadlineExceeded`, kubectl returns that error immediately and
the controller keeps the reason until the pod is Ready (`STATIC`: kubectl
`pkg/polymorphichelpers/rollout_status.go` L75-79 and kubernetes
`pkg/controller/deployment/util/deployment_util.go` `DeploymentTimedOut`, release-1.33).
Wait on the Available condition instead, which the progress deadline does not affect:

```bash
oc -n watermark-demo get pods -w        # or: oc -n watermark-demo logs deploy/vllm-watermark -f
oc -n watermark-demo wait --for=condition=Available deployment/vllm-watermark --timeout=35m
```

then read the Route hosts with the two `oc get route` commands in section 4.0. The script
now does the same: when `oc rollout status` fails it falls back to that `oc wait`, and on a
final failure prints the pod list and the last engine log lines (`STATIC`: this path was
added after the recorded run and has not been exercised on a fresh start; the script runs
under `set -Eeuo pipefail` because without `-E` bash does not run the ERR trap for a
failure inside a function, `EXECUTED` on a local repro, EXPERIMENTS.md 2026-09-16; `oc wait
--for=condition=Available` itself was run read-only against the live Deployment and
returned `condition met`, `EXECUTED`). Re-running the whole script is also safe: every
step is `oc apply`, and `oc diff` against the live objects reports no drift (`EXECUTED`).

---

## 4. Test: generate watermarked text, then detect it

### 4.0 Trust the cluster's ingress certificate

A fresh cluster signs its Routes with a self-signed ingress CA, so `curl` (and the demo
client) refuse the connection until that CA is trusted. Extract it once (`EXECUTED`):

```bash
oc -n openshift-ingress-operator get secret router-ca -o jsonpath='{.data.tls\.crt}' | base64 -d > cluster/router-ca.crt
VLLM=https://$(oc -n watermark-demo get route vllm-watermark -o jsonpath='{.spec.host}')
DET=https://$(oc -n watermark-demo get route watermark-detector -o jsonpath='{.spec.host}')
curl -sS --cacert cluster/router-ca.crt $VLLM/v1/models | python3 -c 'import json,sys; print([m["id"] for m in json.load(sys.stdin)["data"]])'
```

```
['Qwen/Qwen2.5-1.5B-Instruct']
```

The unfiltered response is the OpenAI `ModelList` shape,
`{"object":"list","data":[{"id":"Qwen/Qwen2.5-1.5B-Instruct","object":"model",...}]}`
(full capture in EXPERIMENTS.md), which is also what the demo script reads
(`["data"][0]["id"]`); its `created` epoch and `permission[].id` change on every call,
which is why the command above filters to the model ids.

If your cluster already has a trusted ingress certificate, drop `--cacert`. Alternatively use
`oc -n watermark-demo port-forward svc/vllm-watermark 8000:8000` and
`oc -n watermark-demo port-forward svc/watermark-detector 8080:8080` (two terminals; they
block) and point the URLs at `http://localhost:<port>` (`EXECUTED` on local ports 18000 and
18080, EXPERIMENTS.md 2026-09-16; without `-n` the installer-written kubeconfig has no
current project and the Service lookup fails with `NotFound`).

### 4.1 One request by hand

Generate (default = watermarked):

```bash
curl -s --cacert cluster/router-ca.crt $VLLM/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "Qwen/Qwen2.5-1.5B-Instruct",
  "messages": [{"role": "user", "content": "Explain how vaccines train the immune system."}],
  "max_tokens": 300, "temperature": 1.0, "top_p": 1.0
}' | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"])' > wm.txt
```

Detect:

```bash
python3 -c 'import json; print(json.dumps({"text": open("wm.txt").read()}))' \
  | curl -s --cacert cluster/router-ca.crt $DET/detect -H 'Content-Type: application/json' -d @-
```

Real output from this run (`EXECUTED`; the completion was 300 tokens starting "Vaccines work by
teaching the immune system how to recognize and attack specific viruses or bacteria…"):

```
{"score":617.807335901951,"p_value":2.704772159904535e-46,"num_scored_tokens":300,"is_watermarked":true}
```

Send the watermarked request again and you will normally get the same text back; that is
the fixed key, not a cache (section 1). Seven later calls of this exact request, with and
without `seed`, all returned one 234-word completion (sha256 `5b8a9a94…`, scored 618.114
over 301 tokens as written) that differs from the 228-word one recorded above, so expect
the repetition, not this particular text (`EXECUTED`, EXPERIMENTS.md 2026-09-16 "Fixed-key
determinism").

Now the same prompt with the opt-out field, and detect again:

```bash
curl -s --cacert cluster/router-ca.crt $VLLM/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "Qwen/Qwen2.5-1.5B-Instruct",
  "messages": [{"role": "user", "content": "Explain how vaccines train the immune system."}],
  "max_tokens": 300, "temperature": 1.0, "top_p": 1.0, "watermarking": false
}' | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"])' > plain.txt
python3 -c 'import json; print(json.dumps({"text": open("plain.txt").read()}))' \
  | curl -s --cacert cluster/router-ca.crt $DET/detect -H 'Content-Type: application/json' -d @-
```

Real output (`EXECUTED`; a different 300-token answer to the same prompt):

```
{"score":317.3888859925337,"p_value":0.17178151881976203,"num_scored_tokens":301,"is_watermarked":false}
```

And a human-written sentence (the opening line of *Pride and Prejudice*, 26 tokens) for
comparison:

```
{"score":19.542122906190592,"p_value":0.9070642274102403,"num_scored_tokens":26,"is_watermarked":false}
```

Reading the response: `score` is the summed per-token evidence, `p_value` is
the chance of seeing that score without the key, `num_scored_tokens` is how many
tokens were scored, and `is_watermarked` is `p_value <= threshold` (0.01 here).

`num_scored_tokens` is not the generator's `completion_tokens`. The detector re-encodes
the text it is sent (`tokenizer.encode(text, add_special_tokens=False)` in the server
script) and scores each token whose preceding 4-token context has not already appeared
(`deduplicate_contexts=True`; fact E7), so the count can differ in both directions:

- repeated contexts are dropped, which only lowers it (`STATIC`, detector source);
- when the answer ends before `max_tokens` (`finish_reason: stop`), `completion_tokens`
  counts the `<|im_end|>` stop token, which is not in the returned text: a 19-token
  completion re-tokenized to 18 (`EXECUTED`, `return_token_ids` + `/tokenize`,
  EXPERIMENTS.md 2026-09-16); the 272 → 271 and 282 → 281 rows in section 4.2 fit this
  pattern;
- anything you add is tokenized too. The `print(...)` in the pipelines above appends a
  newline, which becomes one extra token when the answer was cut off mid-sentence at
  `max_tokens`: a fresh completion cut at `max_tokens` (`finish_reason: length`,
  `completion_tokens` 300) re-tokenized to 301 as written (last id 198, the newline) and
  to 300 with the newline stripped, and scored 301 and 300 accordingly (`EXECUTED`,
  `/tokenize` and `/detect`, EXPERIMENTS.md 2026-09-16). That is the most likely source
  of the 301 above. Use `sys.stdout.write(...)` instead of `print` if you want the file
  byte-identical to the completion;
- a sampled token sequence need not be the canonical BPE segmentation of its own text,
  so re-encoding can move the count in either direction: on this deployment
  `/detokenize` of `[279, 265]` (" the" + "re") gives " there", which `/tokenize`
  re-encodes as the single token 1052, and `[45421, 2912]` ("Li" + "opt") gives "Liopt",
  which re-encodes as three tokens `[43, 815, 417]` ("L", "io", "pt") (`EXECUTED`,
  EXPERIMENTS.md 2026-09-16). No drift of this kind was seen in the recorded completions:
  the `max_tokens` completions re-tokenized to exactly their `completion_tokens`, and the
  only differences observed were the stop token and the appended newline above.

### 4.2 Ten prompts, both ways, plus human text

[`scripts/native-watermark-demo.py`](../scripts/native-watermark-demo.py) runs
the loop above for N prompts and also scores human-written passages
(Project Gutenberg text in `benchmarks/data/human_corpus.jsonl`) as a
false-positive check.

The prompts are the first `--n` non-blank lines of
[`benchmarks/prompts.txt`](../benchmarks/prompts.txt), a fixed list of 120 prompts in
five blocks of 24 (open-ended "Explain…" questions, short-story requests, argue-a-position
prompts, "Summarize…" history questions and how-to walkthroughs); `--n 10` therefore uses
the first ten "Explain…" questions, which is why every prompt below starts that way. Pass
`--prompts <file>` (one prompt per line) to run the same check on your own traffic.

The human corpus is generated, not shipped: the data files in `benchmarks/data/` are gitignored (the directory itself is tracked via its `.gitignore` placeholder). Build it
once (150 chunks of 256 tokens, as counted by the served model's tokenizer, from four
public-domain Gutenberg books, seed 42; it needs network access to gutenberg.org and
huggingface.co, plus `pip install transformers requests`):

```bash
python3 benchmarks/fetch_human_corpus.py --model-tokenizer Qwen/Qwen2.5-1.5B-Instruct --out benchmarks/data/human_corpus.jsonl
```

The script's default tokenizer is the 0.5B sibling, `Qwen/Qwen2.5-0.5B-Instruct`, whose
`tokenizer.json`, `vocab.json`, `merges.txt` and `tokenizer_config.json` are sha256-identical
to the 1.5B's at HF `main`, so the flag does not change the output here; pass your own
model's HF id for another family, because the 256-token chunk boundaries (and therefore
the detector's `num_scored_tokens`) depend on it, and expect a different sha256. The file
scored here has sha256 `8601024c868eb704bfe5f2b3e7c536ae4eb1a7f3970bcf524619d24039a04195`
(150 rows); regenerating it with the script defaults and with the flag above both
reproduced that hash byte-for-byte (`EXECUTED` locally, 2026-09-16, EXPERIMENTS.md "Human
corpus provenance"). Because the script fetches live text and the tokenizer at HF `main`
(it has no `--revision` flag), an identical file is only guaranteed while gutenberg.org
serves the same files and the tokenizer revision is unchanged. `--human-jsonl`/`--human-n`
are optional: leave them out to skip the false-positive check, and the script now refuses to
start if the file is missing rather than failing after the GPU run. Note that the demo
requests set only
`temperature` and `top_p`, so the model's other shipped defaults (`top_k 20`,
`repetition_penalty 1.1`; section 4.3) applied to every run in this guide.

```bash
python3 scripts/native-watermark-demo.py --vllm-url $VLLM --detector-url $DET --cacert cluster/router-ca.crt \
  --n 10 --max-tokens 300 --temperature 1.0 --top-p 1.0 \
  --human-jsonl benchmarks/data/human_corpus.jsonl --human-n 50 \
  --json-out /tmp/native-demo.json
```

Real output from this run (`EXECUTED`; full table in `EXPERIMENTS.md`):

```
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
...
[6/10] Explain why some metals conduct electricity so much better than others.
  watermarked    tokens= 172  scored= 170  score=   354.48  p=3.804e-28  -> WATERMARKED
  opt-out        tokens= 282  scored= 281  score=   302.62  p=1.007e-01  -> not detected
...
summary over 10 prompts: watermarked detected 10/10 (TPR 1.000); opt-out flagged 0/10 (FPR 0.000); elapsed 59s

human-written passages (human_corpus.jsonl): flagged 1/50 (FPR 0.020); min p=4.062e-03 median p=5.211e-01
```

Even a 172-token answer (prompt 6) came back at p ≈ 4e-28; the weakest of the ten was
prompt 3 at p ≈ 3.6e-14, still more than eleven orders of magnitude under the 0.01
threshold. In the summary lines, TPR (true-positive rate) is the share of watermarked
answers the detector flagged and FPR (false-positive rate) the share of unwatermarked
texts — opt-out answers, human passages — it wrongly flagged. One of the 50 human passages
scored p = 0.004 and was flagged at the 0.01 threshold: that is what a 1 % threshold means, and
it is why upstream tells you to measure the false-positive rate on your own unwatermarked
traffic with your key before acting on verdicts.

### 4.3 The same run at the model's default temperature

Qwen2.5-1.5B-Instruct ships `generation_config.json` with `temperature 0.7`,
`top_p 0.8`, `top_k 20` and `repetition_penalty 1.1` (`OFFICIAL-SRC`, the model's
[`generation_config.json`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/main/generation_config.json);
the 7B variant ships `repetition_penalty 1.05`,
[generation_config.json](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/generation_config.json),
fetched 2026-09-16, EXPERIMENTS.md, so re-check for other models). vLLM adopts
them as its defaults and fills in each field a request leaves unset (`EXECUTED`: engine
startup log `WARNING [model.py:1772] Default vLLM sampling parameters have been
overridden by the model's generation_config.json: {'repetition_penalty': 1.1,
'temperature': 0.7, 'top_k': 20, 'top_p': 0.8}`, and a seeded opt-out request that
omitted `top_k`/`repetition_penalty` produced the same text as one passing `top_k 20,
repetition_penalty 1.1` and a different text from one passing `top_k 0,
repetition_penalty 1.0`; EXPERIMENTS.md 2026-09-16. `STATIC`:
`vllm/entrypoints/openai/chat_completion/protocol.py` `to_sampling_params` @cd10ed6f
falls back per field). The demo script only ever sends `temperature` and `top_p`, so
`top_k 20` and `repetition_penalty 1.1` were already in effect in section 4.2; passing
`-1` for both flags drops temperature 1.0 → 0.7 and top_p 1.0 → 0.8 and changes nothing
else. (Start vLLM with `--generation-config vllm`, the remedy the engine names in that
warning, if you want its neutral defaults instead.) Less randomness means less signal:

```bash
python3 scripts/native-watermark-demo.py --vllm-url $VLLM --detector-url $DET --cacert cluster/router-ca.crt \
  --n 10 --max-tokens 300 --temperature -1 --top-p -1
```

Real result (`EXECUTED`): 9 of 10 watermarked answers detected, 0 of 10 opt-outs flagged. The
one miss scored p = 0.0198, just above the 0.01 line, and the detected ones sat between
p ≈ 4e-4 and 4e-10 instead of the 3.6e-14 to 5.2e-76 seen at temperature 1.0 (fact E8;
the figures are those of the recorded run, not fixed properties of the setup). Same model,
same key, same ten prompts, same `max_tokens` cap; the only change is how much randomness
the sampler was allowed.

And at temperature 0 (`--temperature 0 --n 5`), nothing is detected: 0 of 5, and the
"watermarked" and "opt-out" texts are byte-identical, because greedy decoding never rolls the
dice. The engine says so in its log:

```
WARNING [gpu_sampler.py:45] Watermarking is enabled, but greedy decoding (temperature=0) cannot be watermarked. This request will use ordinary greedy sampling.
```

This is the single most important operational fact about the scheme: detection
strength depends on how much the model was allowed to choose. Decide your
serving temperature with detection in mind, and measure the rate on your own
traffic before relying on the verdict.

---

## 5. Things to know before using this for real

- **The opt-out field.** `"watermarking": false` is honoured from any client.
  If watermarking is a policy, strip or reject that field at your gateway.
  Likewise decide what you do about `temperature: 0` and `top_k: 1`, which also
  yield unwatermarked text (the second without any log line; section 1) and cannot
  be "stripped" without changing the request.
- **Regenerate returns the same answer.** With one key, the same prompt at the same
  sampling settings normally yields the same watermarked text at any non-zero
  temperature, and `seed` does not change it: `seed` steers only positions sampled
  without the watermark, that is opt-out requests, greedy requests, and positions whose
  4-token context already occurred in the answer, which the default
  `deduplicate_contexts: single_turn` hands to ordinary sampling (`OFFICIAL-SRC`, upstream
  doc "Gumbel-max", `deduplicate_contexts`; `STATIC`, `gpu_sampler.py` L91-104 @cd10ed6f;
  `EXECUTED`, section 1). `n: 2` did return two different completions in this run, neither
  equal to the single-request text, so batching changes the answer where `seed` does not
  (`EXECUTED`; cause not isolated, `OPEN`). If your product has a "regenerate" action, do
  not rely on it for variety: vary the prompt or the sampling, or rotate keys, which
  changes the whole answer set and must be tracked for detection (see "Key handling").
  Upstream's RFC lists diversity-restoring variants; this build ships `dual_key_gumbel`,
  whose per-token key choice goes through the seeded sampler
  (`vllm/v1/watermarking/gumbel.py` L162-171 @cd10ed6f, `STATIC`; not exercised here,
  `OPEN`).
- **Key handling.** The key ends up as an integer on the vLLM command line.
  Observed on this run (`EXECUTED`, EXPERIMENTS.md "Key exposure check"): the
  API-server startup log redacts the argument (`non-default args: {...,
  'watermark_config': '***'}`; upstream lists `watermark_config` in
  `_SENSITIVE_ARG_FIELDS`, `vllm/entrypoints/serve/utils/api_utils.py` @cd10ed6f,
  `STATIC`), and the engine's `core.py:123` config line contains no `watermark` field
  at all (`EXECUTED`). `WatermarkConfig.key` is declared `key: int = Field(ge=0,
  repr=False, exclude=True)` (`vllm/config/watermarking.py:31` @cd10ed6f, `STATIC`), so pydantic
  reprs and JSON dumps should omit it; that was not exercised here. `/server_info` is
  a development endpoint mounted only when `VLLM_SERVER_DEV_MODE=1`
  (`vllm/entrypoints/launchers/api_server/routers.py:34-37`, `vllm/envs.py:167`
  @cd10ed6f, `STATIC`); on this deployment it returned HTTP 404 `{"detail":"Not
  Found"}` while `/version` returned 200 (`EXECUTED`), so it is not an exposure path
  here and what it would show with dev mode on was not tested (`OPEN`; note the text
  form, `str(vllm_config)`, never lists `watermark_config`, so only
  `?config_format=json` could say anything about the key). Never enable dev mode on a
  production serving pod: it also mounts sleep, RPC and cache-reset endpoints. The
  Deployment spec shows only `${WATERMARK_KEY}`, but `ps` inside the container shows
  the full `--watermark-config` JSON (`EXECUTED`). So anyone who can `exec` into the
  pod, or read the Secret, has the key. Keep RBAC on the
  namespace tight, record which key served which traffic, and plan rotation: the
  detector must be told every key that may have produced a text (and correct for
  multiple tests).
- **Keep detector and generator on the same build** until upstream's
  compatibility tests land (open PRs #56801/#56809). The PRF (section 1) is
  versioned, but this is new code.
- **Do not expose `/detect` publicly.** Scores tell an attacker exactly which
  edits weaken the mark. Internal service only; the Route in `20-detector.yaml`
  exists for this demo.
- **Measure false positives with your key.** The p-value assumes independent
  scores; real traffic with repeated boilerplate can drift. Upstream's own
  guidance is to measure the realised rate on unwatermarked traffic before
  trusting `is_watermarked`.
- **Not covered by this scheme** (`OFFICIAL-SRC`, upstream doc "Limitations"; fact
  E6): beam search, and models that replace the vLLM sampler with a custom one.
  Upstream scopes the feature to text ("generated token choices"); non-text outputs and
  multimodal-input models (image/audio in, text out) were not tested here, only the
  text-only `Qwen/Qwen2.5-1.5B-Instruct` was run (`OPEN`). Speculative decoding is
  possible but constrained (`OFFICIAL-SRC`, upstream doc "Speculative decoding"; fact
  E6): it requires `draft_sample_method: probabilistic`, `rejection_sample_method:
  standard` and an autoregressive draft method (`dspark`, `eagle`, `eagle3` or `mtp`;
  parallel drafting only with `dspark`). Only `dual_key_gumbel` watermarks every token
  under it; the plain `gumbel` used here is accepted only with
  `"allow_target_only_watermarking": true`, which leaves accepted draft tokens
  unwatermarked and dilutes the signal in proportion to their share. Generation-side
  context deduplication is not applied on the speculative token paths. Speculative
  decoding was not part of this run (`EXECUTED`: the engine log shows
  `speculative_config=None`).
- **Robustness is bounded.** Insertions, deletions and substitutions break the
  `context_width` contexts that follow them (`OFFICIAL-SRC`, upstream doc). Published
  paraphrase and back-translation attacks on other sampling watermarks remove most of
  the signal: KGW 99.8 % → 9.7 % TPR@1%FPR (true-positive rate at the threshold that
  yields a 1 % false-positive rate) after recursive paraphrase, SynthID F1
  1.0 → 0.71 after one back-translation (`CORROBORATED`, [fact B17](facts.md)).
  Robustness of this Gumbel-max build was not measured in this run (`OPEN`,
  EXPERIMENTS.md 2026-09-16).

---

## 6. Clean up

```bash
oc delete namespace watermark-demo
./scripts/scale-gpu.sh 0          # the GPU node is billable (script not run on this cluster; recorded 2026-08-08 on the previous one, section 2)
```

## Appendix: image pinning and (optional) building your own

The manifests pin the nightly by manifest-list digest
(`vllm/vllm-openai@sha256:20a52b807cef7a3be15d2174d293b5642d437615b20b5930d78a927e709e6bb6`,
commit `cd10ed6f…`, CUDA toolkit 13.0.2, distinct from the node driver's CUDA 13.2 in
section 2). To move to another build:

1. Pick a tag on Docker Hub (`nightly-<commit>` or a release tag) and confirm it
   contains both watermark PRs:
   `gh api repos/vllm-project/vllm/compare/7ee8a6dd013819838da8012ca549d724bee7c6c6...<commit> --jq .status`
   must print `ahead` or `identical`. `7ee8a6dd` is the merge of #56122 (dual-key
   Gumbel-max and generation-side context deduplication; also tag `v0.29.1rc0`) and
   already descends from the #54053 merge `ea40bb9e`, so one check covers both PRs
   (`OFFICIAL-SRC`, compare API: `ea40bb9e...7ee8a6dd` is `ahead` by 141, behind 0;
   fact E3). `behind` or `diverged` (v0.29.0 reports `diverged`) means the build lacks
   the follow-up or the whole feature. Checking against `ea40bb9e` alone is not enough:
   the 2026-09-10 to 09-12 nightlies pass it but have only #54053. The first nightly
   that passes is `nightly-2671fedfc7ae604761990603fc736c0c4f21de57` (2026-09-13).
2. Resolve its digest: `skopeo inspect docker://vllm/vllm-openai:<tag> | jq -r .Digest`.
3. Replace the `image:` line in both manifests, and redeploy generator **and**
   detector together.

You do not need to build anything. If you want a private copy (air-gapped
registry, or an entrypoint wrapper of your own), `skopeo copy
docker://vllm/vllm-openai@sha256:… docker://<your-registry>/vllm-openai:<tag>`
mirrors the image unchanged; a derived image is just `FROM` that digest.
