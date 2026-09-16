#!/usr/bin/env bash
# Deploy (or redeploy) the native-watermarking demo: vLLM with --watermark-config
# plus the reference detector. See docs/guide-native-watermarking.md.
#
# Prereqs: KUBECONFIG pointing at a cluster with one schedulable GPU node
# (scripts/install-gpu-operators.sh + scripts/create-gpu-machineset.sh + scripts/scale-gpu.sh 1);
# it defaults to cluster/auth/kubeconfig, which exists only if `openshift-install ... --dir cluster`
# wrote it. And a key file whose only non-blank, non-comment line is `WATERMARK_KEY=<integer 0..2^64-1>`
# (never commit it):
#   mkdir -p cluster && python3 -c 'import secrets; print("WATERMARK_KEY=%d" % secrets.randbits(64))' > cluster/gumbel-key.env
# WATERMARK_KEY_FILE=<path> overrides the default key file location.
# -E: without errtrace bash does not run the ERR trap for a failure inside a function,
# and the diagnostics in on_fail below are what a failed wait_for should print.
set -Eeuo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export KUBECONFIG="${KUBECONFIG:-${repo_dir}/cluster/auth/kubeconfig}"
key_file="${WATERMARK_KEY_FILE:-${repo_dir}/cluster/gumbel-key.env}"
ns=watermark-demo

if [[ ! -f "$KUBECONFIG" ]]; then
  echo "KUBECONFIG=$KUBECONFIG not found; export KUBECONFIG=<your kubeconfig> (oc does not fall back to ~/.kube/config)" >&2
  exit 2
fi
if [[ ! -f "$key_file" ]]; then
  echo "key file $key_file not found (see header comment)" >&2
  exit 2
fi
# oc --from-env-file skips blank and #-comment lines; every other line must be the single key line.
if [[ $(grep -cvE '^[[:space:]]*(#|$)' "$key_file") -ne 1 ]] \
   || ! grep -qE '^WATERMARK_KEY=[0-9]+$' "$key_file"; then
  echo "$key_file must contain exactly one non-comment line: WATERMARK_KEY=<non-negative integer>" >&2
  exit 2
fi

oc apply -f "${repo_dir}/deploy/native/00-namespace.yaml"
# The key is the only secret; it is created from the local file, never from a manifest.
oc -n "$ns" create secret generic watermark-key --from-env-file="$key_file" \
  --dry-run=client -o yaml | oc apply -f -
oc apply -f "${repo_dir}/deploy/native/10-vllm.yaml"
oc apply -f "${repo_dir}/deploy/native/20-detector.yaml"

# Wait for a Deployment. `oc rollout status` is the right primary wait (it tracks the
# new ReplicaSet on redeploys), but once the Deployment's Progressing condition carries
# ProgressDeadlineExceeded it fails immediately on every call until the pod is Ready, so
# fall back to the Available condition, which the progress deadline does not affect.
# The vLLM timeout exceeds the manifest's progressDeadlineSeconds (1800) on purpose.
wait_for() { # $1 = deployment name, $2 = timeout
  oc -n "$ns" rollout status "deployment/$1" --timeout="$2" \
    || oc -n "$ns" wait --for=condition=Available "deployment/$1" --timeout="$2"
}
on_fail() {
  echo "wait failed; current state:" >&2
  oc -n "$ns" get pods >&2 || true
  oc -n "$ns" logs deploy/vllm-watermark --tail=20 >&2 || true
}
trap on_fail ERR

echo "Waiting for the detector (CPU) and vLLM (GPU; first start pulls the image and downloads the model) ..."
wait_for watermark-detector 15m
wait_for vllm-watermark 35m
trap - ERR

echo
echo "vLLM:     https://$(oc -n "$ns" get route vllm-watermark -o jsonpath='{.spec.host}')"
echo "detector: https://$(oc -n "$ns" get route watermark-detector -o jsonpath='{.spec.host}')"
