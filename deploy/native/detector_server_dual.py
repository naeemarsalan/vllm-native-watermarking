#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Extended detection server. Upstream ships
# examples/basic/online_serving/watermark_detection_server.py, which only wires
# up GumbelWatermarkDetector (single key). Dual-key Gumbel-max needs
# DualKeyGumbelWatermarkDetector, and its `alpha` is a detection-side weight
# that does NOT default to the same value as the generator's alpha:
#
#   DualKeyGumbelWatermarker(..., alpha=0.1)          # generation
#   DualKeyGumbelWatermarkDetector(..., alpha=0.2)    # detection
#
# so a deployer who accepts both defaults is detecting with a different weight
# than they generated with. This server makes the algorithm and alpha explicit
# so that pairing can be measured rather than assumed.
#
# Detector classes are imported from the installed vLLM, so generation and
# detection share one implementation and one PRF version.

import argparse

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from vllm.tokenizers import TokenizerLike, cached_get_tokenizer
from vllm.v1.watermarking import (
    DualKeyGumbelWatermarkDetector,
    GumbelWatermarkDetector,
)

app = FastAPI()
tokenizer: TokenizerLike | None = None
detector = None
CONFIG: dict = {}


class DetectionRequest(BaseModel):
    text: str


class DetectionResponse(BaseModel):
    score: float
    p_value: float
    num_scored_tokens: int
    is_watermarked: bool


@app.get("/config")
def config() -> dict:
    """Report how this detector is configured, so results carry their conditions."""
    return CONFIG


@app.post("/detect")
def detect(request: DetectionRequest) -> DetectionResponse:
    assert tokenizer is not None and detector is not None
    token_ids = tokenizer.encode(request.text, add_special_tokens=False)
    result = detector.detect(token_ids)
    return DetectionResponse(
        score=result.score,
        p_value=result.p_value,
        num_scored_tokens=result.num_scored_tokens,
        is_watermarked=result.is_watermarked,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--key", required=True, type=int)
    parser.add_argument("--algorithm", choices=("gumbel", "dual_key_gumbel"),
                        default="gumbel")
    parser.add_argument("--alpha", type=float, default=None,
                        help="dual-key detection weight for key B; upstream's "
                             "detector default is 0.2 while the generator "
                             "default is 0.1")
    parser.add_argument("--prf", choices=("philox",), default="philox")
    parser.add_argument("--context-width", type=int, default=4)
    parser.add_argument("--p-value-threshold", type=float, default=0.01)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    global tokenizer, detector, CONFIG
    tokenizer = cached_get_tokenizer(args.tokenizer)
    if args.algorithm == "dual_key_gumbel":
        kwargs = dict(
            key=args.key,
            context_width=args.context_width,
            p_value_threshold=args.p_value_threshold,
            prf=args.prf,
        )
        if args.alpha is not None:
            kwargs["alpha"] = args.alpha
        detector = DualKeyGumbelWatermarkDetector(**kwargs)
        effective_alpha = getattr(detector, "alpha", None)
    else:
        detector = GumbelWatermarkDetector(
            key=args.key,
            context_width=args.context_width,
            p_value_threshold=args.p_value_threshold,
            prf=args.prf,
        )
        effective_alpha = None
    CONFIG = {
        "algorithm": args.algorithm,
        "alpha": effective_alpha,
        "prf": args.prf,
        "context_width": args.context_width,
        "p_value_threshold": args.p_value_threshold,
        "tokenizer": args.tokenizer,
    }
    print(f"detector config: {CONFIG}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main(parse_args())
