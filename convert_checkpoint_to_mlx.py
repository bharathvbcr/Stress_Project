#!/usr/bin/env python3
"""Convert a PyTorch/Lightning checkpoint to MLX weights for Apple Silicon inference."""
from __future__ import annotations

import argparse
import json
import logging
import sys

from mlx_convert import convert_checkpoint_to_mlx
from utils import infer_model_feature_dims, load_config, setup_logging

log = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    parser = argparse.ArgumentParser(description="Convert PyTorch checkpoint to MLX .npz weights.")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--checkpoint", required=True, help="Source .ckpt or .pth checkpoint.")
    parser.add_argument(
        "--output",
        default=None,
        help="Destination .mlx.npz path (default: alongside checkpoint with _mlx.npz suffix).",
    )
    parser.add_argument("--no-validate", action="store_true", help="Skip forward-pass validation.")
    parser.add_argument("--atol", type=float, default=0.05, help="Validation tolerance (max abs diff).")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    if not config:
        log.error("Failed to load config: %s", args.config)
        return 1

    seq_dim, static_dim = infer_model_feature_dims(config)
    output = args.output
    if output is None:
        if args.checkpoint.endswith(".ckpt"):
            output = args.checkpoint[:-5] + "_mlx.npz"
        else:
            output = args.checkpoint + ".mlx.npz"

    report = convert_checkpoint_to_mlx(
        config,
        args.checkpoint,
        output,
        input_dim_sequence=seq_dim,
        input_dim_static=static_dim,
        validate=not args.no_validate,
        atol=args.atol,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
