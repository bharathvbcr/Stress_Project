#!/usr/bin/env python3
"""Batch-convert PyTorch/Lightning checkpoints under a directory to MLX weights."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from mlx_convert import convert_checkpoint_to_mlx, ensure_mlx_checkpoint
from utils import infer_model_feature_dims, load_config, setup_logging

log = logging.getLogger(__name__)

_CHECKPOINT_SUFFIXES = (".ckpt", ".pth", ".pt")


def _iter_checkpoints(root: Path, recursive: bool) -> list[Path]:
    if root.is_file():
        return [root] if root.suffix in _CHECKPOINT_SUFFIXES else []
    pattern = "**/*" if recursive else "*"
    paths = []
    for path in sorted(root.glob(pattern)):
        if not path.is_file():
            continue
        if path.suffix not in _CHECKPOINT_SUFFIXES:
            continue
        if path.name.endswith("_mlx.npz"):
            continue
        if path.suffix == ".npz":
            continue
        paths.append(path)
    return paths


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Convert all PyTorch checkpoints in a directory to MLX .npz weights."
    )
    parser.add_argument("--config", default="config.json")
    parser.add_argument(
        "--input-dir",
        default="outputs/models",
        help="Directory (or single checkpoint file) to convert.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory. Defaults to mlx_cache beside each checkpoint.",
    )
    parser.add_argument("--recursive", action="store_true", help="Search subdirectories.")
    parser.add_argument("--no-validate", action="store_true")
    parser.add_argument("--atol", type=float, default=0.05)
    parser.add_argument("--use-cache", action="store_true", help="Use ensure_mlx_checkpoint caching.")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    if not config:
        log.error("Failed to load config: %s", args.config)
        return 1

    seq_dim, static_dim = infer_model_feature_dims(config)
    root = Path(args.input_dir)
    checkpoints = _iter_checkpoints(root, args.recursive)
    if not checkpoints:
        log.error("No checkpoints found under %s", root)
        return 1

    reports = []
    failures = 0
    for ckpt in checkpoints:
        try:
            if args.use_cache:
                out = ensure_mlx_checkpoint(
                    config,
                    str(ckpt),
                    seq_dim,
                    static_dim,
                    cache_dir=str(args.output_dir) if args.output_dir else None,
                )
                reports.append({"checkpoint": str(ckpt), "output_path": out, "mode": "cache"})
                log.info("Cached MLX weights: %s -> %s", ckpt, out)
                continue

            if args.output_dir:
                out_dir = Path(args.output_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                out = out_dir / f"{ckpt.stem}.mlx.npz"
            else:
                out = ckpt.with_name(f"{ckpt.stem}.mlx.npz")

            report = convert_checkpoint_to_mlx(
                config,
                str(ckpt),
                str(out),
                input_dim_sequence=seq_dim,
                input_dim_static=static_dim,
                validate=not args.no_validate,
                atol=args.atol,
            )
            report["checkpoint"] = str(ckpt)
            reports.append(report)
            log.info("Converted %s -> %s", ckpt, out)
        except Exception as exc:
            failures += 1
            log.error("Failed to convert %s: %s", ckpt, exc)

    summary_path = (Path(args.output_dir) if args.output_dir else root) / "mlx_batch_convert_report.json"
    try:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(reports, indent=2), encoding="utf-8")
        log.info("Wrote batch report to %s", summary_path)
    except OSError as exc:
        log.warning("Could not write batch report: %s", exc)

    if failures:
        log.error("Batch conversion finished with %d failure(s).", failures)
        return 1
    log.info("Batch conversion complete (%d checkpoint(s)).", len(reports))
    return 0


if __name__ == "__main__":
    sys.exit(main())
