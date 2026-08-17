"""Command-line entry points for Vision2Autonomy."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from vision2autonomy.edges.canny import canny


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the from-scratch Canny detector.")
    parser.add_argument("input", type=Path, help="input image path")
    parser.add_argument("output", type=Path, help="output edge-map path")
    parser.add_argument("--low", type=float, default=40.0, help="low threshold")
    parser.add_argument("--high", type=float, default=100.0, help="high threshold")
    parser.add_argument("--sigma", type=float, default=1.4, help="Gaussian sigma")
    parser.add_argument("--kernel-size", type=int, default=5, help="odd Gaussian kernel size")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    grayscale = np.asarray(Image.open(args.input).convert("L"))
    edges = canny(
        grayscale,
        low_threshold=args.low,
        high_threshold=args.high,
        gaussian_size=args.kernel_size,
        sigma=args.sigma,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(edges).save(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

