"""Generate deterministic reference images used by Chapters 01 and 02.

Run after installing the project:

    python -m vision2autonomy.examples.reference_images
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from vision2autonomy.edges.canny import CannyResult, canny
from vision2autonomy.image.convolution import convolve2d
from vision2autonomy.image.filters import gaussian_blur

DEFAULT_OUTPUT = Path("docs/assets")


def synthetic_scene(size: int = 256, seed: int = 7) -> np.ndarray:
    """Create a deterministic scene containing several edge types and noise."""

    if size < 64:
        raise ValueError("size must be at least 64 pixels")

    canvas = Image.new("L", (size, size), color=28)
    draw = ImageDraw.Draw(canvas)
    margin = size // 12
    draw.rectangle(
        (margin, margin, size // 2 - margin // 2, size // 2),
        fill=185,
        outline=245,
        width=max(2, size // 96),
    )
    draw.ellipse(
        (size // 2, margin, size - margin, size // 2),
        fill=105,
        outline=225,
        width=max(2, size // 96),
    )
    draw.polygon(
        ((size // 2, size // 2), (margin, size - margin), (size - margin, size - margin)),
        fill=70,
        outline=210,
    )
    draw.line(
        (margin, size * 3 // 4, size - margin, size // 2),
        fill=250,
        width=max(2, size // 64),
    )

    image = np.asarray(canvas, dtype=np.float64)
    noise = np.random.default_rng(seed).normal(0.0, 8.0, image.shape)
    return np.clip(image + noise, 0, 255).astype(np.uint8)


def to_uint8(array: np.ndarray, *, normalize: bool = False) -> np.ndarray:
    """Convert a numerical image to display-safe unsigned bytes."""

    values = np.asarray(array, dtype=np.float64)
    if normalize:
        minimum, maximum = values.min(), values.max()
        if maximum > minimum:
            values = (values - minimum) * (255.0 / (maximum - minimum))
        else:
            values = np.zeros_like(values)
    return np.clip(values, 0, 255).astype(np.uint8)


def labelled_grid(images: Mapping[str, np.ndarray], columns: int = 3) -> Image.Image:
    """Compose equally sized grayscale arrays into a labelled reference grid."""

    if not images:
        raise ValueError("images must not be empty")
    if columns < 1:
        raise ValueError("columns must be positive")

    converted = [(label, Image.fromarray(to_uint8(image)).convert("L")) for label, image in images.items()]
    width, height = converted[0][1].size
    if any(item.size != (width, height) for _, item in converted):
        raise ValueError("all grid images must have the same dimensions")

    label_height = 30
    rows = (len(converted) + columns - 1) // columns
    grid = Image.new("RGB", (columns * width, rows * (height + label_height)), "white")
    draw = ImageDraw.Draw(grid)
    font = ImageFont.load_default()
    for index, (label, image) in enumerate(converted):
        row, column = divmod(index, columns)
        x, y = column * width, row * (height + label_height)
        grid.paste(image.convert("RGB"), (x, y + label_height))
        draw.text((x + 8, y + 9), label, fill="black", font=font)
    return grid


def generate_reference_images(output_dir: Path = DEFAULT_OUTPUT) -> list[Path]:
    """Generate all committed reference images and return their paths."""

    output_dir.mkdir(parents=True, exist_ok=True)
    source = synthetic_scene()
    blurred = gaussian_blur(source, size=5, sigma=1.4)
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=float)
    sharpened = convolve2d(source, sharpen_kernel, padding="reflect")

    result = canny(
        source,
        low_threshold=35,
        high_threshold=90,
        gaussian_size=5,
        sigma=1.4,
        return_intermediates=True,
    )
    assert isinstance(result, CannyResult)

    images: dict[str, Image.Image] = {
        "chapter01_input.png": Image.fromarray(source),
        "chapter01_convolution.png": labelled_grid(
            {
                "Input: noisy synthetic scene": source,
                "Gaussian blur: 5x5, sigma=1.4": to_uint8(blurred),
                "Sharpen convolution: 3x3": to_uint8(sharpened),
            }
        ),
        "chapter02_canny_stages.png": labelled_grid(
            {
                "1. Input": source,
                "2. Gaussian smoothing": to_uint8(result.smoothed),
                "3. Gradient magnitude": to_uint8(result.magnitude, normalize=True),
                "4. Gradient direction": to_uint8(result.direction * (255.0 / 180.0)),
                "5. Non-maximum suppression": to_uint8(result.suppressed, normalize=True),
                "6. Hysteresis result": result.edges,
            }
        ),
        "chapter02_canny_edges.png": Image.fromarray(result.edges),
    }

    paths = []
    for filename, image in images.items():
        path = output_dir / filename
        image.save(path, format="PNG", optimize=True)
        paths.append(path)
    return paths


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    for path in generate_reference_images(args.output_dir):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

