"""Generate deterministic reference images used by Chapters 01 through 03.

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
from vision2autonomy.features.harris import HarrisResult, detect_harris_corners
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

    converted = [(label, Image.fromarray(to_uint8(image)).convert("RGB")) for label, image in images.items()]
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
        grid.paste(image, (x, y + label_height))
        draw.text((x + 8, y + 9), label, fill="black", font=font)
    return grid


def harris_scene(size: int = 256) -> np.ndarray:
    """Create a clean geometric scene with flat, edge, and corner regions."""

    canvas = Image.new("L", (size, size), color=22)
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((64, 64, 192, 192), fill=220)
    draw.polygon(((28, 220), (100, 136), (156, 220)), fill=130)
    draw.line((164, 36, 228, 100), fill=250, width=8)
    return np.asarray(canvas, dtype=np.uint8)


def signed_response_heatmap(response: np.ndarray) -> np.ndarray:
    """Map positive corner responses to warm colors and negative edges to blue."""

    values = np.asarray(response, dtype=np.float64)
    scale = float(np.max(np.abs(values), initial=0.0))
    normalized = values / scale if scale > 0.0 else np.zeros_like(values)
    positive = np.clip(normalized, 0.0, 1.0)
    negative = np.clip(-normalized, 0.0, 1.0)
    heatmap = np.zeros((*values.shape, 3), dtype=np.uint8)
    heatmap[..., 0] = np.uint8(255 * np.sqrt(positive))
    heatmap[..., 1] = np.uint8(210 * positive)
    heatmap[..., 2] = np.uint8(255 * np.sqrt(negative))
    return heatmap


def draw_corner_overlay(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Draw detected Harris corners as red rings on a grayscale image."""

    overlay = Image.fromarray(to_uint8(image)).convert("RGB")
    draw = ImageDraw.Draw(overlay)
    for row, column in corners:
        radius = 6
        draw.ellipse(
            (column - radius, row - radius, column + radius, row + radius),
            outline=(255, 45, 45),
            width=2,
        )
    return np.asarray(overlay)


def save_harris_window_animation(
    image: np.ndarray,
    result: HarrisResult,
    path: Path,
) -> None:
    """Animate a local window moving through flat, edge, and corner regions."""

    key_positions = [(128, 128), (64, 128), (64, 64)]
    centers: list[tuple[int, int]] = []
    for start, end in zip(key_positions, key_positions[1:]):
        for fraction in np.linspace(0.0, 1.0, 10, endpoint=False):
            row = int(round(start[0] + fraction * (end[0] - start[0])))
            column = int(round(start[1] + fraction * (end[1] - start[1])))
            centers.append((row, column))
    centers.extend([key_positions[-1]] * 6)

    eigen_scale = float(result.lambda_max.max(initial=1.0))
    font = ImageFont.load_default()
    frames: list[Image.Image] = []
    for row, column in centers:
        frame = Image.new("RGB", (256, 330), "white")
        frame.paste(Image.fromarray(image).convert("RGB"), (0, 0))
        draw = ImageDraw.Draw(frame)
        radius = 14
        draw.rectangle(
            (column - radius, row - radius, column + radius, row + radius),
            outline=(255, 45, 45),
            width=3,
        )
        draw.ellipse((column - 3, row - 3, column + 3, row + 3), fill=(40, 200, 80))

        lambda_1 = float(result.lambda_max[row, column])
        lambda_2 = float(result.lambda_min[row, column])
        relative_1 = lambda_1 / eigen_scale
        relative_2 = lambda_2 / eigen_scale
        if relative_1 < 0.02:
            region = "FLAT: both changes are small"
        elif lambda_2 < 0.1 * lambda_1:
            region = "EDGE: one direction changes"
        else:
            region = "CORNER: both directions change"

        draw.text((10, 268), region, fill="black", font=font)
        draw.text((10, 290), "lambda max", fill="black", font=font)
        draw.rectangle((82, 290, 82 + int(150 * relative_1), 299), fill=(245, 150, 35))
        draw.text((10, 308), "lambda min", fill="black", font=font)
        draw.rectangle((82, 308, 82 + int(150 * relative_2), 317), fill=(45, 150, 240))
        frames.append(frame)

    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=140,
        loop=0,
        optimize=True,
        disposal=2,
    )


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
    corner_source = harris_scene()
    harris = detect_harris_corners(
        corner_source,
        threshold_rel=0.03,
        min_distance=12,
        max_corners=16,
    )

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
        "chapter03_harris_stages.png": labelled_grid(
            {
                "1. Input: flat, edges, corners": corner_source,
                "2. Largest eigenvalue": to_uint8(harris.lambda_max, normalize=True),
                "3. Smallest eigenvalue": to_uint8(harris.lambda_min, normalize=True),
                "4. Harris response: warm=corner": signed_response_heatmap(harris.response),
                "5. Detected corners": draw_corner_overlay(corner_source, harris.corners),
            }
        ),
        "chapter03_harris_corners.png": Image.fromarray(
            draw_corner_overlay(corner_source, harris.corners)
        ),
    }

    paths = []
    for filename, image in images.items():
        path = output_dir / filename
        image.save(path, format="PNG", optimize=True)
        paths.append(path)
    animation_path = output_dir / "chapter03_harris_window.gif"
    save_harris_window_animation(corner_source, harris, animation_path)
    paths.append(animation_path)
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
