"""A dependency-light Canny edge detector implemented from first principles."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from vision2autonomy.image.convolution import convolve2d
from vision2autonomy.image.filters import gaussian_blur

SOBEL_X = np.array([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
SOBEL_Y = SOBEL_X.T


@dataclass(frozen=True)
class CannyResult:
    """Intermediate products that make the Canny pipeline inspectable."""

    smoothed: NDArray[np.float64]
    magnitude: NDArray[np.float64]
    direction: NDArray[np.float64]
    suppressed: NDArray[np.float64]
    edges: NDArray[np.uint8]


def sobel_gradients(
    image: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return horizontal/vertical gradients, magnitude, and angle in degrees."""

    gx = convolve2d(image, SOBEL_X, padding="reflect")
    gy = convolve2d(image, SOBEL_Y, padding="reflect")
    magnitude = np.hypot(gx, gy)
    direction = (np.rad2deg(np.arctan2(gy, gx)) + 180.0) % 180.0
    return gx, gy, magnitude, direction


def non_maximum_suppression(
    magnitude: ArrayLike,
    direction: ArrayLike,
) -> NDArray[np.float64]:
    """Thin gradient ridges by retaining local maxima along gradient direction."""

    mag = np.asarray(magnitude, dtype=np.float64)
    angle = np.asarray(direction, dtype=np.float64) % 180.0
    if mag.ndim != 2 or angle.shape != mag.shape:
        raise ValueError("magnitude and direction must be equally shaped 2-D arrays")

    output = np.zeros_like(mag)
    if min(mag.shape) < 3:
        return output

    center = mag[1:-1, 1:-1]
    inner_angle = angle[1:-1, 1:-1]
    comparisons = (
        ((inner_angle < 22.5) | (inner_angle >= 157.5), mag[1:-1, :-2], mag[1:-1, 2:]),
        ((inner_angle >= 22.5) & (inner_angle < 67.5), mag[:-2, 2:], mag[2:, :-2]),
        ((inner_angle >= 67.5) & (inner_angle < 112.5), mag[:-2, 1:-1], mag[2:, 1:-1]),
        ((inner_angle >= 112.5) & (inner_angle < 157.5), mag[:-2, :-2], mag[2:, 2:]),
    )
    kept = np.zeros_like(center, dtype=bool)
    for sector, before, after in comparisons:
        kept |= sector & (center >= before) & (center >= after)
    output[1:-1, 1:-1] = np.where(kept, center, 0.0)
    return output


def double_threshold(
    image: ArrayLike,
    low_threshold: float,
    high_threshold: float,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Classify non-zero responses into strong and weak edge candidates."""

    values = np.asarray(image, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("image must be two-dimensional")
    if low_threshold < 0 or high_threshold <= low_threshold:
        raise ValueError("thresholds must satisfy 0 <= low_threshold < high_threshold")

    strong = values >= high_threshold
    weak = (values >= low_threshold) & ~strong & (values > 0)
    return strong, weak


def hysteresis(strong: ArrayLike, weak: ArrayLike) -> NDArray[np.uint8]:
    """Keep weak pixels connected to strong pixels through an 8-neighbour path."""

    strong_mask = np.asarray(strong, dtype=bool)
    weak_mask = np.asarray(weak, dtype=bool).copy()
    if strong_mask.ndim != 2 or weak_mask.shape != strong_mask.shape:
        raise ValueError("strong and weak must be equally shaped 2-D arrays")

    edges = strong_mask.copy()
    queue = deque(map(tuple, np.argwhere(strong_mask)))
    height, width = edges.shape

    while queue:
        row, column = queue.popleft()
        for next_row in range(max(0, row - 1), min(height, row + 2)):
            for next_column in range(max(0, column - 1), min(width, column + 2)):
                if weak_mask[next_row, next_column] and not edges[next_row, next_column]:
                    edges[next_row, next_column] = True
                    queue.append((next_row, next_column))

    return (edges * 255).astype(np.uint8)


def canny(
    image: ArrayLike,
    *,
    low_threshold: float = 40.0,
    high_threshold: float = 100.0,
    gaussian_size: int = 5,
    sigma: float = 1.4,
    return_intermediates: bool = False,
) -> NDArray[np.uint8] | CannyResult:
    """Detect edges in a grayscale image using the complete Canny pipeline."""

    source = np.asarray(image)
    if source.ndim != 2:
        raise ValueError("Canny expects a two-dimensional grayscale image")
    if source.size == 0:
        raise ValueError("image must not be empty")
    if not np.isfinite(source).all():
        raise ValueError("image must contain only finite values")

    smoothed = gaussian_blur(source, size=gaussian_size, sigma=sigma)
    _, _, magnitude, direction = sobel_gradients(smoothed)
    suppressed = non_maximum_suppression(magnitude, direction)
    strong, weak = double_threshold(suppressed, low_threshold, high_threshold)
    edges = hysteresis(strong, weak)

    if return_intermediates:
        return CannyResult(smoothed, magnitude, direction, suppressed, edges)
    return edges

