"""Lucas-Kanade optical flow implemented from first principles.

The brightness constancy assumption I(x, y, t+1) = I(x+u, y+v, t) is
linearized with a first-order Taylor expansion. Inside a small window the
motion is assumed constant, which turns the problem into a 2x2 linear system
per pixel whose solution is the displacement (u, v).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from vision2autonomy.image.convolution import convolve2d

# Central-difference derivative kernels. ``convolve2d`` flips kernels, so these
# are written as correlation kernels: output = (f(x+1) - f(x-1)) / 2.
DX = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, -0.5], [0.0, 0.0, 0.0]])
DY = DX.T


@dataclass(frozen=True)
class OpticalFlowResult:
    """Per-pixel motion and a confidence measure for every location."""

    flow: NDArray[np.float64]       # (H, W, 2) -> (u, v) = (column, row) shift
    confidence: NDArray[np.float64]  # (H, W) smallest eigenvalue of the tensor
    valid: NDArray[np.bool_]         # (H, W) pixels with a well-conditioned system


def _as_gray_pair(first: ArrayLike, second: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    image_a = np.asarray(first, dtype=np.float64)
    image_b = np.asarray(second, dtype=np.float64)
    if image_a.ndim != 2 or image_b.ndim != 2:
        raise ValueError("both frames must be two-dimensional grayscale arrays")
    if image_a.shape != image_b.shape:
        raise ValueError("both frames must have the same shape")
    if image_a.size == 0:
        raise ValueError("frames must not be empty")
    if not np.isfinite(image_a).all() or not np.isfinite(image_b).all():
        raise ValueError("frames must contain only finite values")
    return image_a, image_b


def _box_sum(values: NDArray[np.float64], window_size: int) -> NDArray[np.float64]:
    """Sum every ``window_size`` square with an integral image.

    This is mathematically identical to a uniform convolution with zero
    padding but runs in O(H*W) instead of O(H*W*window_size**2).
    """

    half = window_size // 2
    height, width = values.shape
    integral = np.zeros((height + 1, width + 1), dtype=np.float64)
    integral[1:, 1:] = np.cumsum(np.cumsum(values, axis=0), axis=1)
    top = np.clip(np.arange(height) - half, 0, height)
    bottom = np.clip(np.arange(height) + half + 1, 0, height)
    left = np.clip(np.arange(width) - half, 0, width)
    right = np.clip(np.arange(width) + half + 1, 0, width)
    return (
        integral[np.ix_(bottom, right)]
        - integral[np.ix_(top, right)]
        - integral[np.ix_(bottom, left)]
        + integral[np.ix_(top, left)]
    )


def lucas_kanade_flow(
    first: ArrayLike,
    second: ArrayLike,
    *,
    window_size: int = 15,
    threshold: float = 1e-3,
) -> OpticalFlowResult:
    """Estimate dense optical flow between two consecutive frames.

    Spatial gradients are computed as the average of both frames' central
    differences; the temporal gradient is the frame difference. Inside each
    ``window_size`` window the brightness-constancy equations are accumulated
    into a 2x2 structure tensor, and the flow is the least-squares solution.

    ``confidence`` is the smallest eigenvalue of the tensor: texture in two
    directions gives a large value, a single edge gives a small value, and a
    flat region gives nearly zero. Pixels whose eigenvalue does not exceed
    ``threshold``, and pixels near the border where the window is truncated,
    are marked ``valid = False``.
    """

    image_a, image_b = _as_gray_pair(first, second)
    if window_size < 3 or window_size % 2 == 0:
        raise ValueError("window_size must be an odd integer >= 3")
    if threshold < 0.0:
        raise ValueError("threshold must be non-negative")

    half = window_size // 2
    average = 0.5 * (image_a + image_b)
    ix = convolve2d(average, DX, padding="constant")
    iy = convolve2d(average, DY, padding="constant")
    it = image_b - image_a

    sum_ix2 = _box_sum(ix * ix, window_size)
    sum_iy2 = _box_sum(iy * iy, window_size)
    sum_ixiy = _box_sum(ix * iy, window_size)
    sum_ixt = _box_sum(ix * it, window_size)
    sum_iyt = _box_sum(iy * it, window_size)

    a11 = sum_ix2
    a12 = sum_ixiy
    a22 = sum_iy2
    b1 = sum_ixt
    b2 = sum_iyt

    det = a11 * a22 - a12 * a12
    confidence = 0.5 * (a11 + a22 - np.sqrt((a11 - a22) ** 2 + 4.0 * a12 * a12))

    flow = np.zeros((*image_a.shape, 2), dtype=np.float64)
    solve = det > threshold
    flow[..., 0] = np.where(solve, (-a22 * b1 + a12 * b2) / np.where(solve, det, 1.0), 0.0)
    flow[..., 1] = np.where(solve, (a12 * b1 - a11 * b2) / np.where(solve, det, 1.0), 0.0)

    valid = solve.copy()
    valid[:half, :] = False
    valid[-half:, :] = False
    valid[:, :half] = False
    valid[:, -half:] = False
    return OpticalFlowResult(flow, confidence, valid)