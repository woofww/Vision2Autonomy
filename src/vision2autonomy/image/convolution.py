"""Two-dimensional convolution implemented with NumPy."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

PaddingMode = Literal["constant", "edge", "reflect"]


def convolve2d(
    image: ArrayLike,
    kernel: ArrayLike,
    *,
    padding: PaddingMode = "reflect",
) -> NDArray[np.float64]:
    """Convolve a grayscale image with an odd-sized 2-D kernel.

    The output has the same shape as the input. The kernel is flipped in both
    dimensions, so this function performs convolution rather than correlation.
    """

    source = np.asarray(image, dtype=np.float64)
    weights = np.asarray(kernel, dtype=np.float64)

    if source.ndim != 2:
        raise ValueError("image must be a two-dimensional grayscale array")
    if weights.ndim != 2 or any(size % 2 == 0 for size in weights.shape):
        raise ValueError("kernel must be two-dimensional with odd dimensions")
    if padding not in {"constant", "edge", "reflect"}:
        raise ValueError(f"unsupported padding mode: {padding}")

    pad_y, pad_x = weights.shape[0] // 2, weights.shape[1] // 2
    padded = np.pad(source, ((pad_y, pad_y), (pad_x, pad_x)), mode=padding)
    windows = np.lib.stride_tricks.sliding_window_view(padded, weights.shape)
    flipped = weights[::-1, ::-1]
    return np.einsum("ijkl,kl->ij", windows, flipped, optimize=True)

