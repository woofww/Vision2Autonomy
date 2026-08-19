"""Smoothing filters built from first principles."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from vision2autonomy.image.convolution import convolve2d


def gaussian_kernel(size: int = 5, sigma: float = 1.4) -> NDArray[np.float64]:
    """Return a normalized square Gaussian kernel."""

    if size < 1 or size % 2 == 0:
        raise ValueError("size must be a positive odd integer")
    if sigma <= 0:
        raise ValueError("sigma must be positive")

    radius = size // 2
    coordinates = np.arange(-radius, radius + 1, dtype=np.float64)
    yy, xx = np.meshgrid(coordinates, coordinates, indexing="ij")
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    return kernel / kernel.sum()


def gaussian_blur(
    image: ArrayLike,
    *,
    size: int = 5,
    sigma: float = 1.4,
) -> NDArray[np.float64]:
    """Smooth a grayscale image with a Gaussian kernel."""

    return convolve2d(image, gaussian_kernel(size, sigma), padding="reflect")

