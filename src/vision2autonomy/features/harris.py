"""Harris corner detection implemented with NumPy primitives."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from vision2autonomy.edges.canny import sobel_gradients
from vision2autonomy.image.filters import gaussian_blur


@dataclass(frozen=True)
class HarrisResult:
    """Inspectable products of the Harris detector."""

    response: NDArray[np.float64]
    lambda_max: NDArray[np.float64]
    lambda_min: NDArray[np.float64]
    corners: NDArray[np.int64]


def harris_response(
    image: ArrayLike,
    *,
    k: float = 0.04,
    window_size: int = 5,
    sigma: float = 1.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return Harris response and both structure-tensor eigenvalue maps.

    For each pixel, local gradient products form the second-moment matrix
    ``M``. The response is ``det(M) - k * trace(M) ** 2``.
    """

    source = np.asarray(image)
    if source.ndim != 2:
        raise ValueError("Harris expects a two-dimensional grayscale image")
    if source.size == 0:
        raise ValueError("image must not be empty")
    if not np.isfinite(source).all():
        raise ValueError("image must contain only finite values")
    if not 0.0 < k < 0.25:
        raise ValueError("k must satisfy 0 < k < 0.25")

    gx, gy, _, _ = sobel_gradients(source)
    sxx = gaussian_blur(gx * gx, size=window_size, sigma=sigma)
    syy = gaussian_blur(gy * gy, size=window_size, sigma=sigma)
    sxy = gaussian_blur(gx * gy, size=window_size, sigma=sigma)

    determinant = sxx * syy - sxy * sxy
    trace = sxx + syy
    response = determinant - k * trace * trace

    discriminant = np.sqrt(np.maximum((sxx - syy) ** 2 + 4.0 * sxy * sxy, 0.0))
    lambda_max = 0.5 * (trace + discriminant)
    lambda_min = 0.5 * (trace - discriminant)
    return response, lambda_max, lambda_min


def select_corners(
    response: ArrayLike,
    *,
    threshold_rel: float = 0.01,
    min_distance: int = 8,
    max_corners: int | None = None,
) -> NDArray[np.int64]:
    """Select separated local maxima from a Harris response map.

    Coordinates are returned as ``(row, column)`` pairs ordered from strongest
    to weakest response.
    """

    values = np.asarray(response, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("response must be two-dimensional")
    if not 0.0 < threshold_rel <= 1.0:
        raise ValueError("threshold_rel must satisfy 0 < threshold_rel <= 1")
    if min_distance < 1:
        raise ValueError("min_distance must be positive")
    if max_corners is not None and max_corners < 1:
        raise ValueError("max_corners must be positive when provided")

    maximum = float(values.max(initial=0.0))
    if maximum <= 0.0:
        return np.empty((0, 2), dtype=np.int64)

    radius = min_distance
    padded = np.pad(values, radius, mode="constant", constant_values=-np.inf)
    windows = np.lib.stride_tricks.sliding_window_view(
        padded, (2 * radius + 1, 2 * radius + 1)
    )
    local_maximum = windows.max(axis=(-2, -1))
    candidates = np.argwhere((values >= maximum * threshold_rel) & (values == local_maximum))
    if candidates.size == 0:
        return np.empty((0, 2), dtype=np.int64)

    strengths = values[candidates[:, 0], candidates[:, 1]]
    ordered = candidates[np.argsort(strengths)[::-1]]
    selected: list[NDArray[np.int64]] = []
    min_distance_squared = min_distance * min_distance
    for coordinate in ordered:
        if all(np.sum((coordinate - existing) ** 2) >= min_distance_squared for existing in selected):
            selected.append(coordinate)
            if max_corners is not None and len(selected) >= max_corners:
                break

    return np.asarray(selected, dtype=np.int64).reshape(-1, 2)


def detect_harris_corners(
    image: ArrayLike,
    *,
    k: float = 0.04,
    window_size: int = 5,
    sigma: float = 1.0,
    threshold_rel: float = 0.01,
    min_distance: int = 8,
    max_corners: int | None = None,
) -> HarrisResult:
    """Compute Harris response, eigenvalues, and final corner coordinates."""

    response, lambda_max, lambda_min = harris_response(
        image, k=k, window_size=window_size, sigma=sigma
    )
    corners = select_corners(
        response,
        threshold_rel=threshold_rel,
        min_distance=min_distance,
        max_corners=max_corners,
    )
    return HarrisResult(response, lambda_max, lambda_min, corners)

