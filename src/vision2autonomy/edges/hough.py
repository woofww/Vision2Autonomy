"""Hough transform for line detection, implemented from first principles.

A binary edge map votes for every line that passes through each edge pixel.
The accumulator is a discrete (rho, theta) parameter space; peaks in that
space correspond to image lines that many edge pixels support.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

# theta is measured in radians inside [0, pi), so every image line has a
# unique normal form (rho, theta): rho = x*cos(theta) + y*sin(theta).
THETA_START = 0.0
THETA_END = np.pi


@dataclass(frozen=True)
class HoughLineResult:
    """Detected lines together with the accumulator and drawing support."""

    lines: NDArray[np.float64]      # (N, 2), each row is (rho, theta) in radians
    votes: NDArray[np.float64]      # (N,) accumulator value of each peak
    segments: NDArray[np.float64]   # (N, 4), each row is (x1, y1, x2, y2)
    accumulator: NDArray[np.float64]  # (num_rhos, num_thetas) vote counts
    rhos: NDArray[np.float64]       # rho axis values
    thetas: NDArray[np.float64]     # theta axis values in radians


def _as_edge_mask(edge_image: ArrayLike) -> NDArray[np.bool_]:
    values = np.asarray(edge_image)
    if values.ndim != 2:
        raise ValueError("edge image must be two-dimensional")
    if values.size == 0:
        raise ValueError("edge image must not be empty")
    if not np.isfinite(values).all():
        raise ValueError("edge image must contain only finite values")
    return values != 0


def hough_accumulator(
    edge_image: ArrayLike,
    *,
    rho_resolution: float = 1.0,
    theta_resolution: float = 1.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Vote every edge pixel for all lines passing through it.

    Returns ``(accumulator, rhos, thetas)`` where ``accumulator[i, j]`` counts
    how many edge pixels support the line ``(rhos[i], thetas[j])``.
    """

    if rho_resolution <= 0.0:
        raise ValueError("rho_resolution must be positive")
    if theta_resolution <= 0.0 or theta_resolution >= 180.0:
        raise ValueError("theta_resolution must be in (0, 180) degrees")

    mask = _as_edge_mask(edge_image)
    height, width = mask.shape
    rho_max = float(np.hypot(height, width))
    num_rhos = int(np.ceil(2.0 * rho_max / rho_resolution)) + 1
    rhos = np.linspace(-rho_max, rho_max, num_rhos, dtype=np.float64)
    num_thetas = int(np.ceil(np.degrees(THETA_END - THETA_START) / theta_resolution))
    thetas = np.linspace(THETA_START, THETA_END, num_thetas, endpoint=False, dtype=np.float64)

    rows, columns = np.nonzero(mask)
    if rows.size == 0:
        return np.zeros((num_rhos, num_thetas), dtype=np.float64), rhos, thetas

    # rho = x*cos(theta) + y*sin(theta) for every edge pixel and every theta.
    rho_values = (
        columns[:, None] * np.cos(thetas)[None, :]
        + rows[:, None] * np.sin(thetas)[None, :]
    )
    rho_indices = np.clip(
        np.round((rho_values + rho_max) / rho_resolution).astype(np.int64),
        0,
        num_rhos - 1,
    )
    theta_indices = np.broadcast_to(
        np.arange(num_thetas, dtype=np.int64)[None, :], rho_indices.shape
    )
    accumulator = np.zeros((num_rhos, num_thetas), dtype=np.float64)
    np.add.at(accumulator, (rho_indices.ravel(), theta_indices.ravel()), 1.0)
    return accumulator, rhos, thetas


def hough_peaks(
    accumulator: ArrayLike,
    *,
    num_peaks: int = 10,
    threshold_rel: float = 0.35,
    neighborhood: int = 5,
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Extract the strongest, spatially separated accumulator peaks.

    The accumulator is repeatedly searched for its global maximum. A square
    neighborhood around each accepted peak is suppressed so a thick edge line
    does not produce many near-duplicate (rho, theta) hypotheses.
    """

    votes = np.asarray(accumulator, dtype=np.float64)
    if votes.ndim != 2:
        raise ValueError("accumulator must be two-dimensional")
    if votes.size == 0:
        raise ValueError("accumulator must not be empty")
    if num_peaks < 1:
        raise ValueError("num_peaks must be positive")
    if threshold_rel < 0.0 or threshold_rel > 1.0:
        raise ValueError("threshold_rel must be in [0, 1]")
    if neighborhood < 0:
        raise ValueError("neighborhood must be non-negative")

    working = votes.copy()
    maximum = float(working.max())
    if maximum <= 0.0:
        return np.empty((0, 2), dtype=np.int64), np.empty(0, dtype=np.float64)

    threshold = threshold_rel * maximum
    found_indices: list[list[int]] = []
    found_votes: list[float] = []
    height, width = working.shape

    for _ in range(num_peaks):
        flat_index = int(np.argmax(working))
        row, column = divmod(flat_index, width)
        value = float(working[row, column])
        if value < threshold:
            break
        found_indices.append([row, column])
        found_votes.append(value)
        working[
            max(0, row - neighborhood) : min(height, row + neighborhood + 1),
            max(0, column - neighborhood) : min(width, column + neighborhood + 1),
        ] = 0.0

    return (
        np.asarray(found_indices, dtype=np.int64).reshape(-1, 2),
        np.asarray(found_votes, dtype=np.float64),
    )


def hough_lines(
    edge_image: ArrayLike,
    *,
    num_peaks: int = 10,
    threshold_rel: float = 0.35,
    rho_resolution: float = 1.0,
    theta_resolution: float = 1.0,
    support_tolerance: float = 1.5,
    min_support: int = 15,
) -> HoughLineResult:
    """Detect straight lines in a binary edge map.

    Returns line parameters, accumulator votes, and concrete pixel segments.
    Each segment is derived from the edge pixels that voted for its peak,
    which keeps the output directly drawable and makes the result inspectable.
    """

    if support_tolerance <= 0.0:
        raise ValueError("support_tolerance must be positive")
    if min_support < 1:
        raise ValueError("min_support must be positive")

    accumulator, rhos, thetas = hough_accumulator(
        edge_image,
        rho_resolution=rho_resolution,
        theta_resolution=theta_resolution,
    )
    peak_indices, votes = hough_peaks(
        accumulator,
        num_peaks=num_peaks,
        threshold_rel=threshold_rel,
        neighborhood=max(3, int(round(5.0 / theta_resolution))),
    )
    mask = _as_edge_mask(edge_image)
    rows, columns = np.nonzero(mask)
    if rows.size == 0:
        return HoughLineResult(
            np.empty((0, 2)), np.empty(0), np.empty((0, 4)), accumulator, rhos, thetas
        )

    rho_tolerance = support_tolerance * rho_resolution

    lines: list[list[float]] = []
    segments: list[list[float]] = []
    kept_votes: list[float] = []

    for (rho_index, theta_index), vote in zip(peak_indices, votes):
        rho_peak = float(rhos[rho_index])
        theta_peak = float(thetas[theta_index])

        # All edge pixels geometrically close to this line hypothesis.
        rho_pixel = columns * np.cos(theta_peak) + rows * np.sin(theta_peak)
        rho_distance = np.abs(rho_pixel - rho_peak)
        supporters = rho_distance <= rho_tolerance
        if int(np.count_nonzero(supporters)) < min_support:
            continue

        # Project supporters onto the direction parallel to the line and keep
        # the extremes: that gives a concrete segment inside the image.
        direction = np.array([-np.sin(theta_peak), np.cos(theta_peak)])
        origin = np.array([rho_peak * np.cos(theta_peak), rho_peak * np.sin(theta_peak)])
        projections = (
            columns[supporters] * direction[0] + rows[supporters] * direction[1]
        )
        start_t, end_t = float(projections.min()), float(projections.max())
        if end_t - start_t < min_support - 1:
            continue

        start = origin + start_t * direction
        end = origin + end_t * direction
        lines.append([rho_peak, theta_peak])
        segments.append([start[0], start[1], end[0], end[1]])
        kept_votes.append(float(vote))

    return HoughLineResult(
        np.asarray(lines, dtype=np.float64).reshape(-1, 2),
        np.asarray(kept_votes, dtype=np.float64),
        np.asarray(segments, dtype=np.float64).reshape(-1, 4),
        accumulator,
        rhos,
        thetas,
    )
