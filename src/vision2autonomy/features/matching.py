"""Local patch descriptors, feature matching, and robust homography fitting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class DescriptorSet:
    """Valid keypoints and their row-major normalized patch descriptors."""

    keypoints: NDArray[np.float64]
    descriptors: NDArray[np.float64]


@dataclass(frozen=True)
class MatchResult:
    """Index pairs, descriptor distances, and nearest/second-nearest ratios."""

    pairs: NDArray[np.int64]
    distances: NDArray[np.float64]
    ratios: NDArray[np.float64]


@dataclass(frozen=True)
class HomographyResult:
    """RANSAC homography and the inlier mask over input correspondences."""

    matrix: NDArray[np.float64]
    inliers: NDArray[np.bool_]
    errors: NDArray[np.float64]


def describe_patches(
    image: ArrayLike,
    keypoints: ArrayLike,
    *,
    patch_size: int = 9,
    eps: float = 1e-8,
) -> DescriptorSet:
    """Describe ``(row, column)`` keypoints with zero-mean, unit-norm patches.

    Points whose complete patch would cross the image boundary are omitted.
    Normalization removes additive brightness changes and most global contrast
    changes, while intentionally leaving rotation and scale sensitivity visible.
    """

    source = np.asarray(image, dtype=np.float64)
    points = np.asarray(keypoints, dtype=np.float64)
    if source.ndim != 2 or source.size == 0:
        raise ValueError("image must be a non-empty grayscale array")
    if not np.isfinite(source).all():
        raise ValueError("image must contain only finite values")
    if points.ndim != 2 or points.shape[1:] != (2,) or not np.isfinite(points).all():
        raise ValueError("keypoints must have shape (N, 2) and be finite")
    if patch_size < 3 or patch_size % 2 == 0:
        raise ValueError("patch_size must be an odd integer of at least 3")
    if eps <= 0:
        raise ValueError("eps must be positive")

    radius = patch_size // 2
    valid_points: list[tuple[float, float]] = []
    descriptors: list[NDArray[np.float64]] = []
    height, width = source.shape
    for row_float, column_float in points:
        row, column = int(round(row_float)), int(round(column_float))
        if row - radius < 0 or column - radius < 0 or row + radius >= height or column + radius >= width:
            continue
        patch = source[row - radius : row + radius + 1, column - radius : column + radius + 1].ravel()
        centered = patch - patch.mean()
        norm = float(np.linalg.norm(centered))
        descriptor = centered / max(norm, eps)
        valid_points.append((row_float, column_float))
        descriptors.append(descriptor)

    dimension = patch_size * patch_size
    return DescriptorSet(
        np.asarray(valid_points, dtype=np.float64).reshape(-1, 2),
        np.asarray(descriptors, dtype=np.float64).reshape(-1, dimension),
    )


def match_descriptors(
    descriptors_a: ArrayLike,
    descriptors_b: ArrayLike,
    *,
    ratio_threshold: float = 0.8,
    mutual: bool = True,
) -> MatchResult:
    """Match descriptors using nearest neighbors and Lowe's ratio test."""

    first = np.asarray(descriptors_a, dtype=np.float64)
    second = np.asarray(descriptors_b, dtype=np.float64)
    if first.ndim != 2 or second.ndim != 2 or first.shape[1:] != second.shape[1:]:
        raise ValueError("descriptor arrays must be two-dimensional with equal width")
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("descriptors must be finite")
    if not 0.0 < ratio_threshold <= 1.0:
        raise ValueError("ratio_threshold must satisfy 0 < ratio_threshold <= 1")
    if len(first) == 0 or len(second) < 2:
        return MatchResult(np.empty((0, 2), dtype=np.int64), np.empty(0), np.empty(0))

    squared = np.sum((first[:, None, :] - second[None, :, :]) ** 2, axis=2)
    order = np.argsort(squared, axis=1)[:, :2]
    rows = np.arange(len(first))
    nearest = order[:, 0]
    d1 = np.sqrt(squared[rows, nearest])
    d2 = np.sqrt(squared[rows, order[:, 1]])
    ratios = d1 / np.maximum(d2, np.finfo(np.float64).eps)
    keep = ratios < ratio_threshold
    if mutual:
        reverse_nearest = np.argmin(squared, axis=0)
        keep &= reverse_nearest[nearest] == rows

    selected = np.flatnonzero(keep)
    return MatchResult(
        np.column_stack((selected, nearest[selected])).astype(np.int64),
        d1[selected],
        ratios[selected],
    )


def _normalize_points(points: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    center = points.mean(axis=0)
    shifted = points - center
    mean_distance = float(np.mean(np.linalg.norm(shifted, axis=1)))
    if mean_distance <= np.finfo(np.float64).eps:
        raise ValueError("points must not all coincide")
    scale = np.sqrt(2.0) / mean_distance
    transform = np.array([[scale, 0.0, -scale * center[0]], [0.0, scale, -scale * center[1]], [0.0, 0.0, 1.0]])
    homogeneous = np.column_stack((points, np.ones(len(points))))
    normalized = (transform @ homogeneous.T).T[:, :2]
    return normalized, transform


def estimate_homography(source: ArrayLike, destination: ArrayLike) -> NDArray[np.float64]:
    """Estimate a projective transform with normalized Direct Linear Transform."""

    src = np.asarray(source, dtype=np.float64)
    dst = np.asarray(destination, dtype=np.float64)
    if src.ndim != 2 or src.shape[1:] != (2,) or dst.shape != src.shape or len(src) < 4:
        raise ValueError("source and destination must share shape (N, 2), N >= 4")
    if not np.isfinite(src).all() or not np.isfinite(dst).all():
        raise ValueError("correspondences must be finite")
    src_n, transform_src = _normalize_points(src)
    dst_n, transform_dst = _normalize_points(dst)
    rows = []
    for (x, y), (u, v) in zip(src_n, dst_n):
        rows.extend(([-x, -y, -1, 0, 0, 0, u * x, u * y, u], [0, 0, 0, -x, -y, -1, v * x, v * y, v]))
    _, _, vh = np.linalg.svd(np.asarray(rows, dtype=np.float64), full_matrices=True)
    normalized_h = vh[-1].reshape(3, 3)
    matrix = np.linalg.inv(transform_dst) @ normalized_h @ transform_src
    if abs(matrix[2, 2]) <= np.finfo(np.float64).eps:
        raise ValueError("correspondences produce a degenerate homography")
    return matrix / matrix[2, 2]


def project_points(points: ArrayLike, matrix: ArrayLike) -> NDArray[np.float64]:
    """Project Cartesian ``(x, y)`` points through a 3x3 homography."""

    values = np.asarray(points, dtype=np.float64)
    homography = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[1:] != (2,) or homography.shape != (3, 3):
        raise ValueError("points must have shape (N, 2) and matrix shape (3, 3)")
    projected = (homography @ np.column_stack((values, np.ones(len(values)))).T).T
    if np.any(np.abs(projected[:, 2]) <= np.finfo(np.float64).eps):
        raise ValueError("homography projects points to infinity")
    return projected[:, :2] / projected[:, 2, None]


def ransac_homography(
    source: ArrayLike,
    destination: ArrayLike,
    *,
    threshold: float = 2.0,
    max_iterations: int = 1000,
    seed: int = 0,
) -> HomographyResult:
    """Robustly estimate a homography using four-point random samples."""

    src = np.asarray(source, dtype=np.float64)
    dst = np.asarray(destination, dtype=np.float64)
    if src.ndim != 2 or src.shape[1:] != (2,) or dst.shape != src.shape or len(src) < 4:
        raise ValueError("source and destination must share shape (N, 2), N >= 4")
    if threshold <= 0 or max_iterations < 1:
        raise ValueError("threshold and max_iterations must be positive")

    generator = np.random.default_rng(seed)
    best_mask = np.zeros(len(src), dtype=bool)
    best_error = np.inf
    for _ in range(max_iterations):
        sample = generator.choice(len(src), 4, replace=False)
        try:
            candidate = estimate_homography(src[sample], dst[sample])
            errors = np.linalg.norm(project_points(src, candidate) - dst, axis=1)
        except (ValueError, np.linalg.LinAlgError):
            continue
        mask = errors <= threshold
        score = int(mask.sum())
        mean_error = float(errors[mask].mean()) if score else np.inf
        if score > int(best_mask.sum()) or (score == int(best_mask.sum()) and mean_error < best_error):
            best_mask, best_error = mask, mean_error

    if best_mask.sum() < 4:
        raise ValueError("RANSAC could not find four consistent correspondences")
    matrix = estimate_homography(src[best_mask], dst[best_mask])
    errors = np.linalg.norm(project_points(src, matrix) - dst, axis=1)
    inliers = errors <= threshold
    if inliers.sum() >= 4 and not np.array_equal(inliers, best_mask):
        matrix = estimate_homography(src[inliers], dst[inliers])
        errors = np.linalg.norm(project_points(src, matrix) - dst, axis=1)
        inliers = errors <= threshold
    return HomographyResult(matrix, inliers, errors)
