"""From-scratch camera pose estimation from 3-D to 2-D correspondences.

The implementation uses a transparent normalized DLT solver followed by a
projection onto SO(3). It is designed for learning and initialization;
production PnP systems normally add non-linear refinement.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from vision2autonomy.geometry.calibration import project_points


def _validate_correspondences(
    object_points: ArrayLike,
    image_points: ArrayLike,
    intrinsics: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    world = np.asarray(object_points, dtype=np.float64)
    pixels = np.asarray(image_points, dtype=np.float64)
    k = np.asarray(intrinsics, dtype=np.float64)
    if world.ndim != 2 or world.shape[1:] != (3,):
        raise ValueError("object_points must have shape (N, 3)")
    if pixels.shape != (len(world), 2):
        raise ValueError("image_points must have shape (N, 2)")
    if len(world) < 6:
        raise ValueError("at least six correspondences are required")
    if k.shape != (3, 3):
        raise ValueError("intrinsics must have shape (3, 3)")
    if not np.isfinite(world).all() or not np.isfinite(pixels).all() or not np.isfinite(k).all():
        raise ValueError("correspondences and intrinsics must be finite")
    if np.linalg.matrix_rank(world - world.mean(axis=0), tol=1e-10) < 3:
        raise ValueError("object points must span three dimensions")
    if abs(np.linalg.det(k)) <= np.finfo(np.float64).eps:
        raise ValueError("intrinsics must be invertible")
    return world, pixels, k


def _normalize_world_points(
    points: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    center = points.mean(axis=0)
    shifted = points - center
    mean_distance = float(np.mean(np.linalg.norm(shifted, axis=1)))
    if mean_distance <= np.finfo(np.float64).eps:
        raise ValueError("object points must not all coincide")
    scale = np.sqrt(3.0) / mean_distance
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] *= scale
    transform[:3, 3] = -scale * center
    homogeneous = np.column_stack((points, np.ones(len(points))))
    normalized = (transform @ homogeneous.T).T
    return normalized[:, :3], transform


def _pose_from_projection(
    projection: NDArray[np.float64],
    world: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Project a general 3x4 DLT matrix onto a rigid camera transform."""

    determinant = float(np.linalg.det(projection[:, :3]))
    if abs(determinant) <= np.finfo(np.float64).eps:
        raise ValueError("DLT projection has a singular rotation block")
    # A homogeneous DLT matrix has arbitrary global sign. Choose the sign
    # that gives a positive-determinant rotation block before projecting it
    # onto SO(3); repairing a negative determinant after SVD can otherwise
    # select a 180-degree-flipped pose under small pixel noise.
    matrix = np.sign(determinant) * projection
    u, singular, vh = np.linalg.svd(matrix[:, :3])
    rotation = u @ vh
    if np.linalg.det(rotation) < 0.0:
        raise ValueError("could not recover a proper rotation")
    scale = float(np.mean(singular))
    if scale <= np.finfo(np.float64).eps:
        raise ValueError("DLT projection has zero scale")
    translation = matrix[:, 3] / scale
    positive = int(np.count_nonzero(((rotation @ world.T).T + translation)[:, 2] > 0.0))
    if positive == 0:
        raise ValueError("could not recover a pose with points in front of the camera")
    return rotation, translation


def estimate_pose_dlt(
    object_points: ArrayLike,
    image_points: ArrayLike,
    intrinsics: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Estimate ``R, t`` in ``x ~ K [R | t] X`` using normalized DLT.

    The 3-D points must not all lie on one plane. Pixel coordinates are first
    converted to normalized camera coordinates with ``K^-1``; the recovered
    matrix is then projected to the nearest valid rotation matrix.
    """

    world, pixels, k = _validate_correspondences(object_points, image_points, intrinsics)
    normalized_world, world_transform = _normalize_world_points(world)
    pixel_h = np.column_stack((pixels, np.ones(len(pixels))))
    rays = (np.linalg.inv(k) @ pixel_h.T).T
    normalized_pixels = rays[:, :2] / rays[:, 2, None]

    homogeneous_world = np.column_stack((normalized_world, np.ones(len(world))))
    rows: list[NDArray[np.float64]] = []
    zero = np.zeros(4, dtype=np.float64)
    for point, (x, y) in zip(homogeneous_world, normalized_pixels):
        rows.append(np.concatenate((point, zero, -x * point)))
        rows.append(np.concatenate((zero, point, -y * point)))
    design = np.vstack(rows)
    _, _, vh = np.linalg.svd(design)
    normalized_projection = vh[-1].reshape(3, 4)
    projection = normalized_projection @ world_transform
    return _pose_from_projection(projection, world)


def pose_reprojection_errors(
    object_points: ArrayLike,
    image_points: ArrayLike,
    intrinsics: ArrayLike,
    rotation: ArrayLike,
    translation: ArrayLike,
) -> NDArray[np.float64]:
    """Return Euclidean pixel reprojection error for every correspondence."""

    world = np.asarray(object_points, dtype=np.float64)
    observed = np.asarray(image_points, dtype=np.float64)
    if world.ndim != 2 or world.shape[1:] != (3,) or observed.shape != (len(world), 2):
        raise ValueError("correspondences must have shapes (N, 3) and (N, 2)")
    predicted = project_points(world, intrinsics, rotation, translation)
    return np.linalg.norm(predicted - observed, axis=1)


def camera_center(rotation: ArrayLike, translation: ArrayLike) -> NDArray[np.float64]:
    """Convert world-to-camera extrinsics into the camera center in world coordinates."""

    r = np.asarray(rotation, dtype=np.float64)
    t = np.asarray(translation, dtype=np.float64).reshape(-1)
    if r.shape != (3, 3) or t.shape != (3,):
        raise ValueError("rotation must be 3x3 and translation must have length 3")
    return -r.T @ t


@dataclass(frozen=True)
class PnPRansacResult:
    """Robust PnP output and the consensus set that supports it."""

    rotation: NDArray[np.float64]
    translation: NDArray[np.float64]
    inliers: NDArray[np.bool_]
    errors: NDArray[np.float64]


def ransac_pnp(
    object_points: ArrayLike,
    image_points: ArrayLike,
    intrinsics: ArrayLike,
    *,
    threshold: float = 3.0,
    max_iterations: int = 500,
    seed: int = 0,
) -> PnPRansacResult:
    """Reject bad 3-D/2-D matches with small normalized-DLT hypotheses.

    Six correspondences are the algebraic minimum. When more data are
    available, each hypothesis uses eight points because minimal six-point
    DLT is noticeably more sensitive to pixel noise and near-planar layouts.
    """

    world, pixels, k = _validate_correspondences(object_points, image_points, intrinsics)
    threshold = float(threshold)
    if threshold <= 0.0:
        raise ValueError("threshold must be positive")
    if int(max_iterations) < 1:
        raise ValueError("max_iterations must be positive")

    rng = np.random.default_rng(seed)
    best_mask: NDArray[np.bool_] | None = None
    best_count = -1
    best_median = np.inf
    sample_size = min(8, len(world))
    for _ in range(int(max_iterations)):
        sample = rng.choice(len(world), size=sample_size, replace=False)
        try:
            rotation, translation = estimate_pose_dlt(world[sample], pixels[sample], k)
            errors = pose_reprojection_errors(world, pixels, k, rotation, translation)
        except (ValueError, np.linalg.LinAlgError):
            continue
        mask = errors <= threshold
        count = int(mask.sum())
        median = float(np.median(errors[mask])) if count else np.inf
        if count > best_count or (count == best_count and median < best_median):
            best_count = count
            best_median = median
            best_mask = mask

    if best_mask is None or int(best_mask.sum()) < 6:
        raise ValueError("RANSAC could not find a valid pose")
    rotation, translation = estimate_pose_dlt(world[best_mask], pixels[best_mask], k)
    errors = pose_reprojection_errors(world, pixels, k, rotation, translation)
    inliers = errors <= threshold
    if int(inliers.sum()) >= 6 and not np.array_equal(inliers, best_mask):
        rotation, translation = estimate_pose_dlt(world[inliers], pixels[inliers], k)
        errors = pose_reprojection_errors(world, pixels, k, rotation, translation)
        inliers = errors <= threshold
    return PnPRansacResult(rotation, translation, inliers, errors)
