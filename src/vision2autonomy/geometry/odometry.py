"""Visual-odometry trajectory construction and evaluation primitives.

This module connects triangulated landmarks to the robust PnP solver from
Chapter 09, then provides the trajectory alignment and error metrics needed
to reason about drift.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from vision2autonomy.geometry.pose import camera_center, ransac_pnp


def pose_matrix(rotation: ArrayLike, translation: ArrayLike) -> NDArray[np.float64]:
    """Build a 4x4 world-to-camera rigid transform."""

    r = np.asarray(rotation, dtype=np.float64)
    t = np.asarray(translation, dtype=np.float64).reshape(-1)
    if r.shape != (3, 3) or t.shape != (3,):
        raise ValueError("rotation must be 3x3 and translation must have length 3")
    if not np.isfinite(r).all() or not np.isfinite(t).all():
        raise ValueError("pose values must be finite")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = r
    transform[:3, 3] = t
    return transform


def invert_pose(transform: ArrayLike) -> NDArray[np.float64]:
    """Invert a rigid 4x4 transform without a general matrix inverse."""

    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError("transform must have shape (4, 4)")
    rotation = matrix[:3, :3]
    translation = matrix[:3, 3]
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ translation
    return inverse


def compose_poses(first: ArrayLike, second: ArrayLike) -> NDArray[np.float64]:
    """Compose transforms so the returned matrix applies ``first`` then ``second``."""

    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    if a.shape != (4, 4) or b.shape != (4, 4):
        raise ValueError("both transforms must have shape (4, 4)")
    return b @ a


def accumulate_relative_poses(
    relative_poses: Sequence[ArrayLike],
    initial_pose: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """Convert frame-to-frame transforms into world-to-camera poses."""

    current = (
        np.eye(4, dtype=np.float64)
        if initial_pose is None
        else np.asarray(initial_pose, dtype=np.float64)
    )
    if current.shape != (4, 4):
        raise ValueError("initial_pose must have shape (4, 4)")
    absolute = [current.copy()]
    for relative in relative_poses:
        matrix = np.asarray(relative, dtype=np.float64)
        if matrix.shape != (4, 4):
            raise ValueError("every relative pose must have shape (4, 4)")
        current = matrix @ current
        absolute.append(current.copy())
    return np.stack(absolute)


def trajectory_from_poses(poses: ArrayLike) -> NDArray[np.float64]:
    """Return camera centers from world-to-camera 4x4 poses."""

    matrices = np.asarray(poses, dtype=np.float64)
    if matrices.ndim != 3 or matrices.shape[1:] != (4, 4):
        raise ValueError("poses must have shape (N, 4, 4)")
    return np.stack(
        [camera_center(matrix[:3, :3], matrix[:3, 3]) for matrix in matrices]
    )


@dataclass(frozen=True)
class OdometryResult:
    """Per-frame robust PnP poses for landmarks anchored in the first frame."""

    poses: NDArray[np.float64]
    centers: NDArray[np.float64]
    inlier_counts: NDArray[np.int64]
    median_errors: NDArray[np.float64]


def pnp_odometry(
    landmarks_3d: ArrayLike,
    observations: Sequence[ArrayLike],
    intrinsics: ArrayLike,
    *,
    threshold: float = 3.0,
    max_iterations: int = 500,
    seed: int = 0,
) -> OdometryResult:
    """Estimate a metric camera trajectory from fixed 3-D landmarks.

    Each observation array has shape ``(N, 2)`` and corresponds row-for-row
    with ``landmarks_3d``. Missing measurements are represented by NaN.
    Landmarks may come from the first stereo pair, RGB-D, or an existing map.
    """

    landmarks = np.asarray(landmarks_3d, dtype=np.float64)
    k = np.asarray(intrinsics, dtype=np.float64)
    if landmarks.ndim != 2 or landmarks.shape[1:] != (3,):
        raise ValueError("landmarks_3d must have shape (N, 3)")
    if not observations:
        raise ValueError("observations must not be empty")

    poses: list[NDArray[np.float64]] = []
    counts: list[int] = []
    medians: list[float] = []
    for frame_index, values in enumerate(observations):
        pixels = np.asarray(values, dtype=np.float64)
        if pixels.shape != (len(landmarks), 2):
            raise ValueError("every observation must have shape (N, 2)")
        visible = np.isfinite(pixels).all(axis=1) & np.isfinite(landmarks).all(axis=1)
        if int(visible.sum()) < 6:
            raise ValueError(f"frame {frame_index} has fewer than six visible landmarks")
        result = ransac_pnp(
            landmarks[visible],
            pixels[visible],
            k,
            threshold=threshold,
            max_iterations=max_iterations,
            seed=seed + frame_index,
        )
        poses.append(pose_matrix(result.rotation, result.translation))
        counts.append(int(result.inliers.sum()))
        medians.append(float(np.median(result.errors[result.inliers])))

    pose_array = np.stack(poses)
    return OdometryResult(
        poses=pose_array,
        centers=trajectory_from_poses(pose_array),
        inlier_counts=np.asarray(counts, dtype=np.int64),
        median_errors=np.asarray(medians, dtype=np.float64),
    )


@dataclass(frozen=True)
class TrajectoryAlignment:
    """Similarity transform mapping an estimated trajectory to a reference."""

    aligned: NDArray[np.float64]
    scale: float
    rotation: NDArray[np.float64]
    translation: NDArray[np.float64]


def align_trajectory(
    estimated: ArrayLike,
    reference: ArrayLike,
    *,
    with_scale: bool = False,
) -> TrajectoryAlignment:
    """Align two 3-D trajectories with Umeyama's least-squares method."""

    source = np.asarray(estimated, dtype=np.float64)
    target = np.asarray(reference, dtype=np.float64)
    if source.ndim != 2 or source.shape[1:] != (3,) or target.shape != source.shape:
        raise ValueError("trajectories must share shape (N, 3)")
    if len(source) < 3:
        raise ValueError("at least three poses are required")
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("trajectory values must be finite")

    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = target_centered.T @ source_centered / len(source)
    u, singular, vh = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(u @ vh) < 0.0:
        correction[-1, -1] = -1.0
    rotation = u @ correction @ vh
    if with_scale:
        variance = float(np.mean(np.sum(source_centered**2, axis=1)))
        if variance <= np.finfo(np.float64).eps:
            raise ValueError("estimated trajectory has zero variance")
        scale = float(np.sum(singular * np.diag(correction)) / variance)
    else:
        scale = 1.0
    translation = target_mean - scale * (rotation @ source_mean)
    aligned = (scale * (rotation @ source.T)).T + translation
    return TrajectoryAlignment(aligned, scale, rotation, translation)


def absolute_trajectory_error(
    estimated: ArrayLike,
    reference: ArrayLike,
    *,
    align: bool = True,
    with_scale: bool = False,
) -> float:
    """Return translational ATE RMSE in the reference trajectory's units."""

    source = np.asarray(estimated, dtype=np.float64)
    target = np.asarray(reference, dtype=np.float64)
    if align:
        source = align_trajectory(source, target, with_scale=with_scale).aligned
    elif source.shape != target.shape:
        raise ValueError("trajectories must have the same shape")
    return float(np.sqrt(np.mean(np.sum((source - target) ** 2, axis=1))))


def relative_translation_error(
    estimated: ArrayLike,
    reference: ArrayLike,
    *,
    delta: int = 1,
) -> NDArray[np.float64]:
    """Return per-segment displacement-vector error for a fixed frame gap."""

    source = np.asarray(estimated, dtype=np.float64)
    target = np.asarray(reference, dtype=np.float64)
    if source.ndim != 2 or source.shape[1:] != (3,) or target.shape != source.shape:
        raise ValueError("trajectories must share shape (N, 3)")
    if int(delta) < 1 or int(delta) >= len(source):
        raise ValueError("delta must be between 1 and N-1")
    step = int(delta)
    source_motion = source[step:] - source[:-step]
    target_motion = target[step:] - target[:-step]
    return np.linalg.norm(source_motion - target_motion, axis=1)
