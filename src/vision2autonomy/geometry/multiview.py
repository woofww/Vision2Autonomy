"""Pinhole projection, epipolar geometry, and linear triangulation."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def camera_matrix(intrinsics: ArrayLike, rotation: ArrayLike, translation: ArrayLike) -> NDArray[np.float64]:
    """Build ``P = K [R | t]`` mapping world points to image pixels."""

    k = np.asarray(intrinsics, dtype=np.float64)
    r = np.asarray(rotation, dtype=np.float64)
    t = np.asarray(translation, dtype=np.float64).reshape(-1)
    if k.shape != (3, 3) or r.shape != (3, 3) or t.shape != (3,):
        raise ValueError("intrinsics and rotation must be 3x3; translation must have length 3")
    if not np.isfinite(k).all() or not np.isfinite(r).all() or not np.isfinite(t).all():
        raise ValueError("camera parameters must be finite")
    return k @ np.column_stack((r, t))


def project_points_3d(points: ArrayLike, projection: ArrayLike) -> NDArray[np.float64]:
    """Project world ``(X, Y, Z)`` points through a 3x4 camera matrix."""

    world = np.asarray(points, dtype=np.float64)
    matrix = np.asarray(projection, dtype=np.float64)
    if world.ndim != 2 or world.shape[1:] != (3,) or matrix.shape != (3, 4):
        raise ValueError("points must have shape (N, 3) and projection shape (3, 4)")
    image_h = (matrix @ np.column_stack((world, np.ones(len(world)))).T).T
    if np.any(np.abs(image_h[:, 2]) <= np.finfo(np.float64).eps):
        raise ValueError("a point projects to infinity")
    return image_h[:, :2] / image_h[:, 2, None]


def _normalize_image_points(points: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    center = points.mean(axis=0)
    shifted = points - center
    mean_distance = float(np.mean(np.linalg.norm(shifted, axis=1)))
    if mean_distance <= np.finfo(np.float64).eps:
        raise ValueError("image points must not all coincide")
    scale = np.sqrt(2.0) / mean_distance
    transform = np.array([[scale, 0.0, -scale * center[0]], [0.0, scale, -scale * center[1]], [0.0, 0.0, 1.0]])
    homogeneous = np.column_stack((points, np.ones(len(points))))
    normalized = (transform @ homogeneous.T).T
    return normalized[:, :2], transform


def estimate_fundamental(points_a: ArrayLike, points_b: ArrayLike) -> NDArray[np.float64]:
    """Estimate the rank-two fundamental matrix with normalized eight-point DLT."""

    first = np.asarray(points_a, dtype=np.float64)
    second = np.asarray(points_b, dtype=np.float64)
    if first.ndim != 2 or first.shape[1:] != (2,) or second.shape != first.shape or len(first) < 8:
        raise ValueError("point arrays must share shape (N, 2), N >= 8")
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("image points must be finite")

    first_n, transform_a = _normalize_image_points(first)
    second_n, transform_b = _normalize_image_points(second)
    x, y = first_n.T
    u, v = second_n.T
    design = np.column_stack((u * x, u * y, u, v * x, v * y, v, x, y, np.ones(len(first))))
    _, _, vh = np.linalg.svd(design, full_matrices=True)
    fundamental = vh[-1].reshape(3, 3)
    u_svd, singular, vh_svd = np.linalg.svd(fundamental)
    singular[-1] = 0.0
    rank_two = u_svd @ np.diag(singular) @ vh_svd
    matrix = transform_b.T @ rank_two @ transform_a
    norm = float(np.linalg.norm(matrix))
    if norm <= np.finfo(np.float64).eps:
        raise ValueError("correspondences produce a degenerate fundamental matrix")
    return matrix / norm


def epipolar_lines(points: ArrayLike, fundamental: ArrayLike, *, image: str = "b") -> NDArray[np.float64]:
    """Return lines ``(a,b,c)`` in the opposite image for input pixels."""

    values = np.asarray(points, dtype=np.float64)
    matrix = np.asarray(fundamental, dtype=np.float64)
    if values.ndim != 2 or values.shape[1:] != (2,) or matrix.shape != (3, 3):
        raise ValueError("points must have shape (N, 2) and fundamental shape (3, 3)")
    homogeneous = np.column_stack((values, np.ones(len(values))))
    if image == "b":
        return (matrix @ homogeneous.T).T
    if image == "a":
        return (matrix.T @ homogeneous.T).T
    raise ValueError("image must be 'a' or 'b'")


def sampson_errors(points_a: ArrayLike, points_b: ArrayLike, fundamental: ArrayLike) -> NDArray[np.float64]:
    """Return first-order geometric errors for corresponding image points."""

    first = np.asarray(points_a, dtype=np.float64)
    second = np.asarray(points_b, dtype=np.float64)
    matrix = np.asarray(fundamental, dtype=np.float64)
    if first.ndim != 2 or first.shape[1:] != (2,) or second.shape != first.shape or matrix.shape != (3, 3):
        raise ValueError("correspondences must share shape (N, 2) and fundamental must be 3x3")
    first_h = np.column_stack((first, np.ones(len(first))))
    second_h = np.column_stack((second, np.ones(len(second))))
    fx = (matrix @ first_h.T).T
    ftx = (matrix.T @ second_h.T).T
    residual = np.sum(second_h * fx, axis=1)
    denominator = fx[:, 0] ** 2 + fx[:, 1] ** 2 + ftx[:, 0] ** 2 + ftx[:, 1] ** 2
    return residual**2 / np.maximum(denominator, np.finfo(np.float64).eps)


def triangulate_points(
    projection_a: ArrayLike,
    projection_b: ArrayLike,
    points_a: ArrayLike,
    points_b: ArrayLike,
) -> NDArray[np.float64]:
    """Recover 3D points by solving a four-row homogeneous system per match."""

    first_matrix = np.asarray(projection_a, dtype=np.float64)
    second_matrix = np.asarray(projection_b, dtype=np.float64)
    first = np.asarray(points_a, dtype=np.float64)
    second = np.asarray(points_b, dtype=np.float64)
    if first_matrix.shape != (3, 4) or second_matrix.shape != (3, 4):
        raise ValueError("projection matrices must have shape (3, 4)")
    if first.ndim != 2 or first.shape[1:] != (2,) or second.shape != first.shape:
        raise ValueError("image points must share shape (N, 2)")

    reconstructed = []
    for (x, y), (u, v) in zip(first, second):
        system = np.vstack((
            x * first_matrix[2] - first_matrix[0],
            y * first_matrix[2] - first_matrix[1],
            u * second_matrix[2] - second_matrix[0],
            v * second_matrix[2] - second_matrix[1],
        ))
        _, _, vh = np.linalg.svd(system)
        homogeneous = vh[-1]
        if abs(homogeneous[3]) <= np.finfo(np.float64).eps:
            raise ValueError("triangulation produced a point at infinity")
        reconstructed.append(homogeneous[:3] / homogeneous[3])
    return np.asarray(reconstructed, dtype=np.float64).reshape(-1, 3)
