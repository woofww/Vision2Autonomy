"""Camera calibration: Zhang's planar method with radial-tangential distortion.

The full pipeline is implemented from scratch with NumPy:

1.  Normalized DLT homographies from planar checkerboard corners;
2.  Zhang's closed-form estimate of the intrinsic matrix ``K``;
3.  Per-view extrinsic recovery from each homography;
4.  Levenberg-Marquardt refinement of intrinsics, distortion, and extrinsics
    by minimizing reprojection error.

The distortion model is the Brown-Conrady model used by OpenCV, but every
step here is explicit, testable, and free of framework calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


def planar_object_points(rows: int, cols: int, square_size: float) -> NDArray[np.float64]:
    """Return ``(rows * cols, 3)`` checkerboard corners on the plane ``Z = 0``.

    ``rows`` and ``cols`` are the numbers of *inner* corners, not squares.
    Point ``(0, 0, 0)`` sits at the plane origin and the grid extends along
    the positive X and Y axes.
    """

    rows = int(rows)
    cols = int(cols)
    square_size = float(square_size)
    if rows < 2 or cols < 2:
        raise ValueError("rows and cols must both be at least 2")
    if square_size <= 0.0:
        raise ValueError("square_size must be positive")
    y, x = np.mgrid[0:rows, 0:cols]
    return np.column_stack(
        (x.ravel() * square_size, y.ravel() * square_size, np.zeros(rows * cols))
    )


def _normalize_points(
    points: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Shift points to the centroid and scale them to mean distance sqrt(2)."""

    center = points.mean(axis=0)
    shifted = points - center
    mean_distance = float(np.mean(np.linalg.norm(shifted, axis=1)))
    if mean_distance <= np.finfo(np.float64).eps:
        raise ValueError("points must not all coincide")
    scale = np.sqrt(2.0) / mean_distance
    transform = np.array(
        [
            [scale, 0.0, -scale * center[0]],
            [0.0, scale, -scale * center[1]],
            [0.0, 0.0, 1.0],
        ]
    )
    homogeneous = np.column_stack((points, np.ones(len(points))))
    normalized = (transform @ homogeneous.T).T
    return normalized[:, :2], transform


def estimate_homography_dlt(source: ArrayLike, destination: ArrayLike) -> NDArray[np.float64]:
    """Estimate a 2-D homography with the normalized DLT algorithm."""

    src = np.asarray(source, dtype=np.float64)
    dst = np.asarray(destination, dtype=np.float64)
    if src.ndim != 2 or src.shape[1:] != (2,) or dst.shape != src.shape or len(src) < 4:
        raise ValueError("point arrays must share shape (N, 2), N >= 4")
    if not np.isfinite(src).all() or not np.isfinite(dst).all():
        raise ValueError("points must be finite")

    src_n, source_transform = _normalize_points(src)
    dst_n, destination_transform = _normalize_points(dst)
    x, y = src_n.T
    u, v = dst_n.T
    zero = np.zeros(len(src))
    design = np.vstack(
        (
            np.column_stack((-x, -y, -np.ones(len(src)), zero, zero, zero, u * x, u * y, u)),
            np.column_stack((zero, zero, zero, -x, -y, -np.ones(len(src)), v * x, v * y, v)),
        )
    )
    _, _, vh = np.linalg.svd(design)
    homography = vh[-1].reshape(3, 3)
    matrix = np.linalg.inv(destination_transform) @ homography @ source_transform
    if abs(matrix[2, 2]) <= np.finfo(np.float64).eps:
        raise ValueError("degenerate homography")
    return matrix / matrix[2, 2]


def distort_points(
    points: ArrayLike,
    k1: float = 0.0,
    k2: float = 0.0,
    p1: float = 0.0,
    p2: float = 0.0,
) -> NDArray[np.float64]:
    """Apply radial and tangential (Brown-Conrady) distortion.

    Input and output are *normalized* coordinates, i.e. pixel coordinates
    divided through ``K`` so that the focal length no longer appears.
    """

    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1:] != (2,):
        raise ValueError("points must have shape (N, 2)")
    coefficients = np.array([k1, k2, p1, p2], dtype=np.float64)
    if not np.isfinite(coefficients).all():
        raise ValueError("distortion coefficients must be finite")
    x, y = pts.T
    radius_squared = x * x + y * y
    radial = 1.0 + k1 * radius_squared + k2 * radius_squared * radius_squared
    distorted_x = x * radial + 2.0 * p1 * x * y + p2 * (radius_squared + 2.0 * x * x)
    distorted_y = y * radial + p1 * (radius_squared + 2.0 * y * y) + 2.0 * p2 * x * y
    return np.column_stack((distorted_x, distorted_y))


def undistort_points(
    points: ArrayLike,
    k1: float = 0.0,
    k2: float = 0.0,
    p1: float = 0.0,
    p2: float = 0.0,
    iterations: int = 6,
) -> NDArray[np.float64]:
    """Invert the distortion model with fixed-point iteration."""

    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1:] != (2,):
        raise ValueError("points must have shape (N, 2)")
    if int(iterations) < 1:
        raise ValueError("iterations must be positive")
    x, y = pts.T
    for _ in range(int(iterations)):
        radius_squared = x * x + y * y
        radial = 1.0 + k1 * radius_squared + k2 * radius_squared * radius_squared
        x = (pts[:, 0] - 2.0 * p1 * x * y - p2 * (radius_squared + 2.0 * x * x)) / radial
        y = (pts[:, 1] - p1 * (radius_squared + 2.0 * y * y) - 2.0 * p2 * x * y) / radial
    return np.column_stack((x, y))


def project_points(
    points_3d: ArrayLike,
    intrinsics: ArrayLike,
    rotation: ArrayLike,
    translation: ArrayLike,
    distortion: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """Project world points to pixels: ``x = K (R X + t)`` plus distortion.

    Without distortion this is identical to ``project_points_3d`` from the
    multiview module; with distortion it applies the Brown-Conrady model in
    normalized coordinates before multiplying by ``K``.
    """

    world = np.asarray(points_3d, dtype=np.float64)
    k = np.asarray(intrinsics, dtype=np.float64)
    r = np.asarray(rotation, dtype=np.float64)
    t = np.asarray(translation, dtype=np.float64).reshape(-1)
    if world.ndim != 2 or world.shape[1:] != (3,):
        raise ValueError("points_3d must have shape (N, 3)")
    if k.shape != (3, 3) or r.shape != (3, 3) or t.shape != (3,):
        raise ValueError("intrinsics and rotation must be 3x3; translation must have length 3")
    if (
        not np.isfinite(world).all()
        or not np.isfinite(k).all()
        or not np.isfinite(r).all()
        or not np.isfinite(t).all()
    ):
        raise ValueError("camera parameters and points must be finite")

    camera = (r @ world.T).T + t
    if np.any(camera[:, 2] <= 0.0):
        raise ValueError("all points must be in front of the camera")
    normalized = camera[:, :2] / camera[:, 2, None]
    if distortion is not None:
        coefficients = np.asarray(distortion, dtype=np.float64)
        if coefficients.shape != (4,):
            raise ValueError("distortion must have shape (4,)")
        normalized = distort_points(normalized, *coefficients)
    homogeneous = np.column_stack((normalized, np.ones(len(normalized))))
    pixels = (k @ homogeneous.T).T
    return pixels[:, :2]


def _homography_constraints(homography: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return Zhang's two linear constraints ``(v12, v11 - v22)`` for one H."""

    columns = [homography[:, index] for index in range(3)]
    vectors: list[NDArray[np.float64]] = []
    for first, second in ((0, 1), (0, 0), (1, 1)):
        a, b = columns[first], columns[second]
        vectors.append(
            np.array(
                [
                    a[0] * b[0],
                    a[0] * b[1] + a[1] * b[0],
                    a[1] * b[1],
                    a[2] * b[0] + a[0] * b[2],
                    a[2] * b[1] + a[1] * b[2],
                    a[2] * b[2],
                ]
            )
        )
    return np.vstack((vectors[0], vectors[1] - vectors[2]))


def estimate_intrinsics(homographies: Sequence[ArrayLike]) -> NDArray[np.float64]:
    """Closed-form Zhang estimate of ``K`` from at least three homographies."""

    if len(homographies) < 3:
        raise ValueError("at least three homographies are required")
    matrix = np.vstack([_homography_constraints(np.asarray(h, dtype=np.float64)) for h in homographies])
    _, _, vh = np.linalg.svd(matrix)
    b = vh[-1]
    b11, b12, b22, b13, b23, b33 = b
    denominator = b11 * b22 - b12 * b12
    if abs(denominator) <= np.finfo(np.float64).eps:
        raise ValueError("homographies are degenerate")
    v0 = (b12 * b13 - b11 * b23) / denominator
    lambda_value = b33 - (b13 * b13 + v0 * (b12 * b13 - b11 * b23)) / b11
    if lambda_value <= 0.0:
        # b is determined up to sign; flipping it keeps focal lengths real.
        b = -b
        b11, b12, b22, b13, b23, b33 = b
        v0 = (b12 * b13 - b11 * b23) / denominator
        lambda_value = b33 - (b13 * b13 + v0 * (b12 * b13 - b11 * b23)) / b11
    if lambda_value <= 0.0:
        raise ValueError("could not recover a positive-definite intrinsic matrix")
    alpha = float(np.sqrt(lambda_value / b11))
    beta = float(np.sqrt(lambda_value * b11 / denominator))
    gamma = float(-b12 * alpha * alpha * beta / lambda_value)
    u0 = float(gamma * v0 / beta - b13 * alpha * alpha / lambda_value)
    return np.array([[alpha, gamma, u0], [0.0, beta, v0], [0.0, 0.0, 1.0]])


def estimate_view_pose(
    intrinsics: ArrayLike,
    homography: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Recover rotation and translation from one planar homography."""

    k = np.asarray(intrinsics, dtype=np.float64)
    h = np.asarray(homography, dtype=np.float64)
    k_inv = np.linalg.inv(k)
    r1 = k_inv @ h[:, 0]
    r2 = k_inv @ h[:, 1]
    translation = k_inv @ h[:, 2]
    norm = 0.5 * (np.linalg.norm(r1) + np.linalg.norm(r2))
    if norm <= np.finfo(np.float64).eps:
        raise ValueError("homography is degenerate")
    r1 = r1 / norm
    r2 = r2 / norm
    translation = translation / norm
    r3 = np.cross(r1, r2)
    rotation = np.column_stack((r1, r2, r3))
    u_matrix, _, vh = np.linalg.svd(rotation)
    rotation = u_matrix @ vh
    if np.linalg.det(rotation) < 0.0:
        rotation = -rotation
    return rotation, translation


def _rotation_to_axis_angle(rotation: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert a rotation matrix to a Rodrigues axis-angle vector."""

    trace = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(trace)
    if theta < 1e-10:
        return np.zeros(3)
    axis = np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ]
    )
    return axis * theta / (2.0 * np.sin(theta))


def _axis_angle_to_rotation(axis_angle: ArrayLike) -> NDArray[np.float64]:
    """Convert a Rodrigues axis-angle vector to a rotation matrix."""

    vector = np.asarray(axis_angle, dtype=np.float64).reshape(-1)
    theta = float(np.linalg.norm(vector))
    if theta < 1e-12:
        return np.eye(3)
    axis = vector / theta
    kx, ky, kz = axis
    skew = np.array([[0.0, -kz, ky], [kz, 0.0, -kx], [-ky, kx, 0.0]])
    return (
        np.cos(theta) * np.eye(3)
        + (1.0 - np.cos(theta)) * np.outer(axis, axis)
        + np.sin(theta) * skew
    )


def _levenberg_marquardt(
    residuals,
    initial: ArrayLike,
    iterations: int = 40,
    step: float = 1e-6,
) -> NDArray[np.float64]:
    """Minimize ``sum(residuals(params)**2)`` with a damped Gauss-Newton loop.

    The Jacobian is estimated by forward finite differences, which keeps the
    implementation dependency-free and makes the loop easy to inspect.
    """

    params = np.asarray(initial, dtype=np.float64).copy()
    damping = 1e-3
    residual = np.asarray(residuals(params), dtype=np.float64)
    cost = float(np.sum(residual * residual))
    for _ in range(int(iterations)):
        jacobian = np.empty((residual.size, params.size), dtype=np.float64)
        for index in range(params.size):
            delta = np.zeros(params.size)
            delta[index] = step
            jacobian[:, index] = (np.asarray(residuals(params + delta)) - residual) / step
        normal = jacobian.T @ jacobian
        gradient = jacobian.T @ residual
        diagonal = np.diag(normal).copy()
        diagonal[diagonal <= np.finfo(np.float64).eps] = 1.0
        accepted = False
        for _ in range(12):
            try:
                update = np.linalg.solve(normal + damping * np.diag(diagonal), -gradient)
            except np.linalg.LinAlgError:
                damping *= 10.0
                continue
            candidate = params + update
            candidate_residual = np.asarray(residuals(candidate), dtype=np.float64)
            candidate_cost = float(np.sum(candidate_residual * candidate_residual))
            if candidate_cost < cost:
                params = candidate
                residual = candidate_residual
                cost = candidate_cost
                damping = max(damping * 0.5, 1e-12)
                accepted = True
                break
            damping *= 10.0
            if damping > 1e12:
                break
        if not accepted:
            break
    return params


def reprojection_errors(
    views: Sequence[ArrayLike],
    object_points: ArrayLike,
    intrinsics: ArrayLike,
    distortion: ArrayLike,
    rotations: Sequence[ArrayLike],
    translations: Sequence[ArrayLike],
) -> NDArray[np.float64]:
    """Return per-corner Euclidean reprojection errors across all views."""

    world = np.asarray(object_points, dtype=np.float64)
    k = np.asarray(intrinsics, dtype=np.float64)
    coefficients = np.asarray(distortion, dtype=np.float64)
    errors: list[NDArray[np.float64]] = []
    for index, observed in enumerate(views):
        predicted = project_points(
            world, k, rotations[index], translations[index], coefficients
        )
        errors.append(
            np.linalg.norm(predicted - np.asarray(observed, dtype=np.float64), axis=1)
        )
    return np.concatenate(errors)


def _reprojection_rmse(
    views: Sequence[ArrayLike],
    object_points: ArrayLike,
    intrinsics: ArrayLike,
    distortion: ArrayLike,
    rotations: Sequence[ArrayLike],
    translations: Sequence[ArrayLike],
) -> float:
    errors = reprojection_errors(
        views, object_points, intrinsics, distortion, rotations, translations
    )
    return float(np.sqrt(np.mean(errors * errors)))


def _bilinear_sample(
    image: NDArray[np.float64],
    xs: NDArray[np.float64],
    ys: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Sample a grayscale image at float pixel coordinates."""

    height, width = image.shape[:2]
    x0 = np.floor(xs).astype(np.int64)
    y0 = np.floor(ys).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1
    wx = xs - x0
    wy = ys - y0
    x0 = np.clip(x0, 0, width - 1)
    x1 = np.clip(x1, 0, width - 1)
    y0 = np.clip(y0, 0, height - 1)
    y1 = np.clip(y1, 0, height - 1)
    return (
        image[y0, x0] * (1.0 - wx) * (1.0 - wy)
        + image[y1, x0] * (1.0 - wx) * wy
        + image[y0, x1] * wx * (1.0 - wy)
        + image[y1, x1] * wx * wy
    )


def undistort_image(
    image: ArrayLike,
    intrinsics: ArrayLike,
    distortion: ArrayLike,
) -> NDArray[np.float64]:
    """Remap a distorted grayscale image back to the ideal pinhole image.

    The distorted image stores ideal content at shifted positions, so every
    output pixel finds its source by applying the *forward* distortion model:
    pixel -> normalized -> distorted normalized -> source pixel, then samples
    with bilinear interpolation.
    """

    array = np.asarray(image, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("undistort_image supports grayscale images only")
    height, width = array.shape
    k = np.asarray(intrinsics, dtype=np.float64)
    coefficients = np.asarray(distortion, dtype=np.float64)
    if coefficients.shape != (4,):
        raise ValueError("distortion must have shape (4,)")
    rows, columns = np.indices((height, width))
    pixels = np.column_stack((columns.ravel(), rows.ravel()))
    normalized_h = (np.linalg.inv(k) @ np.column_stack((pixels, np.ones(len(pixels)))).T).T
    distorted = distort_points(normalized_h[:, :2], *coefficients)
    source_h = (k @ np.column_stack((distorted, np.ones(len(distorted)))).T).T
    sampled = _bilinear_sample(array, source_h[:, 0], source_h[:, 1])
    return sampled.reshape(height, width)


@dataclass
class CalibrationResult:
    """Outcome of :func:`calibrate_camera`."""

    intrinsics: NDArray[np.float64]
    distortion: NDArray[np.float64]
    rotations: list[NDArray[np.float64]]
    translations: list[NDArray[np.float64]]
    rmse: float
    initial_intrinsics: NDArray[np.float64]
    initial_rmse: float
    iterations: int


def calibrate_camera(
    views: Sequence[ArrayLike],
    object_points: ArrayLike,
    *,
    iterations: int = 40,
) -> CalibrationResult:
    """Estimate intrinsics, distortion, and per-view poses with Zhang's method.

    ``views`` is one ``(N, 2)`` array of detected corners per image and
    ``object_points`` is the shared ``(N, 3)`` planar checkerboard grid.
    At least three views are required; more views with diverse pose make the
    estimate more stable.
    """

    observed_views = [np.asarray(view, dtype=np.float64) for view in views]
    world = np.asarray(object_points, dtype=np.float64)
    if world.ndim != 2 or world.shape[1:] != (3,):
        raise ValueError("object_points must have shape (N, 3)")
    if len(observed_views) < 3:
        raise ValueError("at least three views are required")
    if any(view.shape != world[:, :2].shape for view in observed_views):
        raise ValueError("every view must contain one corner per object point")
    if any(not np.isfinite(view).all() for view in observed_views):
        raise ValueError("image points must be finite")
    if not np.isfinite(world).all():
        raise ValueError("object points must be finite")

    homographies = [
        estimate_homography_dlt(world[:, :2], view) for view in observed_views
    ]
    intrinsics = estimate_intrinsics(homographies)
    rotations: list[NDArray[np.float64]] = []
    translations: list[NDArray[np.float64]] = []
    for homography in homographies:
        rotation, translation = estimate_view_pose(intrinsics, homography)
        rotations.append(rotation)
        translations.append(translation)

    zero_distortion = np.zeros(4)
    initial_rmse = _reprojection_rmse(
        observed_views, world, intrinsics, zero_distortion, rotations, translations
    )

    def residuals(params: NDArray[np.float64]) -> NDArray[np.float64]:
        fx, fy, cx, cy = params[0:4]
        intrinsics_guess = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
        distortion_guess = params[4:8]
        errors: list[NDArray[np.float64]] = []
        for index, observed in enumerate(observed_views):
            rvec = params[8 + 6 * index : 11 + 6 * index]
            tvec = params[11 + 6 * index : 14 + 6 * index]
            predicted = project_points(
                world,
                intrinsics_guess,
                _axis_angle_to_rotation(rvec),
                tvec,
                distortion_guess,
            )
            errors.append((predicted - observed).ravel())
        return np.concatenate(errors)

    initial = np.concatenate(
        (
            [intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]],
            zero_distortion,
            np.concatenate(
                [
                    np.concatenate((_rotation_to_axis_angle(r), t))
                    for r, t in zip(rotations, translations)
                ]
            ),
        )
    )
    refined = _levenberg_marquardt(residuals, initial, iterations=iterations)
    fx, fy, cx, cy = refined[0:4]
    intrinsics_refined = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    distortion_refined = refined[4:8]
    rotations_refined: list[NDArray[np.float64]] = []
    translations_refined: list[NDArray[np.float64]] = []
    for index in range(len(observed_views)):
        rotations_refined.append(
            _axis_angle_to_rotation(refined[8 + 6 * index : 11 + 6 * index])
        )
        translations_refined.append(refined[11 + 6 * index : 14 + 6 * index])

    final_rmse = _reprojection_rmse(
        observed_views,
        world,
        intrinsics_refined,
        distortion_refined,
        rotations_refined,
        translations_refined,
    )
    return CalibrationResult(
        intrinsics=intrinsics_refined,
        distortion=distortion_refined,
        rotations=rotations_refined,
        translations=translations_refined,
        rmse=final_rmse,
        initial_intrinsics=intrinsics,
        initial_rmse=initial_rmse,
        iterations=int(iterations),
    )