import numpy as np
import pytest

from vision2autonomy.geometry import (
    camera_center,
    estimate_pose_dlt,
    pose_reprojection_errors,
    project_points,
    ransac_pnp,
)


def axis_angle_rotation(vector: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(vector))
    axis = vector / angle
    x, y, z = axis
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


def pose_scene() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    intrinsics = np.array([[760.0, 0.0, 320.0], [0.0, 750.0, 240.0], [0.0, 0.0, 1.0]])
    rotation = axis_angle_rotation(np.array([0.08, -0.16, 0.04]))
    translation = np.array([0.25, -0.15, 7.0])
    rng = np.random.default_rng(12)
    world = rng.uniform([-2.0, -1.5, -1.0], [2.0, 1.5, 1.5], size=(30, 3))
    pixels = project_points(world, intrinsics, rotation, translation)
    return intrinsics, rotation, translation, world, pixels


def rotation_error_degrees(estimated: np.ndarray, expected: np.ndarray) -> float:
    cosine = np.clip((np.trace(estimated @ expected.T) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def test_dlt_recovers_exact_pose() -> None:
    intrinsics, expected_rotation, expected_translation, world, pixels = pose_scene()
    rotation, translation = estimate_pose_dlt(world, pixels, intrinsics)
    assert rotation_error_degrees(rotation, expected_rotation) < 1e-5
    np.testing.assert_allclose(translation, expected_translation, atol=1e-6)
    np.testing.assert_allclose(
        pose_reprojection_errors(world, pixels, intrinsics, rotation, translation),
        0.0,
        atol=1e-5,
    )


def test_ransac_pnp_rejects_bad_landmark_matches() -> None:
    intrinsics, expected_rotation, expected_translation, world, pixels = pose_scene()
    rng = np.random.default_rng(8)
    noisy = pixels + rng.normal(0.0, 0.35, pixels.shape)
    outlier_indices = np.array([2, 8, 17, 24, 28])
    noisy[outlier_indices] += rng.uniform(45.0, 90.0, (len(outlier_indices), 2))
    result = ransac_pnp(world, noisy, intrinsics, threshold=2.0, max_iterations=900, seed=4)

    assert not result.inliers[outlier_indices].any()
    assert int(result.inliers.sum()) >= 24
    assert rotation_error_degrees(result.rotation, expected_rotation) < 0.25
    assert np.linalg.norm(result.translation - expected_translation) < 0.03
    assert float(np.median(result.errors[result.inliers])) < 0.7


def test_camera_center_is_inverse_extrinsic_translation() -> None:
    _, rotation, translation, _, _ = pose_scene()
    center = camera_center(rotation, translation)
    np.testing.assert_allclose(rotation @ center + translation, np.zeros(3), atol=1e-12)


@pytest.mark.parametrize("count", [0, 4, 5])
def test_dlt_requires_six_points(count: int) -> None:
    with pytest.raises(ValueError, match="six"):
        estimate_pose_dlt(np.zeros((count, 3)), np.zeros((count, 2)), np.eye(3))


def test_dlt_rejects_coplanar_points() -> None:
    world = np.column_stack((np.arange(6), np.arange(6) ** 2, np.zeros(6)))
    with pytest.raises(ValueError, match="span three dimensions"):
        estimate_pose_dlt(world, np.zeros((6, 2)), np.eye(3))
