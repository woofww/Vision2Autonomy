import numpy as np
import pytest

from vision2autonomy.geometry import (
    absolute_trajectory_error,
    accumulate_relative_poses,
    align_trajectory,
    compose_poses,
    invert_pose,
    pnp_odometry,
    pose_matrix,
    project_points,
    relative_translation_error,
    trajectory_from_poses,
)


def yaw_rotation(angle: float) -> np.ndarray:
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array([[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]])


def test_pose_inverse_and_composition_return_identity() -> None:
    transform = pose_matrix(yaw_rotation(0.2), [0.4, -0.1, 1.2])
    np.testing.assert_allclose(compose_poses(transform, invert_pose(transform)), np.eye(4), atol=1e-12)


def test_relative_pose_accumulation_recovers_centers() -> None:
    relative = [pose_matrix(np.eye(3), [0.0, 0.0, -1.0]) for _ in range(4)]
    poses = accumulate_relative_poses(relative)
    centers = trajectory_from_poses(poses)
    np.testing.assert_allclose(centers[:, 2], np.arange(5), atol=1e-12)


def test_umeyama_alignment_recovers_metric_and_monocular_trajectories() -> None:
    reference = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 1.0], [2.0, 0.2, 2.0], [3.0, 0.4, 2.5]])
    rotation = yaw_rotation(0.35)
    estimated = (0.4 * (rotation @ reference.T)).T + np.array([4.0, -1.0, 2.0])
    aligned = align_trajectory(estimated, reference, with_scale=True)
    np.testing.assert_allclose(aligned.aligned, reference, atol=1e-12)
    assert aligned.scale == pytest.approx(2.5)
    assert absolute_trajectory_error(estimated, reference, with_scale=True) < 1e-12


def test_pnp_odometry_recovers_synthetic_vehicle_trajectory() -> None:
    intrinsics = np.array([[620.0, 0.0, 320.0], [0.0, 620.0, 180.0], [0.0, 0.0, 1.0]])
    rng = np.random.default_rng(31)
    landmarks = rng.uniform([-5.0, -2.0, 8.0], [5.0, 3.0, 28.0], size=(55, 3))
    centers = np.array([[0.12 * index, 0.0, 0.65 * index] for index in range(7)])
    observations = []
    for index, center in enumerate(centers):
        rotation = yaw_rotation(np.radians(-0.7 * index))
        translation = -rotation @ center
        pixels = project_points(landmarks, intrinsics, rotation, translation)
        pixels += rng.normal(0.0, 0.25, pixels.shape)
        pixels[(index * 3) % len(pixels)] += [50.0, -45.0]
        if index > 1:
            pixels[index : index + 3] = np.nan
        observations.append(pixels)

    result = pnp_odometry(
        landmarks,
        observations,
        intrinsics,
        threshold=3.5,
        max_iterations=600,
        seed=9,
    )
    assert absolute_trajectory_error(result.centers, centers, align=False) < 0.09
    assert np.all(result.inlier_counts >= 45)
    assert np.all(result.median_errors < 3.0)


def test_relative_translation_error_exposes_accumulated_yaw_bias() -> None:
    reference = np.column_stack((np.zeros(8), np.zeros(8), np.arange(8)))
    angle = np.radians(np.arange(8) * 1.0)
    estimated = np.column_stack((np.sin(angle) * np.arange(8), np.zeros(8), np.cos(angle) * np.arange(8)))
    errors = relative_translation_error(estimated, reference, delta=1)
    assert errors[-1] > errors[0]


def test_pnp_odometry_rejects_frames_with_too_few_landmarks() -> None:
    with pytest.raises(ValueError, match="fewer than six"):
        pnp_odometry(np.ones((8, 3)), [np.full((8, 2), np.nan)], np.eye(3))
