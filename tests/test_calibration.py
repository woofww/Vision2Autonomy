import numpy as np
import pytest

from vision2autonomy.examples.reference_images import CalibrationScene, synthetic_calibration_scene
from vision2autonomy.geometry import (
    calibrate_camera,
    distort_points,
    estimate_homography_dlt,
    estimate_intrinsics,
    planar_object_points,
    project_points,
    reprojection_errors,
    undistort_image,
    undistort_points,
)
from vision2autonomy.geometry.multiview import camera_matrix, project_points_3d


def test_planar_object_points_are_on_z_zero_plane() -> None:
    points = planar_object_points(5, 7, 0.025)
    assert points.shape == (35, 3)
    np.testing.assert_allclose(points[:, 2], 0.0)
    np.testing.assert_allclose(points[0, :2], [0.0, 0.0])
    np.testing.assert_allclose(points[1, :2], [0.025, 0.0])


def test_distortion_undistortion_round_trip() -> None:
    rng = np.random.default_rng(3)
    # Real normalized coordinates stay inside the image, where the model is
    # monotonic; far outside the image the polynomial is not invertible.
    points = rng.uniform(-0.8, 0.8, size=(200, 2))
    coefficients = (-0.25, 0.06, 0.004, -0.003)
    distorted = distort_points(points, *coefficients)
    recovered = undistort_points(distorted, *coefficients, iterations=20)
    np.testing.assert_allclose(recovered, points, atol=1e-6)


def test_negative_radial_coefficient_pulls_points_inward() -> None:
    points = np.array([[0.5, 0.5], [-0.6, 0.2], [0.0, 0.8]])
    distorted = distort_points(points, k1=-0.2)
    np.testing.assert_array_less(
        np.linalg.norm(distorted, axis=1), np.linalg.norm(points, axis=1)
    )


def test_distort_points_validates_shape() -> None:
    with pytest.raises(ValueError, match="shape"):
        distort_points(np.zeros((3, 3)))


def test_project_points_matches_pinhole_without_distortion() -> None:
    intrinsics = np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]])
    rotation = np.array([[0.98, 0.0, 0.2], [0.0, 1.0, 0.0], [-0.2, 0.0, 0.98]])
    translation = np.array([-0.5, 0.0, 5.0])
    world = np.array([[-1.0, -0.5, 4.0], [0.0, 0.2, 6.0], [1.2, 0.4, 8.0]])
    projection = camera_matrix(intrinsics, rotation, translation)
    np.testing.assert_allclose(
        project_points(world, intrinsics, rotation, translation),
        project_points_3d(world, projection),
        atol=1e-10,
    )


def test_dlt_homography_recovers_known_transform() -> None:
    rng = np.random.default_rng(4)
    source = rng.uniform(-1.0, 1.0, size=(20, 2))
    homography = np.array(
        [[1.2, 0.15, 0.3], [-0.1, 0.9, -0.2], [0.05, -0.03, 1.0]]
    )
    homogeneous = np.column_stack((source, np.ones(len(source))))
    projected = (homography @ homogeneous.T).T
    destination = projected[:, :2] / projected[:, 2, None]
    estimated = estimate_homography_dlt(source, destination)
    np.testing.assert_allclose(estimated, homography, atol=1e-9)


def test_estimate_intrinsics_requires_three_homographies() -> None:
    with pytest.raises(ValueError, match="at least three"):
        estimate_intrinsics([np.eye(3), np.eye(3)])


def test_calibration_recovers_synthetic_camera_without_noise() -> None:
    scene: CalibrationScene = synthetic_calibration_scene(num_views=8)
    result = calibrate_camera(scene.image_points, scene.object_points, iterations=30)

    np.testing.assert_allclose(
        result.intrinsics[:2, :2], scene.intrinsics[:2, :2], atol=1.5
    )
    np.testing.assert_allclose(
        result.intrinsics[:2, 2], scene.intrinsics[:2, 2], atol=1.5
    )
    assert abs(result.distortion[0] - scene.distortion[0]) < 0.015
    assert abs(result.distortion[1] - scene.distortion[1]) < 0.02
    assert abs(result.distortion[2] - scene.distortion[2]) < 0.003
    assert abs(result.distortion[3] - scene.distortion[3]) < 0.003
    assert result.rmse < 0.05


def test_calibration_is_stable_under_detection_noise() -> None:
    scene: CalibrationScene = synthetic_calibration_scene(num_views=8)
    rng = np.random.default_rng(5)
    noisy_views = [
        view + rng.normal(0.0, 0.3, view.shape) for view in scene.image_points
    ]
    result = calibrate_camera(noisy_views, scene.object_points, iterations=30)

    np.testing.assert_allclose(
        result.intrinsics[:2, :2], scene.intrinsics[:2, :2], atol=6.0
    )
    assert result.rmse < 0.5
    assert result.rmse < result.initial_rmse


def test_calibration_requires_three_views() -> None:
    scene: CalibrationScene = synthetic_calibration_scene(num_views=2)
    with pytest.raises(ValueError, match="at least three views"):
        calibrate_camera(scene.image_points, scene.object_points)


def test_reprojection_errors_shape_matches_rmse() -> None:
    scene: CalibrationScene = synthetic_calibration_scene(num_views=3)
    result = calibrate_camera(scene.image_points, scene.object_points, iterations=20)
    errors = reprojection_errors(
        scene.image_points,
        scene.object_points,
        result.intrinsics,
        result.distortion,
        result.rotations,
        result.translations,
    )
    assert errors.shape == (3 * len(scene.object_points),)
    assert float(np.sqrt(np.mean(errors * errors))) == pytest.approx(result.rmse, rel=1e-9)


def test_undistort_image_recovers_ideal_cell_centers() -> None:
    scene: CalibrationScene = synthetic_calibration_scene(num_views=1, seed=9)
    distorted = scene.images[0]
    corrected = undistort_image(distorted, scene.intrinsics, scene.distortion)
    assert corrected.shape == distorted.shape

    sampled: list[float] = []
    expected: list[float] = []
    for row in range(1, scene.squares_rows):
        for col in range(1, scene.squares_cols):
            x = (col + 0.5) * scene.square_size
            y = (row + 0.5) * scene.square_size
            pixel = project_points(
                [[x, y, 0.0]],
                scene.intrinsics,
                scene.rotations[0],
                scene.translations[0],
            )[0]
            if 2 <= pixel[0] < 398 and 2 <= pixel[1] < 298:
                sampled.append(corrected[int(round(pixel[1])), int(round(pixel[0]))])
                expected.append(245 if (col + row) % 2 == 0 else 28)

    sampled_values = np.asarray(sampled)
    expected_values = np.asarray(expected)
    assert len(sampled_values) > 20
    assert float(np.mean(np.abs(sampled_values - expected_values))) < 12.0