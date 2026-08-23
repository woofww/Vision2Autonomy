import numpy as np
import pytest

from vision2autonomy.geometry import (
    camera_matrix,
    epipolar_lines,
    estimate_fundamental,
    project_points_3d,
    sampson_errors,
    triangulate_points,
)


def stereo_scene() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    intrinsics = np.array([[700.0, 0.0, 320.0], [0.0, 700.0, 240.0], [0.0, 0.0, 1.0]])
    first = camera_matrix(intrinsics, np.eye(3), np.zeros(3))
    second = camera_matrix(intrinsics, np.eye(3), np.array([-0.5, 0.0, 0.0]))
    world = np.array([
        [-1.0, -0.5, 4.0], [0.0, -0.4, 5.0], [1.0, -0.3, 6.0],
        [-0.8, 0.2, 7.0], [0.2, 0.3, 8.0], [1.1, 0.4, 9.0],
        [-1.2, 0.8, 10.0], [0.0, 0.9, 11.0], [1.3, 1.0, 12.0],
        [0.5, -0.9, 6.5], [-0.4, 1.2, 8.5], [1.4, 0.1, 10.5],
    ])
    return first, second, world, project_points_3d(world, first), project_points_3d(world, second)


def test_pinhole_projection_has_inverse_depth_scaling() -> None:
    projection = camera_matrix(np.eye(3), np.eye(3), np.zeros(3))
    pixels = project_points_3d([[2.0, 4.0, 2.0], [2.0, 4.0, 4.0]], projection)
    np.testing.assert_allclose(pixels, [[1.0, 2.0], [0.5, 1.0]])


def test_eight_point_fundamental_satisfies_epipolar_constraint() -> None:
    _, _, _, pixels_a, pixels_b = stereo_scene()
    fundamental = estimate_fundamental(pixels_a, pixels_b)
    assert np.linalg.matrix_rank(fundamental, tol=1e-10) == 2
    np.testing.assert_allclose(sampson_errors(pixels_a, pixels_b, fundamental), 0.0, atol=1e-20)
    lines = epipolar_lines(pixels_a[:2], fundamental)
    points_b_h = np.column_stack((pixels_b[:2], np.ones(2)))
    np.testing.assert_allclose(np.sum(lines * points_b_h, axis=1), 0.0, atol=1e-10)


def test_linear_triangulation_recovers_world_points() -> None:
    first, second, world, pixels_a, pixels_b = stereo_scene()
    reconstructed = triangulate_points(first, second, pixels_a, pixels_b)
    np.testing.assert_allclose(reconstructed, world, atol=1e-10)


def test_noisy_triangulation_remains_close() -> None:
    first, second, world, pixels_a, pixels_b = stereo_scene()
    rng = np.random.default_rng(5)
    reconstructed = triangulate_points(
        first,
        second,
        pixels_a + rng.normal(0, 0.1, pixels_a.shape),
        pixels_b + rng.normal(0, 0.1, pixels_b.shape),
    )
    assert np.median(np.linalg.norm(reconstructed - world, axis=1)) < 0.03


@pytest.mark.parametrize("count", [0, 4, 7])
def test_fundamental_requires_eight_points(count: int) -> None:
    with pytest.raises(ValueError, match="N >= 8"):
        estimate_fundamental(np.zeros((count, 2)), np.zeros((count, 2)))
