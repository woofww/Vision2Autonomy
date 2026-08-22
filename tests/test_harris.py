import numpy as np
import pytest

from vision2autonomy.features.harris import (
    detect_harris_corners,
    harris_response,
    select_corners,
)


def square_image(size: int = 64) -> np.ndarray:
    image = np.zeros((size, size), dtype=np.uint8)
    image[16:48, 16:48] = 255
    return image


def test_constant_image_has_zero_response_and_no_corners() -> None:
    result = detect_harris_corners(np.full((32, 32), 100, dtype=np.uint8))

    np.testing.assert_allclose(result.response, 0.0, atol=1e-12)
    assert result.corners.shape == (0, 2)


def test_square_detects_four_geometric_corners() -> None:
    result = detect_harris_corners(
        square_image(), threshold_rel=0.05, min_distance=8, max_corners=4
    )
    expected = np.array([[16, 16], [16, 47], [47, 16], [47, 47]])

    assert result.corners.shape == (4, 2)
    for corner in expected:
        distance = np.linalg.norm(result.corners - corner, axis=1)
        assert distance.min() <= 2.0


def test_corner_has_two_large_eigenvalues_while_edge_has_one() -> None:
    _, lambda_max, lambda_min = harris_response(square_image())

    corner = (16, 16)
    edge = (16, 32)
    assert lambda_min[corner] > 0.1 * lambda_max[corner]
    assert lambda_min[edge] < 0.02 * lambda_max[edge]


def test_select_corners_orders_by_strength_and_enforces_distance() -> None:
    response = np.zeros((15, 15))
    response[5, 5] = 10.0
    response[6, 6] = 9.0
    response[12, 12] = 8.0

    corners = select_corners(response, threshold_rel=0.5, min_distance=3)

    np.testing.assert_array_equal(corners, [[5, 5], [12, 12]])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"k": 0.0},
        {"k": 0.25},
        {"window_size": 4},
        {"sigma": 0.0},
    ],
)
def test_harris_rejects_invalid_parameters(kwargs) -> None:
    with pytest.raises(ValueError):
        harris_response(square_image(), **kwargs)


def test_harris_rejects_color_input() -> None:
    with pytest.raises(ValueError, match="grayscale"):
        harris_response(np.zeros((10, 10, 3)))

