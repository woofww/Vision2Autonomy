import numpy as np
import pytest

from vision2autonomy.edges.canny import (
    CannyResult,
    canny,
    double_threshold,
    hysteresis,
    non_maximum_suppression,
    sobel_gradients,
)


def test_sobel_returns_zero_for_constant_image() -> None:
    _, _, magnitude, _ = sobel_gradients(np.full((15, 15), 80.0))
    np.testing.assert_allclose(magnitude, 0.0, atol=1e-12)


def test_non_maximum_suppression_retains_only_directional_peak() -> None:
    magnitude = np.zeros((5, 5))
    magnitude[2, 1:4] = [2.0, 8.0, 3.0]
    direction = np.zeros((5, 5))

    result = non_maximum_suppression(magnitude, direction)

    assert result[2, 2] == 8.0
    assert result[2, 1] == 0.0
    assert result[2, 3] == 0.0


def test_hysteresis_keeps_connected_weak_chain_only() -> None:
    strong = np.zeros((6, 6), dtype=bool)
    weak = np.zeros_like(strong)
    strong[1, 1] = True
    weak[2, 2] = True
    weak[3, 3] = True
    weak[5, 5] = True

    result = hysteresis(strong, weak)

    assert result[1, 1] == 255
    assert result[2, 2] == 255
    assert result[3, 3] == 255
    assert result[5, 5] == 0


def test_canny_detects_synthetic_vertical_step() -> None:
    image = np.zeros((64, 64), dtype=np.uint8)
    image[:, 32:] = 255

    result = canny(image, low_threshold=20, high_threshold=50, return_intermediates=True)

    assert isinstance(result, CannyResult)
    edge_locations = np.argwhere(result.edges > 0)
    assert edge_locations.size > 0
    assert set(np.unique(result.edges)).issubset({0, 255})
    assert np.all((edge_locations[:, 1] >= 30) & (edge_locations[:, 1] <= 33))
    assert np.count_nonzero(result.edges[:, :28]) == 0
    assert np.count_nonzero(result.edges[:, 36:]) == 0


def test_canny_constant_image_has_no_edges() -> None:
    result = canny(np.full((32, 32), 128, dtype=np.uint8))
    assert np.count_nonzero(result) == 0


@pytest.mark.parametrize(
    ("low", "high"),
    [(-1.0, 10.0), (10.0, 10.0), (20.0, 10.0)],
)
def test_double_threshold_rejects_invalid_thresholds(low: float, high: float) -> None:
    with pytest.raises(ValueError, match="thresholds"):
        double_threshold(np.ones((3, 3)), low, high)


def test_canny_rejects_color_and_non_finite_images() -> None:
    with pytest.raises(ValueError, match="grayscale"):
        canny(np.zeros((10, 10, 3)))
    with pytest.raises(ValueError, match="finite"):
        canny(np.array([[0.0, np.nan]]))

