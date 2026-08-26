import numpy as np
import pytest

from vision2autonomy.edges import hough_accumulator, hough_lines, hough_peaks


def test_hough_detects_horizontal_and_vertical_lines() -> None:
    image = np.zeros((100, 100), dtype=np.uint8)
    image[40, :] = 255  # horizontal line at row 40
    image[:, 60] = 255  # vertical line at column 60

    result = hough_lines(image, num_peaks=5, threshold_rel=0.1, min_support=5)
    thetas = result.lines[:, 1]
    rhos = result.lines[:, 0]

    horizontal = np.isclose(thetas, np.pi / 2, atol=np.deg2rad(2)) & np.isclose(rhos, 40.0, atol=2.0)
    vertical = np.isclose(thetas, 0.0, atol=np.deg2rad(2)) & np.isclose(rhos, 60.0, atol=2.0)
    assert horizontal.any(), "horizontal line was not detected"
    assert vertical.any(), "vertical line was not detected"


def test_hough_detects_diagonal_line() -> None:
    image = np.zeros((100, 100), dtype=np.uint8)
    rows, columns = np.indices((100, 100))
    image[rows == columns + 10] = 255  # y = x + 10

    result = hough_lines(image, num_peaks=3, threshold_rel=0.1, min_support=5)
    thetas = result.lines[:, 1]
    rhos = result.lines[:, 0]
    expected_theta = 3 * np.pi / 4
    expected_rho = 5 * np.sqrt(2.0)
    assert np.any(
        np.isclose(thetas, expected_theta, atol=np.deg2rad(2))
        & np.isclose(rhos, expected_rho, atol=2.0)
    )


def test_hough_accumulator_covers_parameter_space() -> None:
    image = np.zeros((100, 80), dtype=np.uint8)
    image[50, :] = 255
    accumulator, rhos, thetas = hough_accumulator(image)

    rho_max = float(np.hypot(100, 80))
    assert rhos.min() <= -rho_max
    assert rhos.max() >= rho_max
    assert thetas.min() >= 0.0
    assert thetas.max() < np.pi
    assert accumulator.shape == (len(rhos), len(thetas))
    assert accumulator.sum() == image.sum() // 255 * len(thetas)


def test_hough_peaks_are_strong_and_separated() -> None:
    accumulator = np.zeros((30, 60))
    accumulator[10, 20] = 100.0
    accumulator[10, 21] = 99.0
    accumulator[25, 50] = 80.0

    indices, votes = hough_peaks(accumulator, num_peaks=5, threshold_rel=0.5, neighborhood=3)
    np.testing.assert_array_equal(indices[0], [10, 20])
    np.testing.assert_array_equal(indices[1], [25, 50])
    np.testing.assert_allclose(votes, [100.0, 80.0])


def test_hough_empty_edge_map_yields_no_lines() -> None:
    result = hough_lines(np.zeros((32, 32), dtype=np.uint8))
    assert len(result.lines) == 0
    assert len(result.segments) == 0
    assert result.accumulator.sum() == 0.0


@pytest.mark.parametrize("kwargs", [{"rho_resolution": 0.0}, {"theta_resolution": 0.0}])
def test_hough_rejects_invalid_resolution(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        hough_accumulator(np.ones((8, 8), dtype=np.uint8), **kwargs)


def test_hough_rejects_three_dimensional_input() -> None:
    with pytest.raises(ValueError, match="two-dimensional"):
        hough_lines(np.zeros((8, 8, 3), dtype=np.uint8))


def test_hough_peaks_rejects_empty_accumulator() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        hough_peaks(np.empty((0, 0)))