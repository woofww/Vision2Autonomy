import numpy as np
import pytest

from vision2autonomy.image.morphology import closing, dilate, erode, opening


def test_erosion_removes_one_pixel_border() -> None:
    image = np.ones((5, 5), dtype=np.uint8) * 255
    eroded = erode(image)
    expected = np.zeros((5, 5), dtype=np.uint8)
    expected[1:4, 1:4] = 255
    np.testing.assert_array_equal(eroded, expected)


def test_dilation_expands_single_pixel_to_cross() -> None:
    image = np.zeros((5, 5), dtype=np.uint8)
    image[2, 2] = 255
    dilated = dilate(image)
    expected = np.zeros((5, 5), dtype=np.uint8)
    expected[1:4, 2] = 255
    expected[2, 1:4] = 255
    np.testing.assert_array_equal(dilated, expected)


def test_opening_removes_isolated_specks() -> None:
    image = np.zeros((9, 9), dtype=np.uint8)
    image[2:7, 2:7] = 255
    image[8, 8] = 255  # isolated diagonal speck
    square = np.ones((3, 3), dtype=bool)
    opened = opening(image, square)
    assert opened[8, 8] == 0
    assert opened[2:7, 2:7].all()


def test_cross_opening_rounds_square_corners() -> None:
    image = np.zeros((9, 9), dtype=np.uint8)
    image[2:7, 2:7] = 255
    opened = opening(image)  # default 3x3 cross
    assert opened[3:6, 3:6].all()
    assert opened[2, 2] == 0  # a cross structure cannot restore the corner


def test_closing_bridges_vertical_gap() -> None:
    image = np.zeros((9, 9), dtype=np.uint8)
    image[2:5, 2:7] = 255
    image[6:9, 2:7] = 255  # gap at row 5
    square = np.ones((3, 3), dtype=bool)
    closed = closing(image, square)
    assert closed[5, 2:7].all()


def test_structuring_element_smaller_than_image() -> None:
    image = np.ones((3, 3), dtype=np.uint8)
    eroded = erode(image)
    expected = np.zeros((3, 3), dtype=np.uint8)
    expected[1, 1] = 255
    np.testing.assert_array_equal(eroded, expected)


def test_oversized_structure_erases_or_fills() -> None:
    image = np.ones((3, 3), dtype=np.uint8)
    big = np.ones((5, 5), dtype=bool)
    np.testing.assert_array_equal(erode(image, big), np.zeros((3, 3), dtype=np.uint8))
    np.testing.assert_array_equal(dilate(image, big), np.ones((3, 3), dtype=np.uint8) * 255)


@pytest.mark.parametrize("operation", [erode, dilate, opening, closing])
def test_operations_reject_invalid_input(operation) -> None:
    with pytest.raises(ValueError, match="two-dimensional"):
        operation(np.zeros((2, 2, 2), dtype=np.uint8))
    with pytest.raises(ValueError, match="must not be empty"):
        operation(np.empty((0, 0), dtype=np.uint8))