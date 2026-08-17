import numpy as np
import pytest

from vision2autonomy.image.convolution import convolve2d


def naive_convolution(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    pad_y, pad_x = kernel.shape[0] // 2, kernel.shape[1] // 2
    padded = np.pad(image, ((pad_y, pad_y), (pad_x, pad_x)), mode="constant")
    output = np.zeros(image.shape, dtype=float)
    flipped = kernel[::-1, ::-1]
    for row in range(image.shape[0]):
        for column in range(image.shape[1]):
            patch = padded[row : row + kernel.shape[0], column : column + kernel.shape[1]]
            output[row, column] = np.sum(patch * flipped)
    return output


def test_convolution_matches_naive_reference() -> None:
    image = np.arange(20, dtype=float).reshape(4, 5)
    kernel = np.array([[1.0, 2.0, 0.0], [0.0, -1.0, 3.0], [2.0, 1.0, 0.0]])

    actual = convolve2d(image, kernel, padding="constant")

    np.testing.assert_allclose(actual, naive_convolution(image, kernel))


@pytest.mark.parametrize(
    ("image", "kernel", "message"),
    [
        (np.zeros((2, 2, 3)), np.ones((3, 3)), "grayscale"),
        (np.zeros((3, 3)), np.ones((2, 2)), "odd"),
    ],
)
def test_convolution_rejects_invalid_shapes(image, kernel, message) -> None:
    with pytest.raises(ValueError, match=message):
        convolve2d(image, kernel)


def test_convolution_identity_kernel() -> None:
    image = np.array([[2.0, 4.0], [6.0, 8.0]])
    np.testing.assert_array_equal(convolve2d(image, [[1.0]]), image)

