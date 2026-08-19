import numpy as np
import pytest

from vision2autonomy.image.filters import gaussian_blur, gaussian_kernel


def test_gaussian_kernel_is_normalized_and_symmetric() -> None:
    kernel = gaussian_kernel(5, 1.2)

    assert kernel.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(kernel, kernel[::-1, :])
    np.testing.assert_allclose(kernel, kernel[:, ::-1])
    assert kernel[2, 2] == kernel.max()


@pytest.mark.parametrize(("size", "sigma"), [(0, 1.0), (4, 1.0), (3, 0.0)])
def test_gaussian_kernel_rejects_invalid_parameters(size: int, sigma: float) -> None:
    with pytest.raises(ValueError):
        gaussian_kernel(size, sigma)


def test_gaussian_blur_preserves_constant_image() -> None:
    image = np.full((13, 17), 42.0)
    np.testing.assert_allclose(gaussian_blur(image), image, atol=1e-12)

