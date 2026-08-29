"""Binary morphological operations implemented from first principles.

Erosion and dilation are the two primitive operations. Opening removes small
foreground specks, closing fills small background gaps; both are defined by
composing the primitives. Results use the conventional 0/255 display encoding
that the rest of the repository uses for binary maps.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from numpy.lib.stride_tricks import sliding_window_view

# Default 3x3 cross: a pixel is compared against its four orthogonal neighbors.
CROSS_3 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)


def _as_binary(image: ArrayLike) -> NDArray[np.bool_]:
    values = np.asarray(image)
    if values.ndim != 2:
        raise ValueError("binary image must be two-dimensional")
    if values.size == 0:
        raise ValueError("binary image must not be empty")
    if not np.isfinite(values).all():
        raise ValueError("binary image must contain only finite values")
    return values != 0


def _as_structure(structuring_element: ArrayLike | None) -> NDArray[np.bool_]:
    if structuring_element is None:
        return CROSS_3
    values = np.asarray(structuring_element)
    if values.ndim != 2:
        raise ValueError("structuring element must be two-dimensional")
    if values.size == 0 or not np.any(values):
        raise ValueError("structuring element must contain at least one foreground pixel")
    return values != 0


def _to_uint8(binary: NDArray[np.bool_]) -> NDArray[np.uint8]:
    return (binary * 255).astype(np.uint8)


def erode(
    image: ArrayLike,
    structuring_element: ArrayLike | None = None,
) -> NDArray[np.uint8]:
    """Binary erosion: keep a pixel only when the structure fully fits.

    Background outside the image border is treated as zero, so erosion also
    removes foreground touching the border when the structure extends there.
    """

    binary = _as_binary(image)
    structure = _as_structure(structuring_element)
    window_shape = structure.shape
    if window_shape[0] > binary.shape[0] or window_shape[1] > binary.shape[1]:
        return _to_uint8(np.zeros_like(binary))

    windowed = sliding_window_view(binary, window_shape)
    # Every structure pixel that is foreground must be covered by the window.
    fits = np.all(~structure | windowed, axis=(-2, -1))
    output = np.zeros_like(binary)
    output[
        window_shape[0] // 2 : output.shape[0] - (window_shape[0] - 1) // 2,
        window_shape[1] // 2 : output.shape[1] - (window_shape[1] - 1) // 2,
    ] = fits
    return _to_uint8(output)


def dilate(
    image: ArrayLike,
    structuring_element: ArrayLike | None = None,
) -> NDArray[np.uint8]:
    """Binary dilation: light a pixel when the structure hits foreground."""

    binary = _as_binary(image)
    structure = _as_structure(structuring_element)
    window_shape = structure.shape
    if window_shape[0] > binary.shape[0] or window_shape[1] > binary.shape[1]:
        return _to_uint8(np.ones_like(binary))

    windowed = sliding_window_view(binary, window_shape)
    hits = np.any(windowed & structure, axis=(-2, -1))
    output = np.zeros_like(binary)
    output[
        window_shape[0] // 2 : output.shape[0] - (window_shape[0] - 1) // 2,
        window_shape[1] // 2 : output.shape[1] - (window_shape[1] - 1) // 2,
    ] = hits
    return _to_uint8(output)


def opening(
    image: ArrayLike,
    structuring_element: ArrayLike | None = None,
) -> NDArray[np.uint8]:
    """Morphological opening: erosion followed by dilation.

    Opening removes isolated specks and thin protrusions while preserving the
    overall shape of larger structures.
    """

    return dilate(erode(image, structuring_element), structuring_element)


def closing(
    image: ArrayLike,
    structuring_element: ArrayLike | None = None,
) -> NDArray[np.uint8]:
    """Morphological closing: dilation followed by erosion.

    Closing bridges small gaps and fills narrow holes while preserving the
    overall size of structures.
    """

    return erode(dilate(image, structuring_element), structuring_element)