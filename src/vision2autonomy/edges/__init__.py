"""Classical edge-detection algorithms."""

from vision2autonomy.edges.canny import (
    canny,
    double_threshold,
    hysteresis,
    non_maximum_suppression,
    sobel_gradients,
)

__all__ = [
    "canny",
    "double_threshold",
    "hysteresis",
    "non_maximum_suppression",
    "sobel_gradients",
]

