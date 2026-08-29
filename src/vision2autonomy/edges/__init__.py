"""Classical edge-detection algorithms."""

from vision2autonomy.edges.canny import (
    canny,
    double_threshold,
    hysteresis,
    non_maximum_suppression,
    sobel_gradients,
)
from vision2autonomy.edges.hough import (
    HoughLineResult,
    hough_accumulator,
    hough_lines,
    hough_peaks,
)

__all__ = [
    "HoughLineResult",
    "canny",
    "double_threshold",
    "hough_accumulator",
    "hough_lines",
    "hough_peaks",
    "hysteresis",
    "non_maximum_suppression",
    "sobel_gradients",
]