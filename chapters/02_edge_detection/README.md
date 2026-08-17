# Chapter 02: Edge detection

The first milestone implements the complete Canny pipeline without calling
OpenCV's Canny function:

1. Gaussian smoothing
2. Sobel gradients and orientation
3. Non-maximum suppression
4. Double thresholding
5. Edge tracking by hysteresis

Intermediate arrays are available through `return_intermediates=True`, making
each stage inspectable and independently testable.

