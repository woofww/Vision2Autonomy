# Chapter 02: Edge detection

This chapter builds a complete Canny edge detector without calling
`cv2.Canny`. Canny is useful here not merely as a feature: it connects image
smoothing, discrete derivatives, geometry, classification, and graph traversal
in one pipeline whose intermediate results can all be inspected.

## Learning objectives

You should be able to:

- calculate horizontal and vertical Sobel derivatives;
- convert gradient components into magnitude and direction;
- explain why thresholding the raw gradient produces thick edges;
- thin responses using non-maximum suppression (NMS);
- distinguish strong and weak edge candidates;
- connect weak candidates to strong edges with hysteresis;
- diagnose a Canny result by examining its intermediate images.

## Pipeline

### 1. Gaussian smoothing

Noise produces large local derivatives and therefore false edges. The input is
first convolved with a normalized Gaussian kernel. The example uses a 5 × 5
kernel with `sigma=1.4`.

### 2. Sobel gradients

Horizontal and vertical derivatives are estimated with 3 × 3 Sobel kernels:

```text
Gx = [-1  0  1]    Gy = [-1 -2 -1]
     [-2  0  2]         [ 0  0  0]
     [-1  0  1]         [ 1  2  1]
```

Magnitude and orientation are calculated as:

```text
magnitude = sqrt(Gx² + Gy²)
direction = atan2(Gy, Gx)
```

Direction is normalized to `[0, 180)` because an edge normal pointing at 0° is
equivalent to one pointing at 180°.

### 3. Non-maximum suppression

The gradient image contains ridges several pixels wide. NMS quantizes each
orientation into one of four sectors—0°, 45°, 90°, or 135°—and retains a pixel
only when its magnitude is at least as large as the two neighbours along the
gradient direction. This converts ridges into approximately one-pixel edges.

### 4. Double threshold

Two thresholds classify each remaining response:

```text
magnitude >= high                 strong edge
low <= magnitude < high           weak candidate
magnitude < low                    rejected
```

A single threshold cannot express the difference between a clearly valid edge
and a faint continuation of that edge.

### 5. Hysteresis

All strong pixels enter a queue. The algorithm performs an 8-neighbour graph
search and accepts every weak pixel reachable from a strong pixel. Isolated weak
responses are discarded. The output convention is `255` for an edge and `0`
for background.

## Reproduce the reference result

From the repository root:

```bash
python -m pip install -e ".[dev]"
python -m examples.generate_reference_images
```

The script uses the deterministic synthetic scene documented in Chapter 01 and
saves every display panel under `docs/assets/`.

![Canny intermediate stages](../../docs/assets/chapter02_canny_stages.png)

The final binary edge map is also available separately:

![Final Canny edge map](../../docs/assets/chapter02_canny_edges.png)

### Example parameters

| Parameter | Value | Effect |
|---|---:|---|
| `gaussian_size` | `5` | Size of the smoothing neighbourhood |
| `sigma` | `1.4` | Strength of Gaussian smoothing |
| `low_threshold` | `35` | Lowest response eligible for connection |
| `high_threshold` | `90` | Response accepted as a strong seed |

These thresholds operate on Sobel magnitudes, not normalized display values.
Raising both thresholds produces fewer edges; lowering them admits more texture
and noise. Increasing `sigma` removes fine structure before thresholding.

## Python API

```python
import numpy as np
from PIL import Image

from vision2autonomy.edges.canny import CannyResult, canny

image = np.asarray(Image.open("input.png").convert("L"))
result = canny(
    image,
    low_threshold=35,
    high_threshold=90,
    return_intermediates=True,
)
assert isinstance(result, CannyResult)
Image.fromarray(result.edges).save("edges.png")
```

For a direct command-line run:

```bash
v2a-canny input.png edges.png --low 35 --high 90 --sigma 1.4 --kernel-size 5
```

## Reading the stage image

- Smoothing removes most random speckle while preserving the major shapes.
- Gradient magnitude is bright on both sides of thin high-contrast structures.
- Gradient direction is visualized as intensity solely for inspection; it is
  angular data, not an edge-strength image.
- NMS removes broad gradient responses and retains narrow local maxima.
- Hysteresis preserves connected weak segments while rejecting isolated noise.

## Known limitations and next improvements

- The current NMS uses four direction sectors. Interpolating neighbouring
  magnitudes along the exact angle can improve localization.
- Absolute thresholds depend on image intensity and contrast. Relative or
  percentile-based thresholds can make defaults more adaptive.
- NumPy prioritizes clarity over real-time performance. Later benchmarks will
  compare it with optimized implementations.
- Edge quality should eventually be measured on annotated datasets rather than
  judged only by appearance.

## Verification

```bash
pytest tests/test_canny.py tests/test_cli.py tests/test_examples.py
```

Tests cover constant images, a known vertical step edge, NMS behavior,
8-connected hysteresis, invalid parameters, CLI image output, and deterministic
regeneration of the reference figures.
