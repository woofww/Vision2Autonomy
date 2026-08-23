# Chapter 03: Harris corner detection

[English](README.md) | [简体中文](README.zh-CN.md)

The previous chapters answered where image intensity changes. This chapter asks a more useful question for localization and matching: **which local positions change strongly in two independent directions?** These positions are corners.

Corners support feature matching, tracking, panorama stitching, pose estimation, visual odometry, and SLAM. Harris combines convolution, gradients, and local matrix analysis into a classic, inspectable detector.

![Harris moving-window animation](../../docs/assets/chapter03_harris_window.gif)

## Learning objectives

After this chapter, you should be able to:

- explain how flat regions, edges, and corners respond to window translation;
- derive the second-moment matrix from image gradients;
- classify local structure using its two eigenvalues;
- interpret `R = det(M) - k trace(M)²`;
- select separated corners with a relative threshold and local maxima;
- analyze sensitivity to scale, noise, threshold, and window size.

Prerequisites: Chapter 01 convolution and Gaussian smoothing; Chapter 02 Sobel gradients.

## 1. Why an edge is not enough

Move a small observation window slightly:

- in a **flat region**, every direction looks almost unchanged;
- on an **edge**, motion across the edge changes the window, but motion along it does not;
- at a **corner**, motion in most directions changes the window strongly.

Along a long edge, local appearance cannot determine where we are on that edge—an intuitive form of the aperture problem. A corner constrains both directions and is therefore easier to localize repeatedly.

The animation shows the observation window in red. Orange and blue bars represent the largest and smallest structure-tensor eigenvalues:

```text
flat:   lambda1 small, lambda2 small
edge:   lambda1 large, lambda2 small
corner: lambda1 large, lambda2 large
```

## 2. From window motion to the structure tensor

Measure the squared change after translating a window by `(u, v)`:

```text
E(u, v) = sum w(x, y) [I(x+u, y+v) - I(x, y)]²
```

For a small translation, first-order Taylor expansion gives:

```text
I(x+u, y+v) ≈ I(x, y) + Ix·u + Iy·v
E(u, v) ≈ [u v] M [u v]ᵀ
```

The local second-moment matrix is:

```text
M = [ G(Ix²)   G(IxIy) ]
    [ G(IxIy)  G(Iy²)  ]
```

`Ix` and `Iy` come from Sobel derivatives. `G` denotes Gaussian-weighted aggregation. The matrix summarizes directional gradient structure inside the window.

## 3. Eigenvalue interpretation

The symmetric matrix has real eigenvalues `lambda1 >= lambda2`:

| Region | Largest eigenvalue | Smallest eigenvalue | Meaning |
|---|---:|---:|---|
| Flat | Small | Small | Little change in every direction |
| Edge | Large | Small | One strong direction |
| Corner | Large | Large | Strong change in two directions |

![Harris processing stages](../../docs/assets/chapter03_harris_stages.png)

The largest-eigenvalue map lights up most edges. The smallest-eigenvalue map concentrates around corners. Their contrast is the central idea behind the detector.

## 4. Harris response

Harris combines the matrix invariants into one score:

```text
R = det(M) - k · trace(M)²

det(M)   = lambda1 · lambda2
trace(M) = lambda1 + lambda2
```

- `R` near zero suggests a flat region;
- a strongly negative `R` suggests an edge;
- a strongly positive `R` suggests a corner.

`k` commonly lies between `0.04` and `0.06`; this chapter uses `0.04`. The implementation still returns both eigenvalue maps—not because detection requires them, but because inspecting them makes the model understandable.

## 5. Thresholding and non-maximum suppression

One geometric corner creates a small area of positive response. Direct thresholding would return several coordinates around the same corner.

The selector therefore:

1. rejects responses below `threshold_rel × max(R)`;
2. retains local maxima in a `(2d+1) × (2d+1)` neighborhood;
3. sorts candidates from strongest to weakest;
4. greedily enforces `min_distance` between accepted points;
5. optionally stops at `max_corners`.

Coordinates use NumPy order `(row, column)`. Drawing APIs normally expect `(x, y) = (column, row)`.

## 6. Reproduce the experiment

```bash
python -m pip install -e ".[dev]"
python -m vision2autonomy.examples.reference_images
```

This generates the stage figure, final overlay, and animated moving-window explanation.

![Final Harris corners](../../docs/assets/chapter03_harris_corners.png)

## 7. Python API

```python
import numpy as np
from PIL import Image

from vision2autonomy.features.harris import detect_harris_corners

image = np.asarray(Image.open("input.png").convert("L"))
result = detect_harris_corners(
    image,
    k=0.04,
    window_size=5,
    sigma=1.0,
    threshold_rel=0.03,
    min_distance=12,
    max_corners=100,
)

print(result.corners)  # (row, column)
```

`HarrisResult` contains the response map, both eigenvalue maps, and selected coordinates.

## 8. Parameters

| Parameter | Example | Role | Typical effect when increased |
|---|---:|---|---|
| `k` | `0.04` | Edge penalty | Stricter corner/edge separation |
| `window_size` | `5` | Tensor aggregation window | More stable but coarser localization |
| `sigma` | `1.0` | Gaussian weighting | Smoother local statistics |
| `threshold_rel` | `0.03` | Relative response threshold | Fewer, stronger corners |
| `min_distance` | `12` | Minimum separation | Sparser output |
| `max_corners` | `16` | Output limit | Keeps only stronger candidates |

## 9. Result interpretation

- The square produces strong positive responses at its four corners; long edges are mostly negative.
- Segment endpoints are corners because intensity changes in two directions there.
- Triangle vertices survive while most of each side does not.
- The largest eigenvalue responds to nearly every edge, while the smallest focuses on corners and endpoints.
- The number of red rings depends on threshold, separation, and output limit; it is not an absolute ground-truth count.

## 10. Limitations and next steps

- Harris is rotation-stable but not inherently scale-invariant.
- It detects locations but does not describe them for matching.
- Contrast changes alter gradient magnitude and response scale.
- Pixel-level maxima can be refined to subpixel coordinates.
- Repetitive texture creates ambiguous corners that detection alone cannot resolve.

The next natural topic is local description and feature matching.

## 11. Verification

```bash
pytest tests/test_harris.py tests/test_examples.py tests/test_documentation.py
```

Tests cover zero response on constant images, localization of four known square corners, edge/corner eigenvalue behavior, ordered distance-constrained selection, invalid input, animated GIF frames, reproducible assets, and bilingual links.

The implementation lives in `src/vision2autonomy/features/harris.py`.
## Autonomous-driving role: repeatable landmarks for localization

![Computer-vision algorithms mapped to a driving scene](../../docs/assets/autonomous_driving_cv_map.png)

Amber points mark sign, window, pole, and vehicle corners. Their appearance changes in two directions, making them easier to relocate across frames than plain edges. Visual odometry tracks stable corners to estimate ego motion, while sky and uniform asphalt provide almost none.

**Example:** building corners are usually static landmarks; corners on the lead vehicle move independently and can bias ego-motion unless RANSAC or motion segmentation rejects them.

