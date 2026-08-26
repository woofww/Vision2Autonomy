# Chapter 06: Hough Transforms and Binary Morphology

[English](README.md) | [简体中文](README.zh-CN.md)

> Edges tell us *where* intensity changes. The Hough transform tells us *which geometric shape* those changes form; morphology cleans the binary maps that follow.

## Learning objectives

- Represent a line with the normal form `rho = x*cos(theta) + y*sin(theta)`;
- Build a Hough accumulator by letting every edge pixel vote;
- Extract strong, separated peaks and convert them back to drawable segments;
- Understand how quantization and neighborhood suppression affect detection;
- Implement binary erosion and dilation with a structuring element;
- Compose opening and closing, and know when each is useful.

## 1. From points back to lines: voting

A line in the plane can be written with two parameters `(rho, theta)`:

```math
\rho = x\cos\theta + y\sin\theta
```

`theta` is the angle of the line's normal, and `rho` is the signed distance from the image origin to the line. Every line through a given pixel `(x, y)` is therefore a sine curve in `(rho, theta)` space: fix the pixel, vary `theta`, compute `rho`.

Now reverse the direction. For a set of edge pixels belonging to the same straight line, all their sine curves intersect at the *one* `(rho, theta)` pair describing that line. Instead of searching the image for lines, we count votes in parameter space and keep the strongest intersections.

## 2. The accumulator

The continuous `(rho, theta)` space is quantized into a grid:

- `rho` ranges over `[-sqrt(H^2+W^2), +sqrt(H^2+W^2)]` with resolution `rho_resolution`;
- `theta` ranges over `[0, pi)` with resolution `theta_resolution`.

Each edge pixel increments every accumulator cell its sine curve passes through. A cell with a large count means many edge pixels agree on the same line.

![Road scene, Canny edges, and detected Hough line segments](../../docs/assets/chapter06_hough_lines.png)

The synthetic road scene is processed with Canny, then `hough_lines` returns the strongest peaks. The two lane markings are detected as pairs of lines, one per side of each thick stripe. The horizontal road boundary near the horizon is detected as well.

Peak extraction applies two rules:

1. **Threshold:** a peak must reach `threshold_rel * max_vote` to be accepted;
2. **Suppression:** after accepting a peak, a neighborhood around it is zeroed so a thick edge does not produce many near-duplicate hypotheses.

Smaller `theta_resolution` gives finer angular precision but more cells and more votes per pixel; larger values run faster and are more robust but can merge nearby orientations.

## 3. From parameters to segments

A `(rho, theta)` pair describes an infinite line. To draw it, project the supporting edge pixels onto the line direction `d=(-sin theta, cos theta)` and keep the minimum and maximum projections. The result is a concrete segment with explicit endpoints `(x1, y1, x2, y2)`:

```python
from vision2autonomy.edges import hough_lines

result = hough_lines(edge_map, num_peaks=8, threshold_rel=0.35, min_support=30)
for rho, theta in result.lines:
    print(rho, theta)
for x1, y1, x2, y2 in result.segments:
    print(x1, y1, x2, y2)
```

`min_support` rejects hypotheses supported by only a few pixels, which removes most noise. The remaining failures are informative: a thick stripe produces two parallel line hypotheses, one for each contour. Thinning the edge map first, or merging lines whose `(rho, theta)` are close, gives a single clean hypothesis.

## 4. Binary morphology: erosion and dilation

Morphology operates on binary images using a small binary pattern called a **structuring element**. With the default 3×3 cross, a pixel is compared with its four orthogonal neighbors.

```math
\text{erode}(I)(p) = \min_{s \in S} I(p + s)
\qquad
\text{dilate}(I)(p) = \max_{s \in S} I(p + s)
```

- **Erosion** keeps a pixel only when the whole structuring element fits inside the foreground. It removes one-pixel protrusions, shrinks shapes, and separates barely touching objects.
- **Dilation** lights a pixel when the structuring element touches any foreground pixel. It expands shapes, connects nearby regions, and fills small holes up to the element size.

![Binary morphology stages on a noisy mask](../../docs/assets/chapter06_morphology.png)

The input mask contains isolated specks, a hole, and two bars separated by a one-pixel gap. Erosion shrinks every structure and removes the specks; dilation grows everything and shrinks the hole. Neither primitive alone is a "cleanup" operation: erosion also damages valid structures, and dilation also amplifies noise.

## 5. Opening and closing

Composing the primitives creates the two standard cleanup operations:

```math
\text{opening} = \text{dilate}(\text{erode}(I))
\qquad
\text{closing} = \text{erode}(\text{dilate}(I))
```

- **Opening** first removes small foreground structures (specks, thin protrusions), then restores the remaining shapes to nearly their original size.
- **Closing** first fills small background gaps and holes, then restores the shapes to nearly their original size.

The choice of structuring element matters. The 3×3 square used in the reference figure fills a one-pixel gap completely and restores square corners; the 3×3 cross cannot restore diagonal corners, which is a useful reminder that morphology results are only as meaningful as the chosen structure.

```python
from vision2autonomy.image.morphology import closing, erode, dilate, opening

square = np.ones((3, 3), dtype=bool)
cleaned = closing(opening(mask, square), square)
```

## 6. Reproduce

```bash
python -m pip install -e ".[dev]"
python -m vision2autonomy.examples.reference_images
pytest tests/test_hough.py tests/test_morphology.py -q
```

## 7. Conventions and common failures

| Quantity | Convention | Frequent mistake |
|---|---|---|
| image coordinates | `(x, y) = (column, row)` | using NumPy `(row, column)` in `rho` voting |
| `theta` | radians in `[0, pi)` | forgetting that `(rho, theta)` and `(-rho, theta+pi)` describe the same line |
| `rho` | signed distance in pixels | using unsigned distance for lines on both sides of the origin |
| accumulator | `[rho_index, theta_index]` | plotting the transpose and misreading axes |
| structuring element | foreground = 1 | using a kernel larger than the details you want to preserve |
| binary input | nonzero = foreground | expecting erosion to preserve thin 1-pixel lines |

## 8. Autonomous-driving connection

Lane detection in classic pipelines follows exactly this chapter's loop: grayscale the camera image, detect edges, vote in Hough space, and keep strong lines. A lane-keeping controller then uses the detected line positions to estimate the vehicle's lateral offset and heading.

Morphology appears everywhere a binary mask is produced:

- **Drivable-area segmentation:** opening removes specks caused by shadows and sensor noise; closing fills small holes inside the road region.
- **Lane masks:** broken dashed lines produce many short segments; morphology joins them into continuous lanes before fitting a curve.
- **Object masks:** closing merges fragmented parts of a vehicle or pedestrian, opening removes isolated false positives.
- **Distance maps:** after thresholding disparity or depth, morphology cleans the mask before computing occupancy.

Failure cases are equally instructive: dashed lane lines require connecting segments across gaps, thick lines create double Hough responses, and an oversized structuring element deletes real details such as narrow lanes or small pedestrians.

## 9. Check your understanding

1. Why does a single edge pixel vote for many `(rho, theta)` cells?
2. What happens to the accumulator if `theta_resolution` is halved?
3. Why are thick stripes detected as two parallel lines?
4. Why does erosion followed by dilation not simply restore the input?
5. When would you prefer a cross structuring element over a square one?