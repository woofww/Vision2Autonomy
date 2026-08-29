# Chapter 08: Camera Calibration and Lens Distortion

[English](README.md) | [简体中文](README.zh-CN.md)

> A camera maps the 3-D world onto 2-D pixels, but cheap lenses bend that map. Calibration recovers the exact mapping so that pixels become metric measurements.

## Learning objectives

- Model pinhole projection and the Brown-Conrady radial-tangential distortion;
- Explain why a few views of a flat checkerboard constrain the intrinsic matrix;
- Implement Zhang's closed-form estimate of `K` from planar homographies;
- Refine intrinsics, distortion, and per-view poses by minimizing reprojection error;
- Undistort a real-style image and quantify the remaining error;
- Connect calibration to distance estimation, lane geometry, and sensor fusion in driving.

## 1. Why calibrate?

Every geometric chapter so far assumed a camera matrix. In practice you do not know it: focal length changes with the lens and the focus ring, the principal point is not exactly at the image center, and wide-angle lenses bend straight lines. If a system reports "the front car is 8.4 m away" it is implicitly trusting calibrated intrinsics — a 2 % focal-length error becomes a 2 % distance error, which is more than a lane width at highway speeds.

Calibration is the step that turns pixels into meters:

- **Stereo and triangulation** need `K` to convert disparity into depth;
- **Lane geometry** needs undistorted images so straight lane lines stay straight and curvature is measured correctly;
- **Fusion** with radar, LiDAR, or HD maps needs a common metric coordinate system;
- **Visual odometry** needs `K` to relate pixel motion to camera motion.

## 2. From world to pixel: pinhole plus distortion

The ideal pinhole model maps a world point through a 3×4 projection, but for one camera it is convenient to split the mapping in two steps:

```math
\begin{bmatrix} u \\ v \\ 1 \end{bmatrix}
\sim
K \begin{bmatrix} R & t \end{bmatrix}
\begin{bmatrix} X \\ Y \\ Z \\ 1 \end{bmatrix},
\qquad
K = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}
```

`K` holds the focal lengths `f_x`, `f_y` and the principal point `(c_x, c_y)`; `R`, `t` place the camera in the world. Real lenses deviate from this line, most strongly at the image border. The Brown-Conrady model describes the deviation in *normalized* coordinates `(x, y)` (pixel coordinates divided through `K`):

```math
r^2 = x^2 + y^2
```

```math
x_d = x\,(1 + k_1 r^2 + k_2 r^4) + 2 p_1 x y + p_2 (r^2 + 2 x^2)
```

```math
y_d = y\,(1 + k_1 r^2 + k_2 r^4) + p_1 (r^2 + 2 y^2) + 2 p_2 x y
```

`k_1, k_2` are radial coefficients: a negative `k_1` gives barrel distortion (lines bow outward), a positive one gives pincushion. `p_1, p_2` model tangential distortion, typically caused by the lens not being perfectly parallel to the sensor.

```python
from vision2autonomy.geometry import distort_points, undistort_points

normalized = np.array([[0.5, 0.3], [-0.4, 0.6]])
distorted = distort_points(normalized, k1=-0.22, k2=0.05, p1=0.004, p2=-0.003)
ideal = undistort_points(distorted, k1=-0.22, k2=0.05, p1=0.004, p2=-0.003)
```

![Six synthetic calibration views with different poses](../../docs/assets/chapter08_calibration_targets.png)

The synthetic target above is rendered by the inverse pipeline used throughout the chapter: each pixel is undistorted, unprojected onto the checkerboard plane, and sampled. This is exactly how a real lens would image the pattern.

## 3. Zhang's method: planar homographies give you K

A full calibration target with known 3-D corners is expensive; a printed checkerboard is not. Zhang's insight is that a flat pattern is enough, provided you photograph it from several viewpoints.

For a plane `Z = 0` the projection collapses to a homography `H`:

```math
s \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = K \begin{bmatrix} r_1 & r_2 & t \end{bmatrix} \begin{bmatrix} X \\ Y \\ 1 \end{bmatrix} = H \begin{bmatrix} X \\ Y \\ 1 \end{bmatrix}
```

Each view gives one `3×3` `H` (estimated here with the normalized DLT from Chapter 04). Because `r_1` and `r_2` are orthonormal, every homography imposes two quadratic constraints on the symmetric matrix `B = K^{-T} K^{-1}`. The constraints are linear in the six independent entries of `B`, so stacking them over at least three views yields a linear system whose null-space vector is `B`. Extracting `K` from `B` is a closed-form exercise:

```python
from vision2autonomy.geometry import estimate_homography_dlt, estimate_intrinsics, estimate_view_pose

homographies = [estimate_homography_dlt(object_points[:, :2], view) for view in views]
intrinsics = estimate_intrinsics(homographies)
rotation, translation = estimate_view_pose(intrinsics, homographies[0])
```

The same homographies also recover per-view rotations and translations, which become the initialization for the nonlinear step.

## 4. Nonlinear refinement: reprojection error

The linear estimate ignores distortion and treats each view independently. The final step minimizes the reprojection error over all parameters at once — intrinsics `(f_x, f_y, c_x, c_y)`, distortion `(k_1, k_2, p_1, p_2)`, and every view's rotation and translation:

```math
\min_{K, k, \{R_i, t_i\}} \sum_i \sum_j \| \operatorname{proj}(X_j; K, k, R_i, t_i) - x_{ij} \|^2
```

This chapter implements Levenberg-Marquardt from scratch: a damped Gauss-Newton loop with a finite-difference Jacobian. The damping term smoothly interpolates between gradient descent (far from the solution) and Gauss-Newton (close to it), which is exactly what makes the method robust with a poor linear initialization.

```python
from vision2autonomy.geometry import calibrate_camera, reprojection_errors

result = calibrate_camera(views, object_points, iterations=40)
print(result.intrinsics)     # recovered K
print(result.distortion)     # recovered (k1, k2, p1, p2)
print(result.rmse)           # overall reprojection RMSE in pixels
```

![Distorted view with detected corners beside the undistorted result](../../docs/assets/chapter08_distortion_correction.png)

![Reprojection error vectors and per-view RMSE after refinement](../../docs/assets/chapter08_reprojection_errors.png)

The teaching scene adds 0.35 px of corner noise, like a real corner detector would produce. After refinement the residual vectors are magnified ×40 in the figure; per-view RMSE stays well below 0.5 px, and the recovered focal length is within a few pixels of the truth.

## 5. Reproduce

```bash
python -m pip install -e ".[dev]"
python -m vision2autonomy.examples.reference_images
pytest tests/test_calibration.py -q
```

The animation below shows a distorted view being progressively undistorted with the estimated model — watch the grid straighten from the borders inward:

![Progressive undistortion of a barrel-distorted checkerboard](../../docs/assets/chapter08_undistort.gif)

## 6. Conventions and common failures

| Quantity | Convention | Frequent mistake |
|---|---|---|
| pixel coordinates | `(x, y)` = `(column, row)`, origin top-left | swapping axes or using `(row, column)` |
| normalized coordinates | pixel coordinates divided through `K` | confusing them with raw pixels |
| distortion sign | `k_1 < 0` barrel, `k_1 > 0` pincushion | sign flip makes straight lines bow the wrong way |
| undistortion | source pixel is found with the *forward* model | applying the inverse twice and compounding the error |
| homography | maps plane coordinates `(X, Y)` to pixels | feeding 3-D points with `Z ≠ 0` into the DLT |
| views | at least 3, with varied pose | photographing the board from nearly the same angle |
| refinement | one joint least-squares problem | optimizing each view separately and ignoring distortion |
| RMSE | mean over all corners of all views, in pixels | reporting a single view or a maximum instead |

## 7. Autonomous-driving connection

Calibration sits under every metric claim the car makes:

- **Distance and speed:** a mis-calibrated focal length scales every stereo and mono depth estimate, biasing braking and take-over decisions.
- **Lane geometry:** undistortion keeps lane boundaries straight so curvature and lateral offset are measured from the correct geometry.
- **Sensor fusion:** intrinsics and extrinsics put camera detections, radar points, and LiDAR clouds into one coordinate frame for association and occupancy.
- **Online safety:** production systems re-estimate or monitor calibration drift (vibration, thermal expansion, a bumped camera) because even 1 % of depth bias matters at highway speed.

## 8. Check your understanding

1. Why does a flat checkerboard suffice for calibration instead of a 3-D target?
2. What do `k_1` and `p_1` physically describe, and where is each strongest in the image?
3. Which two orthonormality constraints does Zhang's method extract from each homography?
4. Why is a single view not enough to recover `K`, even with distortion-free data?
5. If the undistorted image looks even more bent, which of the two mapping directions was most likely implemented backwards?