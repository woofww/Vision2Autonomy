# Chapter 10: Visual Odometry, Trajectories, and Drift

[English](README.md) | [简体中文](README.zh-CN.md)

> PnP estimates one camera pose. Visual odometry connects consecutive poses
> into a vehicle trajectory. The hard part is not estimating one motion; it is
> preventing many tiny errors from quietly becoming a large drift.

![Visual-odometry trajectory growing frame by frame](../../docs/assets/chapter10_vo_trajectory.gif)

## Learning objectives

- connect matching, triangulation, tracking, and PnP into a runnable pipeline;
- distinguish world-to-camera, camera-to-world, and relative transforms;
- compose rigid transforms in the correct order;
- separate random noise, systematic bias, local error, and accumulated drift;
- understand metric stereo/RGB-D scale and monocular scale ambiguity;
- evaluate trajectories with alignment, ATE, and relative translation error;
- place VO correctly inside an autonomous-driving localization stack.

Prerequisites: Chapters 04, 05, 07, 08, and 09.

## 1. What is visual odometry?

A mechanical odometer estimates motion from wheel rotation. Visual odometry
estimates camera motion from image changes:

```text
images -> cross-frame associations -> camera poses -> trajectory
```

Odometry is local and sequential. It can be smooth and accurate over short
intervals, but without global constraints its error grows with distance.

## 2. The pipeline implemented here

Common formulations include:

| Formulation | Depth source | Motion solver | Scale |
|---|---|---|---|
| Monocular 2-D/2-D | Epipolar geometry | Essential matrix | Up to scale |
| Stereo 3-D/2-D | Triangulation | PnP | Metric |
| RGB-D 3-D/2-D | Depth sensor | PnP | Metric |
| 3-D/3-D | Depth in both frames | Rigid registration | Metric |

Chapter 10 follows the path already built in this course:

```text
stereo triangulation
        -> landmarks anchored in frame 0
        -> matching or optical-flow tracking
        -> robust PnP in each frame
        -> metric camera centers
```

## 3. Coordinate conventions

Let `T_cw = [R_cw | t_cw]` map world points into camera coordinates:

```text
X_c = R_cw X_w + t_cw
```

Then `T_wc = T_cw⁻¹`, and the camera center is

```text
C_w = -R_cwᵀ t_cw
```

If `T_10` maps camera 0 to camera 1 and `T_21` maps camera 1 to camera 2:

```text
T_20 = T_21 T_10
```

Translation vectors cannot simply be added because each one may be expressed
in a different rotated coordinate frame.

The module exposes `pose_matrix`, `invert_pose`, `compose_poses`,
`accumulate_relative_poses`, and `trajectory_from_poses` so these conventions
remain explicit.

## 4. End-to-end data flow

A teaching stereo VO loop is:

1. calibrate and undistort both cameras;
2. detect stable features;
3. stereo-match and triangulate initial landmarks;
4. track the landmarks into the next frame;
5. estimate the current pose with RANSAC PnP;
6. convert extrinsics into a camera center;
7. remove lost, dynamic, and high-error landmarks;
8. triangulate new landmarks;
9. repeat and record the trajectory.

`pnp_odometry` focuses on steps 5–6 and trajectory output. Missing observations
are represented by `NaN`.

## 5. Reading the result

![Ground-truth and estimated VO trajectories](../../docs/assets/chapter10_vo_trajectory.png)

Green is ground truth, orange is the estimate, gray segments show per-frame
position error, ATE summarizes the full sequence, and final drift measures the
last frame.

This short example repeatedly observes a fixed local landmark set. In a long
drive, old landmarks leave the field of view and noisy new landmarks replace
them, so the coordinate reference itself slowly moves.

## 6. How drift grows

A fixed yaw bias of only a fraction of a degree is hard to notice in one
frame, but every later translation follows a slightly wrong heading:

![Accumulation of a small yaw bias](../../docs/assets/chapter10_drift_comparison.png)

Random errors tend to jitter in different directions and can partly average
out. Calibration, timing, baseline, or heading bias acts repeatedly in the
same direction. Rotation error is especially damaging because it rotates
every future translation.

## 7. Monocular scale ambiguity

Two monocular views recover rotation and translation direction, not metric
translation magnitude. Scaling both the scene and motion leaves the images
unchanged.

Metric scale can come from stereo baseline, RGB-D, IMU and wheel speed, known
objects, camera height, or mapped landmarks. Monocular evaluation therefore
often permits one global scale alignment; metric stereo results normally
should not.

## 8. Absolute trajectory error

ATE RMSE is

```text
sqrt(mean(||p̂ᵢ - pᵢ||²))
```

Before comparison, trajectories may need one global coordinate alignment.
Chapter 10 implements Umeyama similarity alignment:

```text
p_aligned = s R p_est + t
```

Use `with_scale=False` for metric trajectories and `with_scale=True` for a
monocular trajectory with one unknown global scale. Never align or rescale
each local segment independently; that hides drift.

## 9. Relative error

ATE measures global position. Relative error compares displacement over a
fixed frame gap:

```text
Δp_est(i) = p_est(i+δ) - p_est(i)
Δp_ref(i) = p_ref(i+δ) - p_ref(i)
e_rel(i) = ||Δp_est(i) - Δp_ref(i)||
```

High short-gap error indicates unstable local estimation. Low short-gap error
combined with growing ATE means small consistent errors are accumulating.

## 10. Run it

```bash
python -m pip install -e ".[dev]"
python -m pytest tests/test_odometry.py -q
python -m vision2autonomy.examples.reference_images
```

```python
from vision2autonomy.geometry import (
    absolute_trajectory_error,
    align_trajectory,
    pnp_odometry,
    relative_translation_error,
)

result = pnp_odometry(
    landmarks_3d,
    observations_by_frame,
    intrinsics,
    threshold=3.5,
    max_iterations=600,
    seed=9,
)

ate = absolute_trajectory_error(
    result.centers,
    reference_centers,
    align=True,
    with_scale=False,
)
local_errors = relative_translation_error(
    result.centers,
    reference_centers,
    delta=1,
)
```

`OdometryResult` also reports per-frame inlier counts and median inlier
reprojection errors. These often degrade before position drift becomes
visually obvious.

## 11. Failure modes

| Situation | Failure | Mitigation |
|---|---|---|
| Textureless road or tunnel | Too few stable tracks | Better features, longer windows, IMU |
| Dynamic traffic dominates | Static-world assumption breaks | Semantic and motion filtering |
| Fast rotation or blur | Tracking fails | Short exposure, higher frame rate, IMU prediction |
| Near-pure rotation | Weak triangulation baseline | Delay initialization or use depth/IMU |
| Repeated patterns | False motion consensus | Mutual checks, RANSAC, local maps |
| Calibration or timing bias | Systematic drift | Online health checks and synchronization |
| Long drive without revisits | Error cannot reset | Loop closure and global graph optimization |

## 12. Autonomous-driving relevance

VO provides short-term motion through tunnels and urban canyons, ego-motion
for temporal perception and tracking, initialization for map localization,
sensor-failure redundancy, and a stable local frame for accumulated
perception.

A production localization stack normally combines:

```text
visual or lidar odometry
        + IMU
        + wheel speed
        + GNSS / map constraints
        + loop closure and graph optimization
```

## 13. Next step

The classical geometry path is now complete:

```text
pixels -> features -> matches -> 3-D -> pose -> trajectory
```

The next phase begins neural-network foundations. It will first implement a
small dense network, backpropagation, losses, and convolutional layers from
scratch, then use PyTorch as a checked reference rather than starting with a
large detector.

## 14. Check your understanding

1. Why is world-to-camera translation not the vehicle position?
2. Why is transform multiplication order important?
3. How do random noise and focal-length bias affect a trajectory differently?
4. Why does a small rotation bias create large long-range drift?
5. Why may monocular evaluation align scale while metric stereo should not?
6. What does high ATE but low short-gap relative error suggest?
