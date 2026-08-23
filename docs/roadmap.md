# Vision2Autonomy roadmap

[English](roadmap.md) | [简体中文](roadmap.zh-CN.md)

The project grows in small, tested milestones. Each chapter contains an
explanation, a from-scratch implementation, experiments, and automated tests.

## Phase 1 — Image foundations

- Image arrays, intensity ranges, and color spaces
- Convolution and padding
- Gaussian filtering
- Sobel gradients
- Canny edge detection

## Phase 2 — Classical vision

- Harris corners (complete)
- Local patch description, nearest-neighbor matching, and ratio tests (complete)
- Normalized DLT and RANSAC homography (complete)
- Hough transforms and morphology
- Optical flow and tracking

## Phase 3 — Camera geometry

- Camera calibration and distortion
- Homographies and perspective transforms
- Pinhole projection and coordinate transforms (complete)
- Epipolar geometry and the fundamental matrix (complete)
- Stereo triangulation and depth sensitivity (complete)
- Visual odometry

## Phase 4 — Deep vision

- Neural-network foundations
- Classification, detection, and segmentation
- Monocular depth and multi-task learning

## Phase 5 — Driving perception

- Lane and drivable-area perception
- Vehicle, cyclist, and pedestrian detection
- Multi-object tracking and distance estimation
- A real-time, camera-based perception capstone
