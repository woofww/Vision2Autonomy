"""Camera and multi-view geometry primitives."""

from vision2autonomy.geometry.calibration import (
    CalibrationResult,
    calibrate_camera,
    distort_points,
    estimate_homography_dlt,
    estimate_intrinsics,
    estimate_view_pose,
    planar_object_points,
    project_points,
    reprojection_errors,
    undistort_image,
    undistort_points,
)
from vision2autonomy.geometry.multiview import (
    camera_matrix,
    epipolar_lines,
    estimate_fundamental,
    project_points_3d,
    sampson_errors,
    triangulate_points,
)

__all__ = [
    "CalibrationResult",
    "calibrate_camera",
    "camera_matrix",
    "distort_points",
    "epipolar_lines",
    "estimate_fundamental",
    "estimate_homography_dlt",
    "estimate_intrinsics",
    "estimate_view_pose",
    "planar_object_points",
    "project_points",
    "project_points_3d",
    "reprojection_errors",
    "sampson_errors",
    "triangulate_points",
    "undistort_image",
    "undistort_points",
]