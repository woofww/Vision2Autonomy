"""Generate deterministic reference images used by Chapters 01 through 10.

Run after installing the project:

    python -m vision2autonomy.examples.reference_images
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from vision2autonomy.edges.canny import CannyResult, canny
from vision2autonomy.edges.hough import HoughLineResult, hough_lines
from vision2autonomy.image.morphology import closing, dilate, erode, opening
from vision2autonomy.motion.lucas_kanade import OpticalFlowResult, lucas_kanade_flow
from vision2autonomy.features.harris import HarrisResult, detect_harris_corners
from vision2autonomy.features.matching import (
    MatchResult,
    describe_patches,
    match_descriptors,
    ransac_homography,
)
from vision2autonomy.image.convolution import convolve2d
from vision2autonomy.image.filters import gaussian_blur
from vision2autonomy.geometry import (
    absolute_trajectory_error,
    calibrate_camera,
    camera_matrix,
    estimate_fundamental,
    planar_object_points,
    project_points,
    project_points_3d,
    pnp_odometry,
    ransac_pnp,
    triangulate_points,
    undistort_image,
    undistort_points,
)

DEFAULT_OUTPUT = Path("docs/assets")


def synthetic_scene(size: int = 256, seed: int = 7) -> np.ndarray:
    """Create a deterministic scene containing several edge types and noise."""

    if size < 64:
        raise ValueError("size must be at least 64 pixels")

    canvas = Image.new("L", (size, size), color=28)
    draw = ImageDraw.Draw(canvas)
    margin = size // 12
    draw.rectangle(
        (margin, margin, size // 2 - margin // 2, size // 2),
        fill=185,
        outline=245,
        width=max(2, size // 96),
    )
    draw.ellipse(
        (size // 2, margin, size - margin, size // 2),
        fill=105,
        outline=225,
        width=max(2, size // 96),
    )
    draw.polygon(
        ((size // 2, size // 2), (margin, size - margin), (size - margin, size - margin)),
        fill=70,
        outline=210,
    )
    draw.line(
        (margin, size * 3 // 4, size - margin, size // 2),
        fill=250,
        width=max(2, size // 64),
    )

    image = np.asarray(canvas, dtype=np.float64)
    noise = np.random.default_rng(seed).normal(0.0, 8.0, image.shape)
    return np.clip(image + noise, 0, 255).astype(np.uint8)


def to_uint8(array: np.ndarray, *, normalize: bool = False) -> np.ndarray:
    """Convert a numerical image to display-safe unsigned bytes."""

    values = np.asarray(array, dtype=np.float64)
    if normalize:
        minimum, maximum = values.min(), values.max()
        if maximum > minimum:
            values = (values - minimum) * (255.0 / (maximum - minimum))
        else:
            values = np.zeros_like(values)
    return np.clip(values, 0, 255).astype(np.uint8)


def labelled_grid(images: Mapping[str, np.ndarray], columns: int = 3) -> Image.Image:
    """Compose equally sized grayscale arrays into a labelled reference grid."""

    if not images:
        raise ValueError("images must not be empty")
    if columns < 1:
        raise ValueError("columns must be positive")

    converted = [(label, Image.fromarray(to_uint8(image)).convert("RGB")) for label, image in images.items()]
    width, height = converted[0][1].size
    if any(item.size != (width, height) for _, item in converted):
        raise ValueError("all grid images must have the same dimensions")

    label_height = 30
    rows = (len(converted) + columns - 1) // columns
    grid = Image.new("RGB", (columns * width, rows * (height + label_height)), "white")
    draw = ImageDraw.Draw(grid)
    font = ImageFont.load_default()
    for index, (label, image) in enumerate(converted):
        row, column = divmod(index, columns)
        x, y = column * width, row * (height + label_height)
        grid.paste(image, (x, y + label_height))
        draw.text((x + 8, y + 9), label, fill="black", font=font)
    return grid


def harris_scene(size: int = 256) -> np.ndarray:
    """Create a clean geometric scene with flat, edge, and corner regions."""

    canvas = Image.new("L", (size, size), color=22)
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((64, 64, 192, 192), fill=220)
    draw.polygon(((28, 220), (100, 136), (156, 220)), fill=130)
    draw.line((164, 36, 228, 100), fill=250, width=8)
    return np.asarray(canvas, dtype=np.uint8)


def signed_response_heatmap(response: np.ndarray) -> np.ndarray:
    """Map positive corner responses to warm colors and negative edges to blue."""

    values = np.asarray(response, dtype=np.float64)
    scale = float(np.max(np.abs(values), initial=0.0))
    normalized = values / scale if scale > 0.0 else np.zeros_like(values)
    positive = np.clip(normalized, 0.0, 1.0)
    negative = np.clip(-normalized, 0.0, 1.0)
    heatmap = np.zeros((*values.shape, 3), dtype=np.uint8)
    heatmap[..., 0] = np.uint8(255 * np.sqrt(positive))
    heatmap[..., 1] = np.uint8(210 * positive)
    heatmap[..., 2] = np.uint8(255 * np.sqrt(negative))
    return heatmap


def draw_corner_overlay(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Draw detected Harris corners as red rings on a grayscale image."""

    overlay = Image.fromarray(to_uint8(image)).convert("RGB")
    draw = ImageDraw.Draw(overlay)
    for row, column in corners:
        radius = 6
        draw.ellipse(
            (column - radius, row - radius, column + radius, row + radius),
            outline=(255, 45, 45),
            width=2,
        )
    return np.asarray(overlay)


def translated_image(image: np.ndarray, row_shift: int, column_shift: int) -> np.ndarray:
    """Translate an image without wraparound, filling uncovered pixels with zero."""

    output = np.zeros_like(image)
    source_rows = slice(max(0, -row_shift), min(image.shape[0], image.shape[0] - row_shift))
    source_columns = slice(max(0, -column_shift), min(image.shape[1], image.shape[1] - column_shift))
    target_rows = slice(max(0, row_shift), min(image.shape[0], image.shape[0] + row_shift))
    target_columns = slice(max(0, column_shift), min(image.shape[1], image.shape[1] + column_shift))
    output[target_rows, target_columns] = image[source_rows, source_columns]
    return output


def draw_matches(
    first: np.ndarray,
    second: np.ndarray,
    points_a: np.ndarray,
    points_b: np.ndarray,
    matches: MatchResult,
    inliers: np.ndarray | None = None,
) -> Image.Image:
    """Draw side-by-side correspondences, optionally marking RANSAC outliers red."""

    height, width = first.shape
    canvas = Image.new("RGB", (2 * width, height + 30), "white")
    canvas.paste(Image.fromarray(first).convert("RGB"), (0, 30))
    canvas.paste(Image.fromarray(second).convert("RGB"), (width, 30))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 9), "Image A", fill="black", font=ImageFont.load_default())
    draw.text((width + 8, 9), "Image B", fill="black", font=ImageFont.load_default())
    mask = np.ones(len(matches.pairs), dtype=bool) if inliers is None else np.asarray(inliers, dtype=bool)
    for index, (index_a, index_b) in enumerate(matches.pairs):
        row_a, column_a = points_a[index_a]
        row_b, column_b = points_b[index_b]
        color = (38, 174, 96) if mask[index] else (220, 65, 65)
        start = (int(round(column_a)), int(round(row_a)) + 30)
        end = (int(round(column_b)) + width, int(round(row_b)) + 30)
        draw.line((start, end), fill=color, width=2)
        draw.ellipse((start[0] - 3, start[1] - 3, start[0] + 3, start[1] + 3), outline=color, width=2)
        draw.ellipse((end[0] - 3, end[1] - 3, end[0] + 3, end[1] + 3), outline=color, width=2)
    return canvas


def multiview_reference_images() -> tuple[Image.Image, Image.Image]:
    """Render a deterministic rectified stereo pair and its triangulated X-Z view."""

    intrinsics = np.array([[260.0, 0.0, 160.0], [0.0, 260.0, 120.0], [0.0, 0.0, 1.0]])
    projection_a = camera_matrix(intrinsics, np.eye(3), np.zeros(3))
    projection_b = camera_matrix(intrinsics, np.eye(3), np.array([-0.7, 0.0, 0.0]))
    world = np.array([
        [-1.4, -0.8, 4.0], [-0.5, -0.6, 5.0], [0.5, -0.5, 6.0], [1.3, -0.4, 7.0],
        [-1.1, 0.1, 6.5], [-0.2, 0.2, 7.5], [0.7, 0.3, 8.5], [1.4, 0.4, 9.5],
        [-1.3, 0.8, 9.0], [-0.4, 0.9, 10.0], [0.5, 1.0, 11.0], [1.2, 1.1, 12.0],
    ])
    pixels_a = project_points_3d(world, projection_a)
    pixels_b = project_points_3d(world, projection_b)
    fundamental = estimate_fundamental(pixels_a, pixels_b)
    reconstructed = triangulate_points(projection_a, projection_b, pixels_a, pixels_b)

    stereo = Image.new("RGB", (640, 270), (247, 249, 248))
    draw = ImageDraw.Draw(stereo)
    font = ImageFont.load_default()
    draw.text((8, 8), "Left camera", fill=(24, 33, 31), font=font)
    draw.text((328, 8), "Right camera", fill=(24, 33, 31), font=font)
    colors = [(49, 95, 87), (188, 105, 55), (74, 116, 173), (151, 82, 125)]
    for index, (left, right) in enumerate(zip(pixels_a, pixels_b)):
        color = colors[index % len(colors)]
        y = int(round(left[1])) + 25
        draw.line((0, y, 640, y), fill=(216, 225, 221), width=1)
        for offset, point in ((0, left), (320, right)):
            x = int(round(point[0])) + offset
            py = int(round(point[1])) + 25
            draw.ellipse((x - 4, py - 4, x + 4, py + 4), fill=color, outline="white", width=1)
            draw.text((x + 6, py - 6), str(index + 1), fill=color, font=font)

    depth = Image.new("RGB", (560, 340), (247, 249, 248))
    depth_draw = ImageDraw.Draw(depth)
    depth_draw.text((12, 10), "Triangulated points: top view (X-Z)", fill=(24, 33, 31), font=font)
    origin_x, bottom_y = 280, 305
    depth_draw.line((40, bottom_y, 525, bottom_y), fill=(99, 112, 108), width=2)
    depth_draw.line((origin_x, 45, origin_x, bottom_y), fill=(99, 112, 108), width=2)
    depth_draw.text((520, 312), "X", fill=(99, 112, 108), font=font)
    depth_draw.text((286, 42), "Z", fill=(99, 112, 108), font=font)
    for index, (x_value, _, z_value) in enumerate(reconstructed):
        x = int(round(origin_x + x_value * 120))
        y = int(round(bottom_y - (z_value - 3.0) * 28))
        color = colors[index % len(colors)]
        radius = max(3, int(round(9 - z_value * 0.35)))
        depth_draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
        depth_draw.text((x + 6, y - 6), f"{z_value:.0f}m", fill=color, font=font)
    return stereo, depth


def road_scene(size: int = 256, seed: int = 3) -> np.ndarray:
    """Create a synthetic road image with perspective lane lines and noise."""

    canvas = Image.new("L", (size, size), color=205)
    draw = ImageDraw.Draw(canvas)
    horizon = int(size * 0.48)
    draw.polygon(
        ((int(size * 0.14), size - 1), (int(size * 0.39), horizon), (int(size * 0.61), horizon), (int(size * 0.86), size - 1)),
        fill=74,
    )
    draw.line((int(size * 0.24), size - 4, int(size * 0.47), horizon), fill=235, width=3)
    draw.line((int(size * 0.76), size - 4, int(size * 0.53), horizon), fill=235, width=3)
    image = np.asarray(canvas, dtype=np.float64)
    noise = np.random.default_rng(seed).normal(0.0, 5.0, image.shape)
    return np.clip(image + noise, 0, 255).astype(np.uint8)


def draw_hough_overlay(image: np.ndarray, result: HoughLineResult) -> np.ndarray:
    """Draw detected Hough line segments in green over a grayscale image."""

    overlay = Image.fromarray(to_uint8(image)).convert("RGB")
    draw = ImageDraw.Draw(overlay)
    for x1, y1, x2, y2 in result.segments:
        draw.line((x1, y1, x2, y2), fill=(38, 174, 96), width=2)
    return np.asarray(overlay)


def morphology_scene(size: int = 128) -> np.ndarray:
    """Create a binary mask with isolated specks, a hole, and a one-pixel gap."""

    canvas = np.zeros((size, size), dtype=np.uint8)
    canvas[30:90, 30:90] = 255
    canvas[56:64, 56:64] = 0  # hole
    canvas[16:24, 16:52] = 255  # upper bar
    canvas[25:34, 16:52] = 255  # lower bar, one-pixel gap at row 24
    for row, column in [(10, 12), (20, 100), (106, 14), (112, 112), (14, 64), (102, 58)]:
        canvas[row, column] = 255  # isolated specks
    return canvas


def save_harris_window_animation(
    image: np.ndarray,
    result: HarrisResult,
    path: Path,
) -> None:
    """Animate a local window moving through flat, edge, and corner regions."""

    key_positions = [(128, 128), (64, 128), (64, 64)]
    centers: list[tuple[int, int]] = []
    for start, end in zip(key_positions, key_positions[1:]):
        for fraction in np.linspace(0.0, 1.0, 10, endpoint=False):
            row = int(round(start[0] + fraction * (end[0] - start[0])))
            column = int(round(start[1] + fraction * (end[1] - start[1])))
            centers.append((row, column))
    centers.extend([key_positions[-1]] * 6)

    eigen_scale = float(result.lambda_max.max(initial=1.0))
    font = ImageFont.load_default()
    frames: list[Image.Image] = []
    for row, column in centers:
        frame = Image.new("RGB", (256, 330), "white")
        frame.paste(Image.fromarray(image).convert("RGB"), (0, 0))
        draw = ImageDraw.Draw(frame)
        radius = 14
        draw.rectangle(
            (column - radius, row - radius, column + radius, row + radius),
            outline=(255, 45, 45),
            width=3,
        )
        draw.ellipse((column - 3, row - 3, column + 3, row + 3), fill=(40, 200, 80))

        lambda_1 = float(result.lambda_max[row, column])
        lambda_2 = float(result.lambda_min[row, column])
        relative_1 = lambda_1 / eigen_scale
        relative_2 = lambda_2 / eigen_scale
        if relative_1 < 0.02:
            region = "FLAT: both changes are small"
        elif lambda_2 < 0.1 * lambda_1:
            region = "EDGE: one direction changes"
        else:
            region = "CORNER: both directions change"

        draw.text((10, 268), region, fill="black", font=font)
        draw.text((10, 290), "lambda max", fill="black", font=font)
        draw.rectangle((82, 290, 82 + int(150 * relative_1), 299), fill=(245, 150, 35))
        draw.text((10, 308), "lambda min", fill="black", font=font)
        draw.rectangle((82, 308, 82 + int(150 * relative_2), 317), fill=(45, 150, 240))
        frames.append(frame)

    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=140,
        loop=0,
        optimize=True,
        disposal=2,
    )


def optical_flow_frame(
    size: int,
    rect_dx: int,
    rect_dy: int,
    circle_dx: int,
    circle_dy: int,
) -> np.ndarray:
    """Render one synthetic frame with a textured rectangle and circle."""

    # Smooth 2-D textures keep the brightness-constancy linearization valid;
    # hard periodic edges bias the estimate, a useful property for readers to
    # discover by changing the patterns.
    # Long-wavelength patterns keep the single-scale Lucas-Kanade linearization
    # valid for the small displacements used here; short-period or hard-edge
    # textures bias the estimate, a useful property to discover.
    rows, columns = np.indices((size, size))
    background = (
        0.4 * columns
        + 0.25 * rows
        + 2.0 * np.sin(2.0 * np.pi * rows / 36.0)
        + 2.0 * np.sin(2.0 * np.pi * (rows + columns) / 28.0)
    )
    # Pattern content is attached to the object and must shift with it; a
    # pattern fixed to image coordinates would make interior pixels identical
    # across frames and leave only the occlusion boundary visible.
    rect_rows = rows - rect_dy
    rect_cols = columns - rect_dx
    rect_pattern = (
        205.0
        + 10.0 * np.sin(2.0 * np.pi * (rect_rows + rect_cols) / 36.0)
        + 6.0 * np.sin(2.0 * np.pi * (rect_rows + 2.0 * rect_cols) / 30.0)
    )
    circle_rows = rows - circle_dy
    circle_cols = columns - circle_dx
    circle_pattern = (
        32.0
        + 8.0 * np.sin(2.0 * np.pi * (circle_rows + circle_cols) / 30.0)
        + 4.0 * np.sin(2.0 * np.pi * (circle_rows - circle_cols) / 24.0)
    )
    frame = background.copy()
    rect_mask = (
        (columns >= 56 + rect_dx)
        & (columns < 104 + rect_dx)
        & (rows >= 70 + rect_dy)
        & (rows < 118 + rect_dy)
    )
    circle_mask = (rows - (60 + circle_dy)) ** 2 + (columns - (170 + circle_dx)) ** 2 <= 18 ** 2
    frame = np.where(rect_mask, rect_pattern, frame)
    frame = np.where(circle_mask, circle_pattern, frame)
    return np.clip(frame, 0, 255).astype(np.uint8)


def draw_arrow(
    draw: ImageDraw.ImageDraw,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    color: tuple[int, int, int],
    width: int = 2,
) -> None:
    """Draw a line with a small arrowhead on a PIL canvas."""

    draw.line((x1, y1, x2, y2), fill=color, width=width)
    dx, dy = x2 - x1, y2 - y1
    length = float(np.hypot(dx, dy))
    if length <= 1e-9:
        return
    unit_x, unit_y = dx / length, dy / length
    head = 7
    for angle_deg in (150.0, -150.0):
        angle = np.deg2rad(angle_deg)
        tip_x = x2 + head * (unit_x * np.cos(angle) - unit_y * np.sin(angle))
        tip_y = y2 + head * (unit_x * np.sin(angle) + unit_y * np.cos(angle))
        draw.line((x2, y2, tip_x, tip_y), fill=color, width=width)


def draw_flow_arrows(
    image: np.ndarray,
    result: OpticalFlowResult,
    step: int = 10,
    scale: float = 4.0,
) -> np.ndarray:
    """Overlay confident flow vectors on a frame, colored by confidence."""

    overlay = Image.fromarray(to_uint8(image)).convert("RGB")
    draw = ImageDraw.Draw(overlay)
    height, width = image.shape
    for row in range(step // 2, height, step):
        for column in range(step // 2, width, step):
            if not result.valid[row, column]:
                continue
            u, v = result.flow[row, column]
            if float(np.hypot(u, v)) < 0.4:
                continue
            color = (38, 174, 96) if result.confidence[row, column] > 20.0 else (232, 140, 50)
            draw_arrow(draw, column, row, column + scale * u, row + scale * v, color)
    return np.asarray(overlay)


def optical_flow_reference_image(size: int = 256) -> Image.Image:
    """Render frame A beside frame B with the recovered optical flow."""

    frame_a = optical_flow_frame(size, 0, 0, 0, 0)
    frame_b = optical_flow_frame(size, 1, 1, -1, 1)
    result = lucas_kanade_flow(frame_a, frame_b, window_size=15)
    with_flow = draw_flow_arrows(frame_b, result)
    canvas = Image.new("RGB", (2 * size, size + 30), "white")
    canvas.paste(Image.fromarray(frame_a).convert("RGB"), (0, 30))
    canvas.paste(Image.fromarray(with_flow), (size, 30))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((8, 9), "Frame A", fill="black", font=font)
    draw.text((size + 8, 9), "Frame B + Lucas-Kanade flow", fill="black", font=font)
    return canvas


def save_flow_tracking_animation(path: Path, size: int = 256, steps: int = 12) -> None:
    """Animate a moving rectangle and circle with flow arrows on each frame."""

    font = ImageFont.load_default()
    frames: list[Image.Image] = []
    previous = optical_flow_frame(size, 0, 0, 0, 0)
    for index in range(1, steps):
        rect_dx = int(round(18 * index / (steps - 1)))
        rect_dy = int(round(12 * index / (steps - 1)))
        circle_dx = -int(round(14 * index / (steps - 1)))
        circle_dy = int(round(8 * index / (steps - 1)))
        current = optical_flow_frame(size, rect_dx, rect_dy, circle_dx, circle_dy)
        result = lucas_kanade_flow(previous, current, window_size=15)
        with_flow = draw_flow_arrows(current, result)
        frame = Image.new("RGB", (size, size + 30), "white")
        frame.paste(Image.fromarray(with_flow), (0, 30))
        draw = ImageDraw.Draw(frame)
        draw.text((8, 9), f"Tracking frame {index:02d}", fill="black", font=font)
        frames.append(frame)
        previous = current

    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=130,
        loop=0,
        optimize=True,
        disposal=2,
    )


@dataclass
class CalibrationScene:
    """A synthetic planar checkerboard scene with known camera parameters."""

    intrinsics: np.ndarray
    distortion: np.ndarray
    square_size: float
    squares_rows: int
    squares_cols: int
    images: list[np.ndarray]
    object_points: np.ndarray
    image_points: list[np.ndarray]
    rotations: list[np.ndarray]
    translations: list[np.ndarray]


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    """Sample a deterministic random rotation matrix."""

    raw = rng.normal(size=(2, 3))
    first = raw[0] / np.linalg.norm(raw[0])
    second = raw[1] - np.dot(raw[1], first) * first
    second = second / np.linalg.norm(second)
    third = np.cross(first, second)
    return np.column_stack((first, second, third))


def render_checkerboard_view(
    intrinsics: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    distortion: Sequence[float],
    squares_rows: int = 6,
    squares_cols: int = 8,
    square_size: float = 0.032,
    image_size: tuple[int, int] = (300, 400),
    background: int = 190,
    margin: float = 0.02,
) -> np.ndarray:
    """Render the planar checkerboard ``Z = 0`` through a distorted camera.

    Rendering runs the inverse pipeline: every output pixel is undistorted,
    unprojected onto the target plane, and sampled from the checkerboard
    texture. This is the correct way to synthesize what a real lens sees.
    """

    height, width = image_size
    k = np.asarray(intrinsics, dtype=np.float64)
    r = np.asarray(rotation, dtype=np.float64)
    t = np.asarray(translation, dtype=np.float64).reshape(-1)
    coefficients = np.asarray(distortion, dtype=np.float64)
    inverse = np.linalg.inv(k)

    rows_index, columns_index = np.indices((height, width))
    pixels = np.column_stack((columns_index.ravel(), rows_index.ravel()))
    homogeneous = np.column_stack((pixels, np.ones(len(pixels))))
    normalized = undistort_points((inverse @ homogeneous.T).T[:, :2], *coefficients)
    # normalized is already K^-1 @ pixel, so the camera ray is [normalized, 1];
    # applying K^-1 a second time would collapse the whole image onto a point.
    rays = np.column_stack((normalized, np.ones(len(normalized))))
    rotated_translation = r.T @ t
    denominator = rays @ r[:, 2]
    # The ray hits the plane at s = (R^T t)_z / (R^T d)_z; the camera may look
    # up or down at the target, so the scale is valid only when it is positive
    # (the intersection lies in front of the camera).
    numerator = rotated_translation[2]
    visible = denominator * numerator > 0.0
    scale = np.where(visible, numerator / denominator, 0.0)
    camera_points = scale[:, None] * rays
    world_points = (camera_points - t) @ r
    x_world, y_world = world_points[:, 0], world_points[:, 1]
    inside = (
        (x_world >= -margin)
        & (x_world <= squares_cols * square_size + margin)
        & (y_world >= -margin)
        & (y_world <= squares_rows * square_size + margin)
    )
    cell_x = np.floor(x_world / square_size).astype(np.int64)
    cell_y = np.floor(y_world / square_size).astype(np.int64)
    white_cell = (cell_x + cell_y) % 2 == 0
    image = np.where(
        inside & white_cell, 245, np.where(inside, 28, background)
    ).astype(np.uint8)
    return image.reshape(height, width)


def synthetic_calibration_scene(
    num_views: int = 8,
    squares_rows: int = 6,
    squares_cols: int = 8,
    square_size: float = 0.032,
    seed: int = 7,
    image_size: tuple[int, int] = (300, 400),
    distortion: Sequence[float] = (-0.22, 0.05, 0.004, -0.003),
) -> CalibrationScene:
    """Create deterministic synthetic checkerboard views with known parameters."""

    intrinsics = np.array([[330.0, 0.0, 200.0], [0.0, 330.0, 150.0], [0.0, 0.0, 1.0]])
    coefficients = np.asarray(distortion, dtype=np.float64)
    rng = np.random.default_rng(seed)
    plane_x = squares_cols * square_size / 2.0
    plane_y = squares_rows * square_size / 2.0
    object_points = planar_object_points(
        squares_rows - 1, squares_cols - 1, square_size
    )
    width, height = image_size[1], image_size[0]
    images: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    rotations: list[np.ndarray] = []
    translations: list[np.ndarray] = []
    while len(images) < num_views:
        rotation = _random_rotation(rng)
        depth = rng.uniform(0.45, 0.8)
        offset = rng.normal(0.0, 0.03, size=2)
        translation = np.array([0.0, 0.0, depth]) - rotation @ np.array(
            [plane_x + offset[0], plane_y + offset[1], 0.0]
        )
        pixels = project_points(
            object_points, intrinsics, rotation, translation, coefficients
        )
        inside = (
            (pixels[:, 0] > 5.0)
            & (pixels[:, 0] < width - 5.0)
            & (pixels[:, 1] > 5.0)
            & (pixels[:, 1] < height - 5.0)
        )
        if not inside.all():
            continue
        image = render_checkerboard_view(
            intrinsics,
            rotation,
            translation,
            coefficients,
            squares_rows,
            squares_cols,
            square_size,
            image_size,
        )
        images.append(image)
        image_points.append(pixels)
        rotations.append(rotation)
        translations.append(translation)
    return CalibrationScene(
        intrinsics=intrinsics,
        distortion=coefficients,
        square_size=square_size,
        squares_rows=squares_rows,
        squares_cols=squares_cols,
        images=images,
        object_points=object_points,
        image_points=image_points,
        rotations=rotations,
        translations=translations,
    )


def draw_corners_overlay(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Draw checkerboard corner detections as red rings."""

    overlay = Image.fromarray(to_uint8(image)).convert("RGB")
    draw = ImageDraw.Draw(overlay)
    for column, row in points:
        draw.ellipse(
            (column - 3.5, row - 3.5, column + 3.5, row + 3.5),
            outline=(232, 68, 68),
            width=2,
        )
    return np.asarray(overlay)


def draw_reprojection_vectors(
    image: np.ndarray,
    observed: np.ndarray,
    predicted: np.ndarray,
    scale: float = 40.0,
) -> np.ndarray:
    """Magnify residual vectors so small post-refinement errors become visible."""

    overlay = Image.fromarray(to_uint8(image)).convert("RGB")
    draw = ImageDraw.Draw(overlay)
    for (column, row), (predicted_column, predicted_row) in zip(observed, predicted):
        dx = (predicted_column - column) * scale
        dy = (predicted_row - row) * scale
        draw.line((column, row, column + dx, row + dy), fill=(232, 68, 68), width=2)
    return np.asarray(overlay)


def per_view_rmse_bars(scene: CalibrationScene, result) -> np.ndarray:
    """Render a bar chart of per-view reprojection RMSE after refinement."""

    width, height = 400, 300
    canvas = Image.new("RGB", (width, height), (247, 249, 248))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((12, 10), "Per-view reprojection RMSE after refinement", fill=(24, 33, 31), font=font)
    baseline = 245
    left, right = 40, width - 24
    bar_width = (right - left) / len(scene.image_points)
    draw.line((left, baseline, right, baseline), fill=(99, 112, 108), width=2)
    draw.text((right - 38, baseline + 4), "px", fill=(99, 112, 108), font=font)
    for index, observed in enumerate(scene.image_points):
        predicted = project_points(
            scene.object_points,
            result.intrinsics,
            result.rotations[index],
            result.translations[index],
            result.distortion,
        )
        rmse = float(np.sqrt(np.mean(np.sum((predicted - observed) ** 2, axis=1))))
        bar_height = int(round(rmse * 110.0))
        x0 = left + index * bar_width + bar_width * 0.22
        x1 = left + (index + 1) * bar_width - bar_width * 0.22
        draw.rectangle((x0, baseline - bar_height, x1, baseline), fill=(49, 95, 87))
        draw.text(
            (x0, baseline + 4),
            f"{rmse:.2f}",
            fill=(49, 95, 87),
            font=font,
        )
    draw.line((left, baseline - 110, right, baseline - 110), fill=(188, 105, 55), width=1)
    draw.text((right - 92, baseline - 124), "1.0 px reference", fill=(188, 105, 55), font=font)
    return np.asarray(canvas)


def calibration_teaching_assets(
    noise_std: float = 0.35,
    seed: int = 11,
) -> tuple[CalibrationScene, CalibrationResult, Image.Image, Image.Image, Image.Image]:
    """Build the Chapter 08 reference figures from a noisy synthetic scene."""

    scene = synthetic_calibration_scene(num_views=8)
    rng = np.random.default_rng(seed)
    noisy_views = [
        view + rng.normal(0.0, noise_std, view.shape) for view in scene.image_points
    ]
    result = calibrate_camera(noisy_views, scene.object_points, iterations=30)
    selected = 0
    distorted = scene.images[selected]
    undistorted = undistort_image(distorted, result.intrinsics, result.distortion)
    predicted = project_points(
        scene.object_points,
        result.intrinsics,
        result.rotations[selected],
        result.translations[selected],
        result.distortion,
    )

    targets = labelled_grid(
        {f"View {index + 1}": image for index, image in enumerate(scene.images[:6])},
        columns=3,
    )

    correction = Image.new("RGB", (2 * 400, 330), (247, 249, 248))
    correction.paste(Image.fromarray(draw_corners_overlay(distorted, noisy_views[selected])), (0, 30))
    correction.paste(Image.fromarray(to_uint8(undistorted)), (400, 30))
    draw = ImageDraw.Draw(correction)
    font = ImageFont.load_default()
    draw.text((8, 9), "Distorted + detected corners", fill=(24, 33, 31), font=font)
    draw.text((408, 9), "Undistorted with estimated K and k", fill=(24, 33, 31), font=font)

    reprojection = labelled_grid(
        {
            "1. Detected corners (0.35 px noise)": draw_corners_overlay(
                distorted, noisy_views[selected]
            ),
            "2. Residual vectors x40 after refinement": draw_reprojection_vectors(
                distorted, noisy_views[selected], predicted, scale=40.0
            ),
            "3. Per-view RMSE": per_view_rmse_bars(scene, result),
        },
        columns=3,
    )
    return scene, result, targets, correction, reprojection


def save_undistortion_animation(
    path: Path,
    scene: CalibrationScene,
    result: CalibrationResult,
    view_index: int = 0,
    steps: int = 12,
) -> None:
    """Animate the progressive undistortion of one distorted checkerboard view."""

    distorted = scene.images[view_index]
    font = ImageFont.load_default()
    frames: list[Image.Image] = []
    for index in range(steps):
        fraction = index / (steps - 1)
        corrected = undistort_image(
            distorted, scene.intrinsics, scene.distortion * fraction
        )
        frame = Image.new("RGB", (400, 330), "white")
        frame.paste(Image.fromarray(corrected).convert("RGB"), (0, 30))
        draw = ImageDraw.Draw(frame)
        draw.text(
            (8, 9),
            f"Undistortion strength {fraction:.2f}",
            fill="black",
            font=font,
        )
        frames.append(frame)

    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=130,
        loop=0,
        optimize=True,
        disposal=2,
    )


def _yaw_rotation(angle_degrees: float) -> np.ndarray:
    """Return a world-to-camera rotation around the vertical Y axis."""

    angle = np.radians(angle_degrees)
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array(
        [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]]
    )


def pnp_teaching_assets() -> tuple[Image.Image, Image.Image]:
    """Render deterministic 3-D/2-D matches and the recovered camera pose."""

    intrinsics = np.array(
        [[520.0, 0.0, 320.0], [0.0, 520.0, 180.0], [0.0, 0.0, 1.0]]
    )
    rng = np.random.default_rng(19)
    # A mixture of road-surface markings and elevated signs/lamp features.
    # The height variation matters: a nearly planar set makes general PnP DLT
    # ill-conditioned even though a floating-point rank check may say rank 3.
    road_marks = rng.uniform([-3.8, -1.1, 7.5], [3.8, -0.5, 22.0], size=(18, 3))
    elevated = rng.uniform([-4.0, 0.2, 8.0], [4.0, 3.2, 22.5], size=(18, 3))
    world = np.vstack((road_marks, elevated))
    rotation = _yaw_rotation(-4.0)
    true_center = np.array([0.8, 0.1, 0.4])
    translation = -rotation @ true_center
    clean = project_points(world, intrinsics, rotation, translation)
    observed = clean + rng.normal(0.0, 0.3, clean.shape)
    outliers = np.array([3, 10, 14, 27, 33])
    observed[outliers] += np.array(
        [[58.0, -35.0], [-52.0, 44.0], [61.0, 31.0], [-48.0, -39.0], [55.0, 38.0]]
    )
    result = ransac_pnp(
        world, observed, intrinsics, threshold=3.5, max_iterations=1200, seed=23
    )
    predicted = project_points(
        world, intrinsics, result.rotation, result.translation
    )

    image = Image.new("RGB", (640, 390), (242, 246, 244))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((12, 10), "PnP: observed landmarks -> recovered pose -> reprojection", fill=(24, 33, 31), font=font)
    draw.polygon(((205, 370), (435, 370), (380, 115), (260, 115)), fill=(218, 224, 220))
    draw.line((270, 370, 300, 115), fill=(255, 255, 255), width=3)
    draw.line((370, 370, 340, 115), fill=(255, 255, 255), width=3)
    for index, (measurement, estimate) in enumerate(zip(observed, predicted)):
        color = (49, 145, 101) if result.inliers[index] else (220, 65, 65)
        mx, my = measurement
        px, py = estimate
        draw.line((mx, my + 30, px, py + 30), fill=color, width=2)
        draw.ellipse((mx - 4, my + 26, mx + 4, my + 34), outline=color, width=2)
        draw.ellipse((px - 2, py + 28, px + 2, py + 32), fill=(24, 33, 31))
    draw.text((12, 365), "ring = observed   dot = reprojection   green = inlier   red = rejected", fill=(99, 112, 108), font=font)

    center = -result.rotation.T @ result.translation
    top = Image.new("RGB", (560, 390), (247, 249, 248))
    top_draw = ImageDraw.Draw(top)
    top_draw.text((12, 10), "Recovered camera pose: top view (X-Z)", fill=(24, 33, 31), font=font)
    origin_x, bottom_y = 280, 350
    scale_x, scale_z = 55.0, 13.5
    top_draw.line((35, bottom_y, 525, bottom_y), fill=(99, 112, 108), width=2)
    top_draw.line((origin_x, 45, origin_x, bottom_y), fill=(99, 112, 108), width=2)
    for index, point in enumerate(world):
        x = origin_x + point[0] * scale_x
        y = bottom_y - point[2] * scale_z
        color = (49, 145, 101) if result.inliers[index] else (220, 65, 65)
        top_draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color)
    camera_x = origin_x + center[0] * scale_x
    camera_y = bottom_y - center[2] * scale_z
    top_draw.polygon(
        ((camera_x, camera_y - 11), (camera_x - 9, camera_y + 9), (camera_x + 9, camera_y + 9)),
        fill=(188, 105, 55),
    )
    heading = result.rotation.T @ np.array([0.0, 0.0, 1.0])
    top_draw.line(
        (
            camera_x,
            camera_y,
            camera_x + heading[0] * 48,
            camera_y - heading[2] * 48,
        ),
        fill=(188, 105, 55),
        width=3,
    )
    position_error = float(np.linalg.norm(center - true_center))
    median_error = float(np.median(result.errors[result.inliers]))
    top_draw.text((12, 360), f"camera center error {position_error:.3f} m   median reprojection {median_error:.2f} px", fill=(49, 95, 87), font=font)
    return image, top


def save_pnp_motion_animation(path: Path, steps: int = 14) -> None:
    """Animate how forward camera motion changes landmark projections."""

    intrinsics = np.array(
        [[420.0, 0.0, 230.0], [0.0, 420.0, 145.0], [0.0, 0.0, 1.0]]
    )
    world = np.array(
        [[-3.0, -0.7, 9.0], [3.0, -0.7, 9.0], [-3.4, 0.4, 14.0], [3.4, 0.4, 14.0],
         [-3.8, 1.2, 20.0], [3.8, 1.2, 20.0], [-1.2, -0.3, 17.0], [1.2, -0.3, 17.0]]
    )
    font = ImageFont.load_default()
    frames: list[Image.Image] = []
    fractions = list(np.linspace(0.0, 1.0, steps)) + list(np.linspace(1.0, 0.0, steps))
    for fraction in fractions:
        center = np.array([0.8 * np.sin(fraction * np.pi), 0.0, 4.0 * fraction])
        rotation = _yaw_rotation(-5.0 + 10.0 * fraction)
        translation = -rotation @ center
        pixels = project_points(world, intrinsics, rotation, translation)
        frame = Image.new("RGB", (720, 320), (247, 249, 248))
        draw = ImageDraw.Draw(frame)
        draw.text((12, 10), f"camera center = ({center[0]:.1f}, 0.0, {center[2]:.1f}) m", fill=(24, 33, 31), font=font)
        draw.rectangle((10, 30, 470, 320), fill=(232, 237, 234))
        draw.polygon(((145, 319), (335, 319), (285, 80), (195, 80)), fill=(205, 213, 208))
        for index, (x, y) in enumerate(pixels):
            color = (49, 145, 101) if index % 2 == 0 else (188, 105, 55)
            draw.ellipse((x - 5, y + 25, x + 5, y + 35), fill=color, outline="white")
        origin_x, bottom_y = 590, 292
        draw.line((500, bottom_y, 705, bottom_y), fill=(99, 112, 108), width=2)
        draw.line((origin_x, 42, origin_x, bottom_y), fill=(99, 112, 108), width=2)
        for point in world:
            x = origin_x + point[0] * 25
            y = bottom_y - point[2] * 10
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(49, 145, 101))
        camera_x = origin_x + center[0] * 25
        camera_y = bottom_y - center[2] * 10
        draw.polygon(((camera_x, camera_y - 8), (camera_x - 7, camera_y + 7), (camera_x + 7, camera_y + 7)), fill=(188, 105, 55))
        draw.text((515, 302), "top view: PnP recovers this pose", fill=(99, 112, 108), font=font)
        frames.append(frame)
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=110,
        loop=0,
        optimize=True,
        disposal=2,
    )


def visual_odometry_scene():
    """Create a metric landmark track and recover a short vehicle trajectory."""

    intrinsics = np.array(
        [[620.0, 0.0, 320.0], [0.0, 620.0, 180.0], [0.0, 0.0, 1.0]]
    )
    rng = np.random.default_rng(41)
    road = rng.uniform([-5.2, -1.2, 7.5], [5.2, -0.45, 31.0], size=(36, 3))
    structures = rng.uniform([-5.5, 0.1, 8.0], [5.5, 3.6, 32.0], size=(34, 3))
    landmarks = np.vstack((road, structures))
    frame_count = 11
    progress = np.linspace(0.0, 1.0, frame_count)
    centers = np.column_stack(
        (
            0.8 * np.sin(progress * 1.4),
            np.zeros(frame_count),
            6.2 * progress,
        )
    )
    observations: list[np.ndarray] = []
    for frame_index, center in enumerate(centers):
        rotation = _yaw_rotation(-4.5 * progress[frame_index])
        translation = -rotation @ center
        pixels = project_points(landmarks, intrinsics, rotation, translation)
        pixels += rng.normal(0.0, 0.22, pixels.shape)
        outliers = np.array(
            [
                (frame_index * 5 + 3) % len(landmarks),
                (frame_index * 7 + 17) % len(landmarks),
                (frame_index * 11 + 29) % len(landmarks),
            ]
        )
        pixels[outliers] += rng.uniform(38.0, 68.0, (len(outliers), 2)) * rng.choice(
            [-1.0, 1.0], (len(outliers), 2)
        )
        if frame_index >= 3:
            pixels[frame_index : frame_index + 4] = np.nan
        observations.append(pixels)
    result = pnp_odometry(
        landmarks,
        observations,
        intrinsics,
        threshold=3.5,
        max_iterations=700,
        seed=53,
    )
    return centers, result


def draw_vo_trajectory(reference: np.ndarray, estimated: np.ndarray) -> Image.Image:
    """Render reference and estimated camera centers in an X-Z top view."""

    canvas = Image.new("RGB", (680, 420), (247, 249, 248))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text(
        (14, 12),
        "Visual odometry: triangulated landmarks + robust PnP",
        fill=(24, 33, 31),
        font=font,
    )
    left, right, top, bottom = 70, 645, 55, 360
    draw.rounded_rectangle(
        (left, top, right, bottom), radius=12, fill=(232, 237, 234)
    )
    all_points = np.vstack((reference[:, [0, 2]], estimated[:, [0, 2]]))
    minimum = all_points.min(axis=0) - np.array([0.45, 0.35])
    maximum = all_points.max(axis=0) + np.array([0.45, 0.35])
    span = np.maximum(maximum - minimum, 1e-6)

    def screen(point: np.ndarray) -> tuple[float, float]:
        x = left + 25 + (point[0] - minimum[0]) / span[0] * (right - left - 50)
        y = bottom - 20 - (point[2] - minimum[1]) / span[1] * (bottom - top - 40)
        return float(x), float(y)

    reference_xy = [screen(point) for point in reference]
    estimated_xy = [screen(point) for point in estimated]
    for first, second in zip(reference_xy, estimated_xy):
        draw.line((*first, *second), fill=(192, 201, 197), width=1)
    draw.line(reference_xy, fill=(49, 145, 101), width=4, joint="curve")
    draw.line(estimated_xy, fill=(188, 105, 55), width=3, joint="curve")
    for index, (truth, estimate) in enumerate(zip(reference_xy, estimated_xy)):
        draw.ellipse(
            (truth[0] - 4, truth[1] - 4, truth[0] + 4, truth[1] + 4),
            fill=(49, 145, 101),
        )
        draw.ellipse(
            (estimate[0] - 3, estimate[1] - 3, estimate[0] + 3, estimate[1] + 3),
            fill=(188, 105, 55),
        )
        if index in (0, len(reference) - 1):
            draw.text(
                (truth[0] + 6, truth[1] - 7),
                f"frame {index}",
                fill=(49, 95, 87),
                font=font,
            )
    ate = absolute_trajectory_error(estimated, reference, align=False)
    final_error = float(np.linalg.norm(estimated[-1] - reference[-1]))
    draw.text((75, 378), f"ATE RMSE {ate:.3f} m", fill=(49, 95, 87), font=font)
    draw.text(
        (245, 378),
        f"final drift {final_error:.3f} m",
        fill=(188, 105, 55),
        font=font,
    )
    draw.text(
        (420, 378),
        "green truth   orange estimate",
        fill=(99, 112, 108),
        font=font,
    )
    return canvas


def draw_drift_comparison() -> Image.Image:
    """Show how a tiny systematic yaw bias accumulates with distance."""

    canvas = Image.new("RGB", (680, 420), (247, 249, 248))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text(
        (14, 12),
        "The same 0.8 m step, accumulated with a small yaw bias",
        fill=(24, 33, 31),
        font=font,
    )
    origin = np.array([340.0, 365.0])
    scale = 12.5
    draw.line((55, origin[1], 625, origin[1]), fill=(99, 112, 108), width=2)
    colors = [(49, 145, 101), (74, 116, 173), (188, 105, 55), (220, 65, 65)]
    biases = [0.0, 0.15, 0.4, 0.8]
    for color, bias in zip(colors, biases):
        position = np.zeros(2)
        track = [position.copy()]
        heading = 0.0
        for _ in range(28):
            heading += np.radians(bias)
            position += np.array([np.sin(heading), np.cos(heading)]) * 0.8
            track.append(position.copy())
        points = [
            (origin[0] + point[0] * scale, origin[1] - point[1] * scale)
            for point in track
        ]
        draw.line(points, fill=color, width=3, joint="curve")
        final_drift = abs(track[-1][0])
        legend_y = 62 + biases.index(bias) * 25
        draw.line((455, legend_y + 5, 478, legend_y + 5), fill=color, width=3)
        draw.text(
            (486, legend_y),
            f"{bias:.2f} deg/frame -> {final_drift:.2f} m drift",
            fill=color,
            font=font,
        )
    draw.text(
        (14, 394),
        "Random error jitters; systematic bias bends the whole trajectory.",
        fill=(99, 112, 108),
        font=font,
    )
    return canvas


def save_vo_trajectory_animation(
    path: Path,
    reference: np.ndarray,
    estimated: np.ndarray,
) -> None:
    """Animate the estimated trajectory growing frame by frame."""

    frames: list[Image.Image] = []
    for end in list(range(2, len(reference) + 1)) + [len(reference)] * 4:
        frame = draw_vo_trajectory(reference[:end], estimated[:end])
        draw = ImageDraw.Draw(frame)
        draw.text(
            (14, 400),
            f"tracking frame {end - 1} / {len(reference) - 1}",
            fill=(24, 33, 31),
            font=ImageFont.load_default(),
        )
        frames.append(frame)
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=170,
        loop=0,
        optimize=True,
        disposal=2,
    )


def generate_reference_images(output_dir: Path = DEFAULT_OUTPUT) -> list[Path]:
    """Generate all committed reference images and return their paths."""

    output_dir.mkdir(parents=True, exist_ok=True)
    source = synthetic_scene()
    blurred = gaussian_blur(source, size=5, sigma=1.4)
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=float)
    sharpened = convolve2d(source, sharpen_kernel, padding="reflect")

    result = canny(
        source,
        low_threshold=35,
        high_threshold=90,
        gaussian_size=5,
        sigma=1.4,
        return_intermediates=True,
    )
    assert isinstance(result, CannyResult)
    corner_source = harris_scene()
    harris = detect_harris_corners(
        corner_source,
        threshold_rel=0.03,
        min_distance=12,
        max_corners=16,
    )
    shifted = translated_image(corner_source, 12, 18)
    shifted_harris = detect_harris_corners(
        shifted, threshold_rel=0.03, min_distance=12, max_corners=20
    )
    descriptor_a = describe_patches(corner_source, harris.corners, patch_size=11)
    descriptor_b = describe_patches(shifted, shifted_harris.corners, patch_size=11)
    feature_matches = match_descriptors(
        descriptor_a.descriptors,
        descriptor_b.descriptors,
        ratio_threshold=1.0,
        mutual=False,
    )
    # Add two deterministic ambiguous candidates so the teaching image makes
    # RANSAC's rejection step visible instead of relying on random failures.
    distractors = np.array(
        [
            [0, (int(feature_matches.pairs[0, 1]) + 2) % len(descriptor_b.keypoints)],
            [1, (int(feature_matches.pairs[1, 1]) + 3) % len(descriptor_b.keypoints)],
        ],
        dtype=np.int64,
    )
    feature_matches = MatchResult(
        np.vstack((feature_matches.pairs, distractors)),
        np.append(feature_matches.distances, [1.0, 1.0]),
        np.append(feature_matches.ratios, [0.99, 0.99]),
    )
    source_xy = descriptor_a.keypoints[feature_matches.pairs[:, 0]][:, ::-1]
    destination_xy = descriptor_b.keypoints[feature_matches.pairs[:, 1]][:, ::-1]
    robust = ransac_homography(
        source_xy, destination_xy, threshold=2.0, max_iterations=500, seed=11
    )
    stereo_geometry, triangulated_depth = multiview_reference_images()
    road = road_scene()
    road_edges = canny(road, low_threshold=30, high_threshold=85, gaussian_size=5, sigma=1.4)
    road_hough = hough_lines(
        road_edges,
        num_peaks=8,
        threshold_rel=0.35,
        min_support=30,
    )
    square = np.ones((3, 3), dtype=bool)
    morphology_input = morphology_scene()
    morphology_images = {
        "1. Input: specks, hole, gap": morphology_input,
        "2. Erosion": erode(morphology_input),
        "3. Dilation": dilate(morphology_input),
        "4. Opening (removes specks)": opening(morphology_input, square),
        "5. Closing (fills gap)": closing(morphology_input, square),
        "6. Opening + closing cleanup": closing(opening(morphology_input, square), square),
    }
    flow_reference = optical_flow_reference_image()
    calib_scene, calib_result, calibration_targets, distortion_correction, reprojection_figure = (
        calibration_teaching_assets()
    )
    pnp_correspondences, pnp_pose = pnp_teaching_assets()
    vo_reference, vo_result = visual_odometry_scene()

    images: dict[str, Image.Image] = {
        "chapter01_input.png": Image.fromarray(source),
        "chapter01_convolution.png": labelled_grid(
            {
                "Input: noisy synthetic scene": source,
                "Gaussian blur: 5x5, sigma=1.4": to_uint8(blurred),
                "Sharpen convolution: 3x3": to_uint8(sharpened),
            }
        ),
        "chapter02_canny_stages.png": labelled_grid(
            {
                "1. Input": source,
                "2. Gaussian smoothing": to_uint8(result.smoothed),
                "3. Gradient magnitude": to_uint8(result.magnitude, normalize=True),
                "4. Gradient direction": to_uint8(result.direction * (255.0 / 180.0)),
                "5. Non-maximum suppression": to_uint8(result.suppressed, normalize=True),
                "6. Hysteresis result": result.edges,
            }
        ),
        "chapter02_canny_edges.png": Image.fromarray(result.edges),
        "chapter03_harris_stages.png": labelled_grid(
            {
                "1. Input: flat, edges, corners": corner_source,
                "2. Largest eigenvalue": to_uint8(harris.lambda_max, normalize=True),
                "3. Smallest eigenvalue": to_uint8(harris.lambda_min, normalize=True),
                "4. Harris response: warm=corner": signed_response_heatmap(harris.response),
                "5. Detected corners": draw_corner_overlay(corner_source, harris.corners),
            }
        ),
        "chapter03_harris_corners.png": Image.fromarray(
            draw_corner_overlay(corner_source, harris.corners)
        ),
        "chapter04_feature_matches.png": draw_matches(
            corner_source,
            shifted,
            descriptor_a.keypoints,
            descriptor_b.keypoints,
            feature_matches,
        ),
        "chapter04_ransac_inliers.png": draw_matches(
            corner_source,
            shifted,
            descriptor_a.keypoints,
            descriptor_b.keypoints,
            feature_matches,
            robust.inliers,
        ),
        "chapter05_epipolar_geometry.png": stereo_geometry,
        "chapter05_triangulated_depth.png": triangulated_depth,
        "chapter06_hough_lines.png": labelled_grid(
            {
                "Road scene input": road,
                "Canny edges": road_edges,
                "Hough line segments": draw_hough_overlay(road, road_hough),
            }
        ),
        "chapter06_morphology.png": labelled_grid(morphology_images),
        "chapter07_optical_flow.png": flow_reference,
        "chapter08_calibration_targets.png": calibration_targets,
        "chapter08_distortion_correction.png": distortion_correction,
        "chapter08_reprojection_errors.png": reprojection_figure,
        "chapter09_pnp_correspondences.png": pnp_correspondences,
        "chapter09_pose_top_view.png": pnp_pose,
        "chapter10_vo_trajectory.png": draw_vo_trajectory(
            vo_reference, vo_result.centers
        ),
        "chapter10_drift_comparison.png": draw_drift_comparison(),
    }

    paths = []
    for filename, image in images.items():
        path = output_dir / filename
        image.save(path, format="PNG", optimize=True)
        paths.append(path)
    animation_path = output_dir / "chapter03_harris_window.gif"
    save_harris_window_animation(corner_source, harris, animation_path)
    paths.append(animation_path)
    flow_animation_path = output_dir / "chapter07_flow_tracking.gif"
    save_flow_tracking_animation(flow_animation_path)
    paths.append(flow_animation_path)
    calibration_gif_path = output_dir / "chapter08_undistort.gif"
    save_undistortion_animation(calibration_gif_path, calib_scene, calib_result)
    paths.append(calibration_gif_path)
    pnp_gif_path = output_dir / "chapter09_pose_motion.gif"
    save_pnp_motion_animation(pnp_gif_path)
    paths.append(pnp_gif_path)
    vo_gif_path = output_dir / "chapter10_vo_trajectory.gif"
    save_vo_trajectory_animation(vo_gif_path, vo_reference, vo_result.centers)
    paths.append(vo_gif_path)
    return paths


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    for path in generate_reference_images(args.output_dir):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
