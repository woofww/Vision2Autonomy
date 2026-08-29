"""Generate deterministic reference images used by Chapters 01 through 08.

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
    calibrate_camera,
    camera_matrix,
    estimate_fundamental,
    planar_object_points,
    project_points,
    project_points_3d,
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
