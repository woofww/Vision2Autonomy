from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from vision2autonomy.examples.reference_images import (
    generate_reference_images,
    labelled_grid,
    synthetic_scene,
)


@pytest.fixture(scope="module")
def generated_paths(tmp_path_factory: pytest.TempPathFactory) -> list[Path]:
    """Generate the full reference image set once per module."""

    return generate_reference_images(tmp_path_factory.mktemp("figures"))


def test_synthetic_scene_is_deterministic() -> None:
    first = synthetic_scene(seed=11)
    second = synthetic_scene(seed=11)

    np.testing.assert_array_equal(first, second)
    assert first.shape == (256, 256)
    assert first.dtype == np.uint8
    assert first.min() < first.max()


def test_reference_generator_writes_expected_pngs(generated_paths: list[Path]) -> None:
    assert {path.name for path in generated_paths} == {
        "chapter01_input.png",
        "chapter01_convolution.png",
        "chapter02_canny_stages.png",
        "chapter02_canny_edges.png",
        "chapter03_harris_stages.png",
        "chapter03_harris_corners.png",
        "chapter03_harris_window.gif",
        "chapter04_feature_matches.png",
        "chapter04_ransac_inliers.png",
        "chapter05_epipolar_geometry.png",
        "chapter05_triangulated_depth.png",
        "chapter06_hough_lines.png",
        "chapter06_morphology.png",
        "chapter07_optical_flow.png",
        "chapter07_flow_tracking.gif",
        "chapter08_calibration_targets.png",
        "chapter08_distortion_correction.png",
        "chapter08_reprojection_errors.png",
        "chapter08_undistort.gif",
    }
    for path in generated_paths:
        assert path.exists()
        with Image.open(path) as image:
            image.verify()


def test_all_gifs_are_animated(generated_paths: list[Path]) -> None:
    for gif_path in (path for path in generated_paths if path.suffix == ".gif"):
        with Image.open(gif_path) as animation:
            assert animation.is_animated
            assert animation.n_frames >= 10


def test_labelled_grid_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        labelled_grid({})