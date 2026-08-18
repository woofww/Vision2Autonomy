from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from examples.generate_reference_images import (
    generate_reference_images,
    labelled_grid,
    synthetic_scene,
)


def test_synthetic_scene_is_deterministic() -> None:
    first = synthetic_scene(seed=11)
    second = synthetic_scene(seed=11)

    np.testing.assert_array_equal(first, second)
    assert first.shape == (256, 256)
    assert first.dtype == np.uint8
    assert first.min() < first.max()


def test_reference_generator_writes_expected_pngs(tmp_path: Path) -> None:
    paths = generate_reference_images(tmp_path)

    assert {path.name for path in paths} == {
        "chapter01_input.png",
        "chapter01_convolution.png",
        "chapter02_canny_stages.png",
        "chapter02_canny_edges.png",
    }
    for path in paths:
        assert path.exists()
        with Image.open(path) as image:
            image.verify()


def test_labelled_grid_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        labelled_grid({})

