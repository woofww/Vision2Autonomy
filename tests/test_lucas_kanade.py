import numpy as np
import pytest

from vision2autonomy.motion import lucas_kanade_flow


def translated(image: np.ndarray, row_shift: int, column_shift: int) -> np.ndarray:
    """Translate without wraparound, filling uncovered pixels with zero."""

    output = np.zeros_like(image)
    source_rows = slice(max(0, -row_shift), min(image.shape[0], image.shape[0] - row_shift))
    source_columns = slice(max(0, -column_shift), min(image.shape[1], image.shape[1] - column_shift))
    target_rows = slice(max(0, row_shift), min(image.shape[0], image.shape[0] + row_shift))
    target_columns = slice(max(0, column_shift), min(image.shape[1], image.shape[1] + column_shift))
    output[target_rows, target_columns] = image[source_rows, source_columns]
    return output


def textured_frame(size: int = 80) -> np.ndarray:
    """Create a deterministic frame with strong smooth texture in both axes.

    The ramp plus sawtooth stripes keep the brightness-constancy linearization
    valid, so a known integer translation is recovered exactly. Noise or hard
    periodic edges bias the least-squares estimate, which is itself a useful
    property to discover while experimenting.
    """

    rows, columns = np.indices((size, size))
    return np.clip(columns * 2.0 + 12.0 * (rows % 5), 0, 255).astype(np.uint8)


def test_flow_recovers_known_translation() -> None:
    frame_a = textured_frame()
    frame_b = translated(frame_a, row_shift=2, column_shift=3)
    result = lucas_kanade_flow(frame_a, frame_b, window_size=15)

    interior = result.valid[20:-20, 20:-20]
    assert interior.sum() > 0.3 * interior.size
    flows = result.flow[20:-20, 20:-20][interior]
    assert np.abs(np.median(flows[:, 0]) - 3.0) < 0.15
    assert np.abs(np.median(flows[:, 1]) - 2.0) < 0.15


def test_flow_is_zero_for_identical_frames() -> None:
    frame = textured_frame()
    result = lucas_kanade_flow(frame, frame)
    np.testing.assert_allclose(result.flow, 0.0, atol=1e-12)


def test_flat_region_has_low_confidence() -> None:
    frame_a = np.full((60, 60), 120.0, dtype=np.float64)
    frame_b = frame_a.copy()
    result = lucas_kanade_flow(frame_a, frame_b, window_size=11)
    # The inner region is truly flat; only artificial border gradients may fire.
    assert not result.valid[15:-15, 15:-15].any()
    np.testing.assert_allclose(result.confidence[15:-15, 15:-15], 0.0, atol=1e-12)


def test_window_border_is_invalid() -> None:
    frame_a = textured_frame(size=40)
    result = lucas_kanade_flow(frame_a, frame_a, window_size=11)
    half = 5
    assert not result.valid[:half, :].any()
    assert not result.valid[:, :half].any()
    assert result.valid[half:-half, half:-half].any()


def test_single_edge_is_ambiguous() -> None:
    frame = np.zeros((60, 60), dtype=np.float64)
    frame[:, 20:40] = 200.0  # vertical edge, texture in only one direction
    shifted = translated(frame, row_shift=0, column_shift=3)
    result = lucas_kanade_flow(frame, shifted, window_size=11)
    assert not result.valid[15:-15, 15:-15].any(), "a single edge must not yield confident 2-D flow"


@pytest.mark.parametrize("window_size", [0, 2, 4, 6])
def test_window_size_must_be_odd_and_at_least_three(window_size: int) -> None:
    with pytest.raises(ValueError, match="window_size"):
        lucas_kanade_flow(np.zeros((8, 8)), np.zeros((8, 8)), window_size=window_size)


def test_frames_must_be_equally_shaped_grayscale() -> None:
    with pytest.raises(ValueError, match="same shape"):
        lucas_kanade_flow(np.zeros((8, 8)), np.zeros((9, 8)))
    with pytest.raises(ValueError, match="two-dimensional"):
        lucas_kanade_flow(np.zeros((8, 8, 3)), np.zeros((8, 8, 3)))