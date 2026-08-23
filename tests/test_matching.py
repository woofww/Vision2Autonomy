import numpy as np
import pytest

from vision2autonomy.features.matching import (
    describe_patches,
    estimate_homography,
    match_descriptors,
    project_points,
    ransac_homography,
)


def test_patch_descriptor_is_zero_mean_unit_norm_and_skips_border() -> None:
    image = np.arange(100, dtype=float).reshape(10, 10)
    result = describe_patches(image, [[5, 5], [0, 0]], patch_size=3)
    assert result.keypoints.shape == (1, 2)
    np.testing.assert_allclose(result.descriptors.mean(axis=1), 0.0, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(result.descriptors, axis=1), 1.0)


def test_ratio_and_mutual_matching_reject_ambiguous_descriptor() -> None:
    first = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
    second = np.array([[1.0, 0.0], [0.0, 1.0], [0.49, 0.51]])
    result = match_descriptors(first, second, ratio_threshold=0.8, mutual=True)
    np.testing.assert_array_equal(result.pairs, [[0, 0], [1, 2], [2, 1]])
    assert np.all(result.ratios < 0.8)


def test_normalized_dlt_recovers_known_projective_transform() -> None:
    source = np.array([[0, 0], [2, 0], [2, 1], [0, 1], [0.5, 0.6]], dtype=float)
    expected = np.array([[1.2, 0.1, 4.0], [-0.05, 0.9, 3.0], [0.01, -0.02, 1.0]])
    destination = project_points(source, expected)
    estimated = estimate_homography(source, destination)
    np.testing.assert_allclose(estimated, expected, atol=1e-10)


def test_ransac_recovers_transform_and_rejects_outliers() -> None:
    rng = np.random.default_rng(12)
    source = rng.uniform(0, 100, size=(40, 2))
    expected = np.array([[1.0, 0.03, 14.0], [-0.02, 1.0, -8.0], [0.0002, -0.0001, 1.0]])
    destination = project_points(source, expected)
    destination += rng.normal(0, 0.15, destination.shape)
    outliers = np.array([1, 7, 15, 24, 33])
    destination[outliers] = rng.uniform(-80, 180, size=(len(outliers), 2))

    result = ransac_homography(source, destination, threshold=1.0, max_iterations=400, seed=4)

    assert result.inliers.sum() == 35
    assert not result.inliers[outliers].any()
    probe = np.array([[20.0, 30.0], [80.0, 60.0]])
    np.testing.assert_allclose(project_points(probe, result.matrix), project_points(probe, expected), atol=0.25)


@pytest.mark.parametrize("patch_size", [0, 2, 4])
def test_descriptor_rejects_invalid_patch_size(patch_size: int) -> None:
    with pytest.raises(ValueError, match="patch_size"):
        describe_patches(np.zeros((10, 10)), [[5, 5]], patch_size=patch_size)


def test_matching_handles_empty_inputs() -> None:
    result = match_descriptors(np.empty((0, 4)), np.ones((2, 4)))
    assert result.pairs.shape == (0, 2)
