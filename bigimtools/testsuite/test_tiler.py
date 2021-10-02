from __future__ import annotations

import numpy as np
import pytest

from bigimtools import adapters, tiler

# TEST_TILE_SIZES = ((10, 10), (10, 23), (10, 23))
# TEST_INIT = ((1, 1), (5, 10), (10, 5))
TEST_TILE_SIZES = ((64, 64), (32, 64), (64, 32))
TEST_INIT = ((1, 1), (1, 3), (3, 1))


def to_tile(im, tile_size, overlap):
    return adapters.TiledImage.from_dict(
        dict(tiler.split_into_tiles(im, tile_size, overlap)), overlap
    )


def assert_tiledict_almost_equal(actual, desired, **kwargs):
    actual = adapters.TiledImage.from_dict(actual, 0)
    actual = actual.reduce(lambda x: x)

    desired = adapters.TiledImage.from_dict(desired, 0)
    desired = desired.reduce(lambda x: x)

    np.testing.assert_almost_equal(desired, actual, **kwargs)


def test_tiled_mixin(imcamera):
    # imcamera.shape = (512, 512)
    tiles = to_tile(imcamera, (128, 64), (2, 2))
    assert tiles.overlap == (2, 2)
    assert tiles.grid_shape == (4 + 1, 8 + 1)  # por el overlap
    assert tiles.tile_shape == (128, 64)
    assert tiles.tile_shapes == ((128, 64),)
    assert tiles.dtype == np.uint8
    assert tiles.dtypes == (np.uint8,)
    assert tiles.is_homogeneous
    assert tiles.is_filled
    assert tiles.is_index_compatible
    assert tiles.is_compatible_tile(np.ones((128, 64)))
    assert not tiles.is_compatible_tile(np.ones((3, 3)))

    tiles = adapters.TiledImage.from_dict(
        {(0, 0): np.ones((3, 3)), (0, 1): 2 * np.ones((3, 3))},
        overlap=(0, 0),
    )
    assert tiles.reduce_to_dict(np.mean) == {(0, 0): 1, (0, 1): 2}


@pytest.mark.parametrize("tile_size", TEST_TILE_SIZES)
@pytest.mark.parametrize("overlap", ((0, 0), (5, 5), (3, 6), (6, 3)))
def test_round_trip(imcamera, tile_size, overlap):

    tiles = to_tile(imcamera, tile_size, overlap)

    for v in tiles.values():
        assert v.shape == tile_size

    merge = tiler.join_tiles(tiles)

    sh = imcamera.shape
    np.testing.assert_almost_equal(merge[: sh[0], : sh[1]], imcamera)


@pytest.mark.parametrize("tile_size", TEST_TILE_SIZES)
@pytest.mark.parametrize("overlap", ((5, 5), (3, 6), (6, 3)))
@pytest.mark.parametrize("init", TEST_INIT)
@pytest.mark.parametrize(
    "equalizer",
    (
        (
            False,
            lambda t, i: tiler.estimate_corrections_seq(
                t, i, est_func=lambda x, y: 2
            ),
        ),
        (True, lambda t, i: tiler.estimate_corrections(t, i)),
    ),
)
def test_equalize_simple1(
    imcamera, tile_size, overlap, init, equalizer
):
    only_square, equalizer = equalizer
    if only_square and (
        tile_size[0] != tile_size[1] or overlap[0] != overlap[1]
    ):
        pytest.skip(
            "This equalizer currently only works for square tiles/overlaps"
        )

    tiles = to_tile(imcamera, tile_size, overlap)

    corrections = equalizer(tiles, init)

    grount_truth = {k: 2 for k, v in corrections.items()}
    grount_truth[init] = 1.0

    assert_tiledict_almost_equal(corrections, grount_truth)


@pytest.mark.parametrize("tile_size", TEST_TILE_SIZES)
@pytest.mark.parametrize("overlap", ((5, 5), (3, 6), (6, 3)))
@pytest.mark.parametrize("init", TEST_INIT)
@pytest.mark.parametrize(
    "equalizer",
    (
        (
            False,
            lambda t, i: tiler.estimate_corrections_seq(
                t, i, agg_func=lambda x: 3
            ),
        ),
        (True, lambda t, i: tiler.estimate_corrections(t, i)),
    ),
)
def test_equalize_simple2(
    imcamera, tile_size, overlap, init, equalizer
):
    only_square, equalizer = equalizer
    if only_square and (
        tile_size[0] != tile_size[1] or overlap[0] != overlap[1]
    ):
        pytest.skip(
            "This equalizer currently only works for square tiles/overlaps"
        )

    tiles = to_tile(imcamera, tile_size, overlap)

    corrections = equalizer(tiles, init)

    grount_truth = {k: 3 for k, v in corrections.items()}
    grount_truth[init] = 1.0

    assert_tiledict_almost_equal(corrections, grount_truth)


@pytest.mark.parametrize("tile_size", TEST_TILE_SIZES)
@pytest.mark.parametrize("overlap", ((5, 5), (3, 6), (6, 3)))
@pytest.mark.parametrize("init", TEST_INIT)
def test_equalize(imcamera, tile_size, overlap, init):
    tiles = to_tile(imcamera, tile_size, overlap)
    merge = tiler.join_tiles(tiles, tiler.ConstantDict(13))

    sh = imcamera.shape
    np.testing.assert_almost_equal(
        merge[: sh[0], : sh[1]], 13 * imcamera
    )


@pytest.mark.parametrize("tile_size", TEST_TILE_SIZES)
@pytest.mark.parametrize("overlap", ((5, 5), (3, 6), (6, 3)))
@pytest.mark.parametrize("init", TEST_INIT)
@pytest.mark.parametrize(
    "equalizer",
    (
        (False, lambda t, i: tiler.estimate_corrections_seq(t, i)),
        (True, lambda t, i: tiler.estimate_corrections(t, i)),
    ),
)
def test_equalize_change_init(
    imcamera, tile_size, overlap, init, equalizer
):
    only_square, equalizer = equalizer
    if only_square and (
        tile_size[0] != tile_size[1] or overlap[0] != overlap[1]
    ):
        pytest.skip(
            "This equalizer currently only works for square tiles/overlaps"
        )

    tiles = to_tile(imcamera, tile_size, overlap)

    tiles[init] = tiles[init] * 13.0

    corrections = equalizer(tiles, init)

    merge = tiler.join_tiles(tiles, corrections)

    sh = imcamera.shape
    np.testing.assert_almost_equal(
        merge[: sh[0], : sh[1]], imcamera * 13.0
    )


@pytest.mark.parametrize("tile_size", TEST_TILE_SIZES)
@pytest.mark.parametrize("overlap", ((5, 5), (3, 6), (6, 3)))
@pytest.mark.parametrize("init", TEST_INIT)
@pytest.mark.parametrize(
    "equalizer",
    (
        (False, lambda t, i: tiler.estimate_corrections_seq(t, i)),
        (True, lambda t, i: tiler.estimate_corrections(t, i)),
    ),
)
def test_equalize_changed(
    imcamera, tile_size, overlap, init, equalizer
):
    only_square, equalizer = equalizer
    if only_square and (
        tile_size[0] != tile_size[1] or overlap[0] != overlap[1]
    ):
        pytest.skip(
            "This equalizer currently only works for square tiles/overlaps"
        )

    tiles = to_tile(imcamera, tile_size, overlap)

    for ndx0 in (1, 3, 5):
        for ndx1 in (2, 4, 6):
            tiles[(ndx0, ndx1)] = tiles[(ndx0, ndx1)] * 13.0

    corrections = equalizer(tiles, init)

    for ndx0 in (1, 3, 5):
        for ndx1 in (2, 4, 6):
            assert np.isclose(corrections[(ndx0, ndx1)], 1 / 13.0)


def test_build_matrix_simple():
    tile_size = (5, 5)
    overlap = (2, 2)
    tiles = {}
    tmp = np.ones(tile_size)
    tiles[(0, 0)] = tmp
    tiles[(0, 1)] = tmp * 2
    M1, P1, M2, P2 = tiler.build_overlap_matrix(
        adapters.TiledImage.from_dict(tiles, overlap)
    )
    assert M1.shape == (10, 1)
    assert M2.shape == (10, 1)
    assert P1.shape == (1, 2)
    assert P2.shape == (1, 2)


def test_equalize_simple():
    tile_size = (5, 5)
    overlap = (2, 2)
    tiles = {}
    tmp = np.ones(tile_size)
    tiles[(0, 0)] = tmp
    tiles[(0, 1)] = tmp * 2
    result = tiler.estimate_corrections(
        adapters.TiledImage.from_dict(tiles, overlap)
    )
    assert_tiledict_almost_equal(result, {(0, 0): 1, (0, 1): 1 / 2})


# Crashes test suite! Beware
# @pytest.mark.parametrize("tile_size", ((3, 3), (5, 10), (10, 10)))
# @pytest.mark.parametrize("overlap", ((5, 5), (3, 6), (6, 3)))
# @pytest.mark.parametrize("init", ((1, 1), (5, 10), (10, 5)))
# def test_ov_equalize_changed(imcamera, tile_size, overlap, init):

#     tiles = dict(tiler.split_into_tiles(imcamera, tile_size, overlap))

#     for ndx0 in (3, 5, 13):
#         for ndx1 in (4, 7, 20):
#             tiles[(ndx0, ndx1)] = tiles[(ndx0, ndx1)] * 13.0
#     overlap_mat = tiler.overlap_matrix(tiles, overlap)
#     corrections = tiler.coef_matrix_brute_force(overlap_mat)
#     print(corrections)
#     for ndx0 in (3, 5, 13):
#         for ndx1 in (4, 7, 20):
#             assert corrections[(ndx0, ndx1)] == 1 / 13.0
