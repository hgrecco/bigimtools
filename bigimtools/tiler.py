"""
    bigimtools.tiler
    ~~~~~~~~~~~~~~~~

    Functions to manage tiled images.

    :copyright: 2021 by bigimtools Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

from __future__ import annotations

import numpy as np


class ConstantDict:
    """Dummy container class that only returns a single value."""

    def __init__(self, value):
        self.value = value

    def __getitem__(self, item):
        return self.value

    def get(self, item, default):
        return self.value


_NaNArray = ConstantDict(np.nan)
_OneArray = ConstantDict(1)


def estimate_correction(reference_tile, target_tile):
    """Estimate the correction factor to equalize images by
    calculating the median of the |target| / |reference|

    Parameters
    ----------
    reference_tile : np.ndarray
    target_tile : np.ndarray

    Returns
    -------
    number

    """
    if target_tile is None:
        return np.nan
    if reference_tile is None:
        return np.nan

    return np.nanmedian(np.abs(target_tile) / np.abs(reference_tile))


def scan_nearest_first(image_size, init, skip_init=True):
    """Iterate all pixels from closest to farthest.

    Parameters
    ----------
    image_size : (int, int)
    init : (int, int)
        starting coordinates.
    skip_init : bool
        if True, the init point will not be yielded.

    Yields
    ------
    int, int
        coordinates
    """

    [M, N] = np.meshgrid(
        np.arange(0, image_size[0] + 1), np.arange(0, image_size[1] + 1)
    )
    M = M.flatten()
    N = N.flatten()
    dist = np.hypot(M - init[0], N - init[1])

    it = iter(np.argsort(dist.flatten()))
    if skip_init:
        next(it)

    for ndx in it:
        yield M[ndx], N[ndx]


def split_into_tiles(
    img,
    tile_size: tuple[int, int],
    overlap: tuple[int, int],
    fill_value=0,
):
    """Split an image into smaller tiles.

    Parameters
    ----------
    img : object accepting t = obj[a:b, c:d] interface.
        image to split.
    tile_size : (int, int)
        size of the sliding tile_size and therefore of the resulting tiles.
    overlap : (int, int)
        length of the overlap in each dimension.
    fill_value : number

    Yields
    ------
    (int, int), np.ndarray
        tile indices in each dimension and tile content
    """

    sh = img.shape

    if isinstance(tile_size, int):
        tile_size = (tile_size, tile_size)

    if isinstance(overlap, int):
        overlap = (overlap, overlap)

    tsz0, tsz1 = tile_size
    ov0, ov1 = overlap

    ndx1 = 0
    fr1 = 0
    while fr1 < sh[1]:
        ndx0 = 0
        to1 = fr1 + tsz1
        fr0 = 0
        while fr0 < sh[0]:
            to0 = fr0 + tsz0
            tile = img[fr0:to0, fr1:to1]
            if tile.shape == tile_size:
                yield (ndx0, ndx1), tile

            else:
                tmp = np.full(
                    tile_size, fill_value=fill_value, dtype=img.dtype
                )
                sz = tile.shape
                tmp[: sz[0], : sz[1]] = tile
                yield (ndx0, ndx1), tmp

            ndx0 += 1
            fr0 += tsz0 - ov0

        ndx1 += 1
        fr1 += tsz1 - ov1


def equalize_tiles(
    tiles: dict[(int, int), np.ndarray],
    overlap: tuple[int, int],
    init: tuple[int, int],
    est_func=estimate_correction,
    agg_func=np.nanmax,
    scan_func=scan_nearest_first,
):
    """
    Iterate over all tiles generating a new set of equalized tile.

    Each tile is compared to the neighbours that have been already
    equalized, it is then equalized and placed.

    Parameters
    ----------
    tiles : dict[(int, int), np.ndarray)
        maps tile indices to tile.
    overlap : (int, int)
        length of the overlap in each dimension.
    init : (int, int)
        starting tile.
    est_func : callable (np.ndarray, np.ndarray) -> float
        function to estimate the correction factor given two tiles (target, reference).
    agg_func : callable (iterable of numbers) -> number
        function aggregate all corrections values into one.
    scan_func : iterator (np.ndarray, np.ndarray) -> (int, int)
        iterator to scan all tiles in any given order.

    Returns
    -------
    dict[(int, int), float]
        An object mapping tile indices to correction
    """

    out = {init: 1}

    if isinstance(overlap, int):
        overlap = (overlap, overlap)

    ov0, ov1 = overlap

    ndx0s, ndx1s = zip(*tiles.keys())

    for ndx0, ndx1 in scan_func((max(ndx0s), max(ndx1s)), init):
        ndx0 = int(ndx0)
        ndx1 = int(ndx1)
        tile = tiles[(ndx0, ndx1)]

        corr1 = est_func(
            tile[-ov0:, :],
            tiles.get((ndx0 + 1, ndx1), _NaNArray)[:+ov0, :]
            * out.get((ndx0 + 1, ndx1), np.nan),
        )
        corr2 = est_func(
            tile[:+ov0, :],
            tiles.get((ndx0 - 1, ndx1), _NaNArray)[-ov0:, :]
            * out.get((ndx0 - 1, ndx1), np.nan),
        )
        corr3 = est_func(
            tile[:, -ov1:],
            tiles.get((ndx0, ndx1 + 1), _NaNArray)[:, :+ov1]
            * out.get((ndx0, ndx1 + 1), np.nan),
        )
        corr4 = est_func(
            tile[:, :+ov1],
            tiles.get((ndx0, ndx1 - 1), _NaNArray)[:, -ov1:]
            * out.get((ndx0, ndx1 - 1), np.nan),
        )

        corr = agg_func([corr1, corr2, corr3, corr4])
        if np.isnan(corr):
            corr = 1

        out[(ndx0, ndx1)] = corr

    return out


def join_tiles(
    tiles: dict[(int, int), np.ndarray],
    overlap: tuple[int, int],
    corrections: dict[(int, int), float] = _OneArray,
    out=None,
):
    """Join tiles into a single image.

    Parameters
    ----------
    tiles : object accepting t = obj[a, c] interface.
        set of tiles
    overlap : (int, int)
        length of the overlap in each dimension.
    corrections: dict[(int, int), float]
        correction factor for each tile.
    out : object accepting accepting obj[a:b, c:d] = im interface
        object in which the equalized tiles are going to be stored.

    Returns
    -------
    type(out) or np.ndimage
    """

    wall_shape, tile_size, dtype = tiledict_info(tiles)
    tsz0, tsz1 = tile_size

    if isinstance(overlap, int):
        overlap = (overlap, overlap)

    ov0, ov1 = overlap

    if out is None:
        wsz0, wsz1 = wall_shape
        out = np.zeros(
            (wsz0 * (tsz0 - ov0), wsz1 * (tsz1 - ov1)), dtype=dtype
        )

    for (ndx0, ndx1), tile in tiles.items():
        fr0 = ndx0 * (tsz0 - ov0)
        fr1 = ndx1 * (tsz1 - ov1)
        # The or None is to allow for 0 overlap.
        out[fr0 : (fr0 + tsz0 - ov0), fr1 : (fr1 + tsz1 - ov1)] = tile[
            : -ov0 or None, : -ov1 or None
        ] * corrections.get((ndx0, ndx1), 1)

    return out


def correct_tiles(
    tiles: dict[(int, int), np.ndarray],
    overlap: tuple[int, int],
    corrections: dict[(int, int), float] = _OneArray,
    out=None,
):
    """Correct tiles by multiplying each by a factor.

    Parameters
    ----------
    tiles : object accepting t = obj[a, c] interface.
        set of tiles
    overlap : (int, int)
        length of the overlap in each dimension.
    corrections: dict[(int, int), float]
        correction factor for each tile.
    out : object accepting accepting obj[a:b, c:d] = im interface
        object in which the equalized tiles are going to be stored.

    Returns
    -------
    type(out) or np.ndimage
    """

    ov0, ov1 = overlap

    if out is None:
        out = {}

    for (ndx0, ndx1), tile in tiles.items():
        out[ndx0, ndx1] = tile[:-ov0, :-ov1] * corrections.get(
            (ndx0, ndx1), 1
        )

    return out


def tiledict_info(
    tiles: dict[(int, int), np.ndarray], default_dtype=np.float32
):
    """Get a information from group of tiles.

    Parameters
    ----------
    tiles : dict[(int, int), np.ndarray]
        maps tile indices to tile
    default_dtype : dtype
        consensus numpy type to be used if tiles are not homogeneous.

    Returns
    -------
    (int, int), (int, int), dtype
        wall shape, tile shape, dtype
    """
    ndx0s, ndx1s = zip(*tiles.keys())
    shapes = set((el.shape for el in tiles.values()))
    dtypes = set((getattr(v, "dtype", None) for v in tiles.values()))

    if len(shapes) > 1:
        raise ValueError("All tiles must be the same shape.")

    if len(dtypes) == 1:
        dtype = dtypes.pop()
    else:
        dtype = default_dtype

    return (max(ndx0s) + 1, max(ndx1s) + 1), shapes.pop(), dtype
