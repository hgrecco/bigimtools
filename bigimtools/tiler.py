"""
    bigimtools.tiler
    ~~~~~~~~~~~~~~~~

    Functions to manage tiled images.

    :copyright: 2021 by bigimtools Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

from __future__ import annotations

import numpy as np

from . import adapters


def split_into_tiles(
    img,
    tile_size: adapters.IntPair,
    overlap: adapters.IntPair,
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


def join_tiles(
    tiles: adapters.TiledImage,
    corrections: dict[adapters.IntPair, float] = None,
    out=None,
):
    """Join tiles into a single image.

    Parameters
    ----------
    tiles : adapters.TiledImage
        A tiled image.
    corrections: dict[(int, int), float]
        correction factor for each tile.
    out : object accepting accepting obj[a:b, c:d] = im interface
        object in which the equalized tiles are going to be stored.

    Returns
    -------
    type(out) or np.ndimage
    """

    tsz0, tsz1 = tiles.tile_shape
    ov0, ov1 = tiles.overlap

    if out is None:
        gsz0, gsz1 = tiles.grid_shape
        out = np.zeros(
            (gsz0 * (tsz0 - ov0), gsz1 * (tsz1 - ov1)),
            dtype=tiles.dtype if corrections is None else np.float,
        )

    for (ndx0, ndx1), tile in tiles.items():
        fr0 = ndx0 * (tsz0 - ov0)
        fr1 = ndx1 * (tsz1 - ov1)

        this_tile = tile[: -ov0 or None, : -ov1 or None]

        if corrections is not None:
            this_tile = this_tile * corrections.get((ndx0, ndx1), 1)

        # The or None is to allow for 0 overlap.
        out[
            fr0 : (fr0 + tsz0 - ov0), fr1 : (fr1 + tsz1 - ov1)
        ] = this_tile

    return out


def correct_tiles(
    tiles: adapters.TiledImage,
    corrections: dict[adapters.IntPair, float] = None,
    out=None,
):
    """Correct tiles by multiplying each by a factor.

    Parameters
    ----------
    tiles : adapters.TiledImage
        A tiled image.
    corrections: dict[(int, int), float]
        correction factor for each tile.
    out : object accepting accepting obj[a:b, c:d] = im interface
        object in which the equalized tiles are going to be stored.

    Returns
    -------
    type(out) or np.ndimage
    """

    ov0, ov1 = tiles.overlap

    if out is None:
        out = {}

    for (ndx0, ndx1), tile in tiles.items():
        out[ndx0, ndx1] = tile[:-ov0, :-ov1]

        if corrections is not None:
            out[ndx0, ndx1] = out[ndx0, ndx1] * corrections.get(
                (ndx0, ndx1), 1
            )

    return out


######################
# Correction related
######################


def estimate_pair_correction(reference_tile, target_tile) -> float:
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


def scan_nearest_first(
    image_size: adapters.IntPair, init: adapters.IntPair, skip_init=True
):
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


def _yield_overlaps(tiles: adapters.TiledImage, do_not_repeat=True):

    assert tiles.is_filled, "This routine only works for filled images"

    gsh = tiles.grid_shape
    for ndx0 in range(gsh[0]):
        for ndx1 in range(gsh[1]):
            if (ndx0, ndx1) not in tiles:
                continue

            center = ndx0, ndx1

            yield center, *tiles.get_overlap_regions(center, (+1, 0))
            if not do_not_repeat:
                yield center, *tiles.get_overlap_regions(
                    center, (-1, 0)
                )

            yield center, *tiles.get_overlap_regions(center, (0, -1))
            if not do_not_repeat:
                yield center, *tiles.get_overlap_regions(
                    center, (0, +1)
                )


def yield_overlaps(tiles: adapters.TiledImage, do_not_repeat=True):
    """Yield overlapped regions.

    Parameters
    ----------
    tiles : adapters.TiledImage
        An tiled image.
    do_not_repeat : bool
        If True, only right and bottom overlaps will be yielded.
        If False, also top and left overlap will be yielded.

    Yields
    ------
    (int, int), (int, int), np.ndarray, np.ndarray
        center index, neighbour index, region of the center tile, region of the neighbour tile
    """

    for o in _yield_overlaps(tiles, do_not_repeat):
        if o[-1] is not None:
            yield o


def estimate_corrections_seq(
    tiles: adapters.TiledImage,
    init: adapters.IntPair,
    est_func=estimate_pair_correction,
    agg_func=np.nanmax,
    scan_func=scan_nearest_first,
):
    """
    Iterate over all tiles generating a new set of equalized tile.

    Each tile is compared to the neighbours that have been already
    equalized, it is then equalized and placed.

    Parameters
    ----------
    tiles : adapters.TiledImage
        A tiled image.
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

    ndx0s, ndx1s = zip(*tiles.keys())

    for ndx0, ndx1 in scan_func((max(ndx0s), max(ndx1s)), init):
        center = int(ndx0), int(ndx1)

        other_ndx, tile_reg, other_reg = tiles.get_overlap_regions(
            center, (+1, 0)
        )
        corr1 = (
            np.nan
            if other_reg is None
            else est_func(tile_reg, other_reg)
            * out.get(other_ndx, np.nan)
        )

        other_ndx, tile_reg, other_reg = tiles.get_overlap_regions(
            center, (-1, 0)
        )
        corr2 = (
            np.nan
            if other_reg is None
            else est_func(tile_reg, other_reg)
            * out.get(other_ndx, np.nan)
        )

        other_ndx, tile_reg, other_reg = tiles.get_overlap_regions(
            center, (0, +1)
        )
        corr3 = (
            np.nan
            if other_reg is None
            else est_func(tile_reg, other_reg)
            * out.get(other_ndx, np.nan)
        )

        other_ndx, tile_reg, other_reg = tiles.get_overlap_regions(
            center, (0, -1)
        )
        corr4 = (
            np.nan
            if other_reg is None
            else est_func(tile_reg, other_reg)
            * out.get(other_ndx, np.nan)
        )

        corr = agg_func([corr1, corr2, corr3, corr4])
        if np.isnan(corr):
            corr = 1

        out[center] = corr

    return out


def build_overlap_matrix(tiles: adapters.TiledImage):
    """Takes overlapping regions.

    Strategy:

    In matrices M1 and M2, different overlap regions (in the second dimension)
    will be stored linearized (in the first dimension) in matrices M1 and M2.

    1. In a REFerence tile at (ndx0, ndx1)
       pick a NEIghbour (ndx0 + 1, ndx1)
    2. take the flattened pixels in REF and store them in M1
    3. take the flattened pixels in NEI and store them in M2
    4. Pick the NEIghbour (ndx0, ndx1 + 1) and repeat 3 and 4.
    5. Repeat 1 to 4 by iterating aver all tiles in sequential order (see `yield_overlaps`).

    In the process

    +----------+
    |    REF   |
    +----------+
          +----------+
          |    NEI   |
          +----------+
    +-----+----+-----+
    | R   | ov |   N |
    +-----+----+-----+

    Parameters
    ----------
    tiles : adapters.TiledImage
        An tiled image.

    Returns
    -------
    np.ndarray, np.ndarray, np.ndarray, np.ndarray
        (Overlap regions in the center tile, connection matrix between tile and correction
         Overlap regions in the neighbour tile, connection matrix between tile and correction)
    """

    assert tiles.is_filled, "This routine only works for filled images"

    tsh = tiles.tile_shape
    overlap = tiles.overlap

    gsh = tiles.grid_shape
    count_tiles = gsh[0] * gsh[1]

    count_overlap_regions = (
        4 * count_tiles - 2 * gsh[0] - 2 * gsh[1]
    ) // 2

    # Number of overlap pixels in both directions.
    count_overlap_pixels0 = overlap[0] * tsh[1]
    count_overlap_pixels1 = overlap[1] * tsh[0]

    # We distinguish to cases depending if the number of pixels in the "horizontal" or "vertical"
    # overlap regions are the same or not.
    # In the second case we will need to create a larger matrix (wih a lot of zeros)
    # to keep the equation simple.
    isotropic = count_overlap_pixels0 == count_overlap_pixels1
    if isotropic:
        count_overlap_pixels = count_overlap_pixels0
    else:
        count_overlap_pixels = (
            count_overlap_pixels0 + count_overlap_pixels1
        )

    # M1 and M2 will store the overlapping regions for the reference and neighbour tile.
    M1 = np.zeros((count_overlap_pixels, count_overlap_regions))
    M2 = np.zeros((count_overlap_pixels, count_overlap_regions))

    # P1 and P2 is a convenience matrix to link the overlaping regions to the actual tile.
    P1 = np.zeros((count_overlap_regions, count_tiles))
    P2 = np.zeros((count_overlap_regions, count_tiles))

    for ndx, (ref_ndx, nei_ndx, ref_ov_im, nei_ov_im) in enumerate(
        yield_overlaps(tiles)
    ):
        ref, nei = ref_ov_im.flatten(), nei_ov_im.flatten()

        if isotropic:
            M1[:, ndx] = ref
            M2[:, ndx] = nei
        elif len(ref) == count_overlap_pixels0:
            M1[:count_overlap_pixels0, ndx] = ref
            M2[:count_overlap_pixels0, ndx] = nei
        else:
            M1[count_overlap_pixels0:, ndx] = ref
            M2[count_overlap_pixels0:, ndx] = nei

        linear_ref_ndx = np.ravel_multi_index(ref_ndx, gsh)
        linear_nei_ndx = np.ravel_multi_index(nei_ndx, gsh)

        P1[ndx, linear_ref_ndx] = 1.0
        P2[ndx, linear_nei_ndx] = 1.0

    return M1, P1, M2, P2


def estimate_corrections(
    tiles: adapters.TiledImage,
    norm_key: adapters.IntPair = (0, 0),
    norm_value: float = 1.0,
):
    """Estimate correction for each tile via total least square.

    Strategy: Solve the following equation

    M1 @ P1 @ C = M2 @ P2 @ C + B

    where matrices M1 and M2 contains the overlap region of reference and neighbour tiles.
    (reference and neighbour are arbitrary as one tile that is a reference in one case,
    will be a neighbour in another, see `build_overlap_matrix`)
    and C is a vector containing the correction factor for each tile.
    As given any solution C, a scaled vector will be solution as well (including the null solution),
    we use B to pin the element norm_key to norm_value.

    Parameters
    ----------
    tiles : adapters.TiledImage
        An tiled image.
    norm_key : (int, int)
    norm_value : float

    Returns
    -------
    dict[(int, int), float]
    """
    M1, P1, M2, P2 = build_overlap_matrix(tiles)

    newM = np.zeros((M1.shape[0] + 1, P1.shape[1]))
    newM[:-1, :] = M1 @ P1 - M2 @ P2
    newM[
        -1, np.ravel_multi_index(norm_key, tiles.grid_shape)
    ] = norm_value

    b = np.zeros(newM.shape[0])
    b[-1] = 1

    sol, residuals, *_ = np.linalg.lstsq(newM, b, rcond=None)

    return {
        np.unravel_index(k, tiles.grid_shape): sol[k]
        for k in range(sol.size)
    }
