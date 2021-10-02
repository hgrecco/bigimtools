"""
    bigimtools.adapters
    ~~~~~~~~~~~~~~~~~~~

    Classes to wrap data objects like HDF5, numpy arrays and memmaped numpy arrays.

    :copyright: 2021 by bigimtools Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""
from __future__ import annotations

import functools
import json
import pathlib

import numpy as np
from skimage import io as skio

ALL_INTS = (int, np.integer)

IntPair = tuple[int, int]


def check_pair(item):
    """Check that item has 2 integer elements >= 0."""
    if (
        isinstance(item, tuple)
        and len(item) == 2
        and item[0] >= 0
        and item[1] >= 0
    ):
        return item

    raise ValueError(
        f"{item} is not a valid item, should be (int, int)."
    )


def are_all_whole_numbers(iterable):
    """True if all numbers are integers >= 0."""
    return all(
        (isinstance(el, ALL_INTS) and el >= 0) for el in iterable
    )


def check_all_whole_numbers(iterable):
    """Check that all iterables contain integers >= 0 and raises ValueError if None."""
    for el in iterable:
        if not are_all_whole_numbers(el):
            raise ValueError(
                el, [(isinstance(x, int), x >= 0, type(x)) for x in el]
            )


def normalize_overlap(value) -> IntPair:
    """Normalize overlap, raising an exception.

    a: int -> (a, a)
    (a: int, b: int) -> (a, b)
    [a: int, b: int] -> (a, b)
    """
    if isinstance(value, (tuple, list)):
        return check_pair(tuple(value))
    elif isinstance(value, int):
        return value, value
    else:
        raise TypeError(
            f"overlap must be tuple, list, int or str, not {type(value)}"
        )


class OverlapMixin:
    """Mixin class to handle overlapping tiles.

    The derived class must contain an _overlap attribute and be a dict like object.
    """

    overlap: IntPair

    def get(self, item, value=None):
        raise NotImplementedError

    def _shift_to_slices(self, shift: IntPair):
        ov0, ov1 = self.overlap
        if shift == (+1, 0):
            return np.s_[-ov0:, :], np.s_[:+ov0, :]
        elif shift == (-1, 0):
            return np.s_[:+ov0, :], np.s_[-ov0:, :]
        elif shift == (0, +1):
            return np.s_[:, -ov1:], np.s_[:, :+ov1]
        elif shift == (0, -1):
            return np.s_[:, :+ov1], np.s_[:, -ov1:]
        else:
            raise ValueError(f"Cannot translate {shift} to slices")

    def get_overlap_regions(self, center: IntPair, shift: IntPair):
        other_ndx = center[0] + shift[0], center[1] + shift[1]
        if other_ndx not in self:
            return other_ndx, None, None
        s1, s2 = self._shift_to_slices(shift)
        return other_ndx, self.get(center)[s1], self.get(other_ndx)[s2]

    def yield_all_overlap_regions(self, center: IntPair):
        yield self.get_overlap_regions(center, (+1, 0))
        yield self.get_overlap_regions(center, (-1, 0))
        yield self.get_overlap_regions(center, (0, +1))
        yield self.get_overlap_regions(center, (0, -1))


class TiledMixin:
    """Mixin class to handle tiled images

    The derived class must be a dict like object.
    """

    def keys(self):
        raise NotImplementedError

    def values(self):
        raise NotImplementedError

    def items(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def clear_tiled_mixing_cache(self):
        """Clear all cached values. Call this method when mutating the dict."""
        if hasattr(self, "grid_shape"):
            del self.grid_shape
        if hasattr(self, "tile_shape"):
            del self.tile_shape
        if hasattr(self, "tile_shapes"):
            del self.tile_shapes
        if hasattr(self, "dtype"):
            del self.dtype
        if hasattr(self, "dtypes"):
            del self.dtypes
        if hasattr(self, "is_index_compatible"):
            del self.is_index_compatible
        if hasattr(self, "is_homogeneous"):
            del self.is_homogeneous
        if hasattr(self, "is_filled"):
            del self.is_filled

    @functools.cached_property
    def grid_shape(self) -> IntPair:
        """Shape of grid which contain the tiles."""
        ndx0s, ndx1s = zip(*self.keys())
        return len(np.unique(ndx0s)), len(np.unique(ndx1s))

    @functools.cached_property
    def tile_shape(self) -> IntPair:
        """Shape of a tile in a homogeneous."""
        if not self.is_homogeneous:
            raise ValueError(
                "This operation is only valid for homogeneous dataset."
            )
        return self.tile_shapes[0]

    @functools.cached_property
    def tile_shapes(self) -> tuple[IntPair, ...]:
        """Iterable of shapes."""
        return tuple(set(el.shape for el in self.values()))

    @functools.cached_property
    def dtype(self) -> type:
        """Shape of a tile in a homogeneous dataset."""
        if not self.is_homogeneous:
            raise ValueError(
                "This operation is only valid for homogeneous dataset."
            )
        return self.dtypes[0]

    @functools.cached_property
    def dtypes(self) -> tuple[type, ...]:
        """Iterable of shapes."""
        return tuple(set(el.dtype for el in self.values()))

    @functools.cached_property
    def is_index_compatible(self) -> bool:
        """True if all indices are >= 0."""
        ndx0s, ndx1s = zip(*self.keys())
        if not are_all_whole_numbers(ndx0s):
            return False
        if not are_all_whole_numbers(ndx1s):
            return False
        return True

    @functools.cached_property
    def is_homogeneous(self) -> bool:
        """ "True if all tiles have the same shape."""
        return len(self.tile_shapes) == 1

    @functools.cached_property
    def is_filled(self) -> bool:
        """True if the no tiles are missing."""
        return len(self) == np.prod(self.grid_shape)

    def is_compatible_tile(self, tile) -> bool:
        """True if the tile shape is identical with pre-existing tiles.

        True also if no tiles are present.
        """
        if len(self):
            return self.tile_shape == tile.shape
        return True

    def reduce_to_dict(self, func) -> dict[IntPair, object]:
        """Applies func to each"""
        return {k: func(v) for k, v in self.items()}

    def reduce(self, func) -> np.ndarray:
        if not self.is_index_compatible:
            raise ValueError(
                "This function is only available for index compatible objects."
            )
        out = np.zeros(self.grid_shape)
        for k, v in self.reduce_to_dict(func):
            out[k] = v
        return out


class TiledImage(OverlapMixin, TiledMixin):
    """A tiled image with overlap which maps each tile to an entry in a dict.

    Parameters
    ----------
    backend : dict-like object
        a backend to read and write tiles.
    """

    def __init__(self, backend):
        self.backend = backend

    @property
    def overlap(self) -> IntPair:
        return self.backend.get_overlap()

    def __contains__(self, item):
        return item in self.backend

    def __getitem__(self, item):
        return self.backend[item]

    def __setitem__(self, item, value):
        self.backend[item] = value
        self.clear_tiled_mixing_cache()

    def __len__(self):
        return sum(1 for _ in self.backend.keys())

    def keys(self):
        yield from self.backend.keys()

    def values(self):
        yield from self.backend.values()

    def items(self):
        yield from self.backend.items()

    def get(self, item, value=None):
        try:
            return self[item]
        except KeyError:
            return value

    @classmethod
    def from_dict(cls, tiles, overlap):
        db = DictBackend(tiles)
        db.set_overlap(overlap)
        return cls(db)


###########
# Backends
###########


class DictBackend:
    """Store each tile as an entry in a dictionary and the overlap as well.

    Parameters
    ----------
    backend : an opened HDF5 file.
    overlap : int, (int, int) or str
        number of pixels of a square overlap region
        or shape of a overlap region
        or key to get the overlap from the backend.
    """

    __overlap_key = "_overlap"

    def __init__(self, content):
        self.content = content

    def get_overlap(self):
        try:
            return normalize_overlap(self.content[self.__overlap_key])
        except KeyError:
            raise Exception(
                f"No key `{self.__overlap_key}` found to get the overlap value."
            )

    def set_overlap(self, value):
        self.content[self.__overlap_key] = normalize_overlap(value)

    def _from_key(self, key):
        return key

    def _to_key(self, s):
        return s

    def __contains__(self, item):
        return item in self.content

    def __getitem__(self, item):
        return self.content[self._to_key(item)]

    def __setitem__(self, key, value):
        self.content[self._to_key(key)] = value

    def raw_keys(self):
        yield from (
            key
            for key in self.content.keys()
            if key != self.__overlap_key
        )

    def keys(self):
        yield from (self._from_key(key) for key in self.raw_keys())

    def values(self):
        yield from (self.content[key] for key in self.raw_keys())

    def items(self):
        yield from (
            (self._from_key(key), self.content[key])
            for key in self.raw_keys()
        )


class H5PYBackend(DictBackend):
    def _from_key(self, key):
        return tuple(int(string) for string in key.split("_"))

    def _to_key(self, s):
        return "_".join(str(i) for i in check_pair(s))


class FolderBackend:
    """A tile ndarray maps all data to images in a folder.

    Parameters
    ----------
    folder : str or pathlib.Path
        folder in which tiles are going to be saved.
    ext : str
        file extension (png or jpg)
    """

    __overlap_key = "_overlap.json"

    def __init__(self, folder, ext=".png"):
        self.folder = pathlib.Path(folder)
        self.ext = ext

    def get_overlap(self):
        try:
            with self.folder.joinpath(self.__overlap_key).open(
                "r", encoding="utf-8"
            ) as fi:
                return normalize_overlap(json.load(fi))
        except KeyError:
            raise Exception(
                f"No file `{self.__overlap_key}` found to get the overlap value."
            )

    def set_overlap(self, value):
        with self.folder.joinpath(self.__overlap_key).open(
            "w", encoding="utf-8"
        ) as fo:
            return json.dump(value, fo)

    def _from_path(self, path):
        return tuple(int(string) for string in path.stem.split("_"))

    def _to_path(self, s):
        return "_".join(str(i) for i in check_pair(s)) + self.ext

    def __contains__(self, item):
        return self.folder.joinpath(self._to_path(item)).exists()

    def __getitem__(self, item):
        with self.folder.joinpath(self._to_path(item)).open("rb") as fi:
            return skio.imread(fi)

    def __setitem__(self, item, value):
        with self.folder.joinpath(self._to_path(item)).open("wb") as fo:
            skio.imsave(fo, value)

    def raw_keys(self):
        for p in self.folder.iterdir():
            if p.name.startswith("_"):
                continue
            if p.name.endswith(self.ext):
                continue
            yield self._from_path(p)

    def keys(self):
        yield from (self._from_path(key) for key in self.raw_keys())

    def values(self):
        yield from (self[key] for key in self.raw_keys())

    def items(self):
        yield from (
            (self._from_path(key), self[key]) for key in self.raw_keys()
        )


class NDArrayBackend:
    """A tile ndarray maps all data to a single array.

    Parameters
    ----------
    content : np.ndarray
    overlap : (int, int)
        overlap between neighouring tiles
    grid_shape : (int, int)
        shape of the tile container.

    """

    def __init__(
        self, content: np.ndarray, overlap: IntPair, grid_shape: IntPair
    ):
        self.content = content
        self.overlap = normalize_overlap(overlap)
        self._grid_shape = grid_shape
        if np.prod(self._grid_shape) != self.content.shape[0]:
            raise ValueError(
                f"The grid shape {self._grid_shape} is not compatible with {self.content.shape[0]}"
            )

    def get_overlap(self):
        return self.overlap

    def set_overlap(self, value):
        self.overlap = normalize_overlap(value)

    def __contains__(self, item):
        return 0 <= self._to_linear_index(item) < self.content.shape[0]

    def __getitem__(self, item):
        return self.content[self._to_linear_index(item), :, :]

    def __setitem__(self, item, value):
        self.content[self._to_linear_index(item), :, :] = value

    def _from_linear_index(self, ndx):
        return np.unravel_index(ndx, self.grid_shape)

    def _to_linear_index(self, item):
        return np.ravel_multi_index(check_pair(item), self.grid_shape)

    def raw_keys(self):
        yield from range(self.content.shape[0])

    def keys(self):
        yield from (
            self._from_linear_index(key) for key in self.raw_keys()
        )

    def values(self):
        yield from (self[key] for key in self.raw_keys())

    def items(self):
        yield from (
            (self._from_linear_index(key), self[key])
            for key in self.raw_keys()
        )

    @property
    def grid_shape(self):
        return self._grid_shape

    @property
    def tile_shape(self):
        return self.content.shape[1:]

    @property
    def dtype(self):
        return self.content.dtype

    @classmethod
    def from_file(
        cls,
        grid_shape: IntPair,
        tile_shape: IntPair,
        dtype,
        filename,
        mode,
        overlap: IntPair,
    ):
        """Create or use a memmaped array.

        See https://numpy.org/doc/stable/reference/generated/numpy.memmap.html

        Parameters
        ----------
        grid_shape : (int, int)
            shape of the tile container.
        tile_shape : (int, int)
            shape of each tile.
        dtype : dtype
            dtype of the tiles.
        filename : str
            The file name or file object to be used as the array data buffer.
        mode : str
            open mode. See numpy.memmap.
        """
        shape = np.prod(grid_shape) + tile_shape
        return cls(
            np.memmap(filename, dtype=dtype, shape=shape, mode=mode),
            overlap,
            grid_shape,
        )

    @classmethod
    def from_tiles(cls, tiles: TiledImage, overlap: IntPair):
        """Creates from tile dict.

        Parameters
        ----------
        tiles : dict[(int, int), np.ndarray]
            maps tile indices to tile.
        overlap : (int, int)
            length of the overlap in each dimension.
        """
        shape = np.prod(tiles.grid_shape) + tiles.tile_shape
        backend = np.ndarray(shape, dtype=tiles.dtype)
        return cls(backend, overlap, tiles.grid_shape)
