"""
    bigimtools.adapters
    ~~~~~~~~~~~~~~~~~~~

    Classes to wrap data objects like HDF5, numpy arrays and memmaped numpy arrays.

    :copyright: 2021 by bigimtools Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""
from __future__ import annotations

import pathlib

import numpy as np
import PIL.Image

from . import tiler


def check_pair(item):
    if (
        isinstance(item, tuple)
        and len(item) == 2
        and all(isinstance(el, int) for el in item)
    ):
        return item

    raise ValueError(
        f"{item} is not a valid item, should be (int, int)."
    )


class TiledH5PY:
    def __init__(self, content):
        self.content = content

    def __getitem__(self, item):
        return self.content["_".join(str(i) for i in check_pair(item))]

    def __setitem__(self, item, value):
        self.content["_".join(str(i) for i in check_pair(item))] = value
    
    def __len__(self):
        return(len(self.content))
    
    def keys(self):
        def _key_to_tuple(key):
            return tuple(int(string) for string in key.split("_"))
        for key in self.content.keys():
            yield _key_to_tuple(key)

    def values(self):
        return self.content.values()

    def items(self):
        for k in self.keys():
            yield (k, self[k])
    
    def get(self, item, value=None):
        try:
            return self[item]
        except KeyError:
            return value

class TiledFolder:
    """A tile ndarray maps all data to a

    Parameters
    ----------
    folder : str or pathlib.Path
        folder in which tiles are going to be saved.
    ext : str
        file extension (png or jpg)
    """

    def __init__(self, folder, ext=".png"):
        self.folder = pathlib.Path(folder)
        self.ext = ext

    def __getitem__(self, item):
        with self.folder.joinpath(
            "_".join(check_pair(item)) + self.ext
        ).open("r") as fi:
            return PIL.Image.open(fi)

    def __setitem__(self, item, value):
        with self.folder.joinpath(
            "_".join(check_pair(item)) + self.ext
        ).open("w") as fo:
            value.save(fo)

    def keys(self):
        for p in self.folder.iterdir():
            yield p.stem.split("_")


class TiledNDArray:
    """A tile ndarray maps all data to a single array.

    Parameters
    ----------
    wall_shape : (int, int)
        shape of the tile container.
    content : np.ndarray
    """

    def __init__(self, wall_shape: (int, int), content: np.ndarray):
        self.wall_shape = wall_shape
        self.content = content

    @property
    def tile_shape(self):
        return self.content.shape[1:]

    @property
    def dtype(self):
        return self.content.dtype

    @classmethod
    def from_file(
        cls,
        wall_shape: (int, int),
        tile_shape: (int, int),
        dtype,
        filename,
        mode,
    ):
        """Create or use a memmaped array.

        See https://numpy.org/doc/stable/reference/generated/numpy.memmap.html

        Parameters
        ----------
        wall_shape : (int, int)
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
        shape = np.prod(wall_shape) + tile_shape
        return cls(
            wall_shape,
            np.memmap(filename, dtype=dtype, shape=shape, mode=mode),
        )

    @classmethod
    def from_tiles(cls, tiles: dict[(int, int), np.ndarray]):
        """Creates from tile dict.

        Parameters
        ----------
        tiles : dict[(int, int), np.ndarray]
            maps tile indices to tile.
        """
        wall_shape, tile_shape, dtype = tiler.tiledict_info(tiles)
        shape = np.prod(wall_shape) + tile_shape
        content = np.ndarray(shape, dtype=dtype)
        return cls(wall_shape, content)

    def __getitem__(self, item):
        ndx = np.ravel_multi_index(check_pair(item), self.wall_shape)
        return self.content[ndx, :, :]

    def __setitem__(self, item, value):
        ndx = np.ravel_multi_index(check_pair(item), self.wall_shape)
        self.content[ndx, :, :] = value
