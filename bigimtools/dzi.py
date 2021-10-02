"""
    bigimtools.dzi
    ~~~~~~~~~~~~~~

    Create deep zoom image.

    A DZI has two parts: a DZI file (with either a .dzi or .xml extension)
    and a subdirectory of image folders. Each folder in the image
    subdirectory is labeled with its level of resolution. Higher numbers
    correspond to a higher resolution level; inside each folder are the
    image tiles corresponding to that level of resolution, numbered
    consecutively in columns from top left to bottom right.

    :copyright: 2021 by bigimtools Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""
from __future__ import annotations

import enum
import math
import os
import shutil
import xml.dom.minidom
from dataclasses import dataclass

import numpy as np
import PIL
import skimage.io
from skimage import exposure
from skimage import io as skio

from . import adapters, tiler

NS_DEEPZOOM = "http://schemas.microsoft.com/deepzoom/2008"


class RescaleMode(enum.IntEnum):
    NONE = 0
    MIN_MAX = 1
    MAX = 2


class ImageFormat(enum.IntEnum):
    JPEG8 = 0
    PNG8 = 1
    PNG16 = 2
    PNG32 = 3


class ResizeFilters(enum.Enum):
    CUBIC = PIL.Image.CUBIC
    BILINEAR = PIL.Image.BILINEAR
    BICUBIC = PIL.Image.BICUBIC
    NEAREST = PIL.Image.NEAREST
    ANTIALIAS = PIL.Image.ANTIALIAS


def rescale_mode_to_range(obj, rescale_mode):
    """Save array to file, rescaling intensity.

    Parameters
    ----------
    obj : ndarray or dict[Any, np.ndarray]
        numpy ndarray.
    rescale_mode: RescaleMode
        Intensity rescale_mode type or (minimum, maximum) values.
    """

    if rescale_mode is RescaleMode.NONE:
        return None

    if isinstance(obj, np.ndarray):
        rng = obj.min(), obj.max()
    else:
        mn, mx = np.inf, -np.inf
        for v in obj.values():
            mn = min(mn, v.min())
            mx = max(mx, v.max())
        rng = mn, mx

    if rescale_mode is RescaleMode.MAX:
        return 0, rng[1]
    elif rescale_mode is RescaleMode.MIN_MAX:
        return rng
    else:
        raise ValueError("rescale_mode must be RescaleMode")


def save_image(arr, path, fmt, in_range, jpeg_image_quality=0.8):
    """Save array to file, rescaling intensity.

    Parameters
    ----------
    arr : ndarray
        numpy ndarray.
    path : str
        Destination file.
    fmt : ImageFormat
        File format
    in_range: (number, number) or None
        Input intensity range that will be mapped to the full dtype range or None to not rescale.
    jpeg_image_quality : float
        clampled between 0 and 1
    """

    if fmt in (ImageFormat.JPEG8, ImageFormat.PNG8):
        dtype = np.uint8
    elif fmt is ImageFormat.PNG16:
        dtype = np.uint16
    elif fmt is ImageFormat.PNG32:
        dtype = np.uint32
    else:
        raise ValueError("rescale_mode must be a ImageFormat")

    if in_range is None:
        pass
    elif isinstance(in_range, tuple) and len(in_range) == 2:
        arr = exposure.rescale_intensity(arr, in_range, dtype)
    else:
        raise ValueError(
            "rescale_mode must be a tuple of two number or None"
        )

    path = (
        str(path) + "." + ("jpg" if fmt is ImageFormat.JPEG8 else "png")
    )

    # Write empty array as a black pixel
    if len(arr) == 0:
        arr = np.array([[0]])

    if fmt is ImageFormat.JPEG8:
        skimage.io.imsave(
            path, arr, quality=jpeg_image_quality, check_contrast=False
        )
    else:
        skimage.io.imsave(path, arr, check_contrast=False)


@dataclass(frozen=True)
class DeepZoomImageDescriptor:

    width: int
    height: int
    tile_size: int = 254
    tile_overlap: int = 1
    tile_format: str = "jpg"

    @property
    def num_levels(self):
        max_dimension = max(self.width, self.height)
        return int(math.ceil(math.log(max_dimension, 2))) + 1

    @classmethod
    def from_file(cls, filename):
        """Initialize descriptor from an filename."""
        with open(filename, "rb") as fi:
            return cls.from_fp(fi)

    @classmethod
    def from_fp(cls, fp):
        """Initialize descriptor from a file descriptor."""
        doc = xml.dom.minidom.parse(fp)
        image = doc.getElementsByTagName("Image")[0]
        size = doc.getElementsByTagName("Size")[0]
        return cls(
            width=int(size.getAttribute("Width")),
            height=int(size.getAttribute("Height")),
            tile_size=int(image.getAttribute("TileSize")),
            tile_overlap=int(image.getAttribute("Overlap")),
            tile_format=image.getAttribute("Format"),
        )

    def to_file(self, filename, pretty_print_xml=False):
        """Save descriptor file."""
        doc = xml.dom.minidom.Document()
        image = doc.createElementNS(NS_DEEPZOOM, "Image")
        image.setAttribute("xmlns", NS_DEEPZOOM)
        image.setAttribute("TileSize", str(self.tile_size))
        image.setAttribute("Overlap", str(self.tile_overlap))
        image.setAttribute("Format", str(self.tile_format))
        size = doc.createElementNS(NS_DEEPZOOM, "Size")
        size.setAttribute("Width", str(self.width))
        size.setAttribute("Height", str(self.height))
        image.appendChild(size)
        doc.appendChild(image)

        if pretty_print_xml:
            descriptor = doc.toprettyxml(encoding="UTF-8")
        else:
            descriptor = doc.toxml(encoding="UTF-8")

        with open(filename, "wb") as fo:
            fo.write(descriptor)

    @classmethod
    def remove(cls, filename):
        """Remove descriptor file (DZI) and tiles folder."""
        _remove(filename)

    def get_scale(self, level):
        """Scale of a pyramid level."""
        if not 0 <= level < self.num_levels:
            raise ValueError(
                f"Invalid pyramid level ({level}), not in range [0, {self.num_levels})"
            )

        max_level = self.num_levels - 1
        return math.pow(0.5, max_level - level)

    def get_dimensions(self, level):
        """Dimensions of level (width, height)."""
        if not 0 <= level < self.num_levels:
            raise ValueError(
                f"Invalid pyramid level ({level}), not in range [0, {self.num_levels})"
            )

        scale = self.get_scale(level)
        width = int(math.ceil(self.width * scale))
        height = int(math.ceil(self.height * scale))
        return width, height

    def get_num_tiles(self, level):
        """Number of tiles (columns, rows)."""
        if not 0 <= level < self.num_levels:
            raise ValueError(
                f"Invalid pyramid level ({level}), not in range [0, {self.num_levels})"
            )

        w, h = self.get_dimensions(level)
        return (
            int(math.ceil(float(w) / self.tile_size)),
            int(math.ceil(float(h) / self.tile_size)),
        )

    def get_tile_bounds(self, level, column, row):
        """Bounding box of the tile (x1, y1, x2, y2)."""
        if not 0 <= level < self.num_levels:
            raise ValueError(
                f"Invalid pyramid level ({level}), not in range [0, {self.num_levels})"
            )

        offset_x = 0 if column == 0 else self.tile_overlap
        offset_y = 0 if row == 0 else self.tile_overlap
        x = (column * self.tile_size) - offset_x
        y = (row * self.tile_size) - offset_y
        level_width, level_height = self.get_dimensions(level)
        w = (
            self.tile_size
            + (1 if column == 0 else 2) * self.tile_overlap
        )
        h = self.tile_size + (1 if row == 0 else 2) * self.tile_overlap
        w = min(w, level_width - x)
        h = min(h, level_height - y)
        return x, y, x + w, y + h

    def get_tiles(self, level):
        """Iterator for all tiles in the given level. Returns (column, row) of a tile."""
        if not 0 <= level < self.num_levels:
            raise ValueError(
                f"Invalid pyramid level ({level}), not in range [0, {self.num_levels})"
            )

        columns, rows = self.get_num_tiles(level)
        for column in range(columns):
            for row in range(rows):
                yield column, row


def from_image(
    source,
    destination,
    tile_size=254,
    overlap=1,
    resize_filter=ResizeFilters.ANTIALIAS,
    fmt=ImageFormat.PNG32,
    rescale_mode=RescaleMode.MIN_MAX,
    jpeg_image_quality=0.8,
):
    """
    Creates Deep Zoom image from source file and saves it to destination.

    Parameters
    ----------
    source : str or ndarray
        Source file of an image or image
    destination : str
        Destination dzi image.
    tile_size : int
        Size of a tile.
    overlap : int
        Overlap between tiles.
    resize_filter : ResizeFilters
    fmt : ImageFormat
        File format
    rescale_mode: RescaleMode or (number, number)
        Rescale type o tuple of (min, max)
    jpeg_image_quality : float
        clampled between 0 and 1
    """

    if isinstance(source, str):
        img = skio.imread(source)
    else:
        img = source

    if isinstance(tile_size, (tuple, list)):
        if len(tile_size) != 2 or tile_size[0] != tile_size[1]:
            raise ValueError(
                "If 'tile_size' is a tuple, "
                "it must contain two equal integers."
            )
        tile_size = tile_size[0]

    if isinstance(overlap, (tuple, list)):
        if len(overlap) != 2 or overlap[0] != overlap[1]:
            raise ValueError(
                "If 'overlap' is tuple, "
                "it must contain two equal integers."
            )
        overlap = overlap[0]

    if isinstance(rescale_mode, tuple) and len(rescale_mode) == 2:
        in_range = rescale_mode
    else:
        in_range = rescale_mode_to_range(img, rescale_mode)

    sz0, sz1 = img.shape
    descriptor = DeepZoomImageDescriptor(
        width=sz0,
        height=sz1,
        tile_size=tile_size,
        tile_overlap=overlap,
        tile_format="jpg" if fmt is ImageFormat.JPEG8 else "png",
    )

    if not (0 <= jpeg_image_quality <= 1):
        raise ValueError(
            f"jpeg_image_quality has to be in range (0, 1), "
            f"not {jpeg_image_quality}"
        )

    image_files = _get_or_create_path(_get_files_path(destination))

    for level in range(descriptor.num_levels):
        level_dir = _get_or_create_path(
            os.path.join(image_files, str(level))
        )
        level_width, level_height = descriptor.get_dimensions(level)

        if level_width == sz0 and level_height == sz1:
            level_image = img
        else:
            level_image = _pil_resize(
                img, (level_width, level_height), resize_filter.value
            )

        for (ndx0, ndx1) in descriptor.get_tiles(level):
            tile = _crop(
                level_image,
                descriptor.get_tile_bounds(level, ndx0, ndx1),
            )

            tile_path = os.path.join(level_dir, "%s_%s" % (ndx0, ndx1))

            save_image(
                np.asarray(tile),
                tile_path,
                fmt,
                in_range,
                int(jpeg_image_quality * 100),
            )

    descriptor.to_file(destination)


def from_tiles(
    tiles: adapters.TiledImage,
    destination,
    resize_filter=ResizeFilters.ANTIALIAS,
    fmt=ImageFormat.PNG32,
    rescale_mode=RescaleMode.MIN_MAX,
    jpeg_image_quality=0.8,
):
    """
    Creates Deep Zoom image from tiles and saves it to destination.

    Parameters
    ----------
    tiles : adapters.TiledImage
        maps tile indices to tile.
    destination : str
        Destination dzi image.
    overlap : int
        Overlap between tiles.
    resize_filter : ResizeFilters
    fmt : ImageFormat
        File format
    rescale_mode: RescaleMode or (number, number)
        Rescale type o tuple of (min, max)
    jpeg_image_quality : float
        clampled between 0 and 1
    """

    overlap = tiles.overlap

    if isinstance(overlap, (tuple, list)):
        if len(overlap) != 2 or overlap[0] != overlap[1]:
            raise ValueError(
                "If 'overlap' is tuple, "
                "it must contain two equal integers."
            )
        overlap = overlap[0]

    grid_shape, tile_shape = (tiles.grid_shape, tiles.tile_shape)

    if isinstance(tile_shape, (tuple, list)):
        if len(tile_shape) != 2 or tile_shape[0] != tile_shape[1]:
            raise ValueError("Only squared tiles are accepted.")
        tile_shape = tile_shape[0]

    if isinstance(rescale_mode, tuple) and len(rescale_mode) == 2:
        in_range = rescale_mode
    else:
        in_range = rescale_mode_to_range(tiles, rescale_mode)

    tsz0 = tsz1 = tile_shape
    ov0 = ov1 = overlap

    gsz0, gsz1 = grid_shape

    if gsz0 != gsz1:
        raise ValueError("Only squared grids are accepted")

    sz0, sz1 = gsz0 * (tsz0 - ov0), gsz1 * (tsz1 - ov1)

    if not math.log(sz0 / (tsz0 - ov0), 2).is_integer():
        raise ValueError(
            "Only images and divided into a power of 2 tiles are currently accepted."
        )

    descriptor = DeepZoomImageDescriptor(
        width=sz0,
        height=sz1,
        tile_size=tile_shape,
        tile_overlap=overlap,
        tile_format="jpg" if fmt is ImageFormat.JPEG8 else "png",
    )

    if not (0 <= jpeg_image_quality <= 1):
        raise ValueError(
            f"jpeg_image_quality has to be in range (0, 1), "
            f"not {jpeg_image_quality}"
        )

    image_files = _get_or_create_path(_get_files_path(destination))

    for level in reversed(range(descriptor.num_levels)):
        level_dir = _get_or_create_path(
            os.path.join(image_files, str(level))
        )

        for (ndx0, ndx1) in descriptor.get_tiles(level):
            tile_path = os.path.join(level_dir, "%s_%s" % (ndx0, ndx1))

            save_image(
                tiles[(ndx0, ndx1)],
                tile_path,
                fmt,
                in_range,
                int(jpeg_image_quality * 100),
            )

        if level == 0:
            break

        # Here we build the level with less resolution by joining 4 images of the level with more resolution.
        level_dict = {}
        level_width, level_height = descriptor.get_dimensions(level - 1)
        for (ndx0, ndx1) in descriptor.get_tiles(level - 1):
            if len(tiles) == 1:
                level_dict[(0, 0)] = _pil_resize(
                    tiles[(0, 0)],
                    (level_width, level_height),
                    resize_filter.value,
                )
            else:
                parts = {
                    (n, m): tiles[(2 * ndx0 + m, 2 * ndx1 + n)]
                    for (m, n) in ((0, 0), (0, 1), (1, 0), (1, 1))
                }

                joined = tiler.join_tiles(
                    adapters.TiledImage.from_dict(parts, overlap)
                )
                level_dict[(ndx0, ndx1)] = _pil_resize(
                    joined,
                    (tile_shape, tile_shape),
                    resize_filter.value,
                )

        tiles = level_dict

    descriptor.to_file(destination)


def _pil_resize(
    img: np.ndarray,
    size: tuple[(int, int)],
    resize_filter=ResizeFilters.ANTIALIAS,
):
    """Helper function to leverage the efficiency of PIL resize against the practicality of
    working with arrays and scikit-image."""
    return np.array(
        PIL.Image.fromarray(img).resize(size, resize_filter)
    )


def _get_or_create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def _get_files_path(path):
    return os.path.splitext(path)[0] + "_files"


def _crop(arr, bounds):
    a, b, c, d = bounds
    return arr[b:d, a:c]


def _remove(path):
    os.remove(path)
    tiles_path = _get_files_path(path)
    shutil.rmtree(tiles_path)
