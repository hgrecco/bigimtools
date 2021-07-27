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
import PIL.Image

from . import tiler

NS_DEEPZOOM = "http://schemas.microsoft.com/deepzoom/2008"


class ResizeFilters(enum.Enum):
    CUBIC = PIL.Image.CUBIC
    BILINEAR = PIL.Image.BILINEAR
    BICUBIC = PIL.Image.BICUBIC
    NEAREST = PIL.Image.NEAREST
    ANTIALIAS = PIL.Image.ANTIALIAS


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
    fmt="png",
    jpeg_image_quality=0.8,
):
    """
    Creates Deep Zoom image from source file and saves it to destination.

    Parameters
    ----------
    source : str or PIL.Image or ndarray
        Source file of an image or image
    destination : str
        Destination dzi image.
    tile_size : int
        Size of a tile.
    overlap : int
        Overlap between tiles.
    fmt : str
        jpg, png
    resize_filter : ResizeFilters
    jpeg_image_quality : float
        clampled between 0 and 1
    """

    if isinstance(source, np.ndarray):
        img = PIL.Image.fromarray(source)
    elif isinstance(source, PIL.Image.Image):
        img = source
    else:
        img = PIL.Image.open(source)

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

    sz0, sz1 = img.size
    descriptor = DeepZoomImageDescriptor(
        width=sz0,
        height=sz1,
        tile_size=tile_size,
        tile_overlap=overlap,
        tile_format=fmt,
    )

    if not (0 <= jpeg_image_quality <= 1):
        raise ValueError(
            f"jpeg_image_quality has to be in range (0, 1), "
            f"not {jpeg_image_quality}"
        )

    jpeg_image_quality = int(jpeg_image_quality * 100)

    image_files = _get_or_create_path(_get_files_path(destination))

    for level in range(descriptor.num_levels):
        level_dir = _get_or_create_path(
            os.path.join(image_files, str(level))
        )
        level_width, level_height = descriptor.get_dimensions(level)

        if level_width == sz0 and level_height == sz1:
            level_image = img
        else:
            level_image = img.resize(
                (level_width, level_height), resize_filter.value
            )

        for (ndx0, ndx1) in descriptor.get_tiles(level):
            bounds = descriptor.get_tile_bounds(level, ndx0, ndx1)
            tile = level_image.crop(bounds)

            tile_path = os.path.join(
                level_dir, "%s_%s.%s" % (ndx0, ndx1, fmt)
            )

            with open(tile_path, "wb") as tile_file:
                if fmt == "jpg":
                    tile.save(
                        tile_file, "JPEG", quality=jpeg_image_quality
                    )
                else:
                    tile.save(tile_file)

    descriptor.to_file(destination)


def from_tiles(
    tiles: dict[(int, int), np.ndarray],
    destination,
    overlap=1,
    resize_filter=ResizeFilters.ANTIALIAS,
    fmt="png",
    jpeg_image_quality=0.8,
):
    """
    Creates Deep Zoom image from tiles and saves it to destination.

    Parameters
    ----------
    tiles : dict[(int, int), np.ndarray)
        maps tile indices to tile.
    destination : str
        Destination dzi image.
    overlap : int
        Overlap between tiles.
    fmt : str
        jpg, png
    resize_filter : ResizeFilters
    jpeg_image_quality : float
        clampled between 0 and 1
    """

    if isinstance(overlap, (tuple, list)):
        if len(overlap) != 2 or overlap[0] != overlap[1]:
            raise ValueError(
                "If 'overlap' is tuple, "
                "it must contain two equal integers."
            )
        overlap = overlap[0]

    wall_shape, tile_size, dtype = tiler.tiledict_info(tiles)

    if isinstance(tile_size, (tuple, list)):
        if len(tile_size) != 2 or tile_size[0] != tile_size[1]:
            raise ValueError("Only squared tiles are accepted.")
        tile_size = tile_size[0]

    tsz0 = tsz1 = tile_size
    ov0 = ov1 = overlap

    wsz0, wsz1 = wall_shape

    if wsz0 != wsz1:
        raise ValueError("Only squared walls are accepted")

    sz0, sz1 = wsz0 * (tsz0 - ov0), wsz1 * (tsz1 - ov1)

    if not math.log(sz0 / (tsz0 - ov0), 2).is_integer():
        raise ValueError(
            "Only images and divided into a power of 2 tiles are currently accepted."
        )

    descriptor = DeepZoomImageDescriptor(
        width=sz0,
        height=sz1,
        tile_size=tile_size,
        tile_overlap=overlap,
        tile_format=fmt,
    )

    if not (0 <= jpeg_image_quality <= 1):
        raise ValueError(
            f"jpeg_image_quality has to be in range (0, 1), "
            f"not {jpeg_image_quality}"
        )

    jpeg_image_quality = int(jpeg_image_quality * 100)

    image_files = _get_or_create_path(_get_files_path(destination))

    for level in reversed(range(descriptor.num_levels)):
        level_dir = _get_or_create_path(
            os.path.join(image_files, str(level))
        )

        for (ndx0, ndx1) in descriptor.get_tiles(level):
            tile_path = os.path.join(
                level_dir, "%s_%s.%s" % (ndx0, ndx1, fmt)
            )

            tile = PIL.Image.fromarray(tiles[(ndx0, ndx1)])

            with open(tile_path, "wb") as tile_file:
                if fmt == "jpg":
                    tile.save(
                        tile_file, "JPEG", quality=jpeg_image_quality
                    )
                else:
                    tile.save(tile_file)

        if level == 0:
            break

        # Here we build the level with less resolution by joining 4 images of the level with more resolution.
        level_dict = {}
        level_width, level_height = descriptor.get_dimensions(level - 1)
        for (ndx0, ndx1) in descriptor.get_tiles(level - 1):
            if len(tiles) == 1:
                level_dict[(0, 0)] = _resize(
                    tiles[(0, 0)],
                    (level_width, level_height),
                    resize_filter.value,
                )
            else:
                parts = {
                    (n, m): tiles[(2 * ndx0 + m, 2 * ndx1 + n)]
                    for (m, n) in ((0, 0), (0, 1), (1, 0), (1, 1))
                }

                joined = tiler.join_tiles(parts, overlap)
                level_dict[(ndx0, ndx1)] = _resize(
                    joined, (tile_size, tile_size), resize_filter.value
                )

        tiles = level_dict

    descriptor.to_file(destination)


def _get_or_create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def _get_files_path(path):
    return os.path.splitext(path)[0] + "_files"


def _remove(path):
    os.remove(path)
    tiles_path = _get_files_path(path)
    shutil.rmtree(tiles_path)


def _resize(im, shape, resample):
    pim = PIL.Image.fromarray(im)
    return np.asarray(pim.resize(shape, resample))
