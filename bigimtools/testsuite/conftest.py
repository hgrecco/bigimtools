import os
import zipfile
from zipfile import ZipFile

import pytest

# I need to put it here so it is not closed.
import skimage.io as skio

CAMERA_ZIP = ZipFile(
    os.path.join(os.path.dirname(__file__), "camera_dzis.zip"), "r"
)


@pytest.fixture(scope="session")
def camera_dzi():
    return zipfile.Path(CAMERA_ZIP)


@pytest.fixture(scope="session")
def imcamera(camera_dzi):
    with (camera_dzi / "original.png").open("rb") as fi:
        return skio.imread(fi)
