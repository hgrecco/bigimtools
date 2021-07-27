import os
import zipfile
from zipfile import ZipFile

import numpy as np
import PIL
import pytest

# I need to put it here so it is not closed.
CAMERA_ZIP = ZipFile(
    os.path.join(os.path.dirname(__file__), "camera_dzis.zip"), "r"
)


@pytest.fixture(scope="session")
def camera_dzi():
    return zipfile.Path(CAMERA_ZIP)


@pytest.fixture(scope="session")
def imcamera(camera_dzi):
    with (camera_dzi / "original.png").open("rb") as fi:
        source = PIL.Image.open(fi)
        source.load()
    return source


@pytest.fixture(scope="session")
def npcamera(imcamera):
    return np.asarray(imcamera)
