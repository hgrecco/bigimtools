import pathlib

import numpy as np
import PIL.Image
import pytest
from xmldiff.main import diff_files

from bigimtools import dzi


def compare_dz(
    exp_path: pathlib.Path, res_path: pathlib.Path, stem: str
):

    with (res_path / f"{stem}.dzi").open("rb") as f_result:
        with (exp_path / f"{stem}.dzi").open("rb") as f_expected:
            assert not diff_files(f_result, f_expected), f"{stem}.dzi"

    # To ease debug, first check existence
    for p in (exp_path / f"{stem}_files/").iterdir():
        for q in p.iterdir():
            full_path = res_path / (stem + "_files") / p.name / q.name
            assert full_path.exists(), f"{full_path}"

    # Then size
    for p in (exp_path / f"{stem}_files/").iterdir():
        for q in p.iterdir():
            with q.open("rb") as fi:
                im1 = PIL.Image.open(fi)
                im1.load()
            im2 = PIL.Image.open(
                (res_path / (stem + "_files") / p.name / q.name).open(
                    "rb"
                )
            )
            assert im2.size == im1.size, q
            # import matplotlib.pyplot as plt
            # fig, (ax1, ax2) = plt.subplots(2, 1)
            # plt.suptitle((p.name, q.name))
            # ax1.imshow(im1)
            # ax2.imshow(im2)
            # plt.show()
            # print("")

    # Finally content, but cannot check for equality because
    # different shrinkage algorithm image might have been used.
    for p in (exp_path / f"{stem}_files/").iterdir():
        for q in p.iterdir():
            with q.open("rb") as fi:
                im1 = PIL.Image.open(fi)
                im1.load()
            im2 = PIL.Image.open(
                (res_path / (stem + "_files") / p.name / q.name).open(
                    "rb"
                )
            )
            im1 = np.array(im1).flatten()
            im2 = np.array(im2).flatten()
            sel = np.logical_and(np.isfinite(im1), np.isfinite(im2))
            cc = np.corrcoef(im1[sel], im2[sel])[1, 0]
            if np.isnan(cc):
                continue
            assert cc > 0.95, q


@pytest.mark.parametrize("tile_size", (64, 128, 256))
@pytest.mark.parametrize("overlap", (0, 16, 32))
def test_dzi_from_image(
    imcamera, camera_dzi, tmpdir_factory, tile_size, overlap
):

    stem = f"camera_{tile_size}_{overlap}"

    tmpfolder = tmpdir_factory.mktemp("dzi")

    output = tmpfolder.join(stem + ".dzi")
    dzi.from_image(
        imcamera,
        output,
        tile_size=tile_size,
        overlap=overlap,
        rescale=dzi.Rescale.NONE,
        fmt=dzi.Format.PNG8,
    )

    compare_dz(camera_dzi, pathlib.Path(tmpfolder), stem)


@pytest.mark.parametrize("tile_size", (64,))
@pytest.mark.parametrize("overlap", (0,))
def test_dzi_from_tiles(camera_dzi, tmpdir_factory, tile_size, overlap):

    stem = f"camera_{tile_size}_{overlap}"

    tmpfolder = tmpdir_factory.mktemp("dzi")

    tiles = {}
    for q in (camera_dzi / f"{stem}_files/9/").iterdir():
        with q.open("rb") as fi:
            tiles[
                tuple(int(el) for el in q.name[:-4].split("_"))
            ] = np.array(PIL.Image.open(fi))

    output = tmpfolder.join(stem + ".dzi")
    dzi.from_tiles(
        tiles,
        output,
        overlap=overlap,
        rescale=dzi.Rescale.NONE,
        fmt=dzi.Format.PNG8,
    )

    compare_dz(camera_dzi, pathlib.Path(tmpfolder), stem)
