import importlib

import nibabel as nib
import numpy as np
import pytest

from dwitracts import utils


@pytest.mark.parametrize("module", ["main", "utils", "layout", "glm", "plot"])
def test_imports(module):
    importlib.import_module(f"dwitracts.{module}")


def test_run_fsl_without_fsl_setup(tmp_path, monkeypatch):
    # with no fsl.sh where FSLDIR points, the command must still run
    monkeypatch.setenv("FSLDIR", str(tmp_path / "nofsl"))
    out = tmp_path / "ran"
    utils.run_fsl(f"touch {out}")
    assert out.exists()


def test_run_fsl_sources_fsl_setup(tmp_path, monkeypatch):
    setup = tmp_path / "fsl" / "etc" / "fslconf" / "fsl.sh"
    setup.parent.mkdir(parents=True)
    setup.write_text("export SOURCED=yes\n")
    monkeypatch.setenv("FSLDIR", str(tmp_path / "fsl"))
    out = tmp_path / "env"
    utils.run_fsl(f"echo $SOURCED > {out}")
    assert out.read_text().strip() == "yes"


def test_smooth_volume():
    volume = np.zeros((9, 9, 9))
    volume[4, 4, 4] = 1.0
    img = nib.Nifti1Image(volume, np.eye(4))
    smoothed = utils.smooth_volume(volume, img, 2.0)
    assert smoothed.shape == volume.shape
    assert smoothed.sum() == pytest.approx(1.0, rel=1e-3) and smoothed[4, 4, 4] < 1.0
