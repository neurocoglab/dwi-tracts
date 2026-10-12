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


# ----------------------------------------------------------------------------- trajectory voxels

SHAPE = (10, 10, 10)


def _candidates(points_values):
    """Candidates as main.py passes them: values and flat indices in ascending index order."""
    idx = np.array([np.ravel_multi_index(p, SHAPE) for p, _ in points_values])
    order = np.argsort(idx)
    return np.array([v for _, v in points_values], dtype=float)[order], idx[order]


def _original(T, T_idx, prev, max_seg_length):
    """The loop main.py used before choose_trajectory_voxel (with a stable sort)."""
    sidx = T.argsort(kind='stable')[::-1]
    if prev is None:
        return np.unravel_index(T_idx[sidx[0]], SHAPE), True
    for r in range(len(T)):
        idx_r = np.unravel_index(T_idx[sidx[r]], SHAPE)
        if np.linalg.norm(np.subtract(idx_r, prev)) < max_seg_length:
            return idx_r, True
    return np.unravel_index(T_idx[sidx[0]], SHAPE), False


def test_stable_matches_the_original_loop():
    rng = np.random.default_rng(0)
    for _ in range(300):
        T_idx = np.sort(rng.choice(1000, size=rng.integers(1, 40), replace=False))
        T = rng.integers(0, 4, size=T_idx.size) / 3.0  # values in thirds: many ties
        prev = tuple(rng.integers(0, 10, size=3)) if rng.random() < 0.8 else None
        expected = _original(T, T_idx, prev, 4.0)
        got = utils.choose_trajectory_voxel(T, T_idx, SHAPE, prev, None, 4.0, None, 'stable')
        assert got == (tuple(int(i) for i in expected[0]), expected[1])


def test_tie_breaks():
    prev2, prev = (2, 5, 5), (3, 5, 5)  # heading +x
    ahead, side, near_side = (4, 5, 5), (3, 7, 5), (3, 6, 5)
    T, T_idx = _candidates([(ahead, 1.0), (side, 1.0), (near_side, 1.0), ((4, 4, 5), 0.5)])
    choose = lambda tb, **kw: utils.choose_trajectory_voxel(T, T_idx, SHAPE, prev, prev2, 10, None, tb, **kw)[0]
    assert choose('direction') == ahead              # straight on
    assert choose('nearest') in (ahead, near_side)   # both 1 voxel away; highest flat index then
    assert choose('nearest') == max((ahead, near_side), key=lambda p: np.ravel_multi_index(p, SHAPE))
    assert choose('stable') == max((ahead, side, near_side), key=lambda p: np.ravel_multi_index(p, SHAPE))
    centre = np.mean([ahead, side, near_side], axis=0)
    assert choose('centroid') == min((ahead, side, near_side), key=lambda p: np.sum((np.array(p) - centre) ** 2))
    # without a direction yet, 'direction' takes the voxel nearest the previous point
    first = utils.choose_trajectory_voxel(T, T_idx, SHAPE, prev, None, 10, None, 'direction')[0]
    assert first in (ahead, near_side)
    with pytest.raises(ValueError, match="tie_break"):
        choose('random')


def test_angle_constraint():
    prev2, prev = (2, 5, 5), (3, 5, 5)  # heading +x
    sharp, straight = (3, 6, 5), (4, 5, 5)  # 90 degrees vs 0
    T, T_idx = _candidates([(sharp, 1.0), (straight, 0.5)])
    # 0.33333 rad (19 degrees): the sharp turn is rejected for the lower but straight voxel
    assert utils.choose_trajectory_voxel(T, T_idx, SHAPE, prev, prev2, 10, 0.33333) == (straight, True)
    assert utils.choose_trajectory_voxel(T, T_idx, SHAPE, prev, prev2, 10, None) == (sharp, True)
    # nothing within the limit: the highest value anyway, reported as a failure
    T, T_idx = _candidates([(sharp, 1.0), ((3, 4, 5), 0.5)])
    assert utils.choose_trajectory_voxel(T, T_idx, SHAPE, prev, prev2, 10, 0.33333) == (sharp, False)
    # no direction yet (prev2 None): no angle constraint
    assert utils.choose_trajectory_voxel(T, T_idx, SHAPE, prev, None, 10, 0.33333) == (sharp, True)
