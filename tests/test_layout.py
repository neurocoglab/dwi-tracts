"""DwiTractsLayout reproduces the paths main.py used to build from params['general']."""

from dwitracts.layout import REPO_UTILS, DwiTractsLayout, fsl_tool


def demo_params(tmp_path):
    # the demo configuration (project/config_tracts_dmn.json), with the data under tmp_path
    (tmp_path / "subjects.list").write_text("A00030989\nA00013809\n")
    (tmp_path / "dmn.rois").write_text("ACC_L\nPCm_L\n")
    return {
        "general": {"source_dir": str(tmp_path), "project_dir": "nki", "deriv_dir": "deriv", "prefix": "sub-",
                    "sub_dirs": "ses-DS2", "rois_dir": "rois", "rois_list": str(tmp_path / "dmn.rois"),
                    "network_name": "dmn", "subjects_file": str(tmp_path / "subjects.list"), "dwi_dir": "dwi",
                    "tracts_dir": "tract_stats"},
        "dwi_regressions": {"regress_dir": "regress_dwi", "standard_img": "Mean3G"},
    }  # fmt: skip


def test_paths(tmp_path):
    layout = DwiTractsLayout(demo_params(tmp_path))
    dwi = tmp_path / "nki" / "deriv" / "sub-A00013809" / "ses-DS2" / "dwi"
    assert layout.units() == ["A00013809", "A00030989"]
    assert layout.roi_names() == ["ACC_L", "PCm_L"]
    assert layout.roi_dir() == tmp_path / "rois"
    assert layout.group_dir() == tmp_path / "nki" / "tract_stats" / "dmn"
    assert layout.tracking_dir("A00013809") == dwi / "probtrackX" / "dmn"
    assert layout.regress_dir("A00013809") == dwi / "regress_dwi" / "dmn"
    assert layout.dwi("A00013809") == dwi / "sub-A00013809_ses-DS2_dwi_bet.nii.gz"
    assert layout.bvals("A00013809") == dwi / "bedpostX" / "bvals"
    assert layout.bvecs("A00013809") == dwi / "bedpostX" / "bvecs"
    assert layout.fa("A00013809") == dwi / "dti_FA.nii.gz"
    assert layout.diffusivity("A00013809") == dwi / "bedpostX" / "mean_dsamples.nii.gz"
    assert layout.warp_from_template("A00013809") == dwi / "reg3G" / "Mean3G_warp2FA.nii.gz"
    assert layout.warp_to_template("A00013809") == dwi / "reg3G" / "FA_warp2Mean3G.nii.gz"


def test_template_from_the_repository(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # no utils/ here
    assert DwiTractsLayout(demo_params(tmp_path)).template_image() == REPO_UTILS / "Mean3G.nii.gz"


def test_fsl_tool():
    assert fsl_tool("vecreg") == "vecreg"
    assert fsl_tool("vecreg", "/usr/local/fsl") == "/usr/local/fsl/bin/vecreg"
