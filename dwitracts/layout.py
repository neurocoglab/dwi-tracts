"""Where the TSA analysis (main.DwiTracts) finds each unit's inputs and writes its outputs.

The analysis works on "units": subjects, or sessions of a subject. DwiTractsLayout is dwi-tracts'
own folder structure, configured by the "general" parameters:

    <source_dir>/<project_dir>/<deriv_dir>/<prefix><subject>/<sub_dirs>/<dwi_dir>/
        probtrackX/<network_name>/<roi>/   tracking (target_paths_*, target_localdir_*)
        bedpostX/                          bvals, bvecs, mean_dsamples
        reg3G/                             warps between FA and the template
        dti_FA.nii.gz, <prefix><subject>_<sub_dirs>_dwi_bet.nii.gz

DwiTracts(params) uses it by default. A pipeline with its own folder structure passes an object
with the same methods instead: DwiTracts(params, layout=...).
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any, Mapping

REPO_UTILS = Path(__file__).resolve().parents[1] / "utils"  # templates, in a clone of the repository


class DwiTractsLayout:
    def __init__(self, params: Mapping[str, Any]):
        self.general = params["general"]
        self.regress = params.get("dwi_regressions", {})
        source = Path(self.general["source_dir"])
        self.project_dir = source / self.general["project_dir"]
        self._roi_dir = source / self.general["rois_dir"]

    # Group-level folders and lists
    def group_dir(self) -> Path:
        """Outputs of the group stages (average tracts, final tracts, orientations)."""
        return self.project_dir / self.general["tracts_dir"] / self.general["network_name"]

    def roi_dir(self) -> Path:
        return self._roi_dir

    def roi_names(self) -> list[str]:
        with open(self.general["rois_list"], newline="") as handle:
            return [row[0] for row in csv.reader(handle) if row]

    def units(self) -> list[str]:
        with open(self.general["subjects_file"], newline="") as handle:
            return sorted(row[0] for row in csv.reader(handle) if row)

    def template_image(self) -> Path:
        """The template that tracking and the final tracts are in (dwi_regressions.standard_img)."""
        name = self.regress.get("standard_img", "Mean3G")
        for folder in (Path("utils"), REPO_UTILS):  # run from the repository, as dwi-tracts always was
            for candidate in (folder / name, folder / f"{name}.nii.gz", folder / f"{name}.nii"):
                if candidate.is_file():
                    return candidate
        raise FileNotFoundError(f"template {name} not found in utils/ or {REPO_UTILS}")

    # Per-unit inputs and outputs
    def _subject_dir(self, unit: str) -> Path:
        g = self.general
        return self.project_dir / g["deriv_dir"] / f"{g['prefix']}{unit}" / g["sub_dirs"]

    def _dwi_dir(self, unit: str) -> Path:
        return self._subject_dir(unit) / self.general["dwi_dir"]

    def tracking_dir(self, unit: str) -> Path:
        """probtrackx2 output, one folder per seed ROI."""
        return self._dwi_dir(unit) / "probtrackX" / self.general["network_name"]

    def regress_dir(self, unit: str) -> Path:
        """TSA outputs of compute_tsa / process_tsa_subject."""
        return self._subject_dir(unit) / "dwi" / self.regress.get("regress_dir", "regress_dwi") / self.general["network_name"]

    def dwi(self, unit: str) -> Path:
        """Preprocessed, brain-extracted diffusion-weighted images."""
        g = self.general
        return self._dwi_dir(unit) / f"{g['prefix']}{unit}_{g['sub_dirs']}_dwi_bet.nii.gz"

    def bvals(self, unit: str) -> Path:
        return self._dwi_dir(unit) / "bedpostX" / "bvals"

    def bvecs(self, unit: str) -> Path:
        return self._dwi_dir(unit) / "bedpostX" / "bvecs"

    def fa(self, unit: str) -> Path:
        return self._dwi_dir(unit) / "dti_FA.nii.gz"

    def diffusivity(self, unit: str) -> Path:
        """bedpostx's mean diffusivity (mean_dsamples)."""
        return self._dwi_dir(unit) / "bedpostX" / "mean_dsamples.nii.gz"

    def warp_from_template(self, unit: str) -> Path:
        """Warp from the template to the unit's diffusion space (applywarp -w)."""
        return self._dwi_dir(unit) / "reg3G" / "Mean3G_warp2FA.nii.gz"

    def warp_to_template(self, unit: str) -> Path:
        """Warp from the unit's diffusion space to the template."""
        return self._dwi_dir(unit) / "reg3G" / "FA_warp2Mean3G.nii.gz"


def fsl_tool(name: str, fsl_root: str | None = None) -> str:
    """An FSL program: under fsl_root/bin when given (dwi_regressions.fsl_root), else on PATH."""
    return os.path.join(fsl_root, "bin", name) if fsl_root else name
