"""Write the original's and this branch's configs from the demo config (config_tracts_dmn.json):
same inputs and parameters, separate outputs (tract_stats_<which>, regress_<which>).

    python make_configs.py FIDELITY_DIR [ROI ...]      (default ROIs: PCm_L PFCm_L PCG_L ACC_L)

FIDELITY_DIR holds orig/ (the original, exported by run_fidelity.sh) and data/ (dwi-tracts-data);
this writes stage/ (links to the data and ROIs), subjects.list, rois.list, config_orig.json and
config_new.json.
"""

import json
import os
import sys
from pathlib import Path

root = Path(sys.argv[1]).expanduser().resolve()
rois = sys.argv[2:] or ["PCm_L", "PFCm_L", "PCG_L", "ACC_L"]
stage = root / "stage"
stage.mkdir(exist_ok=True)
for name, target in (("nki", root / "data" / "data" / "nki"), ("rois", root / "orig" / "data" / "rois" / "dmn")):
    link = stage / name
    if not link.exists():
        os.symlink(target, link)
subjects = sorted(p.name.removeprefix("sub-") for p in (stage / "nki" / "deriv").glob("sub-*"))
(root / "subjects.list").write_text("".join(f"{s}\n" for s in subjects))
(root / "rois.list").write_text("".join(f"{r}\n" for r in rois))

base = json.loads((root / "orig" / "project" / "config_tracts_dmn.json").read_text())
fsl = os.environ.get("FSLDIR") or str(Path.home() / "fsl")
for which in ("orig", "new"):
    cfg = json.loads(json.dumps(base))
    cfg["general"].update({
        "source_dir": str(stage), "project_dir": "nki", "deriv_dir": "deriv", "rois_dir": "rois",
        "rois_list": str(root / "rois.list"), "subjects_file": str(root / "subjects.list"),
        "tracts_dir": f"tract_stats_{which}", "temp_dir": str(root / f"tmp_{which}"),
        "preproc_config": str(root / "orig" / "project" / "config_preprocess_dmn.json"),
        "clobber": True, "verbose": False, "debug": False,
    })
    cfg["dwi_regressions"].update({"fsl_root": fsl, "regress_dir": f"regress_{which}"})
    if which == "new":  # the original's trajectory rule: no angle limit, ties in stable index order
        cfg["gaussians"].update({"tie_break": "stable", "max_seg_angle": None})
    (root / f"config_{which}.json").write_text(json.dumps(cfg, indent=2) + "\n")
print(f"{len(subjects)} subjects, ROIs {', '.join(rois)}; configs in {root}")
