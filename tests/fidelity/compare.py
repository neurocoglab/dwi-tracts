"""Compare the original's and this branch's TSA outputs file by file.

    python compare.py ORIG_TRACTS_DIR NEW_TRACTS_DIR [ORIG_REGRESS NEW_REGRESS ...]

NIfTI images: same shape, max |difference| (absolute and relative to the original's max |value|);
text files: identical (run logs apart from their start times). Exits 1 if anything is missing or differs beyond the tolerance.
"""

import sys
from pathlib import Path

import nibabel as nib
import numpy as np

REL_TOL = 1e-4  # float32 vs float64 arithmetic and scipy versions differ in the last digits

pairs = [(Path(a), Path(b)) for a, b in zip(sys.argv[1::2], sys.argv[2::2])]
worst, problems, compared = 0.0, [], 0
for orig_dir, new_dir in pairs:
    files = sorted(p.relative_to(orig_dir) for p in orig_dir.rglob("*") if p.is_file())
    for rel in files:
        a, b = orig_dir / rel, new_dir / rel
        if not b.exists():
            problems.append(f"missing in new: {new_dir.name}/{rel}")
            continue
        compared += 1
        if rel.name.endswith((".nii", ".nii.gz")):
            x, y = np.asanyarray(nib.load(a).dataobj, dtype=float), np.asanyarray(nib.load(b).dataobj, dtype=float)
            if x.shape != y.shape:
                problems.append(f"shape {x.shape} vs {y.shape}: {rel}")
                continue
            scale = np.nanmax(np.abs(x)) or 1.0
            diff = float(np.nanmax(np.abs(x - y))) / scale if x.size else 0.0
            worst = max(worst, diff)
            if diff > REL_TOL or not np.array_equal(np.isnan(x), np.isnan(y)):
                problems.append(f"relative difference {diff:.2e}: {rel}")
        elif rel.parts[0] == "log":  # run logs: same entries, apart from each run's start time
            if [l for l in a.read_text().splitlines() if not l.startswith("--New run:")] != \
               [l for l in b.read_text().splitlines() if not l.startswith("--New run:")]:
                problems.append(f"log differs: {rel}")
        elif a.read_bytes() != b.read_bytes() and rel.suffix not in (".log",):
            problems.append(f"text differs: {rel}")
    extra = sorted(p.relative_to(new_dir) for p in new_dir.rglob("*") if p.is_file() and not (orig_dir / p.relative_to(new_dir)).exists())
    problems += [f"only in new: {new_dir.name}/{p}" for p in extra]
print(f"{compared} files compared; largest relative image difference {worst:.2e}")
for p in problems:
    print("  " + p)
sys.exit(1 if problems else 0)
