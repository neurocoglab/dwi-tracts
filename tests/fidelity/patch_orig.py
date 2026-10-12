"""Apply to an exported copy of the original the two changes the comparison should not measure.

    python patch_orig.py ORIG_DIR

- utils.run_fsl: source FSL's setup from $FSLDIR, not the fixed /usr/local/fsl. Without that
  file, no FSL command runs at all (see the commit that fixed it on this branch).
- The corrected beta regression equation (exp(-b d (x.v)^2)): it changes the TSA betas on
  purpose, so both sides use it and the comparison shows only unintended differences.
"""

import sys
from pathlib import Path

PATCHES = {
    "dwitracts/utils.py": [(
        "    cmd = '{0};{1}'.format('. /usr/local/fsl/etc/fslconf/fsl.sh', cmd)",
        "    fsl_setup = os.path.join(os.environ.get('FSLDIR', '/usr/local/fsl'), 'etc', 'fslconf', 'fsl.sh')\n"
        "    if os.path.isfile(fsl_setup):\n"
        "        cmd = '. {0}; {1}'.format(fsl_setup, cmd)",
    )],
    "dwitracts/main.py": [(
        "xx   = np.exp(-(((bb*dd) * xTv)**2))",
        "xx   = np.exp(-((bb*dd) * xTv**2))",
    )],
}

orig = Path(sys.argv[1])
for name, edits in PATCHES.items():
    path = orig / name
    text = path.read_text()
    for old, new in edits:
        if text.count(old) != 1:
            sys.exit(f"{path}: expected one occurrence of {old!r}")
        text = text.replace(old, new)
    path.write_text(text)
print(f"patched {', '.join(PATCHES)} in {orig}")
