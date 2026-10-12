#!/usr/bin/env bash
# This branch against the original dwi-tracts (master) on the demo data: same inputs, outputs compared.
#   tests/fidelity/run_fidelity.sh [FIDELITY_DIR]   (default ~/dwi-tracts-fidelity)
# FIDELITY_DIR must hold data/ (dwi-tracts-data) and env-orig/ (master's environment.yaml plus
# the rft1d fork); see README.md. PYTHON is a Python with this branch installed (default python3).
# ORIG_REF is the original to compare with (default master).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"; repo="$(cd "$here/../.." && pwd)"
root="$(realpath "${1:-$HOME/dwi-tracts-fidelity}")"
python="${PYTHON:-python3}"
rm -rf "$root/orig" && mkdir -p "$root/orig"
git -C "$repo" archive "${ORIG_REF:-master}" | tar -x -C "$root/orig"
"$python" "$here/patch_orig.py" "$root/orig"
"$python" "$here/make_configs.py" "$root"
nki="$root/stage/nki"
rm -rf "$nki"/tract_stats_{orig,new} "$nki"/deriv/sub-*/ses-DS2/dwi/regress_{orig,new}  # logs append across runs
(cd "$root/orig" && "$root/env-orig/bin/python" "$here/run_tsa.py" orig "$root/config_orig.json" "$root/orig")
(cd "$repo" && "$python" "$here/run_tsa.py" new "$root/config_new.json")
args=("$nki/tract_stats_orig/dmn" "$nki/tract_stats_new/dmn")
for sub in $(cat "$root/subjects.list"); do
  args+=("$nki/deriv/sub-$sub/ses-DS2/dwi/regress_orig/dmn" "$nki/deriv/sub-$sub/ses-DS2/dwi/regress_new/dmn")
done
"$python" "$here/compare.py" "${args[@]}"
