# Fidelity check: this branch against the original

`run_fidelity.sh` runs the TSA analysis twice on the demo data ([dwi-tracts-data](https://github.com/neurocoglab/dwi-tracts-data), 3 NKI participants, DMN ROIs):

- once with the original code (`master`, exported by `git archive`, in its Python 3.7 environment with the rft1d fork);
- once with this branch.

It runs the steps of `compute_tract_metrics.py` up to the mean TSA images:

1. average tracts
2. distances
3. unidirectional tracts
4. bidirectional tracts
5. orientations
6. TSA regressions
7. mean images

It then compares every output file. Images may differ by at most 1e-4 of the original's largest absolute value. Text files must be identical.

`patch_orig.py` makes three changes to the exported original before it runs, so that the comparison shows only differences nobody intended:

- `utils.run_fsl` sources FSL's setup from `$FSLDIR` instead of the fixed `/usr/local/fsl`. Without this, no FSL command runs on a machine without that path.
- The corrected beta regression equation. It changes the TSA betas on purpose.
- A stable sort where the core tracts choose among tied voxels. With numpy's default sort, the choice depends on the numpy version, so the original's own results vary with its environment.

## Setup (once)

```bash
F=~/dwi-tracts-fidelity
mkdir -p $F/data
curl -L https://codeload.github.com/neurocoglab/dwi-tracts-data/tar.gz/09a474f6f121993855eb8b40142da600aa349691 | tar -xz -C $F/data --strip-components=1
git show master:environment.yaml > $F/environment-orig.yaml
conda env create -p $F/env-orig -f $F/environment-orig.yaml
$F/env-orig/bin/pip install "git+https://github.com/neurocoglab/rft1d@dfe8bbaad709e23e1e9130bd1ef5ac81114d95cf"
```

FSL must be installed, with `FSLDIR` set.

## Run

From the repository, with this branch installed in the Python given as `PYTHON` (`pip install -e .`):

```bash
PYTHON=.venv/bin/python tests/fidelity/run_fidelity.sh ~/dwi-tracts-fidelity
```

It takes about 25 minutes. It exits 0 when everything matches. Otherwise it lists each file that is missing or differs.
