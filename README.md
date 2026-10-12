# dwi-tracts
DWI probabilistic tractography-based tract-specific metrics.

# Overview
**dwi-tracts** implements four main functions:

* Preprocessing pipeline for DWI data, based on the [FMRIB software library (FSL)](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki), and in particular the [FMRIB diffusion toolbox (FDT)](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FDT)
* The identification of "core" tract trajectories based on probabilistic tractography from a sample of individuals representing a target population, and an arbitrary set of grey matter regions-of-interest (ROIs)
* The estimation of participant- and tract-specific diffusion statistics on the basis of average streamline orientations (tract-specific anisotropy; TSA)
* Linear regression analysis of TSA results, given a set of metadata; this includes cluster-based inference based on one-dimensional random field theory and control of the false discovery rate (FDR)

The relevant theory, implementation, and output are described in more detail in [this preprint](https://doi.org/10.1101/2021.03.05.434061).

The software is organised into several components:

* **preprocess**: All preprocessing pipeline functions, including support for parallel computing
* **main**: Tract trajectory determination, TSA derivation
* **glm**: Linear regression analysis of TSA values
* **plot**: Plotting functions, based on the [Matplotlib](https://matplotlib.org/) and [Seaborn](https://seaborn.pydata.org/) APIs
* **utils**: Utility functions required for the other components

# Installation
**dwi-tracts** is a Python package (`dwitracts`) for Python 3.10 or later (tested with 3.12). Clone this repository and install it with pip, which also installs the Python dependencies:

```console
git clone https://github.com/neurocoglab/dwi-tracts.git
cd dwi-tracts
pip install -e .
```

Or install it straight from GitHub (without the notebooks, demo project and templates):

```console
pip install "dwitracts @ git+https://github.com/neurocoglab/dwi-tracts@develop"
```

With [conda](https://docs.conda.io/), `conda env create --file environment.yaml` makes a `dwitracts` environment with the package and Jupyter installed. Activate it with `conda activate dwitracts`.

# Dependencies

[FSL software](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation) (preferably v6.0 or later) must be installed on your system, as well as the computing cluster if you are running the preprocessing steps there. FSL programs are found on `PATH`, or under `dwi_regressions.fsl_root` when that is set.

The Python dependencies (numpy, scipy, nibabel, nilearn, pandas, statsmodels, matplotlib, seaborn, tqdm and [rft1d](https://github.com/0todd0000/rft1d)) are installed by pip. Earlier versions needed a [Python 3 fork of rft1d](https://github.com/neurocoglab/rft1d). The current upstream release (0.2.8) runs on Python 3 and current numpy, so the fork is no longer needed.

# Using dwi-tracts from another pipeline

By default, `DwiTracts(params)` reads and writes in dwi-tracts' own folder structure, configured by the `general` parameters (see `dwitracts/layout.py`). A pipeline that keeps its outputs elsewhere can pass its own layout: `DwiTracts(params, layout=my_layout)`, where `my_layout` has the methods of `DwiTractsLayout`. Those methods return the paths of the tracking, preprocessed DWI, bvals/bvecs, FA, diffusivity, template warps, the template, and where outputs go.

# Tests

```console
pip install -e ".[test]"
pytest tests
```

`tests/fidelity/` compares this version's TSA results with the original (`master`) on the demo data; see its README.

# Getting started

A Wiki guide is [now available](https://github.com/neurocoglab/dwi-tracts/wiki).

Two [Jupyter notebooks](https://jupyter.org/) are provided to demonstrate the use of the software, using three participants from the [Enhanced NKI Rockland](http://fcon_1000.projects.nitrc.org/indi/enhanced/) sample as a demo dataset, available [here](https://github.com/neurocoglab/dwi-tracts-data) (see the "project" folder for examples of the required parameters, specified as JSON files). Run times are dependent on the input size; the longer steps (typically >30 min) are indicated below:

* **Compute Tract Metrics**: This runs all the steps after preprocessing has been performed, including:
    * Computing bidirectional average distributions
    * Computing tract distances
    * Estimating unidirectional tracts
    * Estimating bidirectional tracts
    * Generating a pass/fail graph in [Pajek](http://vlado.fmf.uni-lj.si/pub/networks/pajek/) format
    * Computing average streamline orientations (>30 min)
    * Computing TSA values (>30 min)
    * Generating average TSA images in [NIFTI](https://nifti.nimh.nih.gov/nifti-1/) format
    * Generating TSA histogram plots
* **GLM Analysis**: This runs all GLM analyses on the generated tract and TSA values:
    * Fitting GLMs (>30 min)
    * Extracting distance-wise GLM statistics (> 30 min)
    * Performing 1D RFT inference and (optional) FDR correction
    * Generating t-value trace plots
    * Generating scatter and violin plots of GLM results
    * Generating aggregate tract and statistics images
    * Generating GLM t-value sum graphs in Pajek format

It is recommended that you make a copy of these notebooks, and copy the "data" and "project" folders to a location other that the repository directory.

To run a notebook, navigate to the source folder using a terminal window, and type:

```console
jupyter notebook
```

which will start the notebook in your web browser. You can then sequentially run the cells in the notebook. Use the `verbose` argument to most functions in order to get detailed console feedback.

This also currently serves as a test function for the correct installation of the software.

# Documentation
Development of a detailed API, including parameter specifications for all functions, is in expected by the end of March, 2021. 

# Citation

If you use this code in your research, please consider citing the article:

Reid AT, Camilleri JA, Hoffstaedter F, Eickhoff SB (2022). Tract-specific statistics based on diffusion-weighted probabilistic tractography. _Commun Biol._ 5:138. doi: [10.1038/s42003-022-03073-w](http://dx.doi.org/10.1038/s42003-022-03073-w)

Earlier preprint is available here:

Reid AT, Camilleri JA, Hoffstaedter F, Eickhoff SB (2021). Tract-specific statistics based on diffusion-weighted probabilistic tractography. _bioRxiv._ doi: [10.1101/2021.03.05.434061v1](https://doi.org/10.1101/2021.03.05.434061)
