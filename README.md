# dwi-tracts
DWI probabilistic tractography-based tract-specific metrics

# Overview
**dwi-tracts** implements four main functions:

* Preprocessing pipeline for DWI data, based on the [FMRIB software library (FSL)](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki), and in particular the [FMRIB diffusion toolbox (FDT)](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FDT)
* The identification of "core" tract trajectories based on probabilistic tractography from a sample of individuals representing a target population, and an arbitrary set of grey matter regions-of-interest (ROIs)
* The estimation of participant- and tract-specific diffusion statistics on the basis of average streamline orientations (tract-specific anisotropy; TSA)
* Linear regression analysis of TSA results, given a set of metadata; this includes cluster-based inference based on one-dimensional random field theory and control of false discovery rate (FDR)

The relevant theory and implementation are described in more detail in this preprint.

The software is organised into several components:

* **preprocess**: All preprocessing pipeline functions, including support for parallel computing
* **main**: Tract trajectory determination, TSA derivation
* **glm**: Linear regression analysis of TSA values
* **plot**: Plotting functions, based on the [Matplotlib](https://matplotlib.org/) and [Seaborn](https://seaborn.pydata.org/) APIs
* **utils**: Utility functions required for the other components

# Installation
**dwi-tracts** can be cloned from this Github repository. 

The dependencies listed below will also have to be installed on your system. An [Anaconda](https://anaconda.org/anaconda) installation of Python 3 is highly recommended, although not necessary.

# Getting started
Two Jupyter notebooks are provided to demonstrate the use of the software, using the [Enhanced NKI Rockland](http://fcon_1000.projects.nitrc.org/indi/enhanced/) sample as a demo dataset (see the "project" folder for examples of the required parameters, specified as JSON files). 

This also currently serves as a test function for the correct installation of the software.

# Documentation
Development of a detailed API, including parameter specifications for all functions, is in expected by the end of March, 2021. A Wiki guide is also in the works.

# Dependencies

These dependencies are included in [Anaconda](https://docs.anaconda.com/anaconda/install/) builds (recommended):
* [pandas](https://pandas.pydata.org/pandas-docs/stable/getting_started/install.html)
* [matplotlib](https://anaconda.org/conda-forge/matplotlib)
* [numpy](https://anaconda.org/anaconda/numpy)
* [scipy](https://anaconda.org/anaconda/scipy)

These need to be installed separately:
* [nibabel](https://anaconda.org/conda-forge/nibabel)
* [statsmodels](https://anaconda.org/anaconda/statsmodels)
* [tqdm](https://anaconda.org/conda-forge/tqdm)
* [seaborn](https://anaconda.org/anaconda/seaborn)
