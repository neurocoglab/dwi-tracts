#!/usr/bin/env python
# coding: utf-8


# Imports
# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')

import json
from dwitracts.main import DwiTracts
import dwitracts.plot as plot
import sys

print('Debug! If you can''t see me, something is wrong')

# Initialize a DwiTract project from argument
config_file = sys.argv[1] #'config_tracts_lcnet.json'

with open(config_file, 'r') as myfile:
    params = json.loads(myfile.read())

my_dwi = DwiTracts( params )

print('Initialising...')
assert my_dwi.initialize()

clobber = params['general']['clobber']
debug = params['general']['debug']
verbose = params['general']['verbose']

# Compute bidirectional average distributions for each tract ([AB + BA] / 2)
print('Computing bidirectional averages...')
assert my_dwi.compute_bidirectional_averages( verbose=verbose, clobber=clobber, debug=debug )



# Compute tract distances (A->B and B->A) for each tract
print('Computing tract distances...')
assert my_dwi.compute_tract_distances( verbose=verbose, clobber=clobber, debug=debug )



# Generate core polylines, gaussian uncertainty fields, and unidirectional tract estimates 
print('Estimating unidirectional tracts...')
assert my_dwi.estimate_unidirectional_tracts( verbose=verbose, clobber=clobber, debug=debug )


# Generate bidirectional tract estimates 
print('Estimating bidirectional tracts...')
assert my_dwi.estimate_bidirectional_tracts( verbose=verbose, clobber=clobber )



# Create a pass/fail Pajek graph
print('Create pass/fail Pajek graph...')
assert my_dwi.tracts_to_pajek( verbose=verbose, clobber=clobber )



# Compute voxel-wise average streamline orientations for each tract
print('Computing voxel-wise average orientations...')
assert my_dwi.compute_average_orientations( verbose=verbose, clobber=clobber )



# Compute tract-specific anisotropy
print('Computing TSA...')
assert my_dwi.compute_tsa( verbose=verbose, clobber=clobber )



# Generate average TSA images for each tract
print('Computing average TSA images per tract...')
assert my_dwi.generate_mean_tsa_images( verbose=verbose )


# Plot TSA histograms
params_plot = {}
params_plot['axis_font'] = 18
params_plot['ticklabel_font'] = 12
params_plot['title_font'] = 18
params_plot['show_labels'] = False
params_plot['show_title'] = False
params_plot['dimensions_tracts'] = (50,40)
params_plot['dimensions_all'] = (50,40)
params_plot['dpi_tracts'] = 150
params_plot['dpi_all'] = 300
params_plot['num_bins'] = 20
params_plot['kde'] = True
params_plot['stat'] = 'density'
params_plot['xlim'] = [-0.25, 0.75]
params_plot['xticks'] = [-0.25, 0.0, 0.25, 0.50, 0.75]
params_plot['color'] = '#2b3ad1'  # blue

tract_names = None

for threshold in [0.1,0.3,0.5]:
    stats = plot.plot_tsa_histograms( params_plot, my_dwi, tract_names=tract_names, threshold=threshold,                                       verbose=verbose, clobber=clobber )

    # Save TSA stats to CSV file
    stats.to_csv('{0}/stats_tsa_{1:02d}.csv'.format(my_dwi.tracts_dir, round(threshold*100)))






