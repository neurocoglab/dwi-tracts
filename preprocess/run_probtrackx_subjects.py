#!/usr/bin/env python

# Runs ProbtrackX on a list of subjects, by submitting jobs to the local
# scheduler (supports qsub and sbatch commands). The job file to be run
# must be called probtrack_job.sh, and must set the scheduler arguments
# (queue, walltime, memory, etc.) and call run_probtrack.py.
#
# Requires that BedpostX has already been run for all subjects.
#
# Target ROI list and other parameters must be specified in a JSON file.
#
# By default, all pairs of ROIs in the list are run as seeds; however,
# will run for only a single ROI if it is passed as the third argument.

# Arguments:
# Arg1: configuration file
# Arg2: [optional] single ROI to run; default=all

# Read a list of subjects, and submit parallel BedpostX jobs for each one
import csv
import sys
import os
import shutil

import subprocess
from subprocess import Popen

import json

cwd = os.getcwd()

def _json_object_hook(d): return namedtuple('X', d.keys())(*d.values())
def json2obj(data): return json.loads(data, object_hook=_json_object_hook)

config_file = sys.argv[1];

with open(config_file, 'r') as myfile:
	json_string=myfile.read() #.replace('\n', '')

config = json.loads(json_string)
config_gen = config['general']
config_ptx = config['probtrackx']
config_sched = config['scheduler']

subjects_file = config_gen['subjects_file']

deriv_dir = '{0}/{1}'.format(config_gen['root_dir'], config_gen['deriv_dir'])

temp_dir = config_gen['temp_dir'];
if not os.path.isdir(temp_dir):
	os.makedirs(temp_dir)

output_dir = config_gen['output_dir']
if not os.path.isdir(output_dir):
	os.makedirs(output_dir)

subjects = [];
with open(subjects_file) as subj_file:
    reader = csv.reader(subj_file)
    for row in reader:
        if len(row) > 0:
            subjects.append(row[0])

print('Processing probtrackx: found {0} subjects'.format(len(subjects)))

rois_dir = config_ptx['roi_dir']
# Read ROI list
rois = []
target_rois = {}

# If this is a JSON file, read as networks and targets
if config_ptx['roi_list'].endswith('.json'):
	with open(config_ptx['roi_list'], 'r') as myfile:
		json_string=myfile.read()

	netconfig = json.loads(json_string)
	networks = netconfig['networks']

	for net in networks:
		for roi in networks[net]:
			rois.append(roi)

else:
	with open(config_ptx['roi_list'],'r') as roi_file:
		reader = csv.reader(roi_file)
		for row in reader:
			rois.append(row[0])

rois2do = rois
if len(sys.argv) > 3:
	rois2do = [sys.argv[3]]

qos_str = ''
if len(config_sbatch['qos']) > 0:
	qos_str = ' --qos={0}'.format(config_sbatch['qos'])

for subject in subjects:

	if config_gen['verbose']:
		print('Subject {0}...'.format(subject))

	# Subject-specific paths
	subj_dir = '{0}/{1}{2}/{3}/dwi' \
						.format(deriv_dir, config_gen['prefix'], subject, config_gen['session'])
	probtrackx_dir = '{0}/probtrackX/{1}'.format(subj_dir, config_ptx['network_name'])

	# Remove existing directory if clobber
	if not os.path.isdir(probtrackx_dir):
		os.makedirs(probtrackx_dir)
	else:
		if config_gen['clobber']:
			shutil.rmtree(probtrackx_dir)
			os.makedirs(probtrackx_dir)

	for roi in rois2do:

		if config_gen['verbose']:
			print('ROI: {0}'.format(roi))
			
		job_file = config_ptx['job_file']

		# Submit to scheduler (unless otherwise specified)
		if config_sched['submit']:

			cmd = '{0} {1} {2} {3} {4}'.format(config_sched['command'], config_ptx['job_file'], \
											   subject, roi, config_file)

			if config_gen['dryrun'] or config_gen['verbose']:
				print(cmd)
				
			if not config_gen['dryrun']:
				os.system(cmd)

		# Run in serial
		else:
			cmd = './{0} {1} {2} {3}'.format(config_ptx['job_file'], \
											 subject, roi, config_file)
			if not config_gen['dryrun']:
				os.system(cmd)
			else:
				print(cmd)
