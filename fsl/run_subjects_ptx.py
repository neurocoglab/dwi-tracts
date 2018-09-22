#!/usr/bin/python

# Arguments:
# Arg1: subjects list file
# Arg2: configuration file

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

#command = sys.argv[1];

subjects_file = sys.argv[1];
config_file = sys.argv[2];

with open(config_file, 'r') as myfile:
	json_string=myfile.read() #.replace('\n', '')

config = json.loads(json_string)
config_qsub = config['qsub']
config_gen = config['general']
config_ptx = config['probtrackx']

deriv_dir = '{0}/{1}'.format(config_gen['root_dir'], config_gen['deriv_dir'])

temp_dir = config_qsub['temp_dir'];
if not os.path.isdir(temp_dir):
	os.makedirs(temp_dir)
	
output_dir = config_qsub['output_dir']
if not os.path.isdir(output_dir):
	os.makedirs(output_dir)

subjects = [];
with open(subjects_file) as subj_file:
    reader = csv.reader(subj_file)
    for row in reader:
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

	for roi in rois:

		if config_gen['verbose']:
			print('ROI: {0}'.format(roi))
	
		# Create temporary script file
		job_file = '{0}/probtrackx-{1}-{2}.sh'.format(temp_dir, subject, roi)
	
		# Write simple command
		with open(job_file, "w") as jobfile:
			jobfile.write('cd {0}; ./run_probtrackx2.py {1} {2} {3}' \
				.format(cwd, subject, roi, config_file))
	
		os.chmod(job_file,0775)
	
		# Submit to scheduler (unless otherwise specified)
		if config_qsub['use_qsub']:
			cmd = 'qsub -l walltime={0} -l select={1}:ncpus={2}:mem={3} ' \
					   '-P {4} {5} -j oe -o {6}/probtrackx-{7}-{8}.out' \
							.format(config_qsub['walltime'], config_qsub['select'], \
									config_qsub['ncpus'], config_qsub['mem'], \
									config_qsub['project_id'], \
									job_file, output_dir, subject, roi)
			   
			print(cmd)
			if not config_gen['dryrun']:
				os.system(cmd)
		# Run in serial
		else:
			if not config_gen['dryrun']:
				os.system(job_file)
			else:
				print(job_file)
				