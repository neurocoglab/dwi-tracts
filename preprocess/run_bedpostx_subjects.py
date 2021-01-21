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

# Load configuration parameters from JSON file
import json
import subprocess
from subprocess import Popen, PIPE, STDOUT 
import csv
import os
from glob import glob

def _json_object_hook(d): return namedtuple('X', d.keys())(*d.values())
def json2obj(data): return json.loads(data, object_hook=_json_object_hook)

config_file = sys.argv[1];

with open(config_file, 'r') as myfile:
    json_string=myfile.read() #.replace('\n', '')

config = json.loads(json_string)
config_gen = config['general']
config_bpx = config['bedpostx']
config_sched = config['scheduler']

if not os.path.isdir(config_gen['root_dir']):
    raise Exception('Root dir does not exist: {}'.format(config_gen['root_dir']))
    
convert_dir = '{0}/{1}'.format(config_gen['root_dir'], config_gen['convert_dir'])
deriv_dir = '{0}/{1}'.format(config_gen['root_dir'], config_gen['deriv_dir'])

# For each subject, run bedpostx

# Load subject names
if not os.path.isdir(deriv_dir):
    os.mkdir(deriv_dir)
    if params_gen['verbose']: 
    	print('Created derivatives dir: {}'.format(deriv_dir))
    
subjects = [];
with open(config_gen['subjects_file']) as subj_file:
    reader = csv.reader(subj_file)
    for row in reader:
        subjects.append(row[0])

if params_gen['verbose']:            
	print('Processing {} subjects'.format(len(subjects)))

for subject in subjects:
	if params_gen['verbose']: 
    	print('Processing subject/session {}'.format(subject))

	if config_sched['submit']:
		cmd = '{0} {1} {2} {3}'.format(config_sched['command'], config_bpx['job_file'], \
    							   subject, config_file)
		sp = Popen(cmd, shell=True, stderr=subprocess.PIPE)
		out, err = sp.communicate()
	else:
		cmd = './{0} {1} {2}'.format(config_bpx['job_file'], \
    							   subject, config_file)
		sp = Popen(cmd, shell=True, stderr=subprocess.PIPE)
		out, err = sp.communicate()
    
	if params_gen['verbose']: 
		if not err:
			print('  Finished {}'.format(subject))
		else:
			print('  {0} failed:\n{1}'.format(subject, err))
		print('Finished subject {}'.format(subject))
    
    
