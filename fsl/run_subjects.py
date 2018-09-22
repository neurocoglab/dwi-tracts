#!/usr/bin/python

# Read a list of subjects, and submit parallel BedpostX jobs for each one
import csv
import sys
import os

import subprocess
from subprocess import Popen 

import json

cwd = os.getcwd()

def _json_object_hook(d): return namedtuple('X', d.keys())(*d.values())
def json2obj(data): return json.loads(data, object_hook=_json_object_hook)

command = sys.argv[1];

subjects_file = sys.argv[2];
config_file = sys.argv[3];

with open(config_file, 'r') as myfile:
	json_string=myfile.read() #.replace('\n', '')

config = json.loads(json_string)
config_qsub = config['qsub']
# print(config_qsub)
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
        
print('Processing {0}: found {1} subjects'.format(command, len(subjects)))

for subject in subjects:
	# Create temporary script file
	job_file = '{0}/{1}-{2}.sh'.format(temp_dir, command, subject)
	
	# Write simple command
	with open(job_file, "w") as jobfile:
		jobfile.write('cd {0}; ./run_{1}.py {2} {3}' \
			.format(cwd, command, subject, config_file))
	
	os.chmod(job_file,0775)
	
	# Submit to scheduler (unless otherwise specified)
	if not config['general']['dryrun'] and config['qsub']['use_qsub']:
		cmd = 'qsub -l walltime={0} -l select={1}:ncpus={2}:mem={3} ' \
				   '-P {4} {5} -j oe -o {6}/{7}-{8}.out' \
						.format(config_qsub['walltime'], config_qsub['select'], \
								config_qsub['ncpus'], config_qsub['mem'], \
								config_qsub['project_id'], \
								job_file, output_dir, command, subject)
			   
		print(cmd)
		os.system(cmd)
	# Run in serial
	else:
		os.system(job_file)
