#!/usr/bin/env python

# Runs BedpostX on a single subject

# Command line arguments to this script:
# Arg1: subject ID
# Arg2: configuration file

import subprocess
from subprocess import Popen, PIPE, STDOUT 
import shutil
import os
import json
import sys
import csv

def run_fsl(cmd):
    sp = Popen(cmd, shell=True, stderr=subprocess.PIPE)
    out, err = sp.communicate()
    return err.decode('ascii')

def process_subject(subject, config):

    config_gen = config['general']
    config_bpx = config['bedpostx']
    config_sched = config['scheduler']

    if not os.path.isdir(config_gen['root_dir']):
        raise Exception('Root dir does not exist: {}'.format(config_gen['root_dir']))

    convert_dir = '{0}/{1}'.format(config_gen['root_dir'], config_gen['convert_dir'])
    deriv_dir = '{0}/{1}'.format(config_gen['root_dir'], config_gen['deriv_dir'])
    
    # Step 1: Eddy current correction
    # eddy_correct <4dinput> <4doutput> <reference_no>    
    fsl_bin = config_gen['fsl_bin']
    
    subj = '{0}{1}'.format(config_gen['prefix'], subject)
    input_dir = '{0}/{1}'.format(convert_dir, subject);
   
    if not input_dir:
        print('Subject {0} has no data.'.format(subject))
        return

    output_dir = '{0}/{1}'.format(deriv_dir, subject)
    
    flag_file = '{0}/{1}/preproc.done'.format(output_dir, config_gen['flags_dir'])
    if not os.path.isfile(flag_file):
	    print('Subject {0} has not been pre-processed. Stopping.'.format(subject))
	    return False
    	
    flag_file = '{0}/{1}/bedpostx.done'.format(output_dir, config_gen['flags_dir'])
    
    bedpostx_dir = '{0}/bedpostX'.format(output_dir)
    bedpostx_dir2 = '{0}.bedpostX'.format(output_dir)
    bedpostx_done = os.path.isdir(bedpostx_dir) or os.path.isdir(bedpostx_dir2)
    
    if bedpostx_done:
        if config_gen['clobber']:
            print('\tBedpostX directory already exists. Clobbering.')
            if os.path.isdir(bedpostx_dir):
                shutil.rmtree(bedpostx_dir)
            if os.path.isdir(bedpostx_dir2):
                shutil.rmtree(bedpostx_dir2)
        else:
            print('\tBedpostX directory already exists. Stopping.')
            return False

    # Step 3: BedpostX (Fitting model)
    #     	3.3: Run BedpostX

    # Run BedpostX
    cmd = '{0}bedpostx_gpu {1} -Q {2}'.format(fsl_bin, output_dir, config_sched['gpu_queue'])
    print('\t{}'.format(cmd))
    err = run_fsl(cmd)
    if err:
        print('\tError running BedpostX [{0}]: {1}'.format(subject,err))
        return False
    
    # Write flag file
    with open(flag_file,'w'):
        print('BedpostX_gpu: Wrote flag file {0}'.format(flag_file))
                
    return True

# Runs when file is called
def _json_object_hook(d): return namedtuple('X', d.keys())(*d.values())
def json2obj(data): return json.loads(data, object_hook=_json_object_hook)

subject = sys.argv[1]

# Load configuration parameters from JSON file
with open(sys.argv[2], 'r') as myfile:
    json_string=myfile.read() #.replace('\n', '')
config = json.loads(json_string)

process_subject(subject, config)
