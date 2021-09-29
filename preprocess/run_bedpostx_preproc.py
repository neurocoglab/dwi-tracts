#!/usr/bin/env python

# Runs BedpostX preprocessing steps on a single subject

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
    config_ptx = config['probtrackx']
    
    prefix = congif_gen['prefix']

    if not os.path.isdir(config_gen['root_dir']):
        raise Exception('Root dir does not exist: {}'.format(config_gen['root_dir']))

    convert_dir = '{0}/{1}'.format(config_gen['root_dir'], config_gen['convert_dir'])
    deriv_dir = '{0}/{1}'.format(config_gen['root_dir'], config_gen['deriv_dir'])
    
    # Step 1: Eddy current correction
    # eddy_correct <4dinput> <4doutput> <reference_no>    
    fsl_bin = config_gen['fsl_bin']
    
    subj = '{0}{1}'.format(config_gen['prefix'], subject)
    input_dir = '{0}/{1}/dwi'.format(convert_dir, subject);
   
    if not input_dir:
        print('Subject {0} has no data.'.format(subject))
        return

    output_dir = '{0}/{1}/dwi'.format(deriv_dir, subject)
    
    flag_file = '{0}/{1}/preproc.done'.format(output_dir, config_gen['flags_dir'])
    if os.path.isfile(flag_file):
    	if not config_gen['clobber']:
    		print('Subject {0} already preprocessed. Skipping.'.format(subject))
    		return False
    	print('Subject {0} already preprocessed. Clobbering.'.format(subject))
    	
    if os.path.isdir(output_dir) and config_gen['clobber']:
        shutil.rmtree(output_dir)
    
    if not os.path.isdir(output_dir):
        os.makedirs('{0}/{1}'.format(output_dir, config_gen['flags_dir']))

    bedpostx_dir = '{0}/bedpostX'.format(output_dir)
    bedpostx_done = os.path.isdir(bedpostx_dir)
    
    if bedpostx_done and not config_gen['clobber']:
        print('\tBedpostX results already exist. Skipping those steps.')
    else:
    
        dwi_img = '{0}/data.nii.gz' \
                        .format(output_dir)

        input_img = '{0}/{1}{2}.nii.gz'.format(input_dir, prefix, subj)

        output_img = '{0}/{1}.nii.gz' \
                        .format(output_dir, config_bpx['eddy_suffix'])

		

        if os.path.exists(dwi_img) and not config_gen['clobber']:
            print('\tEddy output exists for {}. Skipping.'.format(subject))
        else:
            cmd_eddy = '{0}eddy_correct {1} {2} {3}' \
                        .format(fsl_bin, input_img, output_img, config_bpx['eddy_reference'])
            print('\t{}'.format(cmd_eddy))
            err = run_fsl(cmd_eddy)
            if err:
                print('\tError with eddy correction [{0}]: {1}'.format(subject,err))
                return False
            
            print('\tDone eddy correction [{}].'.format(subject))   

            # Rename image to data.nii.gz
            cmd = 'mv {0} {1}'.format(output_img, dwi_img);
            print('\t{}'.format(cmd))
            err = run_fsl(cmd)
            if err:
                print('\tError renaming eddy output image [{0}]: {1}'.format(subject,err))
                return False


        # Step 2: Brain extraction (BET)
        #      2.1: Extract B0 images (using fslroi and bvals file)
        #      2.2: Get mean B0 image (using fslmerge)
        #      2.3: Run BET on the mean B0

        # Check if output already exists 
        bet_img = '{0}/{1}'.format(output_dir, config_bpx['bet_suffix'])

        nodif_img = '{0}/nodif.nii.gz'.format(output_dir)
        
        bval_file = '{0}/{1}{2}.bval'.format(input_dir, prefix, subj)
        bvec_file = '{0}/{1}{2}.bvec'.format(input_dir, prefix, subj)
        bet_mask_img = '{0}/nodif_brain_mask.nii.gz'.format(output_dir)

        if (os.path.exists(bet_img) or os.path.exists(nodif_img)) and not config_gen['clobber']:
            print('\tBET output exists for {}. Skipping.'.format(subject))
        else:

            input_img = dwi_img;
            
            bzeros = []
            with open(bval_file, 'r') as csvfile:
                reader = csv.reader(csvfile, delimiter=' ')
                for row in reader:
                    if len(row) > 0:
                        c = 0
                        for col in row:
                            if len(col) > 0 and int(col) < 100:
                                bzeros.append(c)
                            c+=1;

            # Extract each B0
            for i in range(len(bzeros)):
                cmd = '{0}fslroi {1} {2}/xno_{3}.nii.gz {4} 1' \
                            .format(fsl_bin, input_img, output_dir, i, bzeros[i])
                print('\t{}'.format(cmd))
                err = run_fsl(cmd)
                if err:
                    print('\tError extracting B0 images [{0}]: {1}'.format(subject,err))
                    return False

            # Merge into a 4D image
            input_img = nodif_img
            cmd = '{0}fslmerge -t {1} {2}/xno*.nii.gz' \
                            .format(fsl_bin, input_img, output_dir)
            print('\t{}'.format(cmd))
            err = run_fsl(cmd)
            if err:
                print('\tError merging B0 images [{0}]: {1}'.format(subject,err))
                return False

            # Compute mean B0 from this image
            cmd = '{0}fslmaths {1} -Tmean {1}' \
                            .format(fsl_bin, input_img)
            print('\t{}'.format(cmd))
            err = run_fsl(cmd)
            if err:
                print('\tError computing B0 image [{0}]: {1}'.format(subject,err))
                return False

            # Run BET on the output
            cmd_bet = '{0}bet2 {1} {2} -m -f 0.3' \
                        .format(fsl_bin, input_img, bet_img)
            print('\t{}'.format(cmd_bet))
            err = run_fsl(cmd_bet)
            if err:
                print('\tError with brain extraction (BET) [{0}]: {1}'.format(subject,err))
                return False
            
            # Rename the BET mask
            cmd = 'mv {0}_mask.nii.gz {1}/nodif_brain_mask.nii.gz' \
                        .format(bet_img, output_dir)
            print('\t{}'.format(cmd))
            err = run_fsl(cmd)
            if err:
                print('\tError removing residual B0 images [{0}]: {1}'.format(subject,err))
                return False

            # Clean up the mess
            cmd = 'rm {0}/xno*.nii.gz; rm {0}/sub*'.format(output_dir);
            print('\t{}'.format(cmd))
            run_fsl(cmd) # Ignore any error here
            
            print('\tDone brain extraction (BET) [{}].'.format(subject))

        # Step 3: BedpostX setup (Fitting model)
        #      3.1: Copy bvals and bvecs
        #      3.2: Run dtifit

        # Copy bvals and bvecs
        cmd = 'cp {0} {1}/bvecs; cp {2} {1}/bvals' \
                        .format(bvec_file, output_dir, bval_file)

        print('\t{}'.format(cmd))
        err = run_fsl(cmd)
        if err:
            print('\tError copying bvecs and bvals [{0}]: {1}'.format(subject,err))
            return False

        # Run dtifit
        if os.path.exists('{0}/dti_FA.nii.gz'.format(output_dir)) and not config_gen['clobber']:
            print('\tDTI output exists for {}. Skipping.'.format(subject))
        else:
            cmd = '{0}dtifit -k {1} -m {2} -r {3}/bvecs -b {3}/bvals -o {3}/dti' \
                        .format(fsl_bin, dwi_img, bet_mask_img, output_dir)
            print('\t{}'.format(cmd))
            err = run_fsl(cmd)
            if err:
                print('\tError running dtifit [{0}]: {1}'.format(subject,err))
                return False
           
            print('\tDone dtifit [{}].'.format(subject))

        # Write flag file
        with open(flag_file,'w'):
        	print('Pre-processing: Wrote flag file {0}'.format(flag_file))
                
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
