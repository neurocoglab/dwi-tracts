#!/usr/bin/env python

# Runs BedpostX postprocessing on a single subject

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
    
def recursive_chmod(path, permission):
    for dirpath, dirnames, filenames in os.walk(path):
        os.chmod(dirpath, permission)
        for filename in filenames:
            os.chmod(os.path.join(dirpath, filename), permission)

def process_subject(subject, config):

    config_gen = config['general']
    config_bpx = config['bedpostx']
    config_ptx = config['probtrackx']

    if not os.path.isdir(config_gen['root_dir']):
        raise Exception('Root dir does not exist: {}'.format(config_gen['root_dir']))

    convert_dir = '{0}/{1}'.format(config_gen['root_dir'], config_gen['convert_dir'])
    deriv_dir = '{0}/{1}'.format(config_gen['root_dir'], config_gen['deriv_dir'])
  
    fsl_bin = config_gen['fsl_bin']
    
    subj = '{0}{1}'.format(config_gen['prefix'], subject)
    input_dir = '{0}/{1}'.format(convert_dir, subject);
   
    if not input_dir:
        print('Subject {0} has no data.'.format(subject))
        return

    output_dir = '{0}/{1}'.format(deriv_dir, subject)
    
    flag_file = '{0}/{1}/bedpostx.done'.format(output_dir, config_gen['flags_dir'])
    if not os.path.isfile(flag_file):
        print('Subject {0} has not had BedpostX performed. Stopping.'.format(subject))
        return False
        
    flag_file = '{0}/{1}/postproc.done'.format(output_dir, config_gen['flags_dir'])
    if os.path.isfile(flag_file):
        if not config_gen['clobber']:
            print('Subject {0} already post-processed. Skipping.'.format(subject))
            return False
        print('Subject {0} already post-processed. Clobbering.'.format(subject))
    
    bedpostx_dir = '{0}/bedpostX'.format(output_dir)

    # Clean up
    # Move bedpost directory to proper location
    dummy_dir = '{0}.bedpostX'.format(output_dir)
    cmd = 'mv {0} {1}'.format(dummy_dir, bedpostx_dir)
    print('\t{}'.format(cmd))
    err = run_fsl(cmd)
    if err:
        print('\tError cleaning up BedpostX [{0}]: {1}'.format(subject,err))
        return False

    output_img = '{0}/dwi_{1}.nii.gz' \
                    .format(output_dir, config_bpx['bet_suffix'])

    cmd = 'rm {0}/nodif*; rm {0}/bv*; mv {0}/data.nii.gz {1}' \
                    .format(output_dir, output_img)
    print('\t{}'.format(cmd))
    err = run_fsl(cmd)
    if err:
        print('\tError cleaning up DWI folder [{0}]: {1}'.format(subject,err))
        return False
    
    # Step 4: Warping to template space
    #      4.1: Linear transform (FLIRT)
    #      4.2: Non-linear warp (FNIRT)
    #      4.3: Inverse NL warp (INVWARP)
    
    # Warp to Mean 3G
    reg3g_dir = '{0}/reg3G'.format(output_dir)
        
    test_file = '{0}/Mean3G_warp2FA.nii.gz'.format(output_dir)
        
    if os.path.isfile(test_file) and not config_gen['clobber']:
        print('\tReg3D output exists for {}. Skipping.'.format(subject))
    else:
        if os.path.isdir(reg3g_dir):
            shutil.rmtree(reg3g_dir)
        os.mkdir(reg3g_dir)
        
        # Linear (FLIRT)
        cmd = '{0}flirt -in {1}/dti_FA.nii.gz ' \
                       '-ref utils/{2} ' \
                       '-out {3}/FA_lin2Mean3G.nii.gz ' \
                       '-omat {3}/FA_lin2Mean3G.mat ' \
                       '-bins 256 -cost corratio ' \
                       '-searchrx -180 180 -searchry -180 180 -searchrz -180 180 ' \
                       '-dof 12 -interp spline' \
                            .format(fsl_bin, output_dir, config_bpx['mean3g_ref'], reg3g_dir)
        
        print('\t{}'.format(cmd))
        err = run_fsl(cmd)
        if err:
            print('\tError warping to Mean3G (flirt) [{0}]: {1}'.format(subject,err))
            return False
        
        # Non-linear (FNIRT)
        cmd = '{0}fnirt --in={1}/dti_FA.nii.gz ' \
                       '--ref=utils/{2} ' \
                       '--refmask=utils/{3} ' \
                       '--aff={4}/FA_lin2Mean3G.mat ' \
                       '--config=utils/{5} ' \
                       '--cout={4}/FA_warp2Mean3G.nii.gz ' \
                       '--iout={4}/FA_nlin2Mean3G.nii.gz' \
                            .format(fsl_bin, output_dir, config_bpx['mean3g_ref'], \
                                    config_bpx['mean3g_mask'], reg3g_dir, \
                                    config_bpx['mean3g_config'])
        
        print('\t{}'.format(cmd))
        err = run_fsl(cmd)
        if err:
            print('\tError warping to Mean3G (fnirt) [{0}]: {1}'.format(subject,err))
            return False
        
        # Compute inverse warp
        cmd = '{0}invwarp --warp={1}/FA_warp2Mean3G.nii.gz ' \
                         '--out={1}/Mean3G_warp2FA.nii.gz ' \
                         '--ref={2}/dti_FA.nii.gz' \
                            .format(fsl_bin, reg3g_dir, output_dir)
                    
        print('\t{}'.format(cmd))
        err = run_fsl(cmd)
        if err:
            print('\tError warping to Mean3G (invwarp) [{0}]: {1}'.format(subject,err))
            return False
        else:
            print('\tDone computing transforms [{}].'.format(subject))
            
        # Make permissions wrx for owner AND group, read for other
        recursive_chmod(output_dir, 0o775)
            
        # Write flag file
        with open(flag_file,'w'):
            print('Post-processing: Wrote flag file {0}'.format(flag_file))
                
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
