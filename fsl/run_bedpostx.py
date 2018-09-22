#!/usr/bin/python

import subprocess
import sys
import os
import csv
from subprocess import Popen 

# Load configuration parameters from JSON file
import json

def _json_object_hook(d): return namedtuple('X', d.keys())(*d.values())
def json2obj(data): return json.loads(data, object_hook=_json_object_hook)

config_gen = {}

def main():

	global config

	with open(sys.argv[2], 'r') as myfile:
		json_string=myfile.read() #.replace('\n', '')

	config = json.loads(json_string)
	config_gen = config['general']
	config_bpx = config['bedpostx']
	
	is_dryrun = config_gen['dryrun']
	
# 	print('config_gen file loaded')

	if not os.path.isdir(config_gen['root_dir']):
		raise Exception('Root dir does not exist: {0}'.format(config_gen['root_dir']))
	
	# Run this subject
	subject = sys.argv[1]
	
	append = '';
	if is_dryrun:
		append = ' [DRY RUN]'
	
	print('Processing subject {0}{1}'.format(subject, append))
	if process_subject(subject):
		print('Finished subject {0}'.format(subject))
	else:
		print('Subject {0} failed'.format(subject))

def run_fsl(cmd):

	global config
	is_dryrun = config['general']['dryrun']
	
# 	print('Dry run? {0}'.format(is_dryrun))
	
	if is_dryrun:
# 		print('Not executing command');
		return ''
	else:
		sp = Popen(cmd, shell=True, stderr=subprocess.PIPE)
		out, err = sp.communicate()
		return err
    
def process_subject(subject):
    
	global config
	
	config_gen = config['general']
	config_bpx = config['bedpostx']
    
	raw_dir = '{0}/{1}'.format(config_gen['root_dir'], config_gen['raw_dir'])
	deriv_dir = '{0}/{1}'.format(config_gen['root_dir'], config_gen['deriv_dir'])
    
    # Step 1: Eddy current correction
    # eddy_correct <4dinput> <4doutput> <reference_no>
	fsl_bin = config_gen['fsl_bin']
    
	subj = '{0}{1}'.format(config_gen['prefix'], subject)
	input_dir = '{0}/{1}/{2}/dwi' \
                        .format(raw_dir, subj, config_gen['session'])
	output_dir = '{0}/{1}/{2}/dwi' \
                        .format(deriv_dir, subj, config_gen['session'])
	if not os.path.isdir(output_dir):
		os.makedirs(output_dir)
    
	bedpostx_dir = '{0}/bedpostX' \
                    .format(output_dir)
	bedpostx_done = os.path.exists('{0}/xfms/eye.mat'.format(bedpostx_dir))
    
	if bedpostx_done and not config_gen['clobber']:
		print('\tBedpostX results already exist. Skipping those steps.')
	else:
    
		dwi_img = '{0}/data.nii.gz' \
                        .format(output_dir)

		input_img = '{0}/{1}_{2}_dwi.nii.gz' \
                        .format(input_dir, subj, config_gen['session'])

		output_img = '{0}/{1}_{2}_dwi_{3}.nii.gz' \
                        .format(output_dir, subj, config_gen['session'], config_bpx['eddy_suffix'])

		if os.path.exists(dwi_img) and not config_gen['clobber']:
			print('\tEddy output exists for {0}. Skipping.'.format(subject))
		else:
			cmd_eddy = '{0}eddy_correct {1} {2} {3}' \
                        .format(fsl_bin, input_img, output_img, config_bpx['eddy_reference'])
#             print('\t{0}'.format(cmd_eddy))
			err = run_fsl(cmd_eddy)
			if err:
				print('\tError with eddy correction [{0}]: {1}'.format(subject,err))
				return False
            
			print('\tDone eddy correction [{0}].'.format(subject))   

            # Rename image to data.nii.gz
			cmd = 'mv {0} {1}' \
                            .format(output_img, dwi_img);
#             print('\t{0}'.format(cmd))
			err = run_fsl(cmd)
			if err:
				print('\tError renaming eddy output image [{0}]: {1}'.format(subject,err))
				return False


        # Step 2: Brain extraction (BET)
        #      2.1: Extract B0 images (using fslroi and bvals file)
        #      2.2: Get mean B0 image (using fslmerge)
        #      2.3: Run BET on the mean B0

        # Check if output already exists
		bet_img = '{0}/{1}_{2}_dwi_{3}' \
                        .format(output_dir, subj, config_gen['session'], config_bpx['bet_suffix'])

		nodif_img = '{0}/nodif.nii.gz'.format(output_dir)

		if (os.path.exists(bet_img) or os.path.exists(nodif_img)) and not config_gen['clobber']:
			print('\tBET output exists for {0}. Skipping.'.format(subject))
		else:

			input_img = dwi_img;
			bval_file = '{0}/{1}_{2}_dwi.bval' \
                            .format(input_dir, subj, config_gen['session'])

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
#                 print('\t{0}'.format(cmd))
				err = run_fsl(cmd)
				if err:
					print('\tError extracting B0 images [{0}]: {1}'.format(subject,err))
					return False

            # Merge into a 4D image
			input_img = nodif_img
			cmd = '{0}fslmerge -t {1} {2}/xno*.nii.gz' \
                            .format(fsl_bin, input_img, output_dir)
#             print('\t{0}'.format(cmd))
			err = run_fsl(cmd)
			if err:
				print('\tError merging B0 images [{0}]: {1}'.format(subject,err))
				return False

            # Compute mean B0 from this image
			cmd = '{0}fslmaths {1} -Tmean {1}' \
                            .format(fsl_bin, input_img)
#             print('\t{0}'.format(cmd))
			err = run_fsl(cmd)
			if err:
				print('\tError computing B0 image [{0}]: {1}'.format(subject,err))
				return False

            # Run BET on the output
			cmd_bet = '{0}bet2 {1} {2} -m -f 0.3' \
                        .format(fsl_bin, input_img, bet_img)
#             print('\t{0}'.format(cmd_bet))
			err = run_fsl(cmd_bet)
			if err:
				print('\tError with brain extraction (BET) [{0}]: {1}'.format(subject,err))
				return False
            
            # Rename the BET mask
			cmd = 'mv {0}_mask.nii.gz {1}/nodif_brain_mask.nii.gz' \
                        .format(bet_img, output_dir)
#             print('\t{0}'.format(cmd))
			err = run_fsl(cmd)
			if err:
				print('\tError removing residual B0 images [{0}]: {1}'.format(subject,err))
				return False

			bet_mask_img = '{0}/nodif_brain_mask.nii.gz'.format(output_dir)

            # Clean up the mess
			cmd = 'rm {0}/xno*.nii.gz; rm {0}/sub*'.format(output_dir);
# 			print('\t{0}'.format(cmd))
			run_fsl(cmd) # Ignore any error here
            
			print('\tDone brain extraction (BET) [{0}].'.format(subject))

		# Step 3: BedpostX (Fitting model)
		#      3.1: Copy bvals and bvecs
		#      3.2: Run dtifit
		#      3.3: Run BedpostX

		# Copy bvals and bvecs
		cmd = 'cp {0}/{1}_{2}_dwi.bvec {3}/bvecs; cp {0}/{1}_{2}_dwi.bval {3}/bvals' \
						.format(input_dir, subj, config_gen['session'], output_dir)

#         print('\t{0}'.format(cmd))
		err = run_fsl(cmd)
		if err:
			print('\tError copying bvecs and bvals [{0}]: {1}'.format(subject,err))
			return False

		# Run dtifit
		if os.path.exists('{0}/dti_FA.nii.gz'.format(output_dir)) and not config_gen['clobber']:
			print('\tDTI output exists for {0}. Skipping.'.format(subject))
		else:
			cmd = '{0}dtifit -k {1} -m {2} -r {3}/bvecs -b {3}/bvals -o {3}/dti' \
						.format(fsl_bin, dwi_img, bet_mask_img, output_dir)
#             print('\t{0}'.format(cmd))
			err = run_fsl(cmd)
			if err:
				print('\tError running dtifit [{0}]: {1}'.format(subject,err))
				return False
	   
			print('\tDone dtifit [{0}].'.format(subject))

		# Run BedpostX
		cmd = '{0}bedpostx {1}/' \
					.format(fsl_bin, output_dir)
#         print('\t{0}'.format(cmd))
		err = run_fsl(cmd)
		if err:
			print('\tError running BedpostX [{0}]: {1}'.format(subject,err))
			return False
        
        # Clean up
        # Move bedpost directory to proper location
		dummy_dir = '{0}/{1}/{2}/dwi.bedpostX' \
                        .format(deriv_dir, subj, config_gen['session'])
		if os.path.exists(dummy_dir):
			print('Dummy exists')
			if os.path.exists(bedpostx_dir):
				cmd = 'rm -rf {0}'.format(bedpostx_dir)
				run_fsl(cmd)
				print('Ran command: {0}'.format(cmd))
			cmd = 'mv {0} {1}' \
							.format(dummy_dir, bedpostx_dir)
	#         print('\t{0}'.format(cmd))
			err = run_fsl(cmd)
			print('Ran command: {0}'.format(cmd))
			if err:
				print('\tError running BedpostX [{0}]: {1}'.format(subject,err))
				return False

		output_img = '{0}/{1}_{2}_dwi_{3}.nii.gz' \
                        .format(output_dir, subj, config_gen['session'], config_bpx['bet_suffix'])

		cmd = 'rm {0}/nodif*; rm {0}/bv*; mv {0}/data.nii.gz {1}' \
                        .format(output_dir, output_img)
#         print('\t{0}'.format(cmd))
		err = run_fsl(cmd)
		if err:
			print('\tError cleaning up DWI folder [{0}]: {1}'.format(subject,err))
			return False
        
		print('\tDone BedpostX [{0}].'.format(subject))
    
    # Step 4: Warping to template space
    #      4.1: Linear transform (FLIRT)
    #      4.2: Non-linear warp (FNIRT)
    #      4.3: Inverse NL warp (INVWARP)
    
    # Warp to Mean 3G
	reg3g_dir = '{0}/reg3G' \
                    .format(output_dir)
        
	if os.path.exists('{0}/Mean3G_warp2FA.nii.gz'.format(reg3g_dir)) and not config_gen['clobber']:
		print('\tReg3D output exists for {0}. Skipping.'.format(subject))
	else:
		if not os.path.isdir(reg3g_dir):
			os.makedirs(reg3g_dir)
        
        # Linear (FLIRT)
		cmd = '{0}flirt -in {1}/dti_FA.nii.gz ' \
                       '-ref utils/{2} ' \
                       '-out {3}/FA_lin2Mean3G.nii.gz ' \
                       '-omat {3}/FA_lin2Mean3G.mat ' \
                       '-bins 256 -cost corratio ' \
                       '-searchrx -180 180 -searchry -180 180 -searchrz -180 180 ' \
                       '-dof 12 -interp spline' \
                            .format(fsl_bin, output_dir, config_bpx['mean3g_ref'], reg3g_dir)
        
#         print('\t{0}'.format(cmd))
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
        
#         print('\t{0}'.format(cmd))
		err = run_fsl(cmd)
		if err:
			print('\tError warping to Mean3G (fnirt) [{0}]: {1}'.format(subject,err))
			return False
        
        # Compute inverse warp
		cmd = '{0}invwarp --warp={1}/FA_warp2Mean3G.nii.gz ' \
                         '--out={1}/Mean3G_warp2FA.nii.gz ' \
                         '--ref={2}/dti_FA.nii.gz' \
                            .format(fsl_bin, reg3g_dir, output_dir)
                    
#         print('\t{0}'.format(cmd))
		err = run_fsl(cmd)
		if err:
			print('\tError warping to Mean3G (invwarp) [{0}]: {1}'.format(subject,err))
			return False
		else:
			print('\tDone computing transforms [{0}].'.format(subject))
                
	return True
    
if __name__ == '__main__':
    main()
