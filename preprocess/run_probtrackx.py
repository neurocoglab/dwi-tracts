#!/usr/bin/env python

# Runs ProbtrackX on a single subject, for a specific seed ROI
# Requires that BedpostX has already been run for this subject
# Target ROIs and other parameters must be specified in a JSON file

# Command line arguments to this script:
# Arg1: subject ID
# Arg2: seed ROI
# Arg3: configuration file

import subprocess
import sys
import os
import csv
from subprocess import Popen
import shutil

# Load configuration parameters from JSON file
import json

def _json_object_hook(d): return namedtuple('X', d.keys())(*d.values())
def json2obj(data): return json.loads(data, object_hook=_json_object_hook)

def main():

    global config
    global roi

    roi = sys.argv[2]

    with open(sys.argv[3], 'r') as myfile:
        json_string=myfile.read() #.replace('\n', '')

    config = json.loads(json_string)
    config_gen = config['general']

    is_dryrun = config_gen['dryrun']

    # Run this subject
    subject = sys.argv[1]

    append = '';
    if is_dryrun:
        append = ' [DRY RUN]'

    print('Processing subject {0}{1}'.format(subject, append))
    if probtrackx_subject(subject):
        print('Finished subject {0}'.format(subject))
    else:
        print('Subject {0} failed'.format(subject))

def run_fsl(cmd):

    global config
    is_dryrun = config['general']['dryrun']

#     print('Dry run? {0}'.format(is_dryrun))

    if is_dryrun:
#         print('Not executing command');
        return ''
    else:
        sp = Popen(cmd, shell=True, stderr=subprocess.PIPE)
        out, err = sp.communicate()
        return err

def probtrackx_subject(subject):
    # Generate probabilistic streamlines between all pairs of ROIs

    global config

    # Get configs
    config_gen = config['general']
    config_bpx = config['bedpostx']
    config_ptx = config['probtrackx']

    if config_gen['verbose']:
        print(config_ptx)

    fsl_bin = config_gen['fsl_bin']

    subj = '{0}{1}'.format(config_gen['prefix'], subject)

    deriv_dir = '{0}/{1}'.format(config_gen['root_dir'], config_gen['deriv_dir'])

    # Subject-specific paths
    if len(config_gen['session']) > 0:
        subj_dir = '{0}/{1}/{2}/dwi'.format(deriv_dir, subj, config_gen['session'])
    else:
        subj_dir = '{0}/{1}'.format(deriv_dir, subj)
    bedpostx_dir = '{0}/bedpostX'.format(subj_dir)
    bedpostx_done = os.path.exists('{0}/xfms/eye.mat'.format(bedpostx_dir))
    probtrackx_dir = '{0}/probtrackX/{1}'.format(subj_dir, config_ptx['network_name'])
    rois_dir = config_ptx['roi_dir']

    if not os.path.isdir(probtrackx_dir):
        os.makedirs(probtrackx_dir)
    # else:
#         if config_gen['clobber']:
#             shutil.rmtree(probtrackx_dir)
#             os.makedirs(probtrackx_dir)

    invxfm_img     = '{0}/reg3G/FA_warp2Mean3G.nii.gz'.format(subj_dir)
    xfm_img     = '{0}/reg3G/Mean3G_warp2FA.nii.gz'.format(subj_dir)

#     print('{0} | {1}'.format(xfm_img, invxfm_img)

    # Check whether BedpostX output exists, otherwise fail
    if not bedpostx_done:
        print('No BedpostX output exists at {0}. Skipping subject. [{1}]'.format(bedpostx_dir, subject))
        return False

    if not os.path.exists(xfm_img) or not os.path.exists(invxfm_img):
        print('No warp images (reg3G) exist. Skipping subject. [{0}]'.format(subject))
        return False

    # Read ROI list
    rois = []
    target_rois = {}

    # If this is a JSON file, read as networks and targets
    if config_ptx['roi_list'].endswith('.json'):
        with open(config_ptx['roi_list'], 'r') as myfile:
            json_string=myfile.read()

        netconfig = json.loads(json_string)
        networks = netconfig['networks']
        targets = netconfig['targets']

        for net in networks:
            others = []
            for net2 in targets[net]:
                others = others + networks[net2]
            for roii in networks[net]:
                rois.append(roii)
                target_rois[roii] = others
    else:
        with open(config_ptx['roi_list'],'r') as roi_file:
            reader = csv.reader(roi_file)
            for row in reader:
                rois.append(row[0])

        # All ROIs are targets
        for roii in rois:
            targets = []
            for roi2 in rois:
                if roii is not roi2:
                    targets.append(roi2)
            target_rois[roii] = targets

    # For each ROI seed
    #for roi in rois:

    roi_file = '{0}/{1}.nii'.format(rois_dir, roi)
    roi_list = '{0}/others_{1}.txt'.format(probtrackx_dir, roi)

    cmd_add = ''

    # Build list of other ROIs
    with open(roi_list,'w') as listout:
        for roi2 in target_rois[roi]:
            # Add to ROI list
            listout.write('{0}/{1}.nii\n'.format(rois_dir, roi2))
            # Add to "others" stop image
            if not cmd_add:
                cmd_add = '{0}fslmaths {1}/{2}.nii'.format(fsl_bin, rois_dir, roi2)
            else:
                cmd_add = '{0} -add {1}/{2}.nii'.format(cmd_add, rois_dir, roi2)

    stop_img = '{0}/{1}_others.nii.gz'.format(probtrackx_dir, roi)
    cmd_add = '{0} {1}'.format(cmd_add, stop_img)

    if config_gen['verbose']:
        print(cmd_add)
    err = run_fsl(cmd_add)
    if err:
        print('\tError creating stop mask [{0}]: {1}'.format(subject,err))
        return False

    cmd_pre = '{0}probtrackx2 -V 0 --distthresh={1} --sampvox={2} --forcedir --opd --opathdir ' \
              '-x {3} -l --onewaycondition -c {4} --nsteps={5} --steplength={6} --nsamples={7} --fibthresh={8} --s2tastext ' \
              '--xfm={9} --invxfm={10} -s {11}/merged -m {11}/nodif_brain_mask' \
                .format(fsl_bin, config_ptx['distthresh'], config_ptx['sampvox'], roi_file, \
                       config_ptx['cthr'], config_ptx['nsteps'], config_ptx['steplength'], \
                       config_ptx['nsamples'], config_ptx['fibthresh'], xfm_img, invxfm_img, \
                       bedpostx_dir)

    cmd_net = ' --stop={0} -V 0 --waypoints={0} --waycond=OR --omatrix2 ' \
              ' --target2={0} --os2t --targetmasks={1} --otargetpaths' \
                .format(stop_img, roi_list)

    cmd = '{0} {1} --pd --dir={2}/{3}' \
                .format(cmd_pre, cmd_net, probtrackx_dir, roi)

    if config_gen['verbose']:
        print(cmd)

    err = run_fsl(cmd)
    if err:
        print('\tError running ProbtrackX [{0}]: {1}'.format(subject,err))
        return False

    cmd = '{0} {1} --pd --dir={2}/{3} -o FreeTracking' \
                .format(cmd_pre, cmd_net, probtrackx_dir, roi)
    if config_gen['verbose']:
        print(cmd)
    err = run_fsl(cmd)
    if err:
        print('\tError running ProbtrackX free-tracking [{0}]: {1}'.format(subject,err))
        return False

    print('\tDone tracking for ROI {0} [{1}]'.format(roi, subject))

    # Clean up
    cmd = 'rm {0}; rm {1}'.format(stop_img, roi_list)
    err = run_fsl(cmd)
    if err:
        print('\tError cleaning up ProbtrackX folder [{0}]: {1}'.format(subject,err))
        return False

    return True

if __name__ == '__main__':
    main()
