
# Module "dwitracts"
import statsmodels.api as sm
import math
import nibabel as nib
from nilearn import image, masking
import scipy.stats as stats
import numpy as np
import json
import csv
import os
import glob
from . import utils
import shutil
import time
from tqdm import tnrange, tqdm_notebook, tqdm
import pandas as pd

class DwiTracts:
    
    def __init__(self, params):
        self.params = params
        self.is_init = False
        
    def _json_object_hook(d): return namedtuple('X', d.keys())(*d.values())
    def json2obj(data): return json.loads(data, object_hook=_json_object_hook)
        
    # Initialise this object with its current parameters
    def initialize(self):
        
        params_gen = self.params['general']
        self.source_dir = params_gen['source_dir']
        self.project_dir = os.path.join(self.source_dir, params_gen['project_dir'])
        self.tracts_dir = os.path.join(self.project_dir, params_gen['tracts_dir'], params_gen['network_name'])
        self.rois_dir = os.path.join(self.source_dir, params_gen['rois_dir'], params_gen['network_name'])

        if params_gen['debug']:
            self.debug_dir = os.path.join(self.tracts_dir, 'debug')
            print('Creating debug directory at {0}'.format(self.debug_dir))
            if not os.path.isdir(self.debug_dir):
                os.makedirs(self.debug_dir)

        if not os.path.isdir(self.tracts_dir):
            os.makedirs(self.tracts_dir)        

        self.avr_dir = os.path.join(self.tracts_dir, 'average')
        if not os.path.isdir(self.avr_dir):
            os.makedirs(self.avr_dir)
        self.dist_dir = os.path.join(self.tracts_dir, 'dist')
        if not os.path.isdir(self.dist_dir):
            os.makedirs(self.dist_dir)
        self.polyline_dir = os.path.join(self.tracts_dir, 'polylines')
        if not os.path.isdir(self.polyline_dir):
            os.makedirs(self.polyline_dir)
        self.gauss_dir = os.path.join(self.tracts_dir, 'gaussians')
        if not os.path.isdir(self.gauss_dir):
            os.makedirs(self.gauss_dir)
        self.gauss_fail_dir = os.path.join(self.tracts_dir, 'gaussians.failed')
        if not os.path.isdir(self.gauss_fail_dir):
            os.makedirs(self.gauss_fail_dir)
        self.final_dir = os.path.join(self.tracts_dir, 'final')
        if not os.path.isdir(self.final_dir):
            os.makedirs(self.final_dir)
        self.final_fail_dir = os.path.join(self.tracts_dir, 'final.failed')
        if not os.path.isdir(self.final_fail_dir):
            os.makedirs(self.final_fail_dir)
        self.log_dir = os.path.join(self.tracts_dir, 'log')
        if not os.path.isdir(self.log_dir):
            os.makedirs(self.log_dir)
        self.avdir_dir = os.path.join(self.tracts_dir, 'avrdir')
        if not os.path.isdir(self.avdir_dir):
            os.makedirs(self.avdir_dir)
            
        # Subjects
        subjects = [];
        with open(params_gen['subjects_file']) as subj_file:
            reader = csv.reader(subj_file)
            for row in reader:
                subjects.append(row[0])
        self.subjects = sorted(subjects)
        
        # ROIs
        rois = []
        with open(params_gen['rois_list'],'r') as roi_file:
            reader = csv.reader(roi_file)
            for row in reader:
                rois.append(row[0])
        
        # Determine proper suffix
        roi_suffix = 'nii'
        roi = rois[0]
        roi_file = '{0}/{1}.{2}'.format(self.rois_dir, roi, roi_suffix)
        if not os.path.isfile(roi_file):
            roi_suffix = 'nii.gz'
            roi_file = '{0}/{1}.{2}'.format(self.rois_dir, roi, roi_suffix)
            if not os.path.isfile(roi_file):
                print('No ROI file found: {0}.'.format(roi_file))

        V_img = nib.load(roi_file)
        V_img.header.set_data_dtype(np.float32)

        V = np.zeros(V_img.shape)
        V_min_avr = {}
        V_bin_avr = {}

        # Determine ROI pairs to process
        row_i = {}
        roi_pairs = {}

        if 'roi_pairs' in params_gen and len(params_gen['roi_pairs']) > 0:
            with open(params_gen['roi_pairs'], 'r', newline='') as pairs_file:
                for line in pairs_file:
                    parts = line.strip().split(',')
                    if parts[0] in V_min_avr:
                        row_i = V_min_avr[parts[0]]
                        roi_pairs[parts[0]].append(parts[1])
                    else:
                        row_i = {}
                        roi_pairs[parts[0]] = [parts[1]]
                    row_i[parts[1]] = np.squeeze(V)
                    V_min_avr[parts[0]] = row_i
                    V_bin_avr[parts[0]] = row_i.copy()
        else:    

            # Instantiate list of roi pair volumes
            for i in range (0,len(rois)):
                roi_i = rois[i]
                row_i = {}
                roi_pairs[roi_i] = []
                for j in range (i+1,len(rois)):
                    roi_j = rois[j]
                    roi_pairs[roi_i].append(roi_j)
                    row_i[roi_j] = np.squeeze(V)
                V_min_avr[roi_i] = row_i
                V_bin_avr[roi_i] = row_i.copy()
        
        self.rois = rois
        self.roi_pairs = roi_pairs
        self.V_min_avr = V_min_avr
        self.V_bin_avr = V_bin_avr
        self.V_img = V_img
        self.roi_suffix = roi_suffix
        
        self.is_init = True
        return True
        
        
    # Compute average bidirectional tract distributions based upon 
    # probabilistic tractography.
    #
    # Tractography (Bedpostx, Probtrackx) must have already have been performed
    #
    # verbose:        Whether to print messages to the console
    # clobber:        Whether to overwrite existing output
    #
    def compute_bidirectional_averages( self, clobber=False, verbose=False):
    
        if not self.is_init:
            print( 'DwiTracts object has not yet been initialized!' )
            return False
        
        params_gen = self.params['general']
        output_list = '{0}/subjects_avr_{1}.list'.format(self.tracts_dir, params_gen['network_name'])
        if os.path.isfile( output_list ) and not clobber:
            print(' Output file {0} exists, use "clobber=True" to overwrite!'.format(output_list) )
            return False
                                                         
        
        params_avt = self.params['average_tracts']
        fwhm = params_avt['fwhm']
        bin_thres = params_avt['bin_thres']

        avr_dir = os.path.join(self.tracts_dir, 'average')
        if not os.path.isdir(avr_dir):
            os.makedirs(avr_dir)

        if verbose:
            print('Processing subject-wise tractography...')
        subj_passed = []  

        s = 1
        for subject in tqdm_notebook(self.subjects, 'Progress'):
            prefix_sub = '{0}{1}'.format(params_gen['prefix'], subject)
            if verbose:
                print('  Processing subject: {0} ({1} of {2})'.format(subject, s, len(self.subjects)))
            subject_dir = os.path.join(self.project_dir, params_gen['deriv_dir'], prefix_sub, \
                                       params_gen['sub_dirs'], params_gen['dwi_dir'], \
                                       'probtrackX', params_gen['network_name'])
            passed = False

            for roi_i in self.roi_pairs:
                for roi_j in self.roi_pairs[roi_i]:
                    try:
                        V_ij = nib.load('{0}/{1}/target_paths_{2}.nii.gz'.format(subject_dir, roi_i, roi_j)).get_fdata()
                        V_ji = nib.load('{0}/{1}/target_paths_{2}.nii.gz'.format(subject_dir, roi_j, roi_i)).get_fdata()
                        V_min = np.minimum(V_ij,V_ji) # Minimum of A->B and B->A
                        mx = V_min.max()
                        if mx > 0:
                            V_min = np.divide(V_min, mx) # Normalize to maximum (if not all zeros)      
                        V_min = np.multiply(V_min, np.greater(V_min, bin_thres).astype(np.float16)) # Binarize to bin_thresh
                        V_bin = np.greater(V_min, 0).astype(np.float16)                           
                        self.V_min_avr[roi_i][roi_j] = np.add(self.V_min_avr[roi_i][roi_j], V_min)
                        self.V_bin_avr[roi_i][roi_j] = np.add(self.V_bin_avr[roi_i][roi_j], V_bin)
                        passed = True
                    except Exception as e:
                        if verbose:
                            print('  ** Error processing subject {0} [skipping]: {1}'.format(subject, e))
                        return False

            if passed:
                subj_passed.append(subject)
                s = s + 1

        # Write passed subjects to file
        subjects = subj_passed
        with open(output_list, 'w') as fileout:
            for subject in self.subjects:
                fileout.write('{0}\n'.format(subject))

        # Divide sums
        for roi_i in self.roi_pairs:
            for roi_j in self.roi_pairs[roi_i]:
                self.V_min_avr[roi_i][roi_j] = np.divide(self.V_min_avr[roi_i][roi_j], len(self.subjects))
                self.V_bin_avr[roi_i][roi_j] = np.divide(self.V_bin_avr[roi_i][roi_j], len(self.subjects))

        if verbose:
            print('Done processing subjects.\nSmoothing averages and saving...')

        # Normalize, smooth, and save results
        for roi_i in tqdm_notebook(self.roi_pairs, desc='Smoothing'):
            for roi_j in self.roi_pairs[roi_i]:

                if verbose:
                    print('  Smoothing tract for {0} | {1}'.format(roi_i, roi_j))

                V = self.V_min_avr[roi_i][roi_j]
                # Normalize
                mx = np.max(V)
                if (mx > 0):
                    V = np.divide(V, mx)
                img = nib.Nifti1Image(V, self.V_img.affine, self.V_img.header)
                # Smooth
                V_img = image.smooth_img(img, fwhm)
                # Save
                nib.save(self.V_img, '{0}/avr_min_tract_counts_{1}_{2}.nii.gz'.format(avr_dir, roi_i, roi_j))

                V = self.V_bin_avr[roi_i][roi_j]
                # Normalize
                mx = np.max(V)
                if (mx > 0):
                    V = np.divide(V, mx)
                V_img = nib.Nifti1Image(V, self.V_img.affine, self.V_img.header)
                # Smooth
                V_img = image.smooth_img(img, fwhm)
                # Save
                nib.save(img, '{0}/avr_bin_tract_counts_{1}_{2}.nii.gz'.format(avr_dir, roi_i, roi_j))

        if verbose:
            print('Done smoothing and saving.')
            
        return True
            
    # Compute stepwise distanceas along tracts. 
    #
    # verbose:        Whether to print messages to the console
    # clobber:        Whether to overwrite existing output
    #
    def compute_tract_distances( self, clobber=False, verbose=False):
        
        if not self.is_init:
            print( 'DwiTracts object has not yet been initialized!' )
            return False
        
        if verbose:
            print('\n== Computing tract distances ==\n')

        # Create directories
        params_gauss = self.params['gaussians']
#         sigma_axial = params_gauss['fwhm_axial'] / (2.0*math.sqrt(2.0*math.log(2.0)))
#         sigma_radial = params_gauss['fwhm_radial'] / (2.0*math.sqrt(2.0*math.log(2.0)))
#         max_seg_length = params_gauss['max_seg_length']
#         gauss_max_radius = params_gauss['gauss_max_radius']
#         tract_thresh = params_gauss['tract_thresh']
        
        threshold = self.params['average_tracts']['bin_thres']
        seed_dilate = params_gauss['seed_dilate']

        V_rois = {}
        V_dists = {}

        # Load ROI volumes
        for roi in self.rois:
            roi_file = '{0}/{1}.{2}'.format(self.rois_dir, roi, self.roi_suffix)
            img = nib.load(roi_file)
            V_rois[roi] = np.squeeze(img.get_fdata())

        rois_a = []
        rois_b = []
        for roi_i in self.roi_pairs:
            for roi_j in self.roi_pairs[roi_i]:
                rois_a.append(roi_i)
                rois_b.append(roi_j)  

        # Loop through ROI pairs and compute stepwise distance using a flood-fill
        # approach
        for c in tqdm_notebook(range (0,len(rois_a)), desc="Progress"):
            roi_a = rois_a[c]
            V_roi_a = V_rois[roi_a]
            roi_b = rois_b[c]
            V_roi_b = V_rois[roi_b]
            
            dist_img_ab = '{0}/tract_dist_{1}_{2}.nii.gz'.format(self.dist_dir, roi_a, roi_b)
            dist_img_ba = '{0}/tract_dist_{1}_{2}.nii.gz'.format(self.dist_dir, roi_b, roi_a)

            # Check if this has already been done and if so load it instead of computing it
            # (this can be redone if clobber=True)
            if not clobber and os.path.isfile(dist_img_ab) and os.path.isfile(dist_img_ba):
                V_dists['{0}_{1}'.format(roi_a, roi_b)] = nib.load(dist_img_ab).get_fdata()
                V_dists['{0}_{1}'.format(roi_b, roi_a)] = nib.load(dist_img_ba).get_fdata()
                if verbose:
                    print('  Images already exist for {0}/{1}; loading from file.'.format(roi_a, roi_b))
            else:

                tract_file = '{0}/avr_bin_tract_counts_{1}_{2}.nii.gz'.format(self.avr_dir, roi_a, roi_b)
                if not os.path.isfile(tract_file):
                    tract_file = '{0}/avr_bin_tract_counts_{1}_{2}.nii.gz'.format(self.avr_dir, roi_b, roi_a)

                if not os.path.isfile(tract_file):
                    if verbose:
                        print('  **Warning: No file found for ROI pair {0}/{1}! Skipping...'.format(roi_a, roi_b))
                else:
                    # Load and threshold average tract
                    V_tract = nib.load(tract_file).get_fdata()

                    V_mask = np.greater(V_tract, threshold).astype(np.float)
                    V_tract = np.multiply(V_mask, V_tract)

                    # Compute voxel steps from seed as distances
                    V_dist_a = utils.get_tract_dist(V_mask, V_roi_a, seed_dilate, V_stop=V_roi_b)
                    V_dist_b = utils.get_tract_dist(V_mask, V_roi_b, seed_dilate, V_stop=V_roi_a)
                    V_dist = np.logical_and(V_dist_a>0, V_dist_b>0)

                    if not np.any(V_dist):
                        if verbose:
                            print('  ** Warning: Tracts do not overlap for {0}/{1}. Skipping'.format(roi_a, roi_b))
                    else:

                         # Write distances to file
                        img = nib.Nifti1Image(V_dist_a, self.V_img.affine, self.V_img.header)
                        nib.save(img, dist_img_ab)
                        img = nib.Nifti1Image(V_dist_b, self.V_img.affine, self.V_img.header)
                        nib.save(img, dist_img_ba)

                        if verbose:
                            print('  Done tract {0}/{1}'.format(roi_a, roi_b))

        if verbose:
            print('\n== Done computing tract distances ==\n')
            
        return True
            
            
    # Get center polylines, generate Gaussian uncertainty fields around
    # these polylines, and multiply this with original streamline 
    # distributions to obtain final unidirectional tracts
    #
    # The goal is to estimate the "core" trajectory of each tract between
    # ROI pairs, if this is estimable. 
    #
    # This routine will also generate bidirectional tract estimates, 
    # determined by the average value for each individual estimate
    #
    # Determine polylines representing center of average tract, if they
    # exist. Apply a Gaussian weighting centered on these polylines.
    # Multiply this with the original average tract.   
    #
    # verbose:        Whether to print messages to the console
    # clobber:        Whether to overwrite existing output 
    #
    def estimate_unidirectional_tracts( self, verbose=False, clobber=False ):
        
        # TODO: check flag files for each step to ensure prequisites are met
        if not self.is_init:
            print('DwiTracts object not initialized!')
            return False
        
        if verbose:
            print('\n== Generating polylines and Gaussians ==\n')

        params_gen = self.params['general']
        params_gauss = self.params['gaussians']
        sigma_axial = params_gauss['fwhm_axial'] / (2.0*math.sqrt(2.0*math.log(2.0)))
        sigma_radial = params_gauss['fwhm_radial'] / (2.0*math.sqrt(2.0*math.log(2.0)))
        threshold = params_gauss['threshold']
        max_seg_length = params_gauss['max_seg_length']
        gauss_max_radius = params_gauss['gauss_max_radius']
        tract_thresh = params_gauss['tract_thresh']
        mask_dilate = params_gauss['mask_dilate']

        V_rois = {}
        for roi in self.rois:
            roi_file = '{0}/{1}.{2}'.format(self.rois_dir, roi, self.roi_suffix)
            img = nib.load(roi_file)
            V_rois[roi] = img.get_fdata()

        tract_failed = {}

        ts = time.gmtime()
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", ts)

        # Set up logging
        fail_log = '{0}/failed'.format(self.log_dir)
        with open(fail_log, 'a') as logfile:
            logfile.write('\n\n--New run: {0}\n'.format(timestamp))

        success_log = '{0}/success'.format(self.log_dir)
        with open(success_log, 'a') as logfile:
            logfile.write('\n\n--New run: {0}\n'.format(timestamp))

        violations_log = '{0}/violations'.format(self.log_dir)
        with open(violations_log, 'a') as logfile:
            logfile.write('\n\n--New run: {0}\n'.format(timestamp))

        N_roi = len(self.rois)
        tract_list_ab = []
        tract_list_ba = []
        rois_a = []
        rois_b = []

        for roi_i in self.roi_pairs:
            for roi_j in self.roi_pairs[roi_i]:
                tract_name = '{0}_{1}'.format(roi_i,roi_j)
                tract_list_ab.append(tract_name)
                tract_name = '{0}_{1}'.format(roi_j,roi_i)
                tract_list_ba.append(tract_name)
                rois_a.append(roi_i)
                rois_b.append(roi_j)  

        c = 0

        # Loop through each ROI pair and determine core polylines and gaussians
        # surrounding these polylines (uncertainty field)
        for ii in tqdm_notebook(range(0,len(tract_list_ab)), desc='Progress'):

            tract_name_ab = tract_list_ab[ii]
            tract_name_ba = tract_list_ba[ii]

            roi_a = rois_a[c]
            roi_b = rois_b[c]
            c += 1

            # Target mask is both ROIs
            V_target_a = np.squeeze(V_rois[roi_a])
            V_target_b = np.squeeze(V_rois[roi_b])
            tract_failed[tract_name] = False

            dist_file_ab = '{0}/tract_dist_{1}_{2}.nii.gz'.format(self.dist_dir, roi_a, roi_b)
            dist_file_ba = '{0}/tract_dist_{1}_{2}.nii.gz'.format(self.dist_dir, roi_b, roi_a)

            tract_file = '{0}/avr_min_tract_counts_{1}_{2}.nii.gz'.format(self.avr_dir, roi_a, roi_b)
            if not os.path.isfile(tract_file):
                tract_file = '{0}/avr_min_tract_counts_{1}_{2}.nii.gz'.format(self.avr_dir, roi_b, roi_a)

            if not os.path.isfile(dist_file_ab) and not os.path.isfile(dist_file_ba):
                if verbose:
                    print('  **Warning: No distances file found for ROI pair {0}/{1}! Skipping...'.format(roi_a, roi_b))
                tract_failed[tract_name] = True
                # Log failure
                with open(fail_log,'a') as logfile:
                    logfile.write('{0}|{1},File not found\n'.format(roi_a,roi_b))
            else:

                if (not clobber and
                    os.path.isfile('{0}/tract_final_{1}.nii.gz'.format(self.final_dir, tract_name))):
                    if verbose:
                        print('  Output for {0} already exists. Skipping this tract.'.format(tract_name))

                else:

                    # Dilate A and B to use as stop masks
                    if mask_dilate:
                        V_target_a = utils.dilate_mask(np.greater(V_target_a,0))
                        V_target_b = utils.dilate_mask(np.greater(V_target_b,0))

                    if params_gen['debug']:
                        # Write to debug file
                        img = nib.Nifti1Image(V_target_a, self.V_img.affine, self.V_img.header)
                        nib.save(img, '{0}/dilated_target_a_{1}.nii.gz'.format(self.debug_dir, tract_name))
                        img = nib.Nifti1Image(V_target_b, self.V_img.affine, self.V_img.header)
                        nib.save(img, '{0}/dilated_target_b_{1}.nii.gz'.format(self.debug_dir, tract_name))

                    # Load tract file
                    V_tract = nib.load(tract_file).get_fdata()
                    V_orig = V_tract.copy()
                    # Threshold
                    V_tract[np.less(V_tract, threshold)] = 0

                    dist_files = []; suffixes = []; tract_names = [];
                    if os.path.isfile(dist_file_ab):
                        dist_files.append(dist_file_ab)
                        suffixes.append('ab')
                        tract_names.append(tract_name_ab)
                    if os.path.isfile(dist_file_ba):
                        dist_files.append(dist_file_ba)
                        suffixes.append('ba')
                        tract_names.append(tract_name_ba)

                    # Compute polylines + gaussians in both directions
                    for dist_file, suffix, tract_name in zip(dist_files, suffixes, tract_names):

                        if suffix is 'ab':
                            V_target_1 = V_target_a
                            V_target_2 = V_target_b
                        else:
                            V_target_1 = V_target_b
                            V_target_2 = V_target_a

                        V_dist = np.round(nib.load(dist_file).get_fdata())
                        
                        if params_gen['debug']:
                            print(dist_file)
                            print(suffix)
                            print(tract_name)

                        if V_dist is None:
                            tract_failed[tract_name] = True
                            if verbose:
                                print('  **Warning: Skipping {0}; tract not found.'.format(tract_name))
                            # Log failure
                            with open(fail_log,'a') as logfile:
                                logfile.write('{0},Tract not found\n'.format(tract_name))
                        else:
                            dist_max = int(V_dist.max())
                            if params_gen['debug']:
                                print('  Tract {0} ({1}): max_dist={2}'.format(tract_name, suffix, dist_max))

                            # Compute polylines
                            maxes = np.array([])
                            constraints_failed = 0
                            found_target_a = False
                            found_target_b = False

                            # Start in the middle to avoid endpoint madness
                            d_start = round(dist_max/2)

                            maxes = np.zeros((dist_max,3), dtype=int)

                            for k in tqdm_notebook(range(0, d_start), desc='Distances'.format(c)):

                                for oe in [0,1]:

                                    if oe == 0:
                                        if found_target_a:
                                            d = -1
                                        else:
                                            d = d_start - k
                                    else:
                                        if found_target_b:
                                            d = -1
                                        else:
                                            d = d_start + k - 1

                                    if d >= 0 and d <= dist_max:

                                        T_idx = np.equal(V_dist,d)
                                        T = V_tract[T_idx]
                                        T_idx = np.flatnonzero(T_idx)
                                        sidx = T.argsort()[::-1]
                                        idx_max = np.unravel_index(T_idx[sidx[0]], V_tract.shape)

                                        if k == 0:
                                            maxes[d, :] = idx_max
                                        else:
                                            if oe == 0:
                                                idx_prev = maxes[d+1,:]
                                            else:
                                                idx_prev = maxes[d-1,:]

                                            # Find maximal value which satisfies length constraint
                                            r = 0
                                            is_valid = False
                                            while not is_valid and r < len(T):
                                                idx_r = np.unravel_index(T_idx[sidx[r]], V_tract.shape)
                                                seg_length = math.sqrt((idx_r[0]-idx_prev[0])**2 + \
                                                                       (idx_r[1]-idx_prev[1])**2 + \
                                                                       (idx_r[2]-idx_prev[2])**2)
                                                if seg_length < max_seg_length:
                                                    maxes[d,:] = idx_r
                                                    is_valid = True
                                                r+=1
                                            if not is_valid:
                                                idx_r = np.unravel_index(T_idx[sidx[0]], V_tract.shape)
                                                maxes[d,:] = idx_r
                                                seg_length = math.sqrt((idx_r[0]-idx_prev[0])**2 + \
                                                                       (idx_r[1]-idx_prev[1])**2 + \
                                                                       (idx_r[2]-idx_prev[2])**2)
                                                # Failed to meet constraint; add anyway but note failure
                                                constraints_failed += 1
                                                if params_gen['debug']:
                                                    print('   * Debug: Constraint fail: len ({0}) > max_len({1})'. \
                                                          format(seg_length, max_seg_length))

                                        # Is this voxel adjacent to target? Then ignore any further distances
                                        if not found_target_a and np.any(np.logical_and(np.equal(V_dist,d), \
                                                                                        np.greater(V_target_1,0))):
                                            if params_gen['debug']:
                                                print('  Tract {0} ({1}): Found target A @ d={2}.'.format(tract_name, suffix, d))
                                            found_target_a = True

                                        if not found_target_b and np.any(np.logical_and(np.equal(V_dist,d), \
                                                                                        np.greater(V_target_2,0))):
                                            if params_gen['debug']:
                                                print('  Tract {0} ({1}): Found target B @ d={2}.'.format(tract_name, suffix, d))
                                            found_target_b = True

                                        if found_target_a and found_target_b:
                                            break

                                    # If target was found, we're done
                                    if found_target_a and found_target_b:
                                        break

                            found_target = found_target_a and found_target_b

                            # Only keep non-zero rows
                            maxes = maxes[~np.all(maxes == 0, axis=1)]

                            my_target_gauss = self.gauss_fail_dir
                            my_target_final = self.final_fail_dir

                            if constraints_failed > 0:
                                if verbose:
                                    print('  Tract {0}: Polyline had {1} constraint violation(s).' \
                                              .format(tract_name, constraints_failed))
                                with open(violations_log, 'a') as logfile:
                                    logfile.write('{0},{1}\n'.format(tract_name, constraints_failed))

                                if params_gauss['fail_constraints']:
                                    tract_failed[tract_name] = True
                                else:
                                    my_target_gauss = self.gauss_dir
                                    my_target_final = self.final_dir

                            elif not found_target:
                                if verbose:
                                    print('  Tract {0}: Did not find target ROI.'.format(tract_name))
                                tract_failed[tract_name] = True

                                with open(fail_log, 'a') as logfile:
                                    logfile.write('{0},{1}\n'.format(tract_name, dist_max))
                            else:
                                my_target_gauss = self.gauss_dir
                                my_target_final = self.final_dir
                                with open(success_log, 'a') as logfile:
                                    logfile.write('{0},{1}\n'.format(tract_name, dist_max))

                            if maxes.size == 0:
                                if verbose:
                                    print('Maxes not found for tract {0}!? Failing this tract.'.format(tract_name))
                                tract_failed[tract_name] = True
                                with open(fail_log, 'a') as logfile:
                                    logfile.write('{0},{1}\n'.format(tract_name, dist_max))
                            else:

                                # Smooth the polyline defined by maxes at each distance
                                maxes_vox = np.round(utils.smooth_polyline_ma(maxes, 3))

                                # Transform from voxel to world coordinates and smooth
                                maxes = utils.smooth_polyline_ma(utils.voxel_to_world(maxes, self.V_img.header))

                                # Write polyline to Mgui format
                                utils.write_polyline_mgui(maxes, '{0}/maxes_{1}_sm3.poly3d'.format(self.polyline_dir, \
                                                                                                   tract_name), tract_name)

                                # For each voxel, derive Gaussian for neighbours within gauss_max_radius
                                V_gauss = np.zeros(V_tract.shape)
                                max_radius = round(gauss_max_radius)
                                nv = maxes_vox.shape[0]

                                for v in tqdm_notebook(range(1, nv),desc='Gaussians'):
                                    p = maxes_vox[v,:]
                                    p0 = maxes_vox[v-1,:]
                                    v_axis = p - p0

                                    v_len = np.linalg.norm(v_axis)
                                    if v_len == 0:
                                        if params_gen['debug']:
                                            print('Len @ {0} is zero! p_1={1} | p_2={2}'.format(v,p,p0))
                                    else:
                                        v_axis = np.divide(v_axis, np.linalg.norm(v_axis)) # Unit vector

                                        for i in range(max(0, p[0]-max_radius), min(V_gauss.shape[0]-1, p[0]+max_radius)):
                                            for j in range(max(0, p[1]-max_radius), min(V_gauss.shape[1]-1, p[1]+max_radius)):
                                                for k in range(max(0, p[2]-max_radius), min(V_gauss.shape[2]-1, p[2]+max_radius)):
                                                    vox_ijk = [i,j,k]
                                                    v_d = np.subtract(vox_ijk, p)
                                                    
                                                    # Magnitude of axial component
                                                    v_ax = abs(np.dot(v_d, v_axis))   
                                                    
                                                    # Magnitude of radial component
                                                    v_rad = math.sqrt(v_ax**2 + np.linalg.norm(v_d)**2) 

                                                    # Compute Gaussian, set voxel value to maximal Gaussian encountered
                                                    gauss = stats.norm.pdf(v_ax, 0, sigma_axial) * \
                                                            stats.norm.pdf(v_rad, 0, sigma_radial)
                                                    V_gauss[i,j,k] = max(gauss, V_gauss[i,j,k])

                                # Apply a bit of smoothing
                                V_gauss = utils.smooth_volume(V_gauss, self.V_img, 5)

                                # Multiply with original (unthresholded) tract
                                V_tract = np.multiply(V_orig, V_gauss)
                                V_tract = np.multiply(V_tract, np.greater(V_tract, tract_thresh))
                                V_tract = np.divide(V_tract, V_tract.max())
                                V_dist = np.round(np.multiply(V_dist, np.greater(V_tract, 0)))
                                dist_max = int(np.max(V_dist))

                                # Write results to NIFTI image files
                                img = nib.Nifti1Image(V_gauss, self.V_img.affine, self.V_img.header)
                                nib.save(img, '{0}/gauss_max_{1}_{2}.nii.gz'.format(my_target_gauss, tract_name, suffix))

                                img = nib.Nifti1Image(V_tract, self.V_img.affine, self.V_img.header)
                                nib.save(img, '{0}/tract_final_{1}_{2}.nii.gz'.format(my_target_final, tract_name, suffix))

                                # Normalize tract at each distance
                                for d in range(1, dist_max):
                                    idx = np.nonzero(np.equal(V_dist,d))
                                    if np.any(idx):
                                        V = V_tract[idx]
                                        V = np.divide(V, V.max())
                                        V_tract[idx] = V

                                img = nib.Nifti1Image(V_tract, self.V_img.affine, self.V_img.header)
                                nib.save(img, '{0}/tract_final_norm_{1}_{2}.nii.gz'.format(my_target_final, tract_name, suffix))

                                if verbose:
                                    print('  Computed distances, polylines, and Gaussians for {0}/{1} (direction {2})' \
                                          .format(roi_a, roi_b, suffix))
        if verbose:
            print('\n== Done generating polylines and Gaussians ==\n')
            
        return True
        
        
    # Compute the final bidirectional tract estimates from directional tracts 
    # generated by estimate_unidirectional_tracts
    #
    # verbose:        Whether to print messages to the console
    # clobber:        Whether to overwrite existing output
    # 
    def estimate_bidirectional_tracts( self, verbose=False, clobber=False ):
        
        # TODO: check flag files for each step to ensure prequisites are met
        if not self.is_init:
            print('DwiTracts object not initialized!')
            return False
        
        if verbose:
            print('\n== Computing bidirectional tracts...')

        params_gen = self.params['general']
        params_gauss = self.params['gaussians']
        threshold = params_gauss['threshold']

        V_rois = {}
        for roi in self.rois:
            roi_file = '{0}/{1}.{2}'.format(self.rois_dir, roi, self.roi_suffix)
            img = nib.load(roi_file)
            V_rois[roi] = img.get_fdata()

        N_roi = len(self.rois)
        tract_list = []
        rois_a = []
        rois_b = []

        for roi_a in self.rois:
            for roi_b in [x for x in self.rois if x != roi_a]:
                tract_name = '{0}_{1}'.format(roi_a,roi_b)
                tract_list.append(tract_name)
                rois_a.append(roi_a)
                rois_b.append(roi_b)

        for a in tqdm_notebook(range(0, N_roi-1), desc="Progress"):
            roi_a = self.rois[a]
            for b in range(a+1, N_roi):
                roi_b = self.rois[b]
                ab = '{0}_{1}'.format(roi_a,roi_b)
                ba = '{0}_{1}'.format(roi_b,roi_a)

                tract_file_ab = '{0}/tract_final_{1}_ab.nii.gz'.format(self.final_dir, ab)
                tract_file_ba = '{0}/tract_final_{1}_ba.nii.gz'.format(self.final_dir, ba)

                tract_failed_ab = not os.path.isfile(tract_file_ab)
                tract_failed_ba = not os.path.isfile(tract_file_ba)

                if not (tract_failed_ab and tract_failed_ba):

                    # Unnormalized version
                    if tract_failed_ab:
                        V = nib.load(tract_file_ba).get_fdata()
                    elif tract_failed_ba:
                        V = nib.load(tract_file_ab).get_fdata()
                    else:
                        V_ab = nib.load(tract_file_ab).get_fdata()
                        V_ba = nib.load(tract_file_ba).get_fdata()
                        V = np.max( np.array([ V_ab, V_ba ]), axis=0 )

                    V_blobs = utils.label_blobs(V, threshold)
                    if len(np.unique(V_blobs)) > 2:
                        V_blobs = utils.retain_adjacent_blobs(V_blobs, [V_rois[roi_a], V_rois[roi_b]])
                        if len(np.unique(V_blobs)) > 2:
                            print('  * Tract has multiple tract segments (unfixed): {0}|{1}'.format(roi_a, roi_b))
                        else:
                            print('  * Tract had multiple tract segments (1 retained): {0}|{1}'.format(roi_a, roi_b))
                        V = np.multiply(V, V_blobs>0)

                    img = nib.Nifti1Image(V, self.V_img.affine, self.V_img.header)
                    nib.save(img, '{0}/tract_final_bidir_{1}.nii.gz'.format(self.final_dir, ab))
                    
                    # Create bidirectional distance volume
                    V_dist = utils.get_tract_dist(V, V_rois[roi_a])
                    img = nib.Nifti1Image(V_dist, self.V_img.affine, self.V_img.header)
                    nib.save(img, '{0}/dist_bidir_{1}.nii.gz'.format(self.dist_dir, ab))

                    # Normalized version
                    tract_file_ab = '{0}/tract_final_norm_{1}_ab.nii.gz'.format(self.final_dir, ab)
                    tract_file_ba = '{0}/tract_final_norm_{1}_ba.nii.gz'.format(self.final_dir, ba)
                    tract_failed_ab = not os.path.isfile(tract_file_ab)
                    tract_failed_ba = not os.path.isfile(tract_file_ba)

                    if tract_failed_ab:
                        V = nib.load(tract_file_ba).get_fdata()
                    elif tract_failed_ba:
                        V = nib.load(tract_file_ab).get_fdata()
                    else:
                        V_ab = nib.load(tract_file_ab).get_fdata()
                        V_ba = nib.load(tract_file_ba).get_fdata()
                        V = V_ab + V_ba  #np.max( np.array([ V_ab, V_ba ]), axis=0 )
                        V[np.logical_and(V_ab,V_ba)] = V[np.logical_and(V_ab,V_ba)] / 2

                    # Check for blobs that don't connect ROIs; remove these if found
                    V = np.multiply(V, V_blobs>0)

                    img = nib.Nifti1Image(V, self.V_img.affine, self.V_img.header)
                    nib.save(img, '{0}/tract_final_norm_bidir_{1}.nii.gz'.format(self.final_dir, ab))

                    if verbose:
                        print('  Wrote average bidirectional tract for {0}/{1}'.format(roi_a, roi_b))

        if verbose:
            print('\n== Done generating bidirectional tracts ==\n')
            
        return True

    
       
    # Generates a Pajek graph with success(1) and failure(0) as edge weights
    #
    # verbose:        Whether to print messages to the console
    # clobber:        Whether to overwrite existing output
    #  
    def tracts_to_pajek( self, verbose=False, clobber=False):
        
        # TODO: check flag files for each step to ensure prequisites are met
        if not self.is_init:
            print('DwiTracts object not initialized!')
            return False
        
        N_roi = len(self.rois)
        A = np.zeros((N_roi,N_roi))
        
        # Get ROI center points
        roi_centers = utils.compute_roi_centers( self.rois_dir, self.rois, output_file=None, \
                                                 extension=self.roi_suffix, verbose=False )
        
        # Check final tracts for success/failure
        for a in range(0, N_roi-1):
            roi_a = self.rois[a]
            for b in range(a+1, N_roi):
                roi_b = self.rois[b]
                tract_name = '{0}_{1}'.format(roi_a, roi_b)
                test = '{0}/tract_final_bidir_{1}.nii.gz'.format(self.final_dir, tract_name)
                if os.path.isfile(test):
                    A[a,b] = 1
                    A[b,a] = 1
#                 else:
#                     A[a,b] = -1
#                     A[b,a] = -1
    
        output_file = '{0}/pass_fail.net'.format(self.tracts_dir)
        utils.write_matrix_to_pajek( A, output_file, directed=False, thres_low=-1, labels=self.rois, coords=roi_centers )
        
        if verbose:
            print('Wrote tracts to Pajek graph: {0}'.format(output_file))
            
        return True
    
    
    # Compute average streamline orientations for each tract
    #
    # For each tract that was successfully identified, use the
    # average voxel-wise streamline orientation information generated by ProbtrackX 
    # to compute the average orientation across subjects, for each voxel in the tract.
    #
    # verbose:        Whether to print messages to the console
    # clobber:        Whether to overwrite existing output
    #         
    def compute_average_orientations( self, verbose=False, clobber=False ):
        
        # TODO: check flag files for each step to ensure prequisites are met
        if not self.is_init:
            print('DwiTracts object not initialized!')
            return False
        
        print('\n== Computing average streamline orientations ==\n')

        params_gen = self.params['general']
        params_avdir = self.params['average_directions']
        tract_thresh = params_avdir['threshold']

        tract_list = []
        rois_a = []
        rois_b = []

        for a in range(0, len(self.rois)-1):
            roi_a = self.rois[a]
            for b in range(a+1, len(self.rois)):
                roi_b = self.rois[b]
                tract_name = '{0}_{1}'.format(roi_a,roi_b)
                tract_list.append(tract_name)
                rois_a.append(roi_a)
                rois_b.append(roi_b)

        for c in tqdm_notebook(range(0, len(rois_a)), desc='Total Progress'):
            roi_a = rois_a[c]
            roi_b = rois_b[c]
            tract_name = '{0}_{1}'.format(roi_a,roi_b)
            output_file = '{0}/avrdir_{1}.nii.gz'.format(self.avdir_dir, tract_name)

            if not clobber and os.path.isfile(output_file):
                if verbose:
                    print('  Average orientations already computed for {0}. Skipping.'.format(tract_name))
            else:
                tract_file = '{0}/tract_final_bidir_{1}.nii.gz'.format(self.final_dir, tract_name)

                if not os.path.isfile(tract_file):
                    if verbose:
                        print('  No tract found for {0}. Skipping.'.format(tract_name))
                else:
                    img = nib.load(tract_file)
                    V_tract = img.get_fdata()

                    s = (3,) + V_tract.shape
                    ss = V_tract.shape + (3,)
                    V_avdir = np.zeros(ss)
                    V_denom = np.zeros(ss)

                    V_img3 = None

                    # For each subject, read local direction vectors
                    for subject in tqdm_notebook(self.subjects, desc='{0}'.format(tract_name)):
                        prefix_sub = '{0}{1}'.format(params_gen['prefix'], subject)

                        # AB orientations
                        subject_dir = os.path.join(self.project_dir, params_gen['deriv_dir'], prefix_sub, \
                                                   params_gen['sub_dirs'], params_gen['dwi_dir'], \
                                                   'probtrackX', params_gen['network_name'], roi_a)
                        avdir_file = '{0}/target_localdir_{1}.nii.gz'.format(subject_dir, roi_b)

                        V_img3 = nib.load(avdir_file)
                        V_dir = V_img3.get_fdata()
                        V_dir[np.less(V_tract, tract_thresh)] = 0

                        # Ensure the maximal vector component is positive (we are concerned with
                        # orientation rather than directions); flip those that aren't
                        idx = np.nonzero(V_dir)
                        V_denom[idx] = np.add(V_denom[idx],1)
                        V_dir = utils.make_vectors_positive( V_dir )

                        V_avdir = np.add(V_avdir, V_dir)

                        # BA orientations
                        subject_dir = os.path.join(self.project_dir, params_gen['deriv_dir'], prefix_sub, \
                                                   params_gen['sub_dirs'], params_gen['dwi_dir'], \
                                                   'probtrackX', params_gen['network_name'], roi_b)
                        avdir_file = '{0}/target_localdir_{1}.nii.gz'.format(subject_dir, roi_a)

                        V_img3 = nib.load(avdir_file)
                        V_dir = V_img3.get_fdata()
                        V_dir[np.less(V_tract, tract_thresh)] = 0

                        # Ensure the maximal vector component is positive (we are concerned with
                        # orientation rather than directions); flip those that aren't
                        idx = np.nonzero(V_dir)
                        V_denom[idx] = np.add(V_denom[idx],1)
                        V_dir = utils.make_vectors_positive( V_dir )
                        V_avdir = np.add(V_avdir, V_dir)

                    # Divide to get average, and mask to exclude to this tract
                    idx = np.equal(V_denom,0)
                    V_denom[idx] = 1
                    V_avdir = np.divide(V_avdir, V_denom)

                    nidx = np.transpose(np.nonzero(V_avdir))

                    # Write average directions to file
                    img = nib.Nifti1Image(V_avdir, V_img3.affine, V_img3.header)
                    nib.save(img, output_file)

                    if verbose:
                        print('  Computed average orientations for {0}'.format(tract_name))
        if verbose:
            print('\n== Done computing average streamline orientations ==\n')
            
        return True
     
        
    
    # Compute tract specific anisoptropy (TSA) for all subjects
    #
    # verbose:        Whether to print messages to the console
    # clobber:        Whether to overwrite existing output
    # debug:          Whether to use debug mode (print lots to console)
    #         
    def compute_tsa( self, verbose=False, clobber=False, debug=False ):
        
        # TODO: check flag files for each step to ensure prequisites are met
        if not self.is_init:
            print('DwiTracts object not initialized!')
            return False
        
        failures = []
        
        for subject in tqdm_notebook( self.subjects, desc='Progress' ):           
            failure_count = process_tsa_subject( subject, self, verbose=verbose, debug=debug )
            failures.append(failure_count)
            
        df = pd.DataFrame({'Subject': self.subjects, 'Failures': failures})
        df.to_csv( '{0}/tsa_failures.csv'.format(self.tracts_dir) )
        
        return True
        

# Compute DWI regressions for a single subject
#
# Uses average streamline orientations to obtain tract-specific anisotropy (TSA) measures 
# for each voxel. Voxel-wise regressions are performed to determine how strongly
# the diffusion profile in each voxel loads (in terms of beta cofficients) onto the average 
# orientation for a given tract.
#
# subject:        Subject to process
# my_dwi:         Instance of DwiTracts, which must already been initialized and have 
#                  generated prerequisite outputs (bidirectional tract estimates,
#                  average streamline orientations)
# verbose:        Whether to print messages to the console
# debug:          Whether to use debug mode (print lots to console) - useful for 
#                  debugging FSL errors
#
def process_tsa_subject( subject, my_dwi, verbose=False, debug=False ):

    params_gen = my_dwi.params['general']
    tmp_dir = params_gen['temp_dir']
    if not os.path.isdir(tmp_dir):
        os.makedirs(tmp_dir)
        
    source_dir = params_gen['source_dir']
   
    with open(params_gen['preproc_config'], 'r') as myfile:
        json_string = myfile.read()
                     
    params_preproc = json.loads(json_string)
    params_bpx = params_preproc['bedpostx']
    params_ptx = params_preproc['probtrackx']
    params_regress = my_dwi.params['dwi_regressions']

    fsl_root = params_regress['fsl_root']
    fsl_maths = '{0}/bin/fslmaths'.format(fsl_root)
    fsl_vecreg = '{0}/bin/vecreg'.format(fsl_root)
    fsl_applywarp = '{0}/bin/applywarp'.format(fsl_root)
    fsl_eddy = '{0}/bin/eddy_correct'.format(fsl_root)

    fa_img = params_regress['fa_img']
    standard_img = 'utils/{0}'.format(params_regress['standard_img'])

    # Get ROIs and targets
    tract_names = []
    tract_files = []
    rois_a = []
    rois_b = []

    # Read ROI list
    rois = []
    target_rois = {}
    
    # If this is a JSON file, read as networks and targets
    if params_ptx['roi_list'].endswith('.json'):
        with open(params_ptx['roi_list'], 'r') as myfile:
            json_string=myfile.read()

        netconfig = json.loads(json_string)
        networks = netconfig['networks']
        targets = netconfig['targets']

        for net in networks:
            others = []
            for net2 in targets[net]:
                others = others + networks[net2]
            for roi in networks[net]:
                rois.append(roi)
                target_rois[roi] = others
    else:
        with open(params_gen['rois_list'],'r') as roi_file:
            reader = csv.reader(roi_file)
            for row in reader:
                rois.append(row[0])
        
        # All ROIs are targets
        for a in range(0, len(rois)-1):
            roi_a = rois[a]
            targets = []
            for b in range(a+1, len(rois)):
                roi_b = rois[b]
                targets.append(roi_b)
            target_rois[roi_a] = targets
    
    for a in range(0, len(rois)-1):
        roi_a = rois[a]
        for roi_b in target_rois[roi_a]:
            tract_name = '{0}_{1}'.format(roi_a,roi_b)
            tract_names.append(tract_name)
            tract_file = '{0}/tract_final_bidir_{1}.nii.gz'.format(my_dwi.final_dir, tract_name)
            tract_files.append(tract_file)
            rois_a.append(roi_a)
            rois_b.append(roi_b)

    prefix_sub = '{0}{1}'.format(params_gen['prefix'], subject)
    subject_dir = os.path.join(my_dwi.project_dir, params_gen['deriv_dir'], prefix_sub, params_gen['sub_dirs'])

    dwi_dir = os.path.join(subject_dir, params_gen['dwi_dir'])
    ptx_dir = os.path.join(dwi_dir,'probtrackX', params_gen['network_name'])

    bet_file = '{0}/{1}_{2}_dwi_bet.nii.gz'.format(dwi_dir, prefix_sub, params_gen['sub_dirs'])
    warp_file = '{0}/reg3G/Mean3G_warp2FA.nii.gz'.format(dwi_dir)
    invwarp_file = '{0}/reg3G/FA_warp2Mean3G.nii.gz'.format(dwi_dir)
    bvec_file = '{0}/bedpostX/bvecs'.format(dwi_dir)
    bval_file = '{0}/bedpostX/bvals'.format(dwi_dir)
    fa_file = '{0}/dti_FA.nii.gz'.format(dwi_dir)
    diff_file = '{0}/bedpostX/mean_dsamples.nii.gz'.format(dwi_dir)
    
    # Ensure all the required files exist
    if not os.path.isfile(warp_file) or \
        not os.path.isfile(invwarp_file) or \
        not os.path.isfile(bvec_file) or \
        not os.path.isfile(bval_file) or \
        not os.path.isfile(fa_file) or \
        not os.path.isfile(diff_file):
            
        if verbose:
            print('  * Warning: Subject {0} is missing DWI files. Skipping.'.format(subject))
        return -1
    
    # Create output directory
    subj_output_dir = '{0}/dwi/{1}/{2}'.format(subject_dir, params_regress['regress_dir'], params_gen['network_name'])
    if os.path.isdir(subj_output_dir):
        if not params_regress['force_regress']:
            img_files = glob.glob('*.nii.gz')
            if len(img_files) >= len(tract_name):
                if verbose:
                    print('  * Warning: Output already exists for subject {0}. Skipping.'.format(subject))
                return 0
        else:
            # Remove directory and its contents
            shutil.rmtree(subj_output_dir)

    if not os.path.isdir(subj_output_dir):
        os.makedirs(subj_output_dir)
        
    # Load bvals and bvecs
    bvals = utils.read_bvals(bval_file)
    if bvals is None:
        print('  * Warning: Could not find bvals for subject {0}. Skipping.'.format(subject))
        return -1
    
    bvecs = utils.read_bvecs(bvec_file)
    if bvals is None:
        print('  * Warning: Could not find bvecs for subject {0}. Skipping.'.format(subject))
        return -1

    b0 = np.less_equal(bvals, 50)
    bp = np.greater(bvals, 50)
    bvecs = bvecs[bp,:]
    
    N_img = bp[bp==True].size
    
    # Load diffusion-weighted image
    V_bet = nib.load(bet_file).get_data()
    V_b0 = np.mean(V_bet[:,:,:,b0],3)
    V_bet = V_bet[:,:,:,bp]
    
    # Load mean diffusivity image
    V_img = nib.load(diff_file)
    V_diff = V_img.get_data()
    
    failure_count = 0
    
    # Compute regressions for each tract
    for i in range(0, len(tract_names)):
        tract_name = tract_names[i]
        tract_file_in = '{0}/tract_final_bidir_{1}.nii.gz'.format(my_dwi.final_dir, tract_name)
        tract_file = '{0}/{1}.{2}.tract.nii.gz'.format(tmp_dir, subject, tract_name)
        avrdir_file_in = '{0}/avrdir_{1}.nii.gz'.format(my_dwi.avdir_dir, tract_name)
        avrdir_file = '{0}/{1}.{2}.avrdir.nii.gz'.format(tmp_dir, subject, tract_name)
        
        success = True
        
        if not os.path.isfile(tract_file_in):
            if verbose:
                print('  - Tract {0} not found. Skipping.'.format(tract_name))
        else:
            shell_file = '{0}/{1}/{2}.warp.sh'.format(tmp_dir, subject, tract_name)
            cmd = '{0} -i {1} -o {2} -r {3} -w {4}' \
                  .format(fsl_vecreg, avrdir_file_in, avrdir_file, bet_file, warp_file)
              
            err = utils.run_fsl(cmd)
            if not params_regress['ignore_errors'] and err:
                if verbose:
                    print('  * Error running fsl_vecreg for subject {0} on tract {1}. Skipping.' \
                          .format(subject, tract_name))
                failure_count += 1
                success = False
                if params_regress['fail_on_error']:
                    if verbose:
                        print(err)
                    return failure_count
            elif debug and err:
                print('  * Error running fsl_vecreg for subject {0} on tract {1}.' \
                      .format(subject, tract_name))
                print(err)
                
            if success:             
                cmd = '{0} -i {1} -o {2} -r {3} -w {4}' \
                      .format(fsl_applywarp, tract_file_in, tract_file, bet_file, warp_file)

                err = utils.run_fsl(cmd)
                if not params_regress['ignore_errors'] and err:
                    if verbose:
                        print('  * Error running fsl_applywarp for subject {0} on tract {1}. Skipping.' \
                              .format(subject, tract_name))
                    failure_count += 1
                    success = False
                    if params_regress['fail_on_error']:
                        print(err)
                        return failure_count
                elif debug and err:
                    print('  * Error running fsl_applywarp for subject {0} on tract {1}.' \
                              .format(subject, tract_name))
                    print(err)
    
            if success:
                # Load warped average vectors
                V_tract = nib.load(tract_file).get_data()
                V_avrdir = nib.load(avrdir_file).get_data()
                
                V_beta = np.zeros(V_tract.shape, float)
                V_tvals = np.zeros(V_tract.shape, float)
                V_pvals = np.zeros(V_tract.shape, float)
                
                idx = np.greater(V_tract, params_regress['threshold'])
                fidx = np.transpose(np.nonzero(idx))
                
                if np.any(idx):
                    # Regression per voxel
                    N_idx = fidx.shape[0]
                    v_d = np.zeros([N_idx, 3], float)
                    for i in range(0, N_idx):
                        x = fidx[i,0]
                        y = fidx[i,1]
                        z = fidx[i,2]
                        v_d[i,:] = V_avrdir[x,y,z,:]

                    fidx = fidx[np.any(v_d,axis=1)]
                    N_idx = fidx.shape[0]
                    idx = np.zeros(idx.shape, dtype=bool)
                    for i in range(0, N_idx):
                        x = fidx[i,0]
                        y = fidx[i,1]
                        z = fidx[i,2]
                        idx[x,y,z] = True

                    idx3 = np.repeat(idx[:,:,:,None], N_img, axis=3)

                    x_d     = bvecs
                    dwi_d   = np.reshape(V_bet[idx3],(N_idx, N_img))
                    b0_d    = V_b0[idx]
                    b_d     = bvals[bp]
                    diff_d  = V_diff[idx]

                    # Results
                    betas   = np.zeros(N_idx, float)
                    tvals   = np.zeros(N_idx, float)
                    pvals   = np.zeros(N_idx, float)

                    for vv in range(0, N_idx):
                        b0   = b0_d[vv]
                        yy   = dwi_d[vv,:] / b0
                        bb   = b_d
                        dd   = diff_d[vv]
                        xTv  = np.matmul(v_d[vv,:],x_d.T)
                        xx   = np.exp(-(((bb*dd) * xTv)**2))
                        xx   = sm.add_constant(xx, has_constant='add')

                        results = sm.OLS(yy, xx).fit()
                        betas[vv] = results.params[1]
                        tvals[vv] = results.tvalues[1]
                        pvals[vv] = results.pvalues[1]
                        
                    # Clamp betas
                    betas[betas < params_regress['beta_min']] = params_regress['beta_min']
                    betas[betas > params_regress['beta_max']] = params_regress['beta_max']
                    
                    V_beta[idx] = betas
                    
                    stats = {'betas': V_beta}
                    
                    if params_regress['write_stats']:
                        V_tvals[idx] = tvals
                        V_pvals[idx] = pvals
                        stats['tvals'] = V_tvals
                        stats['pvals'] = V_pvals
                    
                    # Include normalized betas?
                    if params_regress['beta_norm']:
                        betas = (betas - params_regress['beta_min']) / \
                                (params_regress['beta_max'] - params_regress['beta_min'])
                        V_beta_nrm = np.zeros(V_tract.shape, float)
                        V_beta_nrm[idx] = betas
                        stats['beta_norm'] = V_beta_nrm
                    
                # Write results volumes
                
                dwi_files = {}
                mni_files = {}
                
                for stat in stats:
                    dwi_files[stat] = '{0}/{1}_dwi_{2}.nii.gz'.format(subj_output_dir, stat, tract_name)
                    mni_files[stat] = '{0}/{1}_mni_{2}.nii.gz'.format(subj_output_dir, stat, tract_name)
                    img = nib.Nifti1Image(stats[stat], V_img.affine, V_img.header)
                    nib.save(img, dwi_files[stat])

                for stat in stats:
                
                    # Warp back to standard space
                    smooth_file = '{0}/{1}_mni_sm_{2}um_{3}.nii.gz' \
                                  .format(subj_output_dir, stat, int(1000.0*params_regress['beta_sm_fwhm']), tract_name)
                    cmd = '{0} -i {1} -o {2} -r {3} -w {4}' \
                          .format(fsl_applywarp, dwi_files[stat], mni_files[stat], standard_img, invwarp_file)
                    err = utils.run_fsl(cmd)
                    if not params_regress['ignore_errors'] and err:
                        if verbose:
                            print('  * Error running fsl_applywarp on {0} for subject {1} on tract {2}. Skipping.' \
                                .format(stat, subject, tract_name))
                        failure_count += 1
                        if debug:
                            print(cmd)
                            print(err)
                        success = False
                    elif debug and err:
                        print('  * Error running fsl_applywarp on {0} for subject {1} on tract {2}.' \
                                .format(stat, subject, tract_name))
                        print(err)

                    if success:
                        cmd = '{0} {1} -kernel gauss {2} -fmean {3}' \
                              .format(fsl_maths, mni_files[stat], params_regress['beta_sm_fwhm']/2.1231, smooth_file)
                        err = utils.run_fsl(cmd)
                        if not params_regress['ignore_errors'] and err:
                            if verbose:
                                print('  * Error running smoothing {0} for subject {1} on tract {2}. Skipping.' \
                                    .format(stat, subject, tract_name))
                            failure_count += 1
                            if debug:
                                print(cmd)
                                print(err)
                            success = False
                        elif debug and err:
                            print('  * Error running smoothing {0} for subject {1} on tract {2}.' \
                                  .format(stat, subject, tract_name))
                            print(err)

                    if success:
                        cmd = '{0} {1} -mas {2} {3}' \
                              .format(fsl_maths, mni_files[stat], tract_file_in, mni_files[stat])
                        err = utils.run_fsl(cmd)
                        if not params_regress['ignore_errors'] and err:
                            if verbose:
                                print('  * Error running masking results for subject {0} on tract {1}. Skipping.' \
                                    .format(subject, tract_name))
                            failure_count += 1
                            if debug:
                                print(cmd)
                                print(err)
                            success = False
                        elif debug and err:
                            print('  * Error running masking results {0} for subject {1} on tract {2}.' \
                                  .format(stat, subject, tract_name))
                            print(err)

                    if success:
                        cmd = '{0} {1} -mas {2} {3}' \
                              .format(fsl_maths, smooth_file, tract_file_in, smooth_file)
                        err = utils.run_fsl(cmd)
                        if not params_regress['ignore_errors'] and err:
                            if verbose:
                                print('  * Error running masking results for subject {0} on tract {1}. Skipping.' \
                                    .format(subject, tract_name))
                            failure_count += 1
                            if debug:
                                print(cmd)
                                print(err)
                            success = False
                        elif debug and err:
                            print(err)

                        # Check if output is there
                        if success:
                            if not os.path.isfile(mni_files[stat]) or not os.path.isfile(smooth_file):
                                if verbose:
                                    print('  * Error running masking results for subject {0} on tract {1}. Skipping.' \
                                        .format(subject, tract_name))
                                failure_count += 1
                                success = False
                
                    if not params_regress['retain_dwi']:
                        os.remove(dwi_files[stat])
                
                if not success and params_regress['fail_on_error']:
                    return failure_count
                    
                # Remove temporary files
                os.remove(tract_file)
                os.remove(avrdir_file)
                           
                if verbose:
                    if success:
                        print('  Finished tract {0} for subject {1}'.format(tract_name, subject))
                    else:
                        print('  * Warning: Finished tract {0} for subject {1} with failures'.format(tract_name, subject))
    
    return failure_count
