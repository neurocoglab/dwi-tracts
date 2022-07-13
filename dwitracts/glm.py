import os
import csv
import json
import pandas as pd
import copy
import numpy as np
import csv
import nibabel as nib
import statsmodels.api as sm
import scipy.stats as stats
import statsmodels.stats.multitest as smm
import shutil
from tqdm import tnrange, tqdm_notebook, tqdm
from . import utils
import rft1d
import re
import math
from glob import glob
from dwitracts.main import DwiTracts
import seaborn as sns
import matplotlib
import matplotlib.pylab as plt
import math

class DwiTractsGlm:
    
    def __init__(self, params):
        self.params = params
        self.is_init = False
        
    def _json_object_hook(d): return namedtuple('X', d.keys())(*d.values())
    def json2obj(data): return json.loads(data, object_hook=_json_object_hook)
        
    # Initialise this object with its current parameters
    def initialize(self):
        params_gen = self.params['general']
        params_glm = self.params['glm']
        params_tracts = self.params['tracts']
        params_preproc = self.params['preproc']
        params_avdir = params_tracts['average_directions']
        
        my_dwi = DwiTracts( self.params['tracts'] )
        assert my_dwi.initialize()

        subjects = my_dwi.subjects
        subjects = sorted(subjects)
        
        # Set up the directories
        self.source_dir = my_dwi.source_dir
        self.project_dir = my_dwi.project_dir
        self.rois_dir = my_dwi.rois_dir
        self.tracts_dir = my_dwi.tracts_dir
        
        # Read ROIs
        self.rois = my_dwi.rois

        # Determine proper suffix
        self.roi_suffix = 'nii'
        roi = self.rois[0]
        roi_file = '{0}/{1}.{2}'.format(self.rois_dir, roi, self.roi_suffix)
        if not os.path.isfile(roi_file):
            self.roi_suffix = 'nii.gz'
            roi_file = '{0}/{1}.{2}'.format(self.rois_dir, roi, self.roi_suffix)
            if not os.path.isfile(roi_file):
                print('No ROI file found: {0}.'.format(roi_file))
                return False

        V_img = nib.load(roi_file)
        V_img.header.set_data_dtype(np.float32)

        params_regress = params_tracts['dwi_regressions']
        self.final_dir = os.path.join(self.tracts_dir, 'final')
        
        # Set up variables
        # Requires covariate data to have been saved as a Pandas table, in either
        # HDF5 or CSV formats

        params_ptx = params_preproc['probtrackx']
        self.subject_data = pd.read_csv(params_gen['covariates_file'])
        self.subject_data = self.subject_data[self.subject_data['Subject'].isin(subjects)]
        self.subject_data = self.subject_data.sort_values('Subject') #, ignore_index=True)

        if len(subjects) == 0:
            print('No subjects found in subject data file {0}.'.format(params_gen['covariates_file']))
            return False

        Xs = {}
#         subjects = self.subject_data['Subject'].values
        N_sub = len(subjects)
        self.subjects = subjects
        
        # Compile covariate matrix
        for glm in params_glm:

            factors = params_glm[glm]['factors']
            categorical = params_glm[glm]['categorical']
            add_intercept = params_glm[glm]['intercept']
            Xi = np.empty((N_sub, len(factors)+1))

            # Deal with categorical variables
            for factor in factors:
                if categorical[factor]:
                    x = self.subject_data[factor].values
                    vals = np.unique(x) # Return value is sorted; lower value is assigned -1, higher +1
                    if len(vals) != 2:
                        print('Factor {0} has {1} levels. Can only have two.')
                        assert False
                    # Encode as -1, 1
                    xx = x.copy()
                    for j, k in zip([-1, 1], vals):
                        xx[x == k] = j
                    self.subject_data[factor] = xx

            # Add intercept
            i = 0
            if add_intercept:
                Xi[:,0] = np.ones(N_sub) # This is the intercept term
                i = 1

            # Standardize if necessary
            if params_gen['standardized_beta']:
                for factor in factors:
                    if not factor.find('*') < 0:
                        x = self.subject_data[factor].values
                        x = stats.zscore(x)
                        self.subject_data[factor] = x

            for factor in factors:
                if factor.find('*') > -1:
                    # Create interaction term
                    parts = factor.split('*')
                    x = np.ones((N_sub,))
                    for part in parts:
                        x = x * self.subject_data[part].values
                else:
                    x = self.subject_data[factor].values

                Xi[:,i] = x
                i += 1

            Xs[glm] = Xi

            if add_intercept:
                factors.insert(0, 'Intercept')
        
        self.Xs = Xs
        
        # Get tract list
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
            with open(params_ptx['roi_list'],'r') as roi_file:
                reader = csv.reader(roi_file)
                for row in reader:
                    rois.append(row[0])

            # All ROIs are targets
            for i in range(0, len(rois)):
                targets = []
                roi = rois[i]
                for j in range(i+1, len(rois)):
                    roi2 = rois[j]
                    targets.append(roi2)
                target_rois[roi] = targets

        self.rois = rois
        self.target_rois = {}
        self.tract_names = []
        for roi_a in rois:
            targets_a = []
            for roi_b in target_rois[roi_a]:
                tract_name = '{0}_{1}'.format(roi_a,roi_b)
                if tract_name in my_dwi.tracts_final_bidir:
                    self.tract_names.append(tract_name)
                    targets_a.append(roi_b)
            self.target_rois[roi_a] = targets_a
                  
        tract_thresh = params_gen['tract_threshold']
        if tract_thresh < params_avdir['threshold']:
            print('GLM threshold ({0:1.3f}) cannot be less than average direction threshold ({1:1.3f}).' \
                  .format(tract_thresh, params_avdir['threshold']))
            return False
              
        self.tract_threshold = tract_thresh
        
        self.is_init = True
        return True         
        
        
    # Fit all GLMs specified in this object's parameters and write results to file
    #
    # clobber:       Whether to overwrite existing results
    # verbose:       Whether to print progress to screen
    #
    def fit_glms( self, clobber=False, verbose=False ):
        
        if not self.is_init:
            print('Not initialized! You must call "initialize()" before "fit_glms".')
            return False
        
        if verbose:
            print('Computing GLMs for {0} tracts.'.format(len(self.tract_names)))

        params_glm = self.params['glm']
        params_gen = self.params['general']
        params_tracts = self.params['tracts']
        params_regress = params_tracts['dwi_regressions']   
        params_avdir = params_tracts['average_directions']
        
        use_norm = params_avdir['use_normalized']
        
        tract_thresh = self.tract_threshold
        
        N_sub = len(self.subjects)
        
        std_str = ''
        if params_gen['standardized_beta']:
            std_str = '_std'
            
        for glm in params_glm:
            glm_dir = '{0}/{1}/{2}'.format(self.tracts_dir, params_gen['glm_output_dir'], glm)
            if os.path.exists(glm_dir):
                if not clobber:
                    print('Output already exists! Set clobber=True to overwrite.')
                    return False
                shutil.rmtree(glm_dir)

            coef_dir = '{0}/coef'.format(glm_dir);
            tval_dir = '{0}/tval'.format(glm_dir);
            pval_dir = '{0}/pval'.format(glm_dir);
            resid_dir = '{0}/resid'.format(glm_dir);

            os.makedirs(coef_dir)
            os.makedirs(tval_dir)
            os.makedirs(pval_dir)
            os.makedirs(resid_dir)

            if verbose:
                print('Writing results to {0}.'.format(glm_dir))
        
        for tract_name in tqdm_notebook(self.tract_names, 'Progress'):
            
            if use_norm:
                tract_file = '{0}/tract_final_norm_bidir_{1}.nii.gz'.format(self.final_dir, tract_name)
            else:
                tract_file = '{0}/tract_final_bidir_{1}.nii.gz'.format(self.final_dir, tract_name)
            if not os.path.exists(tract_file):
                if verbose:
                    print('   No tract exists for {0}. Skipping.'.format(tract_name))
            else:
                if verbose:
                    print('   Processing {0}'.format(tract_name))

                # Compile 4D volume (N_tracts as 4th dim)
                V_img = nib.load(tract_file)
                V_tract = V_img.get_fdata()
                V_stats = np.zeros(V_tract.shape)

                # Threshold?
#                 thres = params_gen['tract_threshold']
                if tract_thresh > 0:
                    V_tract[V_tract<tract_thresh]=0

                # Tract mask as index vector
                idx = np.flatnonzero(V_tract)
                idx3 = np.nonzero(V_tract)
                idx4 = np.dstack([idx3]*N_sub)
                V_betas = np.zeros((N_sub, idx.size))
                resid_hdr = copy.copy(V_img.header)
                shape = resid_hdr.get_data_shape()
                resid_hdr.set_data_shape(shape + (N_sub,))

                i = 0;
                # Get betas for each subject
                for subject in tqdm_notebook(self.subjects, desc='Read betas: '):
                    prefix_sub = '{0}{1}'.format(params_tracts['general']['prefix'], subject)
                    subject_dir = os.path.join(self.project_dir, params_tracts['general']['deriv_dir'], prefix_sub, \
                                               params_tracts['general']['sub_dirs'])
                    subj_output_dir = '{0}/dwi/{1}/{2}'.format(subject_dir, params_regress['regress_dir'], \
                                                               params_tracts['general']['network_name'])
                    beta_file = '{0}/betas_mni_sm_{1}um_{2}.nii.gz' \
                                          .format(subj_output_dir, int(1000.0*params_regress['beta_sm_fwhm']), tract_name)

                    V_sub = nib.load(beta_file).get_fdata()
                    V_betas[i,:] = V_sub.ravel()[idx]
                    i += 1

                # Now evaluate GLMs for each voxel
                for glm in params_glm:
                    factors = params_glm[glm]['factors']

                    glm_dir = '{0}/{1}/{2}'.format(self.tracts_dir, params_gen['glm_output_dir'], glm)
                    coef_dir = '{0}/coef'.format(glm_dir);
                    tval_dir = '{0}/tval'.format(glm_dir);
                    pval_dir = '{0}/pval'.format(glm_dir);
                    resid_dir = '{0}/resid'.format(glm_dir);

                    X = self.Xs[glm]
                    V_coef = {}
                    V_tval = {}
                    V_pval = {}
                    V_pval_fdr = {}
                    V_sig_fdr = {}
                    V_resid = np.zeros((idx.size, N_sub))

                    for factor in factors:
                        V_coef[factor] = np.zeros(idx.size)
                        V_tval[factor] = np.zeros(idx.size)
                        V_pval[factor] = np.ones(idx.size)
                        V_pval_fdr[factor] = np.ones(idx.size)
                        V_sig_fdr[factor] = np.zeros(idx.size)


                    for j in tqdm_notebook(range(0, idx.size), desc='{0}: '.format(glm) ):
                        y = V_betas[:,j]
                        # Remove outliers?
                        if params_glm[glm]['outlier_z'] > 0:
                            zthres = [y.mean() - params_glm[glm]['outlier_z'] * y.std(), \
                                      y.mean() + params_glm[glm]['outlier_z'] * y.std()]
                        else:
                            zthres = [float('-inf'), float('inf')]
                        idx_ok = np.flatnonzero(np.logical_and(y >= zthres[0], y <= zthres[1]))

                        results = sm.OLS(y[idx_ok], X[idx_ok]).fit()
                        V_resid[j,idx_ok] = results.resid

                        for factor, k in zip(factors, range(0,len(factors))):
                            V_coef[factor][j] = results.params[k]
                            V_tval[factor][j] = results.tvalues[k]
                            V_pval[factor][j] = results.pvalues[k]

                    # Apply FDR correction
                    for factor in factors:
                        try:
                            R = smm.multipletests(V_pval[factor], params_gen['fdr_alpha'], params_gen['fdr_method'])
                            V_pval_fdr[factor] = R[1]
                            V_sig_fdr[factor] = np.logical_not(R[0])
                        except ZeroDivisionError:
                            V_pval_fdr[factor] = V_pval[factor]
                            V_sig_fdr[factor] = np.logical_not(V_pval[factor]>0.05)
                            if verbose:
                                print('      Warning: Zero division in FDR correction (!?) for {0}.'.format(factor))


                    # Write residuals to file
                    output_file_resid = '{0}/{1}{2}.nii.gz'.format(resid_dir, tract_name, std_str)
                    V_resids = np.zeros((V_stats.shape[0], V_stats.shape[1], V_stats.shape[2], N_sub))
                    for s in range(0,N_sub):
                        V_resids[idx3[0],idx3[1],idx3[2],s] = V_resid[:,s]

                    img = nib.Nifti1Image(V_resids, V_img.affine, resid_hdr)
                    if verbose:
                        print('      Writing residuals {0}'.format(img.shape))
                    nib.save(img, output_file_resid)

                    # Write result to file
                    for factor in factors:
                        factor_str = factor.replace('*','X')
                        output_file_coef = '{0}/{1}_{2}{3}.nii.gz'.format(coef_dir, tract_name, factor_str, std_str)
                        output_file_tval = '{0}/{1}_{2}{3}.nii.gz'.format(tval_dir, tract_name, factor_str, std_str)
                        output_file_pval = '{0}/{1}_{2}{3}.nii.gz'.format(pval_dir, tract_name, factor_str, std_str)
                        output_file_pval_fdr = '{0}/fdr_{1}_{2}{3}.nii.gz'.format(pval_dir, tract_name, factor_str, std_str)
                        output_file_pval_sigfdr = '{0}/sigfdr_{1}_{2}{3}.nii.gz'.format(pval_dir, tract_name, factor_str, std_str)

                        V_stats.fill(0)
                        V_stats[idx3] = V_coef[factor]
                        img = nib.Nifti1Image(V_stats, V_img.affine, V_img.header)
                        nib.save(img, output_file_coef)
                        V_stats[idx3] = V_tval[factor]
                        img = nib.Nifti1Image(V_stats, V_img.affine, V_img.header)
                        nib.save(img, output_file_tval)
                        V_stats[idx3] = V_pval[factor]
                        img = nib.Nifti1Image(V_stats, V_img.affine, V_img.header)
                        nib.save(img, output_file_pval)
                        V_stats[idx3] = V_pval_fdr[factor]
                        img = nib.Nifti1Image(V_stats, V_img.affine, V_img.header)
                        nib.save(img, output_file_pval_fdr)
                        V_stats[idx3] = V_sig_fdr[factor]
                        img = nib.Nifti1Image(V_stats, V_img.affine, V_img.header)
                        nib.save(img, output_file_pval_sigfdr)

                        if verbose:
                            print('      Wrote {0}-{1}'.format(glm, factor))
                if verbose:
                    print('   Finished {0}'.format(tract_name))

        if verbose:
            print('Done.')
            
        return True

    # Extract/summarise statistics at each distance along tracts and save results
    #
    # tract_thres:   The threshold defining how much of the tract to use (default=0.1)
    # clobber:       Whether to overwrite existing results
    # verbose:       Whether to print progress to screen
    #
    def extract_distance_traces( self, clobber=False, verbose=True ):
                  
        params_glm = self.params['glm']
        params_gen = self.params['general']
        params_trace = self.params['traces']
        params_avdir = self.params['tracts']['average_directions']
        use_norm = params_avdir['use_normalized']
        
        if params_trace['tract_threshold'] < self.tract_threshold:
            print('Trace threshold ({0:1.3f}) cannot be less than GLM threshold ({1:1.3f}).' \
                  .format(params_trace['tract_threshold'], self.tract_threshold))
            return False
        tract_thresh = params_trace['tract_threshold']
        thresh_str = '{0:02d}'.format(round(tract_thresh*100))
        params_preproc = self.params['preproc']
        params_ptx = params_preproc['probtrackx']
        seed_dilate = self.params['tracts']['gaussians']['seed_dilate']
        
        dist_dir = os.path.join(self.tracts_dir, 'dist')
        final_dir = os.path.join(self.tracts_dir, 'final')
        metric = params_gen['summary_metric']
        
        for glm in params_glm:
            glm_dir = '{0}/{1}/{2}'.format(self.tracts_dir, params_gen['glm_output_dir'], glm)
            output_dir = '{0}/summary-{1}_thr{2}'.format(glm_dir, metric, thresh_str)
            if os.path.exists(output_dir):
                if not clobber:
                    print('Output directory exists! (use "clobber" to overwrite): {0}'.format(output_dir))
                    return False
                shutil.rmtree(output_dir)
            os.makedirs(output_dir)
        
        std_str = ''
        if params_gen['standardized_beta']:
            std_str = '_std'

        nanval = float(params_gen['nan_value'])
        N_sub = len(self.subjects)

        for roi_a in tqdm_notebook(self.rois, 'Progress'):
            for roi_b in self.target_rois[roi_a]:
                
                tract_name = '{0}_{1}'.format(roi_a,roi_b)
                if verbose:
                    print('Evaluating {0}'.format(tract_name))
                if use_norm:
                    tract_file = '{0}/tract_final_norm_bidir_{1}.nii.gz'.format(final_dir, tract_name)
                else:
                    tract_file = '{0}/tract_final_bidir_{1}.nii.gz'.format(final_dir, tract_name)
                if not os.path.exists(tract_file):
                    if verbose:
                        print(' Tract {0} not found.'.format(tract_name))
                    continue
                    
                V_tract = nib.load(tract_file).get_fdata()
                dist_file = '{0}/dist_bidir_{1}_{2}.nii.gz'.format(dist_dir, roi_a, roi_b)
                V_dist = nib.load(dist_file).get_fdata()
                V_dist[V_tract < tract_thresh] = 0

                img_shape = V_dist.shape
                V_dist = V_dist.flatten()
                V_tract = V_tract.flatten()
                dists = np.unique(V_dist)
                dists = dists[dists > 0] # Not interested in zeros
                Nd = dists.size

                if Nd < 3:
                    if verbose:
                        print('   No tract found for {0}. Skipping.'.format(tract_name))
                    continue

                # For each GLM:
                for glm in params_glm:
                    factors = params_glm[glm]['factors']
                    Nf = len(factors)
                    formats = ['%d']
                    for i in range(0,4):
                        for factor in factors:
                            formats.append('%1.7e')

                    glm_dir = '{0}/{1}/{2}'.format(self.tracts_dir, params_gen['glm_output_dir'], glm)
                    output_dir = '{0}/summary-{1}_thr{2}'.format(glm_dir, metric, thresh_str)

                    tval_dir = '{0}/tval'.format(glm_dir, thresh_str);
                    coef_dir = '{0}/coef'.format(glm_dir, thresh_str);
                    pval_dir = '{0}/pval'.format(glm_dir, thresh_str);
                    resid_dir = '{0}/resid'.format(glm_dir, thresh_str);
                    j = 1
                    M = np.zeros((Nd, 4*Nf+1))
                    M[:,0] = dists

                    hdr = 'Distance'

                    # Extract residuals
                    resid_file = '{0}/{1}{2}.nii.gz'.format(resid_dir, tract_name, std_str)
                    V_resids = nib.load(resid_file).get_fdata()

                    # For each factor:
                    for factor in factors:
                        factor_str = factor.replace('*','X')

                        # Read stat image
                        tval_file = '{0}/{1}_{2}{3}.nii.gz'.format(tval_dir, tract_name, factor_str, std_str)
                        coef_file = '{0}/{1}_{2}{3}.nii.gz'.format(coef_dir, tract_name, factor_str, std_str)
                        pval_file = '{0}/{1}_{2}{3}.nii.gz'.format(pval_dir, tract_name, factor_str, std_str)
                        fdrpval_file = '{0}/fdr_{1}_{2}{3}.nii.gz'.format(pval_dir, tract_name, factor_str, std_str)
                        V_tval = nib.load(tval_file).get_fdata().flatten()
                        V_tval_abs = np.abs(V_tval)
                        V_pval = nib.load(pval_file).get_fdata().flatten()
                        V_fdrpval = nib.load(fdrpval_file).get_fdata().flatten()
                        V_coef = nib.load(coef_file).get_fdata().flatten()

                        # Get residuals (can change for each factor if metric == 'max')
                        summary_resids = np.zeros((Nd, N_sub))
                        for d, i in zip(dists, range(0,dists.size)):
                            idx_d = np.flatnonzero(V_dist==d)
                            idx_flat = np.unravel_index(idx_d, img_shape)
                            resids = V_resids[idx_flat[0],idx_flat[1],idx_flat[2],:]
#                             print('Shape: {0}'.format(resids.shape))

                            summary = 0
                            if idx_d.size > 0:
                                V_td = V_tval_abs[idx_d]
                                if metric == 'mean':
                                    summary = np.mean(resids, axis=0)
                                elif metric == 'max':
                                    # Weight by P(tract)
                                    weights = np.power(V_tract[idx_d], params_trace['max_weight_factor'])
                                    wt_tvals = np.abs(np.multiply(weights, V_td))
                                    idx_max = idx_d[np.argmax(wt_tvals)]
                                    idx_max = np.unravel_index(idx_max, img_shape)
                                    summary = V_resids[idx_max[0],idx_max[1],idx_max[2],:]
                                elif metric == 'median':
                                    summary = np.median(resids, axis=0)
                                else:
                                    print('Error: "{0}" is not a valid summary metric'.format(metric))
                                    assert(False)
                
                            
                            summary_resids[i,:] = summary #np.mean(resids.flatten())

                        # Write residuals to file
                        output_file = '{0}/resids_{1}.csv'.format(output_dir, tract_name)
                        np.savetxt(output_file, summary_resids, delimiter=',', \
                                                header='', comments='', \
                                                fmt='%1.8f')

                        # For each distance:
                        for d, i in zip(dists, range(0,dists.size)):
                            # Summarize stat across voxels at this distance (mean, max, median)
                            idx_d = np.flatnonzero(V_dist==d)
                            
                            summary = 0
                            if idx_d.size > 0:
                                if metric == 'mean':
                                    summary_tval = np.mean(V_tval[idx_d])
                                    summary_pval = np.mean(V_pval[idx_d])
                                    summary_fdrpval = np.mean(V_fdrpval[idx_d])
                                    summary_coef = np.mean(V_coef[idx_d])

                                elif metric == 'max':
                                    # Weight by P(tract)
                                    weights = np.power(V_tract[idx_d], params_trace['max_weight_factor'])
                                    wt_tvals = np.abs(np.multiply(weights, V_tval[idx_d]))
                                    idx_max = idx_d[np.argmax(wt_tvals)]
                                    summary_tval = V_tval[idx_max]
                                    summary_pval = V_pval[idx_max]
                                    summary_fdrpval = V_fdrpval[idx_max]
                                    summary_coef = V_coef[idx_max]

                                elif metric == 'median':
                                    summary_tval = np.median(V_tval[idx_d])
                                    summary_pval = np.median(V_pval[idx_d])
                                    summary_fdrpval = np.median(V_fdrpval[idx_d])
                                    summary_coef = np.median(V_coef[idx_d])

                                else:
                                    print('Error: "{0}" is not a valid summary metric'.format(metric))
                                    assert(False)

                            if np.isnan(summary_tval):
                                summary_tval = nanval
                            M[i,j] = summary_coef
                            M[i,j+1] = summary_tval
                            M[i,j+2] = summary_pval
                            M[i,j+3] = summary_fdrpval

                        hdr = '{0},{1}|coef,{1}|tval,{1}|pval,{1}|fdr_pval'.format(hdr,factor_str)
                        j += 4

                    # Write Nd x Ns matrix to CSV file
                    output_file = '{0}/stats_{1}.csv'.format(output_dir, tract_name)

                    np.savetxt(output_file, M, delimiter=',', \
                                            header=hdr, comments='', \
                                            fmt=formats)

                    if verbose:
                        print('   Done {0}'.format(glm) )
        
        return True

    
    # Perform cluster-wise inference using permutation testing, for distance-wise
    # statistics, and save results as (1) ModelGUI polyline files (per GLM/factor); 
    # (2) volume files with thresholded distance-wise summary statistics (per GLM/factor); 
    # and (3) a CSV summary table (per GLM)
    #
    # tract_thres:   The threshold defining how much of the tract to use (default=0.1)
    # alpha:         The alpha level for cluster inference (default=0.05)
    # fdr_method:    FDR method for correction on clusters across all tracts; 
    #                 one of 'none' (don't perform), or any method in 
    #                 statsmodels.stats.multitest.multipletests
    #                 (default='none')
    # fdr_alpha:     The alpha level for FDR correction (default=0.05)
    # clobber:       Whether to overwrite existing results (default=False)
    # verbose:       Whether to print progress to screen (default=False)
    # debug:         Whether to print debug info to screen (default=False)
    #
    
    def extract_distance_trace_clusters( self, clobber=False, verbose=False, debug=False ):
        
        params_glm = self.params['glm']
        params_gen = self.params['general']
        params_tracts = self.params['tracts']
        params_trace = self.params['traces']
        
        if params_trace['tract_threshold'] < self.tract_threshold:
            print('Trace threshold ({0:1.3f}) cannot be less than GLM threshold ({1:1.3f}).' \
                  .format(params_trace['tract_threshold'], self.tract_threshold))
            return False
        tract_thresh = params_trace['tract_threshold']
        thresh_str = '{0:02d}'.format(round(tract_thresh*100))
        
        cluster_alpha = params_trace['cluster_alpha']
        fdr_method = params_trace['fdr_method']
        fdr_alpha = params_trace['fdr_alpha']
        alpha = params_trace['pval_alpha']
        plot_perms = params_trace['plot_permutations']
        if plot_perms:
            plt.ioff()
            sns.set_context("paper", font_scale=0.8)
        N_perm = params_trace['n_permutations']
        N_perm = 500

        metric = params_gen['summary_metric']
        min_clust = params_gen['min_clust']
        dist_dir = os.path.join(self.tracts_dir, 'dist')
        final_dir = os.path.join(self.tracts_dir, 'final')
        N_sub = len(self.subjects)
        
        tract_thresh = params_trace['tract_threshold']
        params_avdir = params_tracts['average_directions']
        
        use_norm = params_avdir['use_normalized']
        
        std_str = ''
        if params_gen['standardized_beta']:
            std_str = '_std'
        
        if N_perm == 0:
            print('Invalid number of permutations: 0')
            return False
        
        for glm in params_glm:
            if verbose:
                print( glm )
            
            glm_dir = '{0}/glms/{1}'.format(self.tracts_dir, glm)
            output_dir = '{0}/summary-{1}_thr{2}'.format(glm_dir, metric, thresh_str)
            if not os.path.exists(output_dir):
                if debug:
                    print('{0} does not exist...'.format(output_dir))
                print('The function extract_distance_traces must be run before extract_distance_trace_clusters!')
                return False

            polyline_dir = '{0}/polylines/perm'.format(self.tracts_dir)
            if not os.path.exists(polyline_dir):
                os.makedirs(polyline_dir)
            figures_dir = '{0}/figures/permutations'.format(glm_dir, metric)
            if plot_perms and not os.path.exists(figures_dir):
                os.makedirs(figures_dir)
                
            pval_output_dir = '{0}/pval-perm_thr{1}'.format(glm_dir, thresh_str)
            tval_output_dir = '{0}/tval-perm_thr{1}'.format(glm_dir, thresh_str)
            
            tval_dir = '{0}/tval'.format(glm_dir)

            pval_dir = '{0}/pval'.format(glm_dir)
            
            if not os.path.isdir(polyline_dir):
                os.makedirs(polyline_dir)
            if os.path.isdir(pval_output_dir):
                if not clobber:
                    print(' Output directory exists! [clobber=True to overwrite]: {0}'.format(pval_output_dir))
                    return False
                shutil.rmtree(pval_output_dir)
            os.makedirs(pval_output_dir)
            if os.path.isdir(tval_output_dir):
                if not clobber:
                    print(' Output directory exists! [clobber=True to overwrite]: {0}'.format(tval_output_dir))
                    return False
                shutil.rmtree(tval_output_dir)
            os.makedirs(tval_output_dir)
                
            factors = params_glm[glm]['factors'].copy()
                        
            has_sig = {}
            tcounts_pos = {}
            tcounts_neg = {}
            pvals_pos = {}
            pvals_neg = {}
            pvals_perm_pos = {}
            pvals_perm_neg = {}
            clusters_pos = {}
            clusters_neg = {}
            tsums_pos = {}
            tsums_neg = {}
            tmax_pos = {}
            tmax_neg = {}
            tmean_all_pos = {}
            tmean_all_neg = {}
            
            for factor in factors[1:]:
                
                tvals_all = {}
                tvals_nt_all = {}
                pvals_all = {}
                clusters_all = {}
                
                # For permutation test
                tvals_perm = {}
                pvals_perm = {}
                
                if verbose:
                    print(' {0}'.format(factor))

                factor_str = factor.replace('*','X')
                tracts_found = []
                
                offset = 0

                for tract_name in tqdm_notebook(self.tract_names,'Performing permutation cluster testing'.format(factor)):
                    if verbose:
                        print(' {0}'.format(tract_name))
                    stats_file = '{0}/stats_{1}.csv'.format(output_dir, tract_name)
                    resids_file = '{0}/resids_{1}.csv'.format(output_dir, tract_name)
                    if not os.path.isfile(stats_file):
                        if verbose:
                            print('Stats not found for {0} at {1}. Skipping.'.format(tract_name, stats_file))
                        continue
                        
                    T = pd.read_csv(stats_file)
                    tvals = T['{0}|tval'.format(factor_str)].values
                    pvals = T['{0}|pval'.format(factor_str)].values
                    tvals_abs = np.abs(tvals)
                    pvals[tvals_abs<0.1] = 1.0
                    t_max = np.max(tvals_abs)
                    N_nodes = tvals.size
                    
                    clusters = utils.get_clusters( pvals, tvals, cluster_alpha, min_clust )
                    
#                     print(tvals)
#                     print(pvals)
#                     print(clusters)
                    cvals = np.unique(clusters)
                    cvals = cvals[cvals>0]

                    tvals_nt = tvals.copy()
                    tvals[clusters==0] = 0
                    pvals[clusters==0] = 1
                    tvals_abs = np.abs(tvals)
                    
                    # If permutations have been specified, apply permutation testing here
                    if cvals.size > 0 and N_perm > 0:
                        
                        if use_norm:
                            tract_file = '{0}/tract_final_norm_bidir_{1}.nii.gz'.format(final_dir, tract_name)
                        else:
                            tract_file = '{0}/tract_final_bidir_{1}.nii.gz'.format(final_dir, tract_name)
                        V_tract = nib.load(tract_file).get_fdata().flatten()
                        tval_file = '{0}/{1}_{2}{3}.nii.gz'.format(tval_dir, tract_name, factor_str, std_str)
                        V_tval = nib.load(tval_file).get_fdata().flatten()
                        pval_file = '{0}/{1}_{2}{3}.nii.gz'.format(pval_dir, tract_name, factor_str, std_str)
                        V_pval = nib.load(pval_file).get_fdata().flatten()
                        dist_file = '{0}/dist/dist_bidir_{1}.nii.gz'.format(self.tracts_dir, tract_name)
                        V_dist = nib.load(dist_file).get_fdata().flatten()
                        V_dist[V_tract < tract_thresh] = 0
                        dists = np.unique(V_dist[V_dist>0]).astype(int)

                        # Generate N_perm random permutations of the t-values
                        t_perm, p_perm = utils.get_permuted_tvals( dists, V_dist, V_tval, N_perm, metric, V_pval=V_pval )
                       
                        # We want a list of maximal cluster t-value sums
                        t_max_sums = np.zeros(N_perm)
                        
                        # Compute clusters for the permuted t-values
                        for p in range(0, N_perm):
                            tvals_p = t_perm[p,:]
                            pvals_p = p_perm[p,:]
                            clusters_p = utils.get_clusters( pvals_p, tvals_p, cluster_alpha, 1 )
                            cvals_p = np.unique(clusters_p)
                            cvals_p = cvals_p[cvals_p>0]
                            tmax = 0
                            for cval in cvals_p:
                                idx_c = clusters_p==cval
                                tmax = max(np.sum(tvals_p[idx_c]), tmax)
                            t_max_sums[p] = tmax
                       
                        if plot_perms:
                            #n_bins = max(min(N_perm/50,100),40)
                            n_bins = 50
                            fig = plt.figure(figsize=[5,5])
                            output_file = '{0}/hist_perm_{1}_{2}_thr{3}' \
                                           .format(figures_dir, factor_str, tract_name, thresh_str)
                            df = pd.DataFrame(data={'Max_tsum': t_max_sums})
                            sns.displot(df, x="Max_tsum", bins=n_bins, stat="probability")
                            idx_alpha = math.ceil(N_perm*(1-alpha))
                            tsort = np.sort(t_max_sums)
                            p_thr = tsort[idx_alpha]
                            for cval in cvals:
                                idx_c = clusters==cval
                                tm = np.sum(tvals_abs[idx_c])
                                clr = [0.2,1.,0.2]
                                if tm < p_thr:
                                    clr = [0.4,0.4,0.4]
                                plt.axvline(tm, 0, 0.5, color=clr, linewidth=0.7)
                            
                            plt.axvline(p_thr, 0, 1., color=[1.,0.2,0.2])
                            plt.savefig( output_file, facecolor=[1,1,1], transparent=True, dpi=300 )
                            plt.close('all');
                    
                        # Compute p-values as the proportion of t_max_sums
                        # greater than the t-value sum of each cluster
                        pvals_p = np.ones(tvals.shape)
                        clusters_p = np.zeros(clusters.shape)
                        tsums = np.zeros(cvals.size)
                        k = 1
                        for cval, j in zip(cvals, range(0,cvals.size)):
                            idx_c = clusters==cval
                            tsum = np.sum(tvals_abs[idx_c])
                            tsums[j] = tsum
                            pval = np.sum(tsum < t_max_sums) / N_perm
                            pvals_p[idx_c] = pval
                            if pval < alpha:
                                clusters_p[idx_c] = k
                                k += 1

                        pvals = pvals_p
                        tvals[pvals >= alpha] = 0
                        clusters = clusters_p

                    tvals_all[tract_name] = tvals
                    tvals_nt_all[tract_name] = tvals_nt
                    pvals_all[tract_name] = pvals
                    clusters_all[tract_name] = clusters

                    tracts_found.append(tract_name)
                    
                        
                # Determine FDR and correct clusters accordingly
                if not fdr_method=='none':
                   
                    pvals = []
                    clusters = []
                    tracts = []

                    for tract_name in tracts_found:
                        pvals_t = pvals_all[tract_name]
                        tvals_t = tvals_all[tract_name]
                        clusters_t = clusters_all[tract_name]

                        cvals = np.unique(clusters_t)
                        cvals = cvals[cvals>0]
                        for c in cvals:
                            pvals.append(np.max(pvals_t[clusters_t==c]))
                            tracts.append(tract_name)
                            clusters.append(c)

                    # Apply FDR to extracted p-values
                    try:
                        R = smm.multipletests(pvals, fdr_alpha, fdr_method)
                        pvals_fdr = R[1]
                        sig_fdr = R[0]
                    except ZeroDivisionError:
                        print(' Could not perform FDR for {0} due to ZeroDivisionError!'.format(factor))
                        return False

                    # Update pvals and tvals with FDR-corrected results
                    pvals_all[tract_name].fill(1)
                    for i in range(0,len(pvals)):
                        tract_name = tracts[i]
                        cluster = clusters_all[tract_name]
                        pvals_all[tract_name][cluster==clusters[i]] = pvals_fdr[i]
                        if not sig_fdr[i]:
                            tvals_all[tract_name][cluster==clusters[i]] = 0

                has_sig[factor] = []
                tcounts_pos[factor] = {}
                tcounts_neg[factor] = {}
                pvals_pos[factor] = {}
                pvals_neg[factor] = {}
                clusters_pos[factor] = {}
                clusters_neg[factor] = {}
                tsums_pos[factor] = {}
                tsums_neg[factor] = {}
                tmax_pos[factor] = {}
                tmax_neg[factor] = {}
                tmean_all_pos[factor] = {}
                tmean_all_neg[factor] = {}        
                        
                for tract_name in tqdm_notebook(tracts_found,'Saving results'.format(factor)):
                    
                    pvals = pvals_all[tract_name]
                    tvals = tvals_all[tract_name]
                    tvals_nt = tvals_nt_all[tract_name]
                    clusters = clusters_all[tract_name]
                    
                    # Append results to CSV file
                    stats_file = '{0}/stats_{1}.csv'.format(output_dir, tract_name)
                    T = pd.read_csv(stats_file)
                    T['{0}|perm_pval'.format(factor_str)] = pvals
                    T['{0}|perm_clusters'.format(factor_str)] = clusters
                    T.to_csv(stats_file, index=False)

                    # Write volume files
                    dist_file = '{0}/dist/dist_bidir_{1}.nii.gz'.format(self.tracts_dir, tract_name)
                    pvals_file = '{0}/{1}_{2}.nii.gz'.format(pval_output_dir, tract_name, factor_str)
                    tvals_file = '{0}/{1}_{2}.nii.gz'.format(tval_output_dir, tract_name, factor_str)
                    V_img = nib.load(dist_file)
                    V_dist = V_img.get_fdata()
                    V_img.header['datatype'] = 16
                    V_tvals = np.zeros(V_dist.shape)
                    V_pvals = np.zeros(V_dist.shape)
                    
                    dists = T['Distance'].values
                    for d, i in zip(dists, range(0,dists.size)):
                        idx = V_dist==d
                        V_pvals[idx] = pvals[i]
                        V_tvals[idx] = tvals[i]

                    nz = np.count_nonzero(V_tvals)
                    nma = np.min(np.abs(V_tvals))

                    img = nib.Nifti1Image(V_pvals, V_img.affine, V_img.header)
                    nib.save(img, pvals_file)
                    img = nib.Nifti1Image(V_tvals, V_img.affine, V_img.header)
                    nib.save(img, tvals_file)

                    if np.any(np.logical_and(tvals != 0, pvals < alpha)):
                        has_sig[factor].append(tract_name)
                        tcounts_pos[factor][tract_name] = np.count_nonzero(tvals>0)
                        tcounts_neg[factor][tract_name] = np.count_nonzero(tvals<0)
                        tsums_pos[factor][tract_name] = np.sum(tvals[tvals>0])
                        tsums_neg[factor][tract_name] = -np.sum(tvals[tvals<0])
                        tmax_pos[factor][tract_name] = 0
                        tmax_neg[factor][tract_name] = 0
                        tmean_all_pos[factor][tract_name] = 0
                        tmean_all_neg[factor][tract_name] = 0

                        if tsums_pos[factor][tract_name] > 0:
                            tmax_pos[factor][tract_name] = np.max(tvals[tvals>0])

                        if np.sum(tvals_nt[tvals_nt>0]) > 0:
                            tmean_all_pos[factor][tract_name] = np.mean(tvals_nt[tvals_nt>0])

                        if tsums_neg[factor][tract_name] > 0:
                            tmax_neg[factor][tract_name] = np.max(-tvals[tvals<0])

                        if np.sum(-tvals_nt[tvals_nt<0]) > 0:
                            tmean_all_neg[factor][tract_name] = np.mean(-tvals_nt[tvals_nt<0])

                    # Make new polyline from bidirectional distance volume
                    # Polyline vertices are maximal tract values at each distance
                    if use_norm:
                        tract_file = '{0}/tract_final_norm_bidir_{1}.nii.gz'.format(final_dir, tract_name)
                    else:
                        tract_file = '{0}/tract_final_bidir_{1}.nii.gz'.format(final_dir, tract_name)
                    V_img = nib.load(tract_file)
                    M = V_img.affine
                    V_tract = V_img.get_fdata()
                    V_dist[V_tract < tract_thresh] = 0
                    polyline = np.ones((dists.size,4))
                    idx_ok = np.zeros(dists.size, dtype=bool)
                    
                    V_dist = V_dist.flatten()
                    V_tract = V_tract.flatten()
                    
                    for d, i in zip(dists, range(0,dists.size)):
                        idx_d = np.flatnonzero(V_dist==d)
                        V_t = V_tract[idx_d]
                        if np.any(idx_d):
                            idx_mx = np.unravel_index(idx_d[np.argmax(V_t)], V_img.shape)
                            polyline[i,0:3] = np.asarray(idx_mx)
                            idx_ok[i] = True

                    # Save to polyline file
                    T = np.transpose(np.stack((tvals_nt,tvals,pvals)))
                    polyline = polyline[idx_ok,:]
                    polyline = np.matmul(M,polyline.T).T
                    polyline = polyline[:,0:3]
                    polyline = utils.smooth_polyline_ma(polyline, window=7)
                    T = T[idx_ok,:]
                    
                    data_names = ['tvals','tvals_thr','pvals']
                    polyline_file = '{0}/tvals_perm_{1}_{2}_{3}.poly3d' \
                                    .format(polyline_dir, tract_name, factor_str, thresh_str)
                    utils.write_polyline_mgui(polyline, polyline_file, factor_str, T, data_names)
                    
            print('   Wrote volumes & polylines.')

            # Write distance-wise 1D-RFT results to CSV file
            df = pd.DataFrame({'Factor': [], 'From': [], 'To': [], 'Direction': [], 'T_count': [], 'T_sum': [], \
                               'T_max': [], 'T_mean_all': []})
            
            for factor in factors[1:]:

                factor_str = factor.replace('*','X')
                # Sort by values
                tnames = np.array(has_sig[factor])

                tcnt_pos = []
                tcnt_neg = []
                tcnt_abs = []
                for tract_name in tnames:
                    tcnt_pos.append(tcounts_pos[factor][tract_name])
                    tcnt_neg.append(tcounts_neg[factor][tract_name])
                    tcnt_abs.append(tcounts_pos[factor][tract_name] + tcounts_neg[factor][tract_name])

                idx = np.argsort(-np.array(tcnt_abs))
                tnames = tnames[idx]

                if debug:
                    print('   Has significant t-values [{0}]:'.format(factor))
                
                for tract_name in tnames:
                    # Parse it
                    idx = [m.start() for m in re.finditer('_',tract_name)]
                    if len(idx) == 1:
                        tract_1 = tract_name[1:idx]
                        tract_2 = tract_name[idx+1:]
                    if not len(idx) % 2 == 0:
                        idx = idx[math.ceil(len(idx) / 2) - 1]
                        tract_1 = tract_name[0:idx]
                        tract_2 = tract_name[idx+1:]
                    else:
                        tract_1 = tract_name
                        tract_2 = tract_name
                        if verbose:
                            print("   Can't parse {0}".format(tract_name))

                    row = pd.DataFrame({'Factor': [factor_str], 'From': [tract_1], 'To': [tract_2], \
                                        'Direction': 'pos', 'T_count': [tcounts_pos[factor][tract_name]], \
                                        'T_sum': [tsums_pos[factor][tract_name]], \
                                        'T_max': [tmax_pos[factor][tract_name]], \
                                        'T_mean_all': [tmean_all_pos[factor][tract_name]]})
                    df = df.append(row, ignore_index=True)
                    row = pd.DataFrame({'Factor': [factor_str], 'From': [tract_1], 'To': [tract_2], \
                                        'Direction': 'neg', 'T_count': [tcounts_neg[factor][tract_name]], \
                                        'T_sum': [tsums_neg[factor][tract_name]], \
                                        'T_max': [tmax_neg[factor][tract_name]], \
                                        'T_mean_all': [tmean_all_neg[factor][tract_name]]})
                    df = df.append(row, ignore_index=True)

            # Save results
            df.to_csv('{0}/tcounts-perm.csv'.format(output_dir), index=False)
            if verbose:
                print(' Wrote results to {0}'.format('{0}/tcounts-perm.csv'.format(output_dir)))

            return True
    
    # Perform 1D random field cluster-wise inference for distance-wise
    # statistics, and save results as (1) ModelGUI polyline files (per GLM/factor); 
    # (2) volume files with thresholded distance-wise summary statistics (per GLM/factor); 
    # and (3) a CSV summary table (per GLM)
    #
    # tract_thres:   The threshold defining how much of the tract to use (default=0.1)
    # alpha:         The alpha level for cluster inference (default=0.05)
    # fdr_method:    FDR method for correction on clusters across all tracts; 
    #                 one of 'none' (don't perform), or any method in 
    #                 statsmodels.stats.multitest.multipletests
    #                 (default='none')
    # fdr_alpha:     The alpha level for FDR correction (default=0.05)
    # clobber:       Whether to overwrite existing results (default=False)
    # verbose:       Whether to print progress to screen (default=False)
    # debug:         Whether to print debug info to screen (default=False)
    #
    def extract_distance_traces_rft1d( self, tract_thres=0.1, alpha=0.05, fdr_method='none', fdr_alpha=0.05, \
                                       clobber=False, verbose=False, debug=False ):
        
        params_glm = self.params['glm']
        params_gen = self.params['general']
        params_trace = self.params['traces']
        params_tracts = self.params['tracts']
        metric = params_gen['summary_metric']
        min_clust = params_gen['min_clust']
        dist_dir = os.path.join(self.tracts_dir, 'dist')
        final_dir = os.path.join(self.tracts_dir, 'final')
        seed_dilate = self.params['tracts']['gaussians']['seed_dilate']
        N_sub = len(self.subjects)
        
        if params_trace['tract_threshold'] < self.tract_threshold:
            print('Trace threshold ({0:1.3f}) cannot be less than GLM threshold ({1:1.3f}).' \
                  .format(params_trace['tract_threshold'], self.tract_threshold))
            return False
        tract_thresh = params_trace['tract_threshold']
        thresh_str = '{0:02d}'.format(round(tract_thresh*100))
        
        params_avdir = params_tracts['average_directions']    
        use_norm = params_avdir['use_normalized']
        
        for glm in params_glm:
            if verbose:
                print( glm )
            
            glm_dir = '{0}/glms/{1}'.format(self.tracts_dir, glm)
            output_dir = '{0}/summary-{1}_thr{2}'.format(glm_dir, metric, thresh_str)
            polyline_dir = '{0}/polylines/rft'.format(self.tracts_dir)
            pval_output_dir = '{0}/pval-rft_thr{1}'.format(glm_dir, thresh_str)
            tval_output_dir = '{0}/tval-rft_thr{1}'.format(glm_dir, thresh_str)
            
            if not os.path.isdir(polyline_dir):
                os.makedirs(polyline_dir)
            if os.path.isdir(pval_output_dir):
                if not clobber:
                    print(' Output directory exists! [clobber=True to overwrite]: {0}'.format(pval_output_dir))
                    return False
                shutil.rmtree(pval_output_dir)
            os.makedirs(pval_output_dir)
            if os.path.isdir(tval_output_dir):
                if not clobber:
                    print(' Output directory exists! [clobber=True to overwrite]: {0}'.format(tval_output_dir))
                    return False
                shutil.rmtree(tval_output_dir)
            os.makedirs(tval_output_dir)
                
            factors = params_glm[glm]['factors'].copy()
                        
            has_sig = {}
            tcounts_pos = {}
            tcounts_neg = {}
            pvals_pos = {}
            pvals_neg = {}
            pvals_perm_pos = {}
            pvals_perm_neg = {}
            clusters_pos = {}
            clusters_neg = {}
            tsums_pos = {}
            tsums_neg = {}
            tmax_pos = {}
            tmax_neg = {}
            tmean_all_pos = {}
            tmean_all_neg = {}
            
            for factor in factors[1:]:
                
                factor_str = factor.replace('*','X')
                tvals_all = {}
                tvals_nt_all = {}
                pvals_all = {}
                clusters_all = {}
                
                # For permutation test
                tvals_perm = {}
                pvals_perm = {}
                
                if verbose:
                    print(' {0}'.format(factor))

                # Get RF parameters from concatenated t-values
                mean_FWHM = 0
                denom = 0

                for tract_name in self.tract_names:
                    stats_file = '{0}/stats_{1}.csv'.format(output_dir, tract_name)
                    resids_file = '{0}/resids_{1}.csv'.format(output_dir, tract_name)
                    if not os.path.isfile(resids_file):
                        if verbose:
                            print('   No output for {0}. Skipping.'.format(tract_name))
                        continue
                    resids = np.loadtxt(resids_file, delimiter=',')
                    nz = np.sum(resids,1) != 0  # Remove where sum is exactly zero (no data here)
                    resids = resids[nz,:]

                    try:
                        FWHM = rft1d.geom.estimate_fwhm(resids.T)
                        if not np.isinf(FWHM):
                            mean_FWHM = mean_FWHM + FWHM
                            denom = denom + 1
                    except:
                        if debug:
                            print('    Tract {0} has too few tests ({1})! Setting this tract to zeros.' \
                                  .format(tract_name, resids.shape[0]))

                mean_FWHM = mean_FWHM / denom    
                df = N_sub - 1

                if debug:
                    print('    Mean FWHM is {0:1.5f}'.format(mean_FWHM))

                tracts_found = []
                
                offset = 0
                for tract_name in tqdm_notebook(self.tract_names,'Performing cluster inference'.format(factor)):
                    stats_file = '{0}/stats_{1}.csv'.format(output_dir, tract_name)
                    resids_file = '{0}/resids_{1}.csv'.format(output_dir, tract_name)
                    if not os.path.isfile(stats_file):
                        continue
                        
                    T = pd.read_csv(stats_file)
                    tvals = T['{0}|tval'.format(factor_str)].values
                    tvals_abs = np.abs(tvals)
                    t_max = np.max(tvals_abs)
                    N_nodes = tvals.size
                    
                    pvals, clusters = utils.get_tvalue_rft1d_clusters( tvals, alpha, df, mean_FWHM, min_clust )

                    tvals_nt = tvals.copy()
                    tvals[pvals > alpha] = 0
                    
                    tvals_all[tract_name] = tvals
                    tvals_nt_all[tract_name] = tvals_nt
                    pvals_all[tract_name] = pvals
                    clusters_all[tract_name] = clusters

                    tracts_found.append(tract_name)

                        
                # Determine FDR and correct clusters accordingly
                if not fdr_method=='none':
                   
                    pvals = []
                    clusters = []
                    tracts = []

                    for tract_name in tracts_found:
                        pvals_t = pvals_all[tract_name]
                        tvals_t = tvals_all[tract_name]
                        clusters_t = clusters_all[tract_name]

                        cls = np.unique(clusters_t)
                        for c in cls:
                            pvals.append(np.max(pvals_t[clusters_t==c]))
                            tracts.append(tract_name)
                            clusters.append(c)

                    # Apply FDR to extracted p-values
                    try:
                        R = smm.multipletests(pvals, fdr_alpha, fdr_method)
                        pvals_fdr = R[1]
                        sig_fdr = R[0]
                    except ZeroDivisionError:
                        print(' Could not perform FDR for {0} due to ZeroDivisionError!'.format(factor))
                        return False

                    # Update pvals and tvals with FDR-corrected results
                    pvals_all[tract_name].fill(1)
                    for i in range(0,len(pvals)):
                        tract_name = tracts[i]
                        cluster = clusters_all[tract_name]
                        pvals_all[tract_name][cluster==clusters[i]] = pvals_fdr[i]
                        if not sig_fdr[i]:
                            tvals_all[tract_name][cluster==clusters[i]] = 0

                has_sig[factor] = []
                tcounts_pos[factor] = {}
                tcounts_neg[factor] = {}
                pvals_pos[factor] = {}
                pvals_neg[factor] = {}
                clusters_pos[factor] = {}
                clusters_neg[factor] = {}
                tsums_pos[factor] = {}
                tsums_neg[factor] = {}
                tmax_pos[factor] = {}
                tmax_neg[factor] = {}
                tmean_all_pos[factor] = {}
                tmean_all_neg[factor] = {}        
                        
                for tract_name in tqdm_notebook(tracts_found,'Saving results'):
                    
                    pvals = pvals_all[tract_name]
                    tvals = tvals_all[tract_name]
                    tvals_nt = tvals_nt_all[tract_name]
                    clusters = clusters_all[tract_name]
                    
                    # Append results to CSV file
                    stats_file = '{0}/stats_{1}.csv'.format(output_dir, tract_name)
                    T = pd.read_csv(stats_file)
                    T['{0}|rft_pval'.format(factor_str)] = pvals
                    T['{0}|rft_clusters'.format(factor_str)] = clusters
                    T.to_csv(stats_file, index=False)

                    # Write volume files
                    dist_file = '{0}/dist/dist_bidir_{1}.nii.gz'.format(self.tracts_dir, tract_name)
                    pvals_file = '{0}/{1}_{2}.nii.gz'.format(pval_output_dir, tract_name, factor_str)
                    tvals_file = '{0}/{1}_{2}.nii.gz'.format(tval_output_dir, tract_name, factor_str)
                    V_img = nib.load(dist_file)
                    V_dist = V_img.get_fdata()
                    V_img.header['datatype'] = 16
                    V_tvals = np.zeros(V_dist.shape)
                    V_pvals = np.zeros(V_dist.shape)
                    
                    dists = T['Distance'].values
                    for d, i in zip(dists, range(0,dists.size)):
                        idx = V_dist==d
                        V_pvals[idx] = pvals[i]
                        V_tvals[idx] = tvals[i]

                    nz = np.count_nonzero(V_tvals)
                    nma = np.min(np.abs(V_tvals))

                    img = nib.Nifti1Image(V_pvals, V_img.affine, V_img.header)
                    nib.save(img, pvals_file)
                    img = nib.Nifti1Image(V_tvals, V_img.affine, V_img.header)
                    nib.save(img, tvals_file)

                    if np.any(np.logical_and(tvals != 0, pvals < alpha)):
                        has_sig[factor].append(tract_name)
                        tcounts_pos[factor][tract_name] = np.count_nonzero(tvals>0)
                        tcounts_neg[factor][tract_name] = np.count_nonzero(tvals<0)
                        tsums_pos[factor][tract_name] = np.sum(tvals[tvals>0])
                        tsums_neg[factor][tract_name] = -np.sum(tvals[tvals<0])
                        tmax_pos[factor][tract_name] = 0
                        tmax_neg[factor][tract_name] = 0
                        tmean_all_pos[factor][tract_name] = 0
                        tmean_all_neg[factor][tract_name] = 0

                        if tsums_pos[factor][tract_name] > 0:
                            tmax_pos[factor][tract_name] = np.max(tvals[tvals>0])

                        if np.sum(tvals_nt[tvals_nt>0]) > 0:
                            tmean_all_pos[factor][tract_name] = np.mean(tvals_nt[tvals_nt>0])

                        if tsums_neg[factor][tract_name] > 0:
                            tmax_neg[factor][tract_name] = np.max(-tvals[tvals<0])

                        if np.sum(-tvals_nt[tvals_nt<0]) > 0:
                            tmean_all_neg[factor][tract_name] = np.mean(-tvals_nt[tvals_nt<0])

                    # Make new polyline from bidirectional distance volume
                    # Polyline vertices are maximal tract values at each distance
                    if use_norm:
                        tract_file = '{0}/tract_final_norm_bidir_{1}.nii.gz'.format(final_dir, tract_name)
                    else:
                        tract_file = '{0}/tract_final_bidir_{1}.nii.gz'.format(final_dir, tract_name)
                    V_img = nib.load(tract_file)
                    M = V_img.affine
                    V_tract = V_img.get_fdata()
                    V_dist[V_tract < tract_thresh] = 0
                    polyline = np.ones((dists.size,4))
                    idx_ok = np.zeros(dists.size, dtype=bool)
                    
                    V_dist = V_dist.flatten()
                    V_tract = V_tract.flatten()
                    
                    for d, i in zip(dists, range(0,dists.size)):
                        idx_d = np.flatnonzero(V_dist==d)
                        V_t = V_tract[idx_d]
                        if np.any(idx_d):
                            idx_mx = np.unravel_index(idx_d[np.argmax(V_t)], V_img.shape)
                            polyline[i,0:3] = np.asarray(idx_mx)
                            idx_ok[i] = True

                    # Save to polyline file
                    T = np.transpose(np.stack((tvals_nt,tvals,pvals)))
                    polyline = polyline[idx_ok,:]
                    polyline = np.matmul(M,polyline.T).T
                    polyline = polyline[:,0:3]
                    polyline = utils.smooth_polyline_ma(polyline, window=7)
                    T = T[idx_ok,:]
                    
                    data_names = ['tvals','tvals_thr','pvals']
                    polyline_file = '{0}/tvals_rft_{1}_{2}_{3}.poly3d' \
                                    .format(polyline_dir, tract_name, factor_str, thresh_str)
                    utils.write_polyline_mgui(polyline, polyline_file, factor_str, T, data_names)
                    
            print('   Wrote volumes & polylines.')

            # Write distance-wise 1D-RFT results to CSV file
            df = pd.DataFrame({'Factor': [], 'From': [], 'To': [], 'Direction': [], 'T_count': [], 'T_sum': [], \
                               'T_max': [], 'T_mean_all': []})
            
            for factor in factors[1:]:

                factor_str = factor.replace('*','X')
                # Sort by values
                tnames = np.array(has_sig[factor])

                tcnt_pos = []
                tcnt_neg = []
                tcnt_abs = []
                for tract_name in tnames:
                    tcnt_pos.append(tcounts_pos[factor][tract_name])
                    tcnt_neg.append(tcounts_neg[factor][tract_name])
                    tcnt_abs.append(tcounts_pos[factor][tract_name] + tcounts_neg[factor][tract_name])

                idx = np.argsort(-np.array(tcnt_abs))
                tnames = tnames[idx]

                if debug:
                    print('   Has significant t-values [{0}]:'.format(factor))
                
                for tract_name in tnames:
#                     if debug:
#                         print('   {0} [t-count (+)={1}] [t-sum (+)={2}] [t-max (+)={3}]' \
#                               .format(tract_name, tcounts_pos[factor][tract_name], tsums_pos[factor] \
#                                       [tract_name], tmax_pos[factor][tract_name]))
#                         print('   {0} [t-count (-)={1}] [t-sum (-)={2}] [t-max (+)={3}]' \
#                               .format(tract_name, tcounts_neg[factor][tract_name], tsums_neg[factor] \
#                                       [tract_name], tmax_neg[factor][tract_name]))
                    # Parse it
                    idx = [m.start() for m in re.finditer('_',tract_name)]
                    if len(idx) == 1:
                        tract_1 = tract_name[1:idx]
                        tract_2 = tract_name[idx+1:]
                    if not len(idx) % 2 == 0:
                        idx = idx[math.ceil(len(idx) / 2) - 1]
                        tract_1 = tract_name[0:idx]
                        tract_2 = tract_name[idx+1:]
                    else:
                        tract_1 = tract_name
                        tract_2 = tract_name
                        if verbose:
                            print("   Can't parse {0}".format(tract_name))

                    row = pd.DataFrame({'Factor': [factor_str], 'From': [tract_1], 'To': [tract_2], \
                                        'Direction': 'pos', 'T_count': [tcounts_pos[factor][tract_name]], \
                                        'T_sum': [tsums_pos[factor][tract_name]], \
                                        'T_max': [tmax_pos[factor][tract_name]], \
                                        'T_mean_all': [tmean_all_pos[factor][tract_name]]})
                    df = df.append(row, ignore_index=True)
                    row = pd.DataFrame({'Factor': [factor_str], 'From': [tract_1], 'To': [tract_2], \
                                        'Direction': 'neg', 'T_count': [tcounts_neg[factor][tract_name]], \
                                        'T_sum': [tsums_neg[factor][tract_name]], \
                                        'T_max': [tmax_neg[factor][tract_name]], \
                                        'T_mean_all': [tmean_all_neg[factor][tract_name]]})
                    df = df.append(row, ignore_index=True)

            # Save results
            df.to_csv('{0}/tcounts-rft.csv'.format(output_dir), index=False)
            if verbose:
                print(' Wrote results to {0}'.format('{0}/tcounts-rft.csv'.format(output_dir)))

            return True
    
    
        
    # Aggregate all tracts and save result as single image file
    # 
    # prefix:     Prefix of the final tract file
    # op:         Aggregation to perform (one of "sum" or "max")
    # threshold:  Threshold to apply to tract probability (default=0)
    # intersects: Path to image file specifying a mask; only tracts that intersect this mask
    #              will be included in the aggregate image. Can be None. (default=None)
    # verbose:    Whether to print progress to screen
    #
    def aggregate_tracts( self, prefix, suffix='', op='max', threshold=0, intersects=None, verbose=False ):
        
        img_files = []
        params_tracts = self.params['tracts']
        
        V_mask = None
        
        if intersects is not None:
            V_mask = nib.load(intersects).get_fdata()
            V_mask = np.squeeze(V_mask)
#             print(V_mask.shape)

        for roi1 in self.rois:
            for roi2 in self.rois:
                fname = '{0}/final/{1}_{2}_{3}.nii.gz'.format(self.tracts_dir, prefix, roi1, roi2)
                if os.path.isfile(fname):
                    img_files.append(fname)
        
        V_img = nib.load(img_files[0])
        V_agg = np.zeros(V_img.shape)

        for file in img_files:
            V = nib.load(file).get_fdata()
            V[V<threshold] = 0
            
            if V_mask is not None:
                V_int = np.logical_and(V_mask, V>0)
                if not np.any(V_int):
#                     if verbose:
#                         print(' {0} not in mask.'.format( file ) ) 
                    continue
            
            if op == 'sum':
                V_agg = V_agg + V
            elif op == 'max':
                V_agg = np.maximum(V_agg,V)
            if verbose:
                print(' Added {0}'.format(os.path.basename(file)))

        img = nib.Nifti1Image(V_agg, V_img.affine, V_img.header)
        output_file = '{0}/{1}_{2}_All_{3}{4}.nii.gz'.format(self.tracts_dir, prefix, \
                                                             params_tracts['general']['network_name'], \
                                                             op, suffix)
        nib.save(img,output_file)
        if verbose:
            print('Output saved to {0}'.format(output_file))
        return True
    
    
    # Aggregate stats voxelwise across tracts for all GLMs & factors and save to image files
    #
    # op:          The aggregation to perform; one of "max" or "mean"
    # pthres:      The p-value threshold to apply (default=0.05)
    # suffix:      Suffix indicating the statistical approach used to obtain the t-statistics
    #               (default = '-rft')
    # intersects:  Path to image file specifying a mask; only tracts that intersect this mask
    #               will be included in the aggregate image. Can be None. (default=None)
    # verbose:     Whether to print progress to screen
    #
    def aggregate_stats( self, op='max', pthres=0.05, suffix='-rft', intersects=None, verbose=False, clobber=False ):
        
        params_glm = self.params['glm']
        V_mask = None
        
        tract_thresh = self.params['traces']['tract_threshold']
        thresh_str = '{0:02d}'.format(round(tract_thresh*100))
        
        if intersects is not None:
            V_mask = nib.load(intersects).get_fdata()
        
        for glm in params_glm:
            if verbose:
                print( glm )
                
            output_dir = '{0}/glms/{1}/aggregate'.format(self.tracts_dir, glm)
            if os.path.isdir(output_dir):
                if not clobber:
                    print('Output directory {0} exists; use "clobber=True" to overwrite.'.format(output_dir))
                    return False
                shutil.rmtree(output_dir)
            os.makedirs(output_dir)
                
            factors = params_glm[glm]['factors'].copy()
            for factor in factors[1:]:
                fname = factor.replace('*','X')
                if verbose:
                    print(' {0}'.format(factor))

                timg_files = glob('{0}/glms/{1}/tval{2}_thr{3}/*_{4}.nii*' \
                                  .format(self.tracts_dir, glm, suffix, thresh_str, fname))
                pimg_files = glob('{0}/glms/{1}/pval{2}_thr{3}/*_{4}.nii*' \
                                  .format(self.tracts_dir, glm, suffix, thresh_str, fname))

                V_img = nib.load(timg_files[0])

                V_agg_pos = np.zeros(V_img.shape)
                V_agg_neg = np.zeros(V_img.shape)

                for tfile,pfile in zip(timg_files,pimg_files):
                    Vt = nib.load(tfile).get_fdata()
                    if not np.any(Vt):
                        continue
                        
                    if V_mask is not None:
                        V_int = np.logical_and(V_mask, np.abs(Vt)>0)
                        if not np.any(V_int):
                            if verbose:
                                print(' {0} not in mask.'.format( timg_files ) )
                            continue
                              
                    Vp = nib.load(pfile).get_fdata()
                    Vt[Vp > pthres] = 0
                    if op=='max':
                        V_agg_pos = np.maximum(V_agg_pos,Vt)
                        V_agg_neg = np.minimum(V_agg_neg,Vt)
                    elif op=='mean':
                        V_agg_pos = V_agg_pos + Vt
                    else:
                        raise Exception('Invalid operation: {0}'.format(op))

                    if verbose:
                        print('  Added {0} [{1}]'.format(os.path.basename(tfile), np.count_nonzero(Vt)))
                            
                if op=='max':          
                    img = nib.Nifti1Image(V_agg_pos, V_img.affine, V_img.header)
                    output_file = '{0}/tstat{1}_{2}_all_pos_max.nii.gz'.format(output_dir, suffix, fname)
                    nib.save(img,output_file)
                    img = nib.Nifti1Image(V_agg_neg, V_img.affine, V_img.header)
                    output_file = '{0}/tstat{1}_{2}_all_neg_min.nii.gz'.format(output_dir, suffix, fname)
                    nib.save(img,output_file)
                else:
                    V_agg_pos = V_agg_pos / len(timg_files)
                    img = nib.Nifti1Image(V_agg_pos, V_img.affine, V_img.header)
                    output_file = '{0}/tstat{1}_{2}_all_mean.nii.gz'.format(output_dir, suffix, fname)
                    nib.save(img,output_file)
                    
                if verbose:
                    print(' Output saved to {0}/glms/{1}'.format(self.tracts_dir, glm))
                    
        return True
    
    
    # Creates Pajek graphs for each GLM/factor, with edge weights representing
    # aggregate statistics across distances (specified by "edge_val") for each ROI pair.
    # 
    # Separate graphs are created for positive and negative effects
    #
    # suffix:      Suffix indicating the statistical approach used to obtain the t-statistics
    #               (default = "-perm")
    # edge_val:    The value to assign as edge weights; one of "tsum" (sum of all t-values), 
    #               "tcount" (count of significant t-values), or tmean (mean of all t-values)
    # verbose:     Whether to print progress to screen
    #
    def create_pajek_graphs( self, suffix='-perm', edge_val='tsum', verbose=False, clobber=False ):
        
        params_glm = self.params['glm']
        params_gen = self.params['general']
        metric = params_gen['summary_metric']
        
        tract_thresh = self.params['traces']['tract_threshold']
        thresh_str = '{0:02d}'.format(round(tract_thresh*100))
        
        roi_centers = utils.compute_roi_centers( self.rois_dir, self.rois, output_file=None, \
                                                 extension=self.roi_suffix, verbose=False )
        
        for glm in params_glm:
            if verbose:
                print( glm )
                
            output_dir = '{0}/glms/{1}/pajek'.format(self.tracts_dir, glm)
            if not os.path.isdir(output_dir):
                os.makedirs(output_dir)
                
            stats_file = '{0}/glms/{1}/summary-{2}_thr{3}/tcounts{4}.csv' \
                            .format(self.tracts_dir, glm, metric, thresh_str, suffix)
            T = pd.read_csv(stats_file)
            # If tract AB has t>1
            T = T[T['T_count']>1]
            T_pos = T[T['Direction']=='pos']
            T_neg = T[T['Direction']=='neg']
            
            factors = params_glm[glm]['factors'].copy()
            for factor in factors[1:]:
                factor_str = factor.replace('*','X')
                if verbose:
                    print(' {0}'.format(factor))
                    
                T_fpos = T_pos[T_pos['Factor']==factor_str]
                T_fneg = T_neg[T_neg['Factor']==factor_str]

                N_fpos = len(T_fpos.index)
                N_fneg = len(T_fneg.index)
                N_roi = len(self.rois)

                # Write stats for this factor
                A = np.zeros((N_roi, N_roi))
                for itr, row in T_fpos.iterrows():
                    ii = self.rois.index(row.From)
                    jj = self.rois.index(row.To)
                    if edge_val == 'tcount':
                        A[ii,jj] = row.T_count
                    elif edge_val == 'tsum':
                        A[ii,jj] = row.T_sum
                    elif edge_val == 'tmean':
                        A[ii,jj] = row.T_mean_all
                    else:
                        raise Exception('Invalid edge_val "{0}"; must be one of "tsum", "tcount", or "tmean".'.format(edge_val))
                    
                output_file = '{0}/{1}{2}_{3}_pos.net'.format(output_dir, edge_val, suffix, factor_str)

                utils.write_matrix_to_pajek( A, output_file, directed=False, labels=self.rois, coords=roi_centers )
                if verbose:
                    print('  Wrote positive Pajek graph for {0}'.format(factor_str))

                #if N_fneg > 0:
                A = np.zeros((N_roi, N_roi))
                for itr, row in T_fneg.iterrows():
                    ii = self.rois.index(row.From)
                    jj = self.rois.index(row.To)
                    if edge_val == 'tcount':
                        A[ii,jj] = row.T_count
                    elif edge_val == 'tsum':
                        A[ii,jj] = row.T_sum
                    elif edge_val == 'tmean':
                        A[ii,jj] = row.T_mean_all
                output_file = '{0}/{1}{2}_{3}_neg.net'.format(output_dir, edge_val, suffix, factor_str)

                utils.write_matrix_to_pajek( A, output_file, directed=False, labels=self.rois, coords=roi_centers )
                if verbose:
                    print('  Wrote negative Pajek graph for {0}'.format(factor_str))
                    
        return True
                    
    
    # Aggregate stats distance-wise (i.e., for each distance along the tract) for all GLMs 
    # & factors.
    # 
    #
    def aggregate_distance_stats( self, verbose=False ):
        
        return False
        
        
        