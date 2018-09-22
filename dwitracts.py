
import statsmodels.api as sm
import math
import nibabel as nib
import numpy as np
import json
import csv
import os
import glob
import utils
import shutil

# Compute DWI regressions for single subject

# Uses the average orientation computed above to obtain tract-specific measures of
# diffusion for each voxel. Voxel-wise regressions are performed to determine how strongly
# the diffusion profile in each voxel loads (in terms of beta cofficients) onto the average 
# orientation for a given tract.

def process_regression_dwi( subject, params ):

    params_gen = params['general']
    
    with open(params_gen['preproc_config'], 'r') as myfile:
        json_string = myfile.read()
                     
    params_preproc = json.loads(json_string)
    params_pgen = params_preproc['general']
    params_bpx = params_preproc['bedpostx']
    params_ptx = params_preproc['probtrackx']

    params_regress = params['dwi_regressions']
    tmp_dir = params_gen['temp_dir']
    source_dir = params_gen['source_dir']
    project_dir = os.path.join(source_dir, params_gen['project_dir'])
    tracts_dir = os.path.join(project_dir, params_gen['tracts_dir'], params_gen['network_name'])
    rois_dir = os.path.join(source_dir, params_gen['rois_dir'], params_gen['network_name'])
    avdir_dir = os.path.join(tracts_dir, 'avrdir')
    final_dir = os.path.join(tracts_dir, 'final')

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
        with open(params_ptx['roi_list'],'r') as roi_file:
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
            tract_file = '{0}/tract_final_bidir_{1}.nii.gz'.format(final_dir, tract_name)
            tract_files.append(tract_file)
            rois_a.append(roi_a)
            rois_b.append(roi_b)

    prefix_sub = '{0}{1}'.format(params_gen['prefix'], subject)
    subject_dir = os.path.join(project_dir, params_gen['deriv_dir'], prefix_sub, params_gen['sub_dirs'])

    dwi_dir = os.path.join(subject_dir, params_gen['dwi_dir'])
    ptx_dir = os.path.join(dwi_dir,'probtrackX', params_gen['network_name'])

    bet_file = '{0}/{1}_{2}_dwi_bet.nii.gz'.format(dwi_dir, prefix_sub, params_pgen['session'])
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
            
        print('  * Warning: Subject {0} is missing DWI files. Skipping.'.format(subject))
        return -1
    
    # Create output directory
    subj_output_dir = '{0}/dwi/{1}/{2}'.format(subject_dir, params_regress['regress_dir'], params_gen['network_name'])
    if os.path.isdir(subj_output_dir):
        if not params_regress['force_regress']:
            img_files = glob.glob('*.nii.gz')
            if len(img_files) >= len(tract_name):
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
        tract_file_in = '{0}/tract_final_bidir_{1}.nii.gz'.format(final_dir, tract_name)
        tract_file = '{0}/{1}.{2}.tract.nii.gz'.format(tmp_dir, subject, tract_name)
        avrdir_file_in = '{0}/avrdir_{1}.nii.gz'.format(avdir_dir, tract_name)
        avrdir_file = '{0}/{1}.{2}.avrdir.nii.gz'.format(tmp_dir, subject, tract_name)
        
        success = True
        
        if not os.path.isfile(tract_file_in):
            if params_gen['verbose']:
                print('  - Tract {0} not found. Skipping.'.format(tract_name))
        else:
            shell_file = '{0}/{1}/{2}.warp.sh'.format(tmp_dir, subject, tract_name)
            cmd = '{0} -i {1} -o {2} -r {3} -w {4}' \
                  .format(fsl_vecreg, avrdir_file_in, avrdir_file, bet_file, warp_file)
            
            err = utils.run_fsl(cmd)
            if err:
                print('  * Error running fsl_vecreg for subject {0} on tract {1}. Skipping.'.format(subject, tract_name))
                failure_count += 1
                success = False
                if params_regress['fail_on_error']:
                    return failure_count
            if success:             
                cmd = '{0} -i {1} -o {2} -r {3} -w {4}' \
                      .format(fsl_applywarp, tract_file_in, tract_file, bet_file, warp_file)
                err = utils.run_fsl(cmd)
                if err:
                    print('  * Error running fsl_applywarp for subject {0} on tract {1}. Skipping.'.format(subject, tract_name))
                    failure_count += 1
                    success = False
                    if params_regress['fail_on_error']:
                        return failure_count
    
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
                        vv_d = v_d[vv,:]
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
                    smooth_file = '{0}/{1}_mni_sm_{2:1.2f}mm_{3}.nii.gz' \
                                  .format(subj_output_dir, stat, params_regress['beta_sm_fwhm'], tract_name)
                    cmd = '{0} -i {1} -o {2} -r {3} -w {4}' \
                          .format(fsl_applywarp, dwi_files[stat], mni_files[stat], standard_img, invwarp_file)
                    err = utils.run_fsl(cmd)
                    if err:
                        print('  * Error running fsl_applywarp on {0} for subject {1} on tract {2}. Skipping.' \
                            .format(stat, subject, tract_name))
                        failure_count += 1
                        if params_gen['debug']:
                            print(cmd)
                            print(err)
                        success = False

                    if success:
                        cmd = '{0} {1} -kernel gauss {2:1.5f} -fmean {3}' \
                              .format(fsl_maths, mni_files[stat], params_regress['beta_sm_fwhm']/2.1231, smooth_file)
                        err = utils.run_fsl(cmd)
                        if err:
                            print('  * Error running smoothing {0} for subject {1} on tract {2}. Skipping.' \
                                .format(stat, subject, tract_name))
                            failure_count += 1
                            if params_gen['debug']:
                                print(cmd)
                                print(err)
                            success = False

                    if success:
                        cmd = '{0} {1} -mas {2} {3}' \
                              .format(fsl_maths, mni_files[stat], tract_file_in, mni_files[stat])
                        err = utils.run_fsl(cmd)
                        if err:
                            print('  * Error running masking results for subject {0} on tract {1}. Skipping.' \
                                .format(subject, tract_name))
                            failure_count += 1
                            if params_gen['debug']:
                                print(cmd)
                                print(err)
                            success = False

                    if success:
                        cmd = '{0} {1} -mas {2} {3}' \
                              .format(fsl_maths, smooth_file, tract_file_in, smooth_file)
                        err = utils.run_fsl(cmd)
                        if err:
                            print('  * Error running masking results for subject {0} on tract {1}. Skipping.' \
                                .format(subject, tract_name))
                            failure_count += 1
                            if params_gen['debug']:
                                print(cmd)
                                print(err)
                            success = False

                        # Check if output is there
                        if success:
                            if not os.path.isfile(mni_files[stat]) or not os.path.isfile(smooth_file):
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
                                    
                if success:
                    print('  Finished tract {0} for subject {1}'.format(tract_name, subject))
                else:
                    print('  * Warning: Finished tract {0} for subject {1} with failures'.format(tract_name, subject))
    
    return failure_count