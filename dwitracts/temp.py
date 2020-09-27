 # Extract/summarise statistics at each distance along tracts and save results
    #
    # tract_thres:   The threshold defining how much of the tract to use (default=0.1)
    # clobber:       Whether to overwrite existing results
    # verbose:       Whether to print progress to screen
    #
    def extract_distance_traces_bak( self, tract_thres=0.1, clobber=False, verbose=True ):
        
        params_glm = self.params['glm']
        params_gen = self.params['general']
        params_preproc = self.params['preproc']
        params_ptx = params_preproc['probtrackx']
        
        dist_dir = os.path.join(self.tracts_dir, 'dist')
        final_dir = os.path.join(self.tracts_dir, 'final')
        metric = params_gen['summary_metric']
        
        for glm in params_glm:
            glm_dir = '{0}/{1}/{2}'.format(self.tracts_dir, params_gen['glm_output_dir'], glm)
            output_dir = '{0}/summary-{1}'.format(glm_dir, metric)
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
                # Directed tract name (GLM is only in one direction)
                tract_name_dir = '{0}_{1}'.format(roi_a,roi_b)
                tract_file = '{0}/tract_final_bidir_{1}.nii.gz'.format(final_dir, tract_name_dir)
                V_tract = nib.load(tract_file).get_fdata()
                V_tract = V_tract > tract_thres

                # Undirected tract names (distances are in both directions)
                tract_names_dist = ['{0}_{1}'.format(roi_a,roi_b),'{0}_{1}'.format(roi_b,roi_a)]
                
                for tract_name in tract_names_dist:
                    if verbose:
                        print('Evaluating {0}'.format(tract_name))

                    # Read distance image
                    dist_file = '{0}/tract_dist_{1}.nii.gz'.format(dist_dir, tract_name)
                    
                    if not os.path.exists(dist_file):
                        if verbose:
                            print('   No output exists for tract {0}. Skipping.'.format(tract_name))
                    else:
                        V_dist = nib.load(dist_file).get_fdata()
                        V_dist[np.logical_not(V_tract)] = 0  # Exclude to final tract
                        
                        V_shape = V_dist.copy()
                        V_dist = V_dist.flatten()
                        dists = np.unique(V_dist)
                        dists = dists[1:] # Not interested in zeros
                        Nd = dists.size
                        
                        if Nd == 0:
                            if verbose:
                                print('   No tract found in direction {0}. Skipping.'.format(tract_name))
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
                            output_dir = '{0}/summary-{1}'.format(glm_dir, metric)
                           
                            tval_dir = '{0}/tval'.format(glm_dir);
                            coef_dir = '{0}/coef'.format(glm_dir);
                            pval_dir = '{0}/pval'.format(glm_dir);
                            resid_dir = '{0}/resid'.format(glm_dir);
                            j = 1
                            M = np.zeros((Nd, 4*Nf+1))
                            M[:,0] = dists

                            hdr = 'Distance'

                            # Extract residuals
                            resid_file = '{0}/{1}{2}.nii.gz'.format(resid_dir, tract_name_dir, std_str)
                            V_resids = nib.load(resid_file).get_fdata()

                            # For each factor:
                            for factor in factors:
                                factor_str = factor.replace('*','X')

                                # Read stat image
                                tval_file = '{0}/{1}_{2}{3}.nii.gz'.format(tval_dir, tract_name_dir, factor_str, std_str)
                                coef_file = '{0}/{1}_{2}{3}.nii.gz'.format(coef_dir, tract_name_dir, factor_str, std_str)
                                pval_file = '{0}/{1}_{2}{3}.nii.gz'.format(pval_dir, tract_name_dir, factor_str, std_str)
                                fdrpval_file = '{0}/fdr_{1}_{2}{3}.nii.gz'.format(pval_dir, tract_name_dir, factor_str, std_str)
                                V_tval = nib.load(tval_file).get_fdata().flatten()
                                V_tval_abs = np.abs(V_tval)
                                V_pval = nib.load(pval_file).get_fdata().flatten()
                                V_fdrpval = nib.load(fdrpval_file).get_fdata().flatten()
                                V_coef = nib.load(coef_file).get_fdata().flatten()

                                # Get residuals (can change for each factor if metric == 'max')
                                summary_resids = np.zeros((Nd, N_sub))
                                i = 0
                                for d in dists:
                                    idx_d = np.flatnonzero(V_dist==d)
                                    idx_flat = np.unravel_index(idx_d, V_shape.shape)
                                    resids = V_resids[idx_flat[0],idx_flat[1],idx_flat[2],:]
                                    V_td = V_tval_abs[idx_d]

                                    summary = 0
                                    if idx_d.size > 0:
                                        if metric == 'mean':
                                            summary = np.mean(resids.flatten())
                                        elif metric == 'max':
                                            idx_max = np.unravel_index(idx_d[np.argmax(V_td)], V_shape.shape)
                                            summary = V_resids[idx_max[0],idx_max[1],idx_max[2],:]
                                        elif metric == 'median':
                                            summary = np.median(resids.flatten())
                                        else:
                                            print('Error: "{0}" is not a valid summary metric'.format(metric))
                                            assert(False)

                                    summary_resids[i,:] = summary
                                    i += 1

                                # Write residuals to file
                                output_file = '{0}/resids_{1}.csv'.format(output_dir, tract_name)
                                np.savetxt(output_file, summary_resids, delimiter=',', \
                                                        header='', comments='', \
                                                        fmt='%1.8f')

                                # For each distance:
                                i = 0
                                for d in dists:
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
                                            idx_max = idx_d[np.argmax(V_tval_abs[idx_d])]
                                            summary_tval = V_tval[idx_max]
                                            summary_pval = V_pval[idx_max]
                                            summary_fdrpval = V_fdrpval[idx_max]
                                            summary_coef = V_coef[idx_max]

                                        elif metric == 'median':
                                            summary_tval = np.median(V_tval[idx_d])
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
                                    i += 1

                                hdr = '{0},{1}|coef,{1}|tval,{1}|pval,{1}|fdr_pval'.format(hdr,factor)
                                j += 4

                            # Write Nd x Ns matrix to CSV file
                            output_file = '{0}/stats_{1}.csv'.format(output_dir, tract_name)

                            np.savetxt(output_file, M, delimiter=',', \
                                                    header=hdr, comments='', \
                                                    fmt=formats)

                            if verbose:
                                print('   Done {0}'.format(glm) )
        
        return True