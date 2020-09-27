import numpy as np
import csv
import nibabel as nib
import os
import statsmodels.api as sm
import shutil
import statsmodels.stats.multitest as smm
import scipy.stats as stats
import rft1d
import seaborn as sns
import matplotlib
import matplotlib.pylab as plt
import pandas as pd
from tqdm import tnrange, tqdm_notebook, tqdm
from . import utils

# Plot distance-wise t-values as a set of horizontal lines
#
# params:            Dict of parameters used to generate results
# alpha:             The alpha threshold to apply to p values
# tract_names:       List of the tracts to plot
# stat_type:         The type of FWE correction applied; one of 'uncorrected' (default),
#                     'fdr', or 'rft'.
# verbose:           Whether to print progress to screen
#
def plot_distance_traces( params, tract_names, alpha=0.5, stat_type='uncorrected', verbose=False ):
    
    params_glm = params['glm']
    params_gen = params['general']
    params_tracts = params['tracts']
    
    source_dir = params_tracts['general']['source_dir']
    project_dir = os.path.join(source_dir, params_tracts['general']['project_dir'])
    tracts_dir = os.path.join(project_dir, params_tracts['general']['tracts_dir'], params_tracts['general']['network_name'])

    # Set font display parameters
    plt.rc('axes', titlesize=params['axis_font']) 
    plt.rc('axes', labelsize=params['axis_font'])
    plt.rc('figure', titlesize=params['title_font'])
    plt.rc('xtick', labelsize=params['xticklabel_font'])
    plt.rc('ytick', labelsize=params['yticklabel_font']) 
        
    metric = params_gen['summary_metric']

    for glm in params_glm:
        factors = params_glm[glm]['factors']

        glm_dir = '{0}/{1}/{2}'.format(tracts_dir, params_gen['glm_output_dir'], glm)
        output_dir = '{0}/summary-{1}'.format(glm_dir, metric)
        
        figures_dir = '{0}/figures'.format(glm_dir)
        if not os.path.isdir(figures_dir):
            shutil.os.makedirs(figures_dir)

        offset = 0
        
        suffix = ''
        if stat_type=='fdr':
            suffix = 'fdr'
        elif stat_type=='rft':
            suffix = 'rft'

        has_sig = {}
        for factor in factors[1:]:
            has_sig[factor] = []
            
        tracts_plotted = []
        for tract_name in tract_names:
            stats_file = '{0}/stats_{1}.csv'.format(output_dir, tract_name)
            if os.path.isfile(stats_file):
                tracts_plotted.append(tract_name)

        for factor in factors[1:]:
            factor_str = factor.replace('*','X')
            plt.figure(figsize=(20,12))
#             offset = len(tracts_plotted)-1
            for tract_name, offset in zip(tracts_plotted, range(len(tracts_plotted)-1,-1,-1)):
                stats_file = '{0}/stats_{1}.csv'.format(output_dir, tract_name)
                T = pd.read_csv(stats_file)
                tvals = T['{0}|tval'.format(factor)].values
                px = ''
                if len(suffix) > 0:
                    px = '{0}_'.format(suffix)
                pvals = T['{0}|{1}pval'.format(factor, px)].values
                tvals[pvals > alpha] = 0
                
                if np.any(np.logical_and(tvals != 0, pvals < alpha)):
                    has_sig[factor].append(tract_name)
                    if verbose:
                        print('{0} | {1}'.format(offset, tract_name))

                tvals = tvals + offset

                plt.plot(tvals)

            plt.yticks(list(range(0,len(tracts_plotted))), tracts_plotted[::-1])
            ax = plt.gca()
            ax.set_xlabel('Distance (mm)')
            if stat_type=='fdr':
                plt.title('T statistics for {0} [FDR corrected]'.format(factor))
            elif stat_type=='rft':
                plt.title('T statistics for {0} [1D-RFT corrected]'.format(factor))
            else:
                plt.title('T statistics for {0} [uncorrected]'.format(factor))
            
            sx = ''
            if stat_type=='fdr' or stat_type=='rft':
                sx = '_{0}'.format(stat_type)
            plt.savefig('{0}/lines_{1}{2}.png'.format(figures_dir, factor_str, sx))
#             plt.show()

# Plot the distance-wise results of GLM analyses for each GLM, factor, and tract. 
#
# params:           Parameters for plotting:
#                      tract_name:     The tract for which to generate plots
#                      glm:            The GLM for which to generate plots
#                      alpha:          The p-value threshold to determine significance
#                      stat_type:      The type of FWE correction applied; one of 
#                                       'uncorrected', 'fdr', or 'rft'.
#                      outlier_z:      The z-value absolute threshold to identify and remove 
#                                       outliers
#                      axis_font:      Font size for axis text
#                      ticklabel_font: Font size for tick labels
#                      title_font:     Font size for figure titles
# my_glm:           DwiTractsGlm object specifying the GLMs and data structures  
# verbose:          Whether to print progress to screen
#
def plot_glm_results_all( params, my_glm, verbose=False ):
    
    for tract in tqdm_notebook(my_glm.tract_names, 'Generating plots'):
        params['tract_name'] = tract
        plot_glm_results( params, my_glm, verbose )
    

# Plot the distance-wise results of GLM analyses for each GLM and factor. 
# Plots are generated by averaging TSA values for all significant clusters; positive and 
# negative clusters are treated separately. 
#
# For categorical variables, violin plots are generated, comparing TSA scores for each level 
# of the variable (currently only supports two levels)
# 
# For continuous variables, scatterplots are generated, showing the relationship
# between the factor and TSA scores, including regression line and confidence interval
#
# For interaction terms with a categorical variable, scatterplots are generated for each 
# level of the categorical variable (currently only supports two levels)
#
# params:           Parameters for plotting:
#                      tract_name:     The tract for which to generate plots
#                      glm:            The GLM for which to generate plots
#                      alpha:          The p-value threshold to determine significance
#                      stat_type:      The type of FWE correction applied; one of 
#                                       'uncorrected', 'fdr', or 'rft'.
#                      outlier_z:      The z-value absolute threshold to identify and remove 
#                                       outliers
#                      axis_font:      Font size for axis text
#                      ticklabel_font: Font size for tick labels
#                      title_font:     Font size for figure titles
# my_glm:           DwiTractsGlm object specifying the GLMs and data structures  
# verbose:          Whether to print progress to screen
#
def plot_glm_results( params, my_glm, verbose=False ):
    
    sns.set(style="white", palette="muted", color_codes=True)
    plot_face_clr = [1,1,1]
    plot_colors = [ 
                    '#1f77b4',  # muted blue
                    '#ff7f0e',  # safety orange
                    '#2ca02c',  # cooked asparagus green
                    '#d62728',  # brick red
                    '#9467bd',  # muted purple
                    '#8c564b',  # chestnut brown
                    '#e377c2',  # raspberry yogurt pink
                    '#7f7f7f',  # middle gray
                    '#bcbd22',  # curry yellow-green
                    '#17becf'   # blue-teal
                ]
    
    params_gen = my_glm.params['general']
    params_glm = my_glm.params['glm']
    params_tracts = my_glm.params['tracts']
    
    source_dir = params_tracts['general']['source_dir']
    project_dir = os.path.join(source_dir, params_tracts['general']['project_dir'])
    tracts_dir = os.path.join(project_dir, params_tracts['general']['tracts_dir'], params_tracts['general']['network_name'])
    
    metric = params_gen['summary_metric']
    glm = params['glm']
    glm_dir = '{0}/{1}/{2}'.format(tracts_dir, params_gen['glm_output_dir'], glm)
    output_dir = '{0}/summary-{1}'.format(glm_dir, metric)

    tract_name = params['tract_name']
    stats_file = '{0}/stats_{1}.csv'.format(output_dir, tract_name)
    if not os.path.isfile(stats_file):
        if verbose:
            print('Tract not found: {0}'.format(tract_name))
        return False
    
    stat_type = params['stat_type']
    categorical = params_glm[glm]['categorical']
    
    figures_dir = '{0}/figures'.format(glm_dir)
    if not os.path.isdir(figures_dir):
        shutil.os.makedirs(figures_dir)
        
    def plot_factor( factor, betas_pos, betas_neg, show_title=True, figsize=None ):
        
        nonlocal categorical
        
        # Average betas for each cluster
        f = factors.index(factor)
        
        # Plot betas versus factor based on type of term (continuous, categorical, or interaction)
        if '*' in factor:
            # Plot as interaction (two X two regression scatterplots)
            plot_interaction( factor, betas_pos, betas_neg, show_title=show_title, figsize=figsize )
        elif categorical[factor]:
            # Plot as categorical (two violin plots)
            plot_violins( factor, betas_pos, betas_neg, show_title=show_title, figsize=figsize )
        else:
            # Plot as continuous (two X one regression scatterplots)
            plot_continuous( factor, betas_pos, betas_neg, show_title=show_title, figsize=figsize )
                             
      
    def load_betas( ):
        
        nonlocal tract_name
        nonlocal my_glm
        nonlocal tracts_dir
        
        subjects = my_glm.subjects
        
        params_tracts = my_glm.params['tracts']
        params_regress = params_tracts['dwi_regressions']
        params_glm = my_glm.params['glm']
                
        dist_file = '{0}/dist/tract_dist_bidir_{1}.nii.gz'.format(tracts_dir, tract_name)
        tract_file = '{0}/final/tract_final_norm_bidir_{1}.nii.gz'.format(tracts_dir, tract_name)
        V_img = nib.load(dist_file)
        V_dist = np.round(V_img.get_fdata().flatten())
        V_tract = nib.load(tract_file).get_fdata().flatten()

        idx_tract = np.flatnonzero(V_tract > params_glm[glm]['tract_thres'])
        V_dist = V_dist[idx_tract]

        N_sub = len(subjects)

        V_betas = np.zeros((N_sub, idx_tract.size))
        # Get betas for each subject
        for subject, i in zip(subjects, range(0,N_sub)):
            prefix_sub = '{0}{1}'.format(params_tracts['general']['prefix'], subject)
            subject_dir = os.path.join(project_dir, params_tracts['general']['deriv_dir'], prefix_sub, \
                                       params_tracts['general']['sub_dirs'])
            subj_output_dir = '{0}/dwi/{1}/{2}'.format(subject_dir, params_regress['regress_dir'], \
                                                       params_tracts['general']['network_name'])
            beta_file = '{0}/betas_mni_sm_{1}um_{2}.nii.gz' \
                                  .format(subj_output_dir, int(1000.0*params_regress['beta_sm_fwhm']), tract_name)

            V_sub = nib.load(beta_file).get_fdata()
            V_betas[i,:] = V_sub.ravel()[idx_tract]
            
        return V_betas, V_dist, idx_tract
    
    
    def get_average_betas( factor, V_betas, V_dist, idx_tract ):
        
        nonlocal verbose
        nonlocal params
        nonlocal my_glm
        nonlocal tracts_dir
        
        subjects = my_glm.subjects
        params_gen = my_glm.params['general']
        params_tracts = my_glm.params['tracts']
        glm_dir = '{0}/{1}/{2}'.format(tracts_dir, params_gen['glm_output_dir'], params['glm'])
        metric = params_gen['summary_metric']
        summary_dir = '{0}/summary-{1}'.format(glm_dir, metric)
    
        factor_str = factor.replace('*','X')
        std_str = ''
        if params_gen['standardized_beta']:
            std_str = '_std'

        N_sub = len(subjects)
        df = N_sub - 1
        alpha = params['alpha']
        stat_type = params['stat_type']
            
        # Get t-values and threshold based on stat_type
        tval_file = '{0}/tval/{1}_{2}{3}.nii.gz'.format(glm_dir, tract_name, factor_str, std_str)
        V_tvals = nib.load(tval_file).get_fdata().flatten()
        V_tvals = V_tvals[idx_tract]
            
        stats_file = '{0}/stats_{1}.csv'.format(summary_dir, tract_name)
        T = pd.read_csv(stats_file)
        tvals = T['{0}|tval'.format(factor)].values.copy()
        tvals_abs = np.abs(tvals)
        N_nodes = tvals.size
        pvals = np.zeros(N_nodes)
        
        if stat_type == 'uncorrected':
            pvals = T['{0}|pval'.format(factor)].values.copy()
        elif stat_type == 'fdr':
            pvals = T['{0}|fdr_pval'.format(factor)].values.copy()
        elif stat_type == 'rft':
            pvals = T['{0}|rft_pval'.format(factor)].values.copy()
        else:
            raise Exception('Invalid statistic type: {0}. Must be one of "uncorrected", "fdr", or "rft"')

        tvals[pvals > alpha] = 0
        dists = T['Distance'].values
        
        # Next average betas for positive and negative clusters separately
        betas_neg = np.zeros(N_sub)
        betas_pos = np.zeros(N_sub)
        bb_pos = np.zeros((N_sub, dists.size))
        bb_neg = np.zeros((N_sub, dists.size))
        count_pos = 0
        count_neg = 0
        for d, i in zip(dists, range(0, dists.size)):
            idx_d = V_dist==d
            if not np.any(idx_d):
                # Probably because tract threshold is too high...
#                 print('No idx at dist {0}'.format(d))
                continue
            V_bd = V_betas[:,idx_d]
            V_t = V_tvals[idx_d]
            if tvals[i] > 0:
                count_pos += 1
                if metric == 'max':
                    bb_pos[:,i] = V_bd[:,np.argmax(V_t)]
                elif metric == 'mean':
                    bb_pos[:,i] = np.mean(V_bd,axis=1)
                elif metric == 'median':
                    bb_pos[:,i] = np.median(V_bd,axis=1)
            if tvals[i] < 0:
                count_neg += 1
                if metric == 'max':
                    bb_neg[:,i] = V_bd[:,np.argmin(V_t)]
                elif metric == 'mean':
                    bb_neg[:,i] = np.mean(V_bd,axis=1)
                elif metric == 'median':
                    bb_neg[:,i] = np.median(V_bd,axis=1)

        if count_pos > 0:
            betas_pos = np.sum(bb_pos,axis=1) / count_pos
            # Remove outliers
            if params['outlier_z'] > 0:
                zthres = [betas_pos.mean() - params['outlier_z'] * betas_pos.std(), \
                          betas_pos.mean() + params['outlier_z'] * betas_pos.std()]
#                 if verbose:
#                     print('{0}: outliers [{1}, {2}]'.format(tract_name, zthres[0], zthres[1]))
            else:
                zthres = [float('-inf'), float('inf')]
            idx_out = np.flatnonzero(~np.logical_and(betas_pos >= zthres[0], betas_pos <= zthres[1]))
            betas_pos[idx_out] = np.nan

        if count_neg > 0:
            betas_neg = np.sum(bb_neg,axis=1) / count_neg
             # Remove outliers
            if params['outlier_z'] > 0:
                zthres = [betas_neg.mean() - params['outlier_z'] * betas_neg.std(), \
                          betas_neg.mean() + params['outlier_z'] * betas_neg.std()]
#                 if verbose:
#                     print('{0}: outliers [{1}, {2}]'.format(tract_name, zthres[0], zthres[1]))
            else:
                zthres = [float('-inf'), float('inf')]
            idx_out = np.flatnonzero(~np.logical_and(betas_neg >= zthres[0], betas_neg <= zthres[1]))
            betas_neg[idx_out] = np.nan
        
        return betas_pos, betas_neg
        
        
    def get_regression_stats( x, y, alpha=0.05, add_constant=True):

        if add_constant:
            x = sm.tools.tools.add_constant(x)

        result = sm.OLS(y, x).fit()
        df = result.get_prediction(x).summary_frame()
        ci = np.vstack((df['mean_ci_lower'], df['mean_ci_upper']))
        y_pred = result.predict(x)

        return result, y_pred, ci
                             
                             
    def plot_interaction( factor, betas_pos, betas_neg, show_title=True, figsize=None ):
        
        nonlocal verbose
        nonlocal params
        nonlocal my_glm
        nonlocal categorical
        nonlocal figures_dir
        nonlocal plot_colors
        
        params_glm = my_glm.params['glm']
        alpha = params['alpha']
        stat_type = params['stat_type']
        glm = params['glm']
        X = my_glm.Xs[glm]
        factors = params_glm[glm]['factors']
        
        # Get component factors and determine which is which
        parts = factor.split('*')
        fac1 = parts[0]
        fac2 = parts[1]
        
        try:
            if categorical[fac2]:
                fac1 = parts[1]
                fac2 = parts[0]
            elif not categorical[fac1]:
                if verbose:
                    print('Interaction {0} is invalid!'.format( factor ))
                return False
            
        except:
            if verbose:
                print('Interaction {0} is invalid!'.format( factor ))
            return False
        
        factor_str = factor.replace('*','X')
        
        i1 = factors.index(fac1)
        i2 = factors.index(fac2)
        
        # Two X two scatterplots for beta_pos, beta_neg
        levels = params_glm[glm]['levels'][fac1]
        if len(levels) != 2:
            print('{0}: Only two-level interaction plots are implemented!'.format( factor ))
            return False
        
        # Ensure labels are ordered corrected (lowest first)
        vals = []
        lnames = []
        for key in levels:
            vals.append(levels[key])
            lnames.append(key)
        if vals[1] < vals[0]:
            lnames = reversed(lnames)
        
        has_pos = np.sum(betas_pos[~np.isnan(betas_pos)] != 0) > 0
        has_neg = np.sum(betas_neg[~np.isnan(betas_neg)] != 0) > 0

        n_rows = int(has_pos) + int(has_neg)
        if n_rows == 0:
#             print('No significant t-values for {0}!'.format(factor))
            return False

        if figsize is None:
            figsize = (30, 14*n_rows)
        else:
            figsize=(figsize[0]*n_rows, figsize[1])
        
        fig, axs = plt.subplots(1, n_rows, sharey=True, figsize=figsize)
#         fig, axs = plt.subplots(n_rows, 1, sharey=True, figsize=figsize)
    
        if n_rows == 1:
            axs = np.array([axs])
           
        if show_title:
            fig.suptitle('TSA v. {0} by {1}: {2}'.format(fac2, fac1, tract_name))
            
        # Sort variables by X2
        X2 = X[:,i2]
        idx = np.argsort(X2)
        X2 = X2[idx]
        X1 = X[idx,i1]
        betas_pos = betas_pos[idx]
        betas_neg = betas_neg[idx]

        idx_row = 0
        
        for title, betas in zip(['Positive', 'Negative'], [betas_pos, betas_neg]):
            if np.sum(betas) != 0:
                
                for s, j in zip([-1,1], [0,1]):
                    idx_s = np.logical_and(~np.isnan(betas), X1==s)
                    x = X2[idx_s]
                    y = betas[idx_s]
                    results, y_est, ci = get_regression_stats(x, y, alpha)
                    r = stats.pearsonr(y_est,y)
                    r2 = results.rsquared

                    axs[idx_row].plot(x, y_est, color=plot_colors[j])
                    axs[idx_row].fill_between(x, ci[0,:], ci[1,:], alpha=0.2)
                    axs[idx_row].plot(x, y, 'o', linewidth=0, label=lnames[j])
#                     axs[idx_row].set_title('{0} interaction [{1}]'.format(title, lnames[j]))
                    
                idx_row += 1
  
        for ax in axs.flat:
            ax.set(xlabel=fac2, ylabel='TSA')
            if params['show_legend']:
                ax.legend(loc='lower right')
        
        suffix = ''
        if stat_type=='fdr' or stat_type=='rft':
            suffix = '_{0}'.format(stat_type)

        plt.savefig( '{0}/scatter_{1}_{2}{3}.png'.format( figures_dir, factor_str, tract_name, suffix ), \
                     facecolor=plot_face_clr, \
                     transparent=True )
        
        plt.close()
        
    def plot_continuous( factor, betas_pos, betas_neg, show_title=True, figsize=None ):
        
        nonlocal my_glm
        nonlocal params
        
        has_pos = np.sum(betas_pos[~np.isnan(betas_pos)] != 0) > 0
        has_neg = np.sum(betas_neg[~np.isnan(betas_neg)] != 0) > 0
        
        alpha = params['alpha']
        stat_type = params['stat_type']
        
        glm = params['glm']
        X = my_glm.Xs[glm]
        params_glm = my_glm.params['glm']
        factors = params_glm[glm]['factors']

        n_rows = int(has_pos) + int(has_neg)
        if n_rows == 0:
            return False

        if figsize is None:
            figsize=(n_rows*15, 14)
        else:
            figsize=(figsize[0]*n_rows, figsize[1])
            
        fig, axs = plt.subplots(1, n_rows, sharey=True, figsize=figsize)
        if n_rows == 1:
            axs = np.array([axs])
            
        if show_title:
            fig.suptitle('TSA v. {0}: {1}'.format(factor, tract_name))
        
        i1 = factors.index(factor)
        X1 = X[:,i1]
        idx = np.argsort(X1)
        
        idx_row = 0
        
        for title, betas in zip(['Positive', 'Negative'], [betas_pos, betas_neg]):
            if np.sum(betas != 0) > 0:
                x = X1[idx]
                y = betas[idx]
                idx_ok = ~np.isnan(y)
                x = x[idx_ok]
                y = y[idx_ok]
                results, y_est, ci = get_regression_stats(x, y, alpha)
                r = stats.pearsonr(y_est,y)
                r2 = results.rsquared

                axs[idx_row].plot(x, y_est)
                axs[idx_row].fill_between(x, ci[0,:], ci[1,:], alpha=0.2)
                axs[idx_row].plot(x, y, 'bo', linewidth=0)
                if show_title:
                    axs[idx_row].set_title('{0} association'.format(title))
                idx_row += 1
  
        for ax in axs.flat:
            ax.set(xlabel=factor, ylabel='TSA')
        
        suffix = ''
        if stat_type=='fdr' or stat_type=='rft':
            suffix = '_{0}'.format(stat_type)
            
        plt.savefig( '{0}/scatter_{1}_{2}{3}.png'.format( figures_dir, factor, tract_name, suffix ), \
                     facecolor=plot_face_clr, \
                     transparent=True )
        
        plt.close()
                                    
              
    def plot_violins( factor, betas_pos, betas_neg, show_title=True, figsize=None ):
        
        nonlocal params
        nonlocal my_glm
        
        params_glm = my_glm.params['glm']
        stat_type = params['stat_type']
        glm = params['glm']
        X = my_glm.Xs[glm]
        factors = params_glm[glm]['factors']
        
        has_pos = np.sum(betas_pos[~np.isnan(betas_pos)] != 0) > 0
        has_neg = np.sum(betas_neg[~np.isnan(betas_neg)] != 0) > 0

        n_rows = int(has_pos) + int(has_neg)
        if n_rows == 0:
            return False
        
        levels = params_glm[glm]['levels'][factor]
        if len(levels) != 2:
            print('{0}: Only two-level interaction plots are implemented!'.format( factor ))
            return False

        # Ensure labels are ordered corrected (lowest first)
        vals = []
        lnames = []
        for key in levels:
            vals.append(levels[key])
            lnames.append(key)
        
        if vals[1] < vals[0]:
            lnames = reversed(lnames)
        
        factor_str = factor.replace('*','X')   
        
        f = factors.index(factor)
        x = X[:,f]
        
        x_txt = []
        for i in x:
            if i == -1:
                x_txt.append(lnames[0])
            else:
                x_txt.append(lnames[1])
        x = np.array(x_txt)
        
        if figsize is None:
            figsize = (14*n_rows, 14)
        else:
            figsize=(figsize[0]*n_rows, figsize[1])
        
        fig, axs = plt.subplots(1, n_rows, sharey=True, figsize=figsize)
        
        if show_title:
            fig.suptitle('TSA [{0} v. {1}]: {2}'.format(lnames[0], lnames[1], tract_name))
        
        titles = ['{0} > {1}'.format(lnames[1], lnames[0]), '{0} > {1}'.format(lnames[0], lnames[1])]
        
        idx_a = 0;
        
        for h, betas, t in zip([has_pos, has_neg], [betas_pos, betas_neg], [0,1]):
            if h:
                if n_rows == 1:
                    axis = axs
                else:
                    axis = axs[idx_a]
                    
                idx_ok = ~np.isnan(betas)

                df = pd.DataFrame(data={'TSA': betas[idx_ok], factor_str: x[idx_ok]})
                df[factor_str].astype('category')
                    
                sns.violinplot(x=factor_str, y='TSA', ax=axis, data=df, order=lnames)
                for v in axis.collections[::2]:
                    v.set_alpha(0.3)
                sns.stripplot(x=factor_str, y='TSA', ax=axis, data=df, order=lnames, s=params['marker_size'])
                
                if show_title:
                    axis.set_title(titles[t])

                idx_a += 1   
         
        suffix = ''
        if stat_type=='fdr' or stat_type=='rft':
            suffix = '_{0}'.format(stat_type)
        
        plt.savefig( '{0}/violin_{1}_{2}{3}.png'.format( figures_dir, factor_str, tract_name, suffix ), \
                     facecolor=plot_face_clr, \
                     transparent=True )
        
        plt.close()


    # Load the betas and plot each factor
    V_betas, V_dist, idx_tract = load_betas( )
    
    # Set font display parameters
    plt.rc('axes', titlesize=params['axis_font']) 
    plt.rc('axes', labelsize=params['axis_font'])
    plt.rc('figure', titlesize=params['title_font'])
    plt.rc('xtick', labelsize=params['ticklabel_font'])
    plt.rc('ytick', labelsize=params['ticklabel_font'])
    plt.rc('legend', fontsize=params['legend_font'])
    plt.rc('legend', markerscale=params['legend_scale'])
    plt.rc('lines', markersize=params['marker_size'])
    plt.rc('lines', linewidth=params['line_width'])
    
    factors = params_glm[glm]['factors'].copy()
    if 'Intercept' in factors:
        factors.remove('Intercept')
    
    for factor in factors:
        betas_pos, betas_neg = get_average_betas( factor, V_betas, V_dist, idx_tract )
        cneg = np.count_nonzero(np.isnan(betas_neg))
        cpos = np.count_nonzero(np.isnan(betas_pos))
        if cneg+cpos > 0:
            if verbose:
                print('  {0} [{1}]: Removed {2} outliers.'.format(tract_name, factor, (cneg+cpos)))
#         if verbose:
#             print('{0}[{1}]: {2} pos, {3} neg'.format(tract_name, factor, np.count_nonzero(betas_pos), \
#                                                                           np.count_nonzero(betas_neg)))
        plot_factor( factor, betas_pos, betas_neg, show_title=params['show_title'], figsize=params['dimensions'] )
        
    return True
    
    
# Plots histograms of TSA distributions across subjects
# for: (1) all tracts; and (2) each individual tract
#
# params:           Parameters for plotting:
#                      axis_font:      Font size for axis text
#                      ticklabel_font: Font size for tick labels
#                      title_font:     Font size for figure titles
# my_glm:           DwiTractsGlm object specifying the GLMs and data structures  
# nbins:            Number of bins to use in histograms
# verbose:          Whether to print progress to screen
#
def plot_tsa_distributions( params, my_glm, nbins=40, verbose=False ):
    
    params_tracts = my_glm.params['tracts']
    params_regress = params_tracts['dwi_regressions']
    params_glm = my_glm.params['glm']
   
    for tract in my_glm.tract_names:
        
        tract_file = '{0}/final/tract_final_norm_bidir_{1}.nii.gz'.format(my_glm.tracts_dir, tract_name)
        V_tract = nib.load(tract_file).get_fdata().flatten()
        idx_tract = np.flatnonzero(V_tract > params_glm[glm]['tract_thres'])

        N_sub = len(my_glm.subjects)
        V_tsa = np.zeros((N_sub, idx_tract.size))
        
        # Get TSA for each subject
        for subject, i in zip(subjects, range(0,N_sub)):
            prefix_sub = '{0}{1}'.format(params_tracts['general']['prefix'], subject)
            subject_dir = os.path.join(project_dir, params_tracts['general']['deriv_dir'], prefix_sub, \
                                       params_tracts['general']['sub_dirs'])
            subj_output_dir = '{0}/dwi/{1}/{2}'.format(subject_dir, params_regress['regress_dir'], \
                                                       params_tracts['general']['network_name'])
            beta_file = '{0}/betas_mni_sm_{1}um_{2}.nii.gz' \
                                  .format(subj_output_dir, int(1000.0*params_regress['beta_sm_fwhm']), tract_name)

            V_sub = nib.load(beta_file).get_fdata()
            V_betas[i,:] = V_sub.ravel()[idx_tract]
    
    