import numpy as np
import math
import rft1d
import nibabel as nib
from nilearn import image
import subprocess
from subprocess import Popen 
import os

# Dilates a 3D mask volume itr times
#
# Returns:      The resulting dilated image
#

def dilate_mask(V_mask, itr=1):
    T = np.copy(V_mask)
    dims = T.shape
    
    for p in range(1,itr+1):
        M = np.copy(T)
        for i in range(1,dims[0]-2):
            for j in range(1,dims[1]-2):
                for k in range(1,dims[2]-2):
                    if M[i,j,k] > 0:
                        T[i-1:i+2,j-1:j+2,k-1:k+2]=1
        
    return T
                        
                        
# Computes "distance" in voxels from a seed ROI within a tract mask,
# by assigning the step count for neighbouring voxels at each iteration
# Algorithm is a basic flood-fill approach.
# 
# V_tract:     The mask volume representing the tract
# V_seed:      The seed ROI volume
# dilate:      The number of voxels to dilate the seed (and stop mask) before starting
# V_stop:      Defines a stop mask; function will stop iterating when this mask is found 
#              (plus the buffer). Can be None (no stop mask).
# stop_buffer: Number of iterations to proceed after stop mask has been found
#
# Returns:     The 3D distance volume
#

def get_tract_dist(V_tract, V_seed, dilate=0, V_stop=None, stop_buffer=6, debug=False):
    if stop_buffer < 0:
        stop_buffer = 0
    
    if dilate > 0:
        V_seed = dilate_mask(V_seed, dilate)
    
    V_current = V_seed.copy()
    V_dist = np.zeros(V_tract.shape, dtype=np.int16)
    V_dist[np.greater(V_current,0)] = 1
    itr_dist = 2
    print_next = False
    itr_max = math.inf
    stop_found = False
    
    while np.any(V_current) and itr_dist < itr_max:
        XYZ_current = np.transpose(np.nonzero(V_current))
        for v in range(0, XYZ_current.shape[0]):
            xyz = XYZ_current[v,:]
            
            # Search all neighbours
            for i in range(max(1, xyz[0]-1), min(V_current.shape[0], xyz[0]+2)):
                for j in range(max(1, xyz[1]-1), min(V_current.shape[1], xyz[1]+2)):
                    for k in range(max(1, xyz[2]-1), min(V_current.shape[2], xyz[2]+2)):

                        if V_tract[i,j,k] > 0 and V_dist[i,j,k] == 0:
                            V_dist[i,j,k] = itr_dist
                            
                        if not stop_found and V_stop is not None and V_stop[i,j,k] > 0:
                            stop_found = True
                            itr_max = itr_dist + stop_buffer
                            if debug:
                                print('Stop mask found @ d={0}.'.format(itr_dist))
                            
        # Next iteration
        V_current.fill(0)
        V_current[np.equal(V_dist, itr_dist)] = 1
        if debug:
            print('End d={0}, any={1}'.format(itr_dist, np.any(V_current)))
        
        itr_dist += 1
    
    return V_dist

# Smooths a polyline represented by vertices with a moving average, 
# with window size window, and returns the result.
# 
# vertices:   Nx3 numpy array containing coordinates constituting the polyline 
# window:     Number of vertices to smooth across for a given vertex. This 
#             should be an odd number; otherwise it will be treated as window+1
#
# Returns:    The resulting polyline
#

def smooth_polyline_ma(vertices, window=5):
    
    polyline = np.copy(vertices)
    win = math.ceil((window-1)/2)
    n = polyline.shape[0]
    
    for i in range(0, n):
        i0 = max(0,i-win)
        i1 = min(i+win,n)
        sub = vertices[i0:i1,:]
        p = np.mean(sub, axis=0)
        polyline[i,:] = p
  
    return polyline


# Converts from voxel to world coordinates, given a (Nibabel) NIFTI1 header object
# Uses the sform transform.
#
# V_coords:      Nx3 numpy aray, containing a set of voxel indices
# header:        A NIFTI1 header object, with the sform defined
#

def voxel_to_world(v_coords, header):
    
    n = v_coords.shape[0]
    T = header.get_sform()
    coords = np.zeros(v_coords.shape)
    
    for i in range(0,n):
        x = v_coords[i,0]
        y = v_coords[i,1]
        z = v_coords[i,2]
    
        coords[i,0] = x * T[0,0] + y * T[0,1] + z * T[0,2] + T[0,3]
        coords[i,1] = x * T[1,0] + y * T[1,1] + z * T[1,2] + T[1,3]
        coords[i,2] = x * T[2,0] + y * T[2,1] + z * T[2,2] + T[2,3]
    
    return coords

# Uses nilearn to smooth the given numpy-format volume. Returns a numpy array.
#
# volume:          3D numpy array representing a volume
# V_img:           Nifti1Image object specifying an affine xfm and header
# fwhm:            Full-width at half-max of the smoothing to be performed
#
# Returns:         The smoothed image
#

def smooth_volume(volume, V_img, fwhm):
    
    img = nib.Nifti1Image(volume, V_img.affine, V_img.header)
    V = image.smooth_img(img, fwhm)
    return V.get_data()
    

# Writes a polyline to file in ModelGUI poly3d format
#
# polyline:        Nx3 numpy array representing coordinates of the polyline vertices
# filename:        Full path to the file location at which to save this polyline
# name:            Name for the polyline
# data:            NxM numpy array of vertex data columns
#

def write_polyline_mgui(polyline, filename, name='', data=None, data_names=None):
    
    if len(name) == 0:
        name = filename
        
    M = 0
    
    if data is not None:
        M = 1
        if len(data.shape) > 1:
            M = data.shape[1]
        if M == 1:
            data_names = [data_names];
    
    with open(filename, 'w') as writer:
        N = polyline.shape[0]
        writer.write('1\n{0} 0 {1}'.format(N, name))
        for i in range(0,M):
            writer.write(' {0}'.format(data_names[i]))
        writer.write('\n');
        
        for i in range(0, N):
            writer.write('{0:1.6f} {1:1.6f} {2:1.6f}'.format(polyline[i,0], polyline[i,1], polyline[i,2]))
            if M == 1:
                writer.write(' {0:1.6f}'.format(data[i]))
            else:
                for j in range(0,M):
                    writer.write(' {0:1.6f}'.format(data[i,j]))
            writer.write('\n');                 
                
    
# Reads a polyline from a ModelGUI poly3d format file
#

def read_polyline_mgui(filename):
    
    polyline = None
    
    with open(filename, 'r') as reader:
        reader.readline()
        line = reader.readline().rstrip()
        parts = line.split()
        N = int(parts[0])
         
        polyline = np.zeros((N,3))
        i = 0
        line = reader.readline()
        while line:
            parts = line.rstrip().split()
            polyline[i,0] = float(parts[0]);
            polyline[i,1] = float(parts[1]);
            polyline[i,2] = float(parts[2]);
            i += 1
            line = reader.readline()
        
    return polyline
    

# Flips all vectors whose major axis (component with maximal magnitude) is negative
#
# V_vect:           XxYxZx3 numpy array of voxel-wise orientation vectors
#
# Returns:          The resulting XxYxZx3 image
#

def make_vectors_positive( V_vect ):
    
    if V_vect.ndim == 4:
        amax = np.amax(np.abs(V_vect),axis=3)
        idxmax = np.argmax(np.abs(V_vect),axis=3)
        idx = np.nonzero(amax)
        x = idxmax[idx]
        idx2 = idx + (x,)
        mxs = V_vect[idx2]
        isneg = np.less(mxs,0)
        idx3=np.transpose(idx)
        idx3 = idx3[isneg,:]

        V_vect[idx3[:,0],idx3[:,1],idx3[:,2],:]=-V_vect[idx3[:,0],idx3[:,1],idx3[:,2],:]

        return V_vect
    
    if V_vect.ndim == 2:
        amax = np.amax(np.abs(V_vect),axis=1)
        idxmax = np.argmax(np.abs(V_vect),axis=1)
        idx = np.nonzero(amax)
        x = idxmax[idx]
        idx2 = idx + (x,)
        mxs = V_vect[idx2]
        isneg = np.less(mxs,0)
        idx3=np.transpose(idx)
        idx3 = idx3[isneg,:]

        V_vect[idx3,:]=-V_vect[idx3,:]

        
        return V_vect
    
    assert False


# Detects disconnected blobs in a binary (thresholded) image, and labels each with
# integers 1,2,3...
#
# V_img:           A 3D image
# threshold:       A threshold used to binarize V_img; set to -Inf to prevent any
#                  thresholding (e.g., if your image is already binarized)
#
# Returns: 	       An integer-labelled 3D image
#

def label_blobs( V_img, threshold=0 ):
    
    V_tract = np.greater(V_img, threshold)
    V_labelled = np.zeros(V_tract.shape)
    XYZ_img = np.transpose(np.nonzero(V_tract))
    
    itr = 1

    while XYZ_img.size:
        XYZ_blob = np.empty((0,3), int)
        xyz = XYZ_img[0,:]
        XYZ_blob = np.append(XYZ_blob, np.array([[xyz[0], xyz[1], xyz[2]]]), axis=0)

        while XYZ_blob.shape[0]:
            xyz = XYZ_blob[0,:]
            XYZ_blob = np.delete(XYZ_blob, 0, 0)

            V_labelled[xyz[0], xyz[1], xyz[2]] = itr

            # Search all neighbours
            for i in range(max(1, xyz[0]-1), min(V_img.shape[0], xyz[0]+2)):
                for j in range(max(1, xyz[1]-1), min(V_img.shape[1], xyz[1]+2)):
                    for k in range(max(1, xyz[2]-1), min(V_img.shape[2], xyz[2]+2)):
                        if V_tract[i,j,k] and V_labelled[i,j,k]==0:
                            V_labelled[i,j,k] = itr
                            XYZ_blob = np.append(XYZ_blob, np.array([[i,j,k]]), axis=0)

        # Removed labelled voxels from tract
        V_tract = np.logical_and(V_tract, np.logical_not(V_labelled))
        XYZ_img = np.transpose(np.nonzero(V_tract))

        itr += 1
    
    return V_labelled

# Given a set of blobs (non-contiguous parts of a tract estimate), determines whether
# each is adjacent to the specified ROIs (usually a pair). 
#
# V_blobs:          An 3D volume containing blobs labelled as integers
# V_rois:           A list of 2D volume masks, one for each ROI
#
# Returns: 			A mask volume for which only the blobs adjacent to all ROIs are 
#                   retained.
#

def retain_adjacent_blobs(V_blobs, V_rois):

    V_mask = np.zeros(V_blobs.shape)
    V_tests = []
    
    for i in range(0, len(V_rois)):
        V = dilate_mask(V_rois[i], 3)
        V_tests.append(V)

    blobs = np.unique(V_blobs).astype(np.int32)
    blobs = blobs[1:len(blobs)]

    for blob in blobs:

        V_b = V_blobs == blob
        is_retained = True
        i=1
        for i in range(0, len(V_tests)):
            V_test = V_tests[i]
            V_wtf = np.logical_and(np.greater(V_test,0), V_b)
            overlap = np.flatnonzero(V_wtf)

            if not overlap.size:
                is_retained = False
                break
            i+=1

        if is_retained:
            V_mask = np.logical_or(V_mask, V_b)
        
    return np.multiply(V_mask, V_blobs)
    
    
# Reads b-values from a text file
#
# Returns:     An Nx1 array of bvals
#

def read_bvals( filename ):
    
    with open(filename, 'r') as csvin:
        line = csvin.readline()
        bvals = list(map(float, line.split()))
        return np.array(bvals)
    
    return None

# Reads b-vectors from a text file
#
# Returns:     An Nx3 array of bvecs
#

def read_bvecs( filename ):
    
    with open(filename, 'r') as csvin:
        
        line = csvin.readline()
        b = list(map(float, line.split()))
        N = len(b)
        bvecs = np.zeros([N,3])
        bvecs[:,0] = np.array(b)
        line = csvin.readline()
        b = list(map(float, line.split()))
        bvecs[:,1] = np.array(b)
        line = csvin.readline()
        b = list(map(float, line.split()))
        bvecs[:,2] = np.array(b)
        
        return bvecs
    
    return None

# Runs a (FSL) system command using the subprocess module
#
# cmd:      Command to execute
# dryrun:   For debugging, returns an empty string
#
# Returns:  An empty string if execution was successful, otherwise an error message
#


def run_fsl(cmd, dryrun=False):
    if dryrun:
        return ''
    cmd = '{0};{1}'.format('. /usr/local/fsl/etc/fslconf/fsl.sh', cmd)
    sp = Popen(cmd, shell=True, stderr=subprocess.PIPE)
    out, err = sp.communicate()
    return err
    
    
    
def write_matrix_to_pajek( M, path, weighted=True, thres_low=0, thres_high=float("inf"), \
                           directed=True, labels=None, coords=None, vertex_values=None):
    
    if len(M.shape) != 2 or M.shape[0] != M.shape[1]:
        raise Exception('Matrix must be square')
    
    with open( path, 'w' ) as fout:
            
        N = M.shape[0]
        
        # Write vertices
        fout.write('*vertices {0}'.format(N))
                   
        for i in range(0,N):
            fout.write('\n {0}'.format(i+1))
            if labels is not None:
                fout.write(' "{0}"'.format(labels[i]))
            else:
                fout.write(' "{0}"'.format(i+1))
                
            if coords is not None:
                fout.write(' {0:1.8f} {1:1.8f} {2:1.8f}'.format(coords[i,0],coords[i,1],coords[i,2]))
                
            if vertex_values is not None:
                fout.write(' x_fact {0} y_fact {0}'.format(vertex_values[i]))
                
        # Write edges
        if directed:
            fout.write('\n*arcs')
        else:
            fout.write('\n*edges')
               
        for i in range(0,N):
            start = 0;
            if directed:
                start = i
            for j in range(start,N):
                if i != j:
                    value = M[i,j]
                    if value > thres_low and value < thres_high:
                        fout.write('\n {0} {1}'.format(i+1, j+1))
                        if weighted:
                            fout.write(' {0:1.8f}'.format(value))

# Find clusters in tvals [Nx1], where abs(tvals) > thres,
# and labels them with integers. Separate clusters are created for positive
# and negative t-values
#
# min_clust:     The minimal width at which to accept a cluster
#
# Returns an array of integers [Nx1] as cluster labels

# def get_tvalue_clusters( tvals, thres, min_clust=0 ):

#     clusters = np.zeros(tvals.size, dtype=int)
#     i = 0;
#     c = 1;
#     in_cluster = False;

#     while i < tvals.size:
#         isclus = tvals[i] > thres
#         if in_cluster:
#             if isclus:
#                 clusters[i] = c
#             else:
#                 in_cluster = False
#                 if np.count_nonzero(clusters==c) < min_clust:
#                     clusters[clusters==c] = 0
#                 else:
#                     c += 1
#         else:
#             if isclus:
#                 clusters[i] = c
#                 in_cluster = True
#         i += 1

#     return clusters

# Aggregates all ROI volumes into a single volume file
#
# rois_dir:    Directory containing image files
# rois:        List of ROIs to aggregate
# output_file: Path to aggregate output volume
# suffix:      Suffix for volumes (default='nii', can also be 'nii.gz')
# verbose:     Whether to print messages to the terminal
#
def aggregate_rois( rois_dir, rois, output_file, suffix='nii', verbose=False ):
    
    img_files = []

    for roi in rois:
        img_files.append('{0}/{1}.{2}'.format(rois_dir, roi, suffix))

    V_img = nib.load(img_files[0])
    V_agg = np.zeros(V.shape)

    for file in img_files:
        V = nib.load(file)
        V_agg = V_agg + V.get_fdata()
        if verbose:
            print('Added {0}'.format(os.path.basename(file)))

    V_agg = np.where(V_agg>0,1,0)
    img = nib.Nifti1Image(V_agg, V_img.affine, V_img.header)
    nib.save(img,output_file)
    if verbose:
        print('Output saved to {0}'.format(output_file))
        
        
# Compute the center of gravity for each ROI in the list, and write to
# a CSV file
#
# rois_dir:          Directory containing the ROI images
# rois:              List of ROIs to compute centers for
# output_file:       Output CSV file (can be None)
# extension:         File extension for volume files (default='nii')
# verbose:           Whether to print progress to screen
#
# return:            numpy array (Nx3) of ROI center points
#
def compute_roi_centers( rois_dir, rois, output_file, extension='nii', verbose=False ):
    
    img_files = []

    for roi in rois:
        img_files.append('{0}/{1}.{2}'.format(rois_dir, roi, extension))
        
    V = nib.load(img_files[0])
    roi_centers = np.zeros((len(img_files),3))
    
    for i, file in zip(range(0,len(img_files)), img_files):
        V = nib.load(file)
        M = V.affine
        nz = np.nonzero(V.get_fdata())
        C = np.array([sum(nz[0])/len(nz[0]),sum(nz[1])/len(nz[1]),sum(nz[2])/len(nz[2]),1])
        pt = np.matmul(M,C.T)[0:3]
        roi_centers[i,:] = pt
#         fn = os.path.basename(file)
#         if extension == 'nii.gz':
#             roi = fn[0:-7]
#         else:
#             roi = fn[0:-4]
        if verbose:
            print('{0}: {1}'.format(rois[i], roi_centers[i,:]))
    
    if not output_file is None:
        with open(output_file,'w+') as outfile:
            for i in range(0,len(img_files)):
                pt = roi_centers[i,:]
                outfile.write('{0},{1:1.4f},{2:1.4f},{3:1.4f}\n'.format(rois[i], pt[0], pt[1], pt[2]))           
            
    return roi_centers

    
    
# Identify and label clusters in a set of t-values and associated p-values
#
# pvals         N x 1 vector of p-values
# tvals         N x 1 vector of t-values
# alpha         cluster-forming threshold
# min_clust     the minimal cluster size [default=1]
#
def get_clusters( pvals, tvals, alpha, min_clust=1 ):

#     print('Min cluster size: {0}'.format(min_clust))
    N_nodes = tvals.size

    clusters = np.zeros(N_nodes, dtype=int)
    c = 1
    in_clust = False
    is_pos = False
    csize = 0

    for i in range(0, N_nodes):
        if pvals[i] < alpha:
            if tvals[i] > 0:
                if in_clust and not is_pos:
                    # Switched sign within cluster;
                    # requires new cluster
                    if csize < min_clust:
                        # Do not keep if less than min_clust
                        clusters[clusters==c] = 0
                    else:
#                         print(1, c, csize, clusters)
                        c += 1
                    csize = 0

                clusters[i] = c
                in_clust = True
                is_pos = True
                csize += 1
            elif tvals[i] < 0:
                if in_clust and is_pos:
                    # Switched sign within cluster;
                    # requires new cluster
                    if csize < min_clust:
                        # Do not keep if less than min_clust
                        clusters[clusters==c] = 0
                    else:
#                         print(2, c, csize, clusters)
                        c += 1
                    csize = 0

                clusters[i] = c
                in_clust = True
                is_pos = False
                csize += 1
        else:
            if in_clust:
                if csize < min_clust:
                    clusters[clusters==c] = 0
                else:
#                     print(3, c, csize, min_clust, clusters)
                    c += 1
                csize = 0
            in_clust = False
          
    return clusters

# Identify and label clusters in a set of t-values using 1-dimensional
# random field theory
def get_tvalue_rft1d_clusters( tvals, alpha, df, fwhm, min_clust ):

    tvals_abs = np.abs(tvals)
    t_max = np.max(tvals_abs)
    N_nodes = tvals.size
    pvals = np.ones(N_nodes)
    clusters = np.zeros(N_nodes)

    t_star = rft1d.t.isf(alpha, df, N_nodes, fwhm)
    if t_star < t_max:
        # Significant main (tract-wise) effect of this factor
        calc = rft1d.geom.ClusterMetricCalculator()
        k_nodes = calc.cluster_extents(tvals_abs, t_star, interp=True)
        k_resels = [kk/fwhm for kk in k_nodes]
        c = len(k_resels)
        rftcalc = rft1d.prob.RFTCalculator(STAT='T', df=(1,df), nodes=N_nodes, FWHM=fwhm)
        k_min = min(k_resels)
        P_set = rftcalc.p.set(c, k_min, t_star)
        P_cluster = [rftcalc.p.cluster(kk, t_star) for kk in k_resels]
        if len(P_cluster) == 1:
            if np.count_nonzero(tvals_abs>t_star) >= min_clust:
                pvals[tvals_abs>t_star] = P_cluster
                clusters = np.zeros(N_nodes)
                clusters[tvals_abs>t_star] = 1
        else:
            clusters = get_clusters(-tvals_abs, tvals, -t_star, min_clust)
            cvals = np.unique(clusters[clusters>0])
            for c in cvals:
                pvals[clusters==c] = P_cluster[c-1]

    return pvals, clusters


# Generates a matrix of t-values [N_perm X N_dist] from permutations of V_dist,
# summarizing the t-values at each distance. Can be used to generate a null
# distribution of distance-wise t-values.
# 
# Parameters:
#
#  dists             1D (N_dist) array of distances to process
#  V_dist            3D (Nx X Ny X Nz) tract distance image
#  V_tval            3D (Nx X Ny X Nz) t-statistic image
#  N_perm            Number of permutations
#  metric            Metric which summarizes t-statistics at a given
#                    distance
#  V_pval            Optional, returns p-values for the permutations.
# 
# Returns:
#
#  A N_perm X N_dist matrix of permuted t-values

def get_permuted_tvals( dists, V_dist, V_tval, N_perm, metric='max', V_pval=None ):

    N_dist = dists.size
    t_perm = np.zeros((N_perm, N_dist))
    idx_tract = V_dist > 0
    V_dist = V_dist[idx_tract].flatten()
    V_tval = V_tval[idx_tract].flatten()
    V_pval = V_pval[idx_tract].flatten()
    V_tval_abs = np.abs(V_tval)
    if V_pval is not None:
        p_perm = np.zeros((N_perm, N_dist))
    else:
        p_perm = None

    for p in range(0, N_perm):
        idx_p = np.random.permutation(V_dist.size)
        V_p = V_dist[idx_p]

        for d, i in zip(dists, range(0,dists.size)):
            # Summarize stat across voxels at this distance (mean, max, median)
            idx_d = np.flatnonzero(V_p==d)

            summary_tval = 0
            if p_perm is not None:
                summary_pval = 1
            if idx_d.size > 0:
                if metric == 'mean':
                    summary_tval = np.mean(V_tval[idx_d])
                    if p_perm is not None:
                        summary_pval = np.mean(V_pval[idx_d])

                elif metric == 'max':
                    idx_max = idx_d[np.argmax(V_tval_abs[idx_d])]
                    summary_tval = V_tval[idx_max]
                    if p_perm is not None:
                        summary_pval = V_pval[idx_max]

                elif metric == 'median':
                    summary_tval = np.median(V_tval[idx_d])
                    if p_perm is not None:
                        summary_pval = np.median(V_pval[idx_d])

                else:
                    print('Error: "{0}" is not a valid summary metric'.format(metric))
                    assert(False)
                    
            t_perm[p,i] = summary_tval
            if p_perm is not None:
                p_perm[p,i] = summary_pval

    return t_perm, p_perm
