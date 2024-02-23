import numpy as np
import scipy
import src.weightedEDFs as wedfs

from sklearn.cluster import KMeans


'''
This module constructs the methods to compute the solutions to DCI problems using
the binning method described in the Distributions-based DCI paper as stated in the README.

The goal is to use this approach to solve Data-Consistent Inversion (DCI) problems as presented in

  Combining Push-Forward Measures and Bayes' Rule to Construct Consistent Solutions to Stochastic Inverse Problems, 
  T. Butler, J. Jakeman, T. Wildey, SIAM J. Sci. Comput., 40(2), A984–A1011 (2018)
  
  URL: https://epubs.siam.org/doi/abs/10.1137/16M1087229

The examples in the examples directory of this repository contain demonstrations of applications of the method.
'''


def distributeWeights(init_samples, bins_of_samples, w_bin):
    '''
    This function distributes a computed set of weights for a set of bins among the samples
    in the bins.
    
    :param init_samples: a numpy array of dimension n-by-d (n=# samples, d=dimension of samples)
    :param init_samples: a numpy array of n containing integers corresponding to the bin IDs that
                         the samples fall into
    :param w_bin: a numpy array of n containing floats corresponding to the weights of the bins
    :rtype: numpy array
    :returns: n-by-n array H
    '''

    n_bins = len(w_bin)
    w = np.empty(len(init_samples))

    for i in range(n_bins):
        bin_inds = (bins_of_samples == i)
        n_i = np.sum(bin_inds)
        w[bin_inds] = (w_bin[i] / n_bins) / n_i if n_i != 0 else 0

    return w


def computePartitionedWeights_kMeans_IID(init_samples, pred_samples, sample_set_2=None,
                                         targ_CDF=None, n_clusters=None):
    '''
    This function distributes a computed set of weights for a set of bins among the samples
    in the bins.
    
    :param init_samples: a numpy array of dimension n-by-d (n=# samples, d=dimension of samples)
    :param pred_samples: a numpy array of n containing integers corresponding to the bin IDs that
                         the samples fall into
    :param w_bin: a numpy array of n containing floats corresponding to the weights of the bins
    :rtype: numpy array
    :returns: n-by-n array H
    '''

    n_samples = len(pred_samples)

    # if n_clusters is not input, then use n_clusters = n_samples / 100
    if n_clusters is None:
        n_clusters = n_samples / 100

    # cluster weights using kMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(pred_samples)
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_

    # compute weights for clusters
    H_cluster = wedfs.compute_H(centers)
    if targ_CDF is not None:
        b_cluster = wedfs.compute_b(centers, targ_CDF=targ_CDF)
    elif sample_set_2 is not None:
        b_cluster = wedfs.compute_b(centers, sample_set_2=sample_set_2)
    w_cluster = wedfs.compute_optimal_w(H_cluster, b_cluster)
    
    # distribute weights evenly in clusters
    w = distributeWeights(init_samples, labels, w_cluster)

    return w, labels, centers, w_cluster


def computePartitionedWeights_regulargrid_IID(init_samples, pred_samples, bbox=None,
                                              sample_set_2=None, targ_CDF=None, n_bins=None,
                                              remove_empty_bins=True):

    n_samples = len(pred_samples)
    dim_D = np.shape(pred_samples)[1]

    # if n_clusters is not input, then use n_clusters = n_samples / 100
    if n_bins is None: # idk this isn't going to work well in multiple dimensions
        n_bins = (n_samples / 100)
    if isinstance(n_bins, int):
        n_bins = [n_bins] * dim_D
        
    if bbox is not None:
        if dim_D == 1:
            min_pred = bbox[0]
            max_pred = bbox[1]
        if dim_D == 2:
            min_pred_D1 = bbox[0][0]
            max_pred_D1 = bbox[0][1]
            min_pred_D2 = bbox[1][0]
            max_pred_D2 = bbox[1][1]
        if dim_D == 2:
            min_pred_D1 = bbox[0][0]
            max_pred_D1 = bbox[0][1]
            min_pred_D2 = bbox[1][0]
            max_pred_D2 = bbox[1][1]
            min_pred_D3 = bbox[2][0]
            max_pred_D3 = bbox[2][1]
    else:
        if dim_D == 1:
            min_pred = np.min(pred_samples[:,0])
            max_pred = np.max(pred_samples[:,0])
        if dim_D == 2:
            min_pred_D1 = np.min(pred_samples[:,0])
            max_pred_D1 = np.max(pred_samples[:,0])
            min_pred_D2 = np.min(pred_samples[:,1])
            max_pred_D2 = np.max(pred_samples[:,1])
        if dim_D == 3:
            min_pred_D1 = np.min(pred_samples[:,0])
            max_pred_D1 = np.max(pred_samples[:,0])
            min_pred_D2 = np.min(pred_samples[:,1])
            max_pred_D2 = np.max(pred_samples[:,1])
            min_pred_D3 = np.min(pred_samples[:,2])
            max_pred_D3 = np.max(pred_samples[:,2])

    # create bins in each dimension
    low_ends = np.empty((np.prod(n_bins), dim_D))
    upp_ends = np.empty((np.prod(n_bins), dim_D))
    if dim_D == 1:
        d_len = (max_pred - min_pred) / n_bins[0]
        low_ends[:,0] = np.linspace(min_pred, max_pred-d_len, n_bins[0])
        upp_ends[:,0] = np.linspace(min_pred+d_len, max_pred, n_bins[0])
    if dim_D == 2:
        d_len_D1 = (max_pred_D1 - min_pred_D1) / n_bins[0]
        d_len_D2 = (max_pred_D2 - min_pred_D2) / n_bins[1]
        low_ends_D1 = np.linspace(min_pred_D1, max_pred_D1-d_len_D1, n_bins[0])
        upp_ends_D1 = np.linspace(min_pred_D1+d_len_D1, max_pred_D1, n_bins[0])
        low_ends_D2 = np.linspace(min_pred_D2, max_pred_D2-d_len_D2, n_bins[1])
        upp_ends_D2 = np.linspace(min_pred_D2+d_len_D2, max_pred_D2, n_bins[1])
        for i in range(n_bins[0]):
            for j in range(n_bins[1]):
                low_ends[i*n_bins[1]+j, 0] = low_ends_D1[i]
                low_ends[i*n_bins[1]+j, 1] = low_ends_D2[j]
                upp_ends[i*n_bins[1]+j, 0] = upp_ends_D1[i]
                upp_ends[i*n_bins[1]+j, 1] = upp_ends_D2[j]
    if dim_D == 3:
        d_len_D1 = (max_pred_D1 - min_pred_D1) / n_bins[0]
        d_len_D2 = (max_pred_D2 - min_pred_D2) / n_bins[1]
        d_len_D3 = (max_pred_D3 - min_pred_D3) / n_bins[2]
        low_ends_D1 = np.linspace(min_pred_D1, max_pred_D1-d_len_D1, n_bins[0])
        upp_ends_D1 = np.linspace(min_pred_D1+d_len_D1, max_pred_D1, n_bins[0])
        low_ends_D2 = np.linspace(min_pred_D2, max_pred_D2-d_len_D2, n_bins[1])
        upp_ends_D2 = np.linspace(min_pred_D2+d_len_D2, max_pred_D2, n_bins[1])
        low_ends_D3 = np.linspace(min_pred_D3, max_pred_D3-d_len_D3, n_bins[2])
        upp_ends_D3 = np.linspace(min_pred_D3+d_len_D3, max_pred_D3, n_bins[2])
        for i in range(n_bins[0]):
            for j in range(n_bins[1]):
                for k in range(n_bins[2]):
                    low_ends[i*n_bins[1]*n_bins[2]+j*n_bins[2]+k, 0] = low_ends_D1[i]
                    upp_ends[i*n_bins[1]*n_bins[2]+j*n_bins[2]+k, 0] = upp_ends_D1[i]
                    low_ends[i*n_bins[1]*n_bins[2]+j*n_bins[2]+k, 1] = low_ends_D2[j]
                    upp_ends[i*n_bins[1]*n_bins[2]+j*n_bins[2]+k, 1] = upp_ends_D2[j]
                    low_ends[i*n_bins[1]*n_bins[2]+j*n_bins[2]+k, 2] = low_ends_D3[k]
                    upp_ends[i*n_bins[1]*n_bins[2]+j*n_bins[2]+k, 2] = upp_ends_D3[k]

    # bin samples
    labels = np.empty(n_samples)
    centers = []
    if dim_D == 1:
        counter = 0
        for i in range(n_bins[0]):
            bin_inds = ((pred_samples[:,0] >= low_ends[i,0]) & (pred_samples[:,0] <= upp_ends[i,0]))
            labels[bin_inds] = counter
            if remove_empty_bins and np.sum(bin_inds) != 0:
                centers.append([(upp_ends[i,0] - low_ends[i,0]) / 2 + low_ends[i,0]])
                counter += 1
            else:
                centers.append([(upp_ends[i,0] - low_ends[i,0]) / 2 + low_ends[i,0]])
                counter += 1
    if dim_D == 2:
        counter = 0
        for i in range(n_bins[0]):
            for j in range(n_bins[1]):
                bin_inds = ((pred_samples[:,0] >= low_ends[i*n_bins[1]+j,0])
                            & (pred_samples[:,0] <= upp_ends[i*n_bins[1]+j,0])
                            & (pred_samples[:,1] >= low_ends[i*n_bins[1]+j,1])
                            & (pred_samples[:,1] <= upp_ends[i*n_bins[1]+j,1]))
                labels[bin_inds] = counter
                if remove_empty_bins and np.sum(bin_inds) != 0:
                    centers.append([(upp_ends[i*n_bins[1]+j,0] - low_ends[i*n_bins[1]+j,0]) / 2  + low_ends[i*n_bins[1]+j,0],
                                    (upp_ends[i*n_bins[1]+j,1] - low_ends[i*n_bins[1]+j,1]) / 2 + low_ends[i*n_bins[1]+j,1]])
                    counter += 1
                else:
                    centers.append([(upp_ends[i*n_bins[1]+j,0] - low_ends[i*n_bins[1]+j,0]) / 2  + low_ends[i*n_bins[1]+j,0],
                                    (upp_ends[i*n_bins[1]+j,1] - low_ends[i*n_bins[1]+j,1]) / 2 + low_ends[i*n_bins[1]+j,1]])
                    counter += 1
    if dim_D == 3:
        counter = 0
        for i in range(n_bins[0]):
            for j in range(n_bins[1]):
                for k in range(n_bins[2]):
                    bin_inds = ((pred_samples[:,0] >= low_ends[i*n_bins[1]*n_bins[2]+j*n_bins[2]+k,0])
                                & (pred_samples[:,0] <= upp_ends[i*n_bins[1]*n_bins[2]+j*n_bins[2]+k,0])
                                & (pred_samples[:,1] >= low_ends[i*n_bins[1]*n_bins[2]+j*n_bins[2]+k,1])
                                & (pred_samples[:,1] <= upp_ends[i*n_bins[1]*n_bins[2]+j*n_bins[2]+k,1])
                                & (pred_samples[:,2] >= low_ends[i*n_bins[1]*n_bins[2]+j*n_bins[2]+k,2])
                                & (pred_samples[:,2] <= upp_ends[i*n_bins[1]*n_bins[2]+j*n_bins[2]+k,2]))
                    labels[bin_inds] = counter
                    if remove_empty_bins and np.sum(bin_inds) != 0:
                        centers.append([(upp_ends[i*n_bins[1]*n_bins[2]+j*n_bins[2]+k,0] - low_ends[i*n_bins[1]*n_bins[2]+j*n_bins[2]+k,0]) / 2 + low_ends[i*n_bins[1]*n_bins[2]+j*n_bins[2]+k,0],
                                        (upp_ends[i*n_bins[1]*n_bins[2]+j*n_bins[2]+k,1] - low_ends[i*n_bins[1]*n_bins[2]+j*n_bins[2]+k,1]) / 2 + low_ends[i*n_bins[1]*n_bins[2]+j*n_bins[2]+k,1],
                                        (upp_ends[i*n_bins[1]*n_bins[2]+j*n_bins[2]+k,2] - low_ends[i*n_bins[1]*n_bins[2]+j*n_bins[2]+k,2]) / 2 + low_ends[i*n_bins[1]*n_bins[2]+j*n_bins[2]+k,2]])
                        counter += 1
                    else:
                        centers.append([(upp_ends[i*n_bins[1]*n_bins[2]+j*n_bins[2]+k,0] - low_ends[i*n_bins[1]*n_bins[2]+j*n_bins[2]+k,0]) / 2 + low_ends[i*n_bins[1]*n_bins[2]+j*n_bins[2]+k,0],
                                        (upp_ends[i*n_bins[1]*n_bins[2]+j*n_bins[2]+k,1] - low_ends[i*n_bins[1]*n_bins[2]+j*n_bins[2]+k,1]) / 2 + low_ends[i*n_bins[1]*n_bins[2]+j*n_bins[2]+k,1],
                                        (upp_ends[i*n_bins[1]*n_bins[2]+j*n_bins[2]+k,2] - low_ends[i*n_bins[1]*n_bins[2]+j*n_bins[2]+k,2]) / 2 + low_ends[i*n_bins[1]*n_bins[2]+j*n_bins[2]+k,2]])
                        counter += 1
                        
    if dim_D > 3:
        print('ERROR: regular grid only works when the output dimesnion is less than or equal to 3')
    
    centers = np.array(centers)

    # compute weights for clusters
    H_bin = wedfs.compute_H(centers)
    if targ_CDF is not None:
        b_bin = wedfs.compute_b(centers, targ_CDF=targ_CDF)
    elif sample_set_2 is not None:
        b_bin = wedfs.compute_b(centers, sample_set_2=sample_set_2)
    w_bin = wedfs.compute_optimal_w(H_bin, b_bin)
    
    # distribute weights evenly in clusters
    w = distributeWeights(init_samples, labels, w_bin)

    return w, labels, centers, w_bin
