import weightedCDFs as wcdfs
from sklearn.cluster import KMeans
import numpy as np
import scipy


def distributeWeights(init_samples, bins_of_samples, w_bin, w_init=None):

    n_bins = len(w_bin)
    w = np.empty(len(init_samples))

    if w_init is None:
        w_init = np.ones(len(init_samples))

    for i in range(n_bins):
        bin_inds = (bins_of_samples == i)
        n_i = np.sum(bin_inds)
        # w[bin_inds] = w_bin[i] / n_i if n_i != 0 else 0
        w[bin_inds] = (w_bin[i] / n_bins) / np.sum(w_init[bin_inds]) if n_i != 0 else 0

    return w


def histogram_dci(init_samples, pred_samples, bbox=None,
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

    else:
        if dim_D == 1:
            min_pred = np.min(pred_samples[:,0])
            max_pred = np.max(pred_samples[:,0])

    # create bins in each dimension
    low_ends = np.empty((np.prod(n_bins), dim_D))
    upp_ends = np.empty((np.prod(n_bins), dim_D))
    if dim_D == 1:
        d_len = (max_pred - min_pred) / n_bins[0]
        low_ends[:,0] = np.linspace(min_pred, max_pred-d_len, n_bins[0])
        upp_ends[:,0] = np.linspace(min_pred+d_len, max_pred, n_bins[0])

    # ok for now this is only going to work in 1 or 2 dimensions because it's fucking recursive again!!!!!!
    # bin samples
    labels = np.empty(n_samples)
    ratios = []
    in_bins_pred = []
    in_bins_obs = []
    n_obs = len(sample_set_2)
    if dim_D == 1:
        counter = 0
        for i in range(n_bins[0]):
            bin_inds = ((pred_samples[:,0] >= low_ends[i,0]) & (pred_samples[:,0] <= upp_ends[i,0]))
            labels[bin_inds] = counter
            ratios.append(np.sum((sample_set_2[:,0] >= low_ends[i,0]) & (sample_set_2[:,0] <= upp_ends[i,0]))/n_obs)
            in_bins_pred.append((np.sum(((pred_samples[:,0] >= low_ends[i,0]) & (pred_samples[:,0] <= upp_ends[i,0])))))
            in_bins_obs.append((np.sum(((sample_set_2[:,0] >= low_ends[i,0]) & (sample_set_2[:,0] <= upp_ends[i,0])))))
            counter += 1

    ratios = np.array(ratios)
    w = weights.distributeWeights(init_samples, labels, ratios)
    len_bin = upp_ends[0,0] - low_ends[0,0]
    return w, ratios, labels, len_bin
    

def computePartitionedWeights_kMeans_iterative(init_samples, pred_samples,
                                               sample_set_2=None, targ_CDF=None,
                                               n_clusters=None):

    n_samples = len(pred_samples)

    # if n_clusters is not input, then use n_clusters = n_samples / 100
    if n_clusters is None:
        n_clusters = n_samples / 100

    w = np.ones(n_samples)
    p_w = np.ones(n_samples)
    for i in range(np.shape(sample_set_2)[1]):
        p_w, _, _, _ = computePartitionedWeights_kMeans_IID(init_samples,
                                                      np.reshape(pred_samples[:,i], (n_samples,1)),
                                                      w_init=p_w,
                                                      sample_set_2=np.reshape(sample_set_2[:,i], (len(sample_set_2),1)),
                                                      n_clusters=n_clusters)

        w *= p_w
    return w


def computePartitionedWeights_regulargrid_iterative(init_samples, pred_samples,
                                                    sample_set_2=None, targ_CDF=None,
                                                    n_clusters=None):

    n_samples = len(pred_samples)

    # if n_clusters is not input, then use n_clusters = n_samples / 100
    if n_clusters is None:
        n_clusters = n_samples / 100

    w = np.ones(n_samples)
    p_w = np.ones(n_samples)
    for i in range(np.shape(sample_set_2)[1]):
        p_w, _, _, _ = computePartitionedWeights_kMeans_IID(init_samples,
                                                      np.reshape(pred_samples[:,i], (n_samples,1)),
                                                      w_init=p_w,
                                                      sample_set_2=np.reshape(sample_set_2[:,i], (len(sample_set_2),1)),
                                                      n_clusters=n_clusters)

        w *= p_w
    return w


def computePartitionedWeights_regulargrid_nonIID(init_samples, pred_samples, des_samples, sample_set_2=None,
                                                 targ_CDF=None, n_bins=None, init_CDF=None):

    n_samples = len(pred_samples)
    m_samples = len(des_samples)
    dim_D = np.shape(pred_samples)[1]

    # if n_clusters is not input, then use n_clusters = n_samples / 100
    if n_bins is None:
        n_bins = n_samples / 100

    # create bins in each dimension
    low_ends = np.empty((n_bins, dim_D))
    upp_ends = np.empty((n_bins, dim_D))
    centers = np.empty((n_bins, dim_D))
    for d in range(dim_D):
        min_pred = np.min(pred_samples[:,d])
        max_pred = np.max(pred_samples[:,d])
        d_len = (max_pred - min_pred) / n_bins
        low_ends[:,d] = np.linspace(min_pred, max_pred-d_len, n_bins)
        upp_ends[:,d] = np.linspace(min_pred+d_len, max_pred, n_bins)
        centers[:,d] = np.linspace(min_pred+d_len/2, max_pred-d_len/2, n_bins)

    # ok for now this is only going to work in 1 or 2 dimensions because it's fucking recursive again!!!!!!
    # bin samples
    labels = np.empty(n_samples)
    if dim_D == 1:
        for i in range(n_bins):
            bin_inds = ((pred_samples[:,0] >= low_ends[i,0]) & (pred_samples[:,0] <= upp_ends[i,0]))
            labels[bin_inds] = i
    if dim_D == 2:
        counter = 0
        for i in range(n_bins):
            for j in range(n_bins):
                bin_inds = ((pred_samples[:,0] >= low_ends[i,0])
                            & (pred_samples[:,0] <= upp_ends[i,0])
                            & (pred_samples[:,1] >= low_ends[i,1])
                            & (pred_samples[:,1] <= upp_ends[i,1]))
                labels[bin_inds] = counter
                counter += 1

    # print('computing cluster weights')
    # compute weights for clusters
    H_bin = wcdfs.compute_H(centers)
    if targ_CDF is not None:
        b_bin = wcdfs.compute_b(centers, targ_CDF=targ_CDF)
    elif sample_set_2 is not None:
        b_bin = wcdfs.compute_b(centers, sample_set_2=sample_set_2)
    w_bin = wcdfs.compute_optimal_w(H_bin, b_bin)

    # print('computing initial weights')
    # compute weights on input
    H_init = wcdfs.compute_H(init_samples)
    # b_init = wcdfs.compute_b(init_samples, targ_CDF=init_CDF)
    b_init = wcdfs.compute_b(init_samples, sample_set_2=des_samples)
    w_init = wcdfs.compute_optimal_w(H_init, b_init)

    # distribute weights in clusters
    w = distributeWeights(init_samples, labels, w_bin, w_init=w_init)

    return w * w_init[:,0], labels, w_init[:,0], w


def computePartitionedWeights_kMeans_nonIID(init_samples, pred_samples, des_samples,
                                            sample_set_2=None, targ_CDF=None, n_clusters=None):

    n_samples = len(pred_samples)
    m_samples = len(des_samples)

    # if n_clusters is not input, then use n_clusters = n_samples / 100
    if n_clusters is None:
        n_clusters = n_samples / 100

    # cluster weights using kMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(pred_samples)
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_

    # compute weights for clusters
    H_cluster = wcdfs.compute_H(centers)
    if targ_CDF is not None:
        b_cluster = wcdfs.compute_b(centers, targ_CDF=targ_CDF)
    elif sample_set_2 is not None:
        b_cluster = wcdfs.compute_b(centers, sample_set_2=des_samples)
    w_cluster = wcdfs.compute_optimal_w(H_cluster, b_cluster)

    # compute weights on input
    H_init = wcdfs.compute_H(init_samples)
    b_init = wcdfs.compute_b(init_samples, sample_set_2=des_samples)
    w_init = wcdfs.compute_optimal_w(H_init, b_init)

    # distribute weights in clusters
    w = distributeWeights(init_samples, labels, w_cluster, w_init=w_init)

    return w * w_init[:,0], labels, w_init[:,0], w


def computePartitionedWeights_kMeans_IID(init_samples, pred_samples, w_init=None,
                                         sample_set_2=None, targ_CDF=None, n_clusters=None):

    n_samples = len(pred_samples)

    # if n_clusters is not input, then use n_clusters = n_samples / 100
    if n_clusters is None:
        n_clusters = n_samples / 100

    # cluster weights using kMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(pred_samples)
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_

    # compute weights for clusters
    H_cluster = wcdfs.compute_H(centers)
    if targ_CDF is not None:
        b_cluster = wcdfs.compute_b(centers, targ_CDF=targ_CDF)
    elif sample_set_2 is not None:
        b_cluster = wcdfs.compute_b(centers, sample_set_2=sample_set_2)
    w_cluster = wcdfs.compute_optimal_w(H_cluster, b_cluster)
    
    # distribute weights evenly in clusters
    w = distributeWeights(init_samples, labels, w_cluster, w_init=w_init)

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

    # ok for now this is only going to work in 1 or 2 dimensions because it's fucking recursive again!!!!!!
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
        print('ERROR')
    
    centers = np.array(centers)

    # compute weights for clusters
    # H_bin = wcdfs.compute_H(centers)
    H_bin = wcdfs.compute_H(upp_ends[:,:])
    if targ_CDF is not None:
        b_bin = wcdfs.compute_b(centers, targ_CDF=targ_CDF)
    elif sample_set_2 is not None:
        b_bin = wcdfs.compute_b(upp_ends[:,:], sample_set_2=sample_set_2)
        # b_bin = wcdfs.compute_b(centers, sample_set_2=sample_set_2)
    w_bin = wcdfs.compute_optimal_w(H_bin, b_bin)
    
    # distribute weights evenly in clusters
    w = distributeWeights(init_samples, labels, w_bin)

    return w, labels, centers, w_bin