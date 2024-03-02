import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import random
from scipy.stats import gaussian_kde as GKDE
from scipy.stats import multivariate_normal
from scipy.stats import norm
import scipy.integrate as integrate
import seaborn as sns

import src.weightedEDFs as wEDFs
import src.binning as binning


def u_k(k, x, t, l, kappa):

    return (2 * l**2 * (-1)**(k+1) / (np.pi * k)
            * np.sin(k * np.pi * x / l)
            * np.exp(-kappa * (k * np.pi)**2 * t) / l**2)


def u(N, x, t, l, kappa):

    u_N = 0 * x
    for k in range(1, N):
        u_N += u_k(k, x, t, l, kappa)

    return u_N


def rejection_sampling(r):

    unifs = np.random.uniform(0,1,len(r))
    M = np.max(r)

    return (unifs < (r / M))


def obs_sample_discrete(n, n_data_pieces, data_splits, obs_dens_pieces):

    where = np.random.uniform(0, 1, n)
    samples = np.zeros(np.shape(where))
    ocm = np.append(0, np.cumsum([obs_dens_pieces[i] * (data_splits[i+1] - data_splits[i]) for i in range(n_data_pieces)]))

    for i in range(0, n_data_pieces):
        samples[(where >= ocm[i]) & (where < ocm[i+1])] = np.random.uniform(data_splits[i], data_splits[i+1], np.sum([(where >= ocm[i]) & (where < ocm[i+1])]))

    return samples


def main():

    # set plotting parameters ==============================================
    plot_directory = './plots'

    if not os.path.exists(plot_directory):
        os.makedirs(plot_directory)

    mpl.rcParams['lines.linewidth'] = 4
    plt.rc('font', size=14)
    plt.rc('axes', titlesize=16)
    plt.rc('axes', labelsize=16)
    mpl.rcParams['lines.markersize'] = 5
    mpl.rcParams['figure.figsize'] = (5.5, 4)
    mpl.rcParams['lines.linewidth'] = 2.5

    CB_color_cycle = ('#377eb8', '#ff7f00', '#4daf4a',
                      '#f781bf', '#a65628', '#984ea3',
                      '#999999', '#e41a1c', '#dede00')
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=CB_color_cycle)
    # ======================================================================


    random.seed(20) # set random seet

    # average value of parameters
    l = 2.
    kappa = 1.

    # parameter ranges
    delta_l = 0.1
    delta_kappa = 0.5

    N = 100 # solution truncation

    # measurement location in time and space
    t = .01
    sensor_loc = 1.2

    # initial distribution
    n_init_samples = 2000
    init_samples = np.random.uniform(0, 1, (n_init_samples,2))
    init_samples[:,0] = init_samples[:,0] * delta_l + l - delta_l / 2
    init_samples[:,1] = init_samples[:,1] * delta_kappa + kappa - delta_kappa / 2

    # push initial samples through QoI map to generate predicted samples
    pred_samples = np.zeros((n_init_samples, 1))
    pred_samples[:, 0] = u(N, sensor_loc, t, init_samples[:,0], init_samples[:,1])

    # generate observed samples from a distribution
    n_obs_samples = 10000
    obs_dist = norm(0.595, 3e-3)
    obs_samples = obs_dist.rvs(n_obs_samples)

    # for plotting
    X = np.linspace(l - delta_l / 2, delta_l + l - delta_l / 2, 100)
    Y = np.linspace(kappa - delta_kappa / 2, delta_kappa + kappa - delta_kappa / 2, 100)
    XX, YY = np.meshgrid(X, Y)
    ZZ = np.zeros(np.shape(XX))


    # contour plot ===========================================================
    for count, x in enumerate(X):
        ZZ[count,:] = u(N, sensor_loc, t, XX[count,:], YY[count,:])

    plt.figure()
    plt.contourf(XX, YY, ZZ, levels=9)
    ax = plt.gca()
    ax.set_aspect(0.18)

    plt.ylabel(r'$\kappa$')
    plt.xlabel(r'$\ell$')
    plt.xticks(ticks=np.linspace(np.min(XX), np.max(XX), 6),
               labels=["{:.2f}".format(x) for x in np.linspace(np.min(XX), np.max(XX), 6)])
    plt.yticks(ticks=np.linspace(np.min(YY), np.max(YY), 6),
               labels=["{:.2f}".format(x) for x in np.linspace(np.min(YY), np.max(YY), 6)])

    cbar = plt.colorbar()
    cbar.set_ticks(np.linspace(np.min(ZZ), np.max(ZZ), 7))
    cbar.set_ticklabels(["{:.4f}".format(x) for x in np.linspace(np.min(ZZ), np.max(ZZ), 7)])

    plt.tight_layout()
    plt.savefig(f'{plot_directory}/contours.png', bbox_inches='tight')
    # ========================================================================


    # density-based method computation
    pred_KDE = GKDE(pred_samples[:,0])
    obs_KDE = GKDE(obs_samples)

    # an example of a bad observed distribution, where the density-based solution fails
    bad_obs_dist = norm(0.613, 7e-3)
    bad_obs_samples = bad_obs_dist.rvs(n_obs_samples)

    bad_obs_KDE = GKDE(bad_obs_samples)



    # distribution comparison plot ===========================================
    plt.figure()
    xx = np.linspace(np.min(pred_samples), np.max(bad_obs_samples), 1000)

    plt.hist(obs_samples, bins=20, alpha=0.3, density=True, label='Observed histogram', rwidth=0.85, color=CB_color_cycle[0]);
    plt.plot(xx, obs_KDE(xx), color=CB_color_cycle[0], label='Observed KDE');

    plt.hist(pred_samples, bins=20, alpha=0.3, density=True, label='Predicted histogram', rwidth=0.85, color=CB_color_cycle[1]);
    plt.plot(xx, pred_KDE(xx), color=CB_color_cycle[1], label='Predicted KDE', ls=':');

    plt.hist(bad_obs_samples, bins=20, alpha=0.3, density=True, label='Pred. violation observed histogram',
             rwidth=0.85, color=CB_color_cycle[2]);
    plt.plot(xx, bad_obs_KDE(xx), color=CB_color_cycle[2], label='Pred. violation observed KDE', ls='-.');

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), shadow=True);
    plt.xlabel(r'$\mathcal{D}$');
    plt.xticks(ticks=np.linspace(np.min(pred_samples), np.max(bad_obs_samples), 6),
               labels=["{:.3f}".format(x) for x in np.linspace(np.min(pred_samples), np.max(bad_obs_samples), 6)]);
    plt.ylabel('Density');

    plt.savefig(f'{plot_directory}/heat_eq_dists.png', bbox_inches='tight')
    # ========================================================================


    # compute the density-based solution weights (radon-nikodym weights) on the initial samples
    r = obs_KDE(pred_samples.T) / pred_KDE(pred_samples.T)
    rn_w = r / n_init_samples
    print(f'For the density-based solution:')
    print(f'    E(r) = {np.mean(r)}') # computes the diagnostic for the density solution

    r_bad = bad_obs_KDE(pred_samples.T) / pred_KDE(pred_samples.T)
    print(f'For the density-based solution that violates the predictability assumption:')
    print(f'    E(r) = {np.mean(r_bad)}')

    # once we have the radon-nikodym weights, we use rejection sampling to find the solution and its push-forward
    update_inds = rejection_sampling(r)
    update_samples = init_samples[update_inds]

    pf_samples = pred_samples[update_inds]
    pf_KDE = GKDE(pf_samples.T)


    # plotting density results on data space =================================
    plt.figure()
    xx = np.linspace(np.min(pred_samples[:,0]), np.max(pred_samples[:,0]), 1000)

    plt.plot(xx, obs_KDE(xx), label=r'$\pi_{obs}$', ls='-');
    plt.plot(xx, pred_KDE(xx), label=r'$\pi_{pred}$', ls=':');
    plt.plot(xx, pf_KDE(xx), label=r'$\pi_{update}$', ls='--');

    plt.xticks(ticks=np.linspace(np.min(pred_samples), np.max(pred_samples), 6),
               labels=["{:.3f}".format(x) for x in np.linspace(np.min(pred_samples), np.max(pred_samples), 6)]);
    plt.xlabel(r'$u$');
    plt.legend(shadow=True);

    plt.tight_layout()
    plt.savefig(f'{plot_directory}/dens_results.png', bbox_inches='tight')
    # ========================================================================


    # plotting density results after rejection sampling ======================
    plt.figure()
    plt.scatter(init_samples[update_inds,0], init_samples[update_inds,1], alpha=0.7)

    ax = plt.gca()
    ax.set_aspect(0.18)

    plt.xlim(1.945, 2.055)
    plt.ylim(0.725, 1.275)
    plt.xlabel(r'$\ell$')
    plt.ylabel(r'$\kappa$')
    plt.xticks(ticks=np.linspace(np.min(XX), np.max(XX), 6),
               labels=["{:.2f}".format(x) for x in np.linspace(np.min(XX), np.max(XX), 6)])
    plt.yticks(ticks=np.linspace(np.min(YY), np.max(YY), 6),
               labels=["{:.2f}".format(x) for x in np.linspace(np.min(YY), np.max(YY), 6)])

    plt.tight_layout()
    plt.savefig(f'{plot_directory}/rejection.png', bbox_inches='tight')
    # ========================================================================


    # plotting density results as weights ====================================
    plt.figure()
    ax = plt.gca()
    ax.set_aspect(0.18)

    plt.scatter(init_samples[:,0], init_samples[:,1], c=r/n_init_samples)

    plt.xlim(1.945, 2.055)
    plt.ylim(0.725, 1.275)
    plt.xlabel(r'$\ell$')
    plt.ylabel(r'$\kappa$')
    plt.xticks(ticks=np.linspace(np.min(XX), np.max(XX), 6),
               labels=["{:.2f}".format(x) for x in np.linspace(np.min(XX), np.max(XX), 6)])
    plt.yticks(ticks=np.linspace(np.min(YY), np.max(YY), 6),
               labels=["{:.2f}".format(x) for x in np.linspace(np.min(YY), np.max(YY), 6)])

    cbar = plt.colorbar()
    cbar.set_ticks(np.linspace(np.min(r/n_init_samples), np.max(r/n_init_samples), 7))
    cbar.set_ticklabels(["{:.4f}".format(x) for x in np.linspace(np.min(r/n_init_samples),
                                                                 np.max(r/n_init_samples), 7)])

    plt.tight_layout()
    plt.savefig(f'{plot_directory}/rn_weights.png', bbox_inches='tight')
    # ========================================================================


    # computing naive optimization-based results
    H = wEDFs.compute_H(np.reshape(pred_samples, (len(pred_samples),1)))
    b = wEDFs.compute_b(np.reshape(pred_samples, (len(pred_samples),1)),
                        sample_set_2=np.reshape(obs_samples, (len(obs_samples),1)))
    w = wEDFs.compute_optimal_w(H, b)


    # plotting naive results as weights ======================================
    plt.figure()
    vmin = 0
    vmax = 0.003
    plt.scatter(init_samples[:,0], init_samples[:,1], c=w/n_init_samples, cmap='viridis', vmin=vmin, vmax=vmax)

    plt.xlabel(r'$\ell$')
    plt.ylabel(r'$\kappa$')
    plt.xticks(ticks=np.linspace(np.min(XX), np.max(XX), 6),
               labels=["{:.2f}".format(x) for x in np.linspace(np.min(XX), np.max(XX), 6)])
    plt.yticks(ticks=np.linspace(np.min(YY), np.max(YY), 6),
               labels=["{:.2f}".format(x) for x in np.linspace(np.min(YY), np.max(YY), 6)])

    cbar = plt.colorbar()
    cbar.set_ticks(np.linspace(vmin, vmax, 7))
    cbar.set_ticklabels(["{:.4f}".format(x) for x in np.linspace(vmin, vmax, 7)])

    plt.tight_layout()
    plt.savefig(f'{plot_directory}/naive_weights.png', bbox_inches='tight')
    # ========================================================================


    # plot weight comparison between density-based and naive =================
    plt.figure()
    plt.scatter(rn_w, w/n_init_samples)
    plt.plot(np.linspace(0, np.max(rn_w), 1000),
             np.linspace(0, np.max(rn_w), 1000), color='k')

    plt.xticks(ticks=np.linspace(np.min(rn_w), np.max(rn_w), 5),
               labels=["{:.4f}".format(x) for x in np.linspace(np.min(rn_w), np.max(rn_w), 5)])
    plt.yticks(ticks=np.linspace(np.min(w/n_init_samples), np.max(w/n_init_samples), 7),
               labels=["{:.4f}".format(x) for x in np.linspace(np.min(w/n_init_samples), np.max(w/n_init_samples), 7)])
    plt.xlabel('Radon-Nikodym weights')
    plt.ylabel('Naïve optimized weights')

    plt.tight_layout()
    plt.savefig(f'{plot_directory}/rn_vs_naive.png')
    # ========================================================================


    # beginning binning-based method computations
    p = 35

    rpartitioned_w, bins, centers, w_center = binning.computePartitionedWeights_regulargrid_IID(init_samples,
                                                                            pred_samples,
                                                                            sample_set_2=np.reshape(obs_samples, (len(obs_samples),1)),
                                                                            n_bins=p)

    # plotting weights from regular gridded binning ==========================
    plt.figure()
    plt.scatter(init_samples[:,0], init_samples[:,1], c=rpartitioned_w)

    plt.xlabel(r'$\ell$')
    plt.ylabel(r'$\kappa$')
    plt.xticks(ticks=np.linspace(np.min(XX), np.max(XX), 6),
               labels=["{:.2f}".format(x) for x in np.linspace(np.min(XX), np.max(XX), 6)])
    plt.yticks(ticks=np.linspace(np.min(YY), np.max(YY), 6),
               labels=["{:.2f}".format(x) for x in np.linspace(np.min(YY), np.max(YY), 6)])

    cbar = plt.colorbar()
    cbar.set_ticks(np.linspace(np.min(rpartitioned_w), np.max(rpartitioned_w), 7))
    cbar.set_ticklabels(["{:.4f}".format(x) for x in np.linspace(np.min(rpartitioned_w),
                                                                 np.max(rpartitioned_w), 7)])

    plt.tight_layout()
    plt.savefig(f'{plot_directory}/regpart_weights.png', bbox_inches='tight')
    # ========================================================================


    # plot weight comparison between reg binning and density =================
    plt.figure()
    plt.scatter(rn_w, rpartitioned_w)
    plt.plot(np.linspace(0, np.max(rn_w), 1000),
             np.linspace(0, np.max(rn_w), 1000), color='k')

    plt.xticks(ticks=np.linspace(np.min(rn_w), np.max(rn_w), 5),
               labels=["{:.4f}".format(x) for x in np.linspace(np.min(rn_w), np.max(rn_w), 5)])
    plt.yticks(ticks=np.linspace(np.min(rpartitioned_w), np.max(rpartitioned_w), 7),
               labels=["{:.4f}".format(x) for x in np.linspace(np.min(rpartitioned_w), np.max(rpartitioned_w), 7)])
    plt.xlabel('Radon-Nikodym weights')
    plt.ylabel('Regular binning weights')

    plt.tight_layout()
    plt.savefig(f'{plot_directory}/rn_vs_regpart.png', bbox_inches='tight')
    # ========================================================================


    # plot regular binning contour sets ======================================
    plt.figure()
    for i in range(p):
        plt.scatter(init_samples[(bins==i),0],init_samples[(bins==i),1])

    plt.xlabel(r'$\ell$')
    plt.ylabel(r'$\kappa$')
    plt.xticks(ticks=np.linspace(np.min(XX), np.max(XX), 6),
               labels=["{:.2f}".format(x) for x in np.linspace(np.min(XX), np.max(XX), 6)])
    plt.yticks(ticks=np.linspace(np.min(YY), np.max(YY), 6),
               labels=["{:.2f}".format(x) for x in np.linspace(np.min(YY), np.max(YY), 6)])

    plt.tight_layout()
    plt.savefig(f'{plot_directory}/regpart_cells.png', bbox_inches='tight')
    # ========================================================================


    # binning with K-means partitioning
    kpartitioned_w, clusters, centers, w_center = binning.computePartitionedWeights_kMeans_IID(init_samples,
                                                                            pred_samples,
                                                                            sample_set_2=np.reshape(obs_samples, (len(obs_samples),1)),
                                                                            n_clusters=p)


    # plot weights from K-means binning ======================================
    plt.figure()
    plt.scatter(init_samples[:,0], init_samples[:,1], c=kpartitioned_w)

    plt.xlabel(r'$\ell$')
    plt.ylabel(r'$\kappa$')
    plt.xticks(ticks=np.linspace(np.min(XX), np.max(XX), 6),
               labels=["{:.2f}".format(x) for x in np.linspace(np.min(XX), np.max(XX), 6)])
    plt.yticks(ticks=np.linspace(np.min(YY), np.max(YY), 6),
               labels=["{:.2f}".format(x) for x in np.linspace(np.min(YY), np.max(YY), 6)])

    cbar = plt.colorbar()
    cbar.set_ticks(np.linspace(np.min(kpartitioned_w), np.max(kpartitioned_w), 7))
    cbar.set_ticklabels(["{:.4f}".format(x) for x in np.linspace(np.min(kpartitioned_w),
                                                                 np.max(kpartitioned_w), 7)])

    plt.tight_layout()
    plt.savefig(f'{plot_directory}/kmeans_part_weights.png', bbox_inches='tight')
    # ========================================================================


    # plot comparison to density weights =====================================
    plt.figure()
    plt.scatter(rn_w, kpartitioned_w)
    plt.plot(np.linspace(0, np.max(rn_w), 1000),
             np.linspace(0, np.max(rn_w), 1000), color='k')

    plt.xticks(ticks=np.linspace(np.min(rn_w), np.max(rn_w), 5),
               labels=["{:.4f}".format(x) for x in np.linspace(np.min(rn_w), np.max(rn_w), 5)])
    plt.yticks(ticks=np.linspace(np.min(kpartitioned_w), np.max(kpartitioned_w), 7),
               labels=["{:.4f}".format(x) for x in np.linspace(np.min(kpartitioned_w), np.max(kpartitioned_w), 7)])
    plt.xlabel('Radon-Nikodym weights')
    plt.ylabel('K-means binning weights')

    plt.tight_layout()
    plt.savefig(f'{plot_directory}/rn_vs_kmeans_part.png')
    # ========================================================================


    # plot bins from K-means on parameter space ==============================
    plt.figure()
    for i in range(p):
        plt.scatter(init_samples[(clusters==i),0], init_samples[(clusters==i),1])

    plt.xlabel(r'$\ell$')
    plt.ylabel(r'$\kappa$')

    plt.xticks(ticks=np.linspace(np.min(XX), np.max(XX), 6),
               labels=["{:.2f}".format(x) for x in np.linspace(np.min(XX), np.max(XX), 6)])
    plt.yticks(ticks=np.linspace(np.min(YY), np.max(YY), 6),
               labels=["{:.2f}".format(x) for x in np.linspace(np.min(YY), np.max(YY), 6)])

    plt.tight_layout()
    plt.savefig(f'{plot_directory}/kmeans_part_cells.png', bbox_inches='tight')
    # ========================================================================


    # plot weights comparison between regular and K-means binning ============
    plt.figure()
    plt.plot(np.linspace(0, np.max(rn_w), 1000),
             np.linspace(0, np.max(rn_w), 1000), color='k', label='Identity')
    plt.scatter(rn_w, rpartitioned_w, label='Regular binning', marker='x')

    plt.xlabel('Radon-Nikodym weights')
    plt.ylabel('Partitioned weights')
    plt.scatter(rn_w, kpartitioned_w, label='K-Means binning')
    plt.xticks(ticks=np.linspace(np.min(rn_w), np.max(rn_w), 5),
               labels=["{:.4f}".format(x) for x in np.linspace(np.min(rn_w), np.max(rn_w), 5)])
    plt.yticks(ticks=np.linspace(np.min(kpartitioned_w), np.max(kpartitioned_w), 7),
               labels=["{:.4f}".format(x) for x in np.linspace(np.min(kpartitioned_w), np.max(kpartitioned_w), 7)])
    plt.legend(shadow=True)

    plt.tight_layout()
    plt.savefig(f'{plot_directory}/regpart_vs_kmeans.png')
    # ========================================================================


    # begin experiment with smaller number of samples and bins
    n_init_samples_small = 200

    H = wEDFs.compute_H(pred_samples[:n_init_samples_small,:])
    b = wEDFs.compute_b(pred_samples[:n_init_samples_small,:], targ_CDF=obs_dist.cdf)
    w = wEDFs.compute_optimal_w(H, b)

    p = 10

    kpartitioned_w, clusters, centers, w_center = binning.computePartitionedWeights_kMeans_IID(init_samples[:n_init_samples_small,:],
                                                                            pred_samples[:n_init_samples_small,:],
                                                                            sample_set_2=np.reshape(obs_samples, (len(obs_samples),1)),
                                                                            n_clusters=p)

    rpartitioned_w, bins, centers, w_center = binning.computePartitionedWeights_regulargrid_IID(init_samples[:n_init_samples_small,:],
                                                                            pred_samples[:n_init_samples_small,:],
                                                                            sample_set_2=np.reshape(obs_samples, (len(obs_samples),1)),
                                                                            n_bins=p)

    isort = np.argsort(pred_samples[:n_init_samples_small,0])
    isort_obs = np.argsort(obs_samples)


    # plot weights EDFs from optimization methods ============================
    plt.figure()
    plt.plot(np.append(np.min(pred_samples[:,0]), np.append(obs_samples[isort_obs], np.max(pred_samples[:,0]))),
             np.append(0, np.append(np.cumsum([1/n_obs_samples]*n_obs_samples), 1)), label=r'$F^m_{obs}$');
    plt.step(pred_samples[isort], np.cumsum(w[isort]/n_init_samples_small),
             label=r'$F^m_{w;pred}$, naïve', ls='dotted');
    plt.step(pred_samples[isort], np.cumsum(rpartitioned_w[isort]),
             label=r'$F^m_{w;pred}$, regular partition', ls='--');
    plt.step(pred_samples[isort], np.cumsum(kpartitioned_w[isort]),
             label=r'$F^m_{w;pred}$, k-means partition', ls='-.');

    plt.xlim(0.584, 0.606)
    plt.xticks(ticks=np.linspace(0.584, 0.606, 6),
               labels=["{:.3f}".format(x) for x in np.linspace(np.min(ZZ), np.max(ZZ), 6)]);
    plt.xlabel(r'$\mathcal{D}$');
    plt.ylabel('Cumulative distribution');
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), shadow=True);

    plt.tight_layout()
    plt.savefig(f'{plot_directory}/small_cdf_compare.png')
    # ========================================================================


    r = obs_KDE(pred_samples[:n_init_samples_small].T) / pred_KDE(pred_samples[:n_init_samples_small].T)
    rn_w = r / n_init_samples_small
    print(f'For the density-based small sample solution:')
    print(f'    E(r) = {np.mean(r)}')

    pf_KDE = GKDE(pred_samples[:n_init_samples_small].T, weights=rn_w)
    pred_samples[:n_init_samples_small, 0] = u(N, sensor_loc, t, init_samples[:n_init_samples_small,0],
                                               init_samples[:n_init_samples_small,1])
    pred_KDE = GKDE(pred_samples[:n_init_samples_small,0])


    # begin computations for the discrete (no-density admitting) example
    data_space = [0.585, 0.6]

    n_data_pieces = 3
    data_splits = (np.array([0, 1/3, 2/3, 1]) * (data_space[1] - data_space[0])) + 0.585

    obs_dens_pieces = [5, 1, 4]
    obs_dens_pieces = obs_dens_pieces / np.sum(obs_dens_pieces)
    obs_dens_pieces = [obs_dens_pieces[i] / (data_splits[i+1] - data_splits[i]) for i in range(n_data_pieces)]

    n_obs_samples = 10000
    obs_samples = obs_sample_discrete(n_obs_samples, n_data_pieces, data_splits, obs_dens_pieces)
    obs_KDE = GKDE(obs_samples)


    # plot distributions for the no density example ==========================
    plt.figure()
    xx = np.linspace(np.min(pred_samples)-0.01, np.max(pred_samples)+0.01, 1000)

    plt.hist(obs_samples, bins=15, alpha=0.3, density=True, label='Observed hist.', rwidth=0.85, color=CB_color_cycle[0]);
    plt.plot(xx, obs_KDE(xx), color=CB_color_cycle[0], label='Observed KDE');

    plt.hist(pred_samples, bins=25, alpha=0.3, density=True, label='Predicted hist.', rwidth=0.85, color=CB_color_cycle[1]);
    plt.plot(xx, pred_KDE(xx), color=CB_color_cycle[1], label='Predicted KDE', ls=':');

    plt.legend(shadow=True);
    plt.xlabel(r'$\mathcal{D}$');
    plt.xlim(np.min(pred_samples[:,0]-0.002), np.max(pred_samples[:,0]+0.015));
    plt.ylabel('Density');

    plt.savefig(f'{plot_directory}/no_density_dists.png', bbox_inches='tight')
    # ========================================================================


    r_no_dens = obs_KDE(pred_samples.T) / pred_KDE(pred_samples.T)
    print(f'For the no-density admitting solution:')
    print(f'    E(r) = {np.mean(r_no_dens)}')

    update_inds = rejection_sampling(r_no_dens)
    update_samples = init_samples[update_inds]

    pf_samples = pred_samples[update_inds]
    pf_KDE = GKDE(pf_samples.T)


    # plot densities for the no-deensity admitting example ===================
    plt.figure()

    xx = np.linspace(np.min(pred_samples[:,0]), np.max(pred_samples[:,0]), 1000)

    plt.plot(xx, obs_KDE(xx), label=r'$\pi_{obs}$');
    plt.plot(xx, pred_KDE(xx), label=r'$\pi_{pred}$', ls=':');
    plt.plot(xx, pf_KDE(xx), label=r'$\pi_{update}$', ls='--');

    plt.xticks(ticks=np.linspace(np.min(pred_samples), np.max(pred_samples), 6),
               labels=["{:.3f}".format(x) for x in np.linspace(np.min(pred_samples), np.max(pred_samples), 6)]);
    plt.xlabel(r'$u$');
    plt.ylabel('Density');
    plt.legend(shadow=True);

    plt.tight_layout()
    plt.savefig(f'{plot_directory}/no_dens_density.png', bbox_inches='tight')
    # ========================================================================


    n_bins = 100

    kpartitioned_w, clusters, centers, w_center = binning.computePartitionedWeights_kMeans_IID(init_samples,
                                                            pred_samples,
                                                            sample_set_2=np.reshape(obs_samples, (len(obs_samples),1)),
                                                            n_clusters=n_bins)


    pf_CDF = np.zeros(np.shape(xx[1:]))
    for count, x in enumerate(xx[1:]):
        pf_CDF[count],temp = integrate.quad(pf_KDE, np.min(pred_samples[:,0]), x)


    # plot resulting weighted EDFs for no density admitting example ==========
    plt.figure()
    isort = np.argsort(pred_samples[:,0])
    isort_obs = np.argsort(obs_samples)

    plt.step(np.append(np.min(pred_samples[:,0]), np.append(obs_samples[isort_obs], np.max(pred_samples[:,0])+0.015)),
             np.append(0, np.append(np.cumsum([1/n_obs_samples]*n_obs_samples), 1)),
             label=r'$F^m_{obs}$');
    plt.step(np.append(pred_samples[isort,0], np.max(pred_samples[:,0])+0.015),
             np.append(np.cumsum([1/n_init_samples]*n_init_samples), 1), label='$F^n_{pred}$', ls=':');
    plt.step(np.append(pred_samples[isort,0], np.max(pred_samples[:,0])+0.015),
             np.append(np.cumsum(kpartitioned_w[isort]), 1), label='$F^n_{pred;w}$', ls='--');
    plt.plot(xx[1:], pf_CDF, label='PF update CDF', ls='--');

    plt.xlabel(r'$\mathcal{D}$');
    plt.xlim(np.min(pred_samples[:,0]-0.002), np.max(pred_samples[:,0]+0.015));
    plt.ylabel('Cumulative distribution');
    plt.legend(loc='lower right', shadow=True);

    plt.tight_layout()
    plt.savefig(f'{plot_directory}/no_dens_cdfs.png', bbox_inches='tight')
    # ========================================================================


    # plot results for no density admitting example ==========================
    plt.figure()
    xx = np.linspace(np.min(pred_samples[:,0]), np.max(pred_samples[:,0]), 1000)

    plt.plot(xx, obs_KDE(xx), label=r'$\pi_{obs}$');
    plt.plot(xx, pred_KDE(xx), label=r'$\pi_{pred}$', ls=':');
    plt.plot(xx, pf_KDE(xx), label=r'$\pi_{update}$', ls='--');

    plt.xticks(ticks=np.linspace(np.min(pred_samples), np.max(pred_samples), 6),
               labels=["{:.3f}".format(x) for x in np.linspace(np.min(pred_samples), np.max(pred_samples), 6)]);
    plt.xlabel(r'$u$');
    plt.ylabel('Density');
    plt.legend(shadow=True);

    plt.tight_layout()
    plt.savefig(f'{plot_directory}/no_dens_densities.png', bbox_inches='tight')
    # ========================================================================


if __name__ == '__main__':

    main()
