import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.stats import gaussian_kde as GKDE
from scipy.stats import multivariate_normal
from scipy.stats import norm
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

    random.seed(10)

    # sets whether you want to use pre-computed data or rerun methods
    rerun_densities = False
    rerun_distributions = False

    l = 2.
    kappa = 1.

    N = 100  # Specify the truncation

    t = .01
    sensor_loc = 1.2

    x = np.linspace(0, l, 2*N)

    # define set B on the output space
    low_x = 2.01
    upp_x = 2.02
    low_y = 0.95
    upp_y = 1.0

    upp_B = u(N, sensor_loc, t, low_x, low_y)
    low_B = u(N, sensor_loc, t, upp_x, upp_y)

    delta_l = 0.1
    delta_kappa = 0.5

    print(f'Upper bound of B = {upp_B}')
    print(f'Lower bound of B = {low_B}')

    # for plotting contour map
    X = np.linspace(l - delta_l / 2, delta_l + l - delta_l / 2, 100)
    Y = np.linspace(kappa - delta_kappa / 2, delta_kappa + kappa - delta_kappa / 2, 100)

    XX, YY = np.meshgrid(X, Y)
    ZZ = np.zeros(np.shape(XX))

    for count, x in enumerate(X):
        ZZ[count,:] = u(N, sensor_loc, t, XX[count,:], YY[count,:])

    d_min = np.min(ZZ)
    d_max = np.max(ZZ)

    # observed distribution
    n_obs_samples = 100000
    obs_dist = norm(0.595, 3e-3)
    obs_samples = obs_dist.rvs(n_obs_samples)
    obs_KDE = GKDE(obs_samples)

    # set initial samples for running density trials
    n_init_samples = 100000

    n_density_trials = 50 # number of density trials

    if rerun_densities:

        true_up_prob_set = []
        for trial in range(n_density_trials):

            print(f'working on density trial {trial}')

            init_samples = np.random.uniform(0, 1, (n_init_samples,2))
            init_samples[:,0] = init_samples[:,0] * delta_l + l - delta_l / 2
            init_samples[:,1] = init_samples[:,1] * delta_kappa + kappa - delta_kappa / 2

            pred_samples = np.zeros((n_init_samples, 1))
            pred_samples[:, 0] = u(N, sensor_loc, t, init_samples[:,0], init_samples[:,1])
            pred_KDE = GKDE(pred_samples[:,0])

            r = obs_KDE(pred_samples.T) / pred_KDE(pred_samples.T)
            rn_w = r / n_init_samples

            true_up_prob_set.append(0)
            for i, isamp in enumerate(init_samples):
                if isamp[0] >= low_x and isamp[0] <= upp_x and isamp[1] >= low_y and isamp[1] <= upp_y:
                    true_up_prob_set[int(trial)] += rn_w[i]

        np.save('./data/convergence_dens_prob_A.npy', np.array(true_up_prob_set))

    else:

        init_samples = np.random.uniform(0, 1, (n_init_samples,2))
        init_samples[:,0] = init_samples[:,0] * delta_l + l - delta_l / 2
        init_samples[:,1] = init_samples[:,1] * delta_kappa + kappa - delta_kappa / 2

        pred_samples = np.zeros((n_init_samples, 1))
        pred_samples[:, 0] = u(N, sensor_loc, t, init_samples[:,0], init_samples[:,1])
        pred_KDE = GKDE(pred_samples[:,0])

        r = obs_KDE(pred_samples.T) / pred_KDE(pred_samples.T)
        rn_w = r / n_init_samples

        true_up_prob_set = np.load('./data/convergence_dens_prob_A.npy')

    obs_prob_set = 0
    for osamp in obs_samples:
        if osamp >= low_B and osamp <= upp_B:
            obs_prob_set += 1 / n_obs_samples

    true_up_prob_set = np.mean(true_up_prob_set)


    # plot set A in parameter space ========================================
    plt.figure()
    ax = plt.gca()

    plt.scatter(init_samples[:,0], init_samples[:,1], c=r/n_init_samples)

    plt.contour(XX, YY, ZZ, levels=[low_B, upp_B], colors=['w', 'w'])

    set_A = patches.Rectangle((low_x, low_y), upp_x-low_x, upp_y-low_y, color='k',
                              fill=None, zorder=10, linewidth=2.5)
    ax.add_patch(set_A)

    plt.text(upp_x-0.002, low_y-0.04, r'$A$', fontsize=16)
    plt.text(upp_x-0.006, upp_y+0.1, r'$Q^{-1}(B)$', fontsize=16, color='w')

    plt.xlabel(r'$\ell$')
    plt.ylabel(r'$\kappa$')
    plt.xticks(ticks=np.linspace(np.min(XX), np.max(XX), 7),
               labels=["{:.3f}".format(x) for x in np.linspace(np.min(XX), np.max(XX), 7)])

    plt.tight_layout()
    plt.savefig(f'{plot_directory}/sets_Lambda.png', bbox_inches='tight')
    # ======================================================================


    # plot set B in data space =============================================
    plt.figure()

    xx = np.linspace(np.min(pred_samples[:,0]), np.max(pred_samples[:,0]), 1000)

    plt.plot(xx, obs_KDE(xx), label=r'$\pi_{obs}$', color='k')
    plt.yticks(fontsize=14)
    plt.text((upp_B-low_B)/2 + low_B, 147, r'$B$', color='r', fontsize=16,
             horizontalalignment='center', verticalalignment='center')
    plt.text(low_B, 147, r'$[$', color='r', fontsize=16, horizontalalignment='center', verticalalignment='center')
    plt.text(upp_B, 147, r'$]$', color='r', fontsize=16, horizontalalignment='center', verticalalignment='center')
    plt.axvspan(low_B, upp_B, alpha=0.3, color='r')

    plt.legend(loc='upper right', shadow=True)
    plt.xlabel(r'$u$')
    plt.xticks(ticks=np.linspace(np.min(xx), np.max(xx), 7),
               labels=["{:.3f}".format(x) for x in np.linspace(np.min(xx), np.max(xx), 7)], color='k')

    plt.tight_layout()
    plt.savefig(f'{plot_directory}/sets_D.png', bbox_inches='tight')
    # ======================================================================


    # setting p and n for distributions tables
    bin_numbers = [20, 40, 60, 80, 100, 120, 140, 160]
    init_samples_numbers = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]

    # run for a single distributions trial
    comp_prob_A = []
    comp_prob_B = []
    count_A = []
    count_B = []

    init_samples_full = np.random.uniform(0, 1, (init_samples_numbers[-1],2))
    init_samples_full[:,0] = init_samples_full[:,0] * delta_l + l - delta_l / 2
    init_samples_full[:,1] = init_samples_full[:,1] * delta_kappa + kappa - delta_kappa / 2
    pred_samples_full = np.zeros((init_samples_numbers[-1], 1))
    pred_samples_full[:, 0] = u(N, sensor_loc, t, init_samples_full[:,0], init_samples_full[:,1])

    for i, n_init_samples in enumerate(init_samples_numbers):

        comp_prob_A.append([])
        comp_prob_B.append([])

        init_samples = init_samples_full[:n_init_samples,:]
        pred_samples = pred_samples_full[:n_init_samples,:]

        for b, n_bins in enumerate(bin_numbers):

            rpartitioned_w, bins, centers, w_center = binning.computePartitionedWeights_regulargrid_IID(init_samples,
                                                                                    pred_samples, bbox=[d_min,d_max],
                                                                                    sample_set_2=np.reshape(obs_samples, (len(obs_samples),1)),
                                                                                    n_bins=n_bins)

            weight_set = 0
            count_in_B = 0
            for count, psamp in enumerate(pred_samples):
                if psamp >= low_B and psamp <= upp_B:
                    weight_set += rpartitioned_w[count]
                    count_in_B += 1

            centers_in_B = 0
            for center in centers:
                if center >= low_B and center <= upp_B:
                    centers_in_B += 1

            up_prob_set = 0
            count_in_A = 0
            for count, isamp in enumerate(init_samples):
                if isamp[0] >= low_x and isamp[0] <= upp_x and isamp[1] >= low_y and isamp[1] <= upp_y:
                    count_in_A += 1
                    up_prob_set += rpartitioned_w[count]

            comp_prob_B[i].append(weight_set)
            comp_prob_A[i].append(up_prob_set)

        count_B.append(count_in_B)
        count_A.append(count_in_A)


    # compute errors in probability of B ===================================
    frow_string = f'        '
    for i in bin_numbers:
        frow_string += f' {i:6d}  '

    print(frow_string)
    for count, b in enumerate(comp_prob_B):
        row_string = f'{init_samples_numbers[count]:5d}   '
        for i in b:
            if i != -999:
                row_string += f'{np.abs(i-obs_prob_set):.5f}  '
            else:
                row_string += f'         '
        print(row_string)

    print()
    print(f'actual probability of B = {obs_prob_set:0.5f}')
    # ======================================================================


    # now run for multiple distributions trials
    n_trials = 100

    if rerun_distributions == True:

        c_A = []
        c_B = []

        for trial in range(n_trials):

            print(f'working on trial {trial}')

            comp_prob_A = []
            comp_prob_B = []

            init_samples_full = np.random.uniform(0, 1, (init_samples_numbers[-1], 2))
            init_samples_full[:,0] = init_samples_full[:,0] * delta_l + l - delta_l / 2
            init_samples_full[:,1] = init_samples_full[:,1] * delta_kappa + kappa - delta_kappa / 2

            pred_samples_full = np.zeros((init_samples_numbers[-1], 1))
            pred_samples_full[:, 0] = u(N, sensor_loc, t, init_samples_full[:,0], init_samples_full[:,1])

            for i, n_init_samples in enumerate(init_samples_numbers):

                comp_prob_A.append([])
                comp_prob_B.append([])

                init_samples = init_samples_full[:n_init_samples,:]

                pred_samples = pred_samples_full[:n_init_samples,:]
                pred_KDE = GKDE(pred_samples[:,0])

                for b, n_bins in enumerate(bin_numbers):

                    rpartitioned_w, bins, centers, w_center = binning.computePartitionedWeights_regulargrid_IID(init_samples,
                                                                                            pred_samples, bbox=[d_min, d_max],
                                                                                            sample_set_2=np.reshape(obs_samples, (len(obs_samples),1)),
                                                                                            n_bins=n_bins)

                    weight_set = 0
                    for count, psamp in enumerate(pred_samples):
                        if psamp >= low_B and psamp <= upp_B:
                            weight_set += rpartitioned_w[count]

                    up_prob_set = 0
                    for count, isamp in enumerate(init_samples):
                        if isamp[0] >= low_x and isamp[0] <= upp_x and isamp[1] >= low_y and isamp[1] <= upp_y:
                            up_prob_set += rpartitioned_w[count]

                    comp_prob_B[i].append(weight_set)
                    comp_prob_A[i].append(up_prob_set)

            c_A.append(comp_prob_A)
            c_B.append(comp_prob_B)

        dist_prob_A = np.array(c_A)
        dist_prob_B = np.array(c_B)

        np.save('./data/convergence_dist_prob_A.npy', dist_prob_A.reshape(n_trials, -1))
        np.save('./data/convergence_dist_prob_B.npy', dist_prob_B.reshape(n_trials, -1))

    else:

        dist_prob_A = np.load('./data/convergence_dist_prob_A.npy').reshape(n_trials,
                                                                            len(init_samples_numbers),
                                                                            len(bin_numbers))
        dist_prob_B = np.load('./data/convergence_dist_prob_B.npy').reshape(n_trials,
                                                                            len(init_samples_numbers),
                                                                            len(bin_numbers))

    # compute mean over trials
    mean_dist_prob_A = np.mean(dist_prob_A, axis=0)
    mean_dist_prob_B = np.mean(dist_prob_B, axis=0)


    # plot heatmap for mean abs error in P(B)  ==============================
    plt.figure(figsize=(12,4.8))

    s = sns.heatmap(np.abs(mean_dist_prob_B-obs_prob_set), annot=True, cmap='viridis', fmt='0.2E',
                yticklabels=init_samples_numbers, xticklabels=bin_numbers,
                norm=colors.LogNorm(vmin=np.min(np.abs(mean_dist_prob_B-obs_prob_set)),
                                    vmax=np.max(np.abs(mean_dist_prob_B-obs_prob_set))))
    s.set_xlabel('$p$')
    s.set_ylabel('$n$')

    plt.tight_layout()
    plt.savefig(f'{plot_directory}/heatmap_meanB.png', bbox_inches='tight')
    # =======================================================================


    # plot heatmap for mean abs error in P(A) ===============================
    plt.figure(figsize=(12,4.8))

    s = sns.heatmap(np.abs(mean_dist_prob_A-true_up_prob_set), annot=True, cmap='viridis', fmt='0.2E',
                    yticklabels=init_samples_numbers, xticklabels=bin_numbers,
                    norm=colors.LogNorm(vmin=np.min(np.abs(mean_dist_prob_A-true_up_prob_set)),
                                        vmax=np.max(np.abs(mean_dist_prob_A-true_up_prob_set))))


    s.set_xlabel('$p$')
    s.set_ylabel('$n$')

    plt.tight_layout()
    plt.savefig(f'{plot_directory}/heatmap_meanA.png', bbox_inches='tight')
    # ========================================================================


    # print table for mean probability of A ==================================
    frow_string = f'        '
    for i in bin_numbers:
        frow_string += f' {i:6d}  '

    print(frow_string)
    for ii, b in enumerate(mean_dist_prob_A):
        row_string = f'{init_samples_numbers[ii]:5d}   '
        for i in b:
            if i != -999:
                row_string += f'{i:.5f}  '
            else:
                row_string += f'         '
        print(row_string)

    print()
    print(f'actual probability of A = {true_up_prob_set:.5f}')
    # ======================================================================


    # standard deviation in error over trials
    std_dist_prob_A = np.std(dist_prob_A, axis=0)
    std_dist_prob_B = np.std(dist_prob_B, axis=0)


    # plot standard deviation of error in prob of B ======================--
    plt.figure(figsize=(12,4.8))

    s = sns.heatmap(np.abs(np.array(std_dist_prob_B)), annot=True, cmap='viridis', fmt='0.2E',
                    yticklabels=init_samples_numbers, xticklabels=bin_numbers)

    s.set_xlabel(r'$p$')
    s.set_ylabel(r'$n$')

    plt.tight_layout()
    plt.savefig(f'{plot_directory}/heatmap_stdB.png', bbox_inches='tight')
    # ======================================================================


    # plot standard deviation of error in prob of A ======================--
    plt.figure(figsize=(12,4.8))

    s = sns.heatmap(np.abs(np.array(std_dist_prob_A)), annot=True, cmap='viridis', fmt='0.2E',
                    yticklabels=init_samples_numbers, xticklabels=bin_numbers)
    s.set_xlabel(r'$p$')
    s.set_ylabel(r'$n$')

    plt.tight_layout()
    plt.savefig(f'{plot_directory}/heatmap_stdA.png', bbox_inches='tight')
    # ======================================================================


if __name__ == '__main__':

    main()
