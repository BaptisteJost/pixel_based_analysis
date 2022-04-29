import IPython
import numpy as np
from astropy import units as u
import matplotlib.pyplot as plt
import os
import IPython


def get_data(path_dir):
    precision_grid = []
    sigmar_grid = []
    sigmaBeta_grid = []
    listdir = [name for name in os.listdir(path_dir) if os.path.isdir(os.path.join(path_dir, name))]
    for name in listdir:
        precision_grid.append(np.load(path_dir+name+'/prior_precision.npy').tolist())
        fisher = np.load(path_dir+name+'/fisher_cosmo.npy')
        sigmaBeta_grid.append([precision_grid[-1][0], np.sqrt(np.linalg.inv(fisher)[1, 1])])
        sigmar_grid.append([precision_grid[-1][0], np.sqrt(np.linalg.inv(fisher)[0, 0])])
    sigmaBeta_grid = np.sort(sigmaBeta_grid, 0)
    sigmar_grid = np.sort(sigmar_grid, 0)
    precision_grid = np.sort(precision_grid, 0)
    return precision_grid[:, 1], sigmar_grid[:, 1], sigmaBeta_grid[:, 1]


path = '/home/baptiste/Documents/these/pixel_based_analysis/results_and_data/full_pipeline/'
precision_1prior, sigmar_1prior, sigmaBeta_1prior = get_data(path+'gridding_1prior_wrerun/')
precision_6priors, sigmar_6priors, sigmaBeta_6priors = get_data(path+'gridding_6prior_wrerun/')

# sigmaBeta_1prior +=

IPython.embed()
plt.scatter(precision_1prior, sigmar_1prior)
plt.scatter(precision_6priors, sigmar_6priors)
ylims = plt.ylim()
plt.close()
plt.vlines(1, ylims[0], ylims[1], label='Wire grid', linestyles='--', color='darkred')
plt.vlines(0.27, ylims[0], ylims[1], label='Tau A', linestyles='--', color='red')
plt.fill_between([0.01, 0.1], ylims[0], ylims[1], color='grey', alpha=0.3, label='Drone')
plt.scatter(precision_1prior, sigmar_1prior,
            label=r'$\sigma (r)$ 1 prior', color='orange')
plt.scatter(precision_6priors, sigmar_6priors,
            label=r'$\sigma (r)$ 6 priors', color='darkred')
plt.xlabel('prior precision in deg')
plt.ylabel(r'$\sigma (r)$')
plt.grid(b=True, linestyle=':')
plt.xscale('log')
plt.legend(loc='upper left')
plt.ylim(ylims[0], ylims[1])
plt.savefig(path+'gridding_1prior_wrerun/'+'sigma_r_wrt_prior.png', bbox_inches='tight', dpi=200)
plt.savefig(path+'gridding_1prior_wrerun/'+'sigma_r_wrt_prior.pdf', bbox_inches='tight', dpi=200)
plt.close()


plt.scatter(precision_1prior, (sigmaBeta_1prior*u.rad).to(u.deg))
plt.scatter(precision_6priors, (sigmaBeta_6priors*u.rad).to(u.deg))
ylims = plt.ylim()
plt.close()
plt.vlines(1, ylims[0], ylims[1], label='Wire grid', linestyles='--', color='darkred')
plt.vlines(0.27, ylims[0], ylims[1], label='Tau A', linestyles='--', color='red')
plt.fill_between([0.01, 0.1], ylims[0], ylims[1], color='grey', alpha=0.3, label='Drone')
plt.scatter(precision_1prior, (sigmaBeta_1prior*u.rad).to(u.deg),
            label=r'$\sigma (\beta_b)$ 1 prior', color='orange')
plt.scatter(precision_6priors, (sigmaBeta_6priors*u.rad).to(u.deg),
            label=r'$\sigma (\beta_b)$ 6 priors', color='darkred')

plt.xlabel('prior precision in deg')
plt.ylabel(r'$\sigma (\beta_b)$ in deg')

plt.grid(b=True, linestyle=':')
plt.xscale('log')
plt.legend(loc='upper left')
plt.ylim(ylims[0], ylims[1])
plt.savefig(path+'gridding_1prior_wrerun/'+'sigma_beta_wrt_prior.png', bbox_inches='tight', dpi=200)
plt.savefig(path+'gridding_1prior_wrerun/'+'sigma_beta_wrt_prior.pdf', bbox_inches='tight', dpi=200)
plt.close()
