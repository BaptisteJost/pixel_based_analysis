# from matplotlib.colors import ListedColormap
# from matplotlib.cm import get_cmap
# import matplotlib.cm as cm
# import matplotlib.colors as mpcolors
# from cycler import cycler
# import matplotlib.animation as animation
# import copy
# import residuals as res
import IPython
import numpy as np
from astropy import units as u
import matplotlib.pyplot as plt
# import argparse
# from datetime import date
# import os
# import configparser
# import bjlib.lib_project as lib
# from json import loads

save_path = '../results_and_data/CMBFRANCE/'
grid_prior = np.load(save_path+'grid_prior_precision(1).npy')
fisher_cosmo = np.load(save_path+'fisher_matrix_grid(1).npy')
fisher_spectral = np.load(save_path+'fisher_spectral_grid(3).npy')

error_r = np.sqrt(np.linalg.inv(fisher_cosmo)[:, 0, 0])
error_b = np.sqrt(np.linalg.inv(fisher_cosmo)[:, 1, 1])
error_spectral = np.sqrt(np.linalg.inv(fisher_spectral))

plt.scatter(grid_prior, error_r)
ylims = plt.ylim()
plt.close()
plt.vlines(1, ylims[0], ylims[1], label='Wire grid', linestyles='--', color='darkred')
plt.vlines(0.27, ylims[0], ylims[1], label='Tau A', linestyles='--', color='red')
# plt.vlines(0.1, ylims[0], ylims[1], label='drone', linestyles='--', color='darkred')
plt.fill_between([0.01, 0.1], ylims[0], ylims[1], color='grey', alpha=0.3, label='Drone')
plt.scatter(grid_prior, error_r, label=r'$\sigma (r)$', color='orange')
plt.xlabel('prior precision in deg')
plt.ylabel(r'$\sigma (r)$')
plt.grid(b=True, linestyle=':')
plt.xscale('log')
plt.legend(loc='upper left')
plt.ylim(ylims[0], ylims[1])
# plt.show()
plt.savefig(save_path+'sigma_r_wrt_prior', bbox_inches='tight', dpi=200)
plt.close()

plt.scatter(grid_prior, (error_b*u.rad).to(u.deg).value, color='orange')
ylims = plt.ylim()
plt.close()
plt.vlines(1, ylims[0], ylims[1], label='Wire grid', linestyles='--', color='darkred')
plt.vlines(0.27, ylims[0], ylims[1], label='Tau A', linestyles='--', color='red')
# plt.vlines(0.1, ylims[0], ylims[1], label='drone', linestyles='--', color='darkred')
plt.fill_between([0.01, 0.1], ylims[0], ylims[1], color='grey', alpha=0.3, label='Drone')
plt.scatter(grid_prior, (error_b*u.rad).to(u.deg).value, color='orange', label=r'$\sigma (\beta)$')

plt.xlabel('Prior precision in deg')
plt.ylabel(r'$\sigma (\beta)$ in deg')

plt.grid(b=True, linestyle=':')
plt.legend(loc='upper left')
plt.xscale('log')
plt.legend(loc='upper left')
plt.ylim(ylims[0], ylims[1])
plt.savefig(save_path+'sigma_beta_wrt_prior', bbox_inches='tight', dpi=200)
plt.close()
# plt.show()
IPython.embed()
