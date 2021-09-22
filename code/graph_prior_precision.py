import IPython
import numpy as np
from astropy import units as u
import matplotlib.pyplot as plt
import argparse
from datetime import date
import os

parser = argparse.ArgumentParser()
parser.add_argument('--folder', type=str)
args = parser.parse_args()

path = '/global/homes/j/jost/these/pixel_based_analysis/results_and_data/prior_gridding/'
# folder_name = args.folder
folder_name = date.today().strftime('%Y%m%d')+'_' + os.environ["SLURM_JOB_ID"]

save_path = path+folder_name+'/'

frequencies = np.array([27,  39,  93, 145, 225, 280])

fisher_cosmo = np.load(save_path+'fisher_matrix_grid.npy')
fisher_spectral = np.load(save_path+'fisher_spectral_grid.npy')
grid_prior = np.load(save_path+'grid_prior_precision.npy')
grid_contral = np.load(save_path+'grid_prior_precision_control.npy')
cosmo = np.load(save_path+'cosmo_results_grid_prior.npy')
spectral = np.load(save_path+'spectral_results_grid_prior.npy')

error_r = np.sqrt(np.linalg.inv(fisher_cosmo)[:, 0, 0])
error_b = np.sqrt(np.linalg.inv(fisher_cosmo)[:, 1, 1])
error_spectral = np.sqrt(np.linalg.inv(fisher_spectral))
# IPython.embed()
plt.errorbar(grid_prior, cosmo[:, 0], yerr=error_r, fmt='o')
plt.xlabel('prior precision in deg')
plt.ylabel('r')
plt.savefig(save_path+'r_wrt_prior', bbox_inches='tight')
plt.close()

plt.errorbar(grid_prior, (cosmo[:, 1]*u.rad).to(u.deg).value,
             yerr=(error_b*u.rad).to(u.deg).value, fmt='o')
plt.xlabel('prior precision in deg')
plt.ylabel('beta')
plt.savefig(save_path+'beta_wrt_prior', bbox_inches='tight')
plt.close()

for i in range(6):
    plt.errorbar(grid_prior, (spectral[:, i]*u.rad).to(u.deg).value,
                 yerr=(error_spectral[:, i, i]*u.rad).to(u.deg).value, fmt='o')
    plt.xlabel('prior precision in deg')
    plt.ylabel('miscal angle at {}Ghz'.format(frequencies[i]))
    plt.savefig(
        save_path+'miscal{}GHz_wrt_prior'.format(frequencies[i]), bbox_inches='tight')
    plt.close()

plt.errorbar(grid_prior, spectral[:, -2],
             yerr=error_spectral[:, -2, -2], fmt='o')
plt.xlabel('prior precision in deg')
plt.ylabel(r'$\beta_d$')
plt.savefig(save_path+'beta_dust_wrt_prior', bbox_inches='tight')
plt.close()

plt.errorbar(grid_prior, spectral[:, -1],
             yerr=error_spectral[:, -1, -1], fmt='o')
plt.xlabel('prior precision in deg')
plt.ylabel(r'$\beta_s$')
plt.savefig(save_path+'beta_synch_wrt_prior', bbox_inches='tight')
plt.close()
