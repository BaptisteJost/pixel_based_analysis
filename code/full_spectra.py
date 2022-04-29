from config import *
import argparse
import IPython
import matplotlib.lines as mlines
from astropy import units as u
import copy
import numpy as np
from getdist.gaussian_mixtures import GaussianND
from getdist import plots, MCSamples
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("folder_end", help="end of the folder name")
args = parser.parse_args()

# save_path_ = save_path_.replace('4', '3')
save_path = save_path_ + args.folder_end + '/'
print()
print(save_path)
ell = np.arange(lmin, lmax+1)
WACAW = np.load(save_path + 'WACAW.npy')
VACAW = np.load(save_path + 'VACAW.npy')
WACAV = np.load(save_path + 'WACAV.npy')
tr_SigmaYY_cmb = np.load(save_path + 'tr_SigmaYY_cmb.npy')
Cl_noise_matrix = np.load(save_path + 'Cl_noise_matrix.npy')
tr_SigmaYY = np.load(save_path + 'tr_SigmaYY.npy')

plt.plot(ell, np.abs(WACAW[0, 0]), label='WACAW')
plt.plot(ell, np.abs(VACAW[0, 0]), label='|VACAW|')
plt.plot(ell, np.abs(WACAV[0, 0]), label='|WACAV|')
plt.plot(ell, Cl_noise_matrix[0, 0], '--', label='Cl_noise_matrix')
plt.plot(ell, np.abs(tr_SigmaYY_cmb[0, 0]), '--', linewidth=2, label='tr_SigmaYY_cmb')
plt.plot(ell, np.abs(tr_SigmaYY[0, 0]), '--', label='tr_SigmaYY tot')
plt.loglog()
plt.legend()
plt.title('EE')
plt.savefig(save_path + 'C_ell_spectra_EE.png', bbox_inches='tight', dpi=200)
plt.close()


plt.plot(ell, np.abs(WACAW[1, 1]), label='WACAW')
plt.plot(ell, np.abs(VACAW[1, 1]), label='|VACAW|')
plt.plot(ell, np.abs(WACAV[1, 1]), label='|WACAV|')
plt.plot(ell, Cl_noise_matrix[1, 1], '--', label='Cl_noise_matrix')
plt.plot(ell, np.abs(tr_SigmaYY_cmb[1, 1]), '--', linewidth=2, label='tr_SigmaYY_cmb')
plt.plot(ell, np.abs(tr_SigmaYY[1, 1]), '--', label='tr_SigmaYY tot')
plt.loglog()
plt.legend()
plt.title('BB')
plt.savefig(save_path + 'C_ell_spectra_BB.png', bbox_inches='tight', dpi=200)
plt.close()

plt.plot(ell, np.abs(WACAW[1, 0]), label='WACAW')
plt.plot(ell, np.abs(VACAW[1, 0]), label='|VACAW|')
plt.plot(ell, np.abs(WACAV[1, 0]), label='|WACAV|')
plt.plot(ell, Cl_noise_matrix[1, 0], '--', label='Cl_noise_matrix')
plt.plot(ell, np.abs(tr_SigmaYY_cmb[1, 0]), '--', linewidth=2, label='tr_SigmaYY_cmb')
plt.plot(ell, np.abs(tr_SigmaYY[1, 0]), '--', label='tr_SigmaYY tot')
plt.loglog()
plt.legend()
plt.title('EB')
plt.savefig(save_path + 'C_ell_spectra_EB.png', bbox_inches='tight', dpi=200)
plt.close()
