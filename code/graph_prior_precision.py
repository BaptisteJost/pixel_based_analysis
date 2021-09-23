from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.cm import get_cmap
import matplotlib.cm as cm
import matplotlib.colors as mpcolors
from cycler import cycler
import matplotlib.animation as animation
import copy
import residuals as res
import IPython
import numpy as np
from astropy import units as u
import matplotlib.pyplot as plt
import argparse
from datetime import date
import os
import configparser
import bjlib.lib_project as lib


parser = argparse.ArgumentParser()
parser.add_argument('--folder', type=str)
args = parser.parse_args()

path = '/global/homes/j/jost/these/pixel_based_analysis/results_and_data/prior_gridding/'
# try:
# except KeyError:
folder_name = args.folder

try:
    save_path = path+folder_name+'/'
except TypeError:
    folder_name = date.today().strftime('%Y%m%d')+'_' + os.environ["SLURM_JOB_ID"]
    save_path = path+folder_name+'/'

config = configparser.ConfigParser()
config.read(save_path+'example.ini')
A_lens_true = float(config['DEFAULT']['a_lens_true'])
r_true = float(config['DEFAULT']['r_true'])
beta_true = float(config['DEFAULT']['beta_true'])*u.rad
path_BB = config['DEFAULT']['path BB']
lmin = int(config['DEFAULT']['lmin'])
lmax = int(config['DEFAULT']['lmax'])
prior_start = float(config['DEFAULT']['prior_start'])
prior_end = float(config['DEFAULT']['prior_end'])


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
plt.ylabel(r'$\beta$ in deg')
plt.savefig(save_path+'beta_wrt_prior', bbox_inches='tight')
plt.close()

plt.scatter(grid_prior, error_r)
plt.xlabel('prior precision in deg')
plt.ylabel(r'$\sigma (r)$')
plt.savefig(save_path+'sigma_r_wrt_prior', bbox_inches='tight')
plt.close()

plt.scatter(grid_prior, (error_b*u.rad).to(u.deg).value)
plt.xlabel('prior precision in deg')
plt.ylabel(r'$\sigma (\beta)$ in deg')
plt.savefig(save_path+'sigma_beta_wrt_prior', bbox_inches='tight')
plt.close()


for i in range(6):
    plt.errorbar(grid_prior, (spectral[:, i]*u.rad).to(u.deg).value,
                 yerr=(error_spectral[:, i, i]*u.rad).to(u.deg).value, fmt='o')
    plt.xlabel('prior precision in deg')
    plt.ylabel('miscal angle at {}Ghz in deg'.format(frequencies[i]))
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

try:
    os.mkdir(save_path+'spectra_fig')
except FileExistsError:
    pass

ps_planck = copy.deepcopy(res.get_Cl_cmbBB(Alens=A_lens_true, r=r_true, path_BB=path_BB))
Cl_fid = {}
Cl_fid['BB'] = res.get_Cl_cmbBB(Alens=A_lens_true, r=r_true,
                                path_BB=path_BB)[2][lmin:lmax+1]
Cl_fid['BuBu'] = res.get_Cl_cmbBB(Alens=0.0, r=1.0, path_BB=path_BB)[2][lmin:lmax+1]
Cl_fid['BlBl'] = res.get_Cl_cmbBB(Alens=1.0, r=0.0, path_BB=path_BB)[2][lmin:lmax+1]
Cl_fid['EE'] = ps_planck[1, lmin:lmax+1]
Cl_cmb_input = np.zeros([4, Cl_fid['EE'].shape[0]])
Cl_cmb_input[1] = copy.deepcopy(Cl_fid['EE'])
Cl_cmb_input[2] = copy.deepcopy(Cl_fid['BB'])
Cl_cmb_input_rot = lib.cl_rotation(Cl_cmb_input.T, beta_true).T


ell = np.arange(lmin, lmax+1)

fig, ax = plt.subplots(2, 2, figsize=(25, 25))
label_string_start_prim = r'input $C_{\ell}^{'
label_string_end_prim = 'primordial r={:1.1e} Rotation {:1.1e}'.format(
    r_true, beta_true.to(u.deg))

ax[0, 0].set_title(r'$C_{\ell}^{EE}$', fontsize=20)
ax[1, 0].set_title(r'$C_{\ell}^{EB}$', fontsize=20)
ax[1, 1].set_title(r'$C_{\ell}^{BB}$', fontsize=20)
for ax_ in ax.reshape(-1):
    ax_.grid(b=True, linestyle=':')
    ax_.set_xlabel(r'$\ell$', fontsize=20)
    ax_.set_ylabel(r'$C_\ell \frac{\ell(\ell+1)}{2\pi} $', fontsize=20)
    ax_.set_xscale('log')
    ax_.set_yscale('log')

colors = get_cmap('viridis')(np.linspace(0, 1/2, 80))
# colors = plt.cm.YlOrBr_r(np.linspace(0, 1, len(grid_prior)))
# plt.gca().set_prop_cycle(cycler('color', colors))

for i in range(len(grid_prior)):
    r = cosmo[i, 0]
    beta = cosmo[i, 1]*u.rad
    # beta = (grid_prior[i]*u.deg).to(u.rad)
    Cl_cmb_model = np.zeros([4, Cl_fid['EE'].shape[0]])
    Cl_cmb_model[1] = copy.deepcopy(Cl_fid['EE'])
    Cl_cmb_model[2] = copy.deepcopy(Cl_fid['BlBl'])*1 + copy.deepcopy(Cl_fid['BuBu']) * r

    Cl_cmb_rot = lib.cl_rotation(Cl_cmb_model.T, beta).T

    label_string_start_fit = r'fit $C_{\ell}^{'
    label_string_end_fit = 'primordial r={:1.1e} Rotation {:1.1e}'.format(r, beta.to(u.deg))

    # color_line = next(ax[0, 0]._get_lines.prop_cycler)['color']

    ax[0, 0].plot(ell, Cl_cmb_rot[1]*ell*(ell+1)/(2*np.pi), linewidth=1.0, color=colors[i])
    ax[1, 1].plot(ell, Cl_cmb_rot[2]*ell*(ell+1)/(2*np.pi), linewidth=1.0, color=colors[i])
    ax[1, 0].plot(ell, Cl_cmb_rot[4]*ell*(ell+1)/(2*np.pi), linewidth=1.0, color=colors[i])

# colors = plt.cm.YlOrBr_r(np.linspace(0, 1, len(grid_prior)))

norm = mpcolors.Normalize(vmin=prior_start, vmax=prior_end)
newcmp = ListedColormap(colors)
cmap = cm.ScalarMappable(norm=norm, cmap=newcmp)
cmap.set_array([])
cbar = plt.colorbar(cmap)
cbar.set_label('prior precision in degrees', fontsize=20)

ax[0, 0].plot(ell, Cl_cmb_input_rot[1]*ell*(ell+1)/(2*np.pi),
              label=label_string_start_prim+'EE}$'+label_string_end_prim, color='orange', linewidth=2.0)
ax[1, 1].plot(ell, Cl_cmb_input_rot[2]*ell*(ell+1)/(2*np.pi),
              label=label_string_start_prim+'BB}$'+label_string_end_prim, color='orange', linewidth=2.0)
ax[1, 0].plot(ell, Cl_cmb_input_rot[4]*ell*(ell+1)/(2*np.pi),
              label=label_string_start_prim+'EB}$'+label_string_end_prim, color='orange', linewidth=2.0)
ax[0, 0].legend(borderaxespad=0., loc='upper left')

plt.savefig(save_path+'spectra_fig'+'/'+'test_spectra_all', bbox_inches='tight')
plt.close(fig)
