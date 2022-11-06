import pixel_based_angle_estimation as pix
import healpy as hp
import numpy as np
from astropy import units as u
import copy
import bjlib.lib_project as lib
import residuals as res
from cycler import cycler
import matplotlib.pyplot as plt
import matplotlib as mpl
import IPython
from potatoe_plot import ticks_format
from matplotlib import ticker

r_true = 0.01
A_lens_true = 1

fsky = 0.1
# lmin = 2
# lmax = 2000
lmin = 2
lmax = 1000
path_BB_local = '/home/baptiste/BBPipe'
path_BB = path_BB_local
IPython.embed()

plot_fg = 1

ps_planck = copy.deepcopy(res.get_Cl_cmbBB(Alens=A_lens_true, r=r_true, path_BB=path_BB))

Cl_fid = {}
Cl_fid['BB'] = res.get_Cl_cmbBB(Alens=A_lens_true, r=r_true,
                                path_BB=path_BB)[2][lmin:lmax+1]

Cl_fid['BuBu'] = res.get_Cl_cmbBB(Alens=0.0, r=1.0, path_BB=path_BB)[2][lmin:lmax+1]
Cl_fid['BlBl'] = res.get_Cl_cmbBB(Alens=1.0, r=0.0, path_BB=path_BB)[2][lmin:lmax+1]
Cl_fid['EE'] = ps_planck[1, lmin:lmax+1]

Cl_cmb_model = np.zeros([4, Cl_fid['EE'].shape[0]])
Cl_cmb_model[1] = copy.deepcopy(Cl_fid['EE'])
Cl_cmb_model[2] = copy.deepcopy(Cl_fid['BlBl'])*1 + copy.deepcopy(Cl_fid['BuBu']) * r_true

beta_range = np.linspace(0, 1, 100)*u.deg
ell = np.arange(lmin, lmax+1)

if plot_fg:
    print('bla')
    import healpy as hp
    import pixel_based_angle_estimation as pix

    INSTRU = 'SAT'
    freq_number = 6
    fsky = 0.1
    lmin_SO = 30
    lmax_SO = 300

    nside = 512
    index_93 = 2
    sky_model = 'c1s0d0'
    sensitiviy_mode = 1
    one_over_f_mode = 1
    A_lens_true = 1

    beta_true = (0.0 * u.deg).to(u.rad)

    true_miscal_angles = np.array([0]*freq_number)*u.rad
    freq_by_instru = [1]*freq_number

    data, model_data = pix.data_and_model_quick(
        miscal_angles_array=true_miscal_angles, bir_angle=beta_true,
        frequencies_by_instrument_array=freq_by_instru, nside=nside,
        sky_model=sky_model, sensitiviy_mode=sensitiviy_mode,
        one_over_f_mode=one_over_f_mode, instrument=INSTRU)

    data.get_mask(path_BB)
    mask = data.mask
    mask[(mask != 0) * (mask != 1)] = 0

    fg_freq_maps_full = data.miscal_matrix.dot(data.mixing_matrix)[
        :, 2:].dot(data.signal[2:])

    fg_93 = np.zeros((3, fg_freq_maps_full.shape[-1]))
    fg_93[1] = fg_freq_maps_full[2*index_93]*mask
    fg_93[2] = fg_freq_maps_full[2*index_93+1]*mask

    del mask, fg_freq_maps_full
    cl_fg93 = hp.anafast(fg_93, lmax=lmax)[:, lmin:]/fsky
    del fg_93
    # ell = np.arange(lmin, lmax+1)

fig, ax = plt.subplots(1, 3, gridspec_kw={'width_ratios': [1, 1, 1]}, figsize=(32, 8))

minval = 0
maxval = 0.7
colors = plt.cm.YlOrBr_r(np.linspace(minval, maxval, 100))
newcmp = mpl.colors.ListedColormap(colors)
plt.gca().set_prop_cycle(cycler('color', colors))

# i = 0
for beta, i in zip(beta_range[1:][::-1], range(len(beta_range)-1)):
    Cl_cmb_rot = lib.cl_rotation(Cl_cmb_model.T, beta).T

    w = 2
    ax[0].plot(ell, ell*(ell+1) * Cl_cmb_rot[1]/(2*np.pi), linewidth=w, color=colors[-(1+i)])
    ax[1].plot(ell, ell*(ell+1) * Cl_cmb_rot[2]/(2*np.pi), linewidth=w, color=colors[-(1+i)])
    ax[2].plot(ell, ell*(ell+1) * np.abs(Cl_cmb_rot[4]) /
               (2*np.pi), linewidth=w, color=colors[-(1+i)])

beta = 0*u.deg
Cl_cmb_rot = lib.cl_rotation(Cl_cmb_model.T, beta).T
ax[0].plot(ell, ell*(ell+1) * Cl_fid['EE']/(2*np.pi), linewidth=3,
           color='purple', label='Lensed CMB spectra')
ax[1].plot(ell, ell*(ell+1) * copy.deepcopy(Cl_fid['BuBu']) * r_true/(2*np.pi),
           linewidth=3,
           color='black', label=r'Primordial B modes, $r=0.01$')
ax[1].plot(ell, ell*(ell+1) * Cl_cmb_rot[2]/(2*np.pi), linewidth=3, color='purple')
ax[2].plot(ell, ell*(ell+1) * np.abs(Cl_cmb_rot[4]) /
           (2*np.pi), linewidth=3, color='purple')

norm = mpl.colors.Normalize(vmin=0, vmax=1)
cmap = mpl.cm.ScalarMappable(norm=norm, cmap=newcmp)  # YlOrBr_r
cmap.set_array([])
cbar = plt.colorbar(cmap)
cbar.set_label('rotation angle in degrees', fontsize=25)
cbar.ax.tick_params(labelsize=20)

if plot_fg:
    alpha_fg = 0.8
    w_fg = 2
    color_fg = 'maroon'
    ax[0].plot(ell, ell*(ell+1) * cl_fg93[1]/(2*np.pi), linewidth=w_fg,
               color=color_fg, alpha=alpha_fg, label='Foregrounds spectra at 93GHz')
    ax[1].plot(ell, ell*(ell+1) * cl_fg93[2]/(2*np.pi),
               linewidth=w_fg, color=color_fg, alpha=alpha_fg)
    ax[2].plot(ell, ell*(ell+1) * cl_fg93[4] /
               (2*np.pi), linewidth=2, color=color_fg, alpha=0.8)
    ax[2].plot(ell, ell*(ell+1) * -cl_fg93[4] /
               (2*np.pi), linewidth=2, color=color_fg, linestyle='--', alpha=0.8)


list_spectra = ['EE', 'BB', '|EB|']
subs = [1.0, 2.0, 5.0]

for i in range(3):
    ax[i].set_xlabel(r'Multipole moment $\ell$', fontsize=25)

    ax[i].set_xscale('log')
    ax[i].set_xlim(lmin, lmax)
    ax[i].set_yscale('log')
    ax[i].set_title('XY = '+list_spectra[i], fontsize=25)
    ylims = ax[i].get_ylim()
    ax[i].fill_between([lmin_SO, lmax_SO], ylims[0], ylims[1], color='grey', alpha=0.3)
    quantile = 0.95
    y_coord = np.exp(np.log(ylims)[0] + (np.log(ylims)[1] - np.log(ylims)[0])*quantile)
    ax[i].text(50, y_coord, 'SO SAT range',  fontsize=15, fontweight='bold')

    ax[i].set_ylim(ylims[0], ylims[1])
    ax[i].grid(b=True, linestyle=':')

    # if i == 2:
    #     ax[i].set_ylabel(
    #         r'$\frac{\ell(\ell+1)}{2\pi} |C_{\ell}^{' + list_spectra[i] + r'}| [\mu K^2]$', fontsize=25)
    #     ax[i].set_yscale('log')
    # else:
    #     ax[i].set_ylabel(
    #         r'$\frac{\ell(\ell+1)}{2\pi} C_{\ell}^{' + list_spectra[i] + r'} [\mu K^2]$', fontsize=25)
    #     ax[i].set_yscale('log')
    ax[i].xaxis.set_major_locator(ticker.LogLocator(subs=subs))
    ax[i].xaxis.set_minor_locator(ticker.LogLocator(subs=subs))
    ax[i].xaxis.set_minor_formatter(ticker.FuncFormatter(ticks_format))
    ax[i].xaxis.set_major_formatter(ticker.FuncFormatter(ticks_format))
    ax[i].tick_params(axis='both', which='major', labelsize=15)
    ax[i].tick_params(axis='both', which='minor', labelsize=8)
# ax[2].set_yscale('symlog')

ax[0].set_ylabel(
    r'$\frac{\ell(\ell+1)}{2\pi} C_{\ell}^{XY} [\mu K^2]$', fontsize=25)

artist_EE, label_EE = ax[0].get_legend_handles_labels()
artist_BB, label_BB = ax[1].get_legend_handles_labels()
artist_EE.append(artist_BB[0])
label_EE.append(label_BB[0])
ax[0].legend(artist_EE, label_EE, fontsize=15, loc='upper left')


plt.subplots_adjust(wspace=0.08)
plt.savefig('birefringence_spectra_fg_3panelsV3.pdf', dpi=200)  # , bbox_inches='tight')
plt.savefig('birefringence_spectra_fg_3panelsV3.png', dpi=200)  # , bbox_inches='tight')
plt.show()
# IPython.embed()

'''=====================NEW PLOT CMB====================='''

r_true = 0.0
A_lens_true = 1

fsky = 0.1
# lmin = 2
# lmax = 2000
lmin = 2
lmax = 1000
lmin_SO = 30
lmax_SO = 300
path_BB_local = '/home/baptiste/BBPipe'
path_BB = path_BB_local

plot_fg = 0

ps_planck = copy.deepcopy(res.get_Cl_cmbBB(Alens=A_lens_true, r=r_true, path_BB=path_BB))

Cl_fid_00 = {}
Cl_fid_00['BB'] = res.get_Cl_cmbBB(Alens=A_lens_true, r=r_true,
                                   path_BB=path_BB)[2][lmin:lmax+1]

Cl_fid_00['BuBu'] = res.get_Cl_cmbBB(Alens=0.0, r=1.0, path_BB=path_BB)[2][lmin:lmax+1]
Cl_fid_00['BlBl'] = res.get_Cl_cmbBB(Alens=1.0, r=0.0, path_BB=path_BB)[2][lmin:lmax+1]
Cl_fid_00['EE'] = ps_planck[1, lmin:lmax+1]

Cl_cmb_model = np.zeros([4, Cl_fid_00['EE'].shape[0]])
Cl_cmb_model[1] = copy.deepcopy(Cl_fid_00['EE'])
Cl_cmb_model[2] = copy.deepcopy(Cl_fid_00['BlBl'])*1 + copy.deepcopy(Cl_fid_00['BuBu']) * r_true


r01 = 0.01
beta35 = 0.35*u.deg
Cl_fid_11 = {}
Cl_fid_11['BB'] = res.get_Cl_cmbBB(Alens=A_lens_true, r=r01,
                                   path_BB=path_BB)[2][lmin:lmax+1]

Cl_fid_11['BuBu'] = res.get_Cl_cmbBB(Alens=0.0, r=1.0, path_BB=path_BB)[2][lmin:lmax+1]
Cl_fid_11['BlBl'] = res.get_Cl_cmbBB(Alens=1.0, r=0.0, path_BB=path_BB)[2][lmin:lmax+1]
Cl_fid_11['EE'] = ps_planck[1, lmin:lmax+1]

Cl_cmb_model11 = np.zeros([4, Cl_fid_11['EE'].shape[0]])
Cl_cmb_model11[1] = copy.deepcopy(Cl_fid_11['EE'])
Cl_cmb_model11[2] = copy.deepcopy(Cl_fid_11['BlBl'])*1 + copy.deepcopy(Cl_fid_11['BuBu']) * r01

Cl_cmb_rot = lib.cl_rotation(Cl_cmb_model.T, beta35).T

ell = np.arange(lmin, lmax+1)

fig, ax = plt.subplots(1, 3, gridspec_kw={'width_ratios': [1, 1, 1]}, figsize=(32, 8))

minval = 0
maxval = 0.7
colors = plt.cm.YlOrBr_r(np.linspace(minval, maxval, 100))
newcmp = mpl.colors.ListedColormap(colors)
plt.gca().set_prop_cycle(cycler('color', colors))

ax[0].plot(ell, ell*(ell+1) * Cl_cmb_model[1]/(2*np.pi), linewidth=3,
           color='darkorange', label=r'$r=0.0$ and $\beta_b = 0^\circ$')
ax[1].plot(ell, ell*(ell+1) * Cl_cmb_model[2]/(2*np.pi),
           linewidth=3,  color='darkorange')

colour_11 = 'darkred'
ax[0].plot(ell, ell*(ell+1) * Cl_cmb_rot[1]/(2*np.pi), '--', linewidth=3,
           color=colour_11, label=r'$r=0.01$ and $\beta_b = 0.35^\circ$')
ax[1].plot(ell, ell*(ell+1) * Cl_cmb_rot[2]/(2*np.pi), '--', linewidth=3, color=colour_11)
ax[2].plot(ell, ell*(ell+1) * np.abs(Cl_cmb_rot[4]) /
           (2*np.pi), '--', linewidth=3, color=colour_11)

list_spectra = ['EE', 'BB', '|EB|']
subs = [1.0, 2.0, 5.0]

for i in range(3):
    ax[i].set_xlabel(r'Multipole moment $\ell$', fontsize=25)

    ax[i].set_xscale('log')
    ax[i].set_xlim(lmin, lmax)
    ax[i].set_yscale('log')
    ax[i].set_title('XY = '+list_spectra[i], fontsize=25)
    ylims = ax[i].get_ylim()
    ax[i].fill_between([lmin_SO, lmax_SO], ylims[0], ylims[1], color='grey', alpha=0.3)
    quantile = 0.95
    y_coord = np.exp(np.log(ylims)[0] + (np.log(ylims)[1] - np.log(ylims)[0])*quantile)
    ax[i].text(50, y_coord, 'SO SAT range',  fontsize=15, fontweight='bold')

    ax[i].set_ylim(ylims[0], ylims[1])
    ax[i].grid(b=True, linestyle=':')

    ax[i].xaxis.set_major_locator(ticker.LogLocator(subs=subs))
    ax[i].xaxis.set_minor_locator(ticker.LogLocator(subs=subs))
    ax[i].xaxis.set_minor_formatter(ticker.FuncFormatter(ticks_format))
    ax[i].xaxis.set_major_formatter(ticker.FuncFormatter(ticks_format))
    ax[i].tick_params(axis='both', which='major', labelsize=15)
    ax[i].tick_params(axis='both', which='minor', labelsize=8)

ax[0].set_ylabel(
    r'$\frac{\ell(\ell+1)}{2\pi} C_{\ell}^{XY} [\mu K^2]$', fontsize=25)

ax[0].legend(fontsize=15, loc='upper left')

plt.subplots_adjust(wspace=0.08)
plt.savefig('CMB_spectra_input.pdf', dpi=300)
plt.show()


'''=====================NEW PLOT FG====================='''

INSTRU = 'SAT'
freq_number = 6
fsky = 0.1
lmin_SO = 30
lmax_SO = 300

nside = 512
index_93 = 2
sky_model = 'c1s0d0'
sensitiviy_mode = 1
one_over_f_mode = 1

beta_true = (0.0 * u.deg).to(u.rad)

true_miscal_angles = np.array([0]*freq_number)*u.rad
freq_by_instru = [1]*freq_number

data_s0d0, model_data_s0d0 = pix.data_and_model_quick(
    miscal_angles_array=true_miscal_angles, bir_angle=beta_true,
    frequencies_by_instrument_array=freq_by_instru, nside=nside,
    sky_model='c1s0d0', sensitiviy_mode=sensitiviy_mode,
    one_over_f_mode=one_over_f_mode, instrument=INSTRU)

data_s0d0.get_mask(path_BB)
mask = data_s0d0.mask
mask[(mask != 0) * (mask != 1)] = 0

fg_freq_maps_full = data_s0d0.miscal_matrix.dot(data_s0d0.mixing_matrix)[
    :, 2:].dot(data_s0d0.signal[2:])

fg_93 = np.zeros((3, fg_freq_maps_full.shape[-1]))
fg_93[1] = fg_freq_maps_full[2*index_93]*mask
fg_93[2] = fg_freq_maps_full[2*index_93+1]*mask

del fg_freq_maps_full
cl_fg93_s0d0 = hp.anafast(fg_93, lmax=lmax)[:, lmin:]/fsky
del fg_93

data_s1d1, model_data_s1d1 = pix.data_and_model_quick(
    miscal_angles_array=true_miscal_angles, bir_angle=beta_true,
    frequencies_by_instrument_array=freq_by_instru, nside=nside,
    sky_model='c1s1d1', sensitiviy_mode=sensitiviy_mode,
    one_over_f_mode=one_over_f_mode, instrument=INSTRU)

fg_freq_maps_full = data_s1d1.miscal_matrix.dot(data_s1d1.mixing_matrix)[
    :, 2:].dot(data_s1d1.signal[2:])

fg_93 = np.zeros((3, fg_freq_maps_full.shape[-1]))
fg_93[1] = fg_freq_maps_full[2*index_93]*mask
fg_93[2] = fg_freq_maps_full[2*index_93+1]*mask

del fg_freq_maps_full
cl_fg93_s1d1 = hp.anafast(fg_93, lmax=lmax)[:, lmin:]/fsky
del fg_93

data_s3d7, model_data_s3d7 = pix.data_and_model_quick(
    miscal_angles_array=true_miscal_angles, bir_angle=beta_true,
    frequencies_by_instrument_array=freq_by_instru, nside=nside,
    sky_model='c1s3d7', sensitiviy_mode=sensitiviy_mode,
    one_over_f_mode=one_over_f_mode, instrument=INSTRU)

fg_freq_maps_full = data_s3d7.miscal_matrix.dot(data_s3d7.mixing_matrix)[
    :, 2:].dot(data_s3d7.signal[2:])

fg_93 = np.zeros((3, fg_freq_maps_full.shape[-1]))
fg_93[1] = fg_freq_maps_full[2*index_93]*mask
fg_93[2] = fg_freq_maps_full[2*index_93+1]*mask

del mask, fg_freq_maps_full
cl_fg93_s3d7 = hp.anafast(fg_93, lmax=lmax)[:, lmin:]/fsky
del fg_93


alpha_fg = 1
w_fg = 2
color_s0d0 = 'maroon'
color_s1d1 = 'darkorange'
color_s3d7 = 'purple'

linestyle_s1d1 = '--'

fig, ax = plt.subplots(1, 3, gridspec_kw={'width_ratios': [1, 1, 1]}, figsize=(32, 8))
ax[0].plot(ell, ell*(ell+1) * cl_fg93_s0d0[1]/(2*np.pi), linewidth=w_fg,
           color=color_s0d0, alpha=alpha_fg, label='s0d0 Foregrounds spectra at 93GHz')
ax[1].plot(ell, ell*(ell+1) * cl_fg93_s0d0[2]/(2*np.pi),
           linewidth=w_fg, color=color_s0d0, alpha=alpha_fg)
ax[2].plot(ell, ell*(ell+1) * cl_fg93_s0d0[4] /
           (2*np.pi), linewidth=2, color=color_s0d0, alpha=0.8)
# ax[2].plot(ell, ell*(ell+1) * -cl_fg93_s0d0[4] /
#            (2*np.pi), linewidth=2, color=color_s0d0, linestyle='--', alpha=0.8)

ax[0].plot(ell, ell*(ell+1) * cl_fg93_s1d1[1]/(2*np.pi), linewidth=w_fg,
           color=color_s1d1, alpha=alpha_fg, label='s1d1 Foregrounds spectra at 93GHz', linestyle=linestyle_s1d1)
ax[1].plot(ell, ell*(ell+1) * cl_fg93_s1d1[2]/(2*np.pi),
           linewidth=w_fg, color=color_s1d1, alpha=alpha_fg, linestyle=linestyle_s1d1)
ax[2].plot(ell, ell*(ell+1) * cl_fg93_s1d1[4] /
           (2*np.pi), linewidth=2, color=color_s1d1, alpha=0.8, linestyle=linestyle_s1d1)
# ax[2].plot(ell, ell*(ell+1) * -cl_fg93_s1d1[4] /
#            (2*np.pi), linewidth=2, color=color_s1d1, linestyle='--', alpha=0.8)

ax[0].plot(ell, ell*(ell+1) * cl_fg93_s3d7[1]/(2*np.pi), linewidth=w_fg,
           color=color_s3d7, alpha=alpha_fg, label='s3d7 Foregrounds spectra at 93GHz')
ax[1].plot(ell, ell*(ell+1) * cl_fg93_s3d7[2]/(2*np.pi),
           linewidth=w_fg, color=color_s3d7, alpha=alpha_fg)
ax[2].plot(ell, ell*(ell+1) * cl_fg93_s3d7[4] /
           (2*np.pi), linewidth=2, color=color_s3d7, alpha=0.8)
# ax[2].plot(ell, ell*(ell+1) * -cl_fg93_s3d7[4] /
#            (2*np.pi), linewidth=2, color=color_s3d7, linestyle='--', alpha=0.8)

for i in range(3):
    ax[i].set_xlabel(r'Multipole moment $\ell$', fontsize=25)

    ax[i].set_xscale('log')
    ax[i].set_xlim(lmin, lmax)
    if i != 2:
        ax[i].set_yscale('log')
    ax[i].set_title('XY = '+list_spectra[i], fontsize=25)
    ylims = ax[i].get_ylim()
    ax[i].fill_between([lmin_SO, lmax_SO], ylims[0], ylims[1], color='grey', alpha=0.3)
    quantile = 0.95
    y_coord = np.exp(np.log(ylims)[0] + (np.log(ylims)[1] - np.log(ylims)[0])*quantile)
    ax[i].text(50, y_coord, 'SO SAT range',  fontsize=15, fontweight='bold')

    ax[i].set_ylim(ylims[0], ylims[1])
    ax[i].grid(b=True, linestyle=':')

    ax[i].xaxis.set_major_locator(ticker.LogLocator(subs=subs))
    ax[i].xaxis.set_minor_locator(ticker.LogLocator(subs=subs))
    ax[i].xaxis.set_minor_formatter(ticker.FuncFormatter(ticks_format))
    ax[i].xaxis.set_major_formatter(ticker.FuncFormatter(ticks_format))
    ax[i].tick_params(axis='both', which='major', labelsize=15)
    ax[i].tick_params(axis='both', which='minor', labelsize=8)

ax[0].set_ylabel(
    r'$\frac{\ell(\ell+1)}{2\pi} C_{\ell}^{XY} [\mu K^2]$', fontsize=25)

ax[0].legend(fontsize=15, loc='lower left')

plt.subplots_adjust(wspace=0.15)
plt.savefig('fg_spectra_input.pdf', dpi=300)
plt.close()
