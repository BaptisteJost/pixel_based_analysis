import time
import IPython
import numpy as np
from astropy import units as u
import bjlib.V3calc as V3
import pixel_based_angle_estimation as pix
import copy
import bjlib.lib_project as lib
import residuals as res
import healpy as hp
import matplotlib.pyplot as plt


def dumb_likelihood_exploration(cosmo_params, Cl_fid, Cl_data, Cl_noise_matrix, ell, fsky):
    r = cosmo_params[0]

    beta = cosmo_params[1]*u.rad
    Cl_cmb_model = np.zeros([4, Cl_fid['EE'].shape[0]])
    Cl_cmb_model[1] = copy.deepcopy(Cl_fid['EE'])
    Cl_cmb_model[2] = copy.deepcopy(Cl_fid['BlBl'])*1 + copy.deepcopy(Cl_fid['BuBu']) * r

    Cl_cmb_rot = lib.cl_rotation(Cl_cmb_model.T, beta).T

    Cl_cmb_rot_matrix = np.zeros([2, 2, Cl_cmb_rot.shape[-1]])
    Cl_cmb_rot_matrix[0, 0] = copy.deepcopy(Cl_cmb_rot[1])
    Cl_cmb_rot_matrix[1, 1] = copy.deepcopy(Cl_cmb_rot[2])
    Cl_cmb_rot_matrix[1, 0] = copy.deepcopy(Cl_cmb_rot[4])
    Cl_cmb_rot_matrix[0, 1] = copy.deepcopy(Cl_cmb_rot[4])

    Cl_model_total = Cl_cmb_rot_matrix + Cl_noise_matrix
    # IPython.embed()

    likelihood = res.cosmo_likelihood_nodeprojection(
        Cl_model_total, Cl_data, ell, fsky)
    return likelihood


def dumb_likelihood_explorationBB(cosmo_params, Cl_fid, Cl_data, Cl_noise_matrix, ell, fsky):
    r = cosmo_params[0]

    beta = cosmo_params[1]*u.rad
    Cl_cmb_model = np.zeros([4, Cl_fid['EE'].shape[0]])
    Cl_cmb_model[1] = copy.deepcopy(Cl_fid['EE'])
    Cl_cmb_model[2] = copy.deepcopy(Cl_fid['BlBl'])*1 + copy.deepcopy(Cl_fid['BuBu']) * r

    Cl_cmb_rot = lib.cl_rotation(Cl_cmb_model.T, beta).T

    Cl_cmb_rot_matrix = np.zeros([2, 2, Cl_cmb_rot.shape[-1]])
    Cl_cmb_rot_matrix[0, 0] = copy.deepcopy(Cl_cmb_rot[1])
    Cl_cmb_rot_matrix[1, 1] = copy.deepcopy(Cl_cmb_rot[2])
    Cl_cmb_rot_matrix[1, 0] = copy.deepcopy(Cl_cmb_rot[4])
    Cl_cmb_rot_matrix[0, 1] = copy.deepcopy(Cl_cmb_rot[4])

    Cl_model_total = Cl_cmb_rot_matrix + Cl_noise_matrix
    # IPython.embed()

    likelihood = cosmo_likelihood_nodeprojectionBB(
        Cl_model_total[1, 1], Cl_data[1, 1], ell, fsky)
    return likelihood


def cosmo_likelihood_nodeprojectionBB(Cl_model_total, Cl_data, ell, fsky):
    inv_model = 1/Cl_model_total
    dof = (2 * ell + 1) * fsky
    dof_over_Cl = dof * inv_model

    first_term_ell = np.einsum('l,l->l', dof_over_Cl, Cl_data)

    # first_term = np.sum(np.trace(first_term_ell))
    first_term = np.sum(first_term_ell)

    # logdetC = np.sum(dof*np.log(np.abs(np.linalg.det(Cl_model_total.T))))
    logdetC = np.sum(dof*np.log(Cl_model_total))

    return first_term + logdetC


def main():
    INSTRU = 'SAT'
    if INSTRU == 'SAT':
        freq_number = 6
        fsky = 0.1
        lmin = 30
        lmax = 300
        nside = 512
        index_93 = 2

    if INSTRU == 'Planck':
        freq_number = 7
        fsky = 0.6
        lmin = 51
        lmax = 1500
        nside = 2048
    path_BB_local = '/home/baptiste/BBPipe'
    path_BB_NERSC = '/global/homes/j/jost/BBPipe'
    path_BB = path_BB_local

    sky_model = 'c1s0d0'
    sensitiviy_mode = 1
    one_over_f_mode = 1
    A_lens_true = 1

    r_true = 0.01
    beta_true = (0.35 * u.deg).to(u.rad)

    true_miscal_angles = np.array([0]*freq_number)*u.rad
    true_miscal_angles[index_93] = (0.28*u.deg).to(u.rad)
    # true_miscal_angles = (np.arange(0.1, 0.5, 0.4 / freq_number)*u.deg).to(u.rad)

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

    # fg_freq_maps_nomiscal = (data.mixing_matrix[:, 2:]).dot(data.signal[2:])
    # fg_93_nomiscal = np.zeros((3, fg_freq_maps_nomiscal.shape[-1]))
    # fg_93_nomiscal[1] = fg_freq_maps_nomiscal[2*index_93]*mask
    # fg_93_nomiscal[2] = fg_freq_maps_nomiscal[2*index_93+1]*mask
    # del fg_freq_maps_nomiscal
    # cl_fg93_nomiscal = hp.anafast(fg_93_nomiscal, lmax=lmax)[:, lmin:]/fsky

    fg_93 = np.zeros((3, fg_freq_maps_full.shape[-1]))
    fg_93[1] = fg_freq_maps_full[2*index_93]*mask
    fg_93[2] = fg_freq_maps_full[2*index_93+1]*mask

    del mask, fg_freq_maps_full
    cl_fg93 = hp.anafast(fg_93, lmax=lmax)[:, lmin:]/fsky
    del fg_93
    ell = np.arange(lmin, lmax+1)

    V3_results = V3.so_V3_SA_noise(sensitiviy_mode, one_over_f_mode,
                                   SAC_yrs_LF=1, f_sky=fsky, ell_max=lmax+1, beam_corrected=True)
    noise_nl = V3_results[1]
    ell_noise = V3_results[0]
    noise_93 = noise_nl[index_93, lmin-2:]
    # IPython.embed()
    ps_planck = copy.deepcopy(res.get_Cl_cmbBB(Alens=A_lens_true, r=r_true, path_BB=path_BB))
    Cl_fid = {}
    Cl_fid['BB'] = res.get_Cl_cmbBB(Alens=A_lens_true, r=r_true,
                                    path_BB=path_BB)[2][lmin:lmax+1]
    Cl_fid['BuBu'] = res.get_Cl_cmbBB(Alens=0.0, r=1.0, path_BB=path_BB)[2][lmin:lmax+1]
    Cl_fid['BlBl'] = res.get_Cl_cmbBB(Alens=1.0, r=0.0, path_BB=path_BB)[2][lmin:lmax+1]
    Cl_fid['EE'] = ps_planck[1, lmin:lmax+1]

    Cl_noise_matrix = np.zeros([2, 2, noise_93.shape[0]])
    Cl_noise_matrix[0, 0] = noise_93
    Cl_noise_matrix[1, 1] = noise_93

    Cl_fg_matrix = np.zeros([2, 2, cl_fg93.shape[-1]])
    Cl_fg_matrix[0, 0] = cl_fg93[1]
    Cl_fg_matrix[1, 1] = cl_fg93[2]
    Cl_fg_matrix[1, 0] = cl_fg93[4]
    Cl_fg_matrix[0, 1] = cl_fg93[4]

    Cl_cmb_model = np.zeros([4, Cl_fid['EE'].shape[0]])
    Cl_cmb_model[1] = copy.deepcopy(Cl_fid['EE'])
    Cl_cmb_model[2] = copy.deepcopy(Cl_fid['BlBl'])*1 + copy.deepcopy(Cl_fid['BuBu']) * r_true

    # Cl_cmb_rot = lib.cl_rotation(copy.deepcopy(Cl_cmb_model.T), beta_true).T
    Cl_cmb_rot = lib.cl_rotation(copy.deepcopy(Cl_cmb_model.T),
                                 beta_true+true_miscal_angles[index_93]).T

    Cl_cmb_rot_matrix = np.zeros([2, 2, Cl_cmb_rot.shape[-1]])
    Cl_cmb_rot_matrix[0, 0] = copy.deepcopy(Cl_cmb_rot[1])
    Cl_cmb_rot_matrix[1, 1] = copy.deepcopy(Cl_cmb_rot[2])
    Cl_cmb_rot_matrix[1, 0] = copy.deepcopy(Cl_cmb_rot[4])
    Cl_cmb_rot_matrix[0, 1] = copy.deepcopy(Cl_cmb_rot[4])
    #
    # Cl_cmb_rot_matrix2 = np.zeros([2, 2, Cl_cmb_rot2.shape[-1]])
    # Cl_cmb_rot_matrix2[0, 0] = copy.deepcopy(Cl_cmb_rot2[1])
    # Cl_cmb_rot_matrix2[1, 1] = copy.deepcopy(Cl_cmb_rot2[2])
    # Cl_cmb_rot_matrix2[1, 0] = copy.deepcopy(Cl_cmb_rot2[4])
    # Cl_cmb_rot_matrix2[0, 1] = copy.deepcopy(Cl_cmb_rot2[4])

    # miscal_93 = data.miscal_matrix[index_93*2:index_93*2 + 2, index_93*2:index_93*2 + 2]
    # test_cmb_rot_miscal = np.einsum('ij,jkl,km->iml', miscal_93.T,
    #                                 copy.deepcopy(Cl_cmb_rot_matrix), miscal_93)
    Cl_data_matrix = copy.deepcopy(Cl_cmb_rot_matrix) + \
        copy.deepcopy(Cl_fg_matrix) + copy.deepcopy(Cl_noise_matrix)

    a = dumb_likelihood_exploration([0, 0], Cl_fid, Cl_data_matrix, Cl_noise_matrix, ell, fsky)

    r_range = np.linspace(0.12, 0.17, 100)
    beta_range = np.linspace(0.0087, 0.014, 100)
    like_grid = np.empty((len(r_range), len(beta_range)))
    r, b = np.meshgrid(r_range, beta_range)

    for i in range(len(r_range)):
        for j in range(len(beta_range)):
            like_grid[i, j] = dumb_likelihood_exploration(
                [r_range[i], beta_range[j]], Cl_fid, Cl_data_matrix, Cl_noise_matrix, ell, fsky)

    IPython.embed()

    np.save('../results_and_data/potatoe/'+'chi2_fg_nomiscal.npy', like_grid)
    np.save('../results_and_data/potatoe/'+'r_range_fg_nomiscal.npy', r)
    np.save('../results_and_data/potatoe/'+'beta_range_fg_nomiscal.npy', b)

    like = np.exp((-like_grid+np.min(like_grid))/2)
    chi2_all_levels = np.min(like_grid) + np.array([6.17, 2.3, 0])
    sigma_levels = np.exp((-chi2_all_levels + np.min(like_grid))/2)

    plt.contour(r, (b*u.rad).to(u.deg), like.T, levels=sigma_levels)
    plt.hlines(beta_true.to(u.deg).value, r_range.min(),
               r_range.max(), colors='black', linestyles='--')
    plt.vlines(r_true, (beta_range*u.rad).to(u.deg).value.min(), (beta_range *
                                                                  u.rad).to(u.deg).value.max(), colors='black', linestyles='--')
    plt.colorbar()
    plt.show()
    exit()


######################################################
# MAIN CALL
if __name__ == "__main__":
    main()
