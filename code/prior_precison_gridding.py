import time
import IPython
import numpy as np
from astropy import units as u
import bjlib.V3calc as V3
import fisher_pixel as fshp
import pixel_based_angle_estimation as pix
import copy
import bjlib.lib_project as lib
from pixel_based_angle_estimation import get_chi_squared_local
from scipy.optimize import minimize
import bjlib.class_faraday as cf
import residuals as res
from mpi4py import MPI
from datetime import date
import os


def parameters2file(path, nside, lmax, lmin, fsky, sky_model, sensitiviy_mode,
                    one_over_f_mode, A_lens_true, r_true, beta_true,
                    true_miscal_angles, prior, prior_indices, nsim, prior_start,
                    prior_end, initmodel_miscal, angle_array_start, cosmo_params, path_BB):
    file_string = '''
    nside = '''+str(nside)+'''
    lmax = '''+str(lmax)+'''
    lmin = '''+str(lmin)+'''
    fsky = '''+str(fsky)+'''
    sky_model = ''' + sky_model + '''
    path to mask = '''+path_BB+'/test_mapbased_param/mask_04000.fits'+''' WARNING: hardcoded
    sensitiviy_mode = '''+str(sensitiviy_mode)+'''
    one_over_f_mode = '''+str(one_over_f_mode)+'''
    A_lens_true = '''+str(A_lens_true)+'''
    r_true = '''+str(r_true)+'''
    beta_true = '''+str(beta_true)+'''
    true_miscal_angles = '''+str(true_miscal_angles)+'''

    prior = '''+str(prior)+'''

    prior_indices = '''+str(prior_indices)+'''
    nsim = '''+str(nsim)+'''
    prior_start = '''+str(prior_start)+''' deg
    prior_end = '''+str(prior_end)+''' deg

    Model initialisaion before minimisation (shouldn't have any impact)
    initmodel_miscal = '''+str(initmodel_miscal)+'''

    Initial values of parameter vectors before initialisation
    angle_array_start = '''+str(angle_array_start)+'''
    cosmo_params = '''+str(cosmo_params)+'''
    '''

    text_file = open(path+'config_'+path[-18:-1]+'.txt', "w+")
    text_file.write(file_string)
    text_file.close()
    return 0


def main():
    comm = MPI.COMM_WORLD
    mpi_rank = MPI.COMM_WORLD.Get_rank()
    size_mpi = comm.Get_size()
    print(mpi_rank, size_mpi)
    root = 0

    NERSC = 1
    path_BB_local = '/home/baptiste/BBPipe'
    path_BB_NERSC = '/global/homes/j/jost/BBPipe'
    if NERSC:
        path_BB = path_BB_NERSC
        pixel_path = '/global/u2/j/jost/these/pixel_based_analysis/'

    else:
        path_BB = path_BB_local
        pixel_path = '/home/baptiste/Documents/these/pixel_based_analysis/'

    if comm.rank == root:
        folder_name = date.today().strftime('%Y%m%d')+'_' + os.environ["SLURM_JOB_ID"]
        # str(np.random.rand())[2:6]
    else:
        folder_name = None

    folder_name = comm.bcast(folder_name, root=root)
    save_path = pixel_path + 'results_and_data/prior_gridding/'+folder_name+'/'
    if comm.rank == root:
        if os.path.exists(save_path):
            print('ERROR: path already exists')
            exit()
        else:
            os.mkdir(save_path)

    nside = 512
    lmax = 300
    lmin = 30
    fsky = 0.1
    sky_model = 'c1s0d0'
    sensitiviy_mode = 0
    one_over_f_mode = 0
    A_lens_true = 1
    # sensitivity_mode
    #     0: threshold,
    #     1: baseline,
    #     2: goal
    # one_over_f_mode
    #     0: pessimistic
    #     1: optimistic
    #     2: none

    r_true = 0.01
    beta_true = (0.35 * u.deg).to(u.rad)

    true_miscal_angles = np.array([0]*6)*u.rad
    # true_miscal_angles = np.arange(0.1, 0.5, 0.4/6)*u.rad
    prior = True
    prior_indices = []
    if prior:
        prior_indices = [2, 3]
    # prior_str = '{:1.1e}rad'.format(prior_precision)
    nsim = 1000

    prior_start = 0.01
    prior_end = 1

    initmodel_miscal = np.array([0]*6)*u.rad
    angle_array_start = np.array([0., 0., 0., 0., 0., 0., 2, -2.5])
    cosmo_params = [0.0, 0.0]

    if comm.rank == root:
        parameters2file(save_path, nside, lmax, lmin, fsky, sky_model, sensitiviy_mode,
                        one_over_f_mode, A_lens_true, r_true, beta_true,
                        true_miscal_angles, prior, prior_indices, nsim, prior_start,
                        prior_end, initmodel_miscal, angle_array_start, cosmo_params, path_BB)

    '''====================================================================='''

    data, model_data = pix.data_and_model_quick(miscal_angles_array=true_miscal_angles, bir_angle=beta_true,
                                                frequencies_array=V3.so_V3_SA_bands(),
                                                frequencies_by_instrument_array=[1, 1, 1, 1, 1, 1], nside=nside,
                                                sky_model=sky_model, sensitiviy_mode=sensitiviy_mode, one_over_f_mode=one_over_f_mode)

    model = pix.get_model(miscal_angles_array=initmodel_miscal, bir_angle=beta_true,
                          frequencies_array=V3.so_V3_SA_bands(),
                          frequencies_by_instrument_array=[1, 1, 1, 1, 1, 1],
                          nside=nside, spectral_params=[1.59, 20, -3],
                          sky_model=sky_model, sensitiviy_mode=sensitiviy_mode, one_over_f_mode=one_over_f_mode)

    '''===========================getting data==========================='''

    S_cmb_name = 'S_cmb_n{}_s{}_r{:1}_b{:1.1e}'.format(nside, nsim, r_true, beta_true.value).replace(
        '.', 'p') + '.npy'
    # print(S_cmb_name)
    S_cmb = np.load(S_cmb_name)

    ASAt = model_data.mix_effectiv[:, :2].dot(S_cmb).dot(model_data.mix_effectiv[:, :2].T)

    fg_freq_maps_full = data.miscal_matrix.dot(data.mixing_matrix)[
        :, 2:].dot(data.signal[2:])
    ddt_fg = np.einsum('ik...,...kj->ijk', fg_freq_maps_full, fg_freq_maps_full.T)

    data.get_mask(path_BB)
    mask = data.mask
    mask[(mask != 0) * (mask != 1)] = 0

    ddt_fg *= mask
    fg_freq_maps = fg_freq_maps_full*mask
    del fg_freq_maps_full

    n_obspix = np.sum(mask == 1)
    del mask
    F = np.sum(ddt_fg, axis=-1)/n_obspix
    data_model = n_obspix*(F + ASAt + model.noise_covariance)

    prior_precision_grid = np.linspace(prior_start, prior_end, size_mpi*2)
    prior_per_rank = 2

    fisher_cosmo_matrix_list = []
    fisher_spectral_list = []
    results_cosmp_list = []

    control_grid = []
    cosmo_results = []
    spectral_results = []

    for prior_precision_deg in prior_precision_grid[prior_per_rank*mpi_rank:prior_per_rank*mpi_rank+prior_per_rank]:
        prior_precision = (prior_precision_deg * u.deg).to(u.rad).value
        print('prior_precision', prior_precision)
        angle_prior = []
        if prior:
            for d in range(6):
                angle_prior.append([true_miscal_angles.value[d], prior_precision, int(d)])
            angle_prior = np.array(angle_prior[prior_indices[0]: prior_indices[-1]])

        params = ['miscal']*6
        params.append('spectral')
        params.append('spectral')

        results_min = minimize(get_chi_squared_local, angle_array_start, args=(
            data_model, model, prior, [], angle_prior, False, True, 1, True, params),
            tol=1e-18, options={'maxiter': 1000}, jac=fshp.spectral_first_deriv, method='SLSQP',
            bounds=((-np.pi/4, np.pi/4), (-np.pi/4, np.pi/4), (-np.pi/4, np.pi/4), (-np.pi/4, np.pi/4), (-np.pi/4, np.pi/4), (-np.pi/4, np.pi/4), (0.5, 2.5), (-5, -1)))

        print('')
        print(results_min)
        print('')

        model_results = pix.get_model(results_min.x[: 6], bir_angle=beta_true,
                                      frequencies_array=V3.so_V3_SA_bands(), frequencies_by_instrument_array=[1, 1, 1, 1, 1, 1], nside=nside,
                                      spectral_params=[results_min.x[-2], 20, results_min.x[-1]],
                                      sky_model=sky_model, sensitiviy_mode=sensitiviy_mode, one_over_f_mode=one_over_f_mode)

        print('results - spectral_true = ', results_min.x[:6] - true_miscal_angles.value)
        print('results - spectral_true = ', results_min.x[-2] - 1.59)
        print('results - spectral_true = ', results_min.x[-1] + 3)

        prior_matrix = np.zeros([len(params), len(params)])
        if prior:
            for i in range(prior_indices[0], prior_indices[-1]):
                prior_matrix[i, i] += 1/prior_precision**2

        ps_planck = copy.deepcopy(res.get_Cl_cmbBB(Alens=A_lens_true, r=r_true, path_BB=path_BB))
        spectra_true = lib.cl_rotation(ps_planck.T, beta_true).T

        diff_list_res = res.get_diff_list(model_results, params)
        diff_diff_list_res = res.get_diff_diff_list(model_results, params)
        fisher_matrix_spectral = fshp.fisher_new(data_model, model_results,
                                                 diff_list_res, diff_diff_list_res, params)
        fisher_matrix_prior_spectral = fisher_matrix_spectral + prior_matrix
        sigma_spectral = np.linalg.inv(fisher_matrix_prior_spectral)

        stat, bias, var, Cl, Cl_cmb, Cl_residuals_matrix, ell, WA_cmb = res.get_residuals(
            model_results, fg_freq_maps, sigma_spectral, lmin, lmax, fsky, params,
            cmb_spectra=spectra_true, true_A_cmb=model_data.mix_effectiv[:, :2])

        '''=================Init and cosmo likelihood estimation================'''
        Cl_fid = {}
        Cl_fid['BB'] = res.get_Cl_cmbBB(Alens=A_lens_true, r=r_true,
                                        path_BB=path_BB)[2][lmin:lmax+1]
        Cl_fid['BuBu'] = res.get_Cl_cmbBB(Alens=0.0, r=1.0, path_BB=path_BB)[2][lmin:lmax+1]
        Cl_fid['BlBl'] = res.get_Cl_cmbBB(Alens=1.0, r=0.0, path_BB=path_BB)[2][lmin:lmax+1]
        Cl_fid['EE'] = ps_planck[1, lmin:lmax+1]

        Cl_noise, ell_noise = res.get_noise_Cl(
            model_results.mix_effectiv, lmax+1, fsky, sensitiviy_mode, one_over_f_mode)
        Cl_noise = Cl_noise[lmin-2:]
        ell_noise = ell_noise[lmin-2:]
        Cl_noise_matrix = np.zeros([2, 2, Cl_noise.shape[0]])
        Cl_noise_matrix[0, 0] = Cl_noise
        Cl_noise_matrix[1, 1] = Cl_noise

        tr_SigmaYY = np.einsum('ij,jimnl->mnl', sigma_spectral, Cl_residuals_matrix['YY'])
        Cl_data = Cl_residuals_matrix['yy'] + Cl_residuals_matrix['zy'] + \
            Cl_residuals_matrix['yz'] + tr_SigmaYY + Cl_noise_matrix

        results_cosmp = minimize(res.likelihood_exploration, cosmo_params, args=(
            Cl_fid, Cl_data, Cl_noise_matrix, tr_SigmaYY, ell, fsky),
            bounds=((-0.01, 0.1), (-np.pi/4, np.pi/4)), tol=1e-18,
            method='L-BFGS-B')

        print('')
        print('results - true cosmo = ', results_cosmp.x - np.array([r_true, beta_true.value]))
        print('')
        print(results_cosmp)
        print('')

        '''==================Fisher cosmo likelihood estimation================='''

        Cl_cmb_model_fish = np.zeros([4, Cl_fid['EE'].shape[0]])
        Cl_cmb_model_fish[1] = copy.deepcopy(Cl_fid['EE'])
        Cl_cmb_model_fish[2] = copy.deepcopy(Cl_fid['BlBl'])*A_lens_true + \
            copy.deepcopy(Cl_fid['BuBu']) * results_cosmp.x[0]

        Cl_cmb_model_fish_r = np.zeros([4, Cl_fid['EE'].shape[0]])
        Cl_cmb_model_fish_r[2] = copy.deepcopy(Cl_fid['BuBu']) * 1

        deriv1 = cf.power_spectra_obj(lib.cl_rotation_derivative(
            Cl_cmb_model_fish.T, results_cosmp.x[1]*u.rad), ell)
        deriv_matrix_beta = cf.power_spectra_obj(np.array(
            [[deriv1.spectra[:, 1], deriv1.spectra[:, 4]],
             [deriv1.spectra[:, 4], deriv1.spectra[:, 2]]]).T, deriv1.ell)

        deriv_matrix_r = cf.power_spectra_obj(lib.get_dr_cov_bir_EB(
            Cl_cmb_model_fish_r.T, results_cosmp.x[1]*u.rad).T, ell)

        Cl_data_spectra = cf.power_spectra_obj(Cl_data.T, ell)

        fish_beta = cf.fisher_pws(Cl_data_spectra, deriv_matrix_beta, fsky)
        fish_r = cf.fisher_pws(Cl_data_spectra, deriv_matrix_r, fsky)
        fish_beta_r = cf.fisher_pws(Cl_data_spectra, deriv_matrix_beta, fsky, deriv2=deriv_matrix_r)
        fisher_cosmo_matrix = np.array([[fish_r, fish_beta_r],
                                        [fish_beta_r, fish_beta]])
        fisher_cosmo_matrix_list.append(fisher_cosmo_matrix)
        results_cosmp_list.append(results_cosmp.x)

        control_grid.append(prior_precision_deg)
        cosmo_results.append(results_cosmp.x)
        spectral_results.append(results_min.x)
        fisher_spectral_list.append(fisher_matrix_prior_spectral)
        print('')
        print('')

    fisher_cosmo_matrix_array = np.array(fisher_cosmo_matrix_list)
    fisher_spectral_array = np.array(fisher_spectral_list)
    results_cosmp_list = np.array(results_cosmp_list)
    control_grid = np.array(control_grid)
    cosmo_results = np.array(cosmo_results)
    spectral_results = np.array(spectral_results)

    fisher_cosmo_matrix_mpi = None
    fisher_spectal_mpi = None
    prior_precision_grid_control = None
    cosmo_results_mpi = None
    spectral_results_mpi = None
    if comm.rank == 0:
        fisher_cosmo_matrix_mpi = np.empty([size_mpi, prior_per_rank, 2, 2])
        fisher_spectal_mpi = np.empty([size_mpi, prior_per_rank, 8, 8])
        prior_precision_grid_control = np.empty([size_mpi, prior_per_rank])
        cosmo_results_mpi = np.empty([size_mpi, prior_per_rank, 2])
        spectral_results_mpi = np.empty([size_mpi, prior_per_rank, 8])

    comm.Gather(fisher_cosmo_matrix_array, fisher_cosmo_matrix_mpi, root)
    comm.Gather(fisher_spectral_array, fisher_spectal_mpi, root)
    comm.Gather(control_grid, prior_precision_grid_control, root)
    comm.Gather(cosmo_results, cosmo_results_mpi, root)
    comm.Gather(spectral_results, spectral_results_mpi, root)

    if comm.rank == 0:
        fisher_cosmo_matrix_mpi_reshape = fisher_cosmo_matrix_mpi.reshape(
            -1, *fisher_cosmo_matrix_mpi.shape[2:])
        fisher_spectal_mpi_reshape = fisher_spectal_mpi.reshape(
            -1, *fisher_spectal_mpi.shape[2:])
        prior_precision_grid_control_reshape = prior_precision_grid_control.reshape(
            -1, *prior_precision_grid_control.shape[2:])
        cosmo_results_mpi_reshape = cosmo_results_mpi.reshape(-1, *cosmo_results_mpi.shape[2:])
        spectral_results_mpi_reshape = spectral_results_mpi.reshape(
            -1, *spectral_results_mpi.shape[2:])

        np.save(save_path+'fisher_matrix_grid.npy', fisher_cosmo_matrix_mpi_reshape)
        np.save(save_path+'fisher_spectral_grid.npy', fisher_spectal_mpi_reshape)
        np.save(save_path+'grid_prior_precision.npy', prior_precision_grid)
        np.save(save_path+'grid_prior_precision_control.npy', prior_precision_grid_control_reshape)
        np.save(save_path+'cosmo_results_grid_prior.npy', cosmo_results_mpi_reshape)
        np.save(save_path+'spectral_results_grid_prior.npy', spectral_results_mpi_reshape)

    exit()


######################################################
# MAIN CALL
if __name__ == "__main__":
    main()
