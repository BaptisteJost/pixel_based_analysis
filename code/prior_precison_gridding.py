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
import configparser
import tracemalloc
import emcee
import matplotlib.pyplot as plt


def parameters2file(path, nside, lmax, lmin, fsky, sky_model, sensitiviy_mode,
                    one_over_f_mode, A_lens_true, r_true, beta_true,
                    true_miscal_angles, prior, prior_indices, nsim, prior_start,
                    prior_end, initmodel_miscal, angle_array_start, cosmo_params, path_BB,
                    INSTRU, method_spectral, method_cosmo, bounds_cosmo, jac_cosmo, jac_spectal, comment):
    config = configparser.ConfigParser()
    config['DEFAULT'] = {}
    config['DEFAULT']['INSTRU'] = INSTRU
    config['DEFAULT']['nside'] = str(nside)
    config['DEFAULT']['lmax'] = str(lmax)
    config['DEFAULT']['lmin'] = str(lmin)
    config['DEFAULT']['fsky'] = str(fsky)
    config['DEFAULT']['sky_model'] = sky_model
    config['DEFAULT']['path BB'] = path_BB
    config['DEFAULT']['mask file'] = path_BB + \
        '/test_mapbased_param/mask_04000.fits'  # WARNING: hardcoded
    if INSTRU == 'SAT':
        config['DEFAULT']['sensitiviy_mode'] = str(sensitiviy_mode)
        config['DEFAULT']['one_over_f_mode'] = str(one_over_f_mode)
    config['DEFAULT']['A_lens_true'] = str(A_lens_true)
    config['DEFAULT']['r_true'] = str(r_true)
    config['DEFAULT']['beta_true'] = str(beta_true.value)
    config['DEFAULT']['true_miscal_angles'] = str(true_miscal_angles.value)

    config['DEFAULT']['prior'] = str(prior)

    config['DEFAULT']['prior_indices'] = str(prior_indices)
    config['DEFAULT']['nsim'] = str(nsim)
    config['DEFAULT']['prior_start'] = str(prior_start)
    config['DEFAULT']['prior_end'] = str(prior_end)

    # Model initialisaion before minimisation (shouldn't have any impact)
    config['DEFAULT']['initmodel_miscal'] = str(initmodel_miscal.value)

    # Initial values of parameter vectors before initialisation
    config['DEFAULT']['angle_array_start'] = str(angle_array_start)
    config['DEFAULT']['cosmo_params'] = str(cosmo_params)
    config['DEFAULT']['method_spectral'] = method_spectral
    config['DEFAULT']['method_cosmo'] = method_cosmo
    config['DEFAULT']['bounds_cosmo'] = str(bounds_cosmo)
    config['DEFAULT']['jac_cosmo'] = str(jac_cosmo)
    config['DEFAULT']['jac_spectal'] = str(jac_spectal)
    config['DEFAULT']['comment'] = str(comment)

    with open(path+'example.ini', 'w+') as configfile:
        config.write(configfile)

    return 0


def main():
    comm = MPI.COMM_WORLD
    mpi_rank = MPI.COMM_WORLD.Get_rank()
    size_mpi = comm.Get_size()
    print(mpi_rank, size_mpi)
    root = 0

    tracemalloc.start()
    # start_init = time.time()

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

    INSTRU = 'SAT'
    if INSTRU == 'SAT':
        freq_number = 6
        fsky = 0.1
        lmin = 30
        lmax = 300
        nside = 512

    if INSTRU == 'Planck':
        freq_number = 7
        fsky = 0.6
        lmin = 51
        lmax = 1500
        nside = 2048

    sky_model = 'c1s0d0'
    sensitiviy_mode = 1
    one_over_f_mode = 1
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

    true_miscal_angles = (np.array([0.28]*freq_number)*u.deg).to(u.rad)
    # true_miscal_angles = (np.arange(0.1, 0.5, 0.4 / freq_number)*u.deg).to(u.rad)
    prior = True
    prior_indices = []
    if prior:
        prior_indices = [2, 3]
    # prior_str = '{:1.1e}rad'.format(prior_precision)
    nsim = 1000

    prior_start = 0.01
    prior_end = 5

    initmodel_miscal = np.array([0]*freq_number)*u.rad
    freq_by_instru = [1]*freq_number

    angle_array_start_list = [0]*freq_number
    angle_array_start_list.append(2)
    angle_array_start_list.append(-2.5)
    angle_array_start = np.array(angle_array_start_list)

    params = ['miscal']*freq_number
    params.append('spectral')
    params.append('spectral')

    miscal_bounds = ((None, None),)*freq_number
    spectral_bounds = ((None, None), (None, None))
    # miscal_bounds = ((-np.pi/4, np.pi/4),)*freq_number
    # spectral_bounds = ((0.5, 2.5), (-5, -1))
    bounds = miscal_bounds + spectral_bounds

    cosmo_params = [0.03, 0.04]
    bounds_cosmo = ((-0.01, 0.1), (-np.pi/4, np.pi/4))
    # bounds_cosmo = ((-0.01, 0.1), (None, None))
    # bounds_cosmo = ((None, None), (None, None))
    method_list = ['CG', 'BFGS', 'Newton-CG', 'L-BFGS-B', 'TNC', 'SLSQP',
                   'TNC', 'TNC']
    method_spectral = 'BFGS'
    machine = np.finfo(float).eps  # * 1e6
    spectral_options = {'maxiter': 1000, 'gtol': machine}
    # 'ftol': machine, 'xtol': machine, 'tol': machine}
    method_cosmo = 'L-BFGS-B'
    # jac_cosmo = None
    jac_spectal = fshp.spectral_first_deriv
    # jac_spectal = None
    jac_cosmo = res.jac_cosmo

    comment = 'test bfgs bounds in function gtol'
    # IPython.embed()

    # current, peak = tracemalloc.get_traced_memory()
    # print('time_init = ', time.time() - start_init)
    # print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")

    if comm.rank == root:
        parameters2file(save_path, nside, lmax, lmin, fsky, sky_model, sensitiviy_mode,
                        one_over_f_mode, A_lens_true, r_true, beta_true,
                        true_miscal_angles, prior, prior_indices, nsim, prior_start,
                        prior_end, initmodel_miscal, angle_array_start, cosmo_params,
                        path_BB, INSTRU, method_spectral, method_cosmo, bounds_cosmo, jac_cosmo, jac_spectal, comment)

    '''====================================================================='''

    # start_data = time.time()
    data, model_data = pix.data_and_model_quick(
        miscal_angles_array=true_miscal_angles, bir_angle=beta_true,
        frequencies_by_instrument_array=freq_by_instru, nside=nside,
        sky_model=sky_model, sensitiviy_mode=sensitiviy_mode,
        one_over_f_mode=one_over_f_mode, instrument=INSTRU)

    # current, peak = tracemalloc.get_traced_memory()
    # print('time data = ', time.time() - start_data)
    # print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")

    # start_model = time.time()
    model = pix.get_model(
        miscal_angles_array=initmodel_miscal, bir_angle=beta_true,
        frequencies_by_instrument_array=freq_by_instru,
        nside=nside, spectral_params=[1.59, 20, -3],
        sky_model=sky_model, sensitiviy_mode=sensitiviy_mode,
        one_over_f_mode=one_over_f_mode, instrument=INSTRU)

    # current, peak = tracemalloc.get_traced_memory()
    # print('time model = ', time.time() - start_model)
    # print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")

    '''===========================getting data==========================='''
    # start_freqmaps = time.time()
    S_cmb_name = 'data/S_cmb_n{}_s{}_r{:1}_b{:1.1e}'.format(nside, nsim, r_true, beta_true.value).replace(
        '.', 'p') + '.npy'
    # print(S_cmb_name)
    S_cmb = np.load(S_cmb_name)

    ASAt = model_data.mix_effectiv[:, :2].dot(S_cmb).dot(model_data.mix_effectiv[:, :2].T)

    fg_freq_maps_full = data.miscal_matrix.dot(data.mixing_matrix)[
        :, 2:].dot(data.signal[2:])
    del data.signal

    # current, peak = tracemalloc.get_traced_memory()
    # print('time_freqmaps = ', time.time() - start_freqmaps)
    # print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")

    # start_mask = time.time()
    data.get_mask(path_BB)
    mask = data.mask
    mask[(mask != 0) * (mask != 1)] = 0

    # ddt_fg *= mask
    fg_freq_maps = fg_freq_maps_full*mask
    del fg_freq_maps_full

    n_obspix = np.sum(mask == 1)
    del mask

    # current, peak = tracemalloc.get_traced_memory()
    # print('time_mask = ', time.time() - start_mask)
    # print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")

    # start_F = time.time()
    shape_ddt = (fg_freq_maps.shape[0], fg_freq_maps.shape[0])
    F = np.empty(shape_ddt)
    for f1 in range(fg_freq_maps.shape[0]):
        for f2 in range(f1, fg_freq_maps.shape[0]):
            F[f1, f2] = np.einsum('i,i->', fg_freq_maps[f1], fg_freq_maps[f2])
            F[f2, f1] = F[f1, f2]
    F /= n_obspix

    # current, peak = tracemalloc.get_traced_memory()
    # print('time_F = ', time.time() - start_F)
    # print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")

    data_model = n_obspix*(F + ASAt + model.noise_covariance)
    true_A_cmb = copy.deepcopy(model_data.mix_effectiv[:, :2])

    # print('diff F =', np.max(np.abs(F2-F)))

    prior_per_rank = 2
    # prior_precision_grid = np.linspace(prior_start, prior_end, size_mpi*prior_per_rank)
    # prior_precision_grid = np.array([0.01, 0.1, 1, 2])  # , 5, 10])
    prior_precision_grid = np.array([0.01, 0.05, 0.1, 0.2, 0.34266667,
                                     0.67533333, 1.008, 1.34066667, 1.67333333, 2.006,
                                     2.33866667, 3.004, 3.33666667, 4.002, 4.66733333,
                                     5.])

    fisher_cosmo_matrix_list = []
    fisher_spectral_list = []
    results_cosmp_list = []

    control_grid = []
    cosmo_results = []
    spectral_results = []

    for prior_precision_deg in prior_precision_grid[prior_per_rank*mpi_rank:prior_per_rank*mpi_rank+prior_per_rank]:
        # for method_spectral in method_list[prior_per_rank*mpi_rank:prior_per_rank*mpi_rank+prior_per_rank]:
        # start_init_min = time.time()
        prior_precision = (prior_precision_deg * u.deg).to(u.rad).value
        # prior_precision_deg = prior_precision_grid[0]
        # prior_precision = (prior_precision_deg * u.deg).to(u.rad).value
        print('prior_precision', prior_precision)
        print('method_spectral', method_spectral)
        angle_prior = []
        if prior:
            for d in range(freq_number):
                angle_prior.append([true_miscal_angles.value[d], prior_precision, int(d)])
            angle_prior = np.array(angle_prior[prior_indices[0]: prior_indices[-1]])
        print('angle_prior = ', angle_prior)

        prior_matrix = np.zeros([len(params), len(params)])
        if prior:
            for i in range(prior_indices[0], prior_indices[-1]):
                prior_matrix[i, i] += 1/prior_precision**2
        # angle_array_start[angle_prior[:, 2].astype(int)] = np.random.normal(
        #     angle_prior[:, 0], angle_prior[:, 1])

        # current, peak = tracemalloc.get_traced_memory()
        # print('time_init_min = ', time.time() - start_init_min)
        # print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")

        # start_min_spectra = time.time()

        ndim = freq_number + 2
        walker_per_dim = 2
        nwalkers = walker_per_dim*ndim
        nsteps = 21000
        discard_num = 1000

        loc = [angle_prior[0, 0]]*freq_number
        loc.append(1.59)
        loc.append(-3)
        loc = np.array(loc)
        loc[angle_prior[:, -1].astype(int).tolist()] = angle_prior[:, 0]
        size = [prior_precision*10]*freq_number
        size.append(0.5)
        size.append(1)
        size = np.array(size)
        size[angle_prior[:, -1].astype(int).tolist()] = angle_prior[:, 1]
        angle_array_start = np.random.normal(loc, size)
        angle_array_start[:6] = np.sign(
            angle_array_start[:6])*(np.abs(angle_array_start[:6]) % (np.pi/4))
        # print('')
        # print('angle_array_start = ', angle_array_start)
        # print('')

        # min_init = minimize(get_chi_squared_local, angle_array_start, args=(
        #     data_model, model, prior, [], angle_prior, False, True, 1, True, params),
        #     tol=1e-18, options=spectral_options, method=method_spectral,
        #     bounds=bounds, jac=jac_spectal)
        # print('')
        # print('min_init = ', min_init)
        # print('')
        #
        # model_results_step1 = pix.get_model(
        #     min_init.x[: freq_number], bir_angle=beta_true,
        #     frequencies_by_instrument_array=freq_by_instru, nside=nside,
        #     spectral_params=[min_init.x[-2], 20, min_init.x[-1]],
        #     sky_model=sky_model, sensitiviy_mode=sensitiviy_mode,
        #     one_over_f_mode=one_over_f_mode, instrument=INSTRU)
        #
        # diff_list_step1 = res.get_diff_list(model_results_step1, params)
        # diff_diff_list_step1 = res.get_diff_diff_list(model_results_step1, params)
        # fisher_matrix_spectral_step1 = fshp.fisher_new(data_model, model_results_step1,
        #                                                diff_list_step1, diff_diff_list_step1, params)
        # fisher_matrix_prior_spectral_step1 = fisher_matrix_spectral_step1 + prior_matrix
        # inv_fisher_spectral_step1 = np.linalg.inv(fisher_matrix_prior_spectral_step1)
        # sigma_spectral_step1 = np.sqrt(np.diag(inv_fisher_spectral_step1))
        #
        # angle_array_start = np.random.normal(
        #     min_init.x, sigma_spectral_step1, (nwalkers, len(size)))
        # angle_array_start[:, :6] = np.sign(
        #     angle_array_start[:, :6])*(np.abs(angle_array_start[:, :6]) % (np.pi/4))
        #
        # sampler = emcee.EnsembleSampler(
        #     nwalkers, ndim, get_chi_squared_local, args=[
        #         data_model, model, prior, [], angle_prior, False, True, 1, False, params],
        #     pool=None)
        # sampler.reset()
        # sampler.run_mcmc(angle_array_start, nsteps, progress=False)
        # flat_samples = sampler.get_chain(discard=discard_num, flat=True)
        # try:
        #     os.mkdir(save_path+'MCMC_spectral')
        # except FileExistsError:
        #     pass
        # np.save(save_path+'MCMC_spectral/' +
        #         'MCMC_spectral_prior{}'.format(prior_precision_deg)+method_spectral+'.npy', flat_samples)
        # mean_mcmc = np.mean(flat_samples, axis=0)
        # chi2_spectra_mcmc = get_chi_squared_local(mean_mcmc, data_model,
        #                                           model, prior, [], angle_prior, False, True, 1, True, params)
        #
        # class results():
        #     def __init__(self, x, fun):
        #         self.x = x
        #         self.fun = fun
        #
        # results_min = results(x=mean_mcmc, fun=chi2_spectra_mcmc)
        #
        # print('')
        # print('mean_mcmc = ', mean_mcmc)
        # print('')
        print('angle_array_start = ', angle_array_start)
        results_min = minimize(get_chi_squared_local, angle_array_start, args=(
            data_model, model, prior, [], angle_prior, False, True, 1, True, params),
            tol=1e-18, options=spectral_options, method=method_spectral,
            bounds=bounds, jac=jac_spectal)
        results_min.x = np.append(true_miscal_angles.value, [1.59, -3])
        # current, peak = tracemalloc.get_traced_memory()
        # print('time_min_spectra = ', time.time() - start_min_spectra)
        # print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")

        print('')
        print(results_min)
        print('')
        print('method_spectral', method_spectral)

        print('true jac = ', fshp.spectral_first_deriv(np.append(true_miscal_angles.value, [1.59, -3]), data_model,
                                                       model, prior, [], angle_prior, False, True, 1, True, params))
        print('results_min.fun = ', results_min.fun)
        chi2_spectra_true = get_chi_squared_local(np.append(true_miscal_angles.value, [1.59, -3]), data_model,
                                                  model, prior, [], angle_prior, False, True, 1, True, params)
        print('results true = ', chi2_spectra_true)
        # print('true < min ? = ', chi2_spectra_true <= min_init.fun)
        # print('true < min ? = ', chi2_spectra_true <= chi2_spectra_mcmc)
        print('true < min ? = ', chi2_spectra_true <= results_min.fun)
        # results_min.x = min_init.x

        # start_model_result = time.time()
        model_results = pix.get_model(
            results_min.x[: freq_number], bir_angle=beta_true,
            frequencies_by_instrument_array=freq_by_instru, nside=nside,
            spectral_params=[results_min.x[-2], 20, results_min.x[-1]],
            sky_model=sky_model, sensitiviy_mode=sensitiviy_mode,
            one_over_f_mode=one_over_f_mode, instrument=INSTRU)

        # current, peak = tracemalloc.get_traced_memory()
        # print('time_model_result = ', time.time() - start_model_result)
        # print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")

        print('prior_precision', prior_precision)
        # print('results min init - spectral_true = ',
        #       ((min_init.x[:freq_number] - true_miscal_angles.value)*u.rad).to(u.deg))
        # print('results mcmc - spectral_true = ',
        #       ((mean_mcmc[:freq_number] - true_miscal_angles.value)*u.rad).to(u.deg))
        print('results - spectral_true = ',
              ((results_min.x[:freq_number] - true_miscal_angles.value)*u.rad).to(u.deg))
        print('results - spectral_true = ', results_min.x[-2] - 1.59)
        print('results - spectral_true = ', results_min.x[-1] + 3)

        # try:
        #     os.mkdir(save_path+'gridding_spectral')
        # except FileExistsError:
        #     pass
        # alpha_range = np.linspace(results_min.x[-3]-0.5 *
        #                           sigma_spectral_step1[-3], results_min.x[-3]+0.5*sigma_spectral_step1[-3], 100)
        #
        # grid_spectral_alpha = []
        # true_spectral = np.append(true_miscal_angles.value, [1.59, -3])
        # for alpha in alpha_range:
        #     spectral_alpha = copy.deepcopy(true_spectral)
        #     spectral_alpha[-3] = alpha
        #     grid_spectral_alpha.append(get_chi_squared_local(
        #         spectral_alpha, data_model,
        #         model, prior, [], angle_prior, False, True, 1, True, params))
        # plt.close()
        # plt.plot(alpha_range, grid_spectral_alpha)
        # interval = max(grid_spectral_alpha) - min(grid_spectral_alpha)
        # plt.vlines(true_miscal_angles.value[-1], min(grid_spectral_alpha) - 0.05*interval, max(
        #     grid_spectral_alpha), linestyles='--', colors='k', label='true')
        # plt.vlines(min_init.x[-3], min(grid_spectral_alpha), max(
        #     grid_spectral_alpha), linestyles='--', colors='red', label='min ini')
        # plt.vlines(mean_mcmc[-3], min(grid_spectral_alpha), max(
        #     grid_spectral_alpha), linestyles='--', colors='blue', label='mcmc')
        # plt.vlines(results_min.x[-3], min(grid_spectral_alpha), max(
        #     grid_spectral_alpha), linestyles='--', colors='orange', label='min results')
        # plt.legend()
        # plt.savefig(save_path+'gridding_spectral/' +
        #             'grid_spectral{}'.format(prior_precision_deg)+method_spectral+'.png')
        # plt.close()

        # start_init_res = time.time()

        ps_planck = copy.deepcopy(res.get_Cl_cmbBB(Alens=A_lens_true, r=r_true, path_BB=path_BB))
        spectra_true = lib.cl_rotation(ps_planck.T, beta_true).T

        diff_list_res = res.get_diff_list(model_results, params)
        diff_diff_list_res = res.get_diff_diff_list(model_results, params)
        fisher_matrix_spectral = fshp.fisher_new(data_model, model_results,
                                                 diff_list_res, diff_diff_list_res, params)
        fisher_matrix_prior_spectral = fisher_matrix_spectral + prior_matrix
        sigma_spectral = np.linalg.inv(fisher_matrix_prior_spectral)

        # current, peak = tracemalloc.get_traced_memory()
        # print('time_init_res = ', time.time() - start_init_res)
        # print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")

        # start_res = time.time()
        sampleMCMC = 0
        if sampleMCMC:
            # print('sigma_spectral = ', sigma_spectral)
            if np.any(np.diag(sigma_spectral) <= 0):
                print('NEGATIVE elements in Fisher !')
                sigma_spectral_step1 = np.sqrt(np.abs(np.diag(sigma_spectral)))
            else:
                sigma_spectral_step1 = np.sqrt(np.diag(sigma_spectral))

            angle_array_start = np.random.normal(
                results_min.x, sigma_spectral_step1, (nwalkers, len(size)))
            angle_array_start[:, :6] = np.sign(
                angle_array_start[:, :6])*(np.abs(angle_array_start[:, :6]) % (np.pi/4))

            sampler = emcee.EnsembleSampler(
                nwalkers, ndim, get_chi_squared_local, args=[
                    data_model, model, prior, [], angle_prior, False, True, 1, False, params],
                pool=None)
            sampler.reset()
            sampler.run_mcmc(angle_array_start, nsteps, progress=False)
            flat_samples = sampler.get_chain(discard=discard_num, flat=True)
            raw_samples = sampler.get_chain()
            try:
                os.mkdir(save_path+'MCMC_spectral')
            except FileExistsError:
                pass
            np.save(save_path+'MCMC_spectral/' +
                    'MCMC_spectral_prior{}'.format(prior_precision_deg)+method_spectral+'.npy', flat_samples)
            np.save(save_path+'MCMC_spectral/' +
                    'MCMC_spectral_prior{}'.format(prior_precision_deg)+method_spectral+'_raw.npy', raw_samples)
            del flat_samples, raw_samples
        stat, bias, var, Cl, Cl_cmb, Cl_residuals_matrix, ell, W_cmb, ddW_cmb = res.get_residuals(
            model_results, fg_freq_maps, sigma_spectral, lmin, lmax, fsky, params,
            cmb_spectra=spectra_true, true_A_cmb=true_A_cmb)

        # current, peak = tracemalloc.get_traced_memory()
        # print('time_res = ', time.time() - start_res)
        # print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")

        WA_cmb = W_cmb.dot(model_results.mix_effectiv[:, :2])
        W_dBdB_cmb = ddW_cmb.dot(model_results.mix_effectiv[:, :2])
        VA_cmb = np.einsum('ij,ij...->...', sigma_spectral, W_dBdB_cmb[:, :])

        print('')
        print('W = ', W_cmb)
        print('A_cmb = ', model_results.mix_effectiv[:, :2])
        print('WA = ', WA_cmb)
        print('')
        print('VA_cmb = ', VA_cmb)
        print('')

        '''=================Init and cosmo likelihood estimation================'''
        # start_init_cosmo = time.time()
        Cl_fid = {}
        Cl_fid['BB'] = res.get_Cl_cmbBB(Alens=A_lens_true, r=r_true,
                                        path_BB=path_BB)[2][lmin:lmax+1]
        Cl_fid['BuBu'] = res.get_Cl_cmbBB(Alens=0.0, r=1.0, path_BB=path_BB)[2][lmin:lmax+1]
        Cl_fid['BlBl'] = res.get_Cl_cmbBB(Alens=1.0, r=0.0, path_BB=path_BB)[2][lmin:lmax+1]
        Cl_fid['EE'] = ps_planck[1, lmin:lmax+1]

        Cl_noise, ell_noise = res.get_noise_Cl(
            model_results.mix_effectiv, lmax+1, fsky,
            sensitiviy_mode=sensitiviy_mode, one_over_f_mode=one_over_f_mode,
            instrument=INSTRU)
        Cl_noise = Cl_noise[lmin-2:]
        ell_noise = ell_noise[lmin-2:]
        Cl_noise_matrix = np.zeros([2, 2, Cl_noise.shape[0]])
        Cl_noise_matrix[0, 0] = Cl_noise
        Cl_noise_matrix[1, 1] = Cl_noise

        tr_SigmaYY = np.einsum('ij,jimnl->mnl', sigma_spectral, Cl_residuals_matrix['YY'])
        Cl_data = Cl_residuals_matrix['yy'] + Cl_residuals_matrix['zy'] + \
            Cl_residuals_matrix['yz'] + tr_SigmaYY + Cl_noise_matrix

        # current, peak = tracemalloc.get_traced_memory()
        # print('time_init_cosmo = ', time.time() - start_init_cosmo)
        # print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")

        # Cl_data = Cl_residuals_matrix['yy'] + tr_SigmaYY + Cl_noise_matrix

        # start_min_cosmo = time.time()
        results_cosmp = minimize(res.likelihood_exploration, cosmo_params, args=(
            Cl_fid, Cl_data, Cl_noise_matrix, tr_SigmaYY, WA_cmb, VA_cmb, ell, fsky),
            tol=1e-18, bounds=bounds_cosmo,
            method=method_cosmo, jac=jac_cosmo)

        # current, peak = tracemalloc.get_traced_memory()
        # print('time_min_cosmo = ', time.time() - start_min_cosmo)
        # print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")

        # bounds=((-0.01, 0.1), (-np.pi/4, np.pi/4)),
        chi2_cosmo_true = res.likelihood_exploration(np.array([r_true, beta_true.value]), Cl_fid, Cl_data,
                                                     Cl_noise_matrix, tr_SigmaYY, WA_cmb,
                                                     VA_cmb, ell, fsky)
        print('')
        print('results - true cosmo = ', results_cosmp.x - np.array([r_true, beta_true.value]))
        print('chi2_cosmo_true =', chi2_cosmo_true)
        print('results_cosmp.fun =', results_cosmp.fun)
        print('true cosmo < results ? = ', chi2_cosmo_true < results_cosmp.fun)
        print('')
        print(results_cosmp)
        print('')
        start_chi2 = time.time()
        chi2cosmo_true = res.likelihood_exploration([r_true, beta_true.value], Cl_fid, Cl_data,
                                                    Cl_noise_matrix, tr_SigmaYY, WA_cmb, VA_cmb, ell, fsky)
        print('time chi2 = ', time.time() - start_chi2)
        print('chi2cosmo true = ', chi2cosmo_true)
        print('delta chi2cosmo = ', results_cosmp.fun - chi2cosmo_true)
        print('delta chi2cosmo relativ = ', (results_cosmp.fun - chi2cosmo_true)/chi2cosmo_true)

        '''==================Fisher cosmo likelihood estimation================='''
        start_fisher = time.time()
        Cl_cmb_model_fish = np.zeros([4, Cl_fid['EE'].shape[0]])
        Cl_cmb_model_fish[1] = copy.deepcopy(Cl_fid['EE'])
        Cl_cmb_model_fish[2] = copy.deepcopy(
            Cl_fid['BlBl'])*A_lens_true + copy.deepcopy(Cl_fid['BuBu']) * results_cosmp.x[0]

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

        current, peak = tracemalloc.get_traced_memory()
        print('time_fisher = ', time.time() - start_fisher)
        print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")

        try:
            os.mkdir(save_path+'gridding_cosmo')
        except FileExistsError:
            pass
        error_b = np.sqrt(np.linalg.inv(fisher_cosmo_matrix)[1, 1])
        beta_range = np.linspace(results_cosmp.x[1]-5 *
                                 error_b, results_cosmp.x[1]+5*error_b, 100)
        grid_cosmo_beta = []
        for b in beta_range:
            grid_cosmo_beta.append(res.likelihood_exploration([r_true, b], Cl_fid, Cl_data,
                                                              Cl_noise_matrix, tr_SigmaYY, WA_cmb, VA_cmb, ell, fsky))
        plt.close()
        plt.plot(beta_range, grid_cosmo_beta)
        plt.vlines(beta_true.value, min(grid_cosmo_beta), max(
            grid_cosmo_beta), linestyles='--', colors='k')
        plt.vlines(results_cosmp.x[1], min(grid_cosmo_beta), max(
            grid_cosmo_beta), linestyles='--', colors='blue')
        plt.savefig(save_path+'gridding_cosmo/'+'grid_prior{}.png'.format(prior_precision_deg))
        plt.close()
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
        fisher_spectal_mpi = np.empty([size_mpi, prior_per_rank,  freq_number+2, freq_number+2])
        prior_precision_grid_control = np.empty([size_mpi, prior_per_rank])
        cosmo_results_mpi = np.empty([size_mpi, prior_per_rank, 2])
        spectral_results_mpi = np.empty([size_mpi, prior_per_rank, freq_number+2])

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
