# from codecarbon import EmissionsTracker
import IPython
from astropy import units as u
import argparse
import os
from bjlib.class_faraday import power_spectra_obj, fisher_pws
from bjlib.lib_project import cl_rotation_derivative, get_dr_cov_bir_EB
from copy import deepcopy
import numpy as np
from fisher_pixel import fisher_new
from scipy.optimize import minimize
from pixel_based_angle_estimation import data_and_model_quick, get_model, \
    get_chi_squared_local, run_MCMC, constrained_chi2
from residuals import get_SFN, get_diff_list, get_diff_diff_list,\
    get_residuals, likelihood_exploration, get_noise_Cl, get_model_WACAW_WACAVeheh, run_double_MC
import shutil
import matplotlib.pyplot as plt
import copy
import time
from residuals import constrained_cosmo
from emcee import EnsembleSampler
# from config import *
import sys

# path_config = '/home/baptiste/Documents/these/pixel_based_analysis/code/config_files'
path_config = './config_files'
sys.path.append(path_config)

if True:  # this is just so that my autotpep8 doesn't put the import above the sys.path.append ...
    from config_test import *


def constrained_cosmo_fixed_r(cosmo_params, Cl_fid, Cl_data, Cl_noise_matrix, dWA_cmb,
                              sigma_spectral, WA_cmb, VA_cmb, total_prior_matrix,
                              true_miscal_angles, pivot_angle_index, eval_angles,
                              ell, fsky, minimisation=True):

    r = 0
    beta = cosmo_params[0]*u.rad
    pivot = cosmo_params[1]*u.rad

    if not minimisation:
        if beta.value < -np.pi/2 or beta.value > np.pi/2:
            print('bla2')
            return -np.inf
        elif pivot.value < -np.pi/2 or pivot.value > np.pi/2:
            print('bla3')
            return -np.inf

    new_angles = copy.deepcopy(eval_angles)
    diff_pivot = pivot.value - eval_angles[pivot_angle_index]

    new_angles += diff_pivot
    new_angles[pivot_angle_index] = pivot.value

    Cl_cmb_model = np.zeros([4, Cl_fid['EE'].shape[0]])
    Cl_cmb_model[1] = copy.deepcopy(Cl_fid['EE'])
    Cl_cmb_model[2] = copy.deepcopy(Cl_fid['BlBl'])*1 + copy.deepcopy(Cl_fid['BuBu']) * r

    Cl_cmb_rot = lib.cl_rotation(Cl_cmb_model.T, beta).T

    Cl_cmb_rot_matrix = np.zeros([2, 2, Cl_cmb_rot.shape[-1]])
    Cl_cmb_rot_matrix[0, 0] = copy.deepcopy(Cl_cmb_rot[1])
    Cl_cmb_rot_matrix[1, 1] = copy.deepcopy(Cl_cmb_rot[2])
    Cl_cmb_rot_matrix[1, 0] = copy.deepcopy(Cl_cmb_rot[4])
    Cl_cmb_rot_matrix[0, 1] = copy.deepcopy(Cl_cmb_rot[4])

    WACAW = np.einsum('ij,jkl,km->iml', WA_cmb, Cl_cmb_rot_matrix, WA_cmb.T)
    WACAV = np.einsum('ij,jkl,km->iml', WA_cmb, Cl_cmb_rot_matrix, VA_cmb.T)
    VACAW = np.einsum('ij,jkl,km->iml', VA_cmb, Cl_cmb_rot_matrix, WA_cmb.T)

    YY_cmb_matrix = np.zeros([sigma_spectral.shape[0], sigma_spectral.shape[0], Cl_cmb_rot_matrix.shape[0],
                              Cl_cmb_rot_matrix.shape[1], Cl_cmb_rot_matrix.shape[2]])
    for i in range(sigma_spectral.shape[0]):
        for ii in range(sigma_spectral.shape[0]):
            YY_cmb_matrix[i, ii] = np.einsum(
                'ij,jkl,km->iml', dWA_cmb[i].T, Cl_cmb_rot_matrix, dWA_cmb[ii])

    tr_SigmaYY = np.einsum('ij,jimnl->mnl', sigma_spectral, YY_cmb_matrix)

    Cl_model_total_ = WACAW + tr_SigmaYY + VACAW + WACAV  # + Cl_noise_matrix

    rot_pivot = np.array([[np.cos(2*diff_pivot), -np.sin(2*diff_pivot)],
                          [np.sin(2*diff_pivot), np.cos(2*diff_pivot)]])

    Cl_model_total = np.einsum('ij,jkl,km->iml', rot_pivot.T,
                               Cl_model_total_, rot_pivot) + Cl_noise_matrix
    Cl_data_rot = Cl_data

    inv_sigma_miscal = np.linalg.inv(sigma_spectral[:-2, :-2])
    angle_relat = np.delete(eval_angles, pivot_angle_index)
    true_prior = np.delete(true_miscal_angles, pivot_angle_index).value
    pivot_true = true_miscal_angles[pivot_angle_index].value

    prior_matrix = np.delete(
        np.delete(total_prior_matrix, pivot_angle_index, 0), pivot_angle_index, 1)[:-2, :-2]
    prior_element_pivot = total_prior_matrix[pivot_angle_index, pivot_angle_index]
    pivot = pivot.value
    inv_model = np.linalg.inv(Cl_model_total.T).T
    dof = (2 * ell + 1) * fsky
    dof_over_Cl = dof * inv_model

    first_term_ell = np.einsum('ijl,jkl->ikl', dof_over_Cl, Cl_data_rot)

    A = angle_relat + pivot - eval_angles[pivot_angle_index]

    radek_jost_prior = ((pivot-pivot_true)**2) * prior_element_pivot +\
        A.T.dot(inv_sigma_miscal).dot(A) -\
        (A.T.dot(inv_sigma_miscal)+true_prior.T.dot(prior_matrix)).dot(np.linalg.inv(inv_sigma_miscal +
                                                                                     prior_matrix)).dot(inv_sigma_miscal.dot(A)+prior_matrix.dot(true_prior)) +\
        np.log(np.linalg.det(inv_sigma_miscal + prior_matrix)) + \
        true_prior.T.dot(prior_matrix).dot(true_prior)

    first_term = np.sum(np.trace(first_term_ell))

    logdetC = np.sum(dof*np.log(np.linalg.det(Cl_model_total.T)))

    likelihood = first_term + logdetC + radek_jost_prior

    if not minimisation:
        return -likelihood/2
    else:
        return likelihood/2


def get_PI(pivot, pivot_angle_index, eval_angles, input_pivot, prior_matrix, inv_sigma_miscal):
    angle_relat = np.delete(eval_angles, pivot_angle_index)

    A = angle_relat + pivot - eval_angles[pivot_angle_index]

    term1 = ((pivot-input_pivot)**2) * prior_element_pivot
    term2 = A.T.dot(inv_sigma_miscal).dot(A)
    term3 = (A.T.dot(inv_sigma_miscal)+angle_relat.T.dot(prior_matrix)).dot(
        np.linalg.inv(inv_sigma_miscal +
                      prior_matrix)).dot(inv_sigma_miscal.dot(A)+prior_matrix.dot(angle_relat))
    return term1, term2, term3


def get_PI2(pivot, pivot_angle_index, eval_angles, input_pivot, prior_matrix, inv_sigma_miscal):
    angle_relat = np.delete(eval_angles, pivot_angle_index)

    C = np.linalg.inv(inv_sigma_miscal + prior_matrix_nopivot)

    mean1 = input_pivot - angle_relat
    mean2 = np.linalg.inv(inv_sigma_miscal - inv_sigma_miscal.dot(C).dot(inv_sigma_miscal)
                          ).dot(inv_sigma_miscal).dot(C).dot(prior_matrix_nopivot).dot(angle_relat)
    term1 = ((pivot-input_pivot)**2) * prior_element_pivot
    term2 = (pivot - mean1 - mean2).T.dot(inv_sigma_miscal -
                                          inv_sigma_miscal.dot(C).dot(inv_sigma_miscal)).dot(pivot - mean1 - mean2)
    return term1, term2


def prior_grid(pivot, eval_angles, true_miscal_angles, total_prior_matrix, pivot_angle_index):
    prior_matrix = np.delete(
        np.delete(total_prior_matrix, pivot_angle_index, 0), pivot_angle_index, 1)[:-2, :-2]
    prior_element_pivot = total_prior_matrix[pivot_angle_index, pivot_angle_index]

    true_prior = np.delete(true_miscal_angles, pivot_angle_index).value
    pivot_true = true_miscal_angles[pivot_angle_index].value

    angle_relat = np.delete(eval_angles, pivot_angle_index)
    A = angle_relat + pivot - eval_angles[pivot_angle_index]

    term1 = ((pivot-pivot_true)**2) * prior_element_pivot

    term2 = A.T.dot(inv_sigma_miscal).dot(A)
    term3 = -(A.T.dot(inv_sigma_miscal)+true_prior.T.dot(prior_matrix)).dot(np.linalg.inv(inv_sigma_miscal +
                                                                                          prior_matrix)).dot(inv_sigma_miscal.dot(A)+prior_matrix.dot(true_prior))

    term_cst = np.log(np.linalg.det(inv_sigma_miscal + prior_matrix)) + \
        true_prior.T.dot(prior_matrix).dot(true_prior)
    radek_jost_prior = term1 + term2 + term3 + term_cst
    return term1, term2, term3, term_cst


def Cl_maps(freq_maps):
    spectra_freq = []
    import healpy as hp
    spectra_freq = np.zeros([12, 12, lmax+1])
    for i in range(int(freq_maps.shape[0]/2)):
        for j in range(int(freq_maps.shape[0]/2)):
            print('Q = ', i*2, 'U = ', j*2+1, 'ax 0 ', i*2, i*2+2, 'ax 1 ', j*2, j*2+2)
            spectra_ = hp.anafast([freq_maps[0]*0, freq_maps[i*2],
                                   freq_maps[j*2+1]], lmax=lmax, nspec=5)/fsky
            spectra_freq[i*2:i*2+2, j*2:j*2+2] = np.array([[spectra_[1], spectra_[4]],
                                                           [spectra_[4], spectra_[2]]])
    return np.array(spectra_freq)


def WfgW(W, Cl_fg_freq):
    WfgWlist = []
    for i in range(6):
        WfgWi = np.einsum('ij,jkl,km->iml', W[:2, i*2:i*2+1], Cl_fg_freq[i], W[:2, i*2:i*2+1].T)
        WfgWlist.append(WfgWi)
    return np.array(WfgWlist)


# tracker = EmissionsTracker()
# tracker.start()

parser = argparse.ArgumentParser()
parser.add_argument("folder_end", help="end of the folder name")
parser.add_argument("--precision_index", help="prior precision in deg !!")
args = parser.parse_args()
# np.radom.seed(1)
save_path = save_path_ + args.folder_end + '/'
print()
print(save_path)
print(args.folder_end)
if os.path.exists(save_path):
    print('ERROR: path already exists')
    answer = input("Do you want to continue? This could lead to files being overwritten (y/n): ")
    if answer == 'n':
        print('exiting program')
        exit()
    elif answer != 'y':
        print('Not a valid answer, exiting program')  # TODO: ask question again instead of quiting
        exit()
    elif answer == 'y':
        print('proceeding')
else:
    os.mkdir(save_path)
# shutil.copy('config.py', save_path)
shutil.copy(path_config+'/config_test.py', save_path)

if prior_gridding:
    # precision_array = np.logspace(-4, np.log10(5), 40)
    # precision_array = np.append(precision_array, [0.01, 0.1, 1])
    # precision_array = np.sort(precision_array) * u.deg.to(u.rad)
    precision_array = np.load(
        '/home/baptiste/Documents/these/pixel_based_analysis/results_and_data/full_pipeline/double_sample_grid/prior_precision_grid.npy') * u.deg.to(u.rad)
    pivot_angle_index = 2
    one_prior = False
    if one_prior:
        prior_indices = [pivot_angle_index, pivot_angle_index+1]
    else:
        prior_indices = [0, freq_number]
    prior_matrix_array = []
    for prior_precision in precision_array:
        prior_matrix = np.zeros([len(params), len(params)])
        for i in range(prior_indices[0], prior_indices[-1]):
            prior_matrix[i, i] += 1/(prior_precision**2)
        prior_matrix_array.append(prior_matrix)
    prior_matrix_array = np.array(prior_matrix_array)
    np.save(save_path + 'precision_array.npy', precision_array)
    np.save(save_path + 'prior_matrix_array.npy', prior_matrix_array)
else:
    precision_array = np.array([prior_precision])
    prior_matrix_array = np.array([prior_matrix])


'''Sky signal simulation'''
data, model_data = data_and_model_quick(
    miscal_angles_array=true_miscal_angles, bir_angle=beta_true*0,
    frequencies_by_instrument_array=freq_by_instru, nside=nside,
    sky_model=sky_model, sensitiviy_mode=sensitiviy_mode,
    one_over_f_mode=one_over_f_mode, instrument=INSTRU,
    overwrite_freq=overwrite_freq, t_obs_years=t_obs_years, SAC_yrs_LF=SAC_yrs_LF)

if INSTRU == 'LiteBIRD':
    S_cmb_name = 'data/New_S_cmb_n{}_s{}_r{:1}_b{:1.1e}'.format(nside, nsim, r_true, beta_true.value).replace(
        '.', 'p') + '.npy'
else:
    S_cmb_name = 'S_cmb_n{}_s{}_r{:1}_b{:1.1e}'.format(nside, nsim, r_true, beta_true.value).replace(
        '.', 'p') + '.npy'
print(S_cmb_name)

start_sfn = time.time()
data_model, fg_freq_maps, n_obspix = get_SFN(data, model_data, path_BB,
                                             S_cmb_name, spectral_flag,
                                             addnoise=add_noise,
                                             fg_angle=fg_angle_config,
                                             dust_angle=dust_angle, synch_angle=synch_angle)
print('time get SFN=', time.time()-start_sfn)

if fg_angle_config is not None:
    import healpy as hp
    fg_spectra_freq = []
    for f in range(len(data.frequencies)):
        TQU_fg_map = np.zeros((3, fg_freq_maps[0].shape[-1]))
        TQU_fg_map[1] = fg_freq_maps[2*f]
        TQU_fg_map[2] = fg_freq_maps[2*f+1]
        fg_spectra = hp.anafast(TQU_fg_map)
        fg_spectra_freq.append(fg_spectra)
        '''
        plt.plot(fg_spectra[1, lmin:lmax+1], label='EE')
        plt.plot(fg_spectra[2, lmin:lmax+1], label='BB')
        plt.plot(fg_spectra[4, lmin:lmax+1], label='EB')
        plt.legend()
        plt.loglog()
        plt.show()
        '''
        del TQU_fg_map, fg_spectra
    np.save(save_path+'fg_spectra_list.npy', fg_spectra_freq)

# IPython.embed()
'''Sky model for spectral likelihood'''
model = get_model(
    miscal_angles_array=initmodel_miscal, bir_angle=beta_true*0,
    frequencies_by_instrument_array=freq_by_instru,
    nside=nside, spectral_params=[1.54, 20, -3],
    sky_model='c1s0d0', sensitiviy_mode=sensitiviy_mode,
    one_over_f_mode=one_over_f_mode, instrument=INSTRU,
    overwrite_freq=overwrite_freq, t_obs_years=t_obs_years, SAC_yrs_LF=SAC_yrs_LF)
model.n_obspix = n_obspix
if not spectral_flag:
    model.bir_matrix = model.bir_matrix[:2, :2]
    model.mixing_matrix = np.array([[1, 0], [0, 1]]*len(model.frequencies))

# IPython.embed()
# pivot_angle_index = 0
angle_constrain_start = np.delete(angle_array_start, pivot_angle_index)
'''General spectral likelihood minimisation'''


start_loop = time.time()
input_angle_array = []
results_min_array = []
angle_eval_array = []
cosmo_array = []
# bias_input_array = np.logspace(-2, np.log10(5), 20)*u.deg.to(u.rad)

# bias_input_array = np.logspace(-4, -2, 20)*u.deg.to(u.rad)
# bias_input_array = np.insert(bias_input_array, 0, 0)
fisher_spectral_array = []
fisher_pivot_array = []
fisher_cosmo_prior_array = []
fisher_cosmo_array = []
bias_input_array = np.array([0])
bias_input = 0*u.deg.to(u.rad)
# IPython.embed()

for prior_precision, prior_matrix in zip(precision_array, prior_matrix_array):
    input_angles[pivot_angle_index] = true_miscal_angles[pivot_angle_index].value + bias_input
    input_angle_array.append(copy.deepcopy(input_angles))

    angle_array_start = np.random.uniform(np.array(bounds)[:, 0],
                                          np.array(bounds)[:, 1])
    angle_constrain_start = np.delete(angle_array_start, pivot_angle_index)
    print(angle_constrain_start)
    results_min = minimize(constrained_chi2, angle_constrain_start, args=(
        data_model, model, pivot_angle_index, input_angles[pivot_angle_index], True, params),
        tol=1e-18, options={'maxiter': 1000}, method='L-BFGS-B',
        bounds=bounds[1:])
    print(results_min)
    print('results - spectral_true = ',
          results_min.x[:freq_number-1] - np.delete(true_miscal_angles, pivot_angle_index).value)
    print('results - spectral_true = ', results_min.x[-2] - 1.54)
    print('results - spectral_true = ', results_min.x[-1] + 3)
    IPython.embed()
    results_min_array.append(results_min.x)

    angle_eval = np.insert(results_min.x[:freq_number-1],
                           pivot_angle_index, input_angles[pivot_angle_index])
    angle_eval_array.append(angle_eval)

    '''Sky model from spectral likelihood results'''
    model_results = get_model(
        angle_eval, bir_angle=beta_true*0,
        frequencies_by_instrument_array=freq_by_instru, nside=nside,
        spectral_params=[results_min.x[-2], 20, results_min.x[-1]],
        sky_model='c1s0d0', sensitiviy_mode=sensitiviy_mode,
        one_over_f_mode=one_over_f_mode, instrument=INSTRU,
        overwrite_freq=overwrite_freq, t_obs_years=t_obs_years,
        SAC_yrs_LF=SAC_yrs_LF)

    '''Spectral Fisher matrix estimation'''
    diff_list_res = get_diff_list(model_results, params)
    diff_diff_list_res = get_diff_diff_list(model_results, params)
    fisher_matrix_spectral = fisher_new(data_model, model_results,
                                        diff_list_res, diff_diff_list_res, params)
    fisher_pivot = np.delete(np.delete(fisher_matrix_spectral,
                                       pivot_angle_index, 0), pivot_angle_index, 1)
    sigma_spectral = np.linalg.inv(fisher_pivot)

    fisher_spectral_array.append(fisher_matrix_spectral)
    fisher_pivot_array.append(fisher_pivot)
    # IPython.embed()
    if spectral_MCMC_flag:
        init_spectral_MCMC = np.random.normal(results_min.x, np.diag(
            sigma_spectral)*2, (spectral_walker_per_dim*(spectral_dim-1), spectral_dim-1))

        sampler_spectral = EnsembleSampler(
            spectral_walker_per_dim*(spectral_dim-1), spectral_dim-1,
            constrained_chi2, args=[data_model, model, pivot_angle_index,
                                    input_angles[pivot_angle_index], False, params])
        sampler_spectral.reset()
        sampler_spectral.run_mcmc(init_spectral_MCMC, nsteps_spectral, progress=True)
        samples_spectral_raw = sampler_spectral.get_chain()
        samples_spectral = sampler_spectral.get_chain(discard=discard_spectral, flat=True)
        np.save(save_path+'samples_spectral_raw.npy', samples_spectral_raw)
        np.save(save_path+'samples_spectral.npy', samples_spectral)

    '''Residuals computation'''
    stat, bias, var, Cl_fg, Cl_cmb, Cl_residuals_matrix, ell, W_cmb, dW_cmb, ddW_cmb = get_residuals(
        model_results, fg_freq_maps, sigma_spectral, lmin, lmax, fsky, params,
        cmb_spectra=spectra_true, true_A_cmb=model_data.mix_effectiv[:, :2], pivot_angle_index=pivot_angle_index)

    WA_cmb = W_cmb.dot(model_results.mix_effectiv[:, :2])
    dWA_cmb = dW_cmb.dot(model_results.mix_effectiv[:, :2])
    W_dBdB_cmb = ddW_cmb.dot(model_results.mix_effectiv[:, :2])
    VA_cmb = np.einsum('ij,ij...->...', sigma_spectral, W_dBdB_cmb[:, :])
    # IPython.embed()
    '''Cosmo likelihood minimisation'''
    Cl_noise, ell_noise = get_noise_Cl(
        model_results.mix_effectiv, lmax+1, fsky,
        sensitiviy_mode, one_over_f_mode,
        instrument=INSTRU, onefreqtest=1-spectral_flag, t_obs_years=t_obs_years, SAC_yrs_LF=SAC_yrs_LF)
    Cl_noise = add_noise*Cl_noise[lmin-2:]
    ell_noise = ell_noise[lmin-2:]
    Cl_noise_matrix = np.zeros([2, 2, Cl_noise.shape[0]])
    Cl_noise_matrix[0, 0] = Cl_noise
    Cl_noise_matrix[1, 1] = Cl_noise
    # Cl_noise_matrix *= 10
    # IPython.embed()
    matrix_fg_yy = np.array([[Cl_fg['yy'][0], Cl_fg['yy'][2]],
                             [Cl_fg['yy'][2], Cl_fg['yy'][1]]])

    tr_SigmaYY = np.einsum('ij,jimnl->mnl', sigma_spectral, Cl_residuals_matrix['YY'])
    Cl_data = Cl_residuals_matrix['yy'] + Cl_residuals_matrix['zy'] + \
        Cl_residuals_matrix['yz'] + tr_SigmaYY + Cl_noise_matrix

    # cosmo_params = [0.01, beta_true.value+prior_precision,
    #                 true_miscal_angles[pivot_angle_index].value + prior_precision]
    # (-0.01, 5)
    # bounds_cosmo2 = ((0., 5), (-np.pi/3, np.pi/3), (-np.pi/3, np.pi/3))
    bounds_cosmo2 = ((1e-5, 1), (-np.pi/3, np.pi/3), (-np.pi/3, np.pi/3))

    pivot_precision = prior_precision
    # (-0.01, 1)
    bounds_rand = ((0.0, 0.01), (- 3*u.deg.to(u.rad), 3*u.deg.to(u.rad)),
                   (input_angles[pivot_angle_index]-3*pivot_precision, input_angles[pivot_angle_index]+3*pivot_precision))

    cosmo_array_start = np.random.uniform(
        np.array(bounds_rand)[:, 0], np.array(bounds_rand)[:, 1])
    # cosmo_array_start = np.array([0.01, 0, true_miscal_angles[pivot_angle_index].value])
    print('cosmo_array_start=', cosmo_array_start)
    # IPython.embed()
    result_cosmo = minimize(constrained_cosmo, cosmo_array_start,
                            args=(Cl_fid, Cl_data, Cl_noise_matrix, dWA_cmb, sigma_spectral,
                                  WA_cmb, VA_cmb, prior_matrix, true_miscal_angles,
                                  pivot_angle_index, angle_eval,  ell, fsky, True),
                            bounds=bounds_cosmo2)
    true_cosmo_params = [r_true, beta_true.value, true_miscal_angles[pivot_angle_index].value]
    print('')
    print('true_cosmo_params = ',
          true_cosmo_params[0], (true_cosmo_params[1])*u.rad.to(u.deg), (true_cosmo_params[2])*u.rad.to(u.deg))
    print('result_cosmo = ',
          result_cosmo.x[0], (result_cosmo.x[1])*u.rad.to(u.deg), (result_cosmo.x[2])*u.rad.to(u.deg))
    print('====')
    print('cosmo res - true (1,rad,rad) =', result_cosmo.x - true_cosmo_params)
    print('====')

    cosmo_array.append(result_cosmo.x)

    '''Fisher cosmo estimation'''

    WACAW, VACAW, WACAV, tr_SigmaYY_cmb, YY_cmb = get_model_WACAW_WACAVeheh(result_cosmo.x,
                                                                            Cl_fid, dWA_cmb, sigma_spectral, WA_cmb, VA_cmb)
    # WACAW, VACAW, WACAV, tr_SigmaYY_cmb, YY_cmb = get_model_WACAW_WACAVeheh(true_cosmo_params,
    #                                                                         Cl_fid, dWA_cmb, sigma_spectral, WA_cmb, VA_cmb)

    diff_pivot_result = result_cosmo.x[-1] - input_angles[pivot_angle_index]
    # diff_pivot_result = 0  # result_cosmo.x[-1] - input_angles[pivot_angle_index]
    rot_pivot = np.array([[np.cos(2*diff_pivot_result), np.sin(2*diff_pivot_result)],
                          [-np.sin(2*diff_pivot_result), np.cos(2*diff_pivot_result)]])

    Cl_data_fish_ = WACAW + VACAW + WACAV + tr_SigmaYY_cmb + Cl_noise_matrix
    Cl_data_fish = np.einsum('ij,jkl,km->iml', rot_pivot, Cl_data_fish_, rot_pivot.T)

    Cl_cmb_model_fish = np.zeros([4, Cl_fid['EE'].shape[0]])
    Cl_cmb_model_fish[1] = deepcopy(Cl_fid['EE'])
    Cl_cmb_model_fish[2] = deepcopy(Cl_fid['BlBl'])*A_lens_true + \
        deepcopy(Cl_fid['BuBu']) * result_cosmo.x[0]

    Cl_cmb_model_fish_r = np.zeros([4, Cl_fid['EE'].shape[0]])
    Cl_cmb_model_fish_r[2] = deepcopy(Cl_fid['BuBu']) * 1

    deriv1 = power_spectra_obj(cl_rotation_derivative(
        Cl_cmb_model_fish.T, result_cosmo.x[1]*u.rad), ell)
    dCldbeta = np.array(
        [[deriv1.spectra[:, 1], deriv1.spectra[:, 4]],
         [deriv1.spectra[:, 4], deriv1.spectra[:, 2]]])
    WACAVdb = np.einsum('ij,jkl,km->iml', WA_cmb, dCldbeta, VA_cmb.T)
    VACAWdb = np.einsum('ij,jkl,km->iml', VA_cmb, dCldbeta, WA_cmb.T)
    WACAWdb = np.einsum('ij,jkl,km->iml', WA_cmb, dCldbeta, WA_cmb.T)

    dCldr = get_dr_cov_bir_EB(
        Cl_cmb_model_fish_r.T, result_cosmo.x[1]*u.rad)
    WACAVdr = np.einsum('ij,jkl,km->iml', WA_cmb, dCldr, VA_cmb.T)
    VACAWdr = np.einsum('ij,jkl,km->iml', VA_cmb, dCldr, WA_cmb.T)
    WACAWdr = np.einsum('ij,jkl,km->iml', WA_cmb, dCldr, WA_cmb.T)

    YY_cmb_matrixdr = np.zeros([sigma_spectral.shape[0], sigma_spectral.shape[0], dCldr.shape[0],
                                dCldr.shape[1], dCldr.shape[2]])
    YY_cmb_matrixdB = np.zeros([sigma_spectral.shape[0], sigma_spectral.shape[0], dCldbeta.shape[0],
                                dCldbeta.shape[1], dCldbeta.shape[2]])

    for i in range(sigma_spectral.shape[0]):
        for ii in range(sigma_spectral.shape[0]):
            YY_cmb_matrixdr[i, ii] = np.einsum(
                'ij,jkl,km->iml', dWA_cmb[i].T, dCldr, dWA_cmb[ii])
            YY_cmb_matrixdB[i, ii] = np.einsum(
                'ij,jkl,km->iml', dWA_cmb[i].T, dCldbeta, dWA_cmb[ii])

    tr_SigmaYYdr = np.einsum('ij,jimnl->mnl', sigma_spectral, YY_cmb_matrixdr)
    tr_SigmaYYdB = np.einsum('ij,jimnl->mnl', sigma_spectral, YY_cmb_matrixdB)

    rot_dB = np.einsum('ij,jkl,km->iml', rot_pivot, (WACAWdb +
                                                     VACAWdb + WACAVdb + tr_SigmaYYdB), rot_pivot.T)
    rot_dr = np.einsum('ij,jkl,km->iml', rot_pivot, (WACAWdr +
                                                     VACAWdr + WACAVdr + tr_SigmaYYdr), rot_pivot.T)

    deriv_rot_pivot = np.array([[-2*np.sin(2*diff_pivot_result), 2*np.cos(2*diff_pivot_result)],
                                [-2*np.cos(2*diff_pivot_result), -2*np.sin(2*diff_pivot_result)]])
    # dCldalpha = np.einsum('ij,jkl,km->iml', deriv_rot_pivot.T, Cl_data_fish_, rot_pivot) + \
    #     np.einsum('ij,jkl,km->iml', rot_pivot.T, Cl_data_fish_, deriv_rot_pivot)
    dCldalpha = np.einsum('ij,jkl,km->iml', deriv_rot_pivot, Cl_data_fish_, rot_pivot.T) + \
        np.einsum('ij,jkl,km->iml', rot_pivot, Cl_data_fish_, deriv_rot_pivot.T)

    deriv_matrix_beta2 = power_spectra_obj(rot_dB.T, deriv1.ell)
    deriv_matrix_r2 = power_spectra_obj(rot_dr.T, ell)
    deriv_matrix_alpha2 = power_spectra_obj(dCldalpha.T, ell)

    Cl_data_spectra = power_spectra_obj(Cl_data_fish.T, ell)

    fish_r2 = fisher_pws(Cl_data_spectra, deriv_matrix_r2, fsky)
    fish_beta2 = fisher_pws(Cl_data_spectra, deriv_matrix_beta2, fsky)
    fish_alpha2 = fisher_pws(Cl_data_spectra, deriv_matrix_alpha2, fsky)
    fish_beta_r2 = fisher_pws(Cl_data_spectra, deriv_matrix_beta2, fsky, deriv2=deriv_matrix_r2)
    fish_alpha_r2 = fisher_pws(Cl_data_spectra, deriv_matrix_alpha2, fsky, deriv2=deriv_matrix_r2)
    fish_alpha_beta2 = fisher_pws(Cl_data_spectra, deriv_matrix_alpha2,
                                  fsky, deriv2=deriv_matrix_beta2)
    fisher_cosmo_matrix = np.array([[fish_r2, fish_beta_r2, fish_alpha_r2],
                                    [fish_beta_r2, fish_beta2, fish_alpha_beta2],
                                    [fish_alpha_r2, fish_alpha_beta2, fish_alpha2]])

    prior_matrix_nopivot = np.delete(
        np.delete(prior_matrix, pivot_angle_index, 0), pivot_angle_index, 1)[:-2, :-2]
    prior_element_pivot = prior_matrix[pivot_angle_index, pivot_angle_index]
    inv_sigma_miscal = np.linalg.inv(sigma_spectral[:-2, :-2])
    # IPython.embed()
    # PI_fisher = 2*prior_element_pivot + 2 * np.ones(5).T.dot(inv_sigma_miscal).dot(np.ones(5)) -\
    #     2 * np.ones(5).T.dot(inv_sigma_miscal).dot(np.linalg.inv(inv_sigma_miscal +
    #                                                              prior_matrix_nopivot)).dot(inv_sigma_miscal).dot(np.ones(5))
    PI_fisher = 2 * prior_element_pivot + 2 * np.ones(freq_number-1).T.dot(inv_sigma_miscal).dot(np.ones(freq_number-1)) -\
        2 * np.ones(freq_number-1).T.dot(inv_sigma_miscal).dot(np.linalg.inv(inv_sigma_miscal +
                                                                             prior_matrix_nopivot)).dot(inv_sigma_miscal).dot(np.ones(freq_number-1))

    fisher_cosmo_matrix_prior = copy.deepcopy(fisher_cosmo_matrix)
    fisher_cosmo_matrix_prior[-1, -1] += PI_fisher/2.
    fisher_cosmo_prior_array.append(fisher_cosmo_matrix_prior)
    sigma_print = np.sqrt(np.linalg.inv(fisher_cosmo_matrix_prior))
    print('prior_precision in deg=', prior_precision*u.rad.to(u.deg))
    print('sigma r =', sigma_print[0, 0], 'sigma beta in deg =', sigma_print[1, 1] *
          u.rad.to(u.deg), 'sigma alpha_ref in deg =', sigma_print[2, 2]*u.rad.to(u.deg))

    fisher_cosmo_array.append(fisher_cosmo_matrix)

    rmin = result_cosmo.x[0]-6*sigma_print[0, 0]  # -1e-5
    rmax = result_cosmo.x[0]+6*sigma_print[0, 0]
    # rmin = -0.01-6*sigma_print[0, 0]  # -1e-5
    # rmax = -0.01+6*sigma_print[0, 0]

    r_range = np.arange(rmin, rmax, (rmax-rmin)/500)
    L_grid_ = []
    fisher_r_grid_ = []
    Cl_fid_matrix = np.array([[Cl_fid['EE'], Cl_fid['EE']*0],
                              [Cl_fid['EE']*0, Cl_fid['BB']]]) + Cl_noise_matrix  # _arianna
    # test_data_Cl
    L_basic = []
    # fisher_sigma_r68 = []
    # fisher_sigma_r34 = []
    for rgrid in r_range:
        fisher_r_grid_.append((rgrid - (result_cosmo.x[0]))**2 / (2*(sigma_print[0, 0]**2)))
        # fisher_sigma_r68.append((rgrid - result_cosmo.x[0])**2 / (2*(sigma_r_basic68**2)))
        # fisher_sigma_r34.append((rgrid - result_cosmo.x[0])**2 / (2*(sigma_r_basic34**2)))
        L_grid_.append(constrained_cosmo([rgrid, 0, true_miscal_angles[pivot_angle_index].value],
                                         Cl_fid, Cl_data, Cl_noise_matrix,
                                         dWA_cmb, sigma_spectral, WA_cmb, VA_cmb, prior_matrix,
                                         true_miscal_angles, pivot_angle_index, angle_eval, ell,
                                         fsky, True))
        # L_grid_.append(constrained_cosmo([rgrid, 0, true_miscal_angles[pivot_angle_index].value],
        #                                  Cl_fid, Cl_fid_matrix, Cl_noise_matrix,
        #                                  dWA_cmb*0, sigma_spectral,
        #                                  np.identity(2), 0*VA_cmb, prior_matrix,
        #                                  true_miscal_angles, pivot_angle_index, true_miscal_angles.value, ell,
        #                                  fsky, True))
        # # Cl_cmb_model_test = np.array(
        #     [[Cl_fid['EE'], Cl_fid['EE']*0], [Cl_fid['EE']*0, Cl_fid['BlBl']+Cl_fid['BuBu'] * rgrid]]) + Cl_noise_matrix
        # inv_model = np.linalg.inv(Cl_cmb_model_test.T).T
        # dof = (2 * ell + 1) * fsky
        # dof_over_Cl = dof * inv_model
        # first_term = np.sum(np.trace(np.einsum('ijl,jkl->ikl', dof_over_Cl, Cl_fid_matrix)))
        # logdetC = np.sum(dof*np.log(np.abs(np.linalg.det(Cl_cmb_model_test.T))))
        # L_basic.append(-(first_term + logdetC)/2)

        Cl_cmb_model_test = np.array(
            [Cl_fid['BlBl']+Cl_fid['BuBu'] * rgrid]) + Cl_noise_matrix[1, 1]  # + Cl_noise_arianna
        inv_model = 1/Cl_cmb_model_test
        dof = (2 * ell + 1) * fsky
        dof_over_Cl = dof * inv_model
        first_term = np.sum(dof_over_Cl[0] * Cl_fid_matrix[1, 1])
        logdetC = np.sum(dof*np.log(Cl_cmb_model_test))
        L_basic.append(-(first_term + logdetC))

    r_range = r_range  # [249:]
    L_grid_ = np.array(L_grid_)  # [249:]
    L_basic = -np.array(L_basic)  # [249:]
    # L_grid = -L_grid_ / np.min(L_grid_)
    fisher_r_grid_ = np.array(fisher_r_grid_)  # [249:]
    # fisher_r_grid = fisher_r_grid_ / np.mex(fisher_r_grid_)
    # fisher_sigma_r68 = np.array(fisher_sigma_r68)
    # fisher_sigma_r34 = np.array(fisher_sigma_r34)

    max_Lgrid = np.nanmax(-L_grid_)
    expLB_grid = np.exp(-L_grid_-np.nanmax(-L_grid_))
    expbasic_grid = np.exp(-L_basic-np.nanmax(-L_basic))
    expfisher_grid = np.exp(-fisher_r_grid_ - 0*np.max(fisher_r_grid_))
    # expfisher_68_grid = np.exp(-fisher_sigma_r68)
    # expfisher_34_grid = np.exp(-fisher_sigma_r34)
    plt.plot(r_range, expLB_grid, label='likelihood grid')
    plt.plot(r_range, expfisher_grid, '--', label='fisher gaussian')
    plt.plot(r_range, expbasic_grid, label='like basic')
    # plt.plot(r_range, expfisher_68_grid, '--', label='fisher gaussian 68')
    # plt.plot(r_range, expfisher_34_grid, '--', label='fisher gaussian 34')
    # plt.vlines(rs_pos[np.argmin(np.abs(cum - 0.68))], 0, 1, label='sigma 68', color='orange')
    # plt.vlines(rs_pos[np.argmin(np.abs(cum - 0.34))], 0, 1,
    #            label='sigma 34', color='orange', linestyles='--')
    # plt.plot(x_axis_arianna, like_arianna**(1/2), label='likelihood Arianna')
    plt.legend()
    plt.xlabel('r')
    plt.show()
    # IPython.embed()
    ind_r_fit = np.nanargmin(L_grid_)
    r_fit = r_range[ind_r_fit]
    # print('logL', L_grid_)
    L = np.exp(-(L_grid_-np.nanmin(L_grid_)))

    rs_pos = r_range[r_range > r_fit]
    plike_pos = L[r_range > r_fit]
    cum = np.cumsum(plike_pos)
    cum /= cum[-1]*2
    sigma_r = rs_pos[np.argmin(np.abs(cum - 0.34))] - r_fit

    ind_r_fit = np.nanargmin(L_basic)
    r_fit = r_range[ind_r_fit]
    # print('logL', L_grid_)
    L = np.exp(-(L_basic-np.nanmin(L_basic)))

    rs_pos = r_range[r_range > r_fit]
    plike_pos = L[r_range > r_fit]
    cum = np.cumsum(plike_pos)
    cum /= cum[-1]*2
    # sigma_r_basic68 = rs_pos[np.argmin(np.abs(cum - 0.68))] - r_fit
    sigma_r_basic34 = rs_pos[np.argmin(np.abs(cum - 0.34))] - r_fit
    # plt.plot(r_range, L_grid)
    # plt.plot(r_range, fisher_r_grid)
    # plt.vlines(result_cosmo.x[0], plt.ylim()[0], plt.ylim()[1], linestyles='--', colors='b')
    # plt.show()
    '''
    bmin = beta_true.value-4*sigma_print[1, 1]
    bmax = beta_true.value+4*sigma_print[1, 1]

    bmin = -0.01
    bmax = 0.01
    b_range = np.arange(bmin, bmax, (bmax-bmin)/500)

    Lb_grid_ = []
    fisher_beta_grid_ = []
    for bgrid in b_range:
        fisher_beta_grid_.append((bgrid - result_cosmo.x[1])**2 / (2*(sigma_print[1, 1])**2))

        Lb_grid_.append(constrained_cosmo([0, bgrid, true_miscal_angles[pivot_angle_index].value], Cl_fid, Cl_data, Cl_noise_matrix,
                                          dWA_cmb, sigma_spectral, WA_cmb, VA_cmb, prior_matrix, true_miscal_angles, pivot_angle_index, angle_eval,  ell, fsky, True))
    Lb_grid_ = np.array(Lb_grid_)
    Lb_grid = -Lb_grid_ / np.min(Lb_grid_)
    fisher_beta_grid_ = np.array(fisher_beta_grid_)
    fisher_beta_grid = fisher_beta_grid_ / np.max(fisher_beta_grid_)
    expLB_grid_beta = np.exp(-Lb_grid_-np.max(-Lb_grid_))
    expfisher_grid_beta = np.exp(-fisher_beta_grid_ - 0*np.max(fisher_beta_grid_))
    plt.plot(b_range, expLB_grid_beta, label='likelihood grid')
    plt.plot(b_range, expfisher_grid_beta, label='fisher gaussian')
    plt.legend()
    plt.xlabel(r'$\beta_b$')
    plt.show()

    amin = true_miscal_angles[pivot_angle_index].value-4*prior_precision
    amax = true_miscal_angles[pivot_angle_index].value+4*prior_precision
    a_range = np.arange(amin, amax, (amax-amin)/500)

    La_grid_ = []
    fisher_alpha_grid_ = []
    for agrid in a_range:
        fisher_alpha_grid_.append((agrid - result_cosmo.x[2])**2 / (2*sigma_print[2, 2]**2))

        La_grid_.append(constrained_cosmo([0, 0, agrid], Cl_fid, Cl_data, Cl_noise_matrix,
                                          dWA_cmb, sigma_spectral, WA_cmb, VA_cmb, prior_matrix, true_miscal_angles, pivot_angle_index, angle_eval,  ell, fsky, True))
    La_grid_ = np.array(La_grid_)
    fisher_alpha_grid_ = np.array(fisher_alpha_grid_)
    expLB_grid_alpha = np.exp(-La_grid_-np.max(-La_grid_))
    expfisher_grid_alpha = np.exp(-fisher_alpha_grid_ - 0*np.max(fisher_alpha_grid_))
    plt.plot(a_range, expLB_grid_alpha)
    plt.plot(a_range, expfisher_grid_alpha)
    plt.xlabel(r'$\alpha_{\rm{ref}}$')
    plt.show()

    IPython.embed()
    '''


input_angle_array = np.array(input_angle_array)
results_min_array = np.array(results_min_array)
angle_eval_array = np.array(angle_eval_array)
cosmo_array = np.array(cosmo_array)
fisher_spectral_array = np.array(fisher_spectral_array)
fisher_pivot_array = np.array(fisher_pivot_array)
fisher_cosmo_prior_array = np.array(fisher_cosmo_prior_array)
fisher_cosmo_array = np.array(fisher_cosmo_array)

print('time loop = ', time.time() - start_loop)
np.save(save_path + 'input_angle_array.npy', input_angle_array)
np.save(save_path + 'results_min_array.npy', results_min_array)
np.save(save_path + 'angle_eval_array.npy', angle_eval_array)
np.save(save_path + 'cosmo_array.npy', cosmo_array)
np.save(save_path + 'bias_input_array.npy', bias_input_array)
np.save(save_path + 'fisher_spectral_array.npy', fisher_spectral_array)
np.save(save_path + 'fisher_pivot_array.npy', fisher_pivot_array)
np.save(save_path + 'fisher_cosmo_prior_array.npy', fisher_cosmo_prior_array)
np.save(save_path + 'fisher_cosmo_array.npy', fisher_cosmo_array)
# IPython.embed()
# emissions: float = tracker.stop()
# print(emissions)

# exit()
if cosmo_MCMC_flag:
    init_cosmo_MCMC = np.random.normal(
        result_cosmo.x, [0.01, prior_precision, prior_precision], (2*3, 3))
    # absolute value in r dimension because LB
    init_cosmo_MCMC[:, 0] = np.abs(init_cosmo_MCMC[:, 0])
    nwalkers_cosmo = 2 * 3
    sampler_cosmo = EnsembleSampler(
        nwalkers_cosmo, 3, constrained_cosmo, args=[
            Cl_fid, Cl_data, Cl_noise_matrix, dWA_cmb,
            sigma_spectral, WA_cmb, VA_cmb, prior_matrix,
            true_miscal_angles, pivot_angle_index, angle_eval,
            ell, fsky, False])
    sampler_cosmo.reset()
    sampler_cosmo.run_mcmc(init_cosmo_MCMC, nsteps_cosmo, progress=True)
    samples_cosmo_raw = sampler_cosmo.get_chain()
    samples_cosmo = sampler_cosmo.get_chain(discard=discard_cosmo, flat=True)
    np.save(save_path+'raw_cosmo_samples.npy', samples_cosmo_raw)
    np.save(save_path+'cosmo_samples.npy', samples_cosmo)
# del samples_cosmo_raw, samples_cosmo


# np.save(save_path+'fisher_cosmo_matrix.npy', fisher_cosmo_matrix)
# np.save(save_path+'fisher_cosmo_matrix_prior.npy', fisher_cosmo_matrix_prior)


pivot_range = np.arange(1.5, 3.5, 2/1000)*u.deg.to(u.rad)
term1_list = []
term2_list = []
term3_list = []
termcst_list = []
for i in pivot_range:
    term1, term2, term3, termcst = prior_grid(i, angle_eval, true_miscal_angles,
                                              prior_matrix, pivot_angle_index)
    term1_list.append(term1)
    term2_list.append(term2)
    term3_list.append(term3)
    termcst_list.append(termcst)

term1_list = np.array(term1_list)
term2_list = np.array(term2_list)
term3_list = np.array(term3_list)
termcst_list = np.array(termcst_list)

totprior = term1_list + term2_list + term3_list + termcst_list
term_sigma = term2_list + term3_list  # + termcst_lis

maxsigma = np.exp(-0.5*(term_sigma-term_sigma.max())).max()

max1 = np.exp(-0.5*(term1_list-term1_list.max())).max()
exp2 = np.exp(-0.5*(term2_list-term2_list.max()))
max2 = exp2[np.isfinite(exp2)].max()

exp3 = np.exp(-0.5*(term3_list-term3_list.max()))
max3 = exp3[np.isfinite(exp3)].max()

# max3 = np.exp(-0.5*(term3_list-term3_list.max())).max()
expcst = np.exp(-0.5*(termcst_list-termcst_list.max()))
maxcst = expcst[np.isfinite(expcst)].max()
# maxcst = np.exp(-0.5*(termcst_list-termcst_list.max())).max()
exptot = np.exp(-0.5*(totprior-totprior.max()))
maxtot = exptot[np.isfinite(exptot)].max()
# maxtot = np.exp(-0.5*(totprior-totprior.max())).max()
gaussian_term1 = np.exp(-0.5*(term1_list-term1_list.max()))/max1
gaussian_termsigma = np.exp(-0.5*(term_sigma-term_sigma.max()))/maxsigma
gaussian_tot = np.exp(-0.5*(totprior-totprior.max()))/maxtot
plt.plot(pivot_range, gaussian_term1)
plt.plot(pivot_range, gaussian_termsigma)
# plt.plot(pivot_range, np.exp(-0.5*(term2_list-term2_list.max()))/max2)
plt.plot(pivot_range, gaussian_tot)
np.save(save_path+'pivot_range.npy', pivot_range)
np.save(save_path+'prior_first_term.npy', gaussian_term1)
np.save(save_path+'prior_second_term.npy', gaussian_termsigma)
np.save(save_path+'prior_total.npy', gaussian_tot)
IPython.embed()

'''
term1v2_list = []
term2v2_list = []
for i in pivot_range:
    term1v2, term2v2 = get_PI2(i, pivot_angle_index, angle_eval,
                               input_angles[pivot_angle_index],
                               prior_matrix_nopivot, inv_sigma_miscal)
    term1v2_list.append(term1v2)
    term2v2_list.append(term2v2)

term1v2_list = np.array(term1v2_list)
term2v2_list = np.array(term2v2_list)
totv2 = term1v2_list + term2v2_list

max1 = np.exp(-0.5*(term1v2_list-term1v2_list.max())).max()
max2 = np.exp(-0.5*(term2v2_list-term2v2_list.max())).max()
maxtot = np.exp(-0.5*(totv2-totv2.max())).max()

plt.plot(pivot_range, np.exp(-0.5*(term1v2_list-term1v2_list.max()))/max1)
plt.plot(pivot_range, np.exp(-0.5*(term2v2_list-term2v2_list.max()))/max2)
plt.plot(pivot_range, np.exp(-0.5*(totv2-totv2.max()))/maxtot)
np.save(save_path+'pivot_range.npy', pivot_range)
np.save(save_path+'prior_first_term.npy', np.exp(-0.5*(term1v2_list-term1v2_list.max()))/max1)
np.save(save_path+'prior_second_term.npy', np.exp(-0.5*(term2v2_list-term2v2_list.max()))/max2)
'''
