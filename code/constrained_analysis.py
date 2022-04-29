import matplotlib.pyplot as plt
import copy
import time
from residuals import constrained_cosmo
from emcee import EnsembleSampler
from config import *
import shutil
from residuals import get_SFN, get_diff_list, get_diff_diff_list,\
    get_residuals, likelihood_exploration, get_noise_Cl, get_model_WACAW_WACAVeheh, run_double_MC
from pixel_based_angle_estimation import data_and_model_quick, get_model, \
    get_chi_squared_local, run_MCMC, constrained_chi2
from scipy.optimize import minimize
from fisher_pixel import fisher_new
import numpy as np
from copy import deepcopy
from bjlib.lib_project import cl_rotation_derivative, get_dr_cov_bir_EB
from bjlib.class_faraday import power_spectra_obj, fisher_pws
import os
import argparse
from astropy import units as u
import IPython


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


parser = argparse.ArgumentParser()
parser.add_argument("folder_end", help="end of the folder name")
parser.add_argument("--precision_index", help="prior precision in deg !!")
args = parser.parse_args()

save_path = save_path_ + args.folder_end + '/'
print()
print(save_path)
print(args.folder_end)
if os.path.exists(save_path):
    print('ERROR: path already exists')
    exit()
else:
    os.mkdir(save_path)
shutil.copy('config.py', save_path)

if prior_gridding:
    precision_array = np.logspace(-4, np.log10(5), 40)
    precision_array = np.append(precision_array, [0.01, 0.1, 1])
    precision_array = np.sort(precision_array) * u.deg.to(u.rad)

    pivot_angle_index = 2
    one_prior = True
    if one_prior:
        prior_indices = [pivot_angle_index, pivot_angle_index+2]
    else:
        prior_indices = [0, 6]
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
    one_over_f_mode=one_over_f_mode, instrument=INSTRU, overwrite_freq=overwrite_freq)

S_cmb_name = 'S_cmb_n{}_s{}_r{:1}_b{:1.1e}'.format(nside, nsim, r_true, beta_true.value).replace(
    '.', 'p') + '.npy'
print(S_cmb_name)

data_model, fg_freq_maps, n_obspix = get_SFN(data, model_data, path_BB,
                                             S_cmb_name, spectral_flag)

'''Sky model for spectral likelihood'''
model = get_model(
    miscal_angles_array=initmodel_miscal, bir_angle=beta_true*0,
    frequencies_by_instrument_array=freq_by_instru,
    nside=nside, spectral_params=[1.54, 20, -3],
    sky_model='c1s0d0', sensitiviy_mode=sensitiviy_mode,
    one_over_f_mode=one_over_f_mode, instrument=INSTRU, overwrite_freq=overwrite_freq)
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
bias_input = 0

for prior_precision, prior_matrix in zip(precision_array, prior_matrix_array):
    input_angles[pivot_angle_index] = true_miscal_angles[pivot_angle_index].value + bias_input
    input_angle_array.append(copy.deepcopy(input_angles))

    results_min = minimize(constrained_chi2, angle_constrain_start, args=(
        data_model, model, pivot_angle_index, input_angles[pivot_angle_index], True, params),
        tol=1e-18, options={'maxiter': 1000}, method='L-BFGS-B',
        bounds=bounds[1:])
    print(results_min)
    print('results - spectral_true = ',
          results_min.x[:freq_number-1] - np.delete(true_miscal_angles, pivot_angle_index).value)
    print('results - spectral_true = ', results_min.x[-2] - 1.54)
    print('results - spectral_true = ', results_min.x[-1] + 3)
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
        one_over_f_mode=one_over_f_mode, instrument=INSTRU, overwrite_freq=overwrite_freq)

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
    IPython.embed()
    '''Cosmo likelihood minimisation'''
    Cl_noise, ell_noise = get_noise_Cl(
        model_results.mix_effectiv, lmax+1, fsky,
        sensitiviy_mode, one_over_f_mode,
        instrument=INSTRU, onefreqtest=1-spectral_flag)
    Cl_noise = Cl_noise[lmin-2:]
    ell_noise = ell_noise[lmin-2:]
    Cl_noise_matrix = np.zeros([2, 2, Cl_noise.shape[0]])
    Cl_noise_matrix[0, 0] = Cl_noise
    Cl_noise_matrix[1, 1] = Cl_noise
    #
    matrix_fg_yy = np.array([[Cl_fg['yy'][0], Cl_fg['yy'][2]],
                             [Cl_fg['yy'][2], Cl_fg['yy'][1]]])

    tr_SigmaYY = np.einsum('ij,jimnl->mnl', sigma_spectral, Cl_residuals_matrix['YY'])
    Cl_data = Cl_residuals_matrix['yy'] + Cl_residuals_matrix['zy'] + \
        Cl_residuals_matrix['yz'] + tr_SigmaYY + Cl_noise_matrix

    cosmo_params = [0.01, beta_true.value+prior_precision,
                    true_miscal_angles[pivot_angle_index].value + prior_precision]

    bounds_cosmo2 = ((-0.01, 5), (-np.pi/3, np.pi/3), (-np.pi/3, np.pi/3))

    pivot_precision = prior_precision
    bounds_rand = ((-0.01, 5), (0, 3*u.deg.to(u.rad)),
                   (input_angles[pivot_angle_index]-3*pivot_precision, input_angles[pivot_angle_index]+3*pivot_precision))

    cosmo_array_start = np.random.uniform(
        np.array(bounds_rand)[:, 0], np.array(bounds_rand)[:, 1])
    print('cosmo_array_start=', cosmo_array_start)
    result_cosmo = minimize(constrained_cosmo, cosmo_array_start,
                            args=(Cl_fid, Cl_data, Cl_noise_matrix, dWA_cmb, sigma_spectral,
                                  WA_cmb, VA_cmb, prior_matrix, input_angles*u.rad,
                                  pivot_angle_index, angle_eval,  ell, fsky, True),
                            bounds=bounds_cosmo2)
    true_cosmo_params = [r_true, beta_true.value, true_miscal_angles[pivot_angle_index].value]
    print('cosmo res - true =', result_cosmo.x - true_cosmo_params)
    cosmo_array.append(result_cosmo.x)

    '''Fisher cosmo estimation'''
    WACAW, VACAW, WACAV, tr_SigmaYY_cmb, YY_cmb = get_model_WACAW_WACAVeheh(result_cosmo.x,
                                                                            Cl_fid, dWA_cmb, sigma_spectral, WA_cmb, VA_cmb)

    diff_pivot_result = result_cosmo.x[-1] - input_angles[pivot_angle_index]
    rot_pivot = np.array([[np.cos(2*diff_pivot_result), np.sin(2*diff_pivot_result)],
                          [-np.sin(2*diff_pivot_result), np.cos(2*diff_pivot_result)]])

    Cl_data_fish_ = WACAW + VACAW + WACAV + tr_SigmaYY_cmb + Cl_noise_matrix
    Cl_data_fish = np.einsum('ij,jkl,km->iml', rot_pivot.T, Cl_data_fish_, rot_pivot)

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

    rot_dB = np.einsum('ij,jkl,km->iml', rot_pivot.T, (WACAWdb +
                                                       VACAWdb + WACAVdb + tr_SigmaYYdB), rot_pivot)
    rot_dr = np.einsum('ij,jkl,km->iml', rot_pivot.T, (WACAWdr +
                                                       VACAWdr + WACAVdr + tr_SigmaYYdr), rot_pivot)

    deriv_rot_pivot = np.array([[-2*np.sin(2*diff_pivot_result), 2*np.cos(2*diff_pivot_result)],
                                [-2*np.cos(2*diff_pivot_result), -2*np.sin(2*diff_pivot_result)]])
    dCldalpha = np.einsum('ij,jkl,km->iml', deriv_rot_pivot.T, Cl_data_fish_, rot_pivot) + \
        np.einsum('ij,jkl,km->iml', rot_pivot.T, Cl_data_fish_, deriv_rot_pivot)

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
    inv_sigma_miscal = np.linalg.inv(sigma_spectral[:5, :5])

    PI_fisher = 2*prior_element_pivot + 2 * np.ones(5).T.dot(inv_sigma_miscal).dot(np.ones(5)) -\
        2 * np.ones(5).T.dot(inv_sigma_miscal).dot(np.linalg.inv(inv_sigma_miscal +
                                                                 prior_matrix_nopivot)).dot(inv_sigma_miscal).dot(np.ones(5))
    fisher_cosmo_matrix_prior = copy.deepcopy(fisher_cosmo_matrix)
    fisher_cosmo_matrix_prior[-1, -1] += PI_fisher/2.
    fisher_cosmo_prior_array.append(fisher_cosmo_matrix_prior)
    fisher_cosmo_array.append(fisher_cosmo_matrix)

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
if cosmo_MCMC_flag:
    init_cosmo_MCMC = np.random.normal(
        result_cosmo.x, [0.01, prior_precision, prior_precision], (2*3, 3))
    nwalkers_cosmo = 2 * 3
    sampler_cosmo = EnsembleSampler(
        nwalkers_cosmo, 3, constrained_cosmo, args=[
            Cl_fid, Cl_data, Cl_noise_matrix, dWA_cmb,
            sigma_spectral, WA_cmb, VA_cmb, prior_matrix,
            angle_eval*u.rad, pivot_angle_index, angle_eval,
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
IPython.embed()


pivot_range = np.arange(0, 2, 2/1000)*u.deg.to(u.rad)
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
