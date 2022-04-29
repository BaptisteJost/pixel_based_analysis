from emcee import EnsembleSampler
from config import *
import shutil
from residuals import get_SFN, get_diff_list, get_diff_diff_list,\
    get_residuals, likelihood_exploration, get_noise_Cl, get_model_WACAW_WACAVeheh, run_double_MC
from pixel_based_angle_estimation import data_and_model_quick, get_model, \
    get_chi_squared_local, run_MCMC
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
    precision_array = np.logspace(-2, np.log10(5), 20)
    precision_array[7] = 0.1
    precision_array[14] = 1
    index_precision = int(args.precision_index)
    print(index_precision)
    print('prior gridding precision =', precision_array[index_precision], 'deg')
    prior_flag = True
    prior_indices = []
    if prior_flag:
        prior_indices = [2, 3]

    prior_precision = (precision_array[index_precision]*u.deg).to(u.rad).value
    np.save(save_path+'prior_precision.npy',
            np.array([index_precision, precision_array[index_precision]]))
    angle_prior = []
    if prior_flag:
        for d in range(freq_number):
            angle_prior.append([true_miscal_angles.value[d], prior_precision, int(d)])
        angle_prior = np.array(angle_prior[prior_indices[0]: prior_indices[-1]])
    prior_matrix = np.zeros([len(params), len(params)])
    if prior_flag:
        for i in range(prior_indices[0], prior_indices[-1]):
            prior_matrix[i, i] += 1/(prior_precision**2)


'''Sky signal simulation'''
data, model_data = data_and_model_quick(
    miscal_angles_array=true_miscal_angles, bir_angle=beta_true*0,
    frequencies_by_instrument_array=freq_by_instru, nside=nside,
    sky_model=sky_model, sensitiviy_mode=sensitiviy_mode,
    one_over_f_mode=one_over_f_mode, instrument=INSTRU, overwrite_freq=overwrite_freq)

S_cmb_name = 'S_cmb_n{}_s{}_r{:1}_b{:1.1e}'.format(nside, nsim, r_true, beta_true.value).replace(
    '.', 'p') + '.npy'
print(S_cmb_name)

data_model, fg_freq_maps, n_obspix = get_SFN(data, model_data, path_BB, S_cmb_name, spectral_flag)

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

IPython.embed()
'''General spectral likelihood minimisation'''
results_min = minimize(get_chi_squared_local, angle_array_start, args=(
    data_model, model, prior_flag, [], angle_prior, False, spectral_flag, 1, True, params),
    tol=1e-18, options={'maxiter': 1000}, jac=spectral_first_deriv, method='L-BFGS-B',
    bounds=bounds)
print(results_min)
print('results - spectral_true = ', results_min.x[:freq_number] - true_miscal_angles.value)
if spectral_flag:
    print('results - spectral_true = ', results_min.x[-2] - 1.54)
    print('results - spectral_true = ', results_min.x[-1] + 3)

if not spectral_flag:
    results_min.x = np.append(results_min.x, [1, -1])

'''Sky model from spectral likelihood results'''
model_results1 = get_model(
    results_min.x[: freq_number], bir_angle=beta_true*0,
    frequencies_by_instrument_array=freq_by_instru, nside=nside,
    spectral_params=[results_min.x[-2], 20, results_min.x[-1]],
    sky_model='c1s0d0', sensitiviy_mode=sensitiviy_mode,
    one_over_f_mode=one_over_f_mode, instrument=INSTRU, overwrite_freq=overwrite_freq)
if not spectral_flag:
    model_results1.bir_matrix = model_results1.bir_matrix[:2, :2]
    model_results1.mixing_matrix = np.array([[1, 0], [0, 1]]*len(model_results1.frequencies))
    model_results1.get_projection_op()

'''Spectral Fisher matrix estimation'''
diff_list_res = get_diff_list(model_results1, params)
diff_diff_list_res = get_diff_diff_list(model_results1, params)
fisher_matrix_spectral = fisher_new(data_model, model_results1,
                                    diff_list_res, diff_diff_list_res, params)
fisher_matrix_prior_spectral = fisher_matrix_spectral + prior_matrix
sigma_spectral = np.linalg.inv(fisher_matrix_prior_spectral)

'''Spectral likelihood MCMC sampling'''
if spectral_MCMC_flag:
    print()
    init_spectral_MCMC = np.random.normal(results_min.x, np.diag(
        sigma_spectral)*2, (spectral_walker_per_dim*spectral_dim, spectral_dim))

    flat_samples, cosmo_list = run_double_MC(
        init_spectral_MCMC, data_model, model, nsteps_spectral, prior=prior_flag,
        fixed_miscal_angles=[], miscal_priors=angle_prior,
        birefringence=birefringence_flag,
        spectral_index=spectral_flag,
        lmin=lmin, lmax=lmax, fsky=fsky,
        sensitiviy_mode=sensitiviy_mode, one_over_f_mode=one_over_f_mode,
        INSTRU=INSTRU,
        cmb_spectra=spectra_true, true_A_cmb=model_data.mix_effectiv[:, :2],
        Cl_fid=Cl_fid, method_cosmo=method_cosmo)
    # flat_samples, flat_samples_raw = run_MCMC(
    #     data_model, model, sampled_miscal_freq=freq_number,
    #     nsteps=nsteps_spectral, discard_num=discard_spectral,
    #     sampled_birefringence=birefringence_flag, prior=prior_flag,
    #     walker_per_dim=spectral_walker_per_dim, angle_prior=angle_prior,
    #     spectral_index=spectral_flag, return_raw_samples=True,
    #     save=False, path=None, parallel=False, nside=nside,
    #     prior_precision=prior_precision, prior_index=prior_indices,
    #     true_miscal_angles=true_miscal_angles, p0=init_spectral_MCMC)

    # np.save(save_path+'raw_spectral_samples.npy', flat_samples_raw)
    np.save(save_path+'spectral_samples.npy', flat_samples)
    np.save(save_path+'cosmo_list.npy', cosmo_list)
    # IPython.embed()
    # res_MC = np.mean(flat_samples, 0)
    # results_min.x = res_MC
    # del flat_samples, flat_samples_raw
    del flat_samples  # , flat_samples_raw

'''Sky model from spectral likelihood results'''
model_results = get_model(
    results_min.x[: freq_number], bir_angle=beta_true*0,
    frequencies_by_instrument_array=freq_by_instru, nside=nside,
    spectral_params=[results_min.x[-2], 20, results_min.x[-1]],
    sky_model='c1s0d0', sensitiviy_mode=sensitiviy_mode,
    one_over_f_mode=one_over_f_mode, instrument=INSTRU, overwrite_freq=overwrite_freq)
if not spectral_flag:
    model_results.bir_matrix = model_results.bir_matrix[:2, :2]
    model_results.mixing_matrix = np.array([[1, 0], [0, 1]]*len(model_results.frequencies))
    model_results.get_projection_op()
'''Spectral Fisher matrix estimation'''
diff_list_res = get_diff_list(model_results, params)
diff_diff_list_res = get_diff_diff_list(model_results, params)
fisher_matrix_spectral = fisher_new(data_model, model_results,
                                    diff_list_res, diff_diff_list_res, params)
fisher_matrix_prior_spectral = fisher_matrix_spectral + prior_matrix
sigma_spectral = np.linalg.inv(fisher_matrix_prior_spectral)
# print('!!! WARNING diag sigma spectral put by hand for test !!!')
# sigma_spectral = np.diag(np.diag(sigma_spectral))
#
# '''Spectral likelihood MCMC sampling'''
# if spectral_MCMC_flag:
#     print()
#     init_spectral_MCMC = np.random.normal(results_min.x, np.diag(
#         sigma_spectral)*2, (spectral_walker_per_dim*spectral_dim, spectral_dim))
#     flat_samples, flat_samples_raw = run_MCMC(
#         data_model, model, sampled_miscal_freq=freq_number,
#         nsteps=nsteps_spectral, discard_num=discard_spectral,
#         sampled_birefringence=birefringence_flag, prior=prior_flag,
#         walker_per_dim=spectral_walker_per_dim, angle_prior=angle_prior,
#         spectral_index=spectral_flag, return_raw_samples=True,
#         save=False, path=None, parallel=False, nside=nside,
#         prior_precision=prior_precision, prior_index=prior_indices,
#         true_miscal_angles=true_miscal_angles, p0=init_spectral_MCMC)
#
#     np.save(save_path+'raw_spectral_samples.npy', flat_samples_raw)
#     np.save(save_path+'spectral_samples.npy', flat_samples)
#     del flat_samples, flat_samples_raw

'''Residuals computation'''
stat, bias, var, Cl_fg, Cl_cmb, Cl_residuals_matrix, ell, W_cmb, dW_cmb, ddW_cmb = get_residuals(
    model_results, fg_freq_maps, sigma_spectral, lmin, lmax, fsky, params,
    cmb_spectra=spectra_true, true_A_cmb=model_data.mix_effectiv[:, :2])

WA_cmb = W_cmb.dot(model_results.mix_effectiv[:, :2])
dWA_cmb = dW_cmb.dot(model_results.mix_effectiv[:, :2])
W_dBdB_cmb = ddW_cmb.dot(model_results.mix_effectiv[:, :2])
VA_cmb = np.einsum('ij,ij...->...', sigma_spectral, W_dBdB_cmb[:, :])

key = 'YY'
shape_temp = np.array(Cl_cmb[key].shape)
shape_temp[-1] = lmax-lmin
temp_matrix = np.zeros(shape_temp)
temp_matrix = np.array([[Cl_fg[key][:, :, 0], Cl_fg[key][:, :, 2]], [
    Cl_fg[key][:, :, 2], Cl_fg[key][:, :, 1]]])
temp_matrix_shape = np.einsum('ijklm->klijm', temp_matrix)
tr_SigmaYY_fg = np.einsum('ij,jimnl->mnl', sigma_spectral, temp_matrix_shape)

# IPython.embed()
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

tr_SigmaYY = np.einsum('ij,jimnl->mnl', sigma_spectral, Cl_residuals_matrix['YY'])
Cl_data = Cl_residuals_matrix['yy'] + Cl_residuals_matrix['zy'] + \
    Cl_residuals_matrix['yz'] + tr_SigmaYY + Cl_noise_matrix
print()
print(cosmo_array_start)
results_cosmp = minimize(likelihood_exploration, cosmo_array_start, args=(
    Cl_fid, Cl_data, Cl_noise_matrix + 0*tr_SigmaYY_fg,  dWA_cmb, sigma_spectral, WA_cmb, VA_cmb, ell, fsky),
    bounds=bounds_cosmo, tol=1e-18,
    method=method_cosmo, jac=jac_cosmo_min)
print(results_cosmp)
print('results - true cosmo = ', results_cosmp.x - np.array([r_true, beta_true.value]))
# IPython.embed()


WACAW, VACAW, WACAV, tr_SigmaYY_cmb, YY_cmb = get_model_WACAW_WACAVeheh(results_cosmp.x,
                                                                        Cl_fid, dWA_cmb, sigma_spectral, WA_cmb, VA_cmb)
np.save(save_path+'WACAW.npy', WACAW)
np.save(save_path+'VACAW.npy', VACAW)
np.save(save_path+'WACAV.npy', WACAV)
np.save(save_path+'Cl_noise_matrix.npy', Cl_noise_matrix)
np.save(save_path+'tr_SigmaYY.npy', tr_SigmaYY)
np.save(save_path+'tr_SigmaYY_cmb.npy', tr_SigmaYY_cmb)
np.save(save_path+'tr_SigmaYY_fg.npy', tr_SigmaYY_fg)
np.save(save_path+'YY.npy', Cl_residuals_matrix['YY'])

# IPython.embed()
'''Fisher cosmo estimation'''
Cl_data_fish = WACAW + VACAW + WACAV + tr_SigmaYY_cmb + Cl_noise_matrix + tr_SigmaYY_fg*0
# Cl_data_fish = Cl_data
Cl_cmb_model_fish = np.zeros([4, Cl_fid['EE'].shape[0]])
Cl_cmb_model_fish[1] = deepcopy(Cl_fid['EE'])
Cl_cmb_model_fish[2] = deepcopy(Cl_fid['BlBl'])*A_lens_true + \
    deepcopy(Cl_fid['BuBu']) * results_cosmp.x[0]

Cl_cmb_model_fish_r = np.zeros([4, Cl_fid['EE'].shape[0]])
Cl_cmb_model_fish_r[2] = deepcopy(Cl_fid['BuBu']) * 1

deriv1 = power_spectra_obj(cl_rotation_derivative(
    Cl_cmb_model_fish.T, results_cosmp.x[1]*u.rad), ell)
dCldbeta = np.array(
    [[deriv1.spectra[:, 1], deriv1.spectra[:, 4]],
     [deriv1.spectra[:, 4], deriv1.spectra[:, 2]]])
WACAVdb = np.einsum('ij,jkl,km->iml', WA_cmb, dCldbeta, VA_cmb.T)
VACAWdb = np.einsum('ij,jkl,km->iml', VA_cmb, dCldbeta, WA_cmb.T)
WACAWdb = np.einsum('ij,jkl,km->iml', WA_cmb, dCldbeta, WA_cmb.T)


# deriv_matrix_beta = power_spectra_obj(np.array(
#     [[deriv1.spectra[:, 1], deriv1.spectra[:, 4]],
#      [deriv1.spectra[:, 4], deriv1.spectra[:, 2]]]).T, deriv1.ell)

dCldr = get_dr_cov_bir_EB(
    Cl_cmb_model_fish_r.T, results_cosmp.x[1]*u.rad)
WACAVdr = np.einsum('ij,jkl,km->iml', WA_cmb, dCldr, VA_cmb.T)
VACAWdr = np.einsum('ij,jkl,km->iml', VA_cmb, dCldr, WA_cmb.T)
WACAWdr = np.einsum('ij,jkl,km->iml', WA_cmb, dCldr, WA_cmb.T)
# deriv_matrix_r = power_spectra_obj(get_dr_cov_bir_EB(
#     Cl_cmb_model_fish_r.T, results_cosmp.x[1]*u.rad).T, ell)

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

deriv_matrix_beta2 = power_spectra_obj((WACAWdb + VACAWdb + WACAVdb + tr_SigmaYYdB).T, deriv1.ell)
deriv_matrix_r2 = power_spectra_obj((WACAWdr + VACAWdr + WACAVdr + tr_SigmaYYdr).T, ell)

# Cl_data_spectra = power_spectra_obj(Cl_data.T, ell)
Cl_data_spectra = power_spectra_obj(Cl_data_fish.T, ell)

# fish_beta = fisher_pws(Cl_data_spectra, deriv_matrix_beta, fsky)
# fish_r = fisher_pws(Cl_data_spectra, deriv_matrix_r, fsky)
# fish_beta_r = fisher_pws(Cl_data_spectra, deriv_matrix_beta, fsky, deriv2=deriv_matrix_r)
# fisher_cosmo_matrix1 = np.array([[fish_r, fish_beta_r],
#                                  [fish_beta_r, fish_beta]])

fish_beta2 = fisher_pws(Cl_data_spectra, deriv_matrix_beta2, fsky)
fish_r2 = fisher_pws(Cl_data_spectra, deriv_matrix_r2, fsky)
fish_beta_r2 = fisher_pws(Cl_data_spectra, deriv_matrix_beta2, fsky, deriv2=deriv_matrix_r2)
fisher_cosmo_matrix = np.array([[fish_r2, fish_beta_r2],
                                [fish_beta_r2, fish_beta2]])

'''Cosmo MCMC '''
if cosmo_MCMC_flag:
    init_cosmo_MCMC = np.random.normal(results_cosmp.x, np.diag(
        np.linalg.inv(fisher_cosmo_matrix))*2, (cosmo_walker_per_dim*cosmo_dim, cosmo_dim))
    nwalkers_cosmo = cosmo_walker_per_dim * cosmo_dim
    sampler_cosmo = EnsembleSampler(
        nwalkers_cosmo, cosmo_dim, likelihood_exploration, args=[
            Cl_fid, Cl_data, Cl_noise_matrix + 0*tr_SigmaYY_fg, dWA_cmb, sigma_spectral, WA_cmb, VA_cmb, ell, fsky, False],
        pool=None)
    sampler_cosmo.reset()
    sampler_cosmo.run_mcmc(init_cosmo_MCMC, nsteps_cosmo, progress=True)
    samples_cosmo_raw = sampler_cosmo.get_chain()
    samples_cosmo = sampler_cosmo.get_chain(discard=discard_cosmo, flat=True)
    np.save(save_path+'raw_cosmo_samples.npy', samples_cosmo_raw)
    np.save(save_path+'cosmo_samples.npy', samples_cosmo)
    del samples_cosmo_raw, samples_cosmo


'''Saving results'''
np.save(save_path+'fisher_cosmo.npy', fisher_cosmo_matrix)
np.save(save_path+'fisher_spectral.npy', fisher_matrix_prior_spectral)
np.save(save_path+'cosmo_results.npy', results_cosmp.x)
np.save(save_path+'spectral_results.npy', results_min.x)
