from config import *
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
import matplotlib.pyplot as plt
import copy
import time
from residuals import constrained_cosmo, get_ys_alms
from emcee import EnsembleSampler
import shutil
import bjlib.lib_project as lib
from residuals import get_SFN, get_diff_list, get_diff_diff_list,\
    get_residuals, likelihood_exploration, get_noise_Cl, \
    get_model_WACAW_WACAVeheh, run_double_MC, multi_freq_get_sky_fg
from residuals import get_W, get_ys_Cls
from bjlib import V3calc as V3

# from multiprocessing import Pool
# from schwimmbad import MPIPool
# import sys
from mpi4py import MPI


def spectral_sampling(spectral_params, ddt, model_skm, total_prior_matrix, prior_centre, minimisation=False):
    angle_eval = spectral_params[:freq_number]*u.rad
    fg_params = spectral_params[freq_number:freq_number+2]

    '''=========Check bounds========='''
    if not 0.5 <= fg_params[-2] < 2.5 or not -5 <= fg_params[-1] <= -1:
        print('bla')
        if not minimisation:
            return -np.inf

    if np.any(np.array(angle_eval) < (-0.5*np.pi)) or np.any(np.array(angle_eval) > (0.5*np.pi)):
        print('blou')
        if not minimisation:
            return -np.inf
    model_skm.evaluate_mixing_matrix([fg_params[-2], 20, fg_params[-1]])

    model_skm.miscal_angles = angle_eval

    model_skm.bir_angle = 0

    model_skm.get_miscalibration_angle_matrix()
    model_skm.get_bir_matrix()

    model_skm.get_projection_op()

    spectral_like = np.einsum('ij,ji->...', -model_skm.projection+model_skm.inv_noise,
                              ddt)
    Prior = (angle_eval.value - prior_centre).T.dot(
        total_prior_matrix[:freq_number, :freq_number]).dot(angle_eval.value - prior_centre)
    if minimisation:
        return -spectral_like + Prior
    else:
        return spectral_like - Prior


def from_spectra_to_cosmo(spectral_params, model_skm, sensitiviy_mode, one_over_f_mode, beam_corrected, one_over_ell):
    angle_eval = spectral_params[:freq_number]*u.rad
    fg_params = spectral_params[freq_number:freq_number+2]

    model_skm.evaluate_mixing_matrix([fg_params[-2], 20, fg_params[-1]])
    model_skm.miscal_angles = angle_eval
    model_skm.bir_angle = 0
    model_skm.get_miscalibration_angle_matrix()
    model_skm.get_bir_matrix()
    model_skm.get_projection_op()

    W = get_W(model_skm)
    A = copy.deepcopy(model_skm.mix_effectiv)

    V3_results = V3.so_V3_SA_noise(sensitiviy_mode, one_over_f_mode,
                                   SAC_yrs_LF=1, f_sky=fsky, ell_max=lmax+1, beam_corrected=beam_corrected)
    if one_over_ell:
        noise_nl = np.repeat(V3_results[1], 2, 0)
    else:
        noise_nl = np.repeat(V3_results[-1], 2, 0)
    nl_inv = 1/noise_nl
    AtNA = np.einsum('fi, fl, fj -> lij', A, nl_inv, A)
    inv_AtNA = np.linalg.inv(AtNA)
    noise_cl = inv_AtNA.swapaxes(-3, -1)[..., lmin-2:]
    Cl_noise = noise_cl[0, 0]

    Cl_noise_matrix = np.zeros([2, 2, Cl_noise.shape[0]])
    Cl_noise_matrix[0, 0] = Cl_noise
    Cl_noise_matrix[1, 1] = Cl_noise

    return Cl_noise_matrix, A, W


def cosmo_sampling(cosmo_params, Cl_cmb_data_matrix, reshape_fg_ys, Cl_fid, Cl_noise_matrix, W, A, minimisation=False):
    r = cosmo_params[0]
    beta = cosmo_params[1]*u.rad

    Wfg_ys = W[:2].dot(reshape_fg_ys)
    Wfg_ys_TEB = np.zeros([3, Wfg_ys.shape[-1]], dtype='complex')
    Wfg_ys_TEB[1:] = Wfg_ys
    temp_Cl_fg = get_ys_Cls(Wfg_ys_TEB, Wfg_ys_TEB, lmax, fsky)[:, lmin:]
    Cl_SM_fg = np.array([[temp_Cl_fg[0], temp_Cl_fg[2]], [
        temp_Cl_fg[2], temp_Cl_fg[1]]])

    Cl_SM_cmb = np.einsum('ij,jkl,km->iml', W[:2], Cl_cmb_data_matrix, W[:2].T)[:, :, lmin:lmax+1]

    WEW = Cl_SM_cmb + Cl_SM_fg
    Cl_data_rot = WEW + Cl_noise_matrix

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

    inv_model = np.linalg.inv(Cl_model_total.T).T
    ell = np.arange(lmin, lmax+1)
    dof = (2 * ell + 1) * fsky
    dof_over_Cl = dof * inv_model

    first_term_ell = np.einsum('ijl,jkl->ikl', dof_over_Cl, Cl_data_rot)

    first_term = np.sum(np.trace(first_term_ell))

    logdetC = np.sum(dof*np.log(np.linalg.det(Cl_model_total.T)))
    likelihood_cosmo = first_term + logdetC
    if minimisation:
        return likelihood_cosmo
    else:
        return -likelihood_cosmo


def spectral_cosmo_onthefly(spectral_params, cosmo_params_list, ddt, model_skm, prior_matrix, prior_centre, Cl_cmb_data_matrix, reshape_fg_ys):
    # start = time.time()
    spec_like = spectral_sampling(spectral_params, ddt, model_skm,
                                  prior_matrix, prior_centre, False)
    # print('time spec', time.time() - start)

    # start = time.time()
    Cl_noise_matrix, A, W = from_spectra_to_cosmo(
        spectral_params, model_skm, sensitiviy_mode=1, one_over_f_mode=2, beam_corrected=False, one_over_ell=False)
    # print('time init W', time.time() - start)

    # start = time.time()
    init_min = np.random.normal([0, 0], [0.1, 0.1], 2)
    # init_min[0] = np.abs(init_min[0])
    cosmo_params = minimize(cosmo_sampling, init_min, args=(Cl_cmb_data_matrix,
                                                            reshape_fg_ys, Cl_fid, Cl_noise_matrix, W, A), bounds=bounds_cosmo)
    # print('time min', time.time() - start)

    # start = time.time()
    cosmo_params_list.append(cosmo_params.x)
    # print('time append', time.time() - start)

    return spec_like


def MCMC_test(ddt, model_skm, prior_matrix, prior_centre, Cl_cmb_data_matrix, reshape_fg_ys, nsteps, discard):

    scatter = [prior_precision]*freq_number
    scatter.append(0.1)
    scatter.append(0.1)
    scatter.append(0.01)
    scatter.append(prior_precision)
    scatter = np.array(scatter)
    centre_random = prior_centre.tolist()
    centre_random.append(1.54)
    centre_random.append(-3)
    centre_random.append(0)
    centre_random.append(0)
    centre_random = np.array(centre_random)
    init_min = np.random.normal(centre_random, scatter, 10)
    init_MCMC = np.random.normal(
        centre_random[:8], scatter[:8], (2*8, 8))
    nwalkers = 2 * 8
    start = time.time()
    sampler = EnsembleSampler(
        nwalkers, 8, spectral_sampling, args=[
            ddt, model_skm, prior_matrix, prior_centre, False])
    sampler.reset()
    start = time.time()
    sampler.run_mcmc(init_MCMC, nsteps, progress=True)
    end = time.time()
    print('time MCMC = ', end - start)
    samples_raw = sampler.get_chain()
    samples = sampler.get_chain(discard=discard, flat=True)

    # spectral_params = minimize(spectral_sampling, init_min[:8], args=(
    #     ddt, model_skm, prior_matrix, prior_centre, True))
    Cl_noise_matrix, A, W = from_spectra_to_cosmo(
        spectral_params.x, model_skm, sensitiviy_mode=1, one_over_f_mode=2, beam_corrected=False, one_over_ell=False)
    cosmo_params = minimize(cosmo_sampling, init_min[8:], args=(Cl_cmb_data_matrix,
                                                                reshape_fg_ys, Cl_fid, Cl_noise_matrix, W, A))
    return 0


def total_likelihood_sampling(total_params, Cl_fid, total_prior_matrix, prior_centre, pivot_angle_index, fsky, ddt, model_skm, true_A_cmb, fg_freq_maps, reshape_fg_ys, minimisation=False, params=None):
    '''=========Check bounds========='''
    # print(total_params)
    angle_array = np.delete(total_params[:freq_number], pivot_angle_index)*u.rad
    fg_params = total_params[freq_number:freq_number+2]
    start_cosmo_params = freq_number+2
    r = total_params[start_cosmo_params]
    beta = total_params[start_cosmo_params+1]*u.rad
    pivot = total_params[pivot_angle_index]*u.rad

    angle_eval = total_params[:freq_number]*u.rad

    if not 0.5 <= fg_params[-2] < 2.5 or not -5 <= fg_params[-1] <= -1:
        print('bla')
        if not minimisation:
            return -np.inf

    if np.any(np.array(angle_array) < (-0.5*np.pi)) or np.any(np.array(angle_array) > (0.5*np.pi)):
        print('blou')
        if not minimisation:
            return -np.inf

    if not minimisation:
        if r > 5 or r < -0.01:
            print('bla1')
            return -np.inf
        elif beta.value < -np.pi/2 or beta.value > np.pi/2:
            print('bla2')
            return -np.inf
        elif pivot.value < -np.pi/2 or pivot.value > np.pi/2:
            print('bla3')
            return -np.inf
    '''=========Spectral likelihood========='''

    model_skm.evaluate_mixing_matrix([fg_params[-2], 20, fg_params[-1]])

    model_skm.miscal_angles = angle_eval

    model_skm.bir_angle = 0

    model_skm.get_miscalibration_angle_matrix()
    model_skm.get_bir_matrix()

    model_skm.get_projection_op()

    spectral_like = np.einsum('ij,ji->...', -model_skm.projection+model_skm.inv_noise,
                              ddt)
    '''
    Cl_noise, ell_noise = get_noise_Cl(
        model_skm.mix_effectiv, lmax+1, fsky,
        sensitiviy_mode, one_over_f_mode,
        instrument=INSTRU, onefreqtest=1-spectral_flag)
    ell_noise = ell_noise[lmin-2:]
    ell = ell_noise
    '''
    ell = np.arange(lmin, lmax+1)
    W = get_W(model_skm)

    V3_results = V3.so_V3_SA_noise(sensitiviy_mode, 2,
                                   SAC_yrs_LF=1, f_sky=fsky, ell_max=lmax+1, beam_corrected=False)
    # noise_nl1 = np.repeat(V3_results[1], 2, 0)
    noise_nl = np.repeat(V3_results[-1], 2, 0)  # test ell white noise

    WNW = np.einsum('fi, fl, fj -> lij', W.T, noise_nl, W.T)
    WcmbNWcmb = np.einsum('fi, fl, fj -> lij', W.T[:, :2], noise_nl, W.T[:, :2])
    noise_clW = WNW.swapaxes(-3, -1)[..., lmin-2:]
    noise_clWcmb = WcmbNWcmb.swapaxes(-3, -1)[..., lmin-2:]

    nl_inv = 1/noise_nl
    # nl_inv_matrix = np.zeros([nl_inv.shape[0], nl_inv.shape[0], nl_inv.shape[-1]])
    # for i in range(12):
    #     nl_inv_matrix[i, i, :] = nl_inv[i]
    A = model_skm.mix_effectiv
    AtNA = np.einsum('fi, fl, fj -> lij', A, nl_inv, A)
    inv_AtNA = np.linalg.inv(AtNA)
    noise_cl = inv_AtNA.swapaxes(-3, -1)[..., lmin-2:]

    '''
    A_cmb = model_skm.mix_effectiv[:, :2]
    AtNAcmb = np.einsum('fi, fl, fj -> lij', A_cmb, nl_inv, A_cmb)
    inv_AtNAcmb = np.linalg.inv(AtNAcmb)
    noise_clcmb = inv_AtNAcmb.swapaxes(-3, -1)[..., lmin-2:]
    '''
    # IPython.embed()
    '''
    Wfg_ys = W.dot(reshape_fg_ys)
    Cl_SM_fg = np.zeros([Wfg_ys.shape[0], Wfg_ys.shape[0], lmax+1-lmin])
    for i in range(Wfg_ys.shape[0]//2):
        for j in range(Wfg_ys.shape[0]//2):
            # print(i, ' : ', 2*i)
            # print(j, ' : ', 2*j + 1)
            Wfg_ys_TEB1 = np.zeros([3, Wfg_ys.shape[-1]], dtype='complex')
            Wfg_ys_TEB1[1] = Wfg_ys[2*i]
            Wfg_ys_TEB1[2] = Wfg_ys[2*i + 1]
            Wfg_ys_TEB2 = np.zeros([3, Wfg_ys.shape[-1]], dtype='complex')
            Wfg_ys_TEB2[1] = Wfg_ys[2*j]
            Wfg_ys_TEB2[2] = Wfg_ys[2*j + 1]
            temp_Cl_fg = get_ys_Cls(Wfg_ys_TEB1, Wfg_ys_TEB2, lmax, fsky)[:, lmin:]
            Cl_SM_fg[2*i, 2*j] = temp_Cl_fg[0]
            Cl_SM_fg[2*j, 2*i] = temp_Cl_fg[0]
            Cl_SM_fg[2*i + 1, 2*j + 1] = temp_Cl_fg[1]
            Cl_SM_fg[2*j + 1, 2*i + 1] = temp_Cl_fg[1]
            Cl_SM_fg[2*i + 1, 2*j] = temp_Cl_fg[2]
            Cl_SM_fg[2*j, 2*i + 1] = temp_Cl_fg[2]
            Cl_SM_fg[2*i, 2*j + 1] = temp_Cl_fg[2]
            Cl_SM_fg[2*j + 1, 2*i] = temp_Cl_fg[2]
    '''
    Wfg_ys = W[:2].dot(reshape_fg_ys)
    Wfg_ys_TEB = np.zeros([3, Wfg_ys.shape[-1]], dtype='complex')
    Wfg_ys_TEB[1:] = Wfg_ys
    temp_Cl_fg = get_ys_Cls(Wfg_ys_TEB, Wfg_ys_TEB, lmax, fsky)[:, lmin:]
    Cl_SM_fg = np.array([[temp_Cl_fg[0], temp_Cl_fg[2]], [
        temp_Cl_fg[2], temp_Cl_fg[1]]])
    # '''
    WA_cmb = W[:2].dot(true_A_cmb)
    # WA_cmb = W.dot(true_A_cmb)
    Cl_cmb_data = np.zeros((2, 2, len(spectra_true[0])))
    Cl_cmb_data[0, 0] = spectra_true[1]
    Cl_cmb_data[1, 1] = spectra_true[2]
    Cl_cmb_data[1, 0] = spectra_true[4]
    Cl_cmb_data[0, 1] = spectra_true[4]
    # Cl_cmb_data_tot = np.zeros([6, 6, len(spectra_true[0])])
    # Cl_cmb_data_tot[:2, :2] = Cl_cmb_data
    Cl_SM_cmb = np.einsum('ij,jkl,km->iml', WA_cmb, Cl_cmb_data, WA_cmb.T)[:, :, lmin:lmax+1]
    # Cl_SM_cmb = np.einsum('ij,jkl,km->iml', WA_cmb, Cl_cmb_data_tot, WA_cmb.T)[:, :, lmin:lmax+1]

    WEW = Cl_SM_cmb + Cl_SM_fg

    # Cl_noise = Cl_noise[lmin-2:]
    # Cl_noise = noise_clcmb[0, 0]
    Cl_noise = noise_cl[0, 0]
    El_noise = noise_cl[0, 0]
    # El_noise = noise_cl
    # El_noise = noise_clWcmb[0, 0]
    # # Cl_noise = noise_clW[0, 0]

    '''
    El_noise_matrix = np.zeros([2, 2, El_noise.shape[0]])
    El_noise_matrix[0, 0] = El_noise
    El_noise_matrix[1, 1] = El_noise
    '''

    Cl_noise_matrix = np.zeros([2, 2, Cl_noise.shape[0]])
    Cl_noise_matrix[0, 0] = Cl_noise
    Cl_noise_matrix[1, 1] = Cl_noise

    Cl_cmb_model = np.zeros([4, Cl_fid['EE'].shape[0]])
    Cl_cmb_model[1] = copy.deepcopy(Cl_fid['EE'])
    Cl_cmb_model[2] = copy.deepcopy(Cl_fid['BlBl'])*1 + copy.deepcopy(Cl_fid['BuBu']) * r

    Cl_cmb_rot = lib.cl_rotation(Cl_cmb_model.T, beta).T

    Cl_cmb_rot_matrix = np.zeros([2, 2, Cl_cmb_rot.shape[-1]])
    Cl_cmb_rot_matrix[0, 0] = copy.deepcopy(Cl_cmb_rot[1])
    Cl_cmb_rot_matrix[1, 1] = copy.deepcopy(Cl_cmb_rot[2])
    Cl_cmb_rot_matrix[1, 0] = copy.deepcopy(Cl_cmb_rot[4])
    Cl_cmb_rot_matrix[0, 1] = copy.deepcopy(Cl_cmb_rot[4])

    # Cl_noise_test = copy.deepcopy(noise_cl)
    # Cl_noise_test[:2, :2] += Cl_cmb_rot_matrix

    Cl_model_total = Cl_cmb_rot_matrix + Cl_noise_matrix
    # Cl_model_total = Cl_noise_test

    Cl_data_rot = WEW + Cl_noise_matrix
    # Cl_data_rot = WEW + El_noise_matrix
    # Cl_data_rot = WEW + noise_cl

    inv_model = np.linalg.inv(Cl_model_total.T).T
    dof = (2 * ell + 1) * fsky
    dof_over_Cl = dof * inv_model

    first_term_ell = np.einsum('ijl,jkl->ikl', dof_over_Cl, Cl_data_rot)

    first_term = np.sum(np.trace(first_term_ell))

    logdetC = np.sum(dof*np.log(np.linalg.det(Cl_model_total.T)))
    # logdetN = np.sum(dof*np.log(np.linalg.det(noise_cl.T)))
    likelihood_cosmo = first_term + logdetC  # - logdetN  # + radek_jost_prior

    '''===========Prior term==========='''
    Prior = (angle_eval.value - prior_centre).T.dot(
        total_prior_matrix[:freq_number, :freq_number]).dot(angle_eval.value - prior_centre)
    # IPython.embed()
    if minimisation:
        # print('tot = ', -(spectral_like - likelihood_cosmo - Prior), ' spec = ',
        #       -spectral_like, ' cosmo = ', likelihood_cosmo, ' Prior = ', Prior)
        # , -spectral_like, likelihood_cosmo, Prior, logdetC, Cl_cmb_rot_matrix, Cl_noise_matrix, WEW, logdetN
        return -(spectral_like - likelihood_cosmo - Prior)
    else:
        return spectral_like - likelihood_cosmo - Prior


def get_SFN_maps(data, model_data, path_BB, S_cmb_map_name, spectral_flag=True):
    S_cmb_map = np.load(S_cmb_map_name)
    ddt_maps = np.einsum('ij,jkp,kl->ilp', model_data.mix_effectiv[:, :2],
                         S_cmb_map, model_data.mix_effectiv[:, :2].T).T
    del S_cmb_map
    data.get_pysm_sky()
    fg_freq_maps_full = data.miscal_matrix.dot(multi_freq_get_sky_fg(data.sky, data.frequencies))
    ddt_maps += np.einsum('ip,jp-> ijp', fg_freq_maps_full, fg_freq_maps_full).T
    data.get_mask(path_BB)
    mask = data.mask
    mask[(mask != 0) * (mask != 1)] = 0

    fg_freq_maps = fg_freq_maps_full*mask
    del fg_freq_maps_full

    ddt_maps += model_data.noise_covariance
    ddt_maps = ddt_maps.T
    ddt_maps *= mask
    n_obspix = np.sum(mask == 1)
    del mask

    # F = np.sum(ddt_fg, axis=-1)/n_obspix
    # data_model = n_obspix*(F*spectral_flag + ASAt + model_data.noise_covariance)
    return ddt_maps, fg_freq_maps*spectral_flag, n_obspix


def acceptance_ratio(likelihood, like_args, p, p_new, prior=None):
    # Return R, using the functions we created before
    if prior is not None:
        return min(1, ((likelihood(p_new, *like_args) / likelihood(p, *like_args)) * (prior(p_new) / prior(p))))
    else:
        return min(1, ((likelihood(p_new, *like_args) / likelihood(p, *like_args))))


def homemade_MCMC(likelihood, like_args, p, n_samples, burn_in, lag=1, prior=None):
    # Create empty list to store samples
    results = []

    # Initialzie a value of p
    # p = np.random.uniform(0, 1)

    # Define model parameters
    # n_samples = 25000
    # burn_in = 5000
    # lag = 5
    shape_param = p.shape
    # Create the MCMC loop
    for i in range(n_samples):
        # Propose a new value of p randomly from a uniform distribution between 0 and 1
        p_new = np.random.random_sample(shape_param)
        # Compute acceptance probability
        R = acceptance_ratio(likelihood, like_args, p, p_new, prior=prior)
        # Draw random sample to compare R to
        u = np.random.random_sample()
        # If R is greater than u, accept the new value of p (set p = p_new)
        if u < R:
            p = p_new
        # Record values after burn in - how often determined by lag
        if i > burn_in and i % lag == 0:
            results.append(p)
    return np.array(results)


def main():
    S_cmb_name = 'data/New_S_cmb_n{}_s{}_r{:1}_b{:1.1e}'.format(nside, nsim, r_true, beta_true.value).replace(
        '.', 'p') + '.npy'
    print(S_cmb_name)
    data, model_data = data_and_model_quick(
        miscal_angles_array=true_miscal_angles, bir_angle=beta_true*0,
        frequencies_by_instrument_array=freq_by_instru, nside=nside,
        sky_model=sky_model, sensitiviy_mode=sensitiviy_mode,
        one_over_f_mode=one_over_f_mode, instrument=INSTRU, overwrite_freq=overwrite_freq)
    true_A_cmb = model_data.mix_effectiv[:, :2]

    # ddt_maps, fg_freq_maps, n_obspix = get_SFN_maps(
    #     data, model_data, path_BB, S_cmb_map_name, spectral_flag)
    ddt, fg_freq_maps, n_obspix = get_SFN(
        data, model_data, path_BB, S_cmb_name, spectral_flag)

    miscal = 1*u.deg.to(u.rad)
    total_params = np.array([miscal, miscal, miscal, miscal, miscal, miscal,
                             1.54, -3., r_true, beta_true.value])
    prior_centre = true_miscal_angles.value
    pivot_angle_index = 2

    Cl_cmb_data = np.zeros((2, 2, len(spectra_true[0])))
    Cl_cmb_data[0, 0] = spectra_true[1]
    Cl_cmb_data[1, 1] = spectra_true[2]
    Cl_cmb_data[1, 0] = spectra_true[4]
    Cl_cmb_data[0, 1] = spectra_true[4]
    Cl_cmb_data_matrix = np.einsum('ij,jkl,km->iml', true_A_cmb,
                                   Cl_cmb_data, true_A_cmb.T)

    '''Sky model from spectral likelihood results'''
    model_skm = get_model(
        total_params[:6], bir_angle=beta_true*0,
        frequencies_by_instrument_array=freq_by_instru, nside=nside,
        spectral_params=[total_params[6], 20, total_params[7]],
        sky_model='c1s0d0', sensitiviy_mode=sensitiviy_mode,
        one_over_f_mode=one_over_f_mode, instrument=INSTRU, overwrite_freq=overwrite_freq)

    # dd_fg = np.einsum('ip,jp->ijp', fg_freq_maps, fg_freq_maps)
    fg_ys = get_ys_alms(y_Q=fg_freq_maps[::2], y_U=fg_freq_maps[1::2], lmax=lmax)
    reshape_fg_ys = fg_ys[:, 1:].reshape([12, 45451])
    IPython.embed()

    scatter = [prior_precision]*freq_number
    scatter.append(0.1)
    scatter.append(0.1)
    scatter.append(0.01)
    scatter.append(prior_precision)
    scatter = np.array(scatter)
    init_min = np.random.normal(total_params, scatter, 10)

    IPython.embed()
    cosmo_params_list = []
    start = time.time()
    spectral_cosmo_onthefly(init_min[:8], cosmo_params_list, ddt, model_skm,
                            prior_matrix, prior_centre, Cl_cmb_data_matrix, reshape_fg_ys)
    end = time.time()
    print('time on the fly = ', end - start)

    nsteps = 13000
    discard = 5000
    cosmo_params_list = []

    init_MCMC = np.random.normal(
        total_params[:8], scatter[:8], (2*8, 8))
    nwalkers = 2 * 8
    start = time.time()
    sampler_spec = EnsembleSampler(
        nwalkers, 8, spectral_sampling, args=[
            ddt, model_skm, prior_matrix, prior_centre])
    sampler_spec.reset()
    start = time.time()
    sampler_spec.run_mcmc(init_MCMC, nsteps, progress=True)
    end = time.time()
    print('time MCMC = ', end - start)
    spec_samples_raw = sampler_spec.get_chain()
    spec_samples = sampler_spec.get_chain(discard=discard, flat=True)

    path = '/home/baptiste/Documents/these/pixel_based_analysis/results_and_data/full_pipeline/test_total_likelihood/'
    np.save(path+'spec_only_samples.npy', spec_samples)
    np.save(path+'spec_only_samples_raw.npy', spec_samples_raw)
    '''
    start = time.time()
    cosmo_params_list = []
    min_info_list = []
    for spectral_params in spec_samples[5000:6000]:
        Cl_noise_matrix, A, W = from_spectra_to_cosmo(
            spectral_params, model_skm, sensitiviy_mode=1, one_over_f_mode=2, beam_corrected=False, one_over_ell=False)

        # init_min = np.random.normal([0, 0], [0.1, 0.1], 2)
        init_min = np.array([r_true, beta_true.value])

        cosmo_params = minimize(cosmo_sampling, init_min, args=(Cl_cmb_data_matrix,
                                                                reshape_fg_ys, Cl_fid, Cl_noise_matrix, W, A), bounds=bounds_cosmo)
        cosmo_params_list.append(cosmo_params.x)
        min_info_list.append(cosmo_params)
    cosmo_params_list = np.array(cosmo_params_list)
    end = time.time()

    print('time min = ', end - start)
    print('time min iter = ', (end - start)/1000)
    '''

    start = time.time()
    cosmo_params_list = []
    cosmo_params_list_allwalkers = []
    step_counter = 0
    for spectral_params in spec_samples[:1000]:
        print('step = ', step_counter)
        Cl_noise_matrix, A, W = from_spectra_to_cosmo(
            spectral_params, model_skm, sensitiviy_mode=1, one_over_f_mode=2, beam_corrected=False, one_over_ell=False)

        # init_min = np.random.normal([0, 0], [0.1, 0.1], 2)
        nsteps = 50
        nwalkers = 2*2
        discard = nsteps - 1
        success = True
        except_counter = 0
        while success:
            try:
                init_MCMC_cosmo = np.random.normal(
                    total_params[8:], scatter[8:], (nwalkers, 2))
                # init_MCMC_cosmo = np.array([[r_true, beta_true.value]]*nwalkers)

                sampler_cosmo = EnsembleSampler(
                    nwalkers, 2, cosmo_sampling, args=[
                        Cl_cmb_data_matrix, reshape_fg_ys, Cl_fid, Cl_noise_matrix, W, A, False])
                sampler_cosmo.reset()
                # start = time.time()
                sampler_cosmo.run_mcmc(init_MCMC_cosmo, nsteps, progress=True)
                success = False
                # print('time MCMC = ', end - start)
            except ValueError:
                except_counter += 1
                print('except !')
                print('except_counter =', except_counter)
                if except_counter == 5:
                    success = False

        cosmo_samples_raw = sampler_cosmo.get_chain()
        cosmo_samples = sampler_cosmo.get_chain(discard=discard, flat=True)

        cosmo_params_list.append(cosmo_samples[0])
        cosmo_params_list_allwalkers.append(cosmo_samples)
        step_counter += 1
    cosmo_params_list = np.array(cosmo_params_list)
    cosmo_params_list_allwalkers = np.array(cosmo_params_list_allwalkers)
    end = time.time()

    print('time min = ', end - start)
    print('time min iter = ', (end - start)/10)

    np.save(path+'cosmo_only_samples.npy', cosmo_params_list)

    start = time.time()
    nsteps = 1000
    test_sample = homemade_MCMC(cosmo_sampling, [
                                Cl_cmb_data_matrix, reshape_fg_ys, Cl_fid, Cl_noise_matrix, W, A, False], init_MCMC_cosmo[0], nsteps, 100)
    end = time.time()
    print('time homemade sampling = ', end - start)
    print('time homemade sampling = ', (end - start)/nsteps)

    spectral_params = minimize(spectral_sampling, init_min[:8], args=(
        ddt, model_skm, prior_matrix, prior_centre, True))

    start = time.time()
    Cl_noise_matrix, A, W = from_spectra_to_cosmo(
        samples[10], model_skm, sensitiviy_mode=1, one_over_f_mode=2, beam_corrected=False, one_over_ell=False)
    cosmo_params = minimize(cosmo_sampling, init_min[8:], args=(Cl_cmb_data_matrix,
                                                                reshape_fg_ys, Cl_fid, Cl_noise_matrix, W, A), bounds=bounds_cosmo)
    print('time min = ', time.time() - start)
    tot_time = 0
    start_loop = time.time()
    for i in range(10):
        start = time.time()
        cosmo_sampling(init_min[8:], Cl_cmb_data_matrix,
                       reshape_fg_ys, Cl_fid, Cl_noise_matrix, W, A)
        tot_time += time.time() - start
    print('end loop=', time.time()-start_loop, ' ', tot_time)
    start = time.time()
    totL, spec, cosmo, prior, logdetC, Cl_model_total, Cl_noise_matrix = total_likelihood(total_params, Cl_fid,
                                                                                          prior_matrix,
                                                                                          prior_centre, pivot_angle_index,
                                                                                          fsky,
                                                                                          ddt, model_skm, true_A_cmb, fg_freq_maps, reshape_fg_ys,
                                                                                          minimisation=True, params=params)
    print('time totL=', time.time()-start)
    param1 = copy.deepcopy(total_params)
    param1[6] = 1.7
    param1[7] = -3.8
    totL, spec, cosmo, prior, logdetC, Cl_model_total1, Cl_noise_matrix1 = total_likelihood(param1, Cl_fid,
                                                                                            prior_matrix,
                                                                                            prior_centre, pivot_angle_index,
                                                                                            fsky,
                                                                                            ddt, model_skm, true_A_cmb, fg_freq_maps, reshape_fg_ys,
                                                                                            minimisation=True, params=params)
    param2 = copy.deepcopy(total_params)
    # param2[:6] += np.arange(1,2,1/6)*u.deg.to(u.rad)
    param2[6] = 1.38
    param2[7] = -2.2
    # param2[8] = 0.02
    # param2[9] += 1*u.deg.to(u.rad)
    totL, spec, cosmo, prior, logdetC, Cl_model_total2, Cl_noise_matrix2 = total_likelihood(param2, Cl_fid,
                                                                                            prior_matrix,
                                                                                            prior_centre, pivot_angle_index,
                                                                                            fsky,
                                                                                            ddt, model_skm, true_A_cmb, fg_freq_maps, reshape_fg_ys,
                                                                                            minimisation=True, params=params)

    i = 0
    j = 1
    ell = np.arange(lmin, lmax+1)
    plt.plot(ell, (Cl_model_total-Cl_noise_matrix)
             [i, j], label='C_model, b_d = 1.54 ; b_s = -3', color='blue')
    # plt.plot(ell, (Cl_noise_matrix)[i, j],
    #          label='Noise, b_d = 1.54 ; b_s = -3', color='blue', linestyle='--')

    plt.plot(ell, (Cl_model_total1-Cl_noise_matrix1)
             [i, j], label='C_model, b_d = 1.7 ; b_s = -3.8', color='orange')
    # plt.plot(ell, (Cl_noise_matrix1)[
    #          i, j], label='Noise, b_d = 1.54 ; b_s = -3', color='orange', linestyle='--')

    plt.plot(ell, (Cl_model_total2-Cl_noise_matrix2)
             [i, j], label='C_model, b_d = 1.38 ; b_s = -2.2', color='green')
    plt.plot(ell, np.abs((Cl_model_total2-Cl_noise_matrix2) - (Cl_model_total-Cl_noise_matrix))
             [i, j], label='C_model, b_d = 1.38 ; b_s = -2.2', color='red')
    # plt.plot(ell, (Cl_noise_matrix2)[
    #          i, j], label='Noise, b_d = 1.54 ; b_s = -3', color='green', linestyle='--')
    plt.legend()
    if i == j:
        plt.loglog()
    title_index = ['E', 'B']
    plt.title(title_index[i]+title_index[j])
    plt.show()
    # IPython.embed()
    # '''
    miscal_grid = np.arange(1.52, 1.58, 0.06/100)
    miscal_grid = np.arange(0.34, 0.36, 0.02/100)*u.deg.to(u.rad)
    miscal_grid = np.arange(0.005, 0.02, 0.015/100)
    totL_grid = []
    spec_grid = []
    cosmo_grid = []
    logdetC_grid = []
    prior_grid = []
    Cl_cmb_grid = []
    Cl_noise_grid = []
    WEW_grid = []
    logdetN_grid = []
    for i in miscal_grid:
        grid_params = copy.deepcopy(total_params)
        grid_params[8] = i
        totL, spec, cosmo, Prior, logdetC, Cl_cmb, Cl_noise, WEW, logdetN = total_likelihood_sampling(grid_params, Cl_fid,
                                                                                                      prior_matrix,
                                                                                                      prior_centre, pivot_angle_index,
                                                                                                      fsky,
                                                                                                      ddt, model_skm, true_A_cmb, fg_freq_maps, reshape_fg_ys,
                                                                                                      minimisation=True, params=params)
        totL_grid.append(totL)
        spec_grid.append(spec)
        cosmo_grid.append(cosmo)
        prior_grid.append(Prior)
        logdetC_grid.append(logdetC)
        Cl_cmb_grid.append(Cl_cmb)
        Cl_noise_grid.append(Cl_noise)
        WEW_grid.append(WEW)
        logdetN_grid.append(logdetN)
    totL_grid = np.array(totL_grid)
    spec_grid = np.array(spec_grid)
    cosmo_grid = np.array(cosmo_grid)
    prior_grid = np.array(prior_grid)
    logdetC_grid = np.array(logdetC_grid)
    Cl_cmb_grid = np.array(Cl_cmb_grid)
    Cl_noise_grid = np.array(Cl_noise_grid)
    WEW_grid = np.array(WEW_grid)
    logdetN_grid = np.array(logdetN_grid)
    plt.plot(miscal_grid, spec_grid+cosmo_grid+prior_grid-logdetN_grid)
    plt.show()
    plt.plot(miscal_grid, spec_grid+cosmo_grid+prior_grid+logdetN_grid)
    plt.show()
    plt.plot(miscal_grid, spec_grid+cosmo_grid+prior_grid)
    plt.show()
    plt.plot(miscal_grid, totL_grid, label='tot')
    plt.plot(miscal_grid, spec_grid, label='spec')
    plt.plot(miscal_grid, cosmo_grid, label='cosmo')
    plt.plot(miscal_grid, np.array(cosmo_grid)-np.array(logdetC_grid), label='trC-1E')
    plt.plot(miscal_grid, logdetC_grid, label='logdetC')
    plt.plot(miscal_grid, prior_grid, label='prior')
    plt.legend()
    plt.show()

    scatter = [prior_precision]*freq_number
    scatter.append(0.1)
    scatter.append(0.1)
    scatter.append(0.01)
    scatter.append(prior_precision)
    scatter = np.array(scatter)
    init_min = np.random.normal(total_params, scatter, 10)
    start = time.time()
    min_totL = minimize(total_likelihood_sampling, init_min,
                        args=(Cl_fid, prior_matrix, prior_centre, pivot_angle_index,
                              fsky, ddt, model_skm, true_A_cmb, fg_freq_maps, reshape_fg_ys,
                              True, params))
    print('time min=', time.time()-start)
    start = time.time()
    min_spectral = minimize(spectral_sampling, init_min,
                            args=(ddt, model_skm, prior_matrix, prior_centre, True))
    print('time min=', time.time()-start)
    '''

    init_MCMC = np.random.normal(
        total_params, scatter, (2*10, 10))
    nwalkers = 2 * 10
    nsteps = 5000
    discard = 500

    sampler = EnsembleSampler(
        nwalkers, 10, total_likelihood_sampling, args=[
            Cl_fid, prior_matrix, prior_centre, pivot_angle_index,
            fsky, ddt, model_skm, true_A_cmb, fg_freq_maps, reshape_fg_ys,
            False, params])
    sampler.reset()
    start = time.time()
    sampler.run_mcmc(init_MCMC, nsteps, progress=True)
    end = time.time()
    multi_time = end - start
    samples_raw = sampler.get_chain()
    samples = sampler.get_chain(discard=discard, flat=True)
    np.save(path+'samples_5K_newP.npy', samples)
    np.save(path+'samples_raw_5K_newP.npy', samples_raw)
    '''

    exit()


######################################################
# MAIN CALL
if __name__ == "__main__":
    main()


def total_likelihood(total_params, Cl_fid,
                     total_prior_matrix,
                     prior_centre, pivot_angle_index,
                     fsky,
                     ddt, model_skm, true_A_cmb, fg_freq_maps, reshape_fg_ys,
                     minimisation=False, params=None):
    '''=========Check bounds========='''
    # print(total_params)
    angle_array = np.delete(total_params[:freq_number], pivot_angle_index)*u.rad
    fg_params = total_params[freq_number:freq_number+2]
    start_cosmo_params = freq_number+2
    r = total_params[start_cosmo_params]
    beta = total_params[start_cosmo_params+1]*u.rad
    pivot = total_params[pivot_angle_index]*u.rad

    angle_eval = total_params[:freq_number]*u.rad

    if not 0.5 <= fg_params[-2] < 2.5 or not -5 <= fg_params[-1] <= -1:
        print('bla')
        if not minimisation:
            return -np.inf

    if np.any(np.array(angle_array) < (-0.5*np.pi)) or np.any(np.array(angle_array) > (0.5*np.pi)):
        print('blou')
        if not minimisation:
            return -np.inf

    if not minimisation:
        if r > 5 or r < -0.01:
            print('bla1')
            return -np.inf
        elif beta.value < -np.pi/2 or beta.value > np.pi/2:
            print('bla2')
            return -np.inf
        elif pivot.value < -np.pi/2 or pivot.value > np.pi/2:
            print('bla3')
            return -np.inf
    '''=========Spectral likelihood========='''

    model_skm.evaluate_mixing_matrix([fg_params[-2], 20, fg_params[-1]])

    model_skm.miscal_angles = angle_eval

    model_skm.bir_angle = 0

    model_skm.get_miscalibration_angle_matrix()
    model_skm.get_bir_matrix()

    model_skm.get_projection_op()

    spectral_like = np.einsum('ij,ji->...', -model_skm.projection+model_skm.inv_noise,
                              ddt)

    '''Spectral Fisher matrix estimation'''
    diff_list_res = get_diff_list(model_skm, params)
    diff_diff_list_res = get_diff_diff_list(model_skm, params)
    fisher_matrix_spectral = fisher_new(ddt, model_skm,
                                        diff_list_res, diff_diff_list_res, params)
    fisher_pivot = np.delete(np.delete(fisher_matrix_spectral,
                                       pivot_angle_index, 0), pivot_angle_index, 1)
    sigma_spectral = np.linalg.inv(fisher_pivot)
    # sigma_spectral = np.linalg.inv(fisher_matrix_spectral)
    # sigma_spectral_tot = np.linalg.inv(fisher_matrix_spectral)

    '''Residuals computation'''
    # IPython.embed()
    start = time.time()
    stat, bias, var, Cl_fg, Cl_cmb, Cl_residuals_matrix, ell, W_cmb, dW_cmb, ddW_cmb = get_residuals(
        model_skm, fg_freq_maps, sigma_spectral, lmin, lmax, fsky, params,
        cmb_spectra=spectra_true, true_A_cmb=true_A_cmb, pivot_angle_index=pivot_angle_index, reshape_fg_ys=reshape_fg_ys)
    # print('time res=', time.time()-start)
    WA_cmb = W_cmb.dot(model_skm.mix_effectiv[:, :2])
    dWA_cmb = dW_cmb.dot(model_skm.mix_effectiv[:, :2])
    W_dBdB_cmb = ddW_cmb.dot(model_skm.mix_effectiv[:, :2])
    VA_cmb = np.einsum('ij,ij...->...', sigma_spectral, W_dBdB_cmb[:, :])
    Cl_noise, ell_noise = get_noise_Cl(
        model_skm.mix_effectiv, lmax+1, fsky,
        sensitiviy_mode, one_over_f_mode,
        instrument=INSTRU, onefreqtest=1-spectral_flag)
    ell_noise = ell_noise[lmin-2:]

    from residuals import get_W
    from bjlib import V3calc as V3
    # IPython.embed()

    V3_results = V3.so_V3_SA_noise(sensitiviy_mode, 1,
                                   SAC_yrs_LF=1, f_sky=fsky, ell_max=lmax+1, beam_corrected=True)
    noise_nl = np.repeat(V3_results[1], 2, 0)

    W = get_W(model_skm)
    WNW = np.einsum('fi, fl, fj -> lij', W.T, noise_nl, W.T)
    WcmbNWcmb = np.einsum('fi, fl, fj -> lij', W.T[:, :2], noise_nl, W.T[:, :2])
    noise_clW = WNW.swapaxes(-3, -1)[..., lmin-2:]
    noise_clWcmb = WcmbNWcmb.swapaxes(-3, -1)[..., lmin-2:]

    nl_inv = 1/noise_nl
    A = model_skm.mix_effectiv
    AtNA = np.einsum('fi, fl, fj -> lij', A, nl_inv, A)
    inv_AtNA = np.linalg.inv(AtNA)
    noise_cl = inv_AtNA.swapaxes(-3, -1)[..., lmin-2:]

    A_cmb = model_skm.mix_effectiv[:, :2]
    AtNAcmb = np.einsum('fi, fl, fj -> lij', A_cmb, nl_inv, A_cmb)
    inv_AtNAcmb = np.linalg.inv(AtNAcmb)
    noise_clcmb = inv_AtNAcmb.swapaxes(-3, -1)[..., lmin-2:]
    # i = 0
    # j = 1
    # plt.plot(ell_noise, noise_cl[i, j], label='cl noise')
    # plt.plot(ell_noise, noise_clcmb[i, j], label='cl noise cmb')
    # plt.plot(ell_noise, noise_clW[i, j], label='clW ')
    # plt.plot(ell_noise, noise_clWcmb[i, j], label='clWcmb')
    # plt.title('one_over_f=1 , beam_corrected=False')
    # plt.legend()
    # plt.show()
    from residuals import get_ys_Cls
    Wfg_ys = W[:2].dot(reshape_fg_ys)
    Wfg_ys_TEB = np.zeros([3, Wfg_ys.shape[-1]], dtype='complex')
    Wfg_ys_TEB[1:] = Wfg_ys
    temp_Cl_fg = get_ys_Cls(Wfg_ys_TEB, Wfg_ys_TEB, lmax, fsky)[:, lmin:]
    Cl_SM_fg = np.array([[temp_Cl_fg[0], temp_Cl_fg[2]], [
        temp_Cl_fg[2], temp_Cl_fg[1]]])

    WA_cmb = W[:2].dot(true_A_cmb)
    Cl_cmb_data = np.zeros((2, 2, len(spectra_true[0])))
    Cl_cmb_data[0, 0] = spectra_true[1]
    Cl_cmb_data[1, 1] = spectra_true[2]
    Cl_cmb_data[1, 0] = spectra_true[4]
    Cl_cmb_data[0, 1] = spectra_true[4]
    Cl_SM_cmb = np.einsum('ij,jkl,km->iml', WA_cmb, Cl_cmb_data, WA_cmb.T)[:, :, lmin:lmax+1]

    WEW = Cl_SM_cmb + Cl_SM_fg

    Cl_noise = Cl_noise[lmin-2:]
    # Cl_noise = noise_clWcmb[0, 0]
    # Cl_noise = noise_clW[0, 0]

    Cl_noise_matrix = np.zeros([2, 2, Cl_noise.shape[0]])
    Cl_noise_matrix[0, 0] = Cl_noise
    Cl_noise_matrix[1, 1] = Cl_noise

    # Cl_noise_cmb, ell_noise_cmb = get_noise_Cl(
    #     model_skm.mix_effectiv[:, :2], lmax+1, fsky,
    #     sensitiviy_mode, one_over_f_mode,
    #     instrument=INSTRU, onefreqtest=1-spectral_flag)
    # Cl_noise_cmb = Cl_noise_cmb[lmin-2:]
    # ell_noise_cmb = ell_noise_cmb[lmin-2:]
    # Cl_noise_matrix_cmb = np.zeros([2, 2, Cl_noise_cmb.shape[0]])
    # Cl_noise_matrix_cmb[0, 0] = Cl_noise_cmb
    # Cl_noise_matrix_cmb[1, 1] = Cl_noise_cmb
    Cl_noise_matrix_cmb = Cl_noise_matrix

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

    # Cl_model_total_ = WACAW + Cl_noise_matrix + tr_SigmaYY + VACAW + WACAV
    Cl_model_total_ = Cl_cmb_rot_matrix + Cl_noise_matrix

    Cl_model_total = Cl_model_total_
    # Cl_model_total = Cl_cmb_rot_matrix + Cl_noise_matrix
    tr_SigmaYY = np.einsum('ij,jimnl->mnl', sigma_spectral, Cl_residuals_matrix['YY'])
    Cl_data = Cl_residuals_matrix['yy'] + Cl_residuals_matrix['zy'] + \
        Cl_residuals_matrix['yz'] + tr_SigmaYY + Cl_noise_matrix
    # Cl_data_rot = Cl_data
    Cl_data_rot = WEW + Cl_noise_matrix

    inv_model = np.linalg.inv(Cl_model_total.T).T
    dof = (2 * ell + 1) * fsky
    dof_over_Cl = dof * inv_model

    first_term_ell = np.einsum('ijl,jkl->ikl', dof_over_Cl, Cl_data_rot)

    first_term = np.sum(np.trace(first_term_ell))

    logdetC = np.sum(dof*np.log(np.linalg.det(Cl_model_total.T)))

    likelihood_cosmo = first_term + logdetC  # + radek_jost_prior

    # new_noise_term = np.trace(model_skm.inv_noise.dot(ddt))
    '''===========Prior term==========='''
    Prior = (angle_eval.value - prior_centre).T.dot(
        total_prior_matrix[:freq_number, :freq_number]).dot(angle_eval.value - prior_centre)
    if minimisation:
        # print('tot = ', -(spectral_like - likelihood_cosmo - Prior), ' spec = ',
        #       -spectral_like, ' cosmo = ', likelihood_cosmo, ' Prior = ', Prior)
        # , -spectral_like, likelihood_cosmo, Prior, logdetC, Cl_model_total, Cl_noise_matrix
        return -(spectral_like - likelihood_cosmo - Prior)
    else:
        return spectral_like - likelihood_cosmo - Prior
