# import IPython
from astropy import units as u
import numpy as np
# from fisher_pixel import fisher_new
# from scipy.optimize import minimize
from pixel_based_angle_estimation import data_and_model_quick, get_model
# import matplotlib.pyplot as plt
import copy
import time
from residuals import get_ys_alms
from emcee import EnsembleSampler
from config_copy import *
# import bjlib.lib_project as lib
from residuals import get_SFN
# , get_diff_list, get_diff_diff_list,\
#     get_residuals, get_noise_Cl, \
#     multi_freq_get_sky_fg
# from residuals import get_W, get_ys_Cls
# from bjlib import V3calc as V3
from mpi4py import MPI
from os import path as p
import tracemalloc
from total_likelihood import spectral_sampling
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="path to save folder")
    args = parser.parse_args()

    save_path = args.folder
    print()
    print(save_path)

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    print(size, ':', rank)
    tracemalloc.start()
    start_init_time = time.time()
    S_cmb_name = '/global/homes/j/jost/these/pixel_based_analysis/code/data/New_S_cmb_n{}_s{}_r{:1}_b{:1.1e}'.format(nside, nsim, r_true, beta_true.value).replace(
        '.', 'p') + '.npy'
    print(S_cmb_name)

    data, model_data = data_and_model_quick(
        miscal_angles_array=true_miscal_angles, bir_angle=beta_true*0,
        frequencies_by_instrument_array=freq_by_instru, nside=nside,
        sky_model=sky_model, sensitiviy_mode=sensitiviy_mode,
        one_over_f_mode=one_over_f_mode, instrument=INSTRU, overwrite_freq=overwrite_freq)
    true_A_cmb = model_data.mix_effectiv[:, :2]

    current, peak = tracemalloc.get_traced_memory()
    print('rank = ', rank,
          f"1Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")

    # if rank == 0:
    ddt, fg_freq_maps, n_obspix = get_SFN(
        data, model_data, path_BB, S_cmb_name, spectral_flag, addnoise=add_noise)

    fg_ys = get_ys_alms(y_Q=fg_freq_maps[::2], y_U=fg_freq_maps[1::2], lmax=lmax)
    reshape_fg_ys = fg_ys[:, 1:].reshape([12, 45451])

    current, peak = tracemalloc.get_traced_memory()
    print('rank = ', rank,
          f" 2Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")

    miscal = 1*u.deg.to(u.rad)
    total_params = np.array([true_miscal_angles[0].value, true_miscal_angles[1].value,
                             true_miscal_angles[2].value, true_miscal_angles[3].value,
                             true_miscal_angles[4].value, true_miscal_angles[5].value,
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
    current, peak = tracemalloc.get_traced_memory()
    print('rank = ', rank,
          f"3Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")
    '''Sky model from spectral likelihood results'''
    model_skm = get_model(
        total_params[:6], bir_angle=beta_true*0,
        frequencies_by_instrument_array=freq_by_instru, nside=nside,
        spectral_params=[total_params[6], 20, total_params[7]],
        sky_model='c1s0d0', sensitiviy_mode=sensitiviy_mode,
        one_over_f_mode=one_over_f_mode, instrument=INSTRU, overwrite_freq=overwrite_freq)
    current, peak = tracemalloc.get_traced_memory()
    print('rank = ', rank,
          f"4Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")

    current, peak = tracemalloc.get_traced_memory()
    print('rank = ', rank,
          f"5Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")
    scatter = [prior_precision]*freq_number
    scatter.append(0.1)
    scatter.append(0.1)
    scatter.append(0.01)
    scatter.append(prior_precision)
    scatter = np.array(scatter)
    init_min = np.random.normal(total_params, scatter, 10)

    # if machine == 'local':
    #     path = '/home/baptiste/Documents/these/pixel_based_analysis/results_and_data/full_pipeline/test_total_likelihood/'
    # elif machine == 'NERSC':
    #     path = '/global/homes/j/jost/these/pixel_based_analysis/results_and_data/total_like/'

    if rank == 0:
        if p.exists(save_path+'spec_samples.npy'):
            print('loading spectral samples...')
            spec_samples = np.load(save_path+'spec_samples.npy')
        else:
            print('generating spectral samples...')

            nsteps = nsteps_spectral
            discard = discard_spectral
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

            np.save(save_path+'spec_samples.npy', spec_samples)
            np.save(save_path+'spec_samples_raw.npy', spec_samples_raw)
    current, peak = tracemalloc.get_traced_memory()
    print('rank = ', rank,
          f"6Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")

    exit()


######################################################
# MAIN CALL
if __name__ == "__main__":
    main()
