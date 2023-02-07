import IPython
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
# from os import path as p
import tracemalloc
from total_likelihood import from_spectra_to_cosmo, cosmo_sampling
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

    if rank == 0:
        ddt, fg_freq_maps, n_obspix = get_SFN(
            data, model_data, path_BB, S_cmb_name, spectral_flag, addnoise=add_noise)

        fg_ys = get_ys_alms(y_Q=fg_freq_maps[::2], y_U=fg_freq_maps[1::2], lmax=lmax)

        reshape_fg_ys = fg_ys[:, 1:].reshape([2*freq_number, fg_ys.shape[-1]])
    else:
        reshape_fg_ys = None
    reshape_fg_ys = comm.bcast(reshape_fg_ys, root=0)
    if rank == 0:
        print(rank)
        print('reshape_fg_ys = ', reshape_fg_ys)
    elif rank == 1:
        print(rank)
        print('reshape_fg_ys = ', reshape_fg_ys)
    current, peak = tracemalloc.get_traced_memory()
    print('rank = ', rank,
          f" 2Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")

    miscal = 1*u.deg.to(u.rad)
    # total_params = np.array([true_miscal_angles[0].value, true_miscal_angles[1].value,
    #                          true_miscal_angles[2].value, true_miscal_angles[3].value,
    #                          true_miscal_angles[4].value, true_miscal_angles[5].value,
    #                          1.54, -3., r_true, beta_true.value])
    total_params = np.append(true_miscal_angles.value.tolist(), [
                             1.54, -3., r_true, beta_true.value])
    prior_centre = true_miscal_angles.value
    # pivot_angle_index = 2

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
        total_params[:freq_number], bir_angle=beta_true*0,
        frequencies_by_instrument_array=freq_by_instru, nside=nside,
        spectral_params=[total_params[freq_number], 20, total_params[freq_number+1]],
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
    scatter.append(0.001)
    scatter.append(prior_precision)
    scatter = np.array(scatter)
    # init_min = np.random.normal(total_params, scatter, 10)

    if rank == 0:
        spec_samples = np.load(save_path+'spec_samples.npy')

    current, peak = tracemalloc.get_traced_memory()
    print('rank = ', rank,
          f"6Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")

    if rank == 0:
        shape_array = np.array(spec_samples.shape)
        print(shape_array, type(shape_array))
    else:
        shape_array = np.empty(2)
    shape_array_bcast = comm.bcast(shape_array, root=0)
    print(rank, ', shape array = ', shape_array_bcast)
    sendbuf = None
    if rank == 0:
        sendbuf = copy.deepcopy(spec_samples.reshape(
            [size, spec_samples.shape[0]//size, spec_samples.shape[1]]))
    recvbuf = np.empty([shape_array_bcast[0]//size, shape_array_bcast[1]])
    recvbuf = comm.scatter(sendbuf, root=0)
    print(rank)
    print(recvbuf)
    end_init_time = time.time()
    print('init time = ', end_init_time - start_init_time)

    start = time.time()
    cosmo_params_list = []
    cosmo_params_list_allwalkers = []
    step_counter = 0
    for spectral_params in recvbuf[:-50]:
        print('step = ', step_counter)
        Cl_noise_matrix, A, W = from_spectra_to_cosmo(
            spectral_params, model_skm, sensitiviy_mode=sensitiviy_mode, one_over_f_mode=one_over_f_mode, beam_corrected=beam_correction, one_over_ell=one_over_ell)

        # init_min = np.random.normal([0, 0], [0.1, 0.1], 2)
        nsteps = 300
        nwalkers = 2*2
        discard = nsteps - 1
        success = True
        except_counter = 0
        while success:
            try:
                seed = rank*(shape_array_bcast[0]//size)*(freq_number*2) + \
                    step_counter*(freq_number*2)+except_counter*2
                print('seed = ', seed)
                np.random.seed(seed)
                spec_param_number = freq_number + 2
                init_MCMC_cosmo = np.random.normal(
                    total_params[spec_param_number:], scatter[spec_param_number:], (nwalkers, 2))
                sampler_cosmo = EnsembleSampler(
                    nwalkers, 2, cosmo_sampling, args=[
                        Cl_cmb_data_matrix, reshape_fg_ys, Cl_fid, add_noise*Cl_noise_matrix, W, A, False])
                sampler_cosmo.reset()
                sampler_cosmo.run_mcmc(init_MCMC_cosmo, nsteps, progress=False)

                # cosmo_samples_raw = sampler_cosmo.get_chain()
                cosmo_samples = sampler_cosmo.get_chain(discard=discard, flat=True)

                success = False
            except ValueError:
                except_counter += 1
                print('except !')
                print('except_counter =', except_counter)
                if except_counter == 5:
                    print('except threshold reached !')
                    cosmo_samples = np.array([[np.nan, np.nan]])

                    success = False
        cosmo_params_list.append(cosmo_samples[0])
        cosmo_params_list_allwalkers.append(cosmo_samples)
        step_counter += 1

    cosmo_params_list = np.array(cosmo_params_list)

    end = time.time()
    print('time min = ', end - start)
    print('time min iter = ', (end - start)/step_counter)
    cosmo_params_list_MPI = None
    if rank == 0:
        cosmo_params_list_MPI = np.empty(
            [size, cosmo_params_list.shape[0], cosmo_params_list.shape[1]])
    comm.Gather(cosmo_params_list, cosmo_params_list_MPI, root=0)
    if rank == 0:
        np.save(save_path+'cosmo_samples_MCMC.npy', cosmo_params_list_MPI)

    cosmo_params_list_allwalkers = np.array(cosmo_params_list_allwalkers)

    print('rank = ', rank,
          f"7Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")

    exit()


######################################################
# MAIN CALL
if __name__ == "__main__":
    main()
