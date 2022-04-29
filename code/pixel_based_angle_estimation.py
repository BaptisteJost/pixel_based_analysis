import time
import IPython
import numpy as np
import bjlib.likelihood_SO as lSO
from astropy import units as u
import bjlib.V3calc as V3
import emcee
from multiprocessing import Pool
# import argparse
from schwimmbad import MPIPool
from schwimmbad import MultiPool

import cloudpickle
import tracemalloc
import os
import sys
from mpi4py import MPI


def get_chi_squared_local(angle_array, ddtPN, model_skm, prior=False,
                          fixed_miscal_angles=[], miscal_priors=[],
                          birefringence=False, spectral_index=False, Ninvfactor=1,
                          minimize=False, params=None):
    # angle_array = np.append(0, angle_array)
    if spectral_index:
        angle_index = - 2
        angle_index_none = angle_index

        if not 0.5 <= angle_array[-2] < 2.5 or not -5 <= angle_array[-1] <= -1:
            print('bla')
            if not minimize:
                return -np.inf
            else:
                return np.inf

        model_skm.evaluate_mixing_matrix([angle_array[-2], 20, angle_array[-1]])

    else:
        angle_index = 0
        angle_index_none = None

    if np.any(np.array(angle_array[:angle_index_none]) < (-0.5*np.pi)) or np.any(np.array(angle_array[:angle_index_none]) > (0.5*np.pi)):
        print('blou')
        if not minimize:
            return -np.inf
        else:
            return np.inf

    if birefringence:
        model_skm.miscal_angles = np.append(angle_array[:-(-angle_index+1)], fixed_miscal_angles)
        model_skm.bir_angle = angle_array[-(-angle_index+1)]

    else:
        model_skm.miscal_angles = np.append(angle_array[:angle_index_none], fixed_miscal_angles)
        model_skm.bir_angle = 0

    model_skm.get_miscalibration_angle_matrix()
    model_skm.get_bir_matrix()

    model_skm.get_projection_op()

    # chi_squared = np.einsum('ij,jip->...', (-model_skm.projection+model_skm.inv_noise),
    #                         ddtPN)
    chi_squared = np.einsum('ij,ji->...', Ninvfactor*(-model_skm.projection+model_skm.inv_noise),
                            ddtPN)

    if prior:
        gaussian_prior = 0
        for l in range(len(miscal_priors)):
            # gaussian_prior += (1/(2*(miscal_priors[l, 1]**2))
            #                    * (angle_array[int(miscal_priors[l, 2])] - miscal_priors[l, 0])**2)
            gaussian_prior += ((angle_array[int(miscal_priors[l, 2])] -
                                miscal_priors[l, 0])**2) / (2*(miscal_priors[l, 1]**2)) + 2*np.log(2*miscal_priors[l, 1]*np.sqrt(2*np.pi))

        if minimize:
            return -chi_squared + gaussian_prior
        else:
            return chi_squared - gaussian_prior
    if minimize:
        return -chi_squared
    else:
        return chi_squared


def data_and_model_quick(miscal_angles_array, frequencies_by_instrument_array, bir_angle=0*u.rad,
                         nside=512, spectral_params=[1.59, 20, -3], sky_model='c1s0d0',
                         sensitiviy_mode=2, one_over_f_mode=2, instrument='SAT'):
    start_datainit = time.time()
    data = lSO.sky_map(bir_angle=bir_angle, miscal_angles=miscal_angles_array,
                       frequencies_by_instrument=frequencies_by_instrument_array,
                       nside=nside, sky_model=sky_model, instrument=instrument)
    print('time data init = ', time.time() - start_datainit)

    start_modelinit = time.time()
    model = lSO.sky_map(bir_angle=bir_angle, miscal_angles=miscal_angles_array,
                        frequencies_by_instrument=frequencies_by_instrument_array,
                        nside=nside, sky_model=sky_model, instrument=instrument)
    print('time model init = ', time.time() - start_modelinit)
    # v3f = V3.so_V3_SA_bands()
    # index = np.in1d(v3f, frequencies_array).nonzero()[0]
    if instrument == 'Planck' and sky_model == 'c1s0d0' and nside == 2048:
        start_import = time.time()
        print('PLANCK MAP IMPORT')
        data.get_frequency()
        data.cmb_freq_maps = np.load('data/pysm_cmbsky_c1_2048.npy')
        data.dust_freq_maps = np.load('data/pysm_dustsky_d0_2048.npy')
        data.sync_freq_maps = np.load('data/pysm_synchsky_s0_2048.npy')
        print('time import = ', time.time() - start_import)

    else:
        start_pysm = time.time()
        data.get_pysm_sky()
        print('time pysm = ', time.time() - start_pysm)

        # data.frequencies = frequencies_array
        start_freq = time.time()
        data.get_frequency()
        print('time freq = ', time.time() - start_freq)

        start_freq_maps = time.time()
        data.get_freq_maps()
        print('time freqmaps = ', time.time() - start_freq_maps)

    start_rot = time.time()
    data.cmb_rotation()
    print('time rot = ', time.time() - start_rot)

    start_signal = time.time()
    data.get_signal()
    print('time signal = ', time.time() - start_signal)

    start_Aev = time.time()
    data.get_A_ev(fix_temp=True)
    print('time Aev = ', time.time() - start_Aev)

    start_mix = time.time()
    data.evaluate_mixing_matrix(spectral_params)
    print('time mix = ', time.time() - start_mix)

    start_miscal = time.time()
    data.get_miscalibration_angle_matrix()
    print('time miscal = ', time.time() - start_miscal)

    start_data = time.time()
    data.get_data()
    print('time data_pix = ', time.time() - start_data)

    # model.frequencies = frequencies_array
    start_modeltotal = time.time()
    model.get_frequency()

    model.get_bir_matrix()
    model.get_A_ev(fix_temp=True)
    model.evaluate_mixing_matrix(spectral_params)

    model.get_miscalibration_angle_matrix()
    model.get_noise(sensitiviy_mode=sensitiviy_mode, one_over_f_mode=one_over_f_mode)
    # model.inv_noise = model.inv_noise[index[0]*2:index[-1]*2 + 2,
    #                                   index[0]*2:index[-1]*2 + 2]
    # model.noise_covariance = model.noise_covariance[index[0]*2:index[-1]*2 + 2,
    #                                                 index[0]*2:index[-1]*2 + 2]
    model.get_projection_op()
    print('time model total =', time.time() - start_modeltotal)

    return data, model


def get_model(miscal_angles_array, frequencies_by_instrument_array,
              bir_angle=0*u.rad, nside=512, spectral_params=[1.59, 20, -3],
              sky_model='c1s0d0', sensitiviy_mode=2, one_over_f_mode=2,
              instrument='SAT'):

    model = lSO.sky_map(bir_angle=bir_angle, miscal_angles=miscal_angles_array,
                        frequencies_by_instrument=frequencies_by_instrument_array,
                        nside=nside, sky_model=sky_model, instrument=instrument)

    model.get_frequency()
    model.get_bir_matrix()
    model.get_A_ev(fix_temp=True)
    model.evaluate_mixing_matrix(spectral_params)

    model.get_miscalibration_angle_matrix()
    model.get_noise(sensitiviy_mode=sensitiviy_mode, one_over_f_mode=one_over_f_mode)

    model.get_projection_op()

    return model


def run_MCMC(ddtn, model, sampled_miscal_freq, nsteps, discard_num,
             sampled_birefringence=False, prior=False,
             walker_per_dim=2, prior_precision=(1*u.arcmin).to(u.rad).value,
             prior_index=[0, 6], spectral_index=False, return_raw_samples=False,
             save=False, path='./prior_tests/', parallel=False, nside=512):

    ndim = sampled_miscal_freq + sampled_birefringence + 2*spectral_index
    nwalkers = walker_per_dim*ndim
    # true_miscal_angles = np.array([0., 0.08333333, 0.16666667, 0.25, 0.33333333,
    #                                0.41666667])*u.rad
    true_miscal_angles = np.array([0]*6)*u.rad

    std_6 = np.array([0.00017337, 0.00016427, 0.00013824, 0.0001337, 0.00013536, 0.00013666])

    if not prior:
        angle_prior = []
    else:
        angle_prior = []
        for d in range(sampled_miscal_freq):
            angle_prior.append([true_miscal_angles.value[d], prior_precision, int(d)])
        angle_prior = np.array(angle_prior[prior_index[0]:prior_index[-1]])

    p0 = np.random.normal(
        true_miscal_angles.value[:sampled_miscal_freq], std_6*2, (nwalkers, sampled_miscal_freq))

    if sampled_birefringence:
        p0_bir = np.array([np.random.normal(0, std_6.max(), (nwalkers))])
        p0 = np.concatenate((p0, p0_bir.T), axis=1)

    if spectral_index:
        p0_spectral = np.array(np.random.normal([1.5, -3], [0.5, 1], (nwalkers, 2)))
        p0 = np.concatenate((p0, p0_spectral), axis=1)

    if parallel:
        with MPIPool() as pool:
            if not pool.is_master():
                pool.wait()
                sys.exit(0)
            sampler = emcee.EnsembleSampler(
                nwalkers, ndim, get_chi_squared_local, args=[
                    data, model, prior, true_miscal_angles.value[sampled_miscal_freq:], angle_prior, sampled_birefringence, spectral_index],
                pool=pool)
            sampler.reset()
            sampler.run_mcmc(p0, nsteps, progress=False)

    else:
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, get_chi_squared_local, args=[
                ddtn, model, prior, true_miscal_angles.value[sampled_miscal_freq:], angle_prior, sampled_birefringence, spectral_index],
            pool=None)
        sampler.reset()
        sampler.run_mcmc(p0, nsteps, progress=True)
    flat_samples = sampler.get_chain(discard=discard_num, flat=True)

    file_name, file_name_raw = get_file_name_sample(
        sampled_miscal_freq, nsteps, discard_num,
        sampled_birefringence, prior,
        prior_precision,
        prior_index, spectral_index, nside)

    print('file_name = ', file_name)

    if return_raw_samples:
        samples_raw = sampler.get_chain()
        if save:
            np.save(path+file_name, flat_samples)
            np.save(path+file_name_raw, samples_raw)
        return flat_samples, samples_raw
    if save:
        np.save(path+file_name, flat_samples)
    return flat_samples


def get_file_name_sample(sampled_miscal_freq, nsteps, discard_num,
                         sampled_birefringence=False, prior=False,
                         prior_precision=(1*u.arcmin).to(u.rad).value,
                         prior_index=[0, 6], spectral_index=False, nside=512):
    file_name = 'priorfix_{}Samples_{}Discard'.format(nsteps, discard_num)
    file_name_raw = 'priorfix_{}Samples_RAW'.format(nsteps)

    file_name += '_v2MiscalAll0'
    file_name_raw += '_v2MiscalAll0'

    file_name += '_MiscalFrom0to{}'.format(sampled_miscal_freq)
    file_name_raw += '_MiscalFrom0to{}'.format(sampled_miscal_freq)

    if prior:
        prior_name = '_PriorPosition{}to{}_Precision{:1.1e}rad'.format(
            prior_index[0], prior_index[-1], prior_precision).replace('.', 'p')
        file_name += prior_name
        file_name_raw += prior_name

    if sampled_birefringence:
        file_name += '_BirSampled'
        file_name_raw += '_BirSampled'

    if spectral_index:
        file_name += '_SpectralSampled'
        file_name_raw += '_SpectralSampled'
    file_name += '_Mask0400_nside'+str(int(nside))
    file_name_raw += '_Mask0400_nside'+str(int(nside))
    file_name += '_FSN_SOf2f'
    file_name_raw += '_FSN_SOf2f'
    file_name += '.npy'
    file_name_raw += '.npy'

    return file_name, file_name_raw


def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--nsteps', type=int, help='MCMC steps',
    #                     default=3000)
    # parser.add_argument('--discard', type=int, help='discard number',
    #                     default=1000)
    # parser.add_argument('--birefringence', help='birefringence sampled',
    #                     action="store_true")
    # parser.add_argument('--spectral', help='spectral indices sampled',
    #                     action="store_true")
    # parser.add_argument('--prior_indices', type=int, nargs='+', help='prior indices',
    #                     default=[0, 6])
    # parser.add_argument('--MPI', help='use MPI default false',
    #                     action="store_true")
    # args = parser.parse_args()
    #
    # nsteps = args.nsteps
    # discard = args.discard
    # birefringence = args.birefringence
    # spectral = args.spectral
    # prior_indices = args.prior_indices
    # wMPI = args.MPI

    nsteps = 11000  # args.nsteps
    discard = 1000  # args.discard
    birefringence = 0  # args.birefringence
    spectral = 1  # args.spectral
    prior_indices = [2, 4]  # args.prior_indices
    prior_flag = True
    # prior_precision = (1*u.arcmin).to(u.rad).value
    prior_precision = (0.01*u.deg).to(u.rad).value
    nside = 128
    wMPI = 0  # args.MPI
    wMPI2 = 0

    path_NERSC = '/global/homes/j/jost/these/pixel_based_analysis/results_and_data/run02032021/'
    # path_local = './test11k/'
    path_local = '/home/baptiste/Documents/these/pixel_based_analysis/results_and_data/SOf2f/samples/'
    path = path_local

    if wMPI2:
        comm = MPI.COMM_WORLD
        mpi_rank = MPI.COMM_WORLD.Get_rank()
        nsim = comm.Get_size()
        print(mpi_rank, nsim)
        print()
        birefringence = 1
        spectral = 1
        if mpi_rank == 0:
            prior_flag = True
            prior_indices = [0, 6]
            print('prior_flag : ', prior_flag, 'prior_indices = ', prior_indices)
        if mpi_rank == 1:
            prior_flag = False
            prior_indices = []
            print('prior_flag : ', prior_flag, 'prior_indices = ', prior_indices)

        comm = MPI.COMM_WORLD
        mpi_rank = MPI.COMM_WORLD.Get_rank()
        nsim = comm.Get_size()
        print(mpi_rank, nsim)
        if mpi_rank > 1:
            prior_indices = [(mpi_rank-2) % 6, ((mpi_rank-2) % 6)+1]
        # if mpi_rank//6 == 0:
        #     birefringence = 1
        #     spectral = 1
        #
        # if mpi_rank//6 == 1:
        #     birefringence = 0
        #     spectral = 1
        #
        # if mpi_rank//6 == 2:
        #     birefringence = 1
        #     spectral = 0
        print('prior_indices = ', prior_indices)
        print('birefringence = ', birefringence)
        print('spectral = ', spectral)
        # exit()

    # birefringence = 1  # args.birefringence
    # spectral = 1  # args.spectral

    # if wMPI:
    #     print('bla')
    #     MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads)
    #     # To avoid issues between numpy and emcee
    #     os.environ["OMP_NUM_THREADS"] = "1"

    tracemalloc.start()
    start = time.time()

    # true_miscal_angles = np.arange(0, 0.5, 0.5/6)*u.rad
    true_miscal_angles = np.array([0]*6)*u.rad

    data6, model6 = data_and_model_quick(miscal_angles_array=true_miscal_angles,
                                         frequencies_array=V3.so_V3_SA_bands(),
                                         frequencies_by_instrument_array=[1, 1, 1, 1, 1, 1], nside=nside)

    print('time initializing in s = ', time.time() - start)
    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")
    # tracemalloc.stop()

    # tracemalloc.start()
    data6.get_mask(path='/home/baptiste/BBPipe')
    # data6.get_mask()

    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")
    # tracemalloc.stop()
    old = 0
    if old:
        d = data6.data
        ddt = np.einsum('ik...,...kj->ijk', d, d.T)
        # data6.ddt = ddt
        ddtPnoise = (ddt+model6.noise_covariance[..., np.newaxis])
        # ddtPnoise_masked = ddtPnoise*data6.mask
        ddtPnoise_masked_cleaned = []  # np.array([p for p in ddtPnoise_masked if np.all(p !=0)])
        for i in range(len(data6.mask)):
            if data6.mask[i]:
                ddtPnoise_masked_cleaned.append(ddtPnoise[:, :, i])
        ddtPnoise_masked_cleaned = np.array(ddtPnoise_masked_cleaned).T

        current, peak = tracemalloc.get_traced_memory()
        print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")
    else:
        S_cmb = np.load('S_cmb_n128_s1000_new.npy')
        ASAt = model6.mix_effectiv[:, :2].dot(S_cmb).dot(model6.mix_effectiv[:, :2].T)
        fg_freq_maps_full = data6.miscal_matrix.dot(data6.mixing_matrix)[
            :, 2:].dot(data6.signal[2:])
        ddt_fg = np.einsum('ik...,...kj->ijk', fg_freq_maps_full, fg_freq_maps_full.T)
        ddt_fg *= data6.mask
        n_obspix = np.sum(data6.mask == 1)
        F = np.sum(ddt_fg, axis=-1)/n_obspix
        ddtPnoise_masked_cleaned = n_obspix*(F + ASAt + model6.noise_covariance)
        current, peak = tracemalloc.get_traced_memory()
        print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")

    attr_remove = 1
    if attr_remove:
        delattr(data6, 'cmb_freq_maps')
        delattr(data6, 'dust_freq_maps')
        delattr(data6, 'sync_freq_maps')
        delattr(data6, 'cmb_freq_rot')
        delattr(data6, 'signal')
        delattr(data6, 'mask')
        delattr(data6, 'data')

        delattr(model6, 'noise_cov_ell')
        delattr(model6, 'inv_noise_ell')
        delattr(model6, 'noise_covariance')
        del data6
        if old:
            del ddt, ddtPnoise

    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")

    angle_prior = []
    for d in range(6):
        angle_prior.append([true_miscal_angles.value[d], prior_precision, int(d)])
    angle_prior = np.array(angle_prior)
    param_array = np.array([0., 0.08333333, 0.16666667, 0.25, 0.33333333,
                            0.41666667, 0., 1.59, -3])

    start = time.time()
    test_ddt_clean = get_chi_squared_local(
        param_array, ddtPnoise_masked_cleaned, model6, True, [], angle_prior, True, True)
    print('temps test = ', time.time() - start)

    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage BEFORE SAMPLE is {current / 10**6}MB; Peak was {peak / 10**6}MB")

    start = time.time()
    flat_samples, flat_samples_raw = run_MCMC(
        ddtPnoise_masked_cleaned, model6, sampled_miscal_freq=6, nsteps=nsteps, discard_num=discard,
        sampled_birefringence=birefringence, prior=prior_flag,
        walker_per_dim=2, prior_precision=prior_precision,
        prior_index=prior_indices, spectral_index=spectral, return_raw_samples=True,
        save=True, path=path, parallel=wMPI, nside=nside)

    print('time sampling in s = ', time.time() - start)
    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage AFTER SAMPLE is {current / 10**6}MB; Peak was {peak / 10**6}MB")
    IPython.embed()
    exit()


######################################################
# MAIN CALL
if __name__ == "__main__":
    main()


'''=========================PURGATORY=========================================

    sampled_miscal_freq = 6
    sampled_birefringence = True
    ndim = sampled_miscal_freq + sampled_birefringence
    nwalkers = 2*ndim
    nsteps = 2500
    discard_num = 100

    std_6 = np.load(
        '/home/baptiste/Documents/these/pixel_based_analysis/code/std_6angles_prior1arcmin.npy')

    prior_precision = (1*u.arcmin).to(u.rad).value
    angle_prior = []
    prior_index = 0

    for d in range(sampled_miscal_freq):
        angle_prior.append([true_miscal_angles.value[d], prior_precision, int(d)])
    angle_prior_total = np.array(angle_prior)
    angle_prior = np.array([angle_prior[prior_index]])
    print(angle_prior)
    print(data6.frequencies[int(angle_prior[:, 2])])

    p0 = np.random.normal(
        true_miscal_angles.value[:sampled_miscal_freq], std_6*2, (nwalkers, sampled_miscal_freq))

    if sampled_birefringence:
        p0_bir = np.array([np.random.normal(0, std_6.max(), (nwalkers))])
        p0 = np.concatenate((p0, p0_bir.T), axis=1)

    sampler6_prior = emcee.EnsembleSampler(
        nwalkers, ndim, get_chi_squared_local, args=[data6, model6, True, true_miscal_angles.value[sampled_miscal_freq:], angle_prior, True])
    # state = sampler.run_mcmc(p0, 10)
    sampler6_prior.reset()
    sampler6_prior.run_mcmc(p0, nsteps, progress=True)

    flat_samples = sampler6_prior.get_chain(discard=discard_num, flat=True)
    print('time sampling in s = ', time.time() - start)

    if sampled_birefringence:
        bir_file = '_birsampled'
    else:
        bir_file = ''
    file_name = 'sample{}_discard{}_prior_position{}_precision{:1.1e}rad'.format(
        nsteps, discard_num, prior_index, prior_precision).replace('.', 'p')+bir_file+'.npy'
    print(file_name)
    np.save('./prior_tests/'+file_name, flat_samples)
    '''
'''====================================================================='''

'''
    quick copy paste before night run
    '''
'''
    sampled_miscal_freq = 0
    sampled_birefringence = True
    ndim = sampled_miscal_freq + sampled_birefringence
    nwalkers = 4*ndim
    nsteps = 5000
    discard_num = 100
    p0_bir = np.array([np.random.normal(0, std_6.max(), (nwalkers))])

    sampler6_prior = emcee.EnsembleSampler(
        nwalkers, ndim, get_chi_squared_local, args=[data6, model6, False, true_miscal_angles.value[sampled_miscal_freq:], [], True])
    sampler6_prior.reset()
    sampler6_prior.run_mcmc(p0_bir.T, nsteps, progress=True)

    flat_samples = sampler6_prior.get_chain(discard=discard_num, flat=True)
    print('time sampling in s = ', time.time() - start)

    if sampled_birefringence:
        bir_file = '_birsampled'
    else:
        bir_file = ''
    # file_name = 'sample{}_discard{}_prior_position0&2_precision{:1.1e}rad'.format(
    #     nsteps, discard_num, prior_precision).replace('.', 'p')+bir_file+'.npy'
    file_name = 'sample5000_discard100_onlybirsampled.npy'
    file_name_raw = 'sample5000_raw_onlybirsampled.npy'

    print(file_name)
    np.save('./prior_tests/'+file_name, flat_samples)
    np.save('./prior_tests/'+file_name_raw, sampler6_prior.get_chain())

    sampler6_prior = emcee.EnsembleSampler(
        nwalkers, ndim, get_chi_squared_local, args=[data6, model6, True, true_miscal_angles.value[sampled_miscal_freq:], [], True])
    sampler6_prior.reset()
    sampler6_prior.run_mcmc(p0, nsteps, progress=True)

    flat_samples = sampler6_prior.get_chain(discard=discard_num, flat=True)
    print('time sampling in s = ', time.time() - start)

    if sampled_birefringence:
        bir_file = '_birsampled'
    else:
        bir_file = ''
    file_name = 'sample{}_discard{}_NOprior_precision{:1.1e}rad'.format(
        nsteps, discard_num, prior_precision).replace('.', 'p')+bir_file+'.npy'
    print(file_name)
    np.save('./prior_tests/'+file_name, flat_samples)
    '''
'''====================================================================='''

'''
    ploted_prior = [5]

    labels = [r'$\alpha_{{{}}}$'.format(i) for i in data6.frequencies]

    fig = corner.corner(flat_samples, labels=labels[:ndim],
                        truths=data6.miscal_angles.value[:ndim] % (0.5*np.pi),
                        quantiles=[0.16, 0.84], show_titles=True,
                        title_kwargs={"fontsize": 12})
    axes = np.array(fig.axes).reshape((ndim, ndim))
    for i in range(ndim):
        for w in range(nwalkers):
            axes[i, i].axvline(p0[w, i], color='g')

    for yi in range(ndim):
        for xi in range(yi):
            # axes[yi, xi].plot(angle_prior[xi, 0], angle_prior[yi, 0], "sr")
            for w in range(nwalkers):
                axes[yi, xi].plot(p0[w, xi], p0[w, yi], "sg")

    # for i in range(len(angle_prior)):
    for i in ploted_prior:
        axes[i, i].axvline(angle_prior_total[i, 0], color='r', linestyle='--')
        # for xi in range(len(angle_prior)):
        #     axes[int(angle_prior[yi, -1]), int(angle_prior[xi, -1])
        #          ].plot(angle_prior[xi, 0], angle_prior[yi, 0], "sr")
        #     print(angle_prior[xi, 0], angle_prior[yi, 0])
        #     print(xi)
        #     print(yi)
        # axes[xi,yi].plot(value2[xi], value2[yi], "sr")

    grid = np.arange(-0.5*np.pi, 0.5*np.pi, np.pi/100)
    grid_chi2 = []
    grid_prior = []
    start = time.time()
    for i in grid:
        chi2_g, prior_g = get_chi_squared_local([i], data6, model6, True,
                                                true_miscal_angles.value[1:],
                                                angle_prior[:1])
        grid_chi2.append(chi2_g)
        grid_prior.append(prior_g)
    print('mean time gridding with prior =', (time.time() - start)/len(grid))
    # serial_time = 535.61
    # with Pool() as pool:
    #     sampler = emcee.EnsembleSampler(nwalkers, ndim, get_chi_squared_local, args=[
    #                                     data6, model6, prior_, [1.5, 2, 2.5, 3]], pool=pool)
    #     start = time.time()
    #     sampler.run_mcmc(p0, nsteps, progress=True)
    #     end = time.time()
    #     multi_time = end - start
    #     print("Multiprocessing took {0:.1f} seconds".format(multi_time))
    #     print("{0:.1f} times faster than serial".format(serial_time / multi_time))
    # IPython.embed()
    flat_samples = sampler.get_chain(discard=10, flat=True)

    fig = corner.corner(
        flat_samples, labels=labels[:ndim], truths=data6.miscal_angles.value)

    results = minimize(get_chi_squared_local, [0.01, 0.01, 0.01, 0.01, 0.01, 0.01], (data, model, prior_),
                       bounds=[(-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi)])
    results2 = minimize(get_chi_squared_local, [0.01], (data2, model2, prior_),
                        bounds=[(-np.pi, np.pi)])

    IPython.embed()
    # print('results = ', results.x)
    # IPython.embed()
    # print('hessian = ', results.hess_inv)

    # visu.corner_norm(results.x, results.hess_inv)
    plt.show()
    # bir_grid, misc_grid = np.meshgrid(grid, grid,
    # indexing='ij')
    start = time.time()
    lSO.get_chi_squared([0, 0, 0], data, model, prior=True)
    print('time chi2 prior = ', time.time() - start)
    # slice_chi2 = np.array([[get_chi_squared([i, j, 0, 0], data, model) for i in grid]
    #                        for j in grid])
    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # ax.plot_surface(grid, misc_grid, slice_chi2, cmap=cm.viridis)
    # plt.show()

    plt.plot(grid, [-lSO.get_chi_squared([i], data, model) for i in grid])
    print('time grid in s = ', time.time() - start)

    # plt.yscale('log')
    plt.show()
    '''

'''

    def get_chi_squared_local(angle_array, data_skm, model_skm, prior=False,
                              fixed_miscal_angles=[], miscal_priors=[],
                              birefringence=False, spectral_index=False):
        if spectral_index:
            angle_index = - 2
            angle_index_none = angle_index

            if not 0.5 <= angle_array[-2] < 2.5 or not -5 <= angle_array[-1] <= -1:
                return -np.inf

            model_skm.evaluate_mixing_matrix([angle_array[-2], 20, angle_array[-1]])

        else:
            angle_index = 0
            angle_index_none = None

        if np.any(np.array(angle_array[:angle_index_none]) < (-0.5*np.pi)) or np.any(np.array(angle_array[:angle_index_none]) > (0.5*np.pi)):
            return -np.inf

        if birefringence:
            model_skm.miscal_angles = np.append(angle_array[:-(-angle_index+1)], fixed_miscal_angles)
            model_skm.bir_angle = angle_array[-(-angle_index+1)]

        else:
            model_skm.miscal_angles = np.append(angle_array[:angle_index_none], fixed_miscal_angles)
            model_skm.bir_angle = 0

        model_skm.get_miscalibration_angle_matrix()
        # model_skm.cmb_rotation()
        model_skm.get_bir_matrix()

        model_skm.get_projection_op()

        chi_squared = np.einsum('ij,jip->...', (-model_skm.projection+model_skm.inv_noise),
                                data_skm.ddt+model_skm.noise_covariance[..., np.newaxis])

        if prior:
            gaussian_prior = 0
            for l in range(len(miscal_priors)):
                gaussian_prior = np.sum((1/(2*(miscal_priors[l, 1]**2)))
                                        * (angle_array[int(miscal_priors[l, 2])] - miscal_priors[l, 0])**2)

            return chi_squared - gaussian_prior

        return chi_squared
    '''
