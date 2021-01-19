import matplotlib.pyplot as plt
import time
import IPython
import numpy as np
from scipy.optimize import minimize
import bjlib.likelihood_SO as lSO
from astropy import units as u
import bjlib.V3calc as V3
import corner
import emcee
from multiprocessing import Pool


def get_chi_squared_local(angle_array, data_skm, model_skm, prior=False, fixed_miscal_angles=[], miscal_priors=[], birefringence=False):
    print(angle_array)
    if np.any(np.array(angle_array) < (-0.5*np.pi)) or np.any(np.array(angle_array) > (0.5*np.pi)):
        # print('return -inf')
        return -np.inf

    # start = time.time()
    # model_skm.miscal_angles = np.append(angle_array, fixed_miscal_angles)  # , angle_array[1],
    # angle_array[2], angle_array[3], angle_array[4], angle_array[5]]  # , angle_array[1], 0]
    # model_skm.frequencies_by_instrument = [6, 0, 0]
    # model_skm.miscal_angles = [1, 2, angle_array[0]]
    # model_skm.bir_angle = 0  # angle_array[1]
    # print('time init angle in s = ', time.time() - start)

    if birefringence:
        model_skm.miscal_angles = np.append(angle_array[:-1], fixed_miscal_angles)
        model_skm.bir_angle = angle_array[-1]

    else:
        model_skm.miscal_angles = np.append(angle_array, fixed_miscal_angles)
        model_skm.bir_angle = 0
    print('model_skm.miscal_angles=', model_skm.miscal_angles)
    print('model_skm.bir_angle=', model_skm.bir_angle)

    # start = time.time()

    model_skm.get_miscalibration_angle_matrix()
    model_skm.cmb_rotation()
    # print('time init matrix in s = ', time.time() - start)

    # if prior:
    #
    #     model_skm.prim_rotation()
    #     model_skm.get_bir_prior()
    #     model_skm.get_projection_op(prior=prior)
    #
    #     d = np.dot(data_skm.mix_effectiv, data_skm.alm_data)
    #     ddt = np.absolute(np.einsum('ik...,...kj->ijk', d, np.matrix.getH(d)))
    #
    #     first_term = model_skm.inv_noise_ell - model_skm.projection
    #
    #     noise_elltot_ = np.append([np.zeros([12, 12])],
    #                               model_skm.noise_cov_ell, 0)
    #     noise_elltot = np.append([np.zeros([12, 12])], noise_elltot_, 0)
    #
    #     ell = hp.Alm.getlm(np.shape(noise_elltot)[0]-1)[0]
    #     noise_cov_lm = [noise_elltot[i] for i in ell]
    #
    #     first_term_tot_ = np.append([np.zeros([12, 12])], first_term, 0)
    #     first_term_tot = np.append([np.zeros([12, 12])], first_term_tot_, 0)
    #
    #     ell = hp.Alm.getlm(np.shape(first_term_tot)[0]-1)[0]
    #     first_term_lm = [first_term_tot[i] for i in ell]
    #
    #     second_term_lm = np.array([noise_cov_lm[i] + ddt[:, :, i]
    #                                for i in range(len(ddt[0, 0, :]))]).T
    #
    #     in_sum = np.einsum('l...ij,jk...l->ikl',
    #                        first_term_lm, second_term_lm)

    # else:
    # d = data_skm.data

    # ddt = np.einsum('ik...,...kj->ijk', d, d.T)
    # start = time.time()

    # ddt = data_skm.ddt
    model_skm.get_projection_op()
    # return model_skm.projection
    # print('time ddt init and projection op in s = ', time.time() - start)

    # print(model_skm.projection)
    # print(model_skm.projection.shape)
    # start = time.time()

    # first_term = model_skm.inv_noise - model_skm.projection
    # print('time 1st term computation in s = ', time.time() - start)

    # start = time.time()

    # second_term = np.array([model_skm.noise_covariance + ddt[:, :, i]
    #                         for i in range(len(ddt[0, 0, :]))]).T
    # print('time 2nd term computation in s = ', time.time() - start)

    # start = time.time()
    # in_sum = np.einsum('ij,jk...l->ikl', first_term, second_term)
    # print('time in sum computation in s = ', time.time() - start)

    chi_squared = np.einsum('ij,jip->...', (-model_skm.projection+model_skm.inv_noise),
                            data_skm.ddt+model_skm.noise_covariance[..., np.newaxis])
    # print('chi_squared =', chi_squared)
    # print('model_skm.projection = ', model_skm.projection)
    # print('model_skm.bir_matrix = ', model_skm.bir_matrix)
    # print('model_skm.mix_effectiv = ', model_skm.mix_effectiv)

    if prior:
        print('PRIOR')
        gaussian_prior = 0
        for l in range(len(miscal_priors)):
            gaussian_prior = np.sum((1/(2*(miscal_priors[l, 1]**2)))
                                    * (angle_array[int(miscal_priors[l, 2])] - miscal_priors[l, 0])**2)
            # print('l =', l)
            # print('angle_array =', angle_array)
            # print('angle_array[int(miscal_priors[l, 2])]=', angle_array[int(miscal_priors[l, 2])])
            # print('miscal_priors[l, :] =', miscal_priors[l, :])
        return chi_squared - gaussian_prior
        # gaussian_prior = 0
        # for l in range(len(miscal_priors)):
        #     gaussian_prior = np.sum(1/(2*miscal_priors[l, 1]**2)
        #                             * (angle_array[l] - miscal_priors[l, 0])**2)
        # return chi_squared - gaussian_prior
        # return chi_squared, gaussian_prior

        # chi_squared += 0
    # start = time.time()
    # sum = np.sum(in_sum, -1)
    # print('time sum in s = ', time.time() - start)
    #
    # start = time.time()
    #
    # chi_squared = np.trace(sum)
    # print('time trace in s = ', time.time() - start)

    return chi_squared


def data_and_model_quick(miscal_angles_array, frequencies_array, frequencies_by_instrument_array, bir_angle=0*u.rad):
    data = lSO.sky_map(bir_angle=bir_angle, miscal_angles=miscal_angles_array,
                       frequencies_by_instrument=frequencies_by_instrument_array)
    model = lSO.sky_map(bir_angle=0*u.rad, miscal_angles=miscal_angles_array*0,
                        frequencies_by_instrument=frequencies_by_instrument_array)

    v3f = V3.so_V3_SA_bands()
    index = np.in1d(v3f, frequencies_array).nonzero()[0]

    data.from_pysm2data()
    model.from_pysm2data()

    data.get_pysm_sky()
    # data.get_frequency()
    data.frequencies = frequencies_array

    data.get_freq_maps()
    data.cmb_rotation()
    data.get_signal()
    data.get_mixing_matrix()
    data.get_miscalibration_angle_matrix()
    data.get_data()
    data.get_noise()
    data.inv_noise = data.inv_noise[index[0]*2:index[-1]*2 + 2,
                                    index[0]*2:index[-1]*2 + 2]
    data.noise_covariance = data.noise_covariance[index[0]*2:index[-1]*2 + 2,
                                                  index[0]*2:index[-1]*2 + 2]

    data.get_projection_op()
    data.data2alm()
    data.get_primordial_spectra()

    model.get_pysm_sky()
    # model.get_frequency()
    model.frequencies = frequencies_array
    model.get_freq_maps()
    model.cmb_rotation()
    model.get_signal()
    model.get_mixing_matrix()
    model.get_miscalibration_angle_matrix()
    model.get_data()
    model.get_noise()
    model.inv_noise = model.inv_noise[index[0]*2:index[-1]*2 + 2,
                                      index[0]*2:index[-1]*2 + 2]
    model.noise_covariance = model.noise_covariance[index[0]*2:index[-1]*2 + 2,
                                                    index[0]*2:index[-1]*2 + 2]
    model.get_projection_op()
    model.data2alm()
    model.get_primordial_spectra()

    return data, model


def main():
    # start = time.time()
    #
    # data = lSO.sky_map(bir_angle=(0.0*u.deg).to(u.rad),
    #                    miscal_angles=[0.5, 1, 1.5, 2, 2.5, 3]*u.rad, frequencies_by_instrument=[1, 1, 1, 1, 1, 1])
    # model = lSO.sky_map(bir_angle=(0.0*u.deg).to(u.rad),
    #                     miscal_angles=[0, 0, 0, 0, 0, 0]*u.rad, frequencies_by_instrument=[1, 1, 1, 1, 1, 1])
    #
    # data.from_pysm2data()
    # model.from_pysm2data()
    #
    # data.get_noise()
    # model.get_noise()
    # # IPython.embed()
    # data.get_projection_op()
    # model.get_projection_op()
    # # IPython.embed()
    #
    # data.data2alm()
    # model.data2alm()
    #
    # data.get_primordial_spectra()
    # model.get_primordial_spectra()

    # print('time chi2 in s = ', time.time() - start)

    start = time.time()
    true_miscal_angles = np.arange(0, 0.5, 0.5/6)*u.rad
    data6, model6 = data_and_model_quick(miscal_angles_array=true_miscal_angles,
                                         frequencies_array=V3.so_V3_SA_bands(),
                                         frequencies_by_instrument_array=[1, 1, 1, 1, 1, 1])

    # data3, model3 = data_and_model_quick(miscal_angles_array=[0.5, 1, 1.5]*u.rad,
    #                                      frequencies_array=data.frequencies[:3],
    #                                      frequencies_by_instrument_array=[1, 1, 1])

    # data4, model4 = data_and_model_quick(miscal_angles_array=[0.5, 1, 1.5, 2]*u.rad,
    #                                      frequencies_array=data.frequencies[:4],
    #                                      frequencies_by_instrument_array=[1, 1, 1, 1])

    # data5, model5 = data_and_model_quick(miscal_angles_array=[0.5, 1, 1.5, 2, 2.5]*u.rad,
    #                                      frequencies_array=data.frequencies[:5],
    #                                      frequencies_by_instrument_array=[1, 1, 1, 1, 1])

    # data1, model1 = data_and_model_quick(miscal_angles_array=[0.5]*u.rad,
    #                                      frequencies_array=np.array([data.frequencies[0]]),
    #                                      frequencies_by_instrument_array=[1])
    print('time initializing in s = ', time.time() - start)

    prior_ = False

    labels = [r'$\alpha_{{{}}}$'.format(i) for i in data6.frequencies]

    # results6 = minimize(get_chi_squared_local, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], (data6, model6, prior_),
    #                     bounds=[(-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi)])

    # start = time.time()
    d = data6.data
    #
    ddt = np.einsum('ik...,...kj->ijk', d, d.T)
    # print('time ddt in s = ', time.time() - start)
    data6.ddt = ddt
    #
    # start = time.time()
    # get_chi_squared_local(flat_samples[-1], data6, model6, False, [3])
    # print('time chi2 in s = ', time.time() - start)
    #
    # repeat_noise = np.tile(model6.noise_covariance.T, (196608, 1, 1)).T
    # start = time.time()
    # test = repeat_noise + ddt
    # print('time test in s = ', time.time() - start)
    #
    # start = time.time()
    # test = np.einsum('ij,jip->...', (-model6.projection+model6.inv_noise),
    #                  ddt+model6.noise_covariance[..., np.newaxis])
    # print('time test in s = ', time.time() - start)
    IPython.embed()

    start = time.time()
    sampled_miscal_freq = 6
    sampled_birefringence = True
    ndim = sampled_miscal_freq + sampled_birefringence
    nwalkers = 2*ndim
    nsteps = 2500
    discard_num = 100

    # p0 = np.random.rand(nwalkers, ndim) % (0.5*np.pi)
    # flat5_5k = np.load(
    #     '/home/baptiste/Documents/these/pixel_based_analysis/code/flat_samples_5k_5angles_100discard_0to0p5_newinitp0.npy')
    # std_5k = np.std(flat5_5k, axis=0)
    # std_6 = np.append(std_5k, std_5k.max())
    std_6 = np.load(
        '/home/baptiste/Documents/these/pixel_based_analysis/code/std_6angles_prior1arcmin.npy')
    # std_6plus1 = np.append(std_6, std_6.max())

    prior_precision = (1*u.arcmin).to(u.rad).value
    angle_prior = []
    prior_index = 0
    # angle_prior_measured = np.random.normal(
    #     true_miscal_angles.value[:ndim], [prior_precision]*ndim, (ndim))
    # for d in range(ndim):
    #     angle_prior.append([angle_prior_measured[d], prior_precision])

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

    '''====================================================================='''

    '''
    quick copy paste before night run
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

    '''====================================================================='''
    ploted_prior = [5]
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

    # IPython.embed()
    exit()


######################################################
# MAIN CALL
if __name__ == "__main__":
    main()
