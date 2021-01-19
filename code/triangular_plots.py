import numpy as np
import matplotlib.pyplot as plt
import corner
from astropy import units as u
from scipy.stats import norm
import IPython

ndim = 6+1
nwalkers = 12
nsteps = 2500
discard_num = 100
prior_precision = (1*u.arcmin).to(u.rad).value

true_miscal_angles = np.array([0., 0.08333333, 0.16666667, 0.25, 0.33333333,
                               0.41666667, 0.])
frequencies = np.array([27,  39,  93, 145, 225, 280])

labels = [r'$\alpha_{{{}}}$'.format(i) for i in frequencies]
labels.append(r'$\beta$')

for i in range(1):
    ploted_prior = i
    file_name = '/home/baptiste/Documents/these/pixel_based_analysis/code/prior_tests/sample2500_discard100_NOprior_precision2p9e-04rad_birsampled.npy'
    # file_name = 'sample{}_discard{}_prior_position{}_precision{:1.1e}rad'.format(
    # nsteps, discard_num, ploted_prior, prior_precision).replace('.', 'p')+'.npy'
    print(file_name)

    # flat_samples = np.load('./prior_tests/'+file_name)
    flat_samples = np.load(file_name)
    IPython.embed()

    fig = corner.corner(flat_samples, labels=labels,
                        truths=true_miscal_angles,
                        quantiles=[0.16, 0.84], show_titles=True, title_fmt='1.2e',
                        title_kwargs={"fontsize": 12}, hist_kwargs={'density': True})
    axes = np.array(fig.axes).reshape((ndim, ndim))

    for ii in range(2):
        ploted_prior = ii
        axes[ploted_prior, ploted_prior].axvline(
            true_miscal_angles[ploted_prior], color='r', linestyle='--')
        min_range = min(flat_samples[:, ploted_prior])
        max_range = max(flat_samples[:, ploted_prior])
        x_range = np.arange(min_range, max_range, (max_range - min_range)/nsteps)
        axes[ploted_prior, ploted_prior].plot(x_range,
                                              norm.pdf(x_range, true_miscal_angles[ploted_prior], np.sqrt(
                                                  2)*prior_precision),
                                              color='red')
    plt.savefig('./prior_tests/'+'samples_Noprior_birsampled')
    # plt.savefig('./prior_tests/'+'samples_prior@27&39GHz_precision{:1.1e}_birsampled'.format(
    #     prior_precision).replace('.', 'p'))
    # IPython.embed()
