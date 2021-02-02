import numpy as np
import matplotlib.pyplot as plt
import corner
from astropy import units as u
from scipy.stats import norm
import IPython
import argparse
import pixel_based_angle_estimation
from mpi4py import MPI


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
# args = parser.parse_args()
#
# nsteps = args.nsteps
# discard = args.discard
# birefringence = args.birefringence
# spectral = args.spectral
# prior_indices = args.prior_indices

nsteps = 100  # args.nsteps
discard = 1  # args.discard
birefringence = 1  # args.birefringence
spectral = 1  # args.spectral
prior_indices = [0, 1]  # args.prior_indices

ndim = 6+birefringence+2*spectral
nwalkers = 2*ndim
prior_precision = (1*u.arcmin).to(u.rad).value
sampled_miscal_freq = 6

path_NERSC = '/global/homes/j/jost/these/pixel_based_analysis/results_and_data/'
path_local = './prior_tests/'

path = path_NERSC

file_name, file_name_raw = pixel_based_angle_estimation.get_file_name_sample(
    sampled_miscal_freq, nsteps, discard,
    sampled_birefringence=birefringence, prior=True,
    prior_precision=prior_precision,
    prior_index=prior_indices, spectral_index=spectral)
print(file_name)

true_miscal_angles = np.array([0., 0.08333333, 0.16666667, 0.25, 0.33333333,
                               0.41666667])
frequencies = np.array([27,  39,  93, 145, 225, 280])

labels = [r'$\alpha_{{{}}}$'.format(i) for i in frequencies]
if birefringence:
    labels.append(r'B')
    true_miscal_angles = np.append(true_miscal_angles, 0.)
if spectral:
    labels.append(r'$\beta_d$')
    labels.append(r'$\beta_s$')
    true_miscal_angles = np.append(true_miscal_angles, [1.59, -3])


print(file_name)

flat_samples = np.load(path+file_name)

fig = corner.corner(flat_samples, labels=labels,
                    truths=true_miscal_angles,
                    quantiles=[0.16, 0.84], show_titles=True, title_fmt='1.2e',
                    title_kwargs={"fontsize": 12}, hist_kwargs={'density': True})
axes = np.array(fig.axes).reshape((ndim, ndim))

for ii in range(prior_indices[0], prior_indices[-1]):
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
plt.savefig(path + 'test' + file_name[:-4])
exit()