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

def flat_and_discard(sample, discard_num):
    flat_samples = sample[discard_num:].reshape(
        sample[discard_num:].shape[0]*sample.shape[1], sample.shape[-1])
    return flat_samples


def multivariate_gaussian(pos, mu, Sigma):
    """Return the multivariate Gaussian distribution on array pos."""

    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2*np.pi)**n * Sigma_det)
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)

    return np.exp(-fac / 2) / N


def multivariate_gaussian_sigmaInv(pos, mu, Sigma_inv):
    """Return the multivariate Gaussian distribution on array pos."""

    n = mu.shape[0]
    Sigma_det = 1/np.linalg.det(Sigma_inv)
    # Sigma_inv = np.linalg.inv(Sigma)
    print('sigma_det =', Sigma_det)
    print('sigma_inv = ', Sigma_inv)
    N = np.sqrt((2*np.pi)**n * Sigma_det)
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)

    return np.exp(-fac / 2) / N


nsteps = 5000  # args.nsteps
discard = 500  # args.discard
birefringence = 1  # args.birefringence
spectral = 1  # args.spectral
prior_indices = [0, 6]  # args.prior_indices
prior_flag = True
nside = 128
fisher_flag = True
plot_2d_fisher_flag = False

wMPI2 = 0
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
"""
if wMPI2:
    comm = MPI.COMM_WORLD
    mpi_rank = MPI.COMM_WORLD.Get_rank()
    nsim = comm.Get_size()
    print(mpi_rank, nsim)
    prior_indices = [mpi_rank % 6, (mpi_rank % 6)+1]
    if mpi_rank//6 == 0:
        birefringence = 1
        spectral = 1

    if mpi_rank//6 == 1:
        birefringence = 0
        spectral = 1

    if mpi_rank//6 == 2:
        birefringence = 1
        spectral = 0
    print('prior_indices = ', prior_indices)
    print('birefringence = ', birefringence)
    print('spectral = ', spectral)
"""

ndim = 6+birefringence+2*spectral
nwalkers = 2*ndim
prior_precision = (1*u.arcmin).to(u.rad).value
sampled_miscal_freq = 6

path_NERSC = '/global/homes/j/jost/these/pixel_based_analysis/results_and_data/'
path_local = './prior_tests/'

path = path_local

file_name, file_name_raw = pixel_based_angle_estimation.get_file_name_sample(
    sampled_miscal_freq, nsteps, discard,
    sampled_birefringence=birefringence, prior=prior_flag,
    prior_precision=prior_precision,
    prior_index=prior_indices, spectral_index=spectral, nside=nside)
print(file_name)
# file_name_raw = '5000Samples_RAW_MiscalFrom0to6_PriorPosition0to6_Precision2p9e-04rad_BirSampled_SpectralSampled_MaskTest_nside512.npy'
# print(file_name_raw)

# true_miscal_angles = np.array([0., 0.08333333, 0.16666667, 0.25, 0.33333333,
#                                0.41666667])
true_miscal_angles = np.array([0]*6)
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
# raw_samples = np.load(path+file_name_raw)
# flat_samples = flat_and_discard(raw_samples, 2000)

# file_name2, file_name_raw2 = pixel_based_angle_estimation.get_file_name_sample(
#     sampled_miscal_freq, nsteps, 2000,
#     sampled_birefringence=birefringence, prior=True,
#     prior_precision=prior_precision,
#     prior_index=prior_indices, spectral_index=spectral)
# IPython.embed()

fig = corner.corner(flat_samples, labels=labels,
                    truths=true_miscal_angles,
                    quantiles=[0.16, 0.84], show_titles=True, title_fmt='1.2e',
                    title_kwargs={"fontsize": 12}, hist_kwargs={'density': True})
axes = np.array(fig.axes).reshape((ndim, ndim))
if prior_flag:
    for ii in range(prior_indices[0], prior_indices[-1]):
        ploted_prior = ii
        print(ii)
        axes[ploted_prior, ploted_prior].axvline(
            true_miscal_angles[ploted_prior], color='r', linestyle='--')
        min_range = min(flat_samples[:, ploted_prior])
        max_range = max(flat_samples[:, ploted_prior])
        x_range = np.arange(min_range, max_range, (max_range - min_range)/nsteps)
        axes[ploted_prior, ploted_prior].plot(x_range,
                                              norm.pdf(x_range, true_miscal_angles[ploted_prior], np.sqrt(
                                                  2)*prior_precision),
                                              color='red')
# x_range_prior = x_range
# norm_prior = norm.pdf(x_range, true_miscal_angles[ploted_prior], np.sqrt(2)*prior_precision)
# IPython.embed()
if fisher_flag:
    cov = np.load('sqrt_inv_fisher_all0_priorposition0to6_prior1arcmin_nside128_mask0400.npy')
    for ii in range(9):
        ploted_prior = ii
        print(ii)
        # axes[ploted_prior, ploted_prior].axvline(
        #     true_miscal_angles[ploted_prior], color='r', linestyle='--')
        min_range = min(flat_samples[:, ploted_prior])
        max_range = max(flat_samples[:, ploted_prior])
        x_range = np.arange(min_range, max_range, (max_range - min_range)/nsteps)
        axes[ploted_prior, ploted_prior].plot(x_range,
                                              norm.pdf(
                                                  x_range, true_miscal_angles[ploted_prior], cov[ii, ii]),
                                              color='green')
# x_range_fisher = x_range
# norm_fisher = norm.pdf(x_range, true_miscal_angles[ploted_prior], cov[ii, ii])
IPython.embed()
if plot_2d_fisher_flag:
    fisher_matrix = np.load('fisher_all0_priorposition0to6_prior1arcmin_nside128_mask0400.npy')
    nsteps = 1000
    for i in range(9):
        for ii in range(i, 9):
            print(i, ii)
            min_range_i = min(flat_samples[:, i])
            max_range_i = max(flat_samples[:, i])

            min_range_ii = min(flat_samples[:, ii])
            max_range_ii = max(flat_samples[:, ii])
            x_range_i = np.arange(min_range_i, max_range_i, (max_range_i - min_range_i)/nsteps)
            x_range_ii = np.arange(min_range_ii, max_range_ii, (max_range_ii - min_range_ii)/nsteps)

            grid_i, grid_ii = np.meshgrid(x_range_i, x_range_ii, indexing='xy')

            mu = np.array([true_miscal_angles[i], true_miscal_angles[ii]])
            pos = np.empty(grid_i.shape + (2,))
            pos[:, :, 0] = grid_i
            pos[:, :, 1] = grid_ii
            fisher_i_ii = np.array([[fisher_matrix[i, i], fisher_matrix[ii, i]],
                                    [fisher_matrix[i, ii], fisher_matrix[ii, ii]]])
            fisher_2D_gaussian = multivariate_gaussian_sigmaInv(pos, mu, fisher_i_ii)
            if i == 7 and ii == 8:
                print(fisher_2D_gaussian)
            cs2 = axes[ii, i].contour(grid_i, grid_ii, fisher_2D_gaussian, colors='g',
                                      linestyles='--')
# plt.savefig(path+'test_2d_fisher')
# file_name_save = '5000Samples_200discard_MiscalFrom0to6_PriorPosition0to6_Precision2p9e-04rad_BirSampled_SpectralSampled_MaskTest_nside512.npy'
if fisher_flag:
    plt.savefig(path + file_name[:-4] + '_fisher')
else:
    plt.savefig(path + file_name[:-4])

# plt.show()
exit()
