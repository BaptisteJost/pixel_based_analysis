import numpy as np
from getdist.gaussian_mixtures import GaussianND
from getdist import plots, MCSamples
from astropy import units as u
import IPython
from mpi4py import MPI
import matplotlib.pyplot as plt
import pixel_based_angle_estimation


plt.switch_backend("Qt5Agg")

nsteps = 5000  # args.nsteps
discard = 1000  # args.discard
birefringence = 1  # args.birefringence
spectral = 1  # args.spectral
prior_indices = [0, 6]  # args.prior_indices
prior_flag = True
nside = 512
fisher_flag = True
plot_2d_fisher_flag = True
ndim = 6+birefringence+2*spectral
nwalkers = 2*ndim
prior_precision = (1*u.arcmin).to(u.rad).value
sampled_miscal_freq = 6
frequencies = np.array([27,  39,  93, 145, 225, 280])

true_miscal_angles = np.array([0]*6)


path_NERSC = '/global/homes/j/jost/these/pixel_based_analysis/results_and_data/run02032021/'
path_local = '/home/baptiste/Documents/these/pixel_based_analysis/results_and_data/run02032021/'
# path_local = '/home/baptiste/Documents/these/pixel_based_analysis/code/test11k/'

path = path_local

wMPI2 = 1
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

    print('prior_indices = ', prior_indices)
    print('birefringence = ', birefringence)
    print('spectral = ', spectral)

file_name, file_name_raw = pixel_based_angle_estimation.get_file_name_sample(
    sampled_miscal_freq, nsteps, discard,
    sampled_birefringence=birefringence, prior=prior_flag,
    prior_precision=prior_precision,
    prior_index=prior_indices, spectral_index=spectral, nside=nside)
print(file_name)


labels = [r'\alpha_{{{}}}'.format(i) for i in frequencies]
if birefringence:
    labels.append(r'B')
    true_miscal_angles = np.append(true_miscal_angles, 0.)
if spectral:
    labels.append(r'\beta_d')
    labels.append(r'\beta_s')
    true_miscal_angles = np.append(true_miscal_angles, [1.59, -3])

markers = {}
for i in range(9):
    markers[labels[i]] = true_miscal_angles[i]

boundaries = {}
for i in range(7):
    boundaries[labels[i]] = [-0.5*np.pi, 0.5*np.pi]
boundaries[labels[7]] = [0.5, 2.5]
boundaries[labels[8]] = [-5, -1]

flat_samples = np.load(path+file_name)
cov = np.load(path+'inv_fisher_prior_'+file_name)

range_bir = max(np.abs(min(flat_samples[:, 6])), np.abs(max(flat_samples[:, 6])))
mean_bir = np.mean(flat_samples[:, 6])
limits_bir = [mean_bir-3*range_bir, mean_bir+3*range_bir]
limits_bir_2 = [min(flat_samples[:, 6]), max(flat_samples[:, 6])]

distsamples = MCSamples(samples=flat_samples, names=labels, labels=labels,
                        ranges=boundaries)
if prior_flag:
    distsamples.smooth_scale_2D = 2
fisher = GaussianND(true_miscal_angles, cov, names=labels, labels=labels)

plot_samples = [distsamples, fisher]
filled = [True, False, False]
contour_colors = ['b', 'r', 'g']
legend_labels = ['Spectral likelihood MCMC sampling', 'Fisher', 'Gaussian priors']
contour_ls = ['', '-.', '--']
contour_lws = [1.5, 1.5, 1.5]
line_args = [{'lw': 1.5, 'color': 'b'}, {'lw': 1.5, 'ls': '-.',
                                         'color': 'r'}, {'lw': 1.5, 'ls': '--', 'color': 'g'}]

if prior_flag:
    prior_num = prior_indices[-1]-prior_indices[0]
    prior_cov = np.zeros([prior_num, prior_num])
    for i in range(prior_num):
        prior_cov[i, i] = prior_precision**2

    prior_samples_num = 100000
    samps_gauss_prior = np.random.multivariate_normal(
        true_miscal_angles[prior_indices[0]:prior_indices[-1]], prior_cov, size=prior_samples_num)

    sample_prior_none = np.zeros([ndim, prior_samples_num])
    counter = 0
    for i in range(prior_indices[0], prior_indices[-1]):
        sample_prior_none[i] = samps_gauss_prior.T[counter]
        counter += 1
    sample_prior_none = sample_prior_none.T
    prior_none = MCSamples(samples=sample_prior_none, names=labels, labels=labels)
    plot_samples.append(prior_none)
    filled.append(False)
    contour_colors.append('g')
    legend_labels.append('Gaussian priors')
    contour_ls.append('--')
    contour_lws.append(1.5)
    line_args.append({'lw': 1.5, 'ls': '--', 'color': 'g'})
# sample_prior_none = samps_gauss_prior.T
# sample_prior_none = np.append(samps_gauss_prior.T, [[0]*samps_gauss_prior.shape[0]], axis=0)
# sample_prior_none = np.append(sample_prior_none, [[0]*samps_gauss_prior.shape[0]], axis=0)
# sample_prior_none = np.append(sample_prior_none, [[0]*samps_gauss_prior.shape[0]], axis=0)
# sample_prior_none = sample_prior_none.T


g = plots.get_subplot_plotter()
g.settings.legend_loc = 'upper right'
g.settings.fontsize = 20
g.settings.legend_fontsize = 20
g.settings.linewidth_contour = 1.5

# if prior_flag and prior_indices[-1] - prior_indices[0] <= 1:
g.settings.line_labels = False

g.triangle_plot(plot_samples, filled=filled, contour_colors=contour_colors,
                markers=markers, legend_labels=legend_labels,
                contour_ls=contour_ls, contour_lws=contour_lws,
                line_args=line_args)

# plt.legend(legend_labels)
'''Title limits generation...'''

sigma_number = 1
marge = distsamples.getMargeStats()
for i in range(ndim):
    param = marge.parWithName(labels[i])
    lim = param.limits[sigma_number - 1]
    mean = param.mean
    upper = lim.upper - mean
    lower = lim.lower - mean
    mean_str = '{:.2e}'.format(mean)
    upper_str = '{:.2e}'.format(upper)
    lower_str = '{:.2e}'.format(lower)
    title_str = mean_str
    title_str += '^{+' + upper_str + '}_{' + lower_str + '}'

    g.subplots[i, i].set_title(r'$'+title_str+'$')
    g.subplots[i, i].axvline(lim.lower, color='gray', ls='--')
    g.subplots[i, i].axvline(lim.upper, color='gray', ls='--')


lines = g.contours_added
legend_args = {}
legend_args['prop'] = {'size': 20}

if prior_flag and g.contours_added[2] is None:
    print('none in fisher for legend')
    line_fisher = g.lines_added.get(2)
    line_fisher.pop('filled', None)
    import matplotlib
    handle_fisher = matplotlib.lines.Line2D([0, 1], [0, 1], **line_fisher)
    lines[2] = handle_fisher


g.legend = g.fig.legend(lines, legend_labels, loc=g.settings.legend_loc, **legend_args)
# IPython.embed()

'''
g.extra_artists = []
delattr(g, 'legend')
a = g.add_legend(legend_labels,g.settings.figure_legend_loc, figure=True)
g.extra_artists = [a]
g._subplots_adjust()
'''
# plt.savefig('test_dist', bbox_inches='tight')
# IPython.embed()
plt.savefig(path + '2getdist' + file_name[:-4], bbox_inches='tight')

exit()
# g.subplots[6, 0].set_ylim(limits_bir_2)

# gauss_prior = GaussianND(
#     true_miscal_angles[prior_indices[0]:prior_indices[-1]], prior_cov,
#     names=labels[prior_indices[0]:prior_indices[-1]],
#     labels=labels[prior_indices[0]:prior_indices[-1]])
# samps_gauss = np.random.multivariate_normal(true_miscal_angles, cov, size=20000)
# gauss_2 = MCSamples(samples=samps_gauss, names=labels, labels=labels)
# gauss_prior_2 = MCSamples(samples=samps_gauss_prior,
#                           names=labels[prior_indices[0]:prior_indices[-1]], labels=labels[prior_indices[0]:prior_indices[-1]])


# g.triangle_plot([distsamples, gauss], filled=[True, False], contour_colors=['b', 'r'],
#                 markers=markers)
# g.triangle_plot([gauss, gauss_2], filled=[True, False], contour_colors=['r', 'black'],
#                 markers=markers)

# g.triangle_plot([gauss_prior, gauss_prior_2], filled=[True, False], contour_colors=['b', 'r'])
# g.subplots[6, 6].set_xlim(limits_bir)

# for i in range(6):
#     print(i)
#     g.subplots[6, i].set_ylim(limits_bir_2)
# for i in range(7, 9):
#     g.subplots[i, 6].set_xlim(limits_bir_2)

# g.triangle_plot([distsamples, gauss, gauss_prior], filled=[True, False, False], contour_colors=['b', 'r', 'b'],
#                 markers=markers)
# IPython.embed()
