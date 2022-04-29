import matplotlib.lines as mlines
from astropy import units as u
import copy
import numpy as np
from getdist.gaussian_mixtures import GaussianND
from getdist import plots, MCSamples
# from astropy import units as u
import IPython
import matplotlib.pyplot as plt
path_local = './samples/'
# save_path = './'
save_path = '/home/baptiste/Documents/these/pixel_based_analysis/results_and_data/MCMCrerun_article/'


rad2deg = (1*u.rad).to(u.deg).value
save_file = 'dist_nside128_10ksamples_1ksim_0p01deg_smooth2'
frequencies = np.array([27,  39,  93, 145, 225, 280])
freq_number = len(frequencies)
spectral = 1
# true_miscal_angles = (np.array([0.28]*freq_number)*u.deg).to(u.rad).value
true_miscal_angles = (np.array([0]*freq_number)*u.deg).to(u.rad).value
# true_miscal_angles = (np.arange(1, 5, 4 / freq_number)*u.deg).to(u.rad)[::-1].value

prior = False
prior_indices = []
# prior_indices = [0, 6]

prior_precion_1deg = 1*u.deg
prior_precion_0p1deg = 0.1*u.deg
prior_precion_0p01deg = 0.01*u.deg

prior_precision_plot = prior_precion_0p1deg.to(u.rad).value


plt.switch_backend("Qt5Agg")

labels = [r'\alpha_{{{}}}'.format(i) for i in frequencies]
if spectral:
    labels.append(r'\beta_d')
    labels.append(r'\beta_s')
    true_miscal_angles = np.append(true_miscal_angles, [1.59, -3])

markers = {}
true_miscal_angles_deg = copy.deepcopy(true_miscal_angles)
true_miscal_angles_deg[:6] *= rad2deg
for i in range(8):
    markers[labels[i]] = true_miscal_angles_deg[i]

boundaries = {}
for i in range(6):
    boundaries[labels[i]] = [-0.5*np.pi*rad2deg, 0.5*np.pi*rad2deg]
boundaries[labels[6]] = [1.4, 1.8]
boundaries[labels[7]] = [-3.5, -2.5]

file_name_1deg = '/home/baptiste/Downloads/' + 'MCMC_spectral_prior1.0BFGS.npy'
# file_name_0p1deg = '/home/baptiste/Downloads/' + 'MCMC_spectral_prior0.1BFGS_raw.npy'
# 'new_20000Samples_RAW_v2MiscalAll0_MiscalFrom0to6_PriorPosition2to3_Precision1p7e-03rad_SpectralSampled_Mask0400_nside512_FSN_SOf2f.npy'
# 'new_60000Samples_RAW_v2MiscalAll0_MiscalFrom0to6_PriorPosition5to6_Precision1p7e-03rad_SpectralSampled_Mask0400_nside512_FSN_SOf2f.npy'
# 'new_60000Samples_RAW_v2MiscalAll0_MiscalFrom0to6_SpectralSampled_Mask0400_nside512_FSN_SOf2f.npy'
file_name_0p1deg = '/home/baptiste/Documents/these/pixel_based_analysis/results_and_data/MCMCrerun_article/' + \
    'new_0miscal_60000Samples_RAW_v2MiscalAll0_MiscalFrom0to6_SpectralSampled_Mask0400_nside512_FSN_SOf2f.npy'
file_name_0p01deg = '/home/baptiste/Downloads/' + 'MCMC_spectral_prior0.01BFGS.npy'

sample_1deg = np.load(file_name_1deg)
sample_0p1deg_raw = np.load(file_name_0p1deg)
sample_0p01deg = np.load(file_name_0p01deg)
# fisher_matrix = np.load('/home/baptiste/Downloads/' + 'fisher_spectral_grid(1).npy')[1]
fisher_matrix = np.load(
    '/home/baptiste/Documents/these/pixel_based_analysis/results_and_data/MCMCrerun_article/' +
    'fisher_spectral_noprior_0miscal.npy')
# 'fisher_spectral_noprior.npy')

cov = np.linalg.inv(fisher_matrix)
# cov[-2:, -2:] = np.linalg.inv(fisher_matrix[-2:, -2:])
# cov[:-2, -2:] = 0
# cov[-2:, :-2] = 0

cov_deg = copy.deepcopy(cov)
cov_deg[:6] *= rad2deg
cov_deg[:, :6] *= rad2deg

raw_shape = sample_0p1deg_raw.shape
new_discard = 25000
sample_0p1deg = sample_0p1deg_raw[new_discard:].reshape(
    (raw_shape[0]-new_discard)*raw_shape[1], raw_shape[2])
sample_0p1deg[:, :6] *= rad2deg


label_1deg = '-2logL, prior precision = {}'.format(prior_precion_1deg)
label_0p1deg = '-2logL, prior precision = {}'.format(prior_precion_0p1deg)
label_0p01deg = '-2logL, prior precision = {}'.format(prior_precion_0p01deg)

# distsamples1deg = MCSamples(samples=sample_1deg, names=labels, labels=labels,
#                             ranges=boundaries, label=label_1deg)
# labels2 = copy.deepcopy(labels)
# for i in range(6):
#     labels2[i] = '$'+labels2[i] + '$ in deg'
# labels2 = [r'$\alpha_{{{}}}$ \text{{in deg}}'.format(i) for i in frequencies]
# if spectral:
#     labels2.append(r'\beta_d')
#     labels2.append(r'\beta_s')
distsamples0p1deg = MCSamples(samples=sample_0p1deg, names=labels, labels=labels,
                              ranges=boundaries, label=label_0p1deg)
# distsamples0p01deg = MCSamples(samples=sample_0p01deg, names=labels, labels=labels,
#                                ranges=boundaries, label=label_0p01deg)
fisher = GaussianND(true_miscal_angles_deg, cov_deg, names=labels, labels=labels)

# distsamples1deg.smooth_scale_2D = 2
if prior:
    distsamples0p1deg.smooth_scale_2D = -1
else:
    distsamples0p1deg.smooth_scale_2D = -1
# distsamples0p01deg.smooth_scale_2D = 2
fisher.smooth_scale_2D = 2
# plot_samples = [distsamples1deg, distsamples0p1deg, distsamples0p01deg]
# plot_samples = [distsamples0p1deg, distsamples0p01deg]
plot_samples = [distsamples0p1deg, fisher]
# filled = [True, True, True]
filled = [True, False]
legend_labels = ['-2logL, prior precision = {}'.format(prior_precion_1deg),
                 '-2logL, prior precision = {}'.format(prior_precion_0p1deg),
                 'Fisher estimation, prior precision = {}'.format(prior_precion_0p1deg)][1:]
contour_lws = [1.5, 2]
# contour_colors = ['b', 'r', 'g']
contour_colors = ['orange', 'maroon']
contour_ls = ['-', '-']

# IPython.embed()


def prior_samples(prior_precision, prior_indices, distrib_centers):
    prior_num = prior_indices[-1]-prior_indices[0]
    prior_cov = np.zeros([prior_num, prior_num])
    prior_precision *= rad2deg
    for i in range(prior_num):
        prior_cov[i, i] = prior_precision**2

    prior_samples_num = 100000
    samps_gauss_prior = np.random.multivariate_normal(
        distrib_centers[prior_indices[0]:prior_indices[-1]], prior_cov, size=prior_samples_num)
    # samps_gauss_prior *= rad2deg

    sample_prior_none = np.zeros([8, prior_samples_num])
    counter = 0
    for i in range(prior_indices[0], prior_indices[-1]):
        sample_prior_none[i] = samps_gauss_prior.T[counter]
        counter += 1
    sample_prior_none = sample_prior_none.T
    prior_none = MCSamples(samples=sample_prior_none, names=labels, labels=labels)
    plot_samples.append(prior_none)
    filled.append(False)
    contour_colors.append('purple')
    label_gauss = 'Gaussian priors precision : {} deg'.format(prior_precision)
    legend_labels.append(label_gauss)
    contour_ls.append('--')
    contour_lws.append(4)
    # line_args.append({'lw': 1.5, 'ls': '--', 'color': 'g'})contour_colors
    return prior_none, label_gauss


def plot_options(plot_samples, legend_labels, index_sigma=-1):
    g = plots.get_subplot_plotter()
    g.settings.legend_loc = 'upper right'
    g.settings.fontsize = 20
    g.settings.legend_fontsize = 20
    g.settings.axes_fontsize = 15
    g.settings.axes_labelsize = 20
    g.settings.title_limit_fontsize = 20
    g.settings.linewidth_contour = 2
    g.settings.line_labels = False
    g.triangle_plot(plot_samples, filled=filled, contour_colors=contour_colors,
                    markers=markers, legend_labels=legend_labels,
                    contour_ls=contour_ls, contour_lws=contour_lws)

    sigma_number = 1
    distsamplesFSN = plot_samples[index_sigma]
    marge = distsamplesFSN.getMargeStats()
    for i in range(8):
        param = marge.parWithName(labels[i])
        lim = param.limits[sigma_number - 1]
        mean = param.mean
        upper = lim.upper - mean
        lower = lim.lower - mean
        mean_str = '{:.2e}'.format(mean)
        # mean_str = mean_str + '$\degree$'
        upper_str = '{:.2e}'.format(upper)
        lower_str = '{:.2e}'.format(lower)
        title_str = mean_str
        title_str += '^{+' + upper_str + '}_{' + lower_str + '}'

        g.subplots[i, i].set_title(r'$'+title_str+'$', fontsize=14)
        g.subplots[i, i].axvline(lim.lower, color='gray', ls='--', lw=2)
        g.subplots[i, i].axvline(lim.upper, color='gray', ls='--', lw=2)

    # legend_args = {}
    # legend_args['prop'] = {'size': 20}
    # lines = g.contours_added
    # g.legend = g.fig.legend(lines, legend_labels, loc=g.settings.legend_loc, **legend_args)
    return g


if prior:
    gaussian_prior_sample, label_gauss = prior_samples(
        prior_precision_plot, prior_indices, true_miscal_angles_deg)
g = plot_options(plot_samples, legend_labels, index_sigma=0)
legend_args = {}
legend_args['prop'] = {'size': 20}
lines = g.contours_added[:-1]
if prior:
    lines.append(mlines.Line2D([], [], **g.lines_added[2]))
else:
    lines.append(mlines.Line2D([], [], **g.lines_added[1]))

g.legend = g.fig.legend(lines, legend_labels, loc=g.settings.legend_loc, **legend_args)

# IPython.embed()

# g.add_legend(legend_labels, **legend_args)

# g.add_1d(gaussian_prior_sample, labels[prior_indices[0]], label=label_gauss, lw=3)
# plt.savefig(save_path+save_file+'_comp_precision_no1deg.png', bbox_inches='tight')
plt.savefig(save_path+'colortestMCMC0p1deg93GHz_noprior_0miscal_25kdiscards.png',
            bbox_inches='tight', dpi=200)
plt.savefig(save_path+'colortestMCMC0p1deg93GHz_noprior_0miscal_25kdiscards.pdf',
            bbox_inches='tight', dpi=200)
plt.savefig(save_path+'colortestMCMC0p1deg93GHz_noprior_0miscal_25kdiscards.svg',
            bbox_inches='tight', dpi=200)

IPython.embed()
