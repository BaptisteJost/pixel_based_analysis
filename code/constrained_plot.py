from config import *
import argparse
import IPython
import matplotlib.lines as mlines
from astropy import units as u
import copy
import numpy as np
from getdist.gaussian_mixtures import GaussianND
from getdist import plots, MCSamples
import matplotlib.pyplot as plt
from full_plots import plot_options


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder_end", help="end of the folder name")
    args = parser.parse_args()

    # save_path__ = save_path_.replace('06', '05')
    save_path = save_path_ + args.folder_end + '/'
    print()
    print(save_path)

    rad2deg = (1*u.rad).to(u.deg).value

    plt.switch_backend("Qt5Agg")

    cosmo_sample = np.load(save_path+'cosmo_samples.npy')
    cosmo_sample[:, 1:] *= rad2deg

    labels = [r'r', r'{\beta_b}', r'{\alpha_p}']
    label_sample = 'Cosmological likelihood'

    markers = {}
    # pivot_angle_index = 2
    alpha_true = true_miscal_angles[pivot_angle_index]
    true_params = [r_true, beta_true.to(u.deg).value, alpha_true.to(u.deg).value]
    markers[labels[0]] = true_params[0]
    markers[labels[1]] = true_params[1]
    markers[labels[2]] = true_params[2]

    boundaries = {}
    boundaries[labels[0]] = [-0.01, 1]
    boundaries[labels[1]] = [-0.5*np.pi*rad2deg, 0.5*np.pi*rad2deg]
    boundaries[labels[2]] = [-0.5*np.pi*rad2deg, 0.5*np.pi*rad2deg]

    distsamples = MCSamples(samples=cosmo_sample, names=labels,
                            labels=labels, ranges=boundaries,
                            label=label_sample)
    # IPython.embed()
    cosmo_fisher = np.load(save_path+'fisher_cosmo_prior_array.npy')[0]
    cosmo_estimation = np.load(save_path+'cosmo_array.npy')[0]
    cov_cosmo = np.linalg.inv(cosmo_fisher)
    cov_cosmo_deg = copy.deepcopy(cov_cosmo)
    cov_cosmo_deg[1:] *= rad2deg
    cov_cosmo_deg[:, 1:] *= rad2deg

    cosmo_estimation[1:] *= rad2deg
    fisher = GaussianND(cosmo_estimation, cov_cosmo_deg, names=labels,
                        labels=labels)

    distsamples.smooth_scale_2D = -1
    distsamples.smooth_scale_1D = -1
    options_dict = {}
    options_dict['plot_samples'] = [distsamples, fisher]
    options_dict['labels'] = labels
    options_dict['markers'] = markers

    options_dict['filled'] = [True, False]

    options_dict['legend_labels'] = [label_sample, 'Fisher estimation']

    options_dict['contour_lws'] = [1.5, 2]
    options_dict['contour_colors'] = ['orange', 'maroon']
    options_dict['contour_ls'] = ['-', '--']

    g = plot_options(options_dict, index_sigma=0)
    legend_args = {}
    legend_args['prop'] = {'size': 8}
    lines = g.contours_added
    plot_pivot = 0
    if plot_pivot:
        pivot_range = np.load(save_path+'pivot_range.npy')
        prior_first_term = np.load(save_path+'prior_first_term.npy')
        prior_second_term = np.load(save_path+'prior_second_term.npy')
        g.subplots[2, 2].plot(pivot_range*rad2deg, prior_first_term,
                              label='prior pivot', color='purple')
        g.subplots[2, 2].plot(pivot_range*rad2deg, prior_second_term,
                              label='prior second term', color='cyan')

        lines.append(g.subplots[2, 2].lines[-2])
        lines.append(g.subplots[2, 2].lines[-1])
        options_dict['legend_labels'].append('prior pivot')
        options_dict['legend_labels'].append('prior second term')
    # lines.append(mlines.Line2D([], [], **g.lines_added[1]))
    #
    g.legend = g.fig.legend(
        lines, options_dict['legend_labels'], loc=g.settings.legend_loc, **legend_args)
    IPython.embed()
    plt.savefig(save_path+'cosmo_MCMC.png', bbox_inches='tight', dpi=200)
    plt.savefig(save_path+'cosmo_MCMC.pdf', bbox_inches='tight', dpi=200)
    plt.close()

    '''Plot Fisher spectral results'''
    frequencies_plot_nopivot = np.delete(frequencies_plot, pivot_angle_index)
    labels = [r'\alpha_{{{}}}'.format(i) for i in frequencies_plot_nopivot]
    if spectral_flag:
        labels.append(r'\beta_d')
        labels.append(r'\beta_s')
        true_params_ = np.append(np.delete(true_miscal_angles.value, pivot_angle_index), [1.54, -3])
    markers = {}
    true_params = copy.deepcopy(true_params_)
    true_params[:5] *= rad2deg
    for i in range(len(labels)):
        markers[labels[i]] = true_params[i]

    spectral_fisher = np.load(save_path+'fisher_pivot_array.npy')[0]
    spectral_estimation = np.load(save_path+'results_min_array.npy')[0]

    cov_spectral = np.linalg.inv(spectral_fisher)
    cov_deg = copy.deepcopy(cov_spectral)
    cov_deg[:5] *= rad2deg
    cov_deg[:, :5] *= rad2deg
    spectral_estimation[:5] *= rad2deg

    fisher_spectral = GaussianND(spectral_estimation, cov_deg, names=labels, labels=labels)

    options_dict = {}
    options_dict['labels'] = labels
    options_dict['markers'] = markers
    options_dict['plot_samples'] = []
    options_dict['filled'] = []

    options_dict['legend_labels'] = []
    options_dict['contour_lws'] = []
    options_dict['contour_colors'] = []
    options_dict['contour_ls'] = []
    title_limit_spectral = 0

    if spectral_MCMC_flag:
        spectral_sample = np.load(save_path+'samples_spectral.npy')
        spectral_sample[:, :5] *= rad2deg
        distsamples = MCSamples(samples=spectral_sample, names=labels, labels=labels,
                                ranges=boundaries, label=label_sample)

        options_dict['plot_samples'].append(distsamples)
        options_dict['filled'].append(True)

        options_dict['legend_labels'].append('MCMC sampling')
        options_dict['contour_lws'].append(1.5)
        options_dict['contour_colors'].append('orange')
        options_dict['contour_ls'].append('-')
        title_limit_spectral = 1

    options_dict['plot_samples'].append(fisher_spectral)
    options_dict['filled'].append(False)

    options_dict['legend_labels'].append('Fisher estimation')
    options_dict['contour_lws'].append(2)
    options_dict['contour_colors'].append('maroon')
    options_dict['contour_ls'].append('--')

    g = plot_options(options_dict, index_sigma=0, title_limit=title_limit_spectral)

    legend_args = {}
    legend_args['prop'] = {'size': 20}
    lines = g.contours_added[:-1]
    lines.append(mlines.Line2D([], [], **g.lines_added[0]))
    g.legend = g.fig.legend(
        lines, options_dict['legend_labels'], loc=g.settings.legend_loc, **legend_args)

    plt.savefig(save_path+'spectral_MCMC.png', bbox_inches='tight', dpi=200)
    plt.savefig(save_path+'spectral_MCMC.pdf', bbox_inches='tight', dpi=200)

    exit()


######################################################
# MAIN CALL
if __name__ == "__main__":
    main()
