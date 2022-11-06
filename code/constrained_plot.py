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
from os.path import exists


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder_end", help="end of the folder name")
    args = parser.parse_args()

    # save_path__ = save_path_.replace('28', '27')
    save_path = save_path_ + args.folder_end + '/'
    print()
    print(save_path)

    rad2deg = (1*u.rad).to(u.deg).value

    plt.switch_backend("Qt5Agg")

    labels = [r'r', r'{\beta_b}', r'{\alpha^{\rm{ref}}}']

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

    options_dict = {}
    options_dict['plot_samples'] = []
    options_dict['filled'] = []

    options_dict['legend_labels'] = []

    options_dict['contour_lws'] = []
    options_dict['contour_colors'] = []
    options_dict['contour_ls'] = []

    if exists(save_path+'cosmo_samples.npy'):
        label_sample = 'Cosmological likelihood'
        cosmo_sample = np.load(save_path+'cosmo_samples.npy')
        cosmo_sample[:, 1:] *= rad2deg

        distsamples = MCSamples(samples=cosmo_sample, names=labels,
                                labels=labels, ranges=boundaries,
                                label=label_sample)
        distsamples.smooth_scale_2D = -1
        distsamples.smooth_scale_1D = -1
        options_dict['plot_samples'].append(distsamples)
        options_dict['filled'].append(True)

        options_dict['legend_labels'].append(label_sample)

        options_dict['contour_lws'].append(1.5)
        options_dict['contour_colors'].append('orange')
        options_dict['contour_ls'].append('-')
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
    options_dict['plot_samples'].append(fisher)
    options_dict['filled'].append(False)

    options_dict['legend_labels'].append('Fisher estimation')

    options_dict['contour_lws'].append(2)
    options_dict['contour_colors'].append('darkred')
    options_dict['contour_ls'].append('-')

    options_dict['labels'] = labels
    options_dict['markers'] = markers

    g = plot_options(options_dict, index_sigma=0, title_limit=0)
    legend_args = {}
    legend_args['prop'] = {'size': 8}
    lines = g.contours_added
    plot_pivot = 1
    if plot_pivot:
        pivot_range = np.load(save_path+'pivot_range.npy')
        prior_first_term = np.load(save_path+'prior_first_term.npy')
        prior_second_term = np.load(save_path+'prior_second_term.npy')
        g.subplots[2, 2].plot(pivot_range*rad2deg, prior_first_term,
                              label='Prior reference', color='darkorange', linestyle='--')
        lines.append(g.subplots[2, 2].lines[-1])
        options_dict['legend_labels'].append('Prior reference')

        if not one_prior:
            g.subplots[2, 2].plot(pivot_range*rad2deg, prior_second_term,
                                  label='Prior second term', color='darkcyan', linestyle='--')
            lines.append(g.subplots[2, 2].lines[-1])
            options_dict['legend_labels'].append('Prior second term')

    title_list = []
    for i in range(len(labels)):
        error_digits = 2
        error_str_init = '{0:.'+str(error_digits)+'g}'
        error_str = error_str_init.format(np.sqrt(cov_cosmo_deg[i, i]))
        print('error str ', error_str)
        error_len = len(error_str)
        mean_str = '{0:.'+str(error_len-2)+'f}'
        print('mean ', mean_str)
        tot_str = ' = '+mean_str.format(cosmo_estimation[i])+'\\pm'+error_str
        print('tot ', tot_str)
        title_list.append('$'+labels[i] + tot_str+'$')
        g.subplots[i, i].set_title(title_list[-1])
    # '$r = 0.0016^{+0.0016}_{-0.0019}$',
    # '${\\beta_b} = 0.002\\pm 0.070$',
    # '$\\alpha_{93} = 2.331\\pm 0.050$'
    # ], dtype='<U32')

    # lines.append(mlines.Line2D([], [], **g.lines_added[1]))
    #
    legend_args = {}
    legend_args['prop'] = {'size': 12}
    g.legend = g.fig.legend(
        lines, options_dict['legend_labels'], loc=g.settings.legend_loc, **legend_args)
    # IPython.embed()
    plt.savefig(save_path+'cosmo_MCMC.png', bbox_inches='tight', dpi=200)
    plt.savefig(save_path+'cosmo_MCMC.pdf', bbox_inches='tight', dpi=200)
    plt.close()

    '''========================================================================================================================='''
    '''Plot Fisher spectral results'''
    frequencies_plot_nopivot = np.delete(frequencies_plot, pivot_angle_index)
    labels = [r'\alpha_{{{}}}'.format(i) for i in frequencies_plot_nopivot]
    if spectral_flag:
        labels.append(r'\beta_d')
        labels.append(r'\beta_s')
        true_params_ = np.append(np.delete(true_miscal_angles.value,
                                           pivot_angle_index), [1.54, -3])
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
    options_dict['contour_colors'].append('darkred')
    options_dict['contour_ls'].append('-')
    # IPython.embed()
    g = plot_options(options_dict, index_sigma=0, title_limit=title_limit_spectral)

    legend_args = {}
    legend_args['prop'] = {'size': 20}
    lines = g.contours_added
    lines.append(mlines.Line2D([], [], **g.lines_added[0]))
    g.legend = g.fig.legend(
        lines, options_dict['legend_labels'], loc='center right', **legend_args)

    title_list = []
    for i in range(len(labels)):
        error_digits = 2
        error_str_init = '{0:.'+str(error_digits)+'g}'
        error_str = error_str_init.format(np.sqrt(cov_deg[i, i]))
        print('error str ', error_str)
        error_len = len(error_str)
        mean_str = '{0:.'+str(error_len-2)+'f}'
        print('mean ', mean_str)
        tot_str = ' = '+mean_str.format(spectral_estimation[i])+'\\pm'+error_str
        print('tot ', tot_str)
        title_list.append('$'+labels[i] + tot_str+'$')
        g.subplots[i, i].set_title(title_list[-1])

    # title_array = np.array(['$\\alpha_{27} = 1.011\\pm 0.083$',
    #                         '$\\alpha_{39} = 1.668\\pm 0.054$',
    #                         '$\\alpha_{145} = 2.998\\pm 0.049$',
    #                         '$\\alpha_{225} = 3.665\\pm 0.049$',
    #                         '$\\alpha_{280} = 4.331\\pm 0.061$',
    #                         '$\\beta_d = 1.3768\\pm 0.0072$',
    #                         '$\\beta_s = -3.0462\\pm 0.0095$'])
    # for i in range(len(title_array)):
    #     print('a')

    plt.savefig(save_path+'spectral_MCMC.png', bbox_inches='tight', dpi=200)
    plt.savefig(save_path+'spectral_MCMC.pdf', bbox_inches='tight', dpi=200)
    IPython.embed()

    exit()


######################################################
# MAIN CALL
if __name__ == "__main__":
    main()
