from config_copy import *
import argparse
# import IPython
from astropy import units as u
import copy
import numpy as np
# from getdist.gaussian_mixtures import GaussianND
from getdist import MCSamples
import matplotlib.pyplot as plt
from full_plots import plot_options, prior_samples
import matplotlib.lines as mlines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="path to save folder")
    args = parser.parse_args()

    save_path = args.folder
    print()
    print(save_path)

    # plt.switch_backend("Qt5Agg")

    rad2deg = (1*u.rad).to(u.deg).value

    '''=========================Spec sample plot=========================='''
    spectral_sample_raw = np.load(save_path+'spec_samples.npy')
    spectral_sample = copy.deepcopy(spectral_sample_raw)
    spectral_sample[:, :freq_number] *= rad2deg

    labels = [r'\alpha_{{{}}}'.format(i) for i in frequencies_plot]
    labels.append(r'\beta_d')
    labels.append(r'\beta_s')
    true_params_ = np.append(true_miscal_angles.value, [1.54, -3])

    markers = {}
    true_params = copy.deepcopy(true_params_)
    true_params[:freq_number] *= rad2deg
    for i in range(freq_number+2):
        markers[labels[i]] = true_params[i]

    boundaries = {}
    for i in range(freq_number):
        boundaries[labels[i]] = [-0.5*np.pi*rad2deg, 0.5*np.pi*rad2deg]
    boundaries[labels[freq_number]] = [1.4, 1.8]
    boundaries[labels[freq_number+1]] = [-3.5, -2.5]

    label_sample = 'Generalised\nSpectral Likelihood'
    distsamples = MCSamples(samples=spectral_sample, names=labels, labels=labels,
                            ranges=boundaries, label=label_sample)

    options_dict = {}
    options_dict['plot_samples'] = [distsamples]
    options_dict['labels'] = labels
    options_dict['markers'] = markers

    options_dict['filled'] = [True]

    options_dict['legend_labels'] = [label_sample]

    options_dict['contour_lws'] = [1.5]
    options_dict['contour_colors'] = ['orange']
    options_dict['contour_ls'] = ['-']
    if prior_flag:
        gaussian_prior_sample, label_gauss = prior_samples(
            (prior_precision*u.rad).to(u.deg).value, prior_indices,
            (angle_prior[:, 0]*u.rad).to(u.deg).value, options_dict, param_number=freq_number+2)
        options_dict['contour_lws'][-1] = 1.5

    g = plot_options(options_dict, index_sigma=0)

    legend_args = {}
    legend_args['prop'] = {'size': 20}
    lines = g.contours_added[:-1]
    if prior_flag:
        lines.append(mlines.Line2D([], [], **g.lines_added[1]))

    g.legend = g.fig.legend(
        lines, options_dict['legend_labels'], loc='center right', **legend_args)

    title_array = []
    for i in range(freq_number+2):
        title_array.append(g.subplots[i, i].title.get_text())

    plt.savefig(save_path+'spectral_MCMC.png', bbox_inches='tight', dpi=200)
    plt.savefig(save_path+'spectral_MCMC.pdf', bbox_inches='tight', dpi=200)
    plt.close()

    '''=======================Cosmo sample plot MIN======================='''
    '''
    cosmo_raw = np.load(save_path + 'cosmo_samples_min.npy')
    cosmo_reshape = cosmo_raw.reshape(cosmo_raw.shape[0]*cosmo_raw.shape[1], 2)
    cosmo_nonan = cosmo_reshape[~np.isnan(cosmo_reshape).any(axis=1)]
    cosmo_sample = copy.deepcopy(cosmo_nonan)
    cosmo_sample[:, 1] *= rad2deg

    labels = [r'r', r'{\beta_b}']
    label_sample = 'Cosmological likelihood'
    markers = {}
    true_params = [r_true, beta_true.to(u.deg).value]
    markers[labels[0]] = true_params[0]
    markers[labels[1]] = true_params[1]

    boundaries = {}
    # boundaries[labels[0]] = [-0.01, 1]
    boundaries[labels[0]] = [-0.0, 1]
    boundaries[labels[1]] = [-0.5*np.pi*rad2deg, 0.5*np.pi*rad2deg]

    distsamples = MCSamples(samples=cosmo_sample, names=labels,
                            labels=labels, ranges=boundaries,
                            label=label_sample)
    options_dict = {}
    options_dict['plot_samples'] = [distsamples]
    options_dict['labels'] = labels
    options_dict['markers'] = markers

    options_dict['filled'] = [True]

    options_dict['legend_labels'] = [label_sample]

    options_dict['contour_lws'] = [1.5]
    options_dict['contour_colors'] = ['orange']
    options_dict['contour_ls'] = ['-']

    g = plot_options(options_dict, index_sigma=0)
    legend_args = {}
    legend_args['prop'] = {'size': 8}
    lines = g.contours_added
    g.legend = g.fig.legend(
        lines, options_dict['legend_labels'], loc=g.settings.legend_loc, **legend_args)

    plt.savefig(save_path+'cosmo_MIN.png', bbox_inches='tight', dpi=200)
    plt.close()
    '''
    '''=======================Cosmo sample plot MCMC======================'''
    cosmo_raw = np.load(save_path + 'cosmo_samples_MCMC.npy')
    cosmo_reshape = cosmo_raw.reshape(cosmo_raw.shape[0]*cosmo_raw.shape[1], 2)
    cosmo_nonan = cosmo_reshape[~np.isnan(cosmo_reshape).any(axis=1)]
    cosmo_sample = copy.deepcopy(cosmo_nonan)
    cosmo_sample[:, 1] *= rad2deg

    labels = [r'r', r'{\beta_b}']
    label_sample = 'Cosmological likelihood'
    markers = {}
    true_params = [r_true, beta_true.to(u.deg).value]
    markers[labels[0]] = true_params[0]
    markers[labels[1]] = true_params[1]

    boundaries = {}
    # boundaries[labels[0]] = [-0.01, 1]
    boundaries[labels[0]] = [-0.004, 1]
    boundaries[labels[1]] = [-0.5*np.pi*rad2deg, 0.5*np.pi*rad2deg]

    distsamples = MCSamples(samples=cosmo_sample, names=labels,
                            labels=labels, ranges=boundaries,
                            label=label_sample)
    options_dict = {}
    options_dict['plot_samples'] = [distsamples]
    options_dict['labels'] = labels
    options_dict['markers'] = markers

    options_dict['filled'] = [True]

    options_dict['legend_labels'] = [label_sample]

    options_dict['contour_lws'] = [1.5]
    options_dict['contour_colors'] = ['orange']
    options_dict['contour_ls'] = ['-']

    g = plot_options(options_dict, index_sigma=0)
    legend_args = {}
    legend_args['prop'] = {'size': 8}
    lines = g.contours_added
    g.legend = g.fig.legend(
        lines, options_dict['legend_labels'], loc=g.settings.legend_loc, **legend_args)

    for i in range(2):
        title_array.append(g.subplots[i, i].title.get_text())
    title_array = np.array(title_array)
    np.save('results_array.npy', title_array)

    plt.savefig(save_path+'cosmo_MCMC.png', bbox_inches='tight', dpi=200)
    plt.savefig(save_path+'cosmo_MCMC.pdf', bbox_inches='tight', dpi=200)
    plt.close()
    exit()


######################################################
# MAIN CALL
if __name__ == "__main__":
    main()
