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
    path = '/home/baptiste/Documents/these/pixel_based_analysis/results_and_data/full_pipeline/test_total_likelihood/'

    rad2deg = (1*u.rad).to(u.deg).value

    plt.switch_backend("Qt5Agg")

    tot_sample = np.load(path+'samples_5K_newP.npy')
    tot_sample[:, :6] *= rad2deg
    tot_sample[:, -1] *= rad2deg

    label_sample = 'total likelihood'
    labels = [r'\alpha_{{{}}}'.format(i) for i in frequencies_plot]
    labels.append(r'\beta_d')
    labels.append(r'\beta_s')
    labels.append(r'r')
    labels.append(r'{\beta_b}')

    miscal = 1  # 1deg
    total_params = np.array([miscal, miscal, miscal, miscal, miscal, miscal,
                             1.54, -3., r_true, beta_true.to(u.deg).value])
    markers = {}
    for i in range(len(labels)):
        markers[labels[i]] = total_params[i]
    distsamples = MCSamples(samples=tot_sample, names=labels, labels=labels,
                            label=label_sample)
    options_dict = {}
    options_dict['labels'] = labels
    options_dict['markers'] = markers
    options_dict['plot_samples'] = [distsamples]
    options_dict['filled'] = [True]

    options_dict['legend_labels'] = ['total likelihood']
    options_dict['contour_lws'] = [1.5]
    options_dict['contour_colors'] = ['orange']
    options_dict['contour_ls'] = ['-']
    title_limit_spectral = 1
    g = plot_options(options_dict, index_sigma=0, title_limit=title_limit_spectral)
    legend_args = {}
    legend_args['prop'] = {'size': 20}
    lines = g.contours_added[:-1]
    lines.append(mlines.Line2D([], [], **g.lines_added[0]))
    g.legend = g.fig.legend(
        lines, options_dict['legend_labels'], loc=g.settings.legend_loc, **legend_args)

    plt.savefig(path+'tot_MCMC_5K_newP.png', bbox_inches='tight', dpi=200)
    plt.savefig(path+'tot_MCMC_5K_newP.pdf', bbox_inches='tight', dpi=200)
    exit()


######################################################
# MAIN CALL
if __name__ == "__main__":
    main()
