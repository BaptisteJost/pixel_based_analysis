# from config import *
import argparse
import IPython
import matplotlib.lines as mlines
from astropy import units as u
import copy
import numpy as np
from getdist.gaussian_mixtures import GaussianND
from getdist import plots, MCSamples
import matplotlib.pyplot as plt


def prior_samples(prior_precision, prior_indices, distrib_centers, options_dict):
    prior_num = prior_indices[-1]-prior_indices[0]
    prior_cov = np.zeros([prior_num, prior_num])
    for i in range(prior_num):
        prior_cov[i, i] = prior_precision**2

    prior_samples_num = 100000
    samps_gauss_prior = np.random.multivariate_normal(
        distrib_centers, prior_cov, size=prior_samples_num)

    sample_prior_none = np.zeros([8, prior_samples_num])
    counter = 0
    for i in range(prior_indices[0], prior_indices[-1]):
        sample_prior_none[i] = samps_gauss_prior.T[counter]
        counter += 1
    sample_prior_none = sample_prior_none.T
    prior_none = MCSamples(samples=sample_prior_none,
                           names=options_dict['labels'], labels=options_dict['labels'])

    options_dict['plot_samples'].append(prior_none)
    options_dict['filled'].append(False)
    options_dict['contour_colors'].append('purple')
    label_gauss = 'Gaussian priors precision : {} deg'.format(prior_precision)
    options_dict['legend_labels'].append(label_gauss)
    options_dict['contour_ls'].append('--')
    options_dict['contour_lws'].append(4)
    return prior_none, label_gauss


def plot_options(options_dict, index_sigma=-1, title_limit=1):
    g = plots.get_subplot_plotter()
    g.settings.legend_loc = 'upper right'
    g.settings.fontsize = 15
    g.settings.legend_fontsize = 15
    g.settings.axes_fontsize = 15
    g.settings.axes_labelsize = 15
    g.settings.title_limit_fontsize = 15
    g.settings.linewidth_contour = 2
    g.settings.line_labels = False
    g.triangle_plot(options_dict['plot_samples'], filled=options_dict['filled'],
                    contour_colors=options_dict['contour_colors'],
                    markers=options_dict['markers'],
                    legend_labels=options_dict['legend_labels'],
                    contour_ls=options_dict['contour_ls'],
                    contour_lws=options_dict['contour_lws'], title_limit=title_limit)
    '''
    sigma_number = 1
    distsamplesFSN = options_dict['plot_samples'][index_sigma]
    marge = distsamplesFSN.getMargeStats()
    params = marge.list()
    for i in range(distsamplesFSN.n):
        param = marge.parWithName(options_dict['labels'][i])
        lim = param.limits[sigma_number - 1]
        mean = param.mean
        upper = lim.upper - mean
        lower = lim.lower - mean
        mean_str = '{:.1e}'.format(mean)
        upper_str = '{:.1e}'.format(upper)
        lower_str = '{:.1e}'.format(lower)
        title_str = params[i] + ' = ' + mean_str
        title_str += '^{+' + upper_str + '}_{' + lower_str + '}'
        IPython.embed()
        g.subplots[i, i].set_title(r'$'+title_str+'$', fontsize=15)
        g.subplots[i, i].axvline(lim.lower, color='gray', ls='--', lw=2)
        g.subplots[i, i].axvline(lim.upper, color='gray', ls='--', lw=2)
    '''
    return g


def get_sample_perwalker(sample, nwalker, burn=None):
    size = int(len(sample)/nwalker)
    sample_pw = []
    for i in range(nwalker):
        sample_pw.append(sample[i*size:(i+1)*size])
    sample_pw = np.array(sample_pw)
    if burn is not None:
        sample_burn = []
        for i in range(nwalker):
            sample_burn.append(sample_pw[i, burn:])
        sample_burn = np.array(sample_burn)
        burn_flat = sample_burn.reshape(nwalker*(size-burn), 2)
        return sample_burn, burn_flat
    return sample_pw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder_end", help="end of the folder name")
    args = parser.parse_args()

    save_path__ = save_path_.replace('23', '22')
    save_path = save_path__ + args.folder_end + '/'
    save_path = '/home/baptiste/Documents/these/pixel_based_analysis/results_and_data/full_pipeline/20220405_constrained35/'
    print()
    print(save_path)

    rad2deg = (1*u.rad).to(u.deg).value

    plt.switch_backend("Qt5Agg")
    import IPython
    IPython.embed()
    '''triangle plot spectral MCMC'''
    if spectral_MCMC_flag:
        labels = [r'\alpha_{{{}}}'.format(i) for i in frequencies_plot]
        if spectral_flag:
            labels.append(r'\beta_d')
            labels.append(r'\beta_s')
            true_params_ = np.append(true_miscal_angles.value, [1.54, -3])

        markers = {}
        true_params = copy.deepcopy(true_params_)
        true_params[:6] *= rad2deg
        for i in range(8):
            markers[labels[i]] = true_params[i]

        boundaries = {}
        for i in range(6):
            boundaries[labels[i]] = [-0.5*np.pi*rad2deg, 0.5*np.pi*rad2deg]
        boundaries[labels[6]] = [1.4, 1.8]
        boundaries[labels[7]] = [-3.5, -2.5]

        spectral_sample = np.load(save_path+'spectral_samples.npy')
        spectral_fisher = np.load(save_path+'fisher_spectral.npy')
        spectral_estimation = np.load(save_path+'spectral_results.npy')

        cov_spectral = np.linalg.inv(spectral_fisher)

        cov_deg = copy.deepcopy(cov_spectral)
        cov_deg[:6] *= rad2deg
        cov_deg[:, :6] *= rad2deg

        spectral_sample[:, :6] *= rad2deg
        spectral_estimation[:6] *= rad2deg

        label_sample = '-2logL, prior precision = {}'.format((prior_precision*u.rad).to(u.deg))

        distsamples = MCSamples(samples=spectral_sample, names=labels, labels=labels,
                                ranges=boundaries, label=label_sample)
        fisher = GaussianND(spectral_estimation, cov_deg, names=labels, labels=labels)
        distsamples.smooth_scale_2D = 1
        fisher.smooth_scale_2D = -1

        options_dict = {}
        options_dict['plot_samples'] = [distsamples, fisher]
        options_dict['labels'] = labels
        options_dict['markers'] = markers

        options_dict['filled'] = [True, False]

        options_dict['legend_labels'] = [label_sample,
                                         'Fisher estimation, prior precision = {}'.format((prior_precision*u.rad).to(u.deg))]
        options_dict['contour_lws'] = [1.5, 2]
        options_dict['contour_colors'] = ['orange', 'maroon']
        options_dict['contour_ls'] = ['-', '-']
        if prior_flag:
            gaussian_prior_sample, label_gauss = prior_samples(
                (prior_precision*u.rad).to(u.deg).value, prior_indices,
                (angle_prior[:, 0]*u.rad).to(u.deg).value, options_dict)

        g = plot_options(options_dict, index_sigma=0)

        legend_args = {}
        legend_args['prop'] = {'size': 20}
        lines = g.contours_added[:-1]
        if prior_flag:
            lines.append(mlines.Line2D([], [], **g.lines_added[2]))
        else:
            lines.append(mlines.Line2D([], [], **g.lines_added[1]))
        import IPython
        IPython.embed()
        g.legend = g.fig.legend(
            lines, options_dict['legend_labels'], loc=g.settings.legend_loc, **legend_args)
        plt.savefig(save_path+'spectral_MCMC.png', bbox_inches='tight', dpi=200)
        plt.savefig(save_path+'spectral_MCMC.pdf', bbox_inches='tight', dpi=200)

    '''triangle plot cosmo MCMC'''

    if cosmo_MCMC_flag:
        IPython.embed()
        cosmo_sample = np.load(save_path+'cosmo_samples.npy')
        cosmo_fisher = np.load(save_path+'fisher_cosmo.npy')
        cosmo_estimation = np.load(save_path+'cosmo_results.npy')
        cosmo_list = np.load(save_path+'cosmo_list.npy')
        flat_list, cosmo_list_burn = get_sample_perwalker(cosmo_list, 16, 4000)
        labels = [r'r', r'{\beta_b}']

        markers = {}
        true_params = [r_true, beta_true.to(u.deg).value]
        markers[labels[0]] = true_params[0]
        markers[labels[1]] = true_params[1]

        boundaries = {}
        boundaries[labels[0]] = [-0.01, 1]
        boundaries[labels[1]] = [-0.5*np.pi*rad2deg, 0.5*np.pi*rad2deg]

        cov_cosmo = np.linalg.inv(cosmo_fisher)

        cov_cosmo_deg = copy.deepcopy(cov_cosmo)
        cov_cosmo_deg[1:] *= rad2deg
        cov_cosmo_deg[:, 1:] *= rad2deg

        cosmo_sample[:, 1:] *= rad2deg
        cosmo_estimation[1:] *= rad2deg

        cosmo_list_burn[:, 1:] *= rad2deg

        label_sample = 'Cosmological likelihood'

        distsamples = MCSamples(samples=cosmo_sample, names=labels,
                                labels=labels, ranges=boundaries,
                                label=label_sample)
        fisher = GaussianND(cosmo_estimation, cov_cosmo_deg, names=labels,
                            labels=labels)
        distsamples.smooth_scale_2D = 1
        fisher.smooth_scale_2D = -1

        label_list = 'Cosmological double MC'
        distlist = MCSamples(samples=cosmo_list_burn, names=labels,
                             labels=labels, ranges=boundaries,
                             label=label_list)
        distlist.smooth_scale_2D = -1
        cov_cosmo_prior = copy.deepcopy(cov_cosmo_deg)
        cov_cosmo_prior[1, 1] = (prior_precision*u.rad).to(u.deg).value ** 2
        fisher_prior = GaussianND(cosmo_estimation, cov_cosmo_prior, names=labels,
                                  labels=labels)

        options_dict = {}
        # options_dict['plot_samples'] = [distsamples, fisher]
        # options_dict['plot_samples'] = [distsamples, fisher, distlist, fisher_prior]
        options_dict['plot_samples'] = [distlist, fisher, distsamples, fisher_prior]
        options_dict['labels'] = labels
        options_dict['markers'] = markers

        options_dict['filled'] = [True, False, False, False]

        options_dict['legend_labels'] = [label_sample,
                                         'Fisher estimation', label_list, 'Fisher prior']
        options_dict['contour_lws'] = [1.5, 2, 2, 2]
        options_dict['contour_colors'] = ['orange', 'maroon', 'green', 'purple']
        options_dict['contour_ls'] = ['-', '-', '--', ':']

        g = plot_options(options_dict, index_sigma=0)

        legend_args = {}
        legend_args['prop'] = {'size': 8}
        lines = g.contours_added[:-1]
        lines.append(mlines.Line2D([], [], **g.lines_added[1]))

        g.legend = g.fig.legend(
            lines, options_dict['legend_labels'], loc=g.settings.legend_loc, **legend_args)
        # IPython.embed()
        plt.savefig(save_path+'cosmo_MCMC.png', bbox_inches='tight', dpi=200)
        plt.savefig(save_path+'cosmo_MCMC.pdf', bbox_inches='tight', dpi=200)
    exit()


######################################################
# MAIN CALL
if __name__ == "__main__":
    main()
