import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
import matplotlib.text as mtext
import IPython
from matplotlib import ticker


class LegendTitle(object):
    def __init__(self, text_props=None):
        self.text_props = text_props or {}
        super(LegendTitle, self).__init__()

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        title = mtext.Text(
            x0, y0,  orig_handle, usetex=False, **self.text_props)
        handlebox.add_artist(title)
        return title


def ticks_format(value, index):
    """
    get the value and returns the value as:
            integer: [0,99]
            1 digit float: [0.1, 0.99]
            n*10^m: otherwise
    To have all the number of the same size they are all returned as latex strings
    """
    exp = np.floor(np.log10(value))
    base = value/10**exp
    if exp == 0 or exp == 1 or exp == 2:
        return '${0:d}$'.format(int(value))
    if exp == -1:
        return '${0:.1f}$'.format(value)
    if exp == -2:
        return '${0:.2f}$'.format(value)
    if exp == -3:
        return '${0:.3f}$'.format(value)
    else:
        return '${0:d}\\times10^{{{1:d}}}$'.format(int(base), int(exp))


def main():

    chi2_all = np.load('../results_and_data/potatoe/'+'chi2_all.npy')
    r_all = np.load('../results_and_data/potatoe/'+'r_range_all.npy')
    beta_all = (np.load('../results_and_data/potatoe/'+'beta_range_all.npy') * u.rad).to(u.deg)

    chi2_fg = np.load('../results_and_data/potatoe/'+'chi2_fg28.npy')
    r_fg = np.load('../results_and_data/potatoe/'+'r_range_fg28.npy')
    beta_fg = (np.load('../results_and_data/potatoe/'+'beta_range_fg28.npy') * u.rad).to(u.deg)

    chi2_miscal = np.load('../results_and_data/potatoe/'+'chi2_miscal.npy')
    r_miscal = np.load('../results_and_data/potatoe/'+'r_range_miscal.npy')
    beta_miscal = (np.load('../results_and_data/potatoe/' +
                           'beta_range_miscal.npy') * u.rad).to(u.deg)

    chi2_miscal_flat = np.load('../results_and_data/potatoe/'+'chi2_miscal_flat28.npy')
    r_miscal_flat = np.load('../results_and_data/potatoe/'+'r_range_miscal_flat28.npy')
    beta_miscal_flat = (np.load('../results_and_data/potatoe/' +
                                'beta_range_miscal_flat28.npy') * u.rad).to(u.deg)

    chi2_miscal_descending = np.load('../results_and_data/potatoe/'+'chi2_miscal_descending.npy')
    r_miscal_descending = np.load('../results_and_data/potatoe/'+'r_range_miscal_descending.npy')
    beta_miscal_descending = (np.load('../results_and_data/potatoe/' +
                                      'beta_range_miscal_descending.npy') * u.rad).to(u.deg)

    chi2_fg_nomiscal = np.load('../results_and_data/potatoe/'+'chi2_fg_nomiscal.npy')
    r_fg_nomiscal = np.load('../results_and_data/potatoe/'+'r_range_fg_nomiscal.npy')
    beta_fg_nomiscal = (np.load('../results_and_data/potatoe/' +
                                'beta_range_fg_nomiscal.npy') * u.rad).to(u.deg)

    chi2_fg_nomiscalBB = np.load('../results_and_data/potatoe/'+'chi2_fg_nomiscal_BB.npy')
    r_fg_nomiscalBB = np.load('../results_and_data/potatoe/'+'r_range_fg_nomiscal_BB.npy')
    beta_fg_nomiscalBB = (np.load('../results_and_data/potatoe/' +
                                  'beta_range_fg_nomiscal_BB.npy') * u.rad).to(u.deg)

    r_true = 0.01
    beta_true = (0.35 * u.deg).to(u.rad)

    beta_max = np.max([beta_all, beta_fg, beta_miscal, beta_miscal_flat,
                       beta_miscal_descending, beta_fg_nomiscal, beta_fg_nomiscalBB])
    beta_min = np.min([beta_all, beta_fg, beta_miscal, beta_miscal_flat,
                       beta_miscal_descending, beta_fg_nomiscal, beta_fg_nomiscalBB])
    r_max = np.max([r_all, r_fg, r_miscal, r_miscal_flat, r_miscal_descending,
                    r_fg_nomiscal, r_fg_nomiscalBB])
    r_min = np.min([r_all, r_fg, r_miscal, r_miscal_flat, r_miscal_descending,
                    r_fg_nomiscal, r_fg_nomiscalBB])

    like_all = np.exp((-chi2_all+np.min(chi2_all))/2)
    like_fg = np.exp((-chi2_fg+np.min(chi2_fg))/2)
    like_miscal = np.exp((-chi2_miscal+np.min(chi2_miscal))/2)

    like_miscal_flat = np.exp((-chi2_miscal_flat+np.min(chi2_miscal_flat))/2)
    like_miscal_descending = np.exp((-chi2_miscal_descending+np.min(chi2_miscal_descending))/2)
    like_fg_nomiscal = np.exp((-chi2_fg_nomiscal+np.min(chi2_fg_nomiscal))/2)

    like_fg_nomiscalBB = np.exp((-chi2_fg_nomiscalBB+np.min(chi2_fg_nomiscalBB))/2)

    chi2_all_levels = np.min(chi2_all) + np.array([6.17, 2.3, 0])
    sigma_levels = np.exp((-chi2_all_levels + np.min(chi2_all))/2)

    '''
    import scipy.stats as st
    sigma_list = np.array([0, 1, 2, 3, 4, 5])
    p_norm = st.norm.cdf(0+sigma_list, 0, 1) - st.norm.cdf(0-sigma_list, 0, 1)
    chi2_2dof = st.chi2.ppf(p_norm, 2)[::-1]
    chi2_all_levels_5s = np.min(chi2_all) + chi2_2dof
    sigma_levels_5s = np.exp((-chi2_all_levels_5s + np.min(chi2_all))/2)
    '''
    colors_plasma = plt.cm.plasma([0.3, 0.2])
    colors_plasma[:, -1] = [0.5, 1]
    colors_gold = plt.cm.YlOrBr([0.4, 0.5])
    colors_gold[:, -1] = [0.5, 1]

    fig, ax = plt.subplots(figsize=(25, 14))
    cs_all = ax.contourf(r_all, beta_all.value, like_all.T,
                         levels=sigma_levels, extend='max', cmap='Blues', alpha=1)
    cs_fg = ax.contourf(r_fg, beta_fg.value, like_fg.T, levels=sigma_levels,
                        extend='max', cmap='RdPu', alpha=1)
    cs_fg_nomiscal = ax.contourf(r_fg_nomiscal, beta_fg_nomiscal.value, like_fg_nomiscal.T,
                                 levels=sigma_levels, extend='max', colors=colors_plasma)

    cs_miscal_flat = ax.contourf(r_miscal_flat, beta_miscal_flat.value, like_miscal_flat.T,
                                 levels=sigma_levels, extend='max', colors=colors_gold)

    cs_miscal = ax.contour(r_miscal, beta_miscal.value, like_miscal.T,
                           levels=sigma_levels, extend='max', colors='olive', linewidths=3)  # cmap='Reds', alpha=0.5)

    cs_miscal_descending = ax.contour(r_miscal_descending, beta_miscal_descending.value, like_miscal_descending.T,
                                      levels=sigma_levels, extend='max', colors='red', linewidths=3)  # cmap='Purples', alpha=0.5)

    cs_fg_nomiscalBB = ax.contour(r_fg_nomiscalBB, beta_fg_nomiscalBB.value, like_fg_nomiscalBB.T,
                                  levels=sigma_levels, extend='max')

    ax.hlines(beta_true.to(u.deg).value, r_min, r_max,
              colors='black', linestyles='--', label='input values')
    ax.vlines(r_true, beta_min, beta_max, colors='black', linestyles='--')
    artist_true, label_true = ax.get_legend_handles_labels()

    artists_all, labels_all = cs_all.legend_elements()
    artists_fg, labels_fg = cs_fg.legend_elements()
    artists_miscal, labels_miscal = cs_miscal.legend_elements()
    artists_miscal_flat, labels_miscal_flat = cs_miscal_flat.legend_elements()
    artists_miscal_descending, labels_miscal_descending = cs_miscal_descending.legend_elements()
    artists_fg_nomiscal, labels_fg_nomiscal = cs_fg_nomiscal.legend_elements()

    artists_fg_nomiscalBB, labels_fg_nomiscalBB = cs_fg_nomiscalBB.legend_elements()

    artist_tot = [artist_true[0], artists_all[1], artists_fg[1], artists_fg_nomiscal[1],
                  artists_miscal_flat[1], artists_miscal[1], artists_miscal_descending[1],
                  artists_fg_nomiscalBB[1]]
    # label_tot = ['Input parameters', 'Full pipeline', r'$93 GHz$ map without foreground cleaning and no miscalibration correction $\alpha_{93} = 0.28^{\circ}$',
    #              r'$93GHz$ map without foreground cleaning and $\alpha_{93} = 0^{\circ}$',
    #              r'No miscalibration correction, $\alpha_{i} = 0.28^{\circ} \quad \forall i $ ',
    #              r'No miscalibration correction, $\alpha_{i} = 0.1^{\circ} + i*0.066^{\circ} $',
    #              r'No miscalibration correction, $\alpha_{i} = 0.1^{\circ} + (5-i)*0.066^{\circ} $',
    #              r'$93GHz$ map without foreground cleaning, miscalibration set to $0^{\circ}$. Only BB used in likelihood.']
    label_tot = [r'Input parameters $r = 0.01$, $\beta_b = 0.35^{\circ}$',
                 'Full pipeline',
                 r'$93 GHz$ map NO foreground cleaning, NO miscalibration correction',
                 r'$93GHz$ map NO foreground cleaning',
                 r'No miscalibration correction',
                 r'No miscalibration correction, $\alpha_{i} = 0.1^{\circ} + i*0.066^{\circ} $',
                 r'No miscalibration correction, $\alpha_{i} = 0.1^{\circ} + (5-i)*0.066^{\circ} $',
                 r'$93GHz$ map NO foreground cleaning. Only BB used in likelihood.']
    plt.legend(artist_tot, label_tot,  loc='lower left', fontsize=20)  # handleheight=2,
    plt.xlabel('$r$', fontsize=25)
    plt.ylabel(r'$\beta_b$ in degrees', fontsize=25)
    plt.xscale('log')
    plt.grid(b=True, linestyle=':')

    # plt.ylim(-0.1, beta_all.value.max())
    # plt.title(r'Cosmological likelihood gridding on $r$ and $\beta_b$', fontsize=25)
    plt.xlim(0.0048, r_max)
    plt.ylim(-1, 1)
    subs = [1.0, 2.0, 5.0]
    ax.xaxis.set_major_locator(ticker.LogLocator(subs=subs))
    ax.xaxis.set_minor_locator(ticker.LogLocator(subs=subs))
    ax.xaxis.set_minor_formatter(ticker.FuncFormatter(ticks_format))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(ticks_format))
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.tick_params(axis='both', which='minor', labelsize=8)
    plt.savefig('../results_and_data/potatoe/'+'cosmo_likelihood_gridding_comparisonV2_CMBFRANCE.pdf',
                dpi=200)  # , bbox_inches='tight')
    plt.savefig('../results_and_data/potatoe/'+'cosmo_likelihood_gridding_comparisonV2_CMBFRANCE.svg',
                dpi=200)
    plt.savefig('../results_and_data/potatoe/'+'cosmo_likelihood_gridding_comparisonV2_CMBFRANCE.png',
                dpi=200)
    plt.show()
    IPython.embed()
    exit()


######################################################
# MAIN CALL
if __name__ == "__main__":
    main()
