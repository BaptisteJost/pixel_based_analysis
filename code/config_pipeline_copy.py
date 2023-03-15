import copy
from bjlib.lib_project import cl_rotation
from copy import deepcopy
from fisher_pixel import spectral_first_deriv
from residuals import jac_cosmo, get_Cl_cmbBB
import numpy as np
from astropy import units as u
from datetime import date


'''Paths and machine'''
machine = 'local'
path_NERSC = '/global/homes/j/jost/these/pixel_based_analysis/results_and_data/run02032021/'
path_local = '/home/baptiste/Documents/these/pixel_based_analysis/results_and_data/MCMCrerun_article/'
path_idark = ''
path_BB_local = '/home/baptiste/BBPipe'
path_BB_NERSC = '/global/homes/j/jost/BBPipe'
path_BB_idark = '/home/jost/code/BBPipe'

if machine == 'local':
    pixel_path = '/home/baptiste/Documents/these/pixel_based_analysis/'
    path = path_local
    path_BB = path_BB_local
elif machine == 'NERSC':
    pixel_path = '/global/u2/j/jost/these/pixel_based_analysis/'
    path = path_NERSC
    path_BB = path_BB_NERSC
elif machine == 'idark':
    pixel_path = '/home/jost/code/pixel_based_analysis/'
    path_BB = path_BB_idark
else:
    print('ERROR : Machine not recognized')
    exit()

save_path_ = pixel_path + 'results_and_data/full_pipeline/debug_LB/' + \
    date.today().strftime('%Y%m%d') + '_constrained_'


'''Cosmology params'''
r_true = 0.0
beta_true = (0.0 * u.deg).to(u.rad)
A_lens_true = 1

test1freq = False

'''Instrument and noise'''
INSTRU = 'LiteBIRD'
if INSTRU == 'SAT':
    freq_number = 6
    fsky = 0.1
    lmin = 30
    lmax = 300
    # lmax = 1000
    nside = 512
    add_noise = 1
    sensitiviy_mode = 1
    one_over_f_mode = 1
    one_over_ell = True
    beam_correction = True
    frequencies_plot = np.array([27,  39,  93, 145, 225, 280])
    overwrite_freq = None
    pivot_angle_index = 2

    if test1freq:
        frequencies_plot = np.array([93])  # test1freq
        overwrite_freq = np.array([93])  # test1freq
        freq_number = 1  # test1freq

elif INSTRU == 'Planck':
    freq_number = 7
    fsky = 0.6
    lmin = 51
    lmax = 1500
    nside = 2048

elif INSTRU == 'LiteBIRD':
    freq_number = 22
    # fsky = 0.49
    lmin = 2
    lmax = 125
    # lmax = 500
    nside = 64
    add_noise = 1
    sensitiviy_mode = None
    one_over_f_mode = None
    one_over_ell = True
    beam_correction = True
    frequencies_plot = np.array([40.,  50.,  60.,  68.,  68.,  78.,  78.,  89.,
                                 89., 100., 119., 140., 100., 119., 140., 166.,
                                 195., 195., 235., 280., 337., 402.])
    overwrite_freq = None
    pivot_angle_index = 9

else:
    print('ERROR : instrument ', INSTRU, ' not supported yet.')
    exit()
freq_by_instru = [1]*freq_number


'''Spectral likelihood minimisation params'''
# if test1freq:
#     spectral_flag = 0  # test1freq
params = ['miscal']*freq_number
# miscal_bounds = ((-np.pi/8, np.pi/8),)*freq_number
#
# if spectral_flag:
params.append('spectral')
params.append('spectral')
#     spectral_bounds = ((0.5, 2.5), (-5, -1))
#     bounds = miscal_bounds + spectral_bounds
# else:
#     bounds = miscal_bounds
# method_spectral = 'L-BFGS-B'
# jac_spectal = spectral_first_deriv
#
# initmodel_miscal = np.array([0]*freq_number)*u.rad
# angle_array_start = np.random.uniform(np.array(bounds)[:, 0],
#                                       np.array(bounds)[:, 1])
'''Prior information'''
prior_gridding = False

# input_angles = copy.deepcopy(true_miscal_angles.value)  # + \

if not prior_gridding:
    prior_flag = True
    prior_indices = []
    if prior_flag:
        # pivot_angle_index = 2
        one_prior = False
        if one_prior:
            prior_indices = [pivot_angle_index, pivot_angle_index+1]
        else:
            prior_indices = [0, freq_number]

        if test1freq:
            prior_indices = [0, 1]  # test1freq
    # prior_precision = (5*u.deg).to(u.rad).value
    # prior_precision = (0.1*u.deg).to(u.rad).value
    # prior_precision = (1.00000000e+00*u.deg).to(u.rad).value
    # prior_precision = (0.001*u.deg).to(u.rad).value
    prior_precision = np.array([49.8, 39.8, 16.1, 1.09, 35.9, 8.6, 13.0, 5.4, 29.4,
                                3.8, 2.1, 1.8, 2.6, 1.2, 1.5, 1.1, 1.8, 3.9, 4.1, 6.8, 17.1, 80.0])*u.arcmin.to(u.rad)
    angle_prior = []
    random_bias = False
    if random_bias:
        np.random.seed(1)
        bias = np.random.normal(0, prior_precision, freq_number)
        print('bias =', bias)
    if prior_flag:
        for d in range(freq_number):
            angle_prior.append([0., prior_precision[d], int(d)])
        angle_prior = np.array(angle_prior[prior_indices[0]: prior_indices[-1]])

    prior_matrix = np.zeros([len(params), len(params)])
    if prior_flag:
        for i in range(prior_indices[0], prior_indices[-1]):
            prior_matrix[i, i] += 1/(prior_precision[i]**2)

'''Fiducial spectra'''
ps_planck = deepcopy(get_Cl_cmbBB(Alens=A_lens_true, r=r_true, path_BB=path_BB))
spectra_true = cl_rotation(ps_planck.T, beta_true).T
Cl_fid = {}
# Cl_fid['BB'] = get_Cl_cmbBB(Alens=A_lens_true, r=r_true, path_BB=path_BB)[2][lmin:lmax+1]
# Cl_fid['BuBu'] = get_Cl_cmbBB(Alens=0.0, r=1.0, path_BB=path_BB)[2][lmin:lmax+1]
# Cl_fid['BlBl'] = get_Cl_cmbBB(Alens=1.0, r=0.0, path_BB=path_BB)[2][lmin:lmax+1]
# Cl_fid['EE'] = ps_planck[1, lmin:lmax+1]
Cl_fid['BB'] = get_Cl_cmbBB(Alens=A_lens_true, r=r_true, path_BB=path_BB)[2][:4000]
Cl_fid['BuBu'] = get_Cl_cmbBB(Alens=0.0, r=1.0, path_BB=path_BB)[2][:4000]
Cl_fid['BlBl'] = get_Cl_cmbBB(Alens=1.0, r=0.0, path_BB=path_BB)[2][:4000]
Cl_fid['EE'] = ps_planck[1, :4000]

# '''Cosmo likelihood minimisation params'''
# bounds_cosmo = ((-0.01, 5), (-np.pi/8, np.pi/8))
# cosmo_array_start = np.random.uniform(np.array(bounds_cosmo)[:, 0],
#                                       np.array(bounds_cosmo)[:, 1])
# # cosmo_array_start = [0.03, 0.04]
# method_cosmo = 'L-BFGS-B'
# jac_cosmo_min = jac_cosmo

'''spectral MCMC options'''
spectral_MCMC_flag = 0
nsteps_spectral = 13000
# nsteps_spectral = 2000  # DEBUG LB
discard_spectral = 5000
# discard_spectral = 500  # DEBUG LB
spectral_walker_per_dim = 2
# spectral_dim = freq_number + birefringence_flag + 2*spectral_flag

'''Cosmo MCMC options'''
cosmo_MCMC_flag = 0
nsteps_cosmo = 30000
discard_cosmo = 15000
cosmo_walker_per_dim = 2
cosmo_dim = 2
