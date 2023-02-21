from astropy import units as u
import numpy as np
from residuals import get_ys_alms
from pixel_based_angle_estimation import data_and_model_quick, get_model
import copy
import time
from emcee import EnsembleSampler
# from config_copy import *
from residuals import get_SFN
from os import path as p
import tracemalloc
from total_likelihood import spectral_sampling, from_spectra_to_cosmo
import argparse
from config_pipeline_copy import *
import healpy as hp
import IPython
import bjlib.lib_project as lib
import pymaster as nmt
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import os
from mpi4py import MPI


def get_ddt(data, mask):
    sim_freq_maps = data
    ddt_full = np.einsum('ik...,...kj->ijk', sim_freq_maps, sim_freq_maps.T)
    mask[(mask != 0) * (mask != 1)] = 0
    ddt_full *= mask
    # n_obspix = np.sum(mask == 1)
    ddt = np.sum(ddt_full, axis=-1)

    return ddt


def binning_definition(nside, lmin=2, lmax=200, fsky=1.0):
    b = nmt.NmtBin(nside, nlb=int(1.5/fsky))
    return b


def get_field(mp_q, mp_u, mask_apo, beam, purify_e=True, purify_b=True, n_iter=3):
    f2y = nmt.NmtField(mask_apo, [mp_q, mp_u], purify_e=purify_e,
                       purify_b=purify_b, beam=beam, n_iter=n_iter)
    return f2y


def compute_master(f_a, f_b, wsp):
    cl_coupled = nmt.compute_coupled_cell(f_a, f_b)
    cl_decoupled = wsp.decouple_cell(cl_coupled)
    return cl_decoupled


def cosmo_like_data(cosmo_params, Cl_noise_matrix, Cl_data, binning_def, fsky, minimisation=False, ell_indices=None):
    r = cosmo_params[0]
    beta = cosmo_params[1]*u.rad

    if not minimisation:
        if r > 5 or r < -1e-5:  # 0.01:
            # print('bla1')
            return -np.inf
        elif beta.value < -np.pi/2 or beta.value > np.pi/2:
            print('bla2')
            return -np.inf

    Cl_data_rot = Cl_data

    Cl_cmb_model = np.zeros([4, Cl_fid['EE'].shape[0]])
    Cl_cmb_model[1] = copy.deepcopy(Cl_fid['EE'])
    Cl_cmb_model[2] = copy.deepcopy(Cl_fid['BlBl'])*1 + copy.deepcopy(Cl_fid['BuBu']) * r

    Cl_cmb_rot_ = lib.cl_rotation(Cl_cmb_model.T, beta).T

    if ell_indices is not None:
        Cl_cmb_rot = binning_def.bin_cell(Cl_cmb_rot_[:, :3*nside])[..., ell_indices]
        # print('Cl_cmb_rot shape =', Cl_cmb_rot.shape)
        ell_eff = binning_def.get_effective_ells()[ell_indices]

    else:
        Cl_cmb_rot = binning_def.bin_cell(Cl_cmb_rot_[:, :3*nside])
        ell_eff = binning_def.get_effective_ells()

    delta_ell = ell_eff[1] - ell_eff[0]
    # IPython.embed()
    # Cl_cmb_rot_EE = binning_def.bin_cell(np.array([Cl_cmb_rot_[1, :3*nside]]))
    # Cl_cmb_rot_BB = binning_def.bin_cell(np.array([Cl_cmb_rot_[2, :3*nside]]))
    # Cl_cmb_rot_EB = binning_def.bin_cell(np.array([Cl_cmb_rot_[4, :3*nside]]))

    Cl_cmb_rot_matrix = np.zeros([2, 2, Cl_cmb_rot.shape[-1]])
    Cl_cmb_rot_matrix[0, 0] = copy.deepcopy(Cl_cmb_rot[1])
    Cl_cmb_rot_matrix[1, 1] = copy.deepcopy(Cl_cmb_rot[2])
    Cl_cmb_rot_matrix[1, 0] = copy.deepcopy(Cl_cmb_rot[4])
    Cl_cmb_rot_matrix[0, 1] = copy.deepcopy(Cl_cmb_rot[4])
    Cl_model_total = Cl_cmb_rot_matrix + Cl_noise_matrix

    inv_model = np.linalg.inv(Cl_model_total.T).T

    dof = (2 * ell_eff + 1) * fsky * delta_ell
    dof_over_Cl = dof * inv_model

    first_term_ell = np.einsum('ijl,jkl->ikl', dof_over_Cl, Cl_data_rot)

    first_term = np.sum(np.trace(first_term_ell))

    logdetC = np.sum(dof*np.log(np.abs(np.linalg.det(Cl_model_total.T))))
    likelihood_cosmo = first_term + logdetC
    if minimisation:
        return likelihood_cosmo/2
    else:
        return -likelihood_cosmo/2


def jac_cosmo_like_data(cosmo_params, Cl_noise_matrix, Cl_data, binning_def, fsky, minimisation=False, ell_indices=None):
    print('DOESN\'T WORK')
    r = cosmo_params[0]
    beta = cosmo_params[1]*u.rad

    Cl_data_rot = Cl_data

    Cl_cmb_model = np.zeros([4, Cl_fid['EE'].shape[0]])
    Cl_cmb_model[1] = copy.deepcopy(Cl_fid['EE'])
    Cl_cmb_model[2] = copy.deepcopy(Cl_fid['BlBl'])*1 + copy.deepcopy(Cl_fid['BuBu']) * r

    Cl_cmb_rot_ = lib.cl_rotation(Cl_cmb_model.T, beta).T

    Cl_cmb_dr = np.zeros([4, Cl_fid['EE'].shape[0]])
    Cl_cmb_dr[1] = copy.deepcopy(Cl_fid['EE'])*0
    Cl_cmb_dr[2] = copy.deepcopy(Cl_fid['BuBu'])
    dCldr_spectra = lib.cl_rotation(Cl_cmb_dr.T, beta).T

    if ell_indices is not None:
        Cl_cmb_rot = binning_def.bin_cell(Cl_cmb_rot_[:, :3*nside])[..., ell_indices]
        # print('Cl_cmb_rot shape =', Cl_cmb_rot.shape)
        ell_eff = binning_def.get_effective_ells()[ell_indices]
        dCldr_spectra = binning_def.bin_cell(dCldr_spectra[:, :3*nside])[..., ell_indices]
        Cl_cmb_model = binning_def.bin_cell(Cl_cmb_model[:, :3*nside])[..., ell_indices]

    else:
        Cl_cmb_rot = binning_def.bin_cell(Cl_cmb_rot_[:, :3*nside])
        ell_eff = binning_def.get_effective_ells()

    dCldr = np.zeros([2, 2, dCldr_spectra.shape[-1]])
    dCldr[0, 0] = dCldr_spectra[1]
    dCldr[1, 1] = dCldr_spectra[2]
    dCldr[1, 0] = dCldr_spectra[4]
    dCldr[0, 1] = dCldr_spectra[4]

    rotation_matrix = np.array([[np.cos(2*beta), np.sin(2*beta)],
                                [-np.sin(2*beta), np.cos(2*beta)]])
    inv_rotation_matrix = np.array([[np.cos(2*beta), -np.sin(2*beta)],
                                    [np.sin(2*beta), np.cos(2*beta)]])
    deriv_rotation_matrix = 2 * np.array([[-np.sin(2*beta), np.cos(2*beta)],
                                          [-np.cos(2*beta), -np.sin(2*beta)]])
    Cl_cmb_model_matrix = np.zeros([2, 2, Cl_cmb_model.shape[-1]])
    Cl_cmb_model_matrix[0, 0] = Cl_cmb_model[1]
    Cl_cmb_model_matrix[1, 1] = Cl_cmb_model[2]

    # dCldbeta_check = deriv_rotation_matrix.dot(Cl_cmb_model_matrix[..., 50]).dot(inv_rotation_matrix) - rotation_matrix.dot(
    #     Cl_cmb_model_matrix[..., 50]).dot(inv_rotation_matrix).dot(deriv_rotation_matrix).dot(inv_rotation_matrix)

    dCldbeta = np.einsum('ij,jkl,km->iml', deriv_rotation_matrix, Cl_cmb_model_matrix, inv_rotation_matrix) -\
        np.einsum('ij,jkl,km,mn,no->iol', rotation_matrix, Cl_cmb_model_matrix,
                  inv_rotation_matrix, deriv_rotation_matrix, inv_rotation_matrix)

    delta_ell = ell_eff[1] - ell_eff[0]

    Cl_cmb_rot_matrix = np.zeros([2, 2, Cl_cmb_rot.shape[-1]])
    Cl_cmb_rot_matrix[0, 0] = copy.deepcopy(Cl_cmb_rot[1])
    Cl_cmb_rot_matrix[1, 1] = copy.deepcopy(Cl_cmb_rot[2])
    Cl_cmb_rot_matrix[1, 0] = copy.deepcopy(Cl_cmb_rot[4])
    Cl_cmb_rot_matrix[0, 1] = copy.deepcopy(Cl_cmb_rot[4])
    Cl_model_total = Cl_cmb_rot_matrix + Cl_noise_matrix

    inv_model = np.linalg.inv(Cl_model_total.T).T

    dof = (2 * ell_eff + 1) * fsky * delta_ell
    dof_over_Cl = dof * inv_model

    first_term_ell = np.einsum('ijl,jkl->ikl', dof_over_Cl, Cl_data_rot)
    first_term = np.sum(np.trace(first_term_ell))
    logdetC = np.sum(dof*np.log(np.abs(np.linalg.det(Cl_model_total.T))))
    likelihood_cosmo = first_term - logdetC

    first_term_ell_dr = dof*np.einsum('ijl,jkl,kml,mnl->inl', -
                                      inv_model, dCldr, inv_model, Cl_data_rot)
    first_term_dr = np.sum(np.trace(first_term_ell_dr))
    second_term_dr = np.sum(np.trace(dof*np.einsum('ijl,jkl->ikl', inv_model, dCldr)))
    jac_dr = first_term - second_term_dr

    first_term_ell_dbeta = dof*np.einsum('ijl,jkl,kml,mnl->inl', -
                                         inv_model, dCldbeta, inv_model, Cl_data_rot)
    first_term_dbeta = np.sum(np.trace(first_term_ell_dr))
    second_term_dbeta = np.sum(np.trace(dof*np.einsum('ijl,jkl->ikl', inv_model, dCldbeta)))
    jac_beta = first_term_dbeta + second_term_dbeta

    return np.array([jac_dr/2, jac_beta/2])


def import_and_smooth_data(instrument, rank, common_beam=None, phase=1, path=None, test_nobeam=False):
    # rank = 0
    data = []
    arcmin2rad = np.pi/(180.0*60.0)
    f = 0
    for freq_tag in instrument.keys():
        print('frequency = ', freq_tag)
        if phase == 1:
            tot_map = hp.read_map('/lustre/work/jost/simulations/LB_phase1/comb/' +
                                  str(rank).zfill(4)+'/'+freq_tag+'_comb_d0s0_white_noise_CMB.fits', field=(0, 1, 2))
        elif phase == 2:
            tot_map = hp.read_map('/global/cfs/cdirs/litebird/simulations/maps/birefringence_project_paper/Phase2/comb/' +
                                  str(rank).zfill(4)+'/'+freq_tag+'_comb_d1s1_white_noise_CMB.fits', field=(0, 1, 2))
        elif phase == 3:
            tot_map = hp.read_map('/global/cfs/cdirs/litebird/simulations/maps/birefringence_project_paper/Phase3_updated_seed_v2/comb/' +
                                  str(rank).zfill(4)+'/'+freq_tag+'_comb_d1s1_white_noise_CMB_polangle.fits', field=(0, 1, 2))
        elif phase is None:
            print('importing mock data, frequency channel #', f)
            tot_map_ = np.load(path)[2*f:2*f+2]
            # IPython.embed()
            tot_map = np.array([np.zeros(tot_map_.shape[1]), tot_map_[0], tot_map_[1]])
            f += 1
        elif phase == 'test' and machine == 'local':
            tot_map = hp.read_map('/home/baptiste/Downloads/LBsim/0000/'
                                  + freq_tag+'_comb_d0s0_white_noise_CMB.fits', field=(0, 1, 2))
        elif phase == 'test' and machine == 'idark':
            tot_map = hp.read_map('/lustre/work/jost/simulations/LB_phase1/comb/0000/'
                                  + freq_tag+'_comb_d0s0_white_noise_CMB.fits', field=(0, 1, 2))

        if not test_nobeam:
            Bl_gauss_fwhm = hp.gauss_beam(
                instrument[freq_tag]['beam']*arcmin2rad, lmax=3*nside, pol=True)[:, 1]
        else:
            print('TEST NO BEAM')
            Bl_gauss_fwhm = np.ones(3*nside+1)

        if common_beam is not None:
            print('   common_beam!=0.0   ')
            Bl_gauss_common = hp.gauss_beam(common_beam*arcmin2rad, lmax=3*nside, pol=True)[:, 1]
        else:
            Bl_gauss_common = np.ones(Bl_gauss_fwhm.shape[0])
        alms = hp.map2alm(tot_map, lmax=3*nside)
        # IPython.embed()
        alms_beamed = []
        for alm_ in alms:
            # hp.almxfl(alm_, Bl_gauss_common/Bl_gauss_fwhm, inplace=True)
            alms_beamed.append(hp.almxfl(alm_, Bl_gauss_common/Bl_gauss_fwhm, inplace=False))
        # tot_map_ = hp.alm2map(alms, nside)
        tot_map = hp.alm2map(alms_beamed, nside)
        # tot_map = tot_map_
        # else:
        #     tot_map = hp.ud_grade(tot_map, nside)

        data.append(tot_map[1])
        data.append(tot_map[2])

    data = np.array(data)
    return data


def main():

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank_mpi = comm.rank  # rank=0
    # rank = 0
    print('MPI size = ', size)
    print('MPI rank = ', rank_mpi)
    # phase = None
    phase = 1
    for map_iter in range(3):
        rank = 3*rank_mpi + map_iter
        print('================================================')
        print("MAP NUMBER = ", rank)
        start = time.time()
        if machine == 'local' or machine == 'NERSC':
            output_dir = pixel_path + 'results_and_data/pipeline_data/phase' + \
                str(phase)+'/'+str(rank).zfill(4) + '/'
            output_dir = pixel_path+'results_and_data/pipeline_data/test_mock/' + \
                str(rank).zfill(4) + '_fullsky_nobeam/'
        elif machine == 'idark':
            output_dir = '/home/jost/results/LB_phase1_fullsky_withbeam/' + str(rank).zfill(4) + '/'
        print('rank=', rank)
        # print('OUTPUT DIR FIXED FOR LOCAL DEBUG!!!!')
        # output_dir = '/home/baptiste/Documents/these/pixel_based_analysis/results_and_data/pipeline_data/debug_beam/'
        # output_dir = '/home/baptiste/Documents/these/pixel_based_analysis/results_and_data/pipeline_data/debug_beam/'
        # output_dir = '/home/baptiste/Documents/these/pixel_based_analysis/results_and_data/pipeline_data/debug_beam_LBsim/'
        # output_dir = '/home/jost/results/debug_beam_test/'
        print('output_dir = ', output_dir)

        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        # autput_dir = '/project/projectdirs/litebird/results/birefringence_project_paper/compsep_maps_fgbuster/Phase4/'+str(rank).zfill(4)

        # instrument = np.load(
        #     '/global/cfs/cdirs/litebird/simulations/maps/PTEP_20201014_d0s0/instrument_LB_IMOv1.npy', allow_pickle=True).item()
        # instrument = np.load('data/instrument_LB_IMOv1.npy', allow_pickle=True).item()
        instrument_LB = np.load(
            pixel_path+'code/data/instrument_LB_IMOv1.npy', allow_pickle=True).item()
        #  reformating for fgbuster
        # instr_ = {}
        # instr_['frequency'] = np.array([instrument_LB[f]['freq'] for f in instrument_LB.keys()])
        # instr_['depth_p'] = np.array([instrument_LB[f]['P_sens'] for f in instrument_LB.keys()])
        # instr_['fwhm'] = np.array([instrument_LB[f]['beam'] for f in instrument_LB.keys()])
        # instr_['depth_i'] = instr_['depth_p']/np.sqrt(2)

        # load map or ddt
        # mask = hp.ud_grade((hp.read_map(pixel_path+'code/data/'+'mask_LB.fits'), nside)

        print('WARNING: FULLSKY TEST')
        '''
        if machine == 'NERSC':
            mask = hp.ud_grade(hp.read_map(
                '/global/cscratch1/sd/josquin/HFI_Mask_GalPlane-apo0_2048_R2.00.fits', field=2), nside_out=nside)
        elif machine == 'local':
            mask = hp.ud_grade(hp.read_map(
                '/home/baptiste/Documents/these/pixel_based_analysis/results_and_data/pipeline_data/HFI_Mask_GalPlane-apo0_2048_R2.00.fits',
                field=2), nside_out=nside)

        mask[(mask != 0) * (mask != 1)] = 0
        '''
        print('WARNING: LB PTEP POWER SPECTRA should be used?')
        print('WARNING: CHECK IF RIGHT MASK')
        # mask = np.ones(mask.shape)
        mask = np.ones(hp.nside2npix(nside))

        # print(fsky)
        fsky = len(np.where(mask != 0)[0])*1.0/len(mask)
        print('fsky = ', fsky)
        print('time mask = ', time.time() - start)
        time_apo = time.time()

        aposize = 5.0
        apotype = 'Smooth'
        mask_apo = nmt.mask_apodization(mask, aposize=aposize, apotype=apotype)
        print('time mask apo = ', time.time() - time_apo)
        time_import = time.time()
        # mask = hp.ud_grade(mask_, nside)
        # del mask_
        # path_data = pixel_path+'code/'+'mock_LB_test_freq_maps_nonoise.npy'

        common_beam = 80.0
        # scaling_factor_for_sensitivity_due_to_transfer_function = np.array([1.03538701, 1.49483298, 1.68544978, 1.87216344, 1.77112946, 1.94462893,
        #                                                                     1.83405669, 1.99590729, 1.87252192, 2.02839885, 2.06834379, 2.09336487,
        #                                                                     1.93022977, 1.98931453, 2.02184001, 2.0436672,  2.05440473, 2.04784698,
        #                                                                     2.08458758, 2.10468234, 2.1148482, 2.13539999])
        # common_beam = 180.0*60.0 / np.pi
        # common_beam = None
        scaling_factor_for_sensitivity_due_to_transfer_function = np.ones(freq_number)

        # print('WARNING: NO BEAM TEST')
        # common_beam = None
        test_nobeam = False
        # scaling_factor_for_sensitivity_due_to_transfer_function = np.array([1]*22)

        # path_data = pixel_path+'code/'+'mock_LB_test_freq_maps.npy'
        if machine == 'local' or machine == 'NERSC':
            path_data = pixel_path+'results_and_data/pipeline_data/test_mock/data/mock_LB_nobeam' + \
                str(rank).zfill(4)+'.npy'
        elif machine == 'idark':
            # path_data = '/home/jost/simu/LB_mock/fullsky_nobeam/mock_LB_nobeam' + \
            #     str(rank).zfill(4)+'.npy'
            path_data = '/home/jost/simu/LB_mock/fullsky_withbeam/mock_LB' + \
                str(rank).zfill(4)+'.npy'
        else:
            print('ERROR: path_data not specified for this machine')
        print('PATH DATA FIXED FOR LOCAL DEBUG!!!!')
        path_data = '/home/baptiste/Downloads/mock_LB0000.npy'
        print('path data = ', path_data)
        # data = np.load(path_data)
        # freq_maps = data*mask
        # ddt = get_ddt(data, mask)
        # data = import_and_smooth_data(instrument_LB, rank, common_beam=common_beam, phase=phase)
        data = import_and_smooth_data(
            instrument_LB, rank, common_beam=common_beam, phase=phase, path=path_data, test_nobeam=test_nobeam)
        freq_maps = data*mask
        # freq_maps *= mask
        ddt = get_ddt(data, mask)
        print('time import = ', time.time() - time_import)
        time_spec_ini = time.time()

        scatter = prior_precision.tolist()
        scatter.append(0.1)  # first spectral index
        scatter.append(0.1)  # first spectral index
        scatter.append(0.01)  # ?? For r ?
        scatter.append(np.mean(prior_precision))  # ?? for beta_b ?
        scatter = np.array(scatter)

        prior_centre = np.array([0]*freq_number)
        total_params = np.append([0]*freq_number, [
            1.54, -3., 0, 0])

        model_skm = get_model(
            total_params[:freq_number], bir_angle=0*u.rad,
            frequencies_by_instrument_array=freq_by_instru, nside=nside,
            spectral_params=[total_params[freq_number], 20, total_params[freq_number+1]],
            sky_model='c1s0d0', sensitiviy_mode=sensitiviy_mode,
            one_over_f_mode=one_over_f_mode, instrument=INSTRU, overwrite_freq=overwrite_freq)
        # print('inv_noise before = ', model_skm.inv_noise)

        model_skm.sensitivity_LB /= scaling_factor_for_sensitivity_due_to_transfer_function
        model_skm.get_noise()
        model_skm.get_projection_op()
        # print('inv_noise after = ', model_skm.inv_noise)

        # nsteps = nsteps_spectral
        # discard = discard_spectral
        param_number = freq_number + 2  # num of msical + spectral index

        init_MCMC = np.random.normal(
            total_params[:param_number], scatter[:param_number], (2*param_number, param_number))
        print('time spec ini = ', time.time() - time_spec_ini)
        time_spec_min = time.time()
        spec_min_success = False
        iter_spec_min = 0
        # IPython.embed()
        while not spec_min_success and iter_spec_min < init_MCMC.shape[0]:
            print('Spectral likelihood minimisation iteration #', iter_spec_min)
            try:
                print('init spec=', init_MCMC[iter_spec_min])
                results_min = minimize(spectral_sampling, init_MCMC[iter_spec_min], args=(
                    ddt, model_skm, prior_matrix, prior_centre, True),
                    tol=1e-18, options={'maxiter': 1000}, method='L-BFGS-B')
                print('try success: ', results_min.success)
            except np.linalg.LinAlgError as err:
                if 'Singular matrix' in str(err):
                    print('ERROR CAUGHT: singular matrix')
                    iter_spec_min += 1
                    print('new iter #', iter_spec_min)
                    print('NEW init spec=', init_MCMC[iter_spec_min])
                    results_min = minimize(spectral_sampling, init_MCMC[iter_spec_min], args=(
                        ddt, model_skm, prior_matrix, prior_centre, True),
                        tol=1e-18, options={'maxiter': 1000}, method='L-BFGS-B')
                    print('except success: ', results_min.success)
                    spec_min_success = results_min.success
                else:
                    raise
            iter_spec_min += 1
            spec_min_success = results_min.success
        print('time spec min = ', time.time() - time_spec_min)
        print('results spec = ', results_min.x)
        print('spec min success = ', results_min.success)
        # IPython.embed()

        np.save(output_dir+'results_spec.npy', results_min.x)
        # results_min.x[-1] = -3
        # freq_maps[:,  np.where(mask == 0)[0]] = hp.UNSEEN

        # results_min.x = [0]*22
        # results_min.x.append(1.54)
        # results_min.x.append(-3)
        # results_min.x = np.array(results_min.x)

        # results_min.x = np.zeros(freq_number+2)
        # results_min.x[-2] = 1.54
        # results_min.x[-1] = -3
        start_min = time.time()
        # for rank in range(99):
        # print(rank)
        # output_dir = pixel_path+'results_and_data/pipeline_data/test_mock/' + \
        #     str(rank).zfill(4) + '_fullsky_nobeam/'
        # print(output_dir)
        # path_data = pixel_path+'results_and_data/pipeline_data/test_mock/data/mock_LB_nobeam' + \
        #     str(rank).zfill(4)+'.npy'
        # print('path data = ', path_data)
        # data = np.load(path_data)
        # freq_maps = data*mask
        # ddt = get_ddt(data, mask)
        Cl_noise_matrix, A, W = from_spectra_to_cosmo(results_min.x, model_skm, sensitiviy_mode=sensitiviy_mode,
                                                      one_over_f_mode=one_over_f_mode,
                                                      beam_corrected=beam_correction,
                                                      one_over_ell=one_over_ell,
                                                      lmin=0, lmax=3*nside-1,
                                                      common_beam=common_beam,
                                                      scaling_factor=scaling_factor_for_sensitivity_due_to_transfer_function,
                                                      test_nobeam=test_nobeam, INSTRU=INSTRU)
        # scaling_factor=np.ones(freq_number) or scaling_factor_for_sensitivity_due_to_transfer_function

        '''in from_spectra_to_cosmo() lmin=0 and lmax=3*nside so that bin_cell has the right ell range as input, indeed it expects those value. bin_cell is so initialised that it then takes care of having the desired lmin. Not necessarily the lmax as we typically set lmax < 3*nside but it is taken care of with indices_ellrange later on.'''
    # 3*nside-1,
        # ell_noise = np.linspace(0, 3*nside-1, 3*nside, dtype=int)

        # noise_lvl = np.array([instrument_LB[f]['P_sens'] for f in instrument_LB.keys()])
        # beam_rad = np.array([instrument_LB[f]['beam']
        #                      for f in instrument_LB.keys()]) * u.arcmin.to(u.rad)
        # for common resolution etc?

        # beam_rad = np.array([common_beam]*freq_number)
        # noise_nl = []
        # from healpy import gauss_beam
        # for f in range(len(noise_lvl)):
        #     Bl = gauss_beam(beam_rad[f], lmax=3*nside-1)  # [2:]
        #     noise = (noise_lvl[f]*np.pi/60/180)**2 * np.ones(len(ell_noise))
        #     noise_nl.append(noise / (Bl**2))
        # noise_nl = np.array(noise_nl)
        # # noise_nl = np.repeat(noise_nl, 2, 0)[..., lmin-2:]
        # noise_nl = np.repeat(noise_nl, 2, 0)[...]
        # nl_inv = 1/noise_nl
        # AtNA = np.einsum('fi, fl, fj -> lij', A, nl_inv, A)
        # inv_AtNA = np.linalg.inv(AtNA)
        # noise_cl = inv_AtNA.swapaxes(-3, -1)
        #
        # Cl_noise = noise_cl[0, 0]
        #
        # Cl_noise_matrix = np.zeros([2, 2, Cl_noise.shape[0]])
        # Cl_noise_matrix[0, 0] = Cl_noise
        # Cl_noise_matrix[1, 1] = Cl_noise

        print('Cl_noise_matrix shape=', Cl_noise_matrix.shape)
        clean_CMB_map = W[:2].dot(freq_maps)
        np.save(output_dir+'output_cmb_map.npy', clean_CMB_map)
        clean_dust_map = W[2:4].dot(freq_maps)
        np.save(output_dir+'output_dust_map.npy', clean_dust_map)
        clean_synch_map = W[4:].dot(freq_maps)
        np.save(output_dir+'output_synch_map.npy', clean_synch_map)

        purify_e = False
        purify_b = False
        if purify_b and fsky > 0.9:
            print('WARNING: if full sky, purify_b should be FALSE')
        if not purify_b and fsky < 0.9:
            print('WARNING: if PARTIAL sky, purify_b should be TRUE')
        # common_beam = 80.0
        # common_beam = 80
        if common_beam is not None:
            Bl_eff = hp.gauss_beam(np.radians(common_beam/60.0), lmax=3*nside+1, pol=True)[:, 1]
        else:
            Bl_eff = np.ones(3*nside+2)
        # Bl_eff = np.ones(3*nside + 1)

        w = nmt.NmtWorkspace()
        b = binning_definition(nside, lmin=lmin, lmax=lmax, fsky=fsky)

        # cltt, clee, clbb, clte = hp.read_cl(
        #     '/project/projectdirs/litebird/simulations/maps/birefringence_project_paper/Cls_Planck2018_for_PTEP_2020_r0.fits')[:, :4000]
        clbb = Cl_fid['BB']
        clee = Cl_fid['EE']
        clte = Cl_fid['EE']*0
        cltt = Cl_fid['EE']*0

        # cltt, clee, clbb, clte = hp.read_cl(
        #     '/project/projectdirs/litebird/simulations/maps/birefringence_project_paper/Cls_Planck2018_for_PTEP_2020_r0.fits')[:, :4000]
        mp_t_sim, mp_q_sim, mp_u_sim = hp.synfast(
            [cltt, clee, clbb, clte], nside=nside, new=True, verbose=False)

        ell_eff_ = b.get_effective_ells()
        # TODO:
        indices_ellrange = np.where((ell_eff_ >= lmin) & (ell_eff_ <= lmax))[0]
        ell_eff = ell_eff_[indices_ellrange]
        np.save(output_dir+'ell_eff.npy', ell_eff)
        np.save(output_dir+'indices_ellrange.npy', indices_ellrange)
        # ell_eff = ell_eff_[(ell_eff_ >= lmin) & (ell_eff_ <= lmax)]
        print('test n_iter namaster')
        n_iter_namaster = 10  # default = 3
        f2y0 = get_field(mp_q_sim, mp_u_sim, mask_apo, Bl_eff,
                         purify_e=purify_e, purify_b=purify_b, n_iter=n_iter_namaster)
        w.compute_coupling_matrix(f2y0, f2y0, b, n_iter=n_iter_namaster)

        field = get_field(clean_CMB_map[0], clean_CMB_map[1], mask_apo, Bl_eff,
                          purify_e=purify_e, purify_b=purify_b, n_iter=n_iter_namaster)
        Cls_data = compute_master(field, field, w)[..., indices_ellrange]
        Cls_data_matrix = np.zeros((2, 2, len(Cls_data[0])))
        Cls_data_matrix[0, 0] = Cls_data[0]
        Cls_data_matrix[1, 1] = Cls_data[3]
        Cls_data_matrix[1, 0] = Cls_data[1]
        Cls_data_matrix[0, 1] = Cls_data[1]

        # Cls_data = hp.anafast([clean_CMB_map[0]*0, clean_CMB_map[0],
        #                        clean_CMB_map[1]])[:, lmin:lmax+1]
        # Cls_data_matrix[0, 0] = Cls_data[1]
        # Cls_data_matrix[1, 1] = Cls_data[2]
        # Cls_data_matrix[1, 0] = Cls_data[4]
        # Cls_data_matrix[0, 1] = Cls_data[4]
        np.save(output_dir+'Cls_data_matrix_fixedspec_debugnamaster.npy', Cls_data_matrix)
        # IPython.embed()

        Cl_noise_matrix_bin_ = b.bin_cell(
            np.array([Cl_noise_matrix[0, 0]]))[..., indices_ellrange]
        print('Cl_noise_matrix_bin_ shape=', Cl_noise_matrix_bin_.shape)
        Cl_noise_matrix_bin = np.zeros((2, 2, len(Cl_noise_matrix_bin_[0])))

        Cl_noise_matrix_bin[0, 0] = Cl_noise_matrix_bin_[0]
        Cl_noise_matrix_bin[1, 1] = Cl_noise_matrix_bin_[0]
        np.save(output_dir+'Cl_noise_matrix_bin.npy', Cl_noise_matrix_bin)
        np.save(output_dir+'Cl_noise_matrix_beforebin.npy', Cl_noise_matrix)

        print('Cl_noise_matrix_bin shape', Cl_noise_matrix_bin.shape)
        print('Cls_data_matrix shape', Cls_data_matrix.shape)

        # test_cosmo = cosmo_like_data([0, 0], Cl_noise_matrix_bin,
        #                              Cls_data_matrix, b, fsky, minimisation=True, ell_indices=indices_ellrange)
        print('time init cosmo/ namaster etc = ', time.time() - start_min)
        time_cosmo_min = time.time()
        bounds_cosmo = ((-1e-5, 5), (-np.pi/8, np.pi/8))
        # init_cosmo = [0, 0]
        init_cosmo = np.random.normal([0, 0], np.abs([bounds_cosmo[0][0], bounds_cosmo[1][0]]))
        print('init cosmo = ', init_cosmo)
        results_min_cosmo = minimize(cosmo_like_data, init_cosmo, args=(
            Cl_noise_matrix_bin, Cls_data_matrix, b, fsky, True, indices_ellrange),
            tol=1e-18, options={'maxiter': 1000}, method='L-BFGS-B', bounds=bounds_cosmo)
        np.save(output_dir+'results_cosmo.npy', results_min_cosmo.x)
        print('results cosmo = ', results_min_cosmo.x)
        rad2deg = 1*u.rad.to(u.deg)
        print('beta deg = ', results_min_cosmo.x[-1]*rad2deg)
        print('results success = ', results_min_cosmo.success)
        if not results_min_cosmo.success:
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print('')
        print('')
        print('time cosmo min =', time.time()-time_cosmo_min)
        '''
        r_grid = np.linspace(bounds_cosmo[0][0], bounds_cosmo[0][1], 1000)
        like_grid_r = []
        for r in r_grid:
            param = [r, 0]
            like_grid_r.append(cosmo_like_data(param, Cl_noise_matrix_bin,
                                               Cls_data_matrix, b, fsky, True, indices_ellrange))
        like_grid_r = np.array(like_grid_r)

        plt.plot(r_grid, like_grid_r)
        plt.savefig(output_dir+'like_grid_r.png')
        plt.close()

        beta_grid = np.linspace(bounds_cosmo[1][0], bounds_cosmo[1][1], 1000)
        like_grid_beta = []
        for beta in beta_grid:
            param = [0, beta]
            like_grid_beta.append(cosmo_like_data(param, Cl_noise_matrix_bin,
                                                  Cls_data_matrix, b, fsky, True, indices_ellrange))
        like_grid_beta = np.array(like_grid_beta)
        plt.plot(beta_grid, like_grid_beta)
        plt.savefig(output_dir+'like_grid_beta.png')
        plt.close()
        '''
        # for rank in range(99):
        #     # print(rank)
        # output_dir = pixel_path+'results_and_data/pipeline_data/test_mock/' + \
        #     str(rank).zfill(4) + '_fullsky_nobeam/'
        # print(output_dir)
        # res_cosmo = np.load(output_dir+'results_cosmo.npy')
        time_plot = time.time()
        Cl_cmb_model = np.zeros([4, Cl_fid['EE'].shape[0]])
        Cl_cmb_model[1] = copy.deepcopy(Cl_fid['EE'])
        Cl_cmb_model[2] = copy.deepcopy(Cl_fid['BlBl'])*1 + \
            copy.deepcopy(Cl_fid['BuBu']) * results_min_cosmo.x[0]
        Cl_cmb_rot = lib.cl_rotation(Cl_cmb_model.T, results_min_cosmo.x[1]*u.rad).T
        Cl_cmb_rot_EE = b.bin_cell(np.array([Cl_cmb_rot[1, :3*nside]]))[0][indices_ellrange]
        Cl_cmb_rot_BB = b.bin_cell(np.array([Cl_cmb_rot[2, :3*nside]]))[0][indices_ellrange]
        Cl_cmb_rot_EB = b.bin_cell(np.array([Cl_cmb_rot[4, :3*nside]]))[0][indices_ellrange]

        Cls_model_matrix_plot = np.zeros((2, 2, len(Cl_cmb_rot_EE)))
        Cls_model_matrix_plot[0, 0] = Cl_cmb_rot_EE
        Cls_model_matrix_plot[1, 1] = Cl_cmb_rot_BB
        Cls_model_matrix_plot[1, 0] = Cl_cmb_rot_EB
        Cls_model_matrix_plot[0, 1] = Cl_cmb_rot_EB
        np.save(output_dir+'Cls_model_matrix_plot.npy', Cls_model_matrix_plot)

        ell_index_min_plot = 0
        norm = ell_eff[ell_index_min_plot:] * (ell_eff[ell_index_min_plot:] + 1) / 2 / np.pi
        plt.plot(ell_eff[ell_index_min_plot:], norm*Cl_cmb_rot_EE[ell_index_min_plot:] +
                 norm*Cl_noise_matrix_bin[0, 0][ell_index_min_plot:], label='model EE  + noise')
        plt.plot(ell_eff[ell_index_min_plot:], norm *
                 Cl_cmb_rot_EE[ell_index_min_plot:], label='model EE')
        plt.plot(ell_eff[ell_index_min_plot:], norm * Cls_data_matrix[0, 0]
                 [ell_index_min_plot:], label='data EE')
        plt.plot(ell_eff[ell_index_min_plot:], -norm*Cls_data_matrix[0, 0]
                 [ell_index_min_plot:], '--', color='green')
        plt.plot(ell_eff[ell_index_min_plot:], norm*Cl_fid['EE']
                 [ell_eff.astype(int)], linestyle='--', label='true primordial EE')
        plt.plot(ell_eff[ell_index_min_plot:], norm*Cl_fid['EE'][ell_eff.astype(int)] + norm *
                 Cl_noise_matrix_bin[0, 0][ell_index_min_plot:], linestyle='--', label='true primordial EE + mean noise')
        plt.legend()
        plt.loglog()
        plt.ylabel(r'$\frac{\ell (\ell+1)}{2\pi} C_\ell^{EE}$')
        plt.xlabel(r'$\ell$')
        plt.savefig(output_dir+'EE_spectra.png', bbox_inches='tight')
        plt.close()

        plt.plot(ell_eff[ell_index_min_plot:], norm*Cl_cmb_rot_BB[ell_index_min_plot:] +
                 norm*Cl_noise_matrix_bin[1, 1][ell_index_min_plot:], label='model BB + noise')
        plt.plot(ell_eff[ell_index_min_plot:], norm *
                 Cl_cmb_rot_BB[ell_index_min_plot:], label='model BB')
        plt.plot(ell_eff[ell_index_min_plot:], norm * Cls_data_matrix[1, 1]
                 [ell_index_min_plot:], label='data BB')
        plt.plot(ell_eff[ell_index_min_plot:], -norm*Cls_data_matrix[1, 1]
                 [ell_index_min_plot:], '--', color='green')
        plt.plot(ell_eff[ell_index_min_plot:], norm*Cl_fid['BB']
                 [ell_eff.astype(int)], linestyle='--', label='true primordial BB')
        plt.plot(ell_eff[ell_index_min_plot:], norm*Cl_fid['BB'][ell_eff.astype(int)] + norm *
                 Cl_noise_matrix_bin[1, 1][ell_index_min_plot:], linestyle='--', label='true primordial BB + mean noise')
        plt.legend()
        plt.loglog()
        plt.ylabel(r'$\frac{\ell (\ell+1)}{2\pi} C_\ell^{BB}$')
        plt.xlabel(r'$\ell$')
        plt.savefig(output_dir+'BB_spectra.png', bbox_inches='tight')
        plt.close()

        plt.plot(ell_eff[ell_index_min_plot:], norm *
                 Cl_cmb_rot_EB[ell_index_min_plot:], label='model EB')
        plt.plot(ell_eff[ell_index_min_plot:], norm*Cls_data_matrix[0, 1]
                 [ell_index_min_plot:], label='data EB')
        # plt.plot(ell_eff[ell_index_min_plot:], -Cls_data_matrix[0, 1]
        #          [ell_index_min_plot:], '--', color='tab:orange')
        plt.legend()
        # plt.loglog()
        plt.xscale('log')
        plt.ylabel(r'$\frac{\ell (\ell+1)}{2\pi} C_\ell^{EB}$')
        plt.xlabel(r'$\ell$')
        plt.savefig(output_dir+'EB_spectra.png', bbox_inches='tight')
        plt.close()
        print('time plot = ', time.time() - time_plot)
        print('')
        print('time one map = ', time.time() - start)
        IPython.embed()

    exit()

    '''
    stat, bias, var, Cl_fg, Cl_cmb, Cl_residuals_matrix, ell, W_cmb, dW_cmb, ddW_cmb = get_residuals(
        model_results, fg_freq_maps, sigma_spectral, lmin, lmax, fsky, params,
        cmb_spectra=spectra_true, true_A_cmb=model_data.mix_effectiv[:, :2], pivot_angle_index=pivot_angle_index)

    WA_cmb = W_cmb.dot(model_results.mix_effectiv[:, :2])
    dWA_cmb = dW_cmb.dot(model_results.mix_effectiv[:, :2])
    W_dBdB_cmb = ddW_cmb.dot(model_results.mix_effectiv[:, :2])
    VA_cmb = np.einsum('ij,ij...->...', sigma_spectral, W_dBdB_cmb[:, :])

    results_min = minimize(spectral_sampling, init_MCMC[0], args=(
        ddt, model_skm, prior_matrix, prior_centre, True), tol=1e-18, options={'maxiter': 1000}, method='L-BFGS-B')

    start = time.time()
    sampler_spec = EnsembleSampler(
        nwalkers, param_number, spectral_sampling, args=[
            ddt, model_skm, prior_matrix, prior_centre])
    sampler_spec.reset()
    start = time.time()
    sampler_spec.run_mcmc(init_MCMC, nsteps, progress=True)
    end = time.time()
    print('time MCMC = ', end - start)
    spec_samples_raw = sampler_spec.get_chain()
    spec_samples = sampler_spec.get_chain(discard=discard, flat=True)

    np.save('spec_samples.npy', spec_samples)
    np.save('spec_samples_raw.npy', spec_samples_raw)
    '''
    exit()


######################################################
# MAIN CALL
if __name__ == "__main__":
    main()
