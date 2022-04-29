import time
import IPython
import numpy as np
import bjlib.likelihood_SO as lSO
from astropy import units as u
import bjlib.V3calc as V3
import pixel_based_angle_estimation as pix
from scipy.linalg import block_diag
import matplotlib.pyplot as plt
import healpy as hp
import copy
from mpi4py import MPI

# import emcee
# from multiprocessing import Pool
# import argparse
# from schwimmbad import MPIPool
# from schwimmbad import MultiPool

# import cloudpickle
# import tracemalloc
# import os
# import sys
# from mpi4py import MPI


def diff_mixing_matrix(model):
    if not model.fix_temp:
        diff_element = model.A.diff(model.frequencies,
                                    model.spectral_indices[0], model.spectral_indices[1], model.spectral_indices[2])
    else:
        diff_element = model.A.diff(model.frequencies,
                                    model.spectral_indices[0], model.spectral_indices[1])
    mix_diff_Bd = []
    if not model.fix_temp:
        mix_diff_Td = []
    mix_diff_Bs = []
    for i in range(len(model.frequencies)):
        mix_diff_Bd.append([0, diff_element[0][i][0], 0])
        if not model.fix_temp:
            mix_diff_Td.append([0, diff_element[1][i][0], 0])
            mix_diff_Bs.append([0, 0, diff_element[2][i][0]])
        else:
            mix_diff_Bs.append([0, 0, diff_element[1][i][0]])
    mix_diff_Bd = np.array(mix_diff_Bd)
    mix_diff_Bd_QU = np.repeat(mix_diff_Bd, 2, 1)
    mix_diff_Bd_QU = np.repeat(mix_diff_Bd_QU, 2, 0)

    if not model.fix_temp:
        mix_diff_Td = np.array(mix_diff_Td)
        mix_diff_Td_QU = np.repeat(mix_diff_Td, 2, 1)
        mix_diff_Td_QU = np.repeat(mix_diff_Td_QU, 2, 0)

    mix_diff_Bs = np.array(mix_diff_Bs)
    mix_diff_Bs_QU = np.repeat(mix_diff_Bs, 2, 1)
    mix_diff_Bs_QU = np.repeat(mix_diff_Bs_QU, 2, 0)

    # mixing_matrix = np.repeat(A_, 2, 0)

    for i in range(np.shape(mix_diff_Bd_QU)[0]):
        for j in range(np.shape(mix_diff_Bd_QU)[1]):
            mix_diff_Bd_QU[i, j] = mix_diff_Bd_QU[i, j] *\
                (((i % 2)-(1-j % 2)) % 2)

            mix_diff_Bs_QU[i, j] = mix_diff_Bs_QU[i, j] *\
                (((i % 2)-(1-j % 2)) % 2)
            if not model.fix_temp:
                mix_diff_Td_QU[i, j] = mix_diff_Td_QU[i, j] *\
                    (((i % 2)-(1-j % 2)) % 2)
    if not model.fix_temp:
        return mix_diff_Bd_QU, mix_diff_Td_QU, mix_diff_Bs_QU
    else:
        return mix_diff_Bd_QU, mix_diff_Bs_QU


def diff_diff_mixing_matrix(model):
    if not model.fix_temp:
        diff_element = model.A.diff_diff(model.frequencies,
                                         model.spectral_indices[0], model.spectral_indices[1], model.spectral_indices[2])
    else:
        diff_element = model.A.diff_diff(model.frequencies,
                                         model.spectral_indices[0], model.spectral_indices[1])
    mix_diff_diff_Bd = []
    if not model.fix_temp:
        mix_diff_diff_Td = []
    mix_diff_diff_Bs = []
    for i in range(len(model.frequencies)):
        mix_diff_diff_Bd.append([0, diff_element[0][0][i][0], 0])
        if not model.fix_temp:
            mix_diff_diff_Td.append([0, diff_element[1][1][i][0], 0])
            mix_diff_diff_Bs.append([0, 0, diff_element[2][2][i][0]])
        else:
            mix_diff_diff_Bs.append([0, 0, diff_element[1][1][i][0]])

    mix_diff_diff_Bd = np.array(mix_diff_diff_Bd)
    mix_diff_diff_Bd_QU = np.repeat(mix_diff_diff_Bd, 2, 1)
    mix_diff_diff_Bd_QU = np.repeat(mix_diff_diff_Bd_QU, 2, 0)

    if not model.fix_temp:
        mix_diff_diff_Td = np.array(mix_diff_diff_Td)
        mix_diff_diff_Td_QU = np.repeat(mix_diff_diff_Td, 2, 1)
        mix_diff_diff_Td_QU = np.repeat(mix_diff_diff_Td_QU, 2, 0)

    mix_diff_diff_Bs = np.array(mix_diff_diff_Bs)
    mix_diff_diff_Bs_QU = np.repeat(mix_diff_diff_Bs, 2, 1)
    mix_diff_diff_Bs_QU = np.repeat(mix_diff_diff_Bs_QU, 2, 0)
    # mixing_matrix = np.repeat(A_, 2, 0)

    for i in range(np.shape(mix_diff_diff_Bd_QU)[0]):
        for j in range(np.shape(mix_diff_diff_Bd_QU)[1]):
            mix_diff_diff_Bd_QU[i, j] = mix_diff_diff_Bd_QU[i, j] *\
                (((i % 2)-(1-j % 2)) % 2)
            if not model.fix_temp:
                mix_diff_diff_Td_QU[i, j] = mix_diff_diff_Td_QU[i, j] *\
                    (((i % 2)-(1-j % 2)) % 2)

            mix_diff_diff_Bs_QU[i, j] = mix_diff_diff_Bs_QU[i, j] *\
                (((i % 2)-(1-j % 2)) % 2)
    if not model.fix_temp:
        return mix_diff_diff_Bd_QU, mix_diff_diff_Td_QU, mix_diff_diff_Bs_QU
    else:
        return mix_diff_diff_Bd_QU, mix_diff_diff_Bs_QU


def diff_miscal_matrix(model):
    i = 0
    freq_num = len(model.frequencies)
    list_diff_miscal = []
    for angle in model.miscal_angles:

        rotation_block = 2*np.array(
            [[-np.sin(2*angle),  np.cos(2*angle)],
             [-np.cos(2*angle), -np.sin(2*angle)]
             ])
        diff_miscal_i = block_diag(
            np.zeros([2*i, 2*i]), rotation_block, np.zeros([2*freq_num - 2 - 2*i, 2*freq_num - 2 - 2*i]))
        list_diff_miscal.append(diff_miscal_i)
        i += 1
    return list_diff_miscal


def diff_diff_miscal_matrix(model):
    i = 0
    freq_num = len(model.frequencies)
    list_diff_miscal = []
    for angle in model.miscal_angles:

        rotation_block = 4*np.array(
            [[-np.cos(2*angle),  -np.sin(2*angle)],
             [np.sin(2*angle), -np.cos(2*angle)]
             ])
        diff_miscal_i = block_diag(
            np.zeros([2*i, 2*i]), rotation_block, np.zeros([2*freq_num - 2 - 2*i, 2*freq_num - 2 - 2*i]))
        list_diff_miscal.append(diff_miscal_i)
        i += 1
    return list_diff_miscal


def diff_bir_matrix(model):
    rotation_block = 2*np.array(
        [[-np.sin(2*model.bir_angle),  np.cos(2*model.bir_angle)],
         [-np.cos(2*model.bir_angle), np.sin(2*model.bir_angle)]
         ])
    diff_bir_matrix = block_diag(rotation_block, np.zeros([4, 4]))
    return diff_bir_matrix


def diff_diff_bir_matrix(model):
    rotation_block = 4*np.array(
        [[-np.cos(2*model.bir_angle),  -np.sin(2*model.bir_angle)],
         [np.sin(2*model.bir_angle), -np.cos(2*model.bir_angle)]
         ])
    diff_bir_matrix = block_diag(rotation_block, np.zeros([4, 4]))
    return diff_bir_matrix


def effectiv_diff_mixing_matrix(diff_index, diff_list, model):
    # print(diff_index)
    if not diff_index//model.miscal_angles.shape[0]:
        # print('misscal')
        effetiv_diff = diff_list[diff_index].dot(model.mixing_matrix).dot(model.bir_matrix)
    elif diff_index == model.miscal_angles.shape[0]:
        # print('birefringence')
        effetiv_diff = model.miscal_matrix.dot(model.mixing_matrix).dot(diff_list[diff_index])
    else:
        # print('spectral')
        effetiv_diff = model.miscal_matrix.dot(diff_list[diff_index]).dot(model.bir_matrix)
    return effetiv_diff


def effectiv_diff_mixing_matrix_new(diff_index, diff_list, model, params):
    if params[diff_index] == 'miscal':
        effetiv_diff = diff_list[diff_index].dot(model.mixing_matrix).dot(model.bir_matrix)

    elif params[diff_index] == 'birefringence':
        effetiv_diff = model.miscal_matrix.dot(model.mixing_matrix).dot(diff_list[diff_index])

    elif params[diff_index] == 'spectral':
        effetiv_diff = model.miscal_matrix.dot(diff_list[diff_index]).dot(model.bir_matrix)

    else:
        print('parameter not recognized, miscal, birefringence or spectral only')
        return None

    return effetiv_diff


def effectiv_doublediff_mixing_matrix(diff_index1, diff_index2, diff_list, diff_diff_list, model):
    # print(diff_index1, diff_index2)
    if diff_index1 == diff_index2:
        if not diff_index1//model.miscal_angles.shape[0]:
            miscal = diff_diff_list[diff_index1]
        else:
            miscal = model.miscal_matrix

        if diff_index1 == model.miscal_angles.shape[0]:
            bir = diff_diff_list[diff_index1]
        else:
            bir = model.bir_matrix

        if diff_index1 == model.miscal_angles.shape[0]+1 or diff_index1 == model.miscal_angles.shape[0]+2:
            spectral = diff_diff_list[diff_index1]
        else:
            spectral = model.mixing_matrix
        double_diff_matrix = miscal.dot(spectral).dot(bir)
        # print('diagonal')
        return double_diff_matrix

    if not diff_index1//model.miscal_angles.shape[0] and not diff_index2//model.miscal_angles.shape[0]:
        # print('double miscal')
        return np.zeros([len(model.frequencies)*2, 6])
    elif (diff_index1 == model.miscal_angles.shape[0]+1 or diff_index1 == model.miscal_angles.shape[0]+2) and (diff_index2 == model.miscal_angles.shape[0]+1 or diff_index2 == model.miscal_angles.shape[0]+2):
        # print('double spectral')
        return np.zeros([len(model.frequencies)*2, 6])
    else:
        if not diff_index1//model.miscal_angles.shape[0]:
            # print('miscal1')
            miscal = diff_list[diff_index1]
        elif not diff_index2//model.miscal_angles.shape[0]:
            # print('miscal2')
            miscal = diff_list[diff_index2]
        else:
            # print('miscal normal')
            miscal = model.miscal_matrix

        if diff_index1 == model.miscal_angles.shape[0]:
            # print('bir1')
            bir = diff_list[diff_index1]
        elif diff_index2 == model.miscal_angles.shape[0]:
            # print('bir2')
            bir = diff_list[diff_index2]
        else:
            # print('bir normal')
            bir = model.bir_matrix

        if diff_index1 == model.miscal_angles.shape[0]+1 or diff_index1 == model.miscal_angles.shape[0]+2:
            # print('spectral1')

            spectral = diff_list[diff_index1]
        elif diff_index2 == model.miscal_angles.shape[0]+1 or diff_index2 == model.miscal_angles.shape[0]+2:
            spectral = diff_list[diff_index2]
            # print('spectral2')
        else:
            spectral = model.mixing_matrix
            # print('spectral normal')

        double_diff_matrix = miscal.dot(spectral).dot(bir)
    return double_diff_matrix


def effectiv_doublediff_mixing_matrix_new(diff_index1, diff_index2, diff_list, diff_diff_list, model, params):
    if diff_index1 == diff_index2:
        if params[diff_index1] == 'miscal':
            miscal = diff_diff_list[diff_index1]
        else:
            miscal = model.miscal_matrix

        if params[diff_index1] == 'birefringence':
            bir = diff_diff_list[diff_index1]
        else:
            bir = model.bir_matrix

        if params[diff_index1] == 'spectral':
            spectral = diff_diff_list[diff_index1]
        else:
            spectral = model.mixing_matrix
        double_diff_matrix = miscal.dot(spectral).dot(bir)
        return double_diff_matrix

    if params[diff_index1] == 'miscal' and params[diff_index2] == 'miscal':
        return np.zeros([len(model.frequencies)*2, 6])
    elif params[diff_index1] == 'spectral' and params[diff_index2] == 'spectral':
        return np.zeros([len(model.frequencies)*2, 6])
    else:
        if params[diff_index1] == 'miscal':
            miscal = diff_list[diff_index1]
        elif params[diff_index2] == 'miscal':
            miscal = diff_list[diff_index2]
        else:
            miscal = model.miscal_matrix

        if params[diff_index1] == 'birefringence':
            bir = diff_list[diff_index1]
        elif params[diff_index2] == 'birefringence':
            bir = diff_list[diff_index2]
        else:
            bir = model.bir_matrix

        if params[diff_index1] == 'spectral':
            spectral = diff_list[diff_index1]
        elif params[diff_index2] == 'spectral':
            spectral = diff_list[diff_index2]
        else:
            spectral = model.mixing_matrix

        double_diff_matrix = miscal.dot(spectral).dot(bir)
    return double_diff_matrix


def fisher(ddt, model, diff_list, diff_diff_list, Ninvfactor=1):
    AtNm1A = model.mix_effectiv.T.dot(model.inv_noise*Ninvfactor).dot(model.mix_effectiv)
    invAtNm1A = np.linalg.inv(AtNm1A)
    Nm1A_invAtNm1A = Ninvfactor*model.inv_noise.dot(model.mix_effectiv).dot(invAtNm1A)
    AtNm1 = model.mix_effectiv.T.dot(model.inv_noise*Ninvfactor)
    # IPython.embed()
    param_num = len(diff_list)
    # fisher_matrix = np.empty([len(model.frequencies)+1+2, len(model.frequencies)+1+2])
    fisher_matrix = np.empty([param_num, param_num])
    # for i in range(len(model.frequencies)+1+2):
    for i in range(param_num):
        # for ii in range(len(model.frequencies)+1+2):
        # for ii in range(i, len(model.frequencies)+1+2):
        for ii in range(i, param_num):
            # print('i=', i, 'ii=', ii)
            # if i == ii:
            #     fisher_matrix[i][ii] = None
            # else:
            A_i = effectiv_diff_mixing_matrix(i, diff_list, model)
            A_ii = effectiv_diff_mixing_matrix(ii, diff_list, model)
            A_i_ii = effectiv_doublediff_mixing_matrix(i, ii, diff_list, diff_diff_list, model)
            # A_ii_i = effectiv_doublediff_mixing_matrix(ii, i, diff_list, model)
            # print('difference ordre derivee = ', A_i_ii - A_ii_i)
            # if not i//6 and not ii//6:
            # print(i, ii)
            # print('A_i', A_i)
            # print('A_ii', A_ii)
            # print('A_i_ii', A_i_ii)

            AitP = A_i.T.dot(Ninvfactor*model.projection)

            term1 = Ninvfactor*model.projection.dot(A_ii).dot(invAtNm1A).dot(AitP)
            term2 = Nm1A_invAtNm1A.dot(A_i_ii.T).dot(Ninvfactor*model.projection)
            term3 = Nm1A_invAtNm1A.dot(A_ii.T).dot(Nm1A_invAtNm1A).dot(AitP)
            term4 = Nm1A_invAtNm1A.dot(A_i.T).dot(
                Nm1A_invAtNm1A).dot(A_ii.T).dot(Ninvfactor*model.projection)
            term5 = Nm1A_invAtNm1A.dot(AitP).dot(A_ii).dot(invAtNm1A).dot(AtNm1)
            # term5 = Nm1A_invAtNm1A.dot(AitP).dot(A_ii).dot(
            #     invAtNm1A).dot(model.mix_effectiv.T).dot(model.projection)

            tot = term1 + term2 - term3 - term4 - term5

            # trace = np.einsum('ij,ji...m->m', tot, ddt)
            if len(ddt.shape) == 3:
                sum_trace = np.einsum('ij,ji...m->', tot, ddt)
            else:
                sum_trace = np.einsum('ij,ji->...', tot, ddt)

            fisher_matrix[i][ii] = -sum_trace*2

            fisher_matrix[ii][i] = -sum_trace*2
            '''
            WARNING : Factor 2 for +transpose term, see Clara's thesis
            '''
            # sterm1 = term_fisher_debug(term1, ddt)
            # sterm2 = term_fisher_debug(term2, ddt)
            # sterm3 = term_fisher_debug(term3, ddt)
            # sterm4 = term_fisher_debug(term4, ddt)
            # sterm5 = term_fisher_debug(term5, ddt)
            # print('i, ii', i, ii)
            # print('sterm1 = ', sterm1)
            # print('sterm2 = ', sterm2)
            # print('sterm3 = ', sterm3)
            # print('sterm4 = ', sterm4)
            # print('sterm5 = ', sterm5)
            # print('sum =', sterm1 + sterm2 - sterm3 - sterm4 - sterm5)

            if i == 11 and ii == 2:
                sterm1 = term_fisher_debug(term1, ddt)
                sterm2 = term_fisher_debug(term2, ddt)
                sterm3 = term_fisher_debug(term3, ddt)
                sterm4 = term_fisher_debug(term4, ddt)
                sterm5 = term_fisher_debug(term5, ddt)
                # sterm5_clara = term_fisher_debug(term5_clara, ddt)

                IPython.embed()

    return fisher_matrix


def fisher_new(ddt, model, diff_list, diff_diff_list, params, Ninvfactor=1):
    AtNm1A = model.mix_effectiv.T.dot(model.inv_noise*Ninvfactor).dot(model.mix_effectiv)
    invAtNm1A = np.linalg.inv(AtNm1A)
    Nm1A_invAtNm1A = Ninvfactor*model.inv_noise.dot(model.mix_effectiv).dot(invAtNm1A)
    AtNm1 = model.mix_effectiv.T.dot(model.inv_noise*Ninvfactor)
    param_num = len(diff_list)
    fisher_matrix = np.empty([param_num, param_num])
    for i in range(param_num):
        for ii in range(i, param_num):
            A_i = effectiv_diff_mixing_matrix_new(i, diff_list, model, params)
            A_ii = effectiv_diff_mixing_matrix_new(ii, diff_list, model, params)
            A_i_ii = effectiv_doublediff_mixing_matrix_new(
                i, ii, diff_list, diff_diff_list, model, params)

            AitP = A_i.T.dot(Ninvfactor*model.projection)

            term1 = Ninvfactor*model.projection.dot(A_ii).dot(invAtNm1A).dot(AitP)
            term2 = Nm1A_invAtNm1A.dot(A_i_ii.T).dot(Ninvfactor*model.projection)
            term3 = Nm1A_invAtNm1A.dot(A_ii.T).dot(Nm1A_invAtNm1A).dot(AitP)
            term4 = Nm1A_invAtNm1A.dot(A_i.T).dot(
                Nm1A_invAtNm1A).dot(A_ii.T).dot(Ninvfactor*model.projection)
            term5 = Nm1A_invAtNm1A.dot(AitP).dot(A_ii).dot(invAtNm1A).dot(AtNm1)

            tot = term1 + term2 - term3 - term4 - term5

            if len(ddt.shape) == 3:
                sum_trace = np.einsum('ij,ji...m->', tot, ddt)
            else:
                sum_trace = np.einsum('ij,ji->...', tot, ddt)

            fisher_matrix[i][ii] = -sum_trace*2

            fisher_matrix[ii][i] = -sum_trace*2
            '''
            WARNING : Factor 2 for +transpose term, see Clara's thesis
            '''

            if i == 11 and ii == 2:
                sterm1 = term_fisher_debug(term1, ddt)
                sterm2 = term_fisher_debug(term2, ddt)
                sterm3 = term_fisher_debug(term3, ddt)
                sterm4 = term_fisher_debug(term4, ddt)
                sterm5 = term_fisher_debug(term5, ddt)
                # sterm5_clara = term_fisher_debug(term5_clara, ddt)

                IPython.embed()

    return fisher_matrix


def spectral_first_deriv(angle_array, ddt, model, prior=False,
                         fixed_miscal_angles=[], miscal_priors=[],
                         birefringence=False, spectral_index=False, Ninvfactor=1,
                         minimize=False, params=None):
    from residuals import get_diff_list
    # IPython.embed()
    if spectral_index:
        model.evaluate_mixing_matrix([angle_array[-2], 20, angle_array[-1]])
    else:
        model.mixing_matrix = np.array([[1, 0], [0, 1]]*len(model.frequencies))
    model.miscal_angles = angle_array[:model.miscal_angles.shape[0]]
    model.bir_angle = 0

    model.get_miscalibration_angle_matrix()
    model.get_bir_matrix()
    if not spectral_index:
        model.bir_matrix = model.bir_matrix[:2, :2]
    model.get_projection_op()
    diff_list = get_diff_list(model, params)

    AtNm1A = model.mix_effectiv.T.dot(model.inv_noise).dot(model.mix_effectiv)
    invAtNm1A = np.linalg.inv(AtNm1A)
    Nm1A_invAtNm1A = model.inv_noise.dot(model.mix_effectiv).dot(invAtNm1A)
    param_num = len(diff_list)
    deriv_vector = np.empty(param_num)
    for i in range(param_num):
        A_i = effectiv_diff_mixing_matrix_new(i, diff_list, model, params)
        AitP = A_i.T.dot(model.projection)
        term = Nm1A_invAtNm1A.dot(AitP)
        deriv = np.sum(np.trace(term.dot(ddt)))

        if prior:
            gaussian_prior_deriv = 0
            if np.any(miscal_priors[:, 2] == i):
                j = np.where(miscal_priors[:, 2] == i)[0][0]
                gaussian_prior_deriv += (angle_array[int(miscal_priors[j, 2])] -
                                         miscal_priors[j, 0]) / (miscal_priors[j, 1]**2)
            deriv_vector[i] = deriv - gaussian_prior_deriv

    # if not 0.5 <= angle_array[-2] < 2.5 or not -5 <= angle_array[-1] <= -1:
    #     print('bla jac')
    #     an_array_1 = np.where(angle_array[-2] > 2.5, -np.inf, angle_array[-2])
    #     an_array_1 = np.where(angle_array[-2] < 0.5, np.inf, angle_array[-2])
    #     an_array_2 = np.where(angle_array[-1] > -1, -np.inf, angle_array[-1])
    #     an_array_2 = np.where(angle_array[-1] < -5, np.inf, angle_array[-1])
    #     angle_array[-2] = an_array_1
    #     angle_array[-1] = an_array_2
    #
    # if np.any(np.array(angle_array[:-2]) < (-0.5*np.pi)) or np.any(np.array(angle_array[:-2]) > (0.5*np.pi)):
    #     print('blou jac neg')
    #     an_array1 = np.where(angle_array[:-2] > 0.5*np.pi, -np.inf, angle_array[:-2])
    #     an_array2 = np.where(angle_array[:-2] < -0.5*np.pi, np.inf, an_array1)
    #     angle_array[:-2] = an_array2

    if minimize:
        return -deriv_vector
    else:
        return deriv_vector


def term_fisher_debug(term, ddt):
    dot = term.dot(ddt)
    sum_trace = np.sum(np.trace(dot))
    return sum_trace


def main():
    true_miscal_angles = np.arange(0.0, 0.5, 0.5/6)*u.rad
    # true_miscal_angles = np.array([0]*6)*u.rad

    prior_precision = (1*u.arcmin).to(u.rad).value
    nside = 128

    nsteps = 11000
    discard = 1000
    birefringence = 1
    spectral = 1
    prior_indices = [0, 6]
    prior_flag = True
    sampled_miscal_freq = 6

    wMPI2 = 0
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
    """
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

    if wMPI2:
        comm = MPI.COMM_WORLD
        mpi_rank = MPI.COMM_WORLD.Get_rank()
        nsim = comm.Get_size()
        print(mpi_rank, nsim)
        prior_indices = [mpi_rank % 6, (mpi_rank % 6)+1]
        if mpi_rank//6 == 0:
            birefringence = 1
            spectral = 1

        if mpi_rank//6 == 1:
            birefringence = 0
            spectral = 1

        if mpi_rank//6 == 2:
            birefringence = 1
            spectral = 0
        print('prior_indices = ', prior_indices)
        print('birefringence = ', birefringence)
        print('spectral = ', spectral)
    """

    data, model = pix.data_and_model_quick(miscal_angles_array=true_miscal_angles, bir_angle=0.1*u.rad,
                                           frequencies_array=V3.so_V3_SA_bands(),
                                           frequencies_by_instrument_array=[1, 1, 1, 1, 1, 1], nside=nside)
    # data.get_mask(path='/home/baptiste/BBPipe')
    IPython.embed()
    path_BB_local = '/home/baptiste/BBPipe'
    path_BB_NERSC = '/global/homes/j/jost/BBPipe'
    path_BB = path_BB_local

    mask_ = hp.read_map(path_BB + "/test_mapbased_param/mask_04000.fits")
    mask = hp.ud_grade(mask_, nside)
    d = data.data
    ddt = np.einsum('ik...,...kj->ijk', d, d.T)
    ddtPnoise = (ddt+model.noise_covariance[..., np.newaxis])
    ddtPnoise_masked_cleaned = []  # np.array([p for p in ddtPnoise_masked if np.all(p !=0)])
    for i in range(len(mask)):
        if mask[i] == 1:
            ddtPnoise_masked_cleaned.append(ddtPnoise[:, :, i])
    ddtPnoise_masked_cleaned = np.array(ddtPnoise_masked_cleaned).T

    diff_list = diff_miscal_matrix(model)
    diff_list.append(diff_bir_matrix(model))
    if not model.fix_temp:
        mix_diff_Bd_QU, mix_diff_Td_QU, mix_diff_Bs_QU = diff_mixing_matrix(model)
    else:
        mix_diff_Bd_QU, mix_diff_Bs_QU = diff_mixing_matrix(model)

    diff_list.append(mix_diff_Bd_QU)
    diff_list.append(mix_diff_Bs_QU)

    diff_diff_list = diff_diff_miscal_matrix(model)
    diff_diff_list.append(diff_diff_bir_matrix(model))
    if not model.fix_temp:
        mix_diff_diff_Bd_QU, mix_diff_diff_Td_QU, mix_diff_diff_Bs_QU = diff_diff_mixing_matrix(
            model)
    else:
        mix_diff_diff_Bd_QU, mix_diff_diff_Bs_QU = diff_diff_mixing_matrix(model)

    diff_diff_list.append(mix_diff_diff_Bd_QU)
    diff_diff_list.append(mix_diff_diff_Bs_QU)
    start = time.time()
    fisher_matrix = fisher(ddtPnoise_masked_cleaned, model, diff_list, diff_diff_list)
    print('time fisher = ', time.time() - start)

    fisher_prior = copy.deepcopy(fisher_matrix)
    for i in range(prior_indices[0], prior_indices[-1]):
        print(i)
        fisher_prior[i, i] += 1/prior_precision**2
    inv_fisher_prior = np.linalg.inv(fisher_prior)
    sqrt_inv_fisher_prior = np.sqrt(inv_fisher_prior)
    # fisher_matrix = copy.deepcopy(fisher_prior)

    path_NERSC = '/global/homes/j/jost/these/pixel_based_analysis/results_and_data/run02032021//'
    path_local = './test11k/'
    path = path_local

    file_name, file_name_raw = pix.get_file_name_sample(
        sampled_miscal_freq, nsteps, discard,
        sampled_birefringence=birefringence, prior=prior_flag,
        prior_precision=prior_precision,
        prior_index=prior_indices, spectral_index=spectral, nside=nside)
    print(file_name)

    np.save(path+'fisher_'+file_name[:-4], fisher_matrix)
    np.save(path+'fisher_prior_'+file_name[:-4], fisher_prior)
    np.save(path+'inv_fisher_prior_'+file_name[:-4], inv_fisher_prior)
    np.save(path+'sqrt_inv_fisher_prior_'+file_name[:-4], sqrt_inv_fisher_prior)

    exit()

    values, vectors = np.linalg.eig(fisher_matrix)
    params = np.append(true_miscal_angles.value, [data.bir_angle.value,
                                                  data.spectral_indices[0], data.spectral_indices[-1]])
    normalised_params = np.empty(len(params))
    conversion_factors = np.empty(len(params))
    index_list = []
    index_list2 = []

    for i in range(len(params)):
        test_vector = np.abs(vectors[i])
        index_2 = np.argmax(test_vector)
        index_list2.append(index_2)

        for l in index_list:
            test_vector[l] = -np.inf
        index = np.argmax(test_vector)
        index_list.append(index)

        if np.abs(values[i]) <= 1:
            conv = 1
        else:
            conv = np.abs(values[i])

        normalised_params[index] = params[index]/np.sqrt(conv)
        conversion_factors[index] = np.sqrt(conv)
    print('index list = ', index_list)
    print('(normalised_params * conversion_factors - params)/params=',
          (normalised_params * conversion_factors - params)/params)
    conversion_matrix = np.einsum('i,j', conversion_factors, conversion_factors.T)

    data_norm, model_norm = pix.data_and_model_quick(miscal_angles_array=normalised_params[:6]*u.rad,
                                                     bir_angle=normalised_params[7]*u.rad,
                                                     frequencies_array=V3.so_V3_SA_bands(),
                                                     frequencies_by_instrument_array=[1, 1, 1, 1, 1, 1], nside=nside,
                                                     spectral_params=[normalised_params[7], 20, normalised_params[8]])
    # d = data_norm.data
    # ddt = np.einsum('ik...,...kj->ijk', d, d.T)
    # ddtPnoise = (ddt+model_norm.noise_covariance[..., np.newaxis])
    # ddtPnoise_masked_cleaned_norm = []  # np.array([p for p in ddtPnoise_masked if np.all(p !=0)])
    # for i in range(len(mask)):
    #     if mask[i] == 1:
    #         ddtPnoise_masked_cleaned_norm.append(ddtPnoise[:, :, i])
    # ddtPnoise_masked_cleaned_norm = np.array(ddtPnoise_masked_cleaned_norm).T

    diff_list_norm = diff_miscal_matrix(model_norm)
    diff_list_norm.append(diff_bir_matrix(model_norm))
    mix_diff_Bd_QU_norm, mix_diff_Td_QU_norm, mix_diff_Bs_QU_norm = diff_mixing_matrix(model_norm)
    diff_list_norm.append(mix_diff_Bd_QU_norm)
    diff_list_norm.append(mix_diff_Bs_QU_norm)

    diff_diff_list_norm = diff_diff_miscal_matrix(model_norm)
    diff_diff_list_norm.append(diff_diff_bir_matrix(model_norm))
    mix_diff_diff_Bd_QU_norm, mix_diff_diff_Td_QU_norm, mix_diff_diff_Bs_QU_norm = diff_diff_mixing_matrix(
        model_norm)
    diff_diff_list_norm.append(mix_diff_diff_Bd_QU_norm)
    diff_diff_list_norm.append(mix_diff_diff_Bs_QU_norm)
    start = time.time()
    fisher_matrix_norm = fisher(ddtPnoise_masked_cleaned, model_norm,
                                diff_list_norm, diff_diff_list_norm)
    print('time fisher = ', time.time() - start)

    inv_fisher = np.linalg.inv(fisher_matrix)
    inv_fisher_norm = np.linalg.inv(fisher_matrix_norm)
    conversion_inv_fisher_norm = np.einsum('ij,ij->ij', inv_fisher_norm, conversion_matrix)

    fisher_norm_prior = copy.deepcopy(fisher_matrix_norm)
    for i in range(6):
        fisher_norm_prior[i, i] -= 1/((prior_precision**2)/conversion_factors[i]**2)
    inv_fisher_norm_prior = np.linalg.inv(fisher_norm_prior)
    conversion_inv_fisher_norm_prior = np.einsum(
        'ij,ij->ij', inv_fisher_norm_prior, conversion_matrix)

    # values_p, vectors_p = np.linalg.eig(fisher_matrix)
    # index_listp = []
    # index_list2p = []
    #
    # for i in range(len(params)):
    #     test_vector = np.abs(vectors_p[i])
    #     index_2p = np.argmax(test_vector)
    #     index_list2p.append(index_2p)
    #
    #     for l in index_list:
    #         test_vector[l] = -np.inf
    #     indexp = np.argmax(test_vector)
    #     index_listp.append(indexp)
    #
    #     # normalised_params[indexp] = params[indexp]/np.sqrt(np.abs(values_p[i]))
    #     # conversion_factors[index] = np.sqrt(np.abs(values[i]))
    # print('index list = ', index_list)
    IPython.embed()

    labels = [r'$\alpha_{{{}}}$'.format(i) for i in model.frequencies]
    labels.append(r'B')
    labels.append(r'$\beta_d$')
    labels.append(r'$\beta_s$')
    ticks = [0, 1, 2, 3, 4, 5, 6, 7, 8]

    IPython.embed()

    plt.matshow(fisher_matrix)
    plt.colorbar()
    plt.xticks(ticks, labels=labels)
    plt.yticks(ticks, labels=labels)
    plt.show()

    plt.matshow(inv_fisher)
    plt.colorbar()
    plt.xticks(ticks, labels=labels)
    plt.yticks(ticks, labels=labels)
    plt.show()

    sqrt_inv_fisher = np.sqrt(np.abs(np.linalg.inv(fisher_matrix)))
    plt.matshow(sqrt_inv_fisher)
    plt.colorbar()
    plt.show()
    sqrt_inv_fisher[6, 6] = np.nan
    plt.matshow(sqrt_inv_fisher)
    plt.colorbar()
    plt.show()

    fisher_no_bir = np.delete(fisher_matrix, 6, 0)
    fisher_no_bir = np.delete(fisher_no_bir, 6, 1)
    plt.matshow(fisher_no_bir)
    plt.colorbar()
    plt.show()

    plt.matshow(np.linalg.inv(fisher_no_bir))
    plt.colorbar()
    ticks2 = [0, 1, 2, 3, 4, 5, 6, 7]
    labels2 = [r'$\alpha_{{{}}}$'.format(i) for i in model.frequencies]
    labels2.append(r'$\beta_d$')
    labels2.append(r'$\beta_s$')
    plt.xticks(ticks2, labels=labels2)
    plt.yticks(ticks2, labels=labels2)
    plt.show()

    plt.matshow(np.sqrt(np.abs(np.linalg.inv(fisher_no_bir))))
    plt.colorbar()
    plt.show()
    IPython.embed()


if __name__ == "__main__":
    main()
