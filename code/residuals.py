from fgbuster.observation_helpers import get_instrument
import time
import IPython
import numpy as np
from astropy import units as u
import bjlib.V3calc as V3
import fisher_pixel as fshp
import pixel_based_angle_estimation as pix
import copy
import healpy as hp
import matplotlib.pyplot as plt
import bjlib.lib_project as lib
from scipy.linalg import logm
from pixel_based_angle_estimation import get_chi_squared_local
from scipy.optimize import minimize
import numdifftools as nd
from scipy.integrate import simps
import bjlib.class_faraday as cf
import emcee
from fisher_pixel import fisher_new
from scipy.linalg import block_diag


def get_diff_list(model, params):
    params_array = np.array(params)

    if sum(params_array == 'miscal') != 0:
        diff_list = fshp.diff_miscal_matrix(model)

    if sum(params_array == 'birefringence'):
        diff_list.append(fshp.diff_bir_matrix(model))

    if sum(params_array == 'spectral') != 0:
        if not model.fix_temp:
            mix_diff_Bd_QU, mix_diff_Td_QU, mix_diff_Bs_QU = fshp.diff_mixing_matrix(model)
        else:
            mix_diff_Bd_QU, mix_diff_Bs_QU = fshp.diff_mixing_matrix(model)

        diff_list.append(mix_diff_Bd_QU)
        diff_list.append(mix_diff_Bs_QU)
        if sum(params_array == 'spectral') != 2:
            print('WARNING : Only 2 spectral parameters supported in get_diff')
    return diff_list


def get_diff_diff_list(model, params):
    params_array = np.array(params)
    if sum(params_array == 'miscal') != 0:
        diff_diff_list = fshp.diff_diff_miscal_matrix(model)
    if sum(params_array == 'birefringence'):
        diff_diff_list.append(fshp.diff_diff_bir_matrix(model))
    if sum(params_array == 'spectral') != 0:
        if not model.fix_temp:
            mix_diff_diff_Bd_QU, mix_diff_diff_Td_QU, mix_diff_diff_Bs_QU = fshp.diff_diff_mixing_matrix(
                model)
        else:
            mix_diff_diff_Bd_QU, mix_diff_diff_Bs_QU = fshp.diff_diff_mixing_matrix(
                model)
        diff_diff_list.append(mix_diff_diff_Bd_QU)
        diff_diff_list.append(mix_diff_diff_Bs_QU)
        if sum(params_array == 'spectral') != 2:
            print('WARNING : Only 2 spectral parameters supported in get_diff_diff')
    return diff_diff_list


def get_W(model):
    AtNm1A = model.mix_effectiv.T.dot(model.inv_noise).dot(model.mix_effectiv)
    invAtNm1A = np.linalg.inv(AtNm1A)

    return invAtNm1A.dot(model.mix_effectiv.T).dot(model.inv_noise)


def get_diff_W(model, diff_list, params, W=None, invAtNm1A=None, return_elements=False):
    if invAtNm1A is None:
        AtNm1A = model.mix_effectiv.T.dot(model.inv_noise).dot(model.mix_effectiv)
        invAtNm1A = np.linalg.inv(AtNm1A)
    if W is None:
        AtN = model.mix_effectiv.T.dot(model.inv_noise)
        W = invAtNm1A.dot(AtN)

    diff_W = []
    invANAdBpBt_list = []
    A_i_list = []
    AitNm1_list = []
    term1_list = []
    term2_list = []

    # param_num = np.shape(diff_list)[0]
    param_num = len(diff_list)
    for i in range(param_num):
        A_i = fshp.effectiv_diff_mixing_matrix_new(i, diff_list, model, params)
        B = A_i.T.dot(model.inv_noise).dot(model.mix_effectiv)
        BpBt = B+B.T
        invANABBT = invAtNm1A.dot(BpBt)
        AitNm1 = A_i.T.dot(model.inv_noise)

        term1 = invANABBT.dot(W)
        term2 = invAtNm1A.dot(AitNm1)
        diff_W_i = -term1+term2

        diff_W.append(diff_W_i)
        invANAdBpBt_list.append(invANABBT)
        A_i_list.append(A_i)
        AitNm1_list.append(AitNm1)
        term1_list.append(term1)
        term2_list.append(term2)
    if return_elements:
        return diff_W, invANAdBpBt_list, A_i_list, AitNm1_list, term1_list, term2_list

    return diff_W


def get_diff_diff_W(model, diff_list, diff_diff_list, params, W, invANAdBpBt_list, A_i_list,
                    AitNm1_list, term1_list, term2_list, invAtNm1A=None):
    if invAtNm1A is None:
        AtNm1A = model.mix_effectiv.T.dot(model.inv_noise).dot(model.mix_effectiv)
        invAtNm1A = np.linalg.inv(AtNm1A)
    diff_diff_W_ = []
    param_num = len(diff_list)

    for i in range(param_num):
        for ii in range(param_num):
            A_i_ii = fshp.effectiv_doublediff_mixing_matrix_new(
                i, ii, diff_list, diff_diff_list, model, params)
            D = A_i_ii.T.dot(model.inv_noise).dot(model.mix_effectiv) + \
                AitNm1_list[i].dot(A_i_list[ii])
            DpDt = D+D.T

            term1 = invANAdBpBt_list[ii].dot(invANAdBpBt_list[i]).dot(W)
            term2 = invANAdBpBt_list[i].dot(invANAdBpBt_list[ii]).dot(W)
            term3 = invANAdBpBt_list[ii].dot(term2_list[i])
            term4 = invANAdBpBt_list[i].dot(term2_list[ii])
            term5 = invAtNm1A.dot(DpDt).dot(W)
            term6 = invAtNm1A.dot(A_i_ii.T).dot(model.inv_noise)
            tot = term1+term2 - term3 - term4 - term5 + term6
            diff_diff_W_.append(tot)
    shape_diff_diff = np.shape(diff_diff_W_)
    # IPython.embed()
    diff_diff_W = np.reshape(
        diff_diff_W_, [param_num, param_num, shape_diff_diff[1], shape_diff_diff[2]])
    return diff_diff_W


def get_Wfg_maps(WQ, WU, freq_maps):
    if not freq_maps.shape[0] == WQ.shape[-1] == WU.shape[-1]:
        print('frequency number doesn\'t align')
        return 0
    if freq_maps.shape[1] == 3:
        print('I/Q/U given in freq_maps. Only Q/U needed')
        # freq_maps = freq_maps_input[:, 1:, :]
        Q_index = 1
        U_index = 2

    else:
        Q_index = 0
        U_index = 1
    WxFg_Q = WQ.dot(freq_maps[:, Q_index])
    WxFg_U = WU.dot(freq_maps[:, U_index])

    return WxFg_Q, WxFg_U


def get_ys_alms(y_Q, y_U, lmax):
    if len(y_Q.shape) == 2:
        # print('if ysalm')
        # shape : db; pixel
        dB_maps = np.zeros((y_Q.shape[0], 3, y_Q.shape[-1]))
        dB_maps[:, 1] = y_Q
        dB_maps[:, 2] = y_U

        alms = np.array([hp.map2alm(stokes_maps, lmax=lmax, iter=3) for stokes_maps in dB_maps])
    else:
        # print('else ysalm')
        stokes_maps = np.zeros((3, y_Q.shape[0]))
        stokes_maps[1] = y_Q
        stokes_maps[2] = y_U

        alms = hp.map2alm(stokes_maps, lmax=lmax, iter=3)

    return alms


def get_ys_Cls(X_alms, Y_alms, lmax, fsky=1):
    if len(X_alms.shape) == 3 or len(Y_alms.shape) == 3:
        if len(X_alms.shape) == len(Y_alms.shape):
            # print('if yCl')
            Cls = np.zeros((X_alms.shape[0], X_alms.shape[0], 3, lmax+1))
            # Cls2 = np.zeros((X_alms.shape[0], X_alms.shape[0], 3, lmax+1))

            for dBx in range(X_alms.shape[0]):
                for dBy in range(Y_alms.shape[0]):
                    spectra_ = hp.alm2cl(X_alms[dBx], Y_alms[dBy], lmax=lmax, nspec=6)
                    spectra = np.array([spectra_[1], spectra_[2], spectra_[4]])
                    Cls[dBx, dBy] = spectra
                    # spectra2_ = hp.alm2cl(Y_alms[dBy], X_alms[dBx], lmax=lmax, nspec=5)
                    # spectra2 = np.array([spectra2_[1], spectra2_[2], spectra2_[4]])
                    # Cls2[dBx, dBy] = spectra2
                    # return Cls, Cls2
        else:
            # print('else1 yCl')

            if len(X_alms.shape) < len(Y_alms.shape):
                print('ERROR: second argument cannot have larger dimension than first')
                return None

            Cls = np.zeros((X_alms.shape[0], 3, lmax+1))
            for dBx in range(X_alms.shape[0]):
                spectra_ = hp.alm2cl(X_alms[dBx], Y_alms, lmax=lmax, nspec=5)
                spectra = np.array([spectra_[1], spectra_[2], spectra_[4]])
                Cls[dBx] = spectra

    else:
        # print('else2yCl')

        spectra_ = hp.alm2cl(X_alms, Y_alms, lmax=lmax, nspec=5)
        spectra = np.array([spectra_[1], spectra_[2], spectra_[4]])
        Cls = spectra

    return Cls / fsky


def get_residuals(model, fg_freq_maps, sigma, lmin, lmax, fsky, params, cmb_spectra=None,
                  true_A_cmb=None, pivot_angle_index=None, reshape_fg_ys=None):
    '''============================computing Ws============================'''
    start_Ws = time.time()
    diff_list = get_diff_list(model, params)
    diff_diff_list = get_diff_diff_list(model, params)

    W = get_W(model)
    diff_W, invANAdBpBt_list, A_i_list, AitNm1_list, term1_list, term2_list = \
        get_diff_W(
            model, diff_list, params, W=W, invAtNm1A=None, return_elements=True)
    diff_diff_W = get_diff_diff_W(model, diff_list, diff_diff_list, params, W, invANAdBpBt_list, A_i_list,
                                  AitNm1_list, term1_list, term2_list, invAtNm1A=None)

    diff_W = np.array(diff_W)
    if pivot_angle_index is not None:
        diff_W = np.delete(diff_W, pivot_angle_index, 0)
        diff_diff_W = np.delete(np.delete(diff_diff_W, pivot_angle_index, 0), pivot_angle_index, 1)
    # print('time Ws = ', time.time() - start_Ws)
    '''========================Computing ys and alms========================'''
    # print('WARNING FSKY !!!!')
    # IPython.embed()

    if reshape_fg_ys is None:
        start_ys = time.time()

        y_Q = W[0].dot(fg_freq_maps)
        y_U = W[1].dot(fg_freq_maps)
        y_alms = get_ys_alms(y_Q=y_Q, y_U=y_U, lmax=lmax)
        del y_Q, y_U

        start_YQU = time.time()
        Y_Q = diff_W[:, 0].dot(fg_freq_maps)
        Y_U = diff_W[:, 1].dot(fg_freq_maps)
        print('time YQU = ', time.time()-start_YQU)
        start_Yalms = time.time()
        Y_alms = get_ys_alms(y_Q=Y_Q, y_U=Y_U, lmax=lmax)
        print('time Yalms = ', time.time()-start_Yalms)

        del Y_Q, Y_U

        V_Q = np.einsum('ij,ij...->...', sigma, diff_diff_W[:, :, 0])
        V_U = np.einsum('ij,ij...->...', sigma, diff_diff_W[:, :, 1])
        z_Q = V_Q.dot(fg_freq_maps)
        z_U = V_U.dot(fg_freq_maps)
        z_alms = get_ys_alms(y_Q=z_Q, y_U=z_U, lmax=lmax)
        del V_Q, V_U, z_Q, z_U
        print('time ys = ', time.time() - start_ys)

    if reshape_fg_ys is not None:
        V = np.einsum('ij,ij...->...', sigma, diff_diff_W[:, :, :2])
        start_ys = time.time()

        Wfg_ys = W[:2].dot(reshape_fg_ys)
        Wfg_ys_TEB = np.zeros([3, Wfg_ys.shape[-1]], dtype='complex')
        Wfg_ys_TEB[1:] = Wfg_ys

        Wfg_Ys = diff_W[:, :2].dot(reshape_fg_ys)
        Wfg_Ys_TEB = np.zeros([diff_W.shape[0], 3, Wfg_Ys.shape[-1]], dtype='complex')
        Wfg_Ys_TEB[:, 1:] = Wfg_Ys

        zfg = V.dot(reshape_fg_ys)
        zfg_TEB = np.zeros([3, zfg.shape[-1]], dtype='complex')
        zfg_TEB[1:] = zfg
        # print('time test ys = ', time.time() - start_ys)

    # '''===========================computing alms==========================='''

    '''===========================computing Cls==========================='''
    start_cl = time.time()
    Cl = {}
    if reshape_fg_ys is None:
        Cl['yy'] = get_ys_Cls(y_alms, y_alms, lmax, fsky)[:, lmin:]
        Cl['YY'] = get_ys_Cls(Y_alms, Y_alms, lmax, fsky)[:, :, :, lmin:]
        Cl['yz'] = get_ys_Cls(y_alms, z_alms, lmax, fsky)[:, lmin:]
        Cl['zy'] = get_ys_Cls(z_alms, y_alms, lmax, fsky)[:, lmin:]

        Cl['Yy'] = get_ys_Cls(Y_alms, y_alms, lmax, fsky)[:, :, lmin:]
        # Cl['yY'] = get_ys_Cls(y_alms, Y_alms, lmax, fsky)
        print('DANGER need to compute yY and zY in residuals ! EB and BE asymmetry otherwise')
        Cl['Yz'] = get_ys_Cls(Y_alms, z_alms, lmax, fsky)[:, :, lmin:]  # attention à checker
        # Cl['zY'] = get_ys_Cls(z_alms, Y_alms, lmax, fsky)  # attention à checker
        del z_alms, Y_alms, y_alms

    else:
        Cl['yy'] = get_ys_Cls(Wfg_ys_TEB, Wfg_ys_TEB, lmax, fsky)[:, lmin:]
        Cl['YY'] = get_ys_Cls(Wfg_Ys_TEB, Wfg_Ys_TEB, lmax, fsky)[:, :, :, lmin:]
        Cl['yz'] = get_ys_Cls(Wfg_ys_TEB, zfg_TEB, lmax, fsky)[:, lmin:]
        Cl['zy'] = get_ys_Cls(zfg_TEB, Wfg_ys_TEB, lmax, fsky)[:, lmin:]

        Cl['Yy'] = get_ys_Cls(Wfg_Ys_TEB, Wfg_ys_TEB, lmax, fsky)[:, :, lmin:]
        # print('DANGER need to compute yY and zY in residuals ! EB and BE asymmetry otherwise')
        Cl['Yz'] = get_ys_Cls(Wfg_Ys_TEB, zfg_TEB, lmax, fsky)[:, :, lmin:]  # attention à checker
    # print('time Cls = ', time.time() - start_cl)

    if cmb_spectra is not None:
        start_clmatrixcmb = time.time()
        Cl_cmb = {}
        # IPython.embed()
        # cl_substract = 0
        # WA_cmb = W[:2].dot(true_A_cmb) - (1-cl_substract)*np.identity(2)
        WA_cmb = W[:2].dot(true_A_cmb)
        # WA_cmb_box = W[:2].dot(true_A_cmb) - np.ones([2, 2])
        W_dB_cmb = diff_W[:, :2, :].dot(true_A_cmb)
        W_dBdB_cmb = diff_diff_W[:, :, :2, :].dot(true_A_cmb)
        # print('WARNING INPUT BIREFRINGENCE IS NOT TAKEN INTO ACCOUNT')
        Cl_matrix = np.zeros((2, 2, len(cmb_spectra[0])))
        Cl_matrix[0, 0] = cmb_spectra[1]
        Cl_matrix[1, 1] = cmb_spectra[2]
        if len(cmb_spectra) == 6:
            Cl_matrix[1, 0] = cmb_spectra[4]
            Cl_matrix[0, 1] = cmb_spectra[4]
        # print('time cl matrix cmb = ', time.time() - start_clmatrixcmb)

        # yy_cmb1 = np.einsum('ij,jkl,km->iml', WA_cmb.T, Cl_matrix, WA_cmb)
        # yy_cmb2 = np.einsum('ij,jkl,km->iml', WA_cmb, Cl_matrix, WA_cmb.T)
        # yy_cmb = yy_cmb2 - Cl_matrix * cl_substract
        start_ycmb = time.time()
        yy_cmb = np.einsum('ij,jkl,km->iml', WA_cmb, Cl_matrix, WA_cmb.T)

        YY_cmb_matrix = np.zeros([sigma.shape[0], sigma.shape[0], Cl_matrix.shape[0],
                                  Cl_matrix.shape[1], Cl_matrix.shape[2]])
        # YY_cmb_matrix2 = np.zeros([sigma.shape[0], sigma.shape[0], Cl_matrix.shape[0],
        #                            Cl_matrix.shape[1], Cl_matrix.shape[2]])
        for i in range(sigma.shape[0]):
            for ii in range(sigma.shape[0]):
                YY_cmb_matrix[i, ii] = np.einsum(
                    'ij,jkl,km->iml', W_dB_cmb[i].T, Cl_matrix, W_dB_cmb[ii])
                # YY_cmb_matrix2[i, ii] = np.einsum(
                #     'ij,jkl,km->iml', W_dB_cmb[i], Cl_matrix, W_dB_cmb[ii].T)
        # print('time yy YY cmb = ', time.time() - start_ycmb)

        start_Vcmb = time.time()
        V_cmb = np.einsum('ij,ij...->...', sigma, W_dBdB_cmb[:, :])
        # V2 = np.einsum('ij,ij...->...', sigma, diff_diff_W[:, :, :2, :])
        # yz_cmb = np.einsum('ji,jkl,km->iml', V_cmb, Cl_matrix, WA_cmb)
        # zy_cmb = np.einsum('ji,jkl,km->iml', WA_cmb, Cl_matrix, V_cmb)
        # yz_cmb1 = []
        # zy_cmb1 = []
        yz_cmb = []
        zy_cmb = []
        for l in range(Cl_matrix.shape[-1]):
            # yz_cmb1.append(V_cmb.T.dot(Cl_matrix[:, :, l]).dot(WA_cmb))
            # zy_cmb1.append(WA_cmb.T.dot(Cl_matrix[:, :, l]).dot(V_cmb))

            yz_cmb.append(V_cmb.dot(Cl_matrix[:, :, l]).dot(WA_cmb.T))
            zy_cmb.append(WA_cmb.dot(Cl_matrix[:, :, l]).dot(V_cmb.T))
        # yz_cmb1 = np.array(yz_cmb1).T
        # zy_cmb1 = np.array(zy_cmb1).T
        yz_cmb = np.array(yz_cmb).T
        zy_cmb = np.array(zy_cmb).T
        # yz_cmb = yz_cmb2
        # zy_cmb = zy_cmb2
        # print('time V, yz, zy cmb = ', time.time() - start_Vcmb)
        # IPython.embed()
        start_Yy = time.time()
        Yy_cmb = np.zeros([sigma.shape[0], Cl_matrix.shape[0],
                           Cl_matrix.shape[1], Cl_matrix.shape[2]])
        # yY_cmb = np.zeros([sigma.shape[0], Cl_matrix.shape[0],
        #                    Cl_matrix.shape[1], Cl_matrix.shape[2]])
        for i in range(sigma.shape[0]):
            Yy_cmb[i] = np.einsum('ij,jkl,km->iml', W_dB_cmb[i].T, Cl_matrix, WA_cmb)
            # yY_cmb[i] = np.einsum('ij,jkl,km->iml', WA_cmb, Cl_matrix, W_dB_cmb[i].T)
        # print('time Yy cmb = ', time.time() - start_Yy)

        start_Yz = time.time()
        Yz_cmb = np.zeros([sigma.shape[0], Cl_matrix.shape[0],
                           Cl_matrix.shape[1], Cl_matrix.shape[2]])
        # zY_cmb = np.zeros([sigma.shape[0], Cl_matrix.shape[0],
        #                    Cl_matrix.shape[1], Cl_matrix.shape[2]])
        for i in range(sigma.shape[0]):
            Yz_cmb[i] = np.einsum('ij,jkl,km->iml', W_dB_cmb[i].T, Cl_matrix, V_cmb)
            # zY_cmb[i] = np.einsum('ij,jkl,km->iml', V_cmb, Cl_matrix, W_dB_cmb[i].T)
        # print('time Yz cmb = ', time.time() - start_Yz)

        Cl_cmb['yy'] = yy_cmb[:, :, lmin:lmax+1]
        Cl_cmb['YY'] = YY_cmb_matrix[:, :, :, :, lmin:lmax+1]
        Cl_cmb['yz'] = yz_cmb[:, :, lmin:lmax+1]
        Cl_cmb['zy'] = zy_cmb[:, :, lmin:lmax+1]

        Cl_cmb['Yy'] = Yy_cmb[:, :, :, lmin:lmax+1]
        # Cl_cmb['yY'] = yY_cmb

        Cl_cmb['Yz'] = Yz_cmb[:, :, :, lmin:lmax+1]
        # Cl_cmb['zY'] = zY_cmb
        # IPython.embed()

    # '''===========================computing alms==========================='''
    #
    # y_alms = get_ys_alms(y_Q=y_Q, y_U=y_U, lmax=lmax)
    # Y_alms = get_ys_alms(y_Q=Y_Q, y_U=Y_U, lmax=lmax)
    # z_alms = get_ys_alms(y_Q=z_Q, y_U=z_U, lmax=lmax)
    # '''===========================computing Cls==========================='''
    # Cl = {}
    # Cl['yy'] = get_ys_Cls(y_alms, y_alms, lmax, fsky)[:, lmin:]
    # Cl['YY'] = get_ys_Cls(Y_alms, Y_alms, lmax, fsky)[:, :, :, lmin:]
    # Cl['yz'] = get_ys_Cls(y_alms, z_alms, lmax, fsky)[:, lmin:]
    # Cl['zy'] = get_ys_Cls(z_alms, y_alms, lmax, fsky)[:, lmin:]
    #
    # Cl['Yy'] = get_ys_Cls(Y_alms, y_alms, lmax, fsky)[:, :, lmin:]
    # # Cl['yY'] = get_ys_Cls(y_alms, Y_alms, lmax, fsky)
    # print('DANGER need to compute yY and zY in residuals ! EB and BE asymmetry otherwise')
    # Cl['Yz'] = get_ys_Cls(Y_alms, z_alms, lmax, fsky)[:, :, lmin:]  # attention à checker
    # # Cl['zY'] = get_ys_Cls(z_alms, Y_alms, lmax, fsky)  # attention à checker

    if cmb_spectra is not None:
        def Cl_adder(Cl_fg, Cl_cmb):
            Cl_out = {}
            key_counter = 0
            for key in Cl_cmb.keys():
                if len(Cl_cmb[key].shape) == 3:
                    key_counter += 1
                    Cl_out[key] = Cl_fg[key] + [Cl_cmb[key][0, 0], Cl_cmb[key]
                                                [1, 1], Cl_cmb[key][0, 1]]
                elif len(Cl_cmb[key].shape) == 5:
                    key_counter += 1
                    temp_cl = np.zeros(Cl_fg[key].shape)
                    temp_cl[:, :, 0] = Cl_cmb[key][:, :, 0, 0]
                    temp_cl[:, :, 1] = Cl_cmb[key][:, :, 1, 1]
                    temp_cl[:, :, 2] = Cl_cmb[key][:, :, 0, 1]
                    Cl_out[key] = Cl_fg[key] + temp_cl
                elif len(Cl_cmb[key].shape) == 4:
                    key_counter += 1
                    temp_cl = np.zeros(Cl_fg[key].shape)
                    temp_cl[:, 0] = Cl_cmb[key][:, 0, 0]
                    temp_cl[:, 1] = Cl_cmb[key][:, 1, 1]
                    temp_cl[:, 2] = Cl_cmb[key][:, 0, 1]
                    Cl_out[key] = Cl_fg[key] + temp_cl
            return Cl_out

        def Cl_adder_matrix(Cl_fg, Cl_cmb, lmin, lmax):
            Cl_out = {}
            key_counter = 0
            for key in Cl_cmb.keys():
                if len(Cl_cmb[key].shape) == 3:
                    key_counter += 1
                    Cl_out[key] = Cl_cmb[key][:, :] + \
                        [[Cl_fg[key][0], Cl_fg[key][2]],
                            [Cl_fg[key][2], Cl_fg[key][1]]]

                elif len(Cl_cmb[key].shape) == 5:
                    key_counter += 1
                    shape_temp = np.array(Cl_cmb[key].shape)
                    shape_temp[-1] = lmax-lmin
                    temp_matrix = np.zeros(shape_temp)
                    temp_matrix = np.array([[Cl_fg[key][:, :, 0], Cl_fg[key][:, :, 2]], [
                        Cl_fg[key][:, :, 2], Cl_fg[key][:, :, 1]]])
                    temp_matrix_shape = np.einsum('ijklm->klijm', temp_matrix)
                    Cl_out[key] = Cl_cmb[key][:, :, :, :] + temp_matrix_shape
                elif len(Cl_cmb[key].shape) == 4:
                    key_counter += 1
                    shape_temp = np.array(Cl_cmb[key].shape)
                    shape_temp[-1] = lmax-lmin
                    temp_matrix = np.zeros(shape_temp)
                    temp_matrix = np.array([[Cl_fg[key][:,  0], Cl_fg[key][:, 2]], [
                        Cl_fg[key][:, 2], Cl_fg[key][:,  1]]])
                    temp_matrix_shape = np.einsum('ijkm->kijm', temp_matrix)
                    Cl_out[key] = Cl_cmb[key][:, :, :] + temp_matrix_shape
            # print('key_counter = ', key_counter)
            return Cl_out
        start_adder = time.time()
        Cl_out = Cl_adder(Cl, Cl_cmb)
        Cl_out_matrix = Cl_adder_matrix(Cl, Cl_cmb, lmin, lmax)
        ell = np.arange(lmin, lmax+1)
        # print('time adder = ', time.time() - start_adder)
        # IPython.embed()
        # Cl_out2 = Cl_adder(Cl, Cl_cmb2)
    '''========================computing residuals========================'''
    if cmb_spectra is not None:
        # IPython.embed()
        start_last = time.time()
        stat = np.einsum('ij,ij...->...', sigma, Cl_out['YY'])
        bias = Cl_out['yy'] + Cl_out['yz'] + Cl_out['zy']
        var = stat**2 + 2 * np.einsum('i...,ij,j...->...',
                                      Cl_out['Yy'], sigma, Cl_out['Yy'])
        # print('time last = ', time.time() - start_last)
        return stat, bias, var, Cl, Cl_cmb, Cl_out_matrix, ell, W[:2], diff_W[:, :2, :], diff_diff_W[:, :, :2, :]

    else:
        stat = np.einsum('ij,ij...->...', sigma, Cl['YY'])
        bias = Cl['yy'] + Cl['yz'] + Cl['zy']
        var = stat**2 + 2 * np.einsum('i...,ij,j...->...', Cl['Yy'], sigma, Cl['Yy'])
        return stat, bias, var, Cl


def cosmo_likelihood(Cl_model, Cl_data, Cl_residuals, sigma_res, ell, fsky):

    # S16, Appendix C
    # Cl_model = Cl_fid['BlBl'] * Alens + Cl_fid['BuBu'] * r_ + Cl_noise
    inv_model = np.linalg.inv(Cl_model.T).T
    dof = (2 * ell + 1) * fsky
    dof_over_Cl = dof * inv_model
    IPython.embed()
    # Eq. C3
    U_1 = np.einsum('ijkml,mnl->ijkn', Cl_residuals['YY'], dof_over_Cl)
    sigma_U = np.zeros(U_1.shape)
    sigma_inv = np.linalg.inv(sigma_res)
    for i in range(2):
        for j in range(2):
            sigma_U[:, :, i, j] = sigma_inv
    U = np.linalg.inv(sigma_U + U_1)

    # Eq. C9
    tr_SigmaYY = np.einsum('ij,jimnl->mnl', sigma_res, Cl_residuals['YY'])
    first_row = np.sum(dof_over_Cl * (
        Cl_data * (1 - np.einsum('ijkm, jikml -> kml', U, Cl_residuals['YY']) * inv_model)
        + tr_SigmaYY))
    second_row = - np.einsum(
        'l, m, ij, mjk, kf, lfi',
        dof_over_Cl, dof_over_Cl, U, Cl_residuals['YY'], sigma_res, Cl_residuals['YY'])
    second_row_EE = - np.einsum(
        'l, m, ij, mjk, kf, lfi',
        dof_over_Cl[0, 0], dof_over_Cl[0, 0], U, Cl_residuals['YY'], sigma_res, Cl_residuals['YY'])
    Cl_Cl = np.einsum('ijl,jkm->iklm', dof_over_Cl, dof_over_Cl)
    trace_UYSY = np.einsum('ijab,jlbck,lm,mncdp->inadkp', U,
                           Cl_residuals['YY'], sigma_res, Cl_residuals['YY'])
    # 'abl, cdm, ijd, jkbem, kf, fidal',
    second_row = - np.einsum(
        'wxl, ywm, ijxb, jkbcl,kn,nicym',
        dof_over_Cl, dof_over_Cl, U, Cl_residuals['YY'], sigma_res, Cl_residuals['YY'])
    trCinvC = first_row + second_row

    # Eq. C10
    first_row = np.sum(dof_over_Cl * (Cl_xF['yy'] + 2 * Cl_xF['yz']))
    # Cyclicity + traspose of scalar + grouping terms -> trace becomes
    # Yy_ell^T U (Yy + 2 Yz)_ell'
    trace = np.einsum('li, ij, mj -> lm',
                      Cl_xF['Yy'], U, Cl_xF['Yy'] + 2 * Cl_xF['Yz'])
    second_row = - _utmv(dof_over_Cl, trace, dof_over_Cl)
    trECinvC = first_row + second_row

    # Eq. C12
    logdetC = np.sum(dof * np.log(Cl_model)) - np.log(np.linalg.det(U))

    # Cl_hat = Cl_obs + tr_SigmaYY

    # Bringing things together
    return trCinvC + trECinvC + logdetC


def cosmo_likelihood_nodeprojection(Cl_model_total, Cl_data, ell, fsky):

    inv_model = np.linalg.inv(Cl_model_total.T).T
    dof = (2 * ell + 1) * fsky
    dof_over_Cl = dof * inv_model

    first_term_ell = np.einsum('ijl,jkl->ikl', dof_over_Cl, Cl_data)

    first_term = np.sum(np.trace(first_term_ell))

    logdetC = np.sum(dof*np.log(np.abs(np.linalg.det(Cl_model_total.T))))

    return first_term + logdetC


def likelihood_exploration(cosmo_params, Cl_fid, Cl_data, Cl_noise_matrix, dWA_cmb, sigma_spectral, WA_cmb, VA_cmb, ell, fsky, minimisation=True):
    r = cosmo_params[0]

    beta = cosmo_params[1]*u.rad
    # print('r', r, ' beta ', beta)
    Cl_cmb_model = np.zeros([4, Cl_fid['EE'].shape[0]])
    Cl_cmb_model[1] = copy.deepcopy(Cl_fid['EE'])
    Cl_cmb_model[2] = copy.deepcopy(Cl_fid['BlBl'])*1 + copy.deepcopy(Cl_fid['BuBu']) * r

    Cl_cmb_rot = lib.cl_rotation(Cl_cmb_model.T, beta).T

    Cl_cmb_rot_matrix = np.zeros([2, 2, Cl_cmb_rot.shape[-1]])
    Cl_cmb_rot_matrix[0, 0] = copy.deepcopy(Cl_cmb_rot[1])
    Cl_cmb_rot_matrix[1, 1] = copy.deepcopy(Cl_cmb_rot[2])
    Cl_cmb_rot_matrix[1, 0] = copy.deepcopy(Cl_cmb_rot[4])
    Cl_cmb_rot_matrix[0, 1] = copy.deepcopy(Cl_cmb_rot[4])

    WACAW = np.einsum('ij,jkl,km->iml', WA_cmb, Cl_cmb_rot_matrix, WA_cmb.T)
    WACAV = np.einsum('ij,jkl,km->iml', WA_cmb, Cl_cmb_rot_matrix, VA_cmb.T)
    VACAW = np.einsum('ij,jkl,km->iml', VA_cmb, Cl_cmb_rot_matrix, WA_cmb.T)
    # Cl_model_total = Cl_cmb_rot_matrix + Cl_noise_matrix + tr_SigmaYY

    YY_cmb_matrix = np.zeros([sigma_spectral.shape[0], sigma_spectral.shape[0], Cl_cmb_rot_matrix.shape[0],
                              Cl_cmb_rot_matrix.shape[1], Cl_cmb_rot_matrix.shape[2]])
    for i in range(sigma_spectral.shape[0]):
        for ii in range(sigma_spectral.shape[0]):
            YY_cmb_matrix[i, ii] = np.einsum(
                'ij,jkl,km->iml', dWA_cmb[i].T, Cl_cmb_rot_matrix, dWA_cmb[ii])

    tr_SigmaYY = np.einsum('ij,jimnl->mnl', sigma_spectral, YY_cmb_matrix)

    Cl_model_total = WACAW + Cl_noise_matrix + tr_SigmaYY + VACAW + WACAV
    # Cl_model_total = WACAW + Cl_noise_matrix + tr_SigmaYY + VACAW + WACAV

    likelihood = cosmo_likelihood_nodeprojection(
        Cl_model_total, Cl_data, ell, fsky)
    # print('-2logL = ', likelihood)
    if not minimisation:
        return -likelihood/2
    else:
        return likelihood/2


def get_model_WACAW_WACAVeheh(cosmo_params, Cl_fid, dWA_cmb, sigma_spectral,
                              WA_cmb, VA_cmb):
    r = cosmo_params[0]
    beta = cosmo_params[1]*u.rad

    Cl_cmb_model = np.zeros([4, Cl_fid['EE'].shape[0]])
    Cl_cmb_model[1] = copy.deepcopy(Cl_fid['EE'])
    Cl_cmb_model[2] = copy.deepcopy(Cl_fid['BlBl'])*1 + copy.deepcopy(Cl_fid['BuBu']) * r

    Cl_cmb_rot = lib.cl_rotation(Cl_cmb_model.T, beta).T

    Cl_cmb_rot_matrix = np.zeros([2, 2, Cl_cmb_rot.shape[-1]])
    Cl_cmb_rot_matrix[0, 0] = copy.deepcopy(Cl_cmb_rot[1])
    Cl_cmb_rot_matrix[1, 1] = copy.deepcopy(Cl_cmb_rot[2])
    Cl_cmb_rot_matrix[1, 0] = copy.deepcopy(Cl_cmb_rot[4])
    Cl_cmb_rot_matrix[0, 1] = copy.deepcopy(Cl_cmb_rot[4])

    WACAW = np.einsum('ij,jkl,km->iml', WA_cmb, Cl_cmb_rot_matrix, WA_cmb.T)
    WACAV = np.einsum('ij,jkl,km->iml', WA_cmb, Cl_cmb_rot_matrix, VA_cmb.T)
    VACAW = np.einsum('ij,jkl,km->iml', VA_cmb, Cl_cmb_rot_matrix, WA_cmb.T)

    YY_cmb_matrix = np.zeros([sigma_spectral.shape[0], sigma_spectral.shape[0],
                              Cl_cmb_rot_matrix.shape[0], Cl_cmb_rot_matrix.shape[1],
                              Cl_cmb_rot_matrix.shape[2]])
    for i in range(sigma_spectral.shape[0]):
        for ii in range(sigma_spectral.shape[0]):
            YY_cmb_matrix[i, ii] = np.einsum(
                'ij,jkl,km->iml', dWA_cmb[i].T, Cl_cmb_rot_matrix, dWA_cmb[ii])

    tr_SigmaYY = np.einsum('ij,jimnl->mnl', sigma_spectral, YY_cmb_matrix)

    return WACAW, VACAW, WACAV, tr_SigmaYY, YY_cmb_matrix


def simple_like(cosmo_params, Cl_fid, Cl_data, Cl_noise_matrix, WA_cmb, ell, fsky):
    r = cosmo_params[0]

    beta = cosmo_params[1]*u.rad

    Cl_cmb_model = np.zeros([4, Cl_fid['EE'].shape[0]])
    Cl_cmb_model[1] = copy.deepcopy(Cl_fid['EE'])
    Cl_cmb_model[2] = copy.deepcopy(Cl_fid['BlBl'])*1 + copy.deepcopy(Cl_fid['BuBu']) * r

    Cl_cmb_rot = lib.cl_rotation(Cl_cmb_model.T, beta).T

    Cl_cmb_rot_matrix = np.zeros([2, 2, Cl_cmb_rot.shape[-1]])
    Cl_cmb_rot_matrix[0, 0] = copy.deepcopy(Cl_cmb_rot[1])
    Cl_cmb_rot_matrix[1, 1] = copy.deepcopy(Cl_cmb_rot[2])
    Cl_cmb_rot_matrix[1, 0] = copy.deepcopy(Cl_cmb_rot[4])
    Cl_cmb_rot_matrix[0, 1] = copy.deepcopy(Cl_cmb_rot[4])

    WACAW = np.einsum('ij,jkl,km->iml', WA_cmb, Cl_cmb_rot_matrix, WA_cmb.T)

    Cl_model_total = WACAW + Cl_noise_matrix

    likelihood = cosmo_likelihood_nodeprojection(
        Cl_model_total, Cl_data, ell, fsky)
    return likelihood/2


def jac_simple_like(cosmo_params, Cl_fid, Cl_data, Cl_noise_matrix, WA_cmb, ell, fsky):
    r = cosmo_params[0]

    beta = cosmo_params[1]*u.rad

    Cl_cmb_model = np.zeros([4, Cl_fid['EE'].shape[0]])
    Cl_cmb_model[1] = copy.deepcopy(Cl_fid['EE'])
    Cl_cmb_model[2] = copy.deepcopy(Cl_fid['BlBl'])*1 + copy.deepcopy(Cl_fid['BuBu']) * r
    Cl_cmb_rot = lib.cl_rotation(Cl_cmb_model.T, beta).T

    Cl_cmb_rot_matrix = np.zeros([2, 2, Cl_cmb_rot.shape[-1]])
    Cl_cmb_rot_matrix[0, 0] = copy.deepcopy(Cl_cmb_rot[1])
    Cl_cmb_rot_matrix[1, 1] = copy.deepcopy(Cl_cmb_rot[2])
    Cl_cmb_rot_matrix[1, 0] = copy.deepcopy(Cl_cmb_rot[4])
    Cl_cmb_rot_matrix[0, 1] = copy.deepcopy(Cl_cmb_rot[4])
    WACAW = np.einsum('ij,jkl,km->iml', WA_cmb, Cl_cmb_rot_matrix, WA_cmb.T)

    Cl_cmb_dr = np.zeros([4, Cl_fid['EE'].shape[0]])
    Cl_cmb_dr[2] = copy.deepcopy(Cl_fid['BuBu'])

    Cl_cmb_rot = lib.cl_rotation(Cl_cmb_model.T, beta).T
    Cl_cmb_dr_rot = lib.cl_rotation(Cl_cmb_dr.T, beta).T
    Cl_cmb_derivrot = lib.cl_rotation_derivative(Cl_cmb_model.T, beta).T

    dCldr = np.zeros([2, 2, Cl_cmb_rot.shape[-1]])
    dCldr[0, 0] = copy.deepcopy(Cl_cmb_dr_rot[1])
    dCldr[1, 1] = copy.deepcopy(Cl_cmb_dr_rot[2])
    dCldr[1, 0] = copy.deepcopy(Cl_cmb_dr_rot[4])
    dCldr[0, 1] = copy.deepcopy(Cl_cmb_dr_rot[4])
    WACAWdr = np.einsum('ij,jkl,km->iml', WA_cmb, dCldr, WA_cmb.T)

    dCldbeta = np.zeros([2, 2, Cl_cmb_rot.shape[-1]])
    dCldbeta[0, 0] = copy.deepcopy(Cl_cmb_derivrot[1])
    dCldbeta[1, 1] = copy.deepcopy(Cl_cmb_derivrot[2])
    dCldbeta[1, 0] = copy.deepcopy(Cl_cmb_derivrot[4])
    dCldbeta[0, 1] = copy.deepcopy(Cl_cmb_derivrot[4])
    WACAWdb = np.einsum('ij,jkl,km->iml', WA_cmb, dCldbeta, WA_cmb.T)

    WdCldr = WACAWdr
    WdCldbeta = WACAWdb
    deriv_list = [WdCldr, WdCldbeta]

    Cl_model_total = WACAW + Cl_noise_matrix
    inv_model = np.linalg.inv(Cl_model_total.T).T
    dof = (2 * ell + 1) * fsky
    dof_over_Cl = dof * inv_model

    d_Cl = []
    for i in range(2):
        first_term_1 = np.einsum('ijl,jkl->ikl', dof_over_Cl, deriv_list[i])
        first_term_2 = np.einsum('ijl,jkl->ikl', first_term_1, inv_model)
        first_term_3 = np.einsum('ijl,jkl->ikl', first_term_2, Cl_data)
        first_term = np.sum(np.trace(first_term_3))

        second_term = np.sum(np.trace(copy.deepcopy(first_term_1)))

        d_Cl.append(-first_term + second_term)
    d_Cl = np.array(d_Cl)
    return d_Cl/2


def run_double_MC(p0, ddtPN, model_skm, nsteps, prior=False,
                  fixed_miscal_angles=[], miscal_priors=[],
                  birefringence=False, spectral_index=False,
                  lmin=30, lmax=300, fsky=0.1,
                  sensitiviy_mode=1, one_over_f_mode=1, INSTRU='SAT',
                  cmb_spectra=None, true_A_cmb=None,
                  Cl_fid=None, method_cosmo=None):
    nwalkers = 16
    ndim = 8
    discard_num = 1
    cosmo_list = []
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, double_MC_function, args=[
            ddtPN, model_skm, prior,
            fixed_miscal_angles, miscal_priors,
            birefringence, spectral_index,
            lmin, lmax, fsky,
            sensitiviy_mode, one_over_f_mode, INSTRU,
            cmb_spectra, true_A_cmb,
            Cl_fid, method_cosmo, cosmo_list])
    sampler.reset()
    sampler.run_mcmc(p0, nsteps, progress=True)
    flat_samples = sampler.get_chain(discard=discard_num, flat=True)
    cosmo_list = np.array(cosmo_list)
    return flat_samples, cosmo_list


def double_MC_function(angle_array, ddtPN, model_skm, prior=False,
                       fixed_miscal_angles=[], miscal_priors=[],
                       birefringence=False, spectral_index=False,
                       lmin=30, lmax=300, fsky=0.1,
                       sensitiviy_mode=1, one_over_f_mode=1, INSTRU='SAT',
                       cmb_spectra=None, true_A_cmb=None,
                       Cl_fid=None, method_cosmo=None, cosmo_list=[],
                       Ninvfactor=1,
                       minimize_flag=False, params=None):
    chi_squared = get_chi_squared_local(angle_array, ddtPN, model_skm, prior,
                                        fixed_miscal_angles, miscal_priors,
                                        birefringence, spectral_index, Ninvfactor,
                                        minimize_flag, params)

    Cl_noise, ell_noise = get_noise_Cl(
        model_skm.mix_effectiv, lmax+1, fsky,
        sensitiviy_mode, one_over_f_mode,
        instrument=INSTRU, onefreqtest=1-spectral_index)
    Cl_noise = Cl_noise[lmin-2:]
    ell_noise = ell_noise[lmin-2:]
    Cl_noise_matrix = np.zeros([2, 2, Cl_noise.shape[0]])
    Cl_noise_matrix[0, 0] = Cl_noise
    Cl_noise_matrix[1, 1] = Cl_noise

    W = get_W(model_skm)
    WA_cmb_true = W[:2].dot(true_A_cmb)
    WA_cmb_est = W[:2].dot(model_skm.mix_effectiv[:, :2])

    Cl_matrix = np.zeros((2, 2, len(cmb_spectra[0])))
    Cl_matrix[0, 0] = cmb_spectra[1]
    Cl_matrix[1, 1] = cmb_spectra[2]
    if len(cmb_spectra) == 6:
        Cl_matrix[1, 0] = cmb_spectra[4]
        Cl_matrix[0, 1] = cmb_spectra[4]
    WACAW_true = np.einsum('ij,jkl,km->iml', WA_cmb_true,
                           Cl_matrix[..., lmin:lmax+1], WA_cmb_true.T)
    Cl_data = WACAW_true + Cl_noise_matrix

    ell = np.arange(lmin, lmax+1)

    bounds_cosmo = ((-0.01, 5), (-np.pi/8, np.pi/8))
    cosmo_array_start = np.random.uniform(np.array(bounds_cosmo)[:, 0],
                                          np.array(bounds_cosmo)[:, 1])

    res_cosmo = minimize(simple_like, cosmo_array_start, args=(
                         Cl_fid, Cl_data, Cl_noise_matrix, WA_cmb_est, ell, fsky),
                         bounds=bounds_cosmo, tol=1e-18, method=method_cosmo)

    cosmo_list.append(res_cosmo.x)

    return chi_squared


def jac_cosmo(cosmo_params, Cl_fid, Cl_data, Cl_noise_matrix, dWA_cmb, sigma_spectral, WA_cmb, VA_cmb, ell, fsky):
    r = cosmo_params[0]
    # IPython.embed()
    beta = cosmo_params[1]*u.rad

    Cl_cmb_model = np.zeros([4, Cl_fid['EE'].shape[0]])
    Cl_cmb_model[1] = copy.deepcopy(Cl_fid['EE'])
    Cl_cmb_model[2] = copy.deepcopy(Cl_fid['BlBl'])*1 + copy.deepcopy(Cl_fid['BuBu']) * r
    Cl_cmb_dr = np.zeros([4, Cl_fid['EE'].shape[0]])
    Cl_cmb_dr[2] = copy.deepcopy(Cl_fid['BuBu'])

    Cl_cmb_rot = lib.cl_rotation(Cl_cmb_model.T, beta).T
    Cl_cmb_dr_rot = lib.cl_rotation(Cl_cmb_dr.T, beta).T
    Cl_cmb_derivrot = lib.cl_rotation_derivative(Cl_cmb_model.T, beta).T

    Cl_cmb_rot_matrix = np.zeros([2, 2, Cl_cmb_rot.shape[-1]])
    Cl_cmb_rot_matrix[0, 0] = copy.deepcopy(Cl_cmb_rot[1])
    Cl_cmb_rot_matrix[1, 1] = copy.deepcopy(Cl_cmb_rot[2])
    Cl_cmb_rot_matrix[1, 0] = copy.deepcopy(Cl_cmb_rot[4])
    Cl_cmb_rot_matrix[0, 1] = copy.deepcopy(Cl_cmb_rot[4])

    WACAV = np.einsum('ij,jkl,km->iml', WA_cmb, Cl_cmb_rot_matrix, VA_cmb.T)
    VACAW = np.einsum('ij,jkl,km->iml', VA_cmb, Cl_cmb_rot_matrix, WA_cmb.T)
    WACAW = np.einsum('ij,jkl,km->iml', WA_cmb, Cl_cmb_rot_matrix, WA_cmb.T)

    dCldr = np.zeros([2, 2, Cl_cmb_rot.shape[-1]])
    dCldr[0, 0] = copy.deepcopy(Cl_cmb_dr_rot[1])
    dCldr[1, 1] = copy.deepcopy(Cl_cmb_dr_rot[2])
    dCldr[1, 0] = copy.deepcopy(Cl_cmb_dr_rot[4])
    dCldr[0, 1] = copy.deepcopy(Cl_cmb_dr_rot[4])
    WACAVdr = np.einsum('ij,jkl,km->iml', WA_cmb, dCldr, VA_cmb.T)
    VACAWdr = np.einsum('ij,jkl,km->iml', VA_cmb, dCldr, WA_cmb.T)
    WACAWdr = np.einsum('ij,jkl,km->iml', WA_cmb, dCldr, WA_cmb.T)

    dCldbeta = np.zeros([2, 2, Cl_cmb_rot.shape[-1]])
    dCldbeta[0, 0] = copy.deepcopy(Cl_cmb_derivrot[1])
    dCldbeta[1, 1] = copy.deepcopy(Cl_cmb_derivrot[2])
    dCldbeta[1, 0] = copy.deepcopy(Cl_cmb_derivrot[4])
    dCldbeta[0, 1] = copy.deepcopy(Cl_cmb_derivrot[4])
    WACAVdb = np.einsum('ij,jkl,km->iml', WA_cmb, dCldbeta, VA_cmb.T)
    VACAWdb = np.einsum('ij,jkl,km->iml', VA_cmb, dCldbeta, WA_cmb.T)
    WACAWdb = np.einsum('ij,jkl,km->iml', WA_cmb, dCldbeta, WA_cmb.T)
    # deriv_list = [dCldr, dCldbeta]

    YY_cmb_matrix = np.zeros([sigma_spectral.shape[0], sigma_spectral.shape[0], Cl_cmb_rot_matrix.shape[0],
                              Cl_cmb_rot_matrix.shape[1], Cl_cmb_rot_matrix.shape[2]])
    YY_cmb_matrixdr = np.zeros([sigma_spectral.shape[0], sigma_spectral.shape[0], Cl_cmb_rot_matrix.shape[0],
                                Cl_cmb_rot_matrix.shape[1], Cl_cmb_rot_matrix.shape[2]])
    YY_cmb_matrixdB = np.zeros([sigma_spectral.shape[0], sigma_spectral.shape[0], Cl_cmb_rot_matrix.shape[0],
                                Cl_cmb_rot_matrix.shape[1], Cl_cmb_rot_matrix.shape[2]])

    for i in range(sigma_spectral.shape[0]):
        for ii in range(sigma_spectral.shape[0]):
            YY_cmb_matrix[i, ii] = np.einsum(
                'ij,jkl,km->iml', dWA_cmb[i].T, Cl_cmb_rot_matrix, dWA_cmb[ii])
            YY_cmb_matrixdr[i, ii] = np.einsum(
                'ij,jkl,km->iml', dWA_cmb[i].T, dCldr, dWA_cmb[ii])
            YY_cmb_matrixdB[i, ii] = np.einsum(
                'ij,jkl,km->iml', dWA_cmb[i].T, dCldbeta, dWA_cmb[ii])

    tr_SigmaYY = np.einsum('ij,jimnl->mnl', sigma_spectral, YY_cmb_matrix)
    tr_SigmaYYdr = np.einsum('ij,jimnl->mnl', sigma_spectral, YY_cmb_matrixdr)
    tr_SigmaYYdB = np.einsum('ij,jimnl->mnl', sigma_spectral, YY_cmb_matrixdB)

    WdCldr = WACAWdr + VACAWdr + WACAVdr + tr_SigmaYYdr
    WdCldbeta = WACAWdb + VACAWdb + WACAVdb + tr_SigmaYYdB
    deriv_list = [WdCldr, WdCldbeta]

    Cl_model_total = WACAW + Cl_noise_matrix + tr_SigmaYY + VACAW + WACAV
    inv_model = np.linalg.inv(Cl_model_total.T).T
    dof = (2 * ell + 1) * fsky
    dof_over_Cl = dof * inv_model

    d_Cl = []
    for i in range(2):
        first_term_1 = np.einsum('ijl,jkl->ikl', dof_over_Cl, deriv_list[i])
        first_term_2 = np.einsum('ijl,jkl->ikl', first_term_1, inv_model)
        first_term_3 = np.einsum('ijl,jkl->ikl', first_term_2, Cl_data)
        first_term = np.sum(np.trace(first_term_3))

        second_term = np.sum(np.trace(copy.deepcopy(first_term_1)))

        d_Cl.append(-first_term + second_term)
    d_Cl = np.array(d_Cl)
    # d_Cl[0] *= -1
    print(d_Cl)
    return d_Cl/2


def constrained_cosmo(cosmo_params, Cl_fid, Cl_data, Cl_noise_matrix, dWA_cmb,
                      sigma_spectral, WA_cmb, VA_cmb, total_prior_matrix,
                      true_miscal_angles, pivot_angle_index, eval_angles,
                      ell, fsky, minimisation=True):

    r = cosmo_params[0]
    beta = cosmo_params[1]*u.rad
    # beta = 0*u.rad
    pivot = cosmo_params[2]*u.rad
    # pivot = true_miscal_angles[pivot_angle_index]
    # print('ahhh', r, beta, pivot)
    # import bjlib.lib_project as lib

    if not minimisation:
        if r > 5 or r < -0.01:
            # if r > 5 or r < -1e-6:
            print('bla1')
            return -np.inf
        elif beta.value < -np.pi/2 or beta.value > np.pi/2:
            print('bla2')
            return -np.inf
        elif pivot.value < -np.pi/2 or pivot.value > np.pi/2:
            print('bla3')
            return -np.inf

    new_angles = copy.deepcopy(eval_angles)
    diff_pivot = pivot.value - eval_angles[pivot_angle_index]

    new_angles += diff_pivot
    new_angles[pivot_angle_index] = pivot.value

    Cl_cmb_model = np.zeros([4, Cl_fid['EE'].shape[0]])
    Cl_cmb_model[1] = copy.deepcopy(Cl_fid['EE'])
    Cl_cmb_model[2] = copy.deepcopy(Cl_fid['BlBl'])*1 + copy.deepcopy(Cl_fid['BuBu']) * r

    Cl_cmb_rot = lib.cl_rotation(Cl_cmb_model.T, beta).T

    Cl_cmb_rot_matrix = np.zeros([2, 2, Cl_cmb_rot.shape[-1]])
    Cl_cmb_rot_matrix[0, 0] = copy.deepcopy(Cl_cmb_rot[1])
    Cl_cmb_rot_matrix[1, 1] = copy.deepcopy(Cl_cmb_rot[2])
    Cl_cmb_rot_matrix[1, 0] = copy.deepcopy(Cl_cmb_rot[4])
    Cl_cmb_rot_matrix[0, 1] = copy.deepcopy(Cl_cmb_rot[4])

    WACAW = np.einsum('ij,jkl,km->iml', WA_cmb, Cl_cmb_rot_matrix, WA_cmb.T)
    WACAV = np.einsum('ij,jkl,km->iml', WA_cmb, Cl_cmb_rot_matrix, VA_cmb.T)
    VACAW = np.einsum('ij,jkl,km->iml', VA_cmb, Cl_cmb_rot_matrix, WA_cmb.T)

    YY_cmb_matrix = np.zeros([sigma_spectral.shape[0], sigma_spectral.shape[0], Cl_cmb_rot_matrix.shape[0],
                              Cl_cmb_rot_matrix.shape[1], Cl_cmb_rot_matrix.shape[2]])
    for i in range(sigma_spectral.shape[0]):
        for ii in range(sigma_spectral.shape[0]):
            YY_cmb_matrix[i, ii] = np.einsum(
                'ij,jkl,km->iml', dWA_cmb[i].T, Cl_cmb_rot_matrix, dWA_cmb[ii])

    tr_SigmaYY = np.einsum('ij,jimnl->mnl', sigma_spectral, YY_cmb_matrix)

    Cl_model_total_ = WACAW + tr_SigmaYY + VACAW + WACAV  # + Cl_noise_matrix

    rot_pivot = np.array([[np.cos(2*diff_pivot), -np.sin(2*diff_pivot)],
                          [np.sin(2*diff_pivot), np.cos(2*diff_pivot)]])

    Cl_model_total = np.einsum('ij,jkl,km->iml', rot_pivot.T,
                               Cl_model_total_, rot_pivot) + Cl_noise_matrix
    Cl_data_rot = Cl_data

    # removing spectral indices rows/columns
    inv_sigma_miscal = np.linalg.inv(sigma_spectral[:-2, :-2])
    angle_relat = np.delete(eval_angles, pivot_angle_index)
    true_prior = np.delete(true_miscal_angles, pivot_angle_index).value
    # angle_relat_true = np.delete(true_miscal_angles, pivot_angle_index).value
    pivot_true = true_miscal_angles[pivot_angle_index].value

    prior_matrix = np.delete(
        np.delete(total_prior_matrix, pivot_angle_index, 0), pivot_angle_index, 1)[:-2, :-2]
    prior_element_pivot = total_prior_matrix[pivot_angle_index, pivot_angle_index]
    pivot = pivot.value
    # print('prior matrix in function', prior_matrix)
    inv_model = np.linalg.inv(Cl_model_total.T).T
    dof = (2 * ell + 1) * fsky
    dof_over_Cl = dof * inv_model

    first_term_ell = np.einsum('ijl,jkl->ikl', dof_over_Cl, Cl_data_rot)

    A = angle_relat + pivot - eval_angles[pivot_angle_index]

    # radek_jost_prior = ((pivot-pivot_true)**2) * prior_element_pivot +\
    #     A.T.dot(inv_sigma_miscal).dot(A) -\
    #     (A.T.dot(inv_sigma_miscal)+angle_relat.T.dot(prior_matrix)).dot(np.linalg.inv(inv_sigma_miscal +
    #                                                                                   prior_matrix)).dot(inv_sigma_miscal.dot(A)+prior_matrix.dot(angle_relat)) +\
    #     np.log(np.linalg.det(inv_sigma_miscal + prior_matrix)) + \
    #     angle_relat.T.dot(prior_matrix).dot(angle_relat)

    radek_jost_prior = ((pivot-pivot_true)**2) * prior_element_pivot +\
        A.T.dot(inv_sigma_miscal).dot(A) -\
        (A.T.dot(inv_sigma_miscal)+true_prior.T.dot(prior_matrix)).dot(np.linalg.inv(inv_sigma_miscal +
                                                                                     prior_matrix)).dot(inv_sigma_miscal.dot(A)+prior_matrix.dot(true_prior)) +\
        np.log(np.linalg.det(inv_sigma_miscal + prior_matrix)) + \
        true_prior.T.dot(prior_matrix).dot(true_prior)
    # radek_jost_prior +=

    first_term = np.sum(np.trace(first_term_ell))

    # logdetC = np.sum(dof*np.log(np.abs(np.linalg.det(Cl_model_total.T))))
    # print('arg: ', r, beta, pivot)
    # print('Cl_model_total[:,: 100]', Cl_model_total[:, : 100])
    # print('np.linalg.det(Cl_model_total.T)[100]', np.linalg.det(Cl_model_total.T)[100])
    logdetC = np.sum(dof*np.log(np.linalg.det(Cl_model_total.T)))
    # print('logdetC', logdetC)
    # print('(pivot-pivot_true)', (pivot-pivot_true))
    # print('A', A)
    # print('radek_jost_prior=', radek_jost_prior)
    # print('first_term', first_term)
    # print('logdetC', logdetC)
    likelihood = first_term + logdetC + radek_jost_prior
    # import math
    # if math.isnan(likelihood):
    #     IPython.embed()
    if not minimisation:
        return -likelihood/2
    else:
        return likelihood/2


def get_Cl_cmbBB(Alens=1., r=0.001, path_BB='.'):
    power_spectrum = hp.read_cl(
        path_BB + '/test_mapbased_param/Cls_Planck2018_lensed_scalar.fits')[:, :4000]
    if Alens != 1.:
        power_spectrum[2] *= Alens
    if r:
        power_spectrum += r *\
            hp.read_cl(
                path_BB + '/test_mapbased_param/Cls_Planck2018_unlensed_scalar_and_tensor_r1.fits')[:, :4000]
    return power_spectrum


def get_noise_Cl(A, lmax, fsky, sensitiviy_mode=2, one_over_f_mode=2, instrument='SAT',
                 onefreqtest=False, t_obs_years=5, SAC_yrs_LF=1, model_skm=None):
    if instrument == 'SAT':
        V3_results = V3.so_V3_SA_noise(sensitiviy_mode, one_over_f_mode,
                                       SAC_yrs_LF=SAC_yrs_LF, f_sky=fsky, ell_max=lmax,
                                       beam_corrected=True, t_obs_years=t_obs_years)
        noise_nl = np.repeat(V3_results[1], 2, 0)
        if onefreqtest:
            noise_nl = np.repeat(V3_results[1], 2, 0)[4:6]

        ell_noise = V3_results[0]

    elif instrument == 'Planck':
        instru_planck = get_instrument('planck_P')
        noise_lvl = instru_planck['sens_P']
        beam_rad = (instru_planck['beams']*u.arcmin).to(u.rad).value
        ell_noise = np.linspace(2, lmax-1, lmax-2, dtype=int)
        noise_nl = []
        for f in range(len(noise_lvl)):
            Bl = hp.gauss_beam(beam_rad[f], lmax=lmax-1)[2:]
            noise = (noise_lvl[f]*np.pi/60/180)**2 * np.ones(len(ell_noise))
            noise_nl.append(noise / (Bl**2))
        noise_nl = np.array(noise_nl)
        noise_nl = np.repeat(noise_nl, 2, 0)
        # noise_nl = (noise_lvl*np.pi/60/180)**2 * ell_noise
    if instrument == 'SAT+Planck':
        V3_results = V3.so_V3_SA_noise(sensitiviy_mode, one_over_f_mode,
                                       SAC_yrs_LF=SAC_yrs_LF, f_sky=fsky, ell_max=lmax,
                                       beam_corrected=True, t_obs_years=t_obs_years)
        ell_noise = V3_results[0]
        # noise_nl = np.repeat(V3_results[1], 2, 0)
        if SAC_yrs_LF == 0:  # removing two first frequencies as LF is off
            N_ell = V3_results[1][2:]
        else:
            N_ell = V3_results[1]
        noise_nl = np.repeat(N_ell, 2, 0)
        planck_noise_lvl = model_skm.planck_sens_p  # in uk-arcmin
        # as it is the sensitivity for polarisation already, no sqrt(2) factor needed
        planck_noise_lvl *= np.pi / 180 / 60  # from arcmin to rad
        # rescaling to match SO sky fraction
        f_sky_planck = 1  # with what fsky were the noise lvl computed ?
        planck_noise_lvl *= np.sqrt(fsky) / np.sqrt(f_sky_planck)

        planck_beam_rad = model_skm.planck_beams * u.arcmin.to(u.rad)
        planck_nl = []
        for f in range(len(planck_beam_rad)):
            planck_nl_nobeam = planck_noise_lvl[f]**2 * np.ones(len(ell_noise))
            Bl = hp.gauss_beam(planck_beam_rad[f], lmax=lmax-1)[2:]
            planck_nl.append(planck_nl_nobeam[f] / Bl**2)
        planck_nl = np.array(planck_nl)
        planck_nl = np.repeat(planck_nl, 2, 0)
        noise_nl = np.append(noise_nl, planck_nl, 0)

    elif instrument == 'LiteBIRD':
        print('WARNING NO 1/F FOR NOW !! (get_noise_Cl)')
        ell_noise = np.linspace(2, lmax-1, lmax-2, dtype=int)

        instrument_LB = np.load('data/instrument_LB_IMOv1.npy', allow_pickle=True).item()

        noise_lvl = np.array([instrument_LB[f]['P_sens'] for f in instrument_LB.keys()])
        beam_rad = np.array([instrument_LB[f]['beam']
                             for f in instrument_LB.keys()]) * u.arcmin.to(u.rad)
        noise_nl = []
        for f in range(len(noise_lvl)):
            Bl = hp.gauss_beam(beam_rad[f], lmax=lmax-1)[2:]
            noise = (noise_lvl[f]*np.pi/60/180)**2 * np.ones(len(ell_noise))
            noise_nl.append(noise / (Bl**2))
        noise_nl = np.array(noise_nl)
        noise_nl = np.repeat(noise_nl, 2, 0)

    else:
        print('Only SAT Planck and LiteBIRD supported for Cl_noise for now')

    nl_inv = 1/noise_nl
    AtNA = np.einsum('fi, fl, fj -> lij', A, nl_inv, A)
    inv_AtNA = np.linalg.inv(AtNA)
    noise_cl = inv_AtNA.swapaxes(-3, -1)[0, 0]
    # IPython.embed()
    return noise_cl, ell_noise


def multivariate_gaussian(pos, mu, Sigma):
    """Return the multivariate Gaussian distribution on array pos."""

    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2*np.pi)**n * Sigma_det)
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)

    return np.exp(-fac / 2) / N


def get_ranges(r_bounds, beta_bounds, r_true, beta_true, results_mini, nsteps_r, nsteps_beta):
    r_range = np.arange(r_bounds[0], r_bounds[1], (r_bounds[1]-r_bounds[0])/nsteps_r)
    beta_range = np.arange(beta_bounds[0], beta_bounds[1],
                           (beta_bounds[1]-beta_bounds[0])/nsteps_beta)
    if not np.any(r_range == r_true):
        r_range = np.sort(np.append(r_range, r_true))
    if not np.any(beta_range == beta_true):
        beta_range = np.sort(np.append(beta_range, beta_true))
    if not np.any(r_range == results_mini[0]):
        r_range = np.sort(np.append(r_range, results_mini[0]))
    if not np.any(beta_range == results_mini[1]):
        beta_range = np.sort(np.append(beta_range, results_mini[1]))

    r_index = np.where(r_range == r_true)[0][0]
    beta_index = np.where(beta_range == beta_true)[0][0]
    rmini_index = np.where(r_range == results_mini[0])[0][0]
    betamini_index = np.where(beta_range == results_mini[1])[0][0]
    return r_range, beta_range, r_index, beta_index, rmini_index, betamini_index


def get_chi2_grid(r_range, beta_range, Cl_fid, Cl_data,
                  Cl_noise_matrix, tr_SigmaYY, ell, fsky):
    likelihood_grid = []
    jac_grid_r = []
    jac_grid_beta = []
    for r in r_range:
        for beta in beta_range:
            likelihood_grid.append(likelihood_exploration([r, beta], Cl_fid, Cl_data,
                                                          Cl_noise_matrix, tr_SigmaYY, ell, fsky))
            jac = jac_cosmo([r, beta], Cl_fid, Cl_data,
                            Cl_noise_matrix, tr_SigmaYY, ell, fsky)
            jac_grid_r.append(jac[0])
            jac_grid_beta.append(jac[1])
    likelihood_grid = np.array(likelihood_grid)
    jac_grid_r = np.array(jac_grid_r)
    jac_grid_beta = np.array(jac_grid_beta)

    chi2_mesh = np.reshape(likelihood_grid, (-1, len(beta_range)))
    jac_mesh_r = np.reshape(jac_grid_r, (-1, len(beta_range)))
    jac_mesh_beta = np.reshape(jac_grid_beta, (-1, len(beta_range)))

    betaxx, ryy = np.meshgrid(beta_range, r_range)
    pos = np.empty(betaxx.shape + (2,))
    pos[:, :, 0] = betaxx
    pos[:, :, 1] = ryy
    return chi2_mesh, jac_mesh_r, jac_mesh_beta, betaxx, ryy, pos


def chi2_2_likenorm(chi2_mesh, pos, r_index, beta_index):
    r_range = pos[:, 0, 1]
    beta_range = pos[0, :, 0]

    like_mesh = np.exp((-chi2_mesh+np.min(chi2_mesh))/2)
    # norm_mesh = np.sum(like_mesh)

    if len(r_range) != len(beta_range):
        print('WARNING : approximate norm used in chi2_2_likenorm. Dimensions should be of same length')
        square_area = (r_range[1]-r_range[0])*(beta_range[1]-beta_range[0])
        total_area = square_area * len(r_range) * len(beta_range)
        norm_mesh = np.sum(like_mesh) / total_area
    else:
        norm_mesh = simps([simps(like_mesh_r, r_range) for like_mesh_r in like_mesh], beta_range)
    # IPython.embed()
    like_mesh /= norm_mesh

    like_r = np.exp((-chi2_mesh[:, beta_index]+np.min(chi2_mesh[:, beta_index]))/2)
    norm_r = simps(like_r, r_range)
    like_r /= norm_r

    like_beta = np.exp((-chi2_mesh[r_index, :]+np.min(chi2_mesh[r_index, :]))/2)
    norm_beta = simps(like_beta, beta_range)
    like_beta /= norm_beta

    return like_mesh, like_r, like_beta


def sigma2_int(like_mesh, like_r, like_beta, param_grid, r_index, beta_index):

    r_range = param_grid[:, 0, 1]
    beta_range = param_grid[0, :, 0]

    mean_r = simps(like_r*r_range, r_range)
    mean_rsquared = simps(like_r*(r_range**2), r_range)
    sigma2_r = mean_rsquared - mean_r**2

    mean_beta = simps(like_beta*beta_range, beta_range)
    mean_betasquared = simps(like_beta*(beta_range**2), beta_range)
    sigma2_beta = mean_betasquared - mean_beta**2
    if len(r_range) != len(beta_range):
        print('WARNING : approximate int used in sigma2_int(). Dimensions should be of same length')
        print('DO NOT use the result for sigma2_beta_r')
        mean_beta_r = 0
        for i in range(len(r_range)):
            for j in range(len(beta_range)):
                mean_beta_r += like_mesh[i, j]*r_range[i]*beta_range[j]
    else:
        mean_beta_r = simps([simps(like_mesh_r*r_range, r_range)
                             for like_mesh_r in like_mesh]*beta_range, beta_range)

    sigma2_beta_r = mean_beta_r - mean_r * mean_beta

    return sigma2_r, sigma2_beta, sigma2_beta_r


def multi_freq_get_sky_fg(sky, freq, dust_angle=None, synch_angle=None):
    freq_maps = []
    import pysm

    for f in freq:
        dust_freq_maps = sky.dust(f) * \
            pysm.convert_units('K_RJ', 'K_CMB', f)
        if dust_angle is not None:
            rotation_block = np.array(
                [[np.cos(2*dust_angle),  np.sin(2*dust_angle)],
                 [-np.sin(2*dust_angle), np.cos(2*dust_angle)]
                 ])
            dust_freq_maps[1:] = rotation_block.dot(dust_freq_maps[1:])

        sync_freq_maps = sky.synchrotron(f) *\
            pysm.convert_units('K_RJ', 'K_CMB', f)
        if synch_angle is not None:
            rotation_block = np.array(
                [[np.cos(2*synch_angle),  np.sin(2*synch_angle)],
                 [-np.sin(2*synch_angle), np.cos(2*synch_angle)]
                 ])
            sync_freq_maps[1:] = rotation_block.dot(sync_freq_maps[1:])

        fg_f = dust_freq_maps + sync_freq_maps
        freq_maps.append(fg_f[1:])
    freq_maps_array = np.array(freq_maps)
    shape = freq_maps_array.shape
    return np.reshape(freq_maps_array, (shape[0]*shape[1], shape[2]))


def get_fg_polar_mixing(freq_number, fg_mixing_angle_input=0):
    fg_mixing_angle = fg_mixing_angle_input * u.rad
    fg_mixing_angle = fg_mixing_angle

    rotation_block = np.array(
        [[np.cos(2*fg_mixing_angle),  np.sin(2*fg_mixing_angle)],
         [-np.sin(2*fg_mixing_angle), np.cos(2*fg_mixing_angle)]
         ])
    fg_rot_matrix = rotation_block

    for f in range(freq_number-1):
        fg_rot_matrix = block_diag(fg_rot_matrix, rotation_block)

    return fg_rot_matrix


def get_SFN(data, model_data, path_BB, S_cmb_name, spectral_flag=True, addnoise=1, fg_angle=None, dust_angle=None, synch_angle=None):
    S_cmb = np.load(S_cmb_name)

    ASAt = model_data.mix_effectiv[:, :2].dot(S_cmb).dot(model_data.mix_effectiv[:, :2].T)

    data.get_pysm_sky()
    if fg_angle is not None:
        # IPython.embed()
        print('angle pol fg = ', fg_angle, 'rad', fg_angle*u.rad.to(u.deg), 'deg')
        fg_rot_matrix = get_fg_polar_mixing(len(data.frequencies), fg_angle)
        fg_freq_maps_full = data.miscal_matrix.dot(fg_rot_matrix.dot(
            multi_freq_get_sky_fg(data.sky, data.frequencies, dust_angle, synch_angle)))
    else:
        fg_freq_maps_full = data.miscal_matrix.dot(
            multi_freq_get_sky_fg(data.sky, data.frequencies, dust_angle, synch_angle))

    ddt_fg = np.einsum('ik...,...kj->ijk', fg_freq_maps_full, fg_freq_maps_full.T)
    data.get_mask(path_BB)
    mask = data.mask
    mask[(mask != 0) * (mask != 1)] = 0

    ddt_fg *= mask
    fg_freq_maps = fg_freq_maps_full*mask
    del fg_freq_maps_full

    n_obspix = np.sum(mask == 1)
    del mask
    F = np.sum(ddt_fg, axis=-1)/n_obspix
    data_model = n_obspix*(F*spectral_flag + ASAt + addnoise*model_data.noise_covariance)
    return data_model, fg_freq_maps*spectral_flag, n_obspix


def main():
    INSTRU = 'SAT'
    if INSTRU == 'SAT':
        freq_number = 6
        fsky = 0.1
        lmax = 300
        lmin = 30

    if INSTRU == 'Planck':
        freq_number = 7
        fsky = 0.6
        lmax = 500
        lmin = 51
    nside = 512

    sky_model = 'c1s0d0'
    sensitiviy_mode = 1
    one_over_f_mode = 1
    A_lens_true = 1
    # sensitivity_mode
    #     0: threshold,
    #     1: baseline,
    #     2: goal
    # one_over_f_mode
    #     0: pessimistic
    #     1: optimistic
    #     2: none

    r_true = 0.01
    r_str = '_r0p01'
    beta_true = (0.35 * u.deg).to(u.rad)
    # beta_true = (0.0*u.deg).to(u.rad)

    true_miscal_angles = (np.arange(1, 5, 4 / freq_number)*u.deg).to(u.rad)  # [::-1]
    # true_miscal_angles = np.array([0]*freq_number)*u.rad
    # true_miscal_angles = (np.array([0.28] * freq_number)*u.deg).to(u.rad)
    # true_miscal_angles = np.array([0]*6)*u.rad
    # true_miscal_angles[0] = 0.4333*u.rad

    prior = True
    prior_indices = []
    if prior:
        # prior_indices = [5, 6]
        # prior_indices = [2, 3]
        prior_indices = [2, 3]
    prior_precision = (0.1 * u.deg).to(u.rad).value
    prior_str = '{:1.1e}rad'.format(prior_precision)

    save_path = '/home/baptiste/Documents/these/pixel_based_analysis/results_and_data/test/'

    path_BB_local = '/home/baptiste/BBPipe'
    path_BB_NERSC = '/global/homes/j/jost/BBPipe'
    path_BB = path_BB_local

    nsim = 1000

    initmodel_miscal = np.array([0]*freq_number)*u.rad
    freq_by_instru = [1]*freq_number

    angle_array_start_list = [0]*freq_number
    angle_array_start_list.append(2)
    angle_array_start_list.append(-2.5)
    angle_array_start = np.array(angle_array_start_list)

    params = ['miscal']*freq_number
    params.append('spectral')
    params.append('spectral')

    miscal_bounds = ((-np.pi/4, np.pi/4),)*freq_number
    spectral_bounds = ((0.5, 2.5), (-5, -1))
    bounds = miscal_bounds + spectral_bounds
    cosmo_params = [0.03, 0.04]
    bounds_cosmo = ((-0.01, 0.1), (-np.pi/4, np.pi/4))
    # bounds_cosmo = ((-0.01, 0.1), (None, None))
    # bounds_cosmo = ((None, None), (None, None))
    method_cosmo = 'L-BFGS-B'
    # jac_cosmo_min = None
    jac_cosmo_min = jac_cosmo

    '''====================================================================='''

    data, model_data = pix.data_and_model_quick(
        miscal_angles_array=true_miscal_angles, bir_angle=beta_true,
        frequencies_by_instrument_array=freq_by_instru, nside=nside,
        sky_model=sky_model, sensitiviy_mode=sensitiviy_mode,
        one_over_f_mode=one_over_f_mode, instrument=INSTRU)

    model = pix.get_model(
        miscal_angles_array=initmodel_miscal, bir_angle=beta_true,
        frequencies_by_instrument_array=freq_by_instru,
        nside=nside, spectral_params=[1.54, 20, -3],
        sky_model='c1s0d0', sensitiviy_mode=sensitiviy_mode,
        one_over_f_mode=one_over_f_mode, instrument=INSTRU)

    '''===========================getting data==========================='''

    S_cmb_name = 'S_cmb_n{}_s{}_r{:1}_b{:1.1e}'.format(nside, nsim, r_true, beta_true.value).replace(
        '.', 'p') + '.npy'
    print(S_cmb_name)
    S_cmb = np.load(S_cmb_name)

    ASAt = model_data.mix_effectiv[:, :2].dot(S_cmb).dot(model_data.mix_effectiv[:, :2].T)

    data.get_pysm_sky()
    fg_freq_maps_full = data.miscal_matrix.dot(multi_freq_get_sky_fg(data.sky, data.frequencies))
    # fg_freq_maps_full = data.miscal_matrix.dot(data.mixing_matrix)[
    #     :, 2:].dot(data.signal[2:])
    ddt_fg = np.einsum('ik...,...kj->ijk', fg_freq_maps_full, fg_freq_maps_full.T)
    # IPython.embed()
    data.get_mask(path_BB)
    mask = data.mask
    mask[(mask != 0) * (mask != 1)] = 0

    ddt_fg *= mask
    fg_freq_maps = fg_freq_maps_full*mask
    # IPython.embed()
    del fg_freq_maps_full

    n_obspix = np.sum(mask == 1)
    del mask
    F = np.sum(ddt_fg, axis=-1)/n_obspix
    data_model = n_obspix*(F + ASAt + model_data.noise_covariance)

    # angle_array_start = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 2, -2.5])
    # angle_array_start = np.array([0., 0., 0., 0., 0., 0., 1.59, -3])

    angle_prior = []
    if prior:
        for d in range(freq_number):
            angle_prior.append([true_miscal_angles.value[d], prior_precision, int(d)])
            # angle_prior.append([(1*u.deg).to(u.rad).value, prior_precision, int(d)])
        angle_prior = np.array(angle_prior[prior_indices[0]: prior_indices[-1]])

    alpha_grid = np.linspace(-2*prior_precision/30, 2*prior_precision/30, 1000)
    chi2_grid = []
    angle_array_grid = true_miscal_angles.value.tolist()
    angle_array_grid.append(1.54)
    angle_array_grid.append(-3)
    model.n_obspix = n_obspix
    '''
    for i in alpha_grid:
        angle_array_grid[2] = i
        chi2_grid.append(get_chi_squared_local(angle_array_grid, data_model,
                                               model, prior, [], angle_prior, False, True, 1, False))
    '''
    # np.save('grid_prior0p01deg_b0_r0_n128_max', chi2_grid)
    # results_min = minimize(get_chi_squared_local, angle_array_start, args=(
    #                        data_model, model, prior, [], angle_prior, False, True, 1, True),
    #                        bounds=((-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5), (0.5, 2.5), (-5, -1)), tol=1e-18)

    # results_min = minimize(get_chi_squared_local, angle_array_start, args=(
    #                        data_model, model, prior, [], angle_prior, False, True, 1, True),
    #                        tol=1e-18, method='L-BFGS-B', options={'maxiter': 1000},
    #                        bounds=((-np.pi/4, np.pi/4), (-np.pi/4, np.pi/4), (-np.pi/4, np.pi/4), (-np.pi/4, np.pi/4), (-np.pi/4, np.pi/4), (-np.pi/4, np.pi/4), (0.5, 2.5), (-5, -1)))

    # IPython.embed()
    # angle_array_start = np.random.uniform(np.array(bounds)[:, 0],
    #                                       np.array(bounds)[:, 1])
    results_min = minimize(get_chi_squared_local, angle_array_start, args=(
        data_model, model, prior, [], angle_prior, False, True, 1, True, params),
        tol=1e-18, options={'maxiter': 1000}, jac=fshp.spectral_first_deriv, method='L-BFGS-B',
        bounds=bounds)
    print(results_min)
    # results_min.x[:freq_number] = 0
    # results_min.x[-2] = 1.59
    # results_min.x[-1] = -3
    chi2_min = get_chi_squared_local(results_min.x, data_model, model,
                                     prior, [], angle_prior, False, True, 1, True)
    # chi2_minjac = get_chi_squared_local(results_minjac.x, data_model, model,
    #                                     prior, [], angle_prior, False, True, 1, True)
    param_true = copy.deepcopy(true_miscal_angles.value.tolist())  # [0]*freq_number
    param_true.append(1.54)
    param_true.append(-3)
    chi2_true = get_chi_squared_local(param_true, data_model, model,
                                      prior, [], angle_prior, False, True, 1, True)
    print('delta chi2 = ', chi2_min - chi2_true)
    print('delta alpha = ', true_miscal_angles.value[2] - results_min.x[2])
    # IPython.embed()
    # results_min = minimize(get_chi_squared_local, angle_array_start[1:], args=(
    #                        data_model, model, prior, [], angle_prior, False, True, 1, True),
    #                        bounds=((-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5), (0.5, 2.5), (-5, -1)), tol=1e-18)
    # results_min.x = np.append(0, results_min.x)
    # print('')
    # print(results_min.x)
    # for i in range(6):
    #     results_min.x[i] = (1*u.deg).to(u.rad).value
    # print(results_min.x)
    # print('')

    model_results = pix.get_model(
        results_min.x[: freq_number], bir_angle=beta_true,
        frequencies_by_instrument_array=freq_by_instru, nside=nside,
        spectral_params=[results_min.x[-2], 20, results_min.x[-1]],
        sky_model='c1s0d0', sensitiviy_mode=sensitiviy_mode,
        one_over_f_mode=one_over_f_mode, instrument=INSTRU)
    print('results - spectral_true = ', results_min.x[:freq_number] - true_miscal_angles.value)
    print('results - spectral_true = ', results_min.x[-2] - 1.54)
    print('results - spectral_true = ', results_min.x[-1] + 3)
    # IPython.embed()
    # np.save(save_path+'miscal_mini_pior2to4_'+prior_str+r_str, results_min.x)

    prior_matrix = np.zeros([len(params), len(params)])
    if prior:
        for i in range(prior_indices[0], prior_indices[-1]):
            prior_matrix[i, i] += 1/(prior_precision**2)

    ps_planck = copy.deepcopy(get_Cl_cmbBB(Alens=A_lens_true, r=r_true, path_BB=path_BB))
    spectra_true = lib.cl_rotation(ps_planck.T, beta_true).T

    diff_list_res = get_diff_list(model_results, params)
    diff_diff_list_res = get_diff_diff_list(model_results, params)
    fisher_matrix_spectral = fshp.fisher_new(data_model, model_results,
                                             diff_list_res, diff_diff_list_res, params)
    fisher_matrix_prior_spectral = fisher_matrix_spectral + prior_matrix
    sigma_spectral = np.linalg.inv(fisher_matrix_prior_spectral)

    start_residuals = time.time()

    stat, bias, var, Cl, Cl_cmb, Cl_residuals_matrix, ell, W_cmb, ddW_cmb = get_residuals(
        model_results, fg_freq_maps, sigma_spectral, lmin, lmax, fsky, params,
        cmb_spectra=spectra_true, true_A_cmb=model_data.mix_effectiv[:, :2])
    print('residuals estimation time = ', time.time() - start_residuals)
    WA_cmb = W_cmb.dot(model_results.mix_effectiv[:, :2])

    W_dBdB_cmb = ddW_cmb.dot(model_results.mix_effectiv[:, :2])
    VA_cmb = np.einsum('ij,ij...->...', sigma_spectral, W_dBdB_cmb[:, :])
    # IPython.embed()
    # np.save(save_path+'sigma_miscal_eval_pior2to4_'+prior_str+r_str, sigma_spectral)
    # np.save(save_path+'stat_residuals_eval_pior2to4_'+prior_str+r_str, stat)
    # np.save(save_path+'bias_residuals_eval_pior2to4_'+prior_str+r_str, bias)
    # np.save(save_path+'var_residuals_eval_pior2to4_'+prior_str+r_str, var)
    # np.save(save_path+'ell_pior2to4_'+prior_str+r_str, ell)

    '''=================Init and cosmo likelihood estimation================'''
    Cl_fid = {}
    Cl_fid['BB'] = get_Cl_cmbBB(Alens=A_lens_true, r=r_true, path_BB=path_BB)[2][lmin:lmax+1]
    Cl_fid['BuBu'] = get_Cl_cmbBB(Alens=0.0, r=1.0, path_BB=path_BB)[2][lmin:lmax+1]
    Cl_fid['BlBl'] = get_Cl_cmbBB(Alens=1.0, r=0.0, path_BB=path_BB)[2][lmin:lmax+1]
    Cl_fid['EE'] = ps_planck[1, lmin:lmax+1]

    Cl_noise, ell_noise = get_noise_Cl(
        model_results.mix_effectiv, lmax+1, fsky,
        sensitiviy_mode, one_over_f_mode,
        instrument=INSTRU)
    Cl_noise = Cl_noise[lmin-2:]
    ell_noise = ell_noise[lmin-2:]
    Cl_noise_matrix = np.zeros([2, 2, Cl_noise.shape[0]])
    Cl_noise_matrix[0, 0] = Cl_noise
    Cl_noise_matrix[1, 1] = Cl_noise

    # Cl_residuals_matrix = copy.deepcopy(Cl_cmb)
    tr_SigmaYY = np.einsum('ij,jimnl->mnl', sigma_spectral, Cl_residuals_matrix['YY'])
    Cl_data = Cl_residuals_matrix['yy'] + Cl_residuals_matrix['zy'] + \
        Cl_residuals_matrix['yz'] + tr_SigmaYY + Cl_noise_matrix

    # Cl_data = Cl_residuals_matrix['yy'] + tr_SigmaYY + Cl_noise_matrix

    chi2cosmo_true = likelihood_exploration([r_true, beta_true.value], Cl_fid, Cl_data,
                                            Cl_noise_matrix, tr_SigmaYY, WA_cmb, VA_cmb, ell, fsky)

    cosmo_params = [0.03, 0.04]
    # IPython.embed()
    # cosmo_array_start = np.random.uniform(np.array(bounds_cosmo)[:, 0],
    #                                       np.array(bounds_cosmo)[:, 1])
    results_cosmp = minimize(likelihood_exploration, cosmo_params, args=(
        Cl_fid, Cl_data, Cl_noise_matrix, tr_SigmaYY, WA_cmb, VA_cmb, ell, fsky),
        bounds=bounds_cosmo, tol=1e-18,
        method=method_cosmo, jac=jac_cosmo_min)
    print('')
    print(results_cosmp)
    print('')
    print('delta chi2cosmo = ', results_cosmp.fun - chi2cosmo_true)
    print('results - true cosmo = ', results_cosmp.x - np.array([r_true, beta_true.value]))
    IPython.embed()

    r_range = np.linspace(0, 0.02, 100)
    # beta_range = np.linspace(0.0017, 0.009, 100)
    beta_range = np.linspace(0.005, 0.014, 100)

    like_grid = np.empty((len(r_range), len(beta_range)))
    r, b = np.meshgrid(r_range, beta_range)
    for i in range(len(r_range)):
        for j in range(len(beta_range)):
            like_grid[i, j] = likelihood_exploration(
                [r_range[i], beta_range[j]], Cl_fid, Cl_data, Cl_noise_matrix,
                tr_SigmaYY, WA_cmb, VA_cmb, ell, fsky)
    np.save('../results_and_data/potatoe/'+'chi2_miscal_flat.npy', like_grid)
    np.save('../results_and_data/potatoe/'+'r_range_miscal_flat.npy', r)
    np.save('../results_and_data/potatoe/'+'beta_range_miscal_flat.npy', b)

    grid_r = np.linspace(-0.01, 0.02, 100)
    grid_beta = np.linspace(-0.08, 0.08, 100)
    param_grid = [r_true, beta_true.value]
    chi2r = []
    chi2_beta = []
    jac_r = []
    jac_beta = []
    for r in grid_r:
        param_grid[0] = r
        chi2r.append(likelihood_exploration(param_grid, Cl_fid, Cl_data,
                                            Cl_noise_matrix, tr_SigmaYY, ell, fsky))
        jac_r.append(jac_cosmo(param_grid, Cl_fid, Cl_data,
                               Cl_noise_matrix, tr_SigmaYY, ell, fsky))
    chi2r = np.array(chi2r)
    jac_r = np.array(jac_r)

    param_grid = [r_true, beta_true.value]
    for b in grid_beta:
        param_grid[1] = b
        chi2_beta.append(likelihood_exploration(param_grid, Cl_fid, Cl_data,
                                                Cl_noise_matrix, tr_SigmaYY, ell, fsky))
        jac_beta.append(jac_cosmo(param_grid, Cl_fid, Cl_data,
                                  Cl_noise_matrix, tr_SigmaYY, ell, fsky))
    chi2_beta = np.array(chi2_beta)
    jac_beta = np.array(jac_beta)
    plt.plot(grid_r, chi2r)
    plt.vlines(r_true, min(chi2r), max(chi2r), colors='black')
    plt.savefig('test_gridr_noYY')
    plt.close()
    plt.plot(grid_r, jac_r[:, 0])
    plt.vlines(r_true, min(jac_r[:, 0]), max(jac_r[:, 0]), colors='black')
    plt.savefig('test_jacr')
    plt.close()

    plt.plot(grid_beta, chi2_beta)
    plt.vlines(beta_true.value, min(chi2_beta), max(chi2_beta), colors='black')
    plt.savefig('test_gridbeta_noYY')
    plt.close()
    plt.plot(grid_beta, jac_beta[:, 1])
    plt.vlines(beta_true.value, min(jac_beta[:, 1]), max(jac_beta[:, 1]), colors='black')
    plt.savefig('test_jacbeta')
    plt.close()

    '''==================Fisher cosmo likelihood estimation================='''

    Cl_cmb_model_fish = np.zeros([4, Cl_fid['EE'].shape[0]])
    Cl_cmb_model_fish[1] = copy.deepcopy(Cl_fid['EE'])
    Cl_cmb_model_fish[2] = copy.deepcopy(Cl_fid['BlBl'])*A_lens_true + \
        copy.deepcopy(Cl_fid['BuBu']) * results_cosmp.x[0]

    Cl_cmb_model_fish_r = np.zeros([4, Cl_fid['EE'].shape[0]])
    Cl_cmb_model_fish_r[2] = copy.deepcopy(Cl_fid['BuBu']) * 1

    deriv1 = cf.power_spectra_obj(lib.cl_rotation_derivative(
        Cl_cmb_model_fish.T, results_cosmp.x[1]*u.rad), ell)
    deriv_matrix_beta = cf.power_spectra_obj(np.array(
        [[deriv1.spectra[:, 1], deriv1.spectra[:, 4]],
         [deriv1.spectra[:, 4], deriv1.spectra[:, 2]]]).T, deriv1.ell)

    deriv_matrix_r = cf.power_spectra_obj(lib.get_dr_cov_bir_EB(
        Cl_cmb_model_fish_r.T, results_cosmp.x[1]*u.rad).T, ell)

    Cl_data_spectra = cf.power_spectra_obj(Cl_data.T, ell)

    fish_beta = cf.fisher_pws(Cl_data_spectra, deriv_matrix_beta, fsky)
    fish_r = cf.fisher_pws(Cl_data_spectra, deriv_matrix_r, fsky)
    fish_beta_r = cf.fisher_pws(Cl_data_spectra, deriv_matrix_beta, fsky, deriv2=deriv_matrix_r)
    fisher_cosmo_matrix = np.array([[fish_r, fish_beta_r],
                                    [fish_beta_r, fish_beta]])

    IPython.embed()

    '''==================================================================================================='''
    '''==================================================================================================='''

    spectra_true_r = copy.deepcopy(get_Cl_cmbBB(
        Alens=A_lens_true, r=r_true, path_BB=path_BB))
    spectra_fit_r = copy.deepcopy(get_Cl_cmbBB(
        Alens=A_lens_true, r=results_cosmp.x[0], path_BB=path_BB))

    spectra_fit_rot = lib.cl_rotation(spectra_fit_r.T, results_cosmp.x[1]*u.rad).T
    spectra_bias_rot2 = lib.cl_rotation(spectra_true_r.T, (2*u.deg).to(u.rad)).T
    spectra_bias_rot1 = lib.cl_rotation(spectra_true_r.T, (1*u.deg).to(u.rad)).T
    tr_SigmaYY = np.einsum('ij,jimnl->mnl', sigma_spectral, Cl_residuals_matrix['YY'])
    tr_SigmaYY01d = np.einsum('ij,jimnl->mnl', sigma_spectral, Cl_residuals_matrix01d['YY'])

    # plt.plot(ell, spectra_data_rot[2]*ell*(ell+1)/(2*np.pi), label='BB data')
    plt.plot(ell, spectra_data_matrix[1, 1]*ell*(ell+1)/(2*np.pi), label='BB data + noise')
    # plt.plot(ell, bias[2][lmin:lmax+1]*ell*(ell+1)/(2*np.pi), label='BB syst')
    plt.plot(ell, (Cl_residuals_matrix['yy']+Cl_residuals_matrix['yz']+Cl_residuals_matrix['zy'])
             [1, 1]*ell*(ell+1)/(2*np.pi), label='BB syst1')
    plt.plot(ell, tr_SigmaYY[1, 1] * ell*(ell+1)/(2*np.pi), label='BB stat')
    # plt.plot(ell, (spectra_data_rot[2]+bias[2][lmin:lmax+1])
    #          * ell*(ell+1)/(2*np.pi), label='BB syst + data')
    # plt.plot(ell, (spectra_data_rot[2]+(Cl_residuals_matrix['yy']+2*Cl_residuals_matrix['yz']+tr_SigmaYY)
    #                [1, 1])*ell*(ell+1)/(2*np.pi), label='BB data + syst1 + stat')
    i = 1
    j = 1
    k = 2
    spec = 'BB'
    plt.plot(ell, (spectra_data_matrix[i, j]+(Cl_residuals_matrix['yy']+Cl_residuals_matrix['yz']+Cl_residuals_matrix['zy']+tr_SigmaYY)
                   [i, j])*ell*(ell+1)/(2*np.pi), label=spec+' data + noise + syst1 + stat')
    plt.plot(ell, (spectra_fit_rot[k][lmin:lmax+1]+(Cl_noise_matrix+Cl_residuals_matrix['yy']+Cl_residuals_matrix['yz']+Cl_residuals_matrix['zy']+tr_SigmaYY)
                   [i, j])*ell*(ell+1)/(2*np.pi), label=spec+' FIT + noise + syst1 + stat')

    plt.plot(ell, spectra_bias_rot1[k][lmin:lmax+1]*ell *
             (ell+1)/(2*np.pi), label=spec+' 1deg ', linestyle='-.')
    plt.plot(ell, spectra_bias_rot2[k][lmin:lmax+1]*ell *
             (ell+1)/(2*np.pi), label=spec+' 2deg ', linestyle='-.')
    plt.plot(ell, (spectra_bias_rot2[k][lmin:lmax+1]+tr_SigmaYY[i, j]+Cl_noise_matrix[i, j])*ell *
             (ell+1)/(2*np.pi), label=spec+' 1deg + stat res + noise ', linestyle='-.')
    plt.legend()

    plt.plot(ell, spectra_fit_rot[2][lmin:lmax+1]*ell *
             (ell+1)/(2*np.pi), label='BB fit', linestyle='-.')
    plt.legend()
    ''' plot residuals poster'''
    spectra_prim = copy.deepcopy(get_Cl_cmbBB(Alens=1, r=r_true, path_BB=path_BB))
    spectra_prim_rot1 = lib.cl_rotation(spectra_prim.T, (1*u.deg).to(u.rad)).T
    tr_SigmaYY_cmb = np.einsum('ij,jimnl->mnl', sigma_spectral, Cl_cmb['YY'])
    tr_SigmaYYfg = np.einsum('ij,jiml->ml', sigma_spectral, Cl['YY'])

    spectra_fit_r = copy.deepcopy(get_Cl_cmbBB(
        Alens=A_lens_true, r=results_cosmp.x[0], path_BB=path_BB))
    spectra_fit_rot = lib.cl_rotation(spectra_fit_r.T, results_cosmp.x[1]*u.rad).T

    figure = plt.gcf()
    largeur = 10.
    hauteur = largeur * 16./9.
    figure.set_size_inches(hauteur, largeur)
    spec = 'EB'
    if spec == 'EE':
        i = 0
        j = 0
        k = 1
        l = 0

    if spec == 'BB':
        i = 1
        j = 1
        k = 2
        l = 1

    if spec == 'EB':
        i = 0
        j = 1
        k = 4
        l = 2

    if spec == 'EE' or spec == 'BB':
        # plt.plot(ell, Cl_fid[spec]*ell*(ell+1)/(2*np.pi),
        #          label=r'$C_{\ell}^{' + spec + '}$ lensing+primordial r=0.01', lw=2, color='blue')
        plt.plot(ell, spectra_prim[k][lmin:lmax+1]*ell *
                 (ell+1)/(2*np.pi), label=r'$C_{\ell}^{' + spec + '}$ primordial r=0.01', lw=2, color='orange')
    plt.plot(ell, spectra_fit_rot[k][lmin:lmax+1]*ell*(ell+1)/(2*np.pi),
             label=r'$C_{\ell}^{' + spec + '}$ fit result', lw=2, color='turquoise')
    # plt.plot(ell, (Cl_residuals_matrix['yy']+Cl_residuals_matrix['yz']+Cl_residuals_matrix['zy'])
    #          [1, 1]*ell*(ell+1)/(2*np.pi), label='systematic residuals, bias=1deg', linestyle='-.', lw=2)
    plt.plot(ell, spectra_prim_rot1[k][lmin:lmax+1]*ell *
             (ell+1)/(2*np.pi), label=r'$C_{\ell}^{'+spec+'}$ primordial r=0.01 Rotation 1deg', lw=2, color='green', linestyle=':')
    if spec == 'EE' or spec == 'BB':
        plt.plot(ell, (spectra_prim_rot1[k]+(1-2*np.cos(2*1*u.deg))*spectra_prim[k])[lmin:lmax+1]*ell *
                 (ell+1)/(2*np.pi), label=r'$C_{\ell}^{'+spec+'}$ primordial r=0.01 Rotation 1deg - '+spec+'prim', lw=2, color='red')
        plt.plot(ell, (spectra_prim_rot1[k]-spectra_prim[k])[lmin:lmax+1]*ell *
                 (ell+1)/(2*np.pi), label=r'$C_{\ell}^{'+spec+'}$ primordial r=0.01 Rotation 1deg - '+spec+'prim', lw=2, color='red')

        # plt.plot(ell, (spectra_fit_rot[k]+(1-2*np.cos(2*results_cosmp.x[1]*u.rad))*spectra_fit_r[k])[lmin:lmax+1]*ell *
        #          (ell+1)/(2*np.pi), label=r'$C_{\ell}^{'+spec+'}$ primordial r=0.01 Rotation 1deg - BBprim', lw=2, color='turquoise', linestyle=':')
    else:
        plt.plot(ell, (spectra_prim_rot1[k]-np.sin(2*1*u.deg)*(spectra_prim[1]-spectra_prim[2]))[lmin:lmax+1]*ell *
                 (ell+1)/(2*np.pi), label=r'$C_{\ell}^{'+spec+'}$ primordial r=0.01 Rotation 1deg - BBprim', lw=2, color='red')

        # plt.plot(ell, (spectra_fit_rot[k]-np.sin(2*results_cosmp.x[1]*u.rad)*(spectra_fit_r[1]-spectra_fit_r[2]))[lmin:lmax+1]*ell *
        #          (ell+1)/(2*np.pi), label=r'$C_{\ell}^{'+spec+'}$ primordial r=0.01 Rotation 1deg - BBprim', lw=2, color='turquoise', linestyle=':')
    # plt.plot(ell, (Cl_residuals_matrix01d['yy']+Cl_residuals_matrix01d['yz']+Cl_residuals_matrix01d['zy'])
    #          [1, 1]*ell*(ell+1)/(2*np.pi), label='systematic residuals, bias=0p1deg', linestyle='-.', lw=2)

    plt.plot(ell, (Cl_cmb['yy']+Cl_cmb['yz']+Cl_cmb['zy'])
             [i, j][lmin:lmax+1]*ell*(ell+1)/(2*np.pi), label='systematic residuals CMB '+spec+', bias=1deg', linestyle='-.', lw=2, color='purple')
    plt.plot(ell, (Cl['yy']+Cl['yz']+Cl['zy'])
             [l][lmin:]*ell*(ell+1)/(2*np.pi), label='systematic residuals foregrounds '+spec+', bias=1deg', linestyle='-.', lw=2, color='brown')
    # plt.plot(ell, tr_SigmaYY[1, 1] * ell*(ell+1)/(2*np.pi),
    #          label='statistical residuals, bias=1deg', linestyle='--', lw=2)
    plt.plot(ell, tr_SigmaYY_cmb[i, j][lmin:lmax+1] * ell*(ell+1)/(2*np.pi),
             label='statistical residuals CMB '+spec+', bias=1deg', linestyle='--', lw=2, color='magenta')
    plt.plot(ell, tr_SigmaYYfg[l, lmin:] * ell*(ell+1)/(2*np.pi),
             label='statistical residuals foregrounds '+spec+', bias=1deg', linestyle='--', lw=2, color='grey')
    # plt.plot(ell, tr_SigmaYY01d[1, 1] * ell*(ell+1)/(2*np.pi),
    #          label='statistical residuals, bias=1deg', linestyle='-.', lw=2)
    plt.xlabel(r'$\ell$', fontsize=20)
    plt.ylabel(r'$C_\ell \frac{\ell(\ell+1)}{2\pi} $', fontsize=20)
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.legend(prop={'size': 20})
    plt.savefig('testresiduals_poster_2bias', bbox_inches='tight')
    plt.close()

    plt.plot(ell, (Cl_residuals_matrix['yy']+2*Cl_residuals_matrix['yz'])
             [0, 1]*ell*(ell+1)/(2*np.pi), label='EB syst1')
    plt.plot(ell, tr_SigmaYY[0, 1] * ell*(ell+1)/(2*np.pi), label='EB stat')
    plt.plot(ell, (Cl_residuals_matrix['yy']+2*Cl_residuals_matrix['yz']+tr_SigmaYY)
             [0, 1]*ell*(ell+1)/(2*np.pi), label='EB syst1 + stat')
    plt.plot(ell, spectra_fit_rot[4][lmin:lmax+1]*ell *
             (ell+1)/(2*np.pi), label='EB fit', linestyle='-.')
    plt.plot(ell, spectra_bias_rot[4][lmin:lmax+1]*ell *
             (ell+1)/(2*np.pi), label='EB bias', linestyle='-.')

    plt.legend()

    hess_fun = nd.Hessian(likelihood_exploration)
    hess_eval = hess_fun(results_cosmp.x, Cl_fid, Cl_residuals_matrix,
                         Cl_noise_matrix, sigma_spectral, ell, fsky, WA_cmb)
    sigma_hess = np.sqrt(np.linalg.inv(hess_eval))
    hess_inv_min = results_cosmp.hess_inv(results_cosmp.x)

    np.save(save_path+'cosmo_mini_pior2to4_'+prior_str+r_str, results_cosmp.x)
    np.save(save_path+'sigma_numdiff_cosmo_eval_pior2to4_'+prior_str+r_str, sigma_hess)
    np.save(save_path+'hessinv_mini_cosmo_eval_pior2to4_'+prior_str+r_str, hess_inv_min)

    '''==================================================================================================='''
    '''==================================================================================================='''

    beta_range = np.arange(0.0, 0.02, 0.02/100)

    r_range = np.arange(-0.01, 0.02, 0.021/100)

    r_range2, beta_range2, r_index, beta_index, rmini_index, betamini_index = get_ranges(
        [-0.2, 0.5], [-np.pi/4, np.pi/4], r_true, beta_true.value, results_cosmp.x, 5, 5)
    # beta_index = 0
    start = time.time()
    chi2_mesh, jac_mesh_r, jac_mesh_beta, betaxx, ryy, pos = get_chi2_grid(r_range2, beta_range2, Cl_fid, Cl_data,
                                                                           Cl_noise_matrix, tr_SigmaYY, ell, fsky)
    print(time.time() - start)
    like_mesh, like_r, like_beta = chi2_2_likenorm(chi2_mesh, pos, r_index, beta_index)
    sigma2_r, sigma2_beta, sigma2_beta_r = sigma2_int(
        like_mesh, like_r, like_beta, pos, r_index, beta_index)

    r_range = np.arange(-sigma_hess[0, 0]*3, sigma_hess[0, 0]*3, sigma_hess[0, 0]*6/100)
    beta_range = np.arange(-sigma_hess[1, 1]*3, sigma_hess[1, 1]*3, sigma_hess[1, 1]*6/100)

    # fisher_2D_gaussian = multivariate_gaussian(
    # pos, results_cosmp.x, np.flip(np.linalg.inv(hess_eval)))

    # cs = ax.contourf(betaxx, ryy, np.exp((-chi2_mesh+np.min(chi2_mesh))), levels=levels)
    # cs2 = ax.contour(betaxx, ryy, fisher_2D_gaussian /
    #                  np.max(fisher_2D_gaussian), levels=cs.levels, colors='r', linestyles='--')
    sigma_matrix = np.array([[sigma2_r, 0], [0, sigma2_beta]])
    fisher_2D_gaussian = multivariate_gaussian(
        pos, np.flip(results_cosmp.x), np.flip(np.linalg.inv(fisher_cosmo_matrix)))
    fisher_2D_gaussian *= np.max(like_mesh) / np.max(fisher_2D_gaussian)

    int_2D_gaussian = multivariate_gaussian(
        pos, np.flip(results_cosmp.x), np.flip(sigma_matrix))
    int_2D_gaussian *= np.max(like_mesh) / np.max(int_2D_gaussian)
    fig, ax = plt.subplots()
    levels = np.arange(0, 1+1/8, 1/8)*np.max(like_mesh)
    cs = ax.contourf(betaxx, ryy, like_mesh, levels=levels)
    cs2 = ax.contour(betaxx, ryy, fisher_2D_gaussian, levels=cs.levels, colors='r', linestyles='--')
    cs3 = ax.contour(betaxx, ryy, int_2D_gaussian, levels=cs.levels, colors='black', linestyles=':')

    cbar = fig.colorbar(cs)
    cbar.add_lines(cs2)
    cbar.ax.set_xlabel(r'$\mathcal{L}$')
    plt.ylabel(r'$r$')
    plt.xlabel(r'Briefringence angle $\beta$ in radian')
    plt.title(r'Joint likelihood on $r$ and $\alpha$ with $r_{input} =$'+'{},'.format(r_true)+r' $\beta_{input}=$'+'{}'.format(
        beta_true))
    h1, _ = cs2.legend_elements()
    ax.legend([h1[0]], ["Hessian prediction"])
    plt.savefig(save_path+'gridding_cosmo_0p1deg',  bbox_inches='tight')

    start = time.time()
    sigma_r_min = np.sqrt(hess_inv_min)[0]

    r_range = np.arange(0, results_cosmp.x[0] +
                        3*sigma_hess[0, 0], (results_cosmp.x[0] + 3*sigma_hess[0, 0])/100)

    likelihood_grid_r = []
    for r in r_range:
        likelihood_grid_r.append(likelihood_exploration([r, beta_true.value], Cl_fid, Cl_data,
                                                        Cl_noise_matrix, tr_SigmaYY, ell, fsky))
    print('time 1 = ', time.time() - start)

    likelihood_grid_r = np.array(likelihood_grid_r)

    factor_test = 1.
    hess_logLr = (r_range - results_cosmp.x[0])**2 / \
        (2 * ((sigma_hess[0, 0]*factor_test)**2)) + np.log((sigma_hess[0, 0]*factor_test)
                                                           * np.sqrt(2*np.pi))
    hess_logLr_norm = hess_logLr - np.min(hess_logLr) + np.min(likelihood_grid_r)

    sigma_r_min = np.sqrt(hess_inv_min)[0]
    hess_logLr_min = (r_range - results_cosmp.x[0])**2 / \
        (2 * ((sigma_r_min*factor_test)**2)) + np.log((sigma_r_min*factor_test)
                                                      * np.sqrt(2*np.pi))
    hess_logLr_min_norm = hess_logLr_min - np.min(hess_logLr_min) + np.min(likelihood_grid_r)

    sigma_r_int = np.sqrt(sigma2_r)
    hess_logLr_int = (r_range - results_cosmp.x[0])**2 / \
        (2 * ((sigma_r_int*factor_test)**2)) + np.log((sigma_r_int*factor_test)
                                                      * np.sqrt(2*np.pi))
    hess_logLr_int_norm = hess_logLr_int - np.min(hess_logLr_int) + np.min(likelihood_grid_r)

    plt.plot(r_range, likelihood_grid_r, label='-2logL')
    plt.vlines(r_true, np.min(likelihood_grid_r), np.max(
        likelihood_grid_r), color='black', linestyles='--')
    plt.vlines(results_cosmp.x[0], np.min(likelihood_grid_r), np.max(
        likelihood_grid_r), color='black', linestyles='-.')

    plt.plot(r_range, hess_logLr_norm, linestyle='--', label='numdifftool hessian')
    # plt.plot(r_range, hess_logLr_min_norm, linestyle='--', label='minimization hessian')
    plt.plot(r_range, hess_logLr_int_norm, linestyle='--', label='sigma integral')
    plt.legend()
    plt.show()

    plt.plot(r_range, likelihood_grid_r[:, 0], label='first term')
    plt.plot(r_range, likelihood_grid_r[:, 1], label='second term')
    plt.plot(r_range, likelihood_grid_r[:, 2], label='logdet term')
    min_r = np.min(np.sum(likelihood_grid_r, axis=-1))
    max_r = np.max(np.sum(likelihood_grid_r, axis=-1))
    plt.vlines(r_true, min_r, max_r, color='black', linestyles='--')
    plt.plot(r_range, np.sum(likelihood_grid_r, axis=-1), label='total')
    plt.legend()

    beta_range = np.arange(
        beta_true.value-3*sigma_hess[1, 1], beta_true.value+3*sigma_hess[1, 1], 6*sigma_hess[1, 1]/100)

    likelihood_grid_beta = []
    likelihood_grid_beta2 = []
    for beta in beta_range:
        likelihood_grid_beta.append(likelihood_exploration_new([r_true, beta], Cl_fid, Cl_residuals_matrix,
                                                               Cl_noise_matrix, sigma_res, ell, fsky, WA_cmb))
        likelihood_grid_beta2.append(likelihood_exploration_new2([r_true, beta], Cl_fid, Cl_data,
                                                                 Cl_noise_matrix, tr_SigmaYY, ell, fsky))

    likelihood_grid_beta = np.array(likelihood_grid_beta)
    likelihood_grid_beta2 = np.array(likelihood_grid_beta2)

    factor_test = 1
    hess_logLbeta = (beta_range - results_cosmp_new.x[1])**2 / \
        (2 * ((sigma_hess[1, 1]*factor_test)**2)) + np.log((sigma_hess[1, 1]*factor_test)
                                                           * np.sqrt(2*np.pi))
    hess_logLbeta_norm = hess_logLbeta - np.min(hess_logLbeta) + np.min(likelihood_grid_beta)

    sigma_beta_min = np.sqrt(hess_inv_min)[1]
    hess_logLbeta_min = (beta_range - results_cosmp_new.x[1])**2 / \
        (2 * ((sigma_beta_min*factor_test)**2)) + np.log((sigma_beta_min*factor_test)
                                                         * np.sqrt(2*np.pi))
    hess_logLbeta_min_norm = hess_logLbeta_min - \
        np.min(hess_logLbeta_min) + np.min(likelihood_grid_beta)

    sigma_beta_int = np.sqrt(sigma2_beta)
    hess_logLbeta_int = (beta_range - results_cosmp_new.x[1])**2 / \
        (2 * ((sigma_beta_int*factor_test)**2)) + np.log((sigma_beta_int*factor_test)
                                                         * np.sqrt(2*np.pi))
    hess_logLbeta_int_norm = hess_logLbeta_int - \
        np.min(hess_logLbeta_int) + np.min(likelihood_grid_beta)

    plt.plot(beta_range, likelihood_grid_beta, label='-2logL')
    plt.vlines(beta_true.value, np.min(likelihood_grid_beta), np.max(
        likelihood_grid_beta), color='black', linestyles='--')
    plt.vlines(results_cosmp_new.x[1], np.min(likelihood_grid_beta), np.max(
        likelihood_grid_beta), color='black', linestyles='-.')

    plt.plot(beta_range, hess_logLbeta_norm, linestyle='--', label='numdifftool hessian')
    # plt.plot(beta_range, hess_logLbeta_min_norm, linestyle='--', label='minimization hessian')
    plt.plot(beta_range, hess_logLbeta_int_norm, linestyle='--', label='sigma integral')
    plt.legend()
    plt.show()

    plt.plot(beta_range, likelihood_grid_beta[:, 0], label='first term')
    plt.plot(beta_range, likelihood_grid_beta[:, 1], label='second term')
    plt.plot(beta_range, likelihood_grid_beta[:, 2], label='logdet term')
    min_beta = np.min(np.sum(likelihood_grid_beta, axis=-1))
    max_beta = np.max(np.sum(likelihood_grid_beta, axis=-1))
    plt.vlines(beta_true.value, min_beta, max_beta, color='black', linestyles='--')
    plt.plot(beta_range, np.sum(likelihood_grid_beta, axis=-1), label='total')
    plt.legend()
    lmin_plot = lmin
    lmax_plot = lmax+1
    # ell = np.arange(lmin_plot, lmax_plot)
    Cl_BBprimonly = get_Cl_cmbBB(Alens=0, r=0.01, path_BB=path_BB)[2]
    plt.plot(ell, ps_planck[1][lmin_plot:lmax_plot]*ell*(ell+1), label='CMB EE')
    plt.plot(ell, stat[0][lmin_plot:lmax_plot]*ell*(ell+1),
             label='stat EE', color='orange', linestyle='--')
    plt.plot(ell, bias[0][lmin_plot:lmax_plot]*ell*(ell+1),
             label='syst EE', color='orange', linestyle=':')
    plt.plot(ell, var[0][lmin_plot:lmax_plot]*ell*(ell+1),
             label='var EE', color='orange', linestyle='-.')
    plt.plot(ell, stat_noCMB[0][lmin_plot:lmax_plot]*ell*(ell+1),
             label='stat EE noCMB', color='red', linestyle='--')
    plt.plot(ell, bias_noCMB[0][lmin_plot:lmax_plot]*ell*(ell+1),
             label='syst EE noCMB', color='red', linestyle=':')
    plt.plot(ell, var_noCMB[0][lmin_plot:lmax_plot]*ell*(ell+1),
             label='var EE noCMB', color='red', linestyle='-.')

    plt.plot(ell, ps_planck[2][lmin_plot:lmax_plot]*ell*(ell+1), label='CMB BB')
    plt.plot(ell, Cl_BBprimonly[lmin_plot:lmax_plot]*ell*(ell+1), label='CMB BB prim r=0.01')
    plt.plot(ell, stat[1][lmin_plot:lmax_plot]*ell*(ell+1),
             label='stat BB', color='blue', linestyle='--')
    plt.plot(ell, np.abs(bias[1][lmin_plot:lmax_plot])*ell*(ell+1),
             label='syst BB', color='blue', linestyle=':')
    plt.plot(ell, var[1][lmin_plot:lmax_plot]*ell*(ell+1),
             label='var BB', color='orange', linestyle='-.')
    plt.legend()
    plt.loglog()
    plt.plot(ell, stat_noCMB[1][lmin_plot:lmax_plot]*ell*(ell+1),
             label='stat BB noCMB', color='red', linestyle='--')
    plt.plot(ell, bias_noCMB[1][lmin_plot:lmax_plot]*ell*(ell+1),
             label='syst BB noCMB', color='red', linestyle=':')
    plt.plot(ell, var_noCMB[1][lmin_plot:lmax_plot]*ell*(ell+1),
             label='var BB noCMB', color='red', linestyle='-.')

    # plt.plot(ell, ps_planck[4][lmin_plot:lmax_plot]*ell*(ell+1), label='CMB EB')
    plt.plot(ell, stat[2][lmin_plot:lmax_plot]*ell*(ell+1),
             label='stat EB', color='orange', linestyle='--')
    plt.plot(ell, bias[2][lmin_plot:lmax_plot]*ell*(ell+1),
             label='syst EB', color='orange', linestyle=':')
    plt.plot(ell, var[2][lmin_plot:lmax_plot]*ell*(ell+1),
             label='var EB', color='orange', linestyle='-.')
    plt.legend()
    plt.yscale('symlog')
    plt.xscale('log')

    plt.plot(ell, stat_noCMB[2][lmin_plot:lmax_plot]*ell*(ell+1),
             label='stat EB noCMB', color='red', linestyle='--')
    plt.plot(ell, bias_noCMB[2][lmin_plot:lmax_plot]*ell*(ell+1),
             label='syst EB noCMB', color='red', linestyle=':')
    plt.plot(ell, var_noCMB[2][lmin_plot:lmax_plot]*ell*(ell+1),
             label='var EB noCMB', color='red', linestyle='-.')

    exit()


######################################################
# MAIN CALL
if __name__ == "__main__":
    main()

'''================================Purgatory================================'''
'''
    spectra_data_matrix = np.zeros([2, 2, lmax+1-lmin])
    spectra_data_matrix[0, 0] = copy.deepcopy(spectra_true[1][lmin:lmax+1])
    spectra_data_matrix[1, 1] = copy.deepcopy(spectra_true[2][lmin:lmax+1])
    spectra_data_matrix[1, 0] = copy.deepcopy(spectra_true[4][lmin:lmax+1])
    spectra_data_matrix[0, 1] = copy.deepcopy(spectra_true[4][lmin:lmax+1])
    spectra_data_matrix += copy.deepcopy(Cl_noise_matrix)

    start = time.time()
    likelihood_grid = []
    for r in r_range:
        for beta in beta_range:
            likelihood_grid.append(likelihood_exploration([r, beta], Cl_fid, spectra_data_matrix,
                                                          Cl_residuals_matrix, Cl_noise_matrix, sigma_res, ell, fsky))
    likelihood_grid = np.array(likelihood_grid)
    chi2_mesh = np.reshape(likelihood_grid, (-1, len(beta_range)))
    print('time gird = ', time.time() - start)
    betaxx, ryy = np.meshgrid(beta_range, r_range)
    pos = np.empty(betaxx.shape + (2,))
    pos[:, :, 0] = betaxx
    pos[:, :, 1] = ryy
'''
'''
def cosmo_likelihood_nodeprojection_new(Cl_model, Cl_residuals, Cl_noise_matrix, sigma_res, ell, fsky, WA_cmb):
    residuals_factor = 1
    tr_SigmaYY = np.einsum('ij,jimnl->mnl', sigma_res, Cl_residuals['YY'])*residuals_factor
    # cross_terms = +np.einsum('ij,jkl->ikl', WA_cmb, Cl_model) + np.einsum(
    #     'ijl,jk->ikl', Cl_model, WA_cmb.T) - Cl_model

    # wacaw = np.einsum('ij,jkl,km->iml', WA_cmb, Cl_model, WA_cmb.T)
    # Cl_model_total = wacaw + Cl_noise_matrix + tr_SigmaYY  # + cross_terms

    Cl_model_total = Cl_model + Cl_noise_matrix + tr_SigmaYY  # + cross_terms

    inv_model = np.linalg.inv(Cl_model_total.T).T
    dof = (2 * ell + 1) * fsky
    dof_over_Cl = dof * inv_model

    Cl_data = Cl_residuals['yy'] + Cl_residuals['zy'] + \
        Cl_residuals['yz'] + tr_SigmaYY + Cl_noise_matrix
    first_term_ell = []
    second_term_ell = []
    logm_list = []
    for l in range(dof_over_Cl.shape[-1]):
        first_term_ell.append(dof_over_Cl[:, :, l].dot((Cl_data)[:, :, l]))
        # second_term_ell.append(dof_over_Cl[:, :, l].dot(
        #     (Cl_residuals['yy'] + Cl_residuals['zy'] + Cl_residuals['yz'])[:, :, l]))
        logm_list.append(logm(Cl_model_total[:, :, l]))

    first_term_ell = np.array(first_term_ell)
    # second_term_ell = np.array(second_term_ell)*residuals_factor
    logm_list = np.array(logm_list)

    first_term = np.sum(np.trace(first_term_ell.T))
    # second_term = np.sum(np.trace(second_term_ell.T))
    logdetC = np.sum(dof*np.trace(logm_list.T))

    return first_term + logdetC
'''
'''
def cosmo_likelihood_nodeprojection(Cl_model, Cl_data, Cl_residuals, sigma_res, ell, fsky):
    # Cl data should be CMB + noise !
    residuals_factor = 1
    tr_SigmaYY = np.einsum('ij,jimnl->mnl', sigma_res, Cl_residuals['YY'])*residuals_factor
    # cross_terms = +np.einsum('ij,jkl->ikl', WA_cmb, Cl_cmb_rot_matrix) + np.einsum(
    #     'ijl,jk->ikl', Cl_cmb_rot_matrix, WA_cmb.T)  # Cl_cmb_rot_matrix.dot(WA_cmb.T)
    Cl_model_total = Cl_model+tr_SigmaYY  # Cl_model should be CMB + Noise !
    # Cl_model_total = Cl_model  # Cl_model should be CMB + Noise !
    inv_model = np.linalg.inv(Cl_model_total.T).T
    dof = (2 * ell + 1) * fsky
    dof_over_Cl = dof * inv_model

    # first_dep = dof_over_C  l*(Cl_data+tr_SigmaYY)
    # first_term = np.sum(np.trace(first_dep))
    # first_term = np.trace(np.einsum('ijl,kmn->im', dof_over_Cl, (Cl_data+tr_SigmaYY)))
    # IPython.embed()
    first_term_ell = []
    second_term_ell = []
    logm_list = []
    # logm_list_sum = []
    for l in range(dof_over_Cl.shape[-1]):
        first_term_ell.append(dof_over_Cl[:, :, l].dot((Cl_data+tr_SigmaYY)[:, :, l]))
        second_term_ell.append(dof_over_Cl[:, :, l].dot(
            (Cl_residuals['yy'] + Cl_residuals['zy'] + Cl_residuals['yz'])[:, :, l]))
        logm_list.append(logm(Cl_model_total[:, :, l]))
        # logm_list_sum.append(logm(Cl_model_total[:, :, l])*dof[l])

    first_term_ell = np.array(first_term_ell)
    second_term_ell = np.array(second_term_ell)*residuals_factor
    logm_list = np.array(logm_list)
    # logm_list_sum = np.array(logm_list_sum)

    first_term = np.sum(np.trace(first_term_ell.T))
    second_term = np.sum(np.trace(second_term_ell.T))
    logdetC = np.sum(dof*np.trace(logm_list.T))
    # first_sum = np.sum(first_term_ell)
    # second_sum = np.sum(second_term_ell)
    # logdetC_sum = np.sum(logm_list_sum)
    # IPython.embed()
    # plt.plot(ell, (Cl_data+tr_SigmaYY)[0, 1], label='data EB: CMB + noise + stat')
    # plt.plot(ell, Cl_model_total[0, 1], label='model EB: CMB + noise + stat res', linestyle='-.')
    # plt.plot(ell, tr_SigmaYY[0, 1], label='EB stat res')
    # plt.plot(ell, (Cl_residuals['yy'] + Cl_residuals['zy'] +
    #                Cl_residuals['yz'])[0, 1], label='EB syst res')
    # plt.legend()
    # plt.plot(np.abs(first_term_ell[:, 0, 0]))
    # plt.plot(np.abs(first_term_ell[:, 1, 1]))
    # plt.plot(np.abs(second_term_ell[:, 0, 1]))
    # plt.plot(np.abs(second_term_ell[:, 1, 0]))

    # return first_term, second_term, logdetC
    # return first_term + second_term + logdetC, (Cl_residuals['yy'] + Cl_residuals['zy'] + Cl_residuals['yz']), tr_SigmaYY
    return first_term + second_term + logdetC
    # return first_sum + second_sum + logdetC_sum
'''
'''
def get_diff_list(model):
    diff_list = fshp.diff_miscal_matrix(model)
    diff_list.append(fshp.diff_bir_matrix(model))
    if not model.fix_temp:
        mix_diff_Bd_QU, mix_diff_Td_QU, mix_diff_Bs_QU = fshp.diff_mixing_matrix(model)
    else:
        mix_diff_Bd_QU, mix_diff_Bs_QU = fshp.diff_mixing_matrix(model)

    diff_list.append(mix_diff_Bd_QU)
    diff_list.append(mix_diff_Bs_QU)
    return diff_list

def get_diff_diff_list(model):
    diff_diff_list = fshp.diff_diff_miscal_matrix(model)
    diff_diff_list.append(fshp.diff_diff_bir_matrix(model))
    if not model.fix_temp:
        mix_diff_diff_Bd_QU, mix_diff_diff_Td_QU, mix_diff_diff_Bs_QU = fshp.diff_diff_mixing_matrix(
            model)
    else:
        mix_diff_diff_Bd_QU, mix_diff_diff_Bs_QU = fshp.diff_diff_mixing_matrix(
            model)
    diff_diff_list.append(mix_diff_diff_Bd_QU)
    diff_diff_list.append(mix_diff_diff_Bs_QU)
    return diff_diff_list


def get_diff_W(model, diff_list, W=None, invAtNm1A=None, return_elements=False):
    if invAtNm1A is None:
        start = time.time()
        AtNm1A = model.mix_effectiv.T.dot(model.inv_noise).dot(model.mix_effectiv)
        invAtNm1A = np.linalg.inv(AtNm1A)
        print('time computing invANA = ', time.time()-start)
    if W is None:
        AtN = model.mix_effectiv.T.dot(model.inv_noise)
        W = invAtNm1A.dot(AtN)

    diff_W = []
    invANAdBpBt_list = []
    A_i_list = []
    AitNm1_list = []
    term1_list = []
    term2_list = []

    param_num = len(diff_list)
    for i in range(param_num):
        A_i = fshp.effectiv_diff_mixing_matrix(i, diff_list, model)
        B = A_i.T.dot(model.inv_noise).dot(model.mix_effectiv)
        BpBt = B+B.T
        invANABBT = invAtNm1A.dot(BpBt)
        AitNm1 = A_i.T.dot(model.inv_noise)

        term1 = invANABBT.dot(W)
        term2 = invAtNm1A.dot(AitNm1)
        diff_W_i = -term1+term2

        diff_W.append(diff_W_i)
        invANAdBpBt_list.append(invANABBT)
        A_i_list.append(A_i)
        AitNm1_list.append(AitNm1)
        term1_list.append(term1)
        term2_list.append(term2)
    if return_elements:
        return diff_W, invANAdBpBt_list, A_i_list, AitNm1_list, term1_list, term2_list

    return diff_W


def get_diff_diff_W(model, diff_list, diff_diff_list, W, invANAdBpBt_list, A_i_list,
                    AitNm1_list, term1_list, term2_list, invAtNm1A=None):
    if invAtNm1A is None:
        AtNm1A = model.mix_effectiv.T.dot(model.inv_noise).dot(model.mix_effectiv)
        invAtNm1A = np.linalg.inv(AtNm1A)
    start = time.time()
    diff_diff_W_ = []
    # param_num = np.shape(diff_list)[0]
    param_num = len(diff_list)

    for i in range(param_num):
        for ii in range(param_num):
            A_i_ii = fshp.effectiv_doublediff_mixing_matrix(i, ii, diff_list, diff_diff_list, model)
            D = A_i_ii.T.dot(model.inv_noise).dot(model.mix_effectiv) + \
                AitNm1_list[i].dot(A_i_list[ii])
            DpDt = D+D.T

            term1 = invANAdBpBt_list[ii].dot(invANAdBpBt_list[i]).dot(W)
            term2 = invANAdBpBt_list[i].dot(invANAdBpBt_list[ii]).dot(W)
            term3 = invANAdBpBt_list[ii].dot(term2_list[i])
            term4 = invANAdBpBt_list[i].dot(term2_list[ii])
            term5 = invAtNm1A.dot(DpDt).dot(W)
            term6 = invAtNm1A.dot(A_i_ii.T).dot(model.inv_noise)
            tot = term1+term2 - term3 - term4 - term5 + term6
            diff_diff_W_.append(tot)
    shape_diff_diff = np.shape(diff_diff_W_)
    # IPython.embed()
    print('time diff diff W = ', time.time()-start)
    diff_diff_W = np.reshape(
        diff_diff_W_, [param_num, param_num, shape_diff_diff[1], shape_diff_diff[2]])
    return diff_diff_W
'''

'''
def get_residuals(model, fg_freq_maps, sigma, lmax, fsky, cmb_spectra=None, true_A_cmb=None):
    ============================computing Ws============================
    diff_list = get_diff_list(model)
    diff_diff_list = get_diff_diff_list(model)
    # IPython.embed()
    start = time.time()
    W = get_W(model)
    diff_W, invANAdBpBt_list, A_i_list, AitNm1_list, term1_list, term2_list = get_diff_W(
        model, diff_list, W=W, invAtNm1A=None, return_elements=True)
    diff_diff_W = get_diff_diff_W(model, diff_list, diff_diff_list, W, invANAdBpBt_list, A_i_list,
                                  AitNm1_list, term1_list, term2_list, invAtNm1A=None)
    print('time W, WdB, WdBdB = ', time.time()-start)

    diff_W = np.array(diff_W)
    #
    # params = ['miscal']*6
    # # params.append('birefringence')
    # params.append('spectral')
    # params.append('spectral')
    # diff_list_new = get_diff_list_new(model, params)
    # diff_diff_list_new = get_diff_diff_list_new(model, params)
    # # IPython.embed()
    # diff_W, invANAdBpBt_list_new, A_i_list_new, AitNm1_list_new, term1_list_new, term2_list_new = get_diff_W_new(
    #     model, diff_list_new, params, W=W, invAtNm1A=None, return_elements=True)
    # diff_diff_W = get_diff_diff_W_new(model, diff_list_new, diff_diff_list_new, params, W, invANAdBpBt_list_new, A_i_list_new,
    #                                   AitNm1_list_new, term1_list_new, term2_list_new, invAtNm1A=None)
    # diff_W = np.array(diff_W)

    ===========================Computing ys===========================
    print('WARNING FSKY !!!!')

    y_Q = W[0].dot(fg_freq_maps)
    y_U = W[1].dot(fg_freq_maps)
    Y_Q = diff_W[:, 0].dot(fg_freq_maps)
    Y_U = diff_W[:, 1].dot(fg_freq_maps)
    V_Q = np.einsum('ij,ij...->...', sigma, diff_diff_W[:, :, 0])
    V_U = np.einsum('ij,ij...->...', sigma, diff_diff_W[:, :, 1])
    z_Q = V_Q.dot(fg_freq_maps)
    z_U = V_U.dot(fg_freq_maps)

    if cmb_spectra is not None:
        WA_cmb = W[:2].dot(true_A_cmb) - np.identity(2)
        W_dB_cmb = diff_W[:, :2, :].dot(true_A_cmb)
        W_dBdB_cmb = diff_diff_W[:, :, :2, :].dot(true_A_cmb)
        Cl_matrix = np.zeros((2, 2, len(cmb_spectra[0])))
        Cl_matrix[0, 0] = cmb_spectra[1]
        Cl_matrix[1, 1] = cmb_spectra[1]
        Cl_matrix2 = np.zeros((2, 2, len(cmb_spectra[0])))
        Cl_matrix2[0, 0] = cmb_spectra[1]
        Cl_matrix2[1, 1] = cmb_spectra[2]
        yy_cmb1 = np.einsum('ij,ijl,ij->ijl', WA_cmb.T, Cl_matrix, WA_cmb)
        YY_cmb1 = np.einsum('ij,ijl,ij->ijl', W_dB_cmb[0].T, Cl_matrix, W_dB_cmb[0])
        yy_cmb2 = np.einsum('ij,ijl,ij->ijl', WA_cmb.T, Cl_matrix2, WA_cmb)
        YY_cmb2 = np.einsum('ij,ijl,ij->ijl', W_dB_cmb[0].T, Cl_matrix2, W_dB_cmb[0])

        YY_cmb_matrix = np.zeros([sigma.shape[0], sigma.shape[0], Cl_matrix.shape[0],
                                  Cl_matrix.shape[1], Cl_matrix.shape[2]])
        for i in range(sigma.shape[0]):
            for ii in range(sigma.shape[0]):
                YY_cmb_matrix[i, ii] = np.einsum(
                    'ij,ijl,ij->ijl', W_dB_cmb[i].T, Cl_matrix, W_dB_cmb[ii])

        YY_cmb_matrix2 = np.zeros([sigma.shape[0], sigma.shape[0], Cl_matrix.shape[0],
                                   Cl_matrix.shape[1], Cl_matrix.shape[2]])
        for i in range(sigma.shape[0]):
            for ii in range(sigma.shape[0]):
                YY_cmb_matrix2[i, ii] = np.einsum(
                    'ij,ijl,ij->ijl', W_dB_cmb[i].T, Cl_matrix2, W_dB_cmb[ii])
        V_cmb = np.einsum('ij,ij...->...', sigma, W_dBdB_cmb[:, :])
        yz_cmb = np.einsum('ji,jkl,km->iml', V_cmb, Cl_matrix, WA_cmb)
        yz_cmb2 = np.einsum('ji,jkl,km->iml', V_cmb, Cl_matrix2, WA_cmb)
        # zy_cmb = np.einsum('ji,jkl,km->iml', WA_cmb, Cl_matrix, V_cmb)

        Yy_cmb = np.zeros([sigma.shape[0], Cl_matrix.shape[0],
                           Cl_matrix.shape[1], Cl_matrix.shape[2]])
        for i in range(sigma.shape[0]):
            Yy_cmb[i] = np.einsum('ij,jkl,km->iml', W_dB_cmb[i].T, Cl_matrix2, WA_cmb)

        Yz_cmb = np.zeros([sigma.shape[0], Cl_matrix.shape[0],
                           Cl_matrix.shape[1], Cl_matrix.shape[2]])
        for i in range(sigma.shape[0]):
            Yz_cmb[i] = np.einsum('ij,jkl,km->iml', W_dB_cmb[i].T, Cl_matrix2, V_cmb)

    ===========================computing alms===========================

    y_alms = get_ys_alms(y_Q=y_Q, y_U=y_U, lmax=lmax)
    Y_alms = get_ys_alms(y_Q=Y_Q, y_U=Y_U, lmax=lmax)
    z_alms = get_ys_alms(y_Q=z_Q, y_U=z_U, lmax=lmax)
    ===========================computing Cls===========================

    yy = get_ys_Cls(y_alms, y_alms, lmax, fsky)
    YY = get_ys_Cls(Y_alms, Y_alms, lmax, fsky)
    yz = get_ys_Cls(y_alms, z_alms, lmax, fsky)
    zy = get_ys_Cls(z_alms, y_alms, lmax, fsky)

    Yy = get_ys_Cls(Y_alms, y_alms, lmax, fsky)
    Yz = get_ys_Cls(Y_alms, z_alms, lmax, fsky)  # attention à checker
    ========================computing residuals========================

    stat = np.einsum('ij, ij... -> ...', sigma, YY)
    bias = yy + yz + zy
    var = stat**2 + 2 * np.einsum('i..., ij, j... -> ...', Yy, sigma, Yy)
    # Clres = bias + stat

    return stat, bias, var
'''

'''
def likelihood_exploration_new(cosmo_params, Cl_fid, Cl_residuals, Cl_noise_matrix, sigma_res, ell, fsky, WA_cmb):
    r = cosmo_params[0]

    beta = cosmo_params[1]*u.rad
    print('r', r, ' beta ', beta)
    Cl_cmb_model = np.zeros([4, Cl_fid['EE'].shape[0]])
    Cl_cmb_model[1] = copy.deepcopy(Cl_fid['EE'])
    Cl_cmb_model[2] = copy.deepcopy(Cl_fid['BlBl'])*1 + copy.deepcopy(Cl_fid['BuBu']) * r

    Cl_cmb_rot = lib.cl_rotation(Cl_cmb_model.T, beta).T

    Cl_cmb_rot_matrix = np.zeros([2, 2, Cl_cmb_rot.shape[-1]])
    Cl_cmb_rot_matrix[0, 0] = copy.deepcopy(Cl_cmb_rot[1])
    Cl_cmb_rot_matrix[1, 1] = copy.deepcopy(Cl_cmb_rot[2])
    Cl_cmb_rot_matrix[1, 0] = copy.deepcopy(Cl_cmb_rot[4])
    Cl_cmb_rot_matrix[0, 1] = copy.deepcopy(Cl_cmb_rot[4])

    # Cl_model = copy.deepcopy(Cl_cmb_rot_matrix) + copy.deepcopy(Cl_noise_matrix)

    likelihood = cosmo_likelihood_nodeprojection_new(
        Cl_cmb_rot_matrix, Cl_residuals, Cl_noise_matrix, sigma_res, ell, fsky, WA_cmb)

    return likelihood
'''

'''
def likelihood_exploration(cosmo_params, Cl_fid, Cl_data, Cl_residuals, Cl_noise_matrix, sigma_res, ell, fsky):
    r = cosmo_params[0]

    beta = cosmo_params[1]*u.rad
    print('r', r, ' beta ', beta)
    Cl_cmb_model = np.zeros([4, Cl_fid['EE'].shape[0]])
    Cl_cmb_model[1] = copy.deepcopy(Cl_fid['EE'])
    Cl_cmb_model[2] = copy.deepcopy(Cl_fid['BlBl'])*1 + copy.deepcopy(Cl_fid['BuBu']) * r
    # path_BB = '/home/baptiste/BBPipe'
    # Cl_cmb_model = copy.deepcopy(get_Cl_cmbBB(
    #     Alens=1, r=r, path_BB=path_BB))[:, ell[0]:ell[-1]+1]

    Cl_cmb_rot = lib.cl_rotation(Cl_cmb_model.T, beta).T

    Cl_cmb_rot_matrix = np.zeros([2, 2, Cl_cmb_rot.shape[-1]])
    Cl_cmb_rot_matrix[0, 0] = copy.deepcopy(Cl_cmb_rot[1])
    Cl_cmb_rot_matrix[1, 1] = copy.deepcopy(Cl_cmb_rot[2])
    Cl_cmb_rot_matrix[1, 0] = copy.deepcopy(Cl_cmb_rot[4])
    Cl_cmb_rot_matrix[0, 1] = copy.deepcopy(Cl_cmb_rot[4])

    Cl_model = copy.deepcopy(Cl_cmb_rot_matrix) + copy.deepcopy(Cl_noise_matrix)
    # IPython.embed()
    # likelihood, syst_res, stat_res = cosmo_likelihood_nodeprojection(
    # Cl_model, Cl_data, Cl_residuals, sigma_res, ell, fsky)
    likelihood = cosmo_likelihood_nodeprojection(
        Cl_model, Cl_data, Cl_residuals, sigma_res, ell, fsky)

    # return likelihood, Cl_model-(Cl_data+residuals)
    return likelihood
'''
'''
    stat_full, bias_full, var_full = get_residuals(model, fg_freq_maps_full, sigma_full, lmax, 1)

    Clres1 = stat + bias

    ============================computing Ws============================
    diff_list = get_diff_list(model)
    diff_diff_list = get_diff_diff_list(model)
    # IPython.embed()
    start = time.time()
    W = get_W(model)
    diff_W, invANAdBpBt_list, A_i_list, AitNm1_list, term1_list, term2_list = get_diff_W(
        model, diff_list, W=W, invAtNm1A=None, return_elements=True)
    diff_diff_W = get_diff_diff_W(model, diff_list, diff_diff_list, W, invANAdBpBt_list, A_i_list,
                                  AitNm1_list, term1_list, term2_list, invAtNm1A=None)
    print('time W, WdB, WdBdB = ', time.time()-start)
    diff_W = np.array(diff_W)

    ===========================test fgbuster===========================
    noise_diag = []
    for i in range(6):
        noise_diag.append(model.inv_noise[i*2, i*2])
    inv_noise6 = np.diag(noise_diag)
    if not model.fix_temp:
        A_dB_maxL = model.A.diff(model.frequencies,
                                 model.spectral_indices[0], model.spectral_indices[1], model.spectral_indices[2])
        A_dBdB_maxL = model.A.diff_diff(
            model.frequencies, model.spectral_indices[0], model.spectral_indices[1], model.spectral_indices[2])
    else:
        A_dB_maxL = model.A.diff(model.frequencies,
                                 model.spectral_indices[0], model.spectral_indices[1])
        A_dBdB_maxL = model.A.diff_diff(
            model.frequencies, model.spectral_indices[0], model.spectral_indices[1])

    W_maxL = fgcos.W(model.A_, invN=inv_noise6)
    W_dB_maxL = fgcos.W_dB(model.A_, A_dB_maxL, model.A.comp_of_dB, invN=inv_noise6)
    W_dBdB_maxL = fgcos.W_dBdB(model.A_, A_dB_maxL, A_dBdB_maxL,
                               model.A.comp_of_dB, invN=inv_noise6)

    # if not model.fix_temp:
    #     delta_W_Q = ((W[::2, ::2] - W_maxL)/W_maxL)[0]
    #     delta_W_U = ((W[1::2, 1::2] - W_maxL)/W_maxL)[0]
    #
    #     delta_diff_W_Q = ((diff_W[-2:, ::2, ::2] - W_dB_maxL[::2])/W_dB_maxL[::2])[0]
    #     delta_diff_W_U = ((diff_W[-2:, 1::2, 1::2] - W_dB_maxL[::2])/W_dB_maxL[::2])[0]
    #
    #     delta_diffdiff_W_Q = ((diff_diff_W[-2:, -2:, ::2, ::2] -
    #                            W_dBdB_maxL[::2, ::2])/W_dBdB_maxL[::2, ::2])[0]
    #     delta_diffdiff_W_U = ((diff_diff_W[-2:, -2:, 1::2, 1::2] -
    #                            W_dBdB_maxL[::2, ::2])/W_dBdB_maxL[::2, ::2])[0]
    # else:
    #     delta_W_Q = (W[0, ::2] - W_maxL[0])/W_maxL[0]
    #     delta_W_U = (W[1, 1::2] - W_maxL[0])/W_maxL[0]
    #
    #     delta_diff_W_Q = (diff_W[-2:, 0, ::2] - W_dB_maxL[:, 0])/W_dB_maxL[:, 0]
    #     delta_diff_W_U = (diff_W[-2:, 1, 1::2] - W_dB_maxL[:, 0])/W_dB_maxL[:, 0]
    #
    #     delta_diffdiff_W_Q = (diff_diff_W[-2:, -2:, 0, ::2] -
    #                           W_dBdB_maxL[:, :, 0])/W_dBdB_maxL[:, :, 0]
    #     delta_diffdiff_W_U = (diff_diff_W[-2:, -2:, 1, 1::2] -
    #                           W_dBdB_maxL[:, :, 0])/W_dBdB_maxL[:, :, 0]
    #
    # print('biggest diff in W_cmb Q w/ fgbuster = ', np.max(np.abs(delta_W_Q)))
    # print('biggest diff in W_cmb U w/ fgbuster = ', np.max(np.abs(delta_W_U)))
    # print('biggest diff in dB_W_cmb Q w/ fgbuster = ', np.max(np.abs(delta_diff_W_Q)))
    # print('biggest diff in dB_W_cmb U w/ fgbuster = ', np.max(np.abs(delta_diff_W_U)))
    # print('biggest diff in dBdB_W_cmb Q w/ fgbuster = ',
    #       np.max(np.abs(delta_diffdiff_W_Q)))
    # print('biggest diff in dBdB_W_cmb U w/ fgbuster = ',
    #       np.max(np.abs(delta_diffdiff_W_U)))

    ===========================foregrounds Cls===========================
    # fg_freq_maps = data.miscal_matrix.dot(data.mixing_matrix)[:, 2:].dot(data.signal[2:])
    # fg_freq_maps = data.mixing_matrix[:, 2:].dot(data.signal[2:])

    fg_freq_maps_reshaped = np.reshape(
        fg_freq_maps, [fg_freq_maps.shape[0]//2, 2, fg_freq_maps.shape[-1]])
    d_spectra = np.ones(
        (len(data.frequencies), 3, fg_freq_maps.shape[-1]), dtype=fg_freq_maps.dtype)
    d_spectra[:, 1:] = fg_freq_maps_reshaped
    d_spectra[:, 0] *= mask  # mask is already applied on fg_freq_maps, only T elements need it

    fg_freq_maps_reshaped_full = np.reshape(
        fg_freq_maps_full, [fg_freq_maps.shape[0]//2, 2, fg_freq_maps.shape[-1]])
    d_spectra_full = np.ones(
        (len(data.frequencies), 3, fg_freq_maps.shape[-1]), dtype=fg_freq_maps.dtype)
    d_spectra_full[:, 1:] = fg_freq_maps_reshaped_full

    # Compute cross-spectra
    start = time.time()

    n_freqs = len(data.frequencies)

    almBs = [hp.map2alm(freq_map, lmax=lmax, iter=10)[2] for freq_map in d_spectra]
    Cl_fgs = np.zeros((n_freqs, n_freqs, lmax+1), dtype=fg_freq_maps_reshaped.dtype)
    for f1 in range(n_freqs):
        for f2 in range(n_freqs):
            if f1 > f2:
                Cl_fgs[f1, f2] = Cl_fgs[f2, f1]
            else:
                Cl_fgs[f1, f2] = hp.alm2cl(almBs[f1], almBs[f2], lmax=lmax)
    Cl_fgs = Cl_fgs[..., lmin:] / fsky
    # print('WARNING FSKY !!! Cl_fgs')
    print('time fg spectra = ', time.time()-start)

    =============================xForecast=============================
    components = [CMB(), Dust(model.dust_freq, temp=20),
                  Synchrotron(model.synchrotron_freq)]
    # instrument_xforecast = {'frequency': model.frequencies, 'Sens_P': V3.so_V3_SA_noise(
    #     2, 2, 1, 0.1, nside*3)[2]}

    class instrument_xforecast:
        def __init__(self, nside=128):
            self.nside = nside
            self.Frequencies = np.array([27,  39,  93, 145, 225, 280])
            self.frequency = np.array([27,  39,  93, 145, 225, 280])
            # self.white_noise = V3.so_V3_SA_noise(2, 2, 1, 0.1, nside*3)[2]
            self.Sens_P = V3.so_V3_SA_noise(2, 2, 1, 0.1, nside*3)[2]
            self.Beams = [0]*6  # [hp.nside2resol(self.nside, arcmin=True)]*6
            self.fsky = 0.1
    test_class_instru = instrument_xforecast(nside=nside)
    # fgcos.xForecast(components, instrument_xforecast, d_spectra, lmin=0, lmax=lmax)
    # d_spectra1 = copy.deepcopy(d_spectra)
    # d_spectra1[0, 0] = np.ones(d_spectra.shape[-1])
    res_xForecast = fgcos.xForecast(components, test_class_instru,
                                    d_spectra[:, 1:, :], lmin=lmin, lmax=lmax, r=0)
    res_xForecast_full = fgcos.xForecast(components, test_class_instru,
                                         d_spectra_full[:, 1:, :], lmin=lmin, lmax=lmax, r=0)
    # Cl_xForecast = res_xForecast.stat + res_xForecast.bias

    if res_xForecast.Sigma.shape[0] == 3:
        sigmaX = res_xForecast.Sigma[::2, ::2]
    else:
        sigmaX = res_xForecast.Sigma  # [::2, ::2]
    #
    # fisher_matrix = fshp.fisher(ddtPnoise_masked_cleaned, model, diff_list, diff_diff_list)
    # prior_indices = [0, 6]
    # prior_precision = (1 * u.arcmin).to(u.rad).value
    #
    # fisher_prior = copy.deepcopy(fisher_matrix)
    # for i in range(prior_indices[0], prior_indices[-1]):
    #     # print(i)
    #     fisher_prior[i, i] += 1/prior_precision**2
    # sigma = np.linalg.inv(fisher_prior)

    =============V computation and comparaison with fgbuster=============
    # V_Q = np.einsum('ij,ij...->...', sigma, diff_diff_W[:, :, 0])
    # V_U = np.einsum('ij,ij...->...', sigma, diff_diff_W[:, :, 1])

    # QU_dBdB_delta = diff_diff_W[:, :, 0, ::2] - diff_diff_W[:, :, 1, 1::2]

    # sigma_noB = np.delete(np.delete(sigma, 6, 0), 6, 1)
    # diff_diff_W_noB = np.delete(np.delete(diff_diff_W, 6, 0), 6, 1)
    # V_Q_noB = np.einsum('ij,ij...->...', sigma_noB, diff_diff_W_noB[:, :, 0])
    # V_U_noB = np.einsum('ij,ij...->...', sigma_noB, diff_diff_W_noB[:, :, 1])

    # V_fg = np.einsum('ij,ij...->...', sigmaX, W_dBdB_maxL[::2, ::2, 0])
    #
    # V_comfgQ = np.einsum('ij,ij...->...', sigmaX, diff_diff_W[-2:, -2:, 0])
    # V_comfgU = np.einsum('ij,ij...->...', sigmaX, diff_diff_W[-2:, -2:, 1])
    #
    # delta_V_Q = (V_fg-V_comfgQ[::2])/V_fg
    # delta_V_U = (V_fg-V_comfgU[1::2])/V_fg
    # print('biggest diff in V Q w/ fgbuster = ', np.max(np.abs(delta_V_Q)))
    # print('biggest diff in V U w/ fgbuster = ', np.max(np.abs(delta_V_U)))

    ===========================Computing ys===========================
    print('WARNING FSKY !!!!')

    y_Q1 = W[0].dot(fg_freq_maps)
    y_U1 = W[1].dot(fg_freq_maps)
    Y_Q1 = diff_W[:, 0].dot(fg_freq_maps)
    Y_U1 = diff_W[:, 1].dot(fg_freq_maps)
    # z_Q1 = V_Q.dot(fg_freq_maps)
    # z_U1 = V_U.dot(fg_freq_maps)

    ===========================computing alms===========================

    y_alms1 = get_ys_alms(y_Q=y_Q1, y_U=y_U1, lmax=lmax)
    # Y_alms1 = get_ys_alms(y_Q=Y_Q1, y_U=Y_U1, lmax=lmax)
    # z_alms1 = get_ys_alms(y_Q=z_Q1, y_U=z_U1, lmax=lmax)
    ===========================computing Cls===========================

    yy1 = get_ys_Cls(y_alms1, y_alms1, lmax, fsky)
    # YY1 = get_ys_Cls(Y_alms1, Y_alms1, lmax)
    # yz1 = get_ys_Cls(y_alms1, z_alms1, lmax)
    # zy1 = get_ys_Cls(z_alms1, y_alms1, lmax)
    #
    # Yy1 = get_ys_Cls(Y_alms1, y_alms1, lmax)
    # Yz1 = get_ys_Cls(Y_alms1, z_alms1, lmax)
    ========================computing residuals========================
    #
    # stat1 = np.einsum('ij, ij... -> ...', sigma, YY1)
    # bias1 = yy1 + yz1 + zy1
    # var1 = stat1**2 + 2 * np.einsum('i..., ij, j... -> ...', Yy1, sigma, Yy1)
    # Clres1 = bias1 + stat1

    =====================computing residuals fg comp=====================


    # yy1_fgcomp1 = get_ys_Cls(y_alms1[::2], y_alms1[::2], lmax)

    Y_alms_fgcomp1 = get_ys_alms(y_Q=Y_Q1[-2:], y_U=Y_U1[-2:], lmax=lmax)
    YY_fgcomp1 = get_ys_Cls(Y_alms_fgcomp1, Y_alms_fgcomp1, lmax, fsky)
    Yy_gfcomp1 = get_ys_Cls(Y_alms_fgcomp1, y_alms1, lmax, fsky)

    V_Q_fgcomp1 = np.einsum('ij,ji...->...', sigmaX, diff_diff_W[-2:, -2:, 0])
    V_U_fgcomp1 = np.einsum('ij,ji...->...', sigmaX, diff_diff_W[-2:, -2:, 1])
    z_Q_fgcomp1 = V_Q_fgcomp1.dot(fg_freq_maps)
    z_U_fgcomp1 = V_U_fgcomp1.dot(fg_freq_maps)
    z_alms_fgcomp1 = get_ys_alms(y_Q=z_Q_fgcomp1, y_U=z_U_fgcomp1, lmax=lmax)
    yz_fgcomp1 = get_ys_Cls(y_alms1, z_alms_fgcomp1, lmax, fsky)
    zy_fgcomp1 = get_ys_Cls(z_alms_fgcomp1, y_alms1, lmax, fsky)

    bias_fgcomp1 = yy1 + yz_fgcomp1 + zy_fgcomp1
    bias_fgcomp2 = yy1 + 2*yz_fgcomp1

    stat_fgcomp1 = np.einsum('ij, ij... -> ...', sigmaX, YY_fgcomp1)
    var_fgcomp1 = stat_fgcomp1**2 + 2 * np.einsum('i..., ij, j... -> ...',
                                                  Yy_gfcomp1, sigmaX, Yy_gfcomp1)
    Clres_fgcomp1 = bias_fgcomp1 + stat_fgcomp1

    ====================computing fgbuster residuals====================
    V_maxL = np.einsum('ij,ij...->...', sigma[-2:, -2:], W_dBdB_maxL[::2, ::2, 0])
    Cl_xF = {}
    Cl_xF['yy'] = fgcos._utmv(W_maxL[0], Cl_fgs.T, W_maxL[0])  # (ell,)
    Cl_xF['YY'] = fgcos._mmm(W_dB_maxL[::2, 0], Cl_fgs.T,
                             W_dB_maxL[::2, 0].T)  # (ell, param, param)
    Cl_xF['yz'] = fgcos._utmv(W_maxL[0], Cl_fgs.T, V_maxL)  # (ell,)
    Cl_xF['zy'] = fgcos._utmv(V_maxL, Cl_fgs.T,  W_maxL[0])  # (ell,)

    Cl_xF['Yy'] = fgcos._mmv(W_dB_maxL[::2, 0], Cl_fgs.T, W_maxL[0])  # (ell, param)
    Cl_xF['Yz'] = fgcos._mmv(W_dB_maxL[::2, 0], Cl_fgs.T, V_maxL)  # (ell, param)

    bias_fgbuster = Cl_xF['yy'] + Cl_xF['yz'] + Cl_xF['zy']
    stat_fgbuster = np.einsum('ij, lij -> l', sigma[-2:, -2:], Cl_xF['YY'])
    Clres_fgbuster = bias_fgbuster + stat_fgbuster

    ====================computing fgbuster residuals====================
    # V_maxL_all = np.einsum('ij,ij...->...', res_xForecast.Sigma, W_dBdB_maxL[:, :, 0])
    # Cl_xF_all = {}
    # Cl_xF_all['yy'] = fgcos._utmv(W_maxL[0], Cl_fgs.T, W_maxL[0])  # (ell,)
    # Cl_xF_all['YY'] = fgcos._mmm(W_dB_maxL[:, 0], Cl_fgs.T,
    #                              W_dB_maxL[:, 0].T)  # (ell, param, param)
    # Cl_xF_all['yz'] = fgcos._utmv(W_maxL[0], Cl_fgs.T, V_maxL_all)  # (ell,)
    # Cl_xF_all['zy'] = fgcos._utmv(V_maxL_all, Cl_fgs.T,  W_maxL[0])  # (ell,)
    #
    # Cl_xF_all['Yy'] = fgcos._mmv(W_dB_maxL[:, 0], Cl_fgs.T, W_maxL[0])  # (ell, param)
    # Cl_xF_all['Yz'] = fgcos._mmv(W_dB_maxL[:, 0], Cl_fgs.T, V_maxL_all)  # (ell, param)
    #
    # bias_fgbuster_all = Cl_xF_all['yy'] + Cl_xF_all['yz'] + Cl_xF_all['zy']
    # stat_fgbuster_all = np.einsum('ij, lij -> l', res_xForecast.Sigma, Cl_xF_all['YY'])
    # Clres_fgbuster_all = bias_fgbuster_all + stat_fgbuster_all
    #
    # plt.plot(Clres_fgbuster, label='fgbuster')
    # # plt.plot(Clres_fgcomp[1], label='comp')
    # plt.plot(Clres_fgcomp1[1], label='comp1')
    #
    # plt.xscale('log')
    # plt.yscale('symlog')
    # plt.legend()
    # plt.show()
    #
    # plt.plot(bias_fgbuster, label='bias fgbuster')
    # # plt.plot(bias_fgcomp[1], label='bias comp')
    # plt.plot(bias_fgcomp1[1], label='bias comp1')
    #
    # plt.xscale('log')
    # plt.yscale('symlog')
    # plt.legend()
    # plt.show()
    #
    # plt.plot(stat_fgbuster, label='stat fgbuster')
    # # plt.plot(stat_fgcomp[1], label='stat comp')
    # plt.plot(stat_fgcomp1[1], label='stat comp1')
    #
    # plt.xscale('log')
    # plt.yscale('symlog')
    # plt.legend()
    # plt.show()
    #

    =========================computing CMB Cls=========================
    cmb_maps = np.zeros(
        (3, data.signal.shape[-1]), dtype=data.signal.dtype)
    cmb_maps[1:] = data.signal[:2]
    cmb_maps *= mask
    cmb_tot_cls = hp.anafast(cmb_maps, lmax=lmax, iter=10)
    Clcmb = [cmb_tot_cls[1], cmb_tot_cls[2], cmb_tot_cls[4]]

    =================computing fgbuster comp xForecast=================
    # sigmaX = res_xForecast.Sigma  # [::2, ::2]
    Cl_xFX = {}

    if not model.fix_temp:
        V_maxLX = np.einsum('ij,ij...->...', sigmaX, W_dBdB_maxL[::2, ::2, 0])
        Cl_xFX['YY'] = fgcos._mmm(W_dB_maxL[::2, 0], Cl_fgs.T,
                                  W_dB_maxL[::2, 0].T)  # (ell, param, param)
        Cl_xFX['Yy'] = fgcos._mmv(W_dB_maxL[::2, 0], Cl_fgs.T, W_maxL[0])  # (ell, param)
        Cl_xFX['Yz'] = fgcos._mmv(W_dB_maxL[::2, 0], Cl_fgs.T, V_maxLX)  # (ell, param)

    else:
        V_maxLX = np.einsum('ij,ij...->...', sigmaX, W_dBdB_maxL[:, :, 0])
        Cl_xFX['YY'] = fgcos._mmm(W_dB_maxL[:, 0], Cl_fgs.T,
                                  W_dB_maxL[:, 0].T)  # (ell, param, param)
        Cl_xFX['Yy'] = fgcos._mmv(W_dB_maxL[:, 0], Cl_fgs.T, W_maxL[0])  # (ell, param)
        Cl_xFX['Yz'] = fgcos._mmv(W_dB_maxL[:, 0], Cl_fgs.T, V_maxLX)  # (ell, param)

    Cl_xFX['yy'] = fgcos._utmv(W_maxL[0], Cl_fgs.T, W_maxL[0])  # (ell,)
    Cl_xFX['yz'] = fgcos._utmv(W_maxL[0], Cl_fgs.T, V_maxLX)  # (ell,)
    Cl_xFX['zy'] = fgcos._utmv(V_maxLX, Cl_fgs.T,  W_maxL[0])  # (ell,)

    bias_fgbusterX = Cl_xFX['yy'] + Cl_xFX['yz'] + Cl_xFX['zy']
    bias_fgbusterX2 = Cl_xFX['yy'] + 2*Cl_xFX['yz']

    stat_fgbusterX = np.einsum('ij, lij -> l', sigmaX, Cl_xFX['YY'])
    Clres_fgbusterX = bias_fgbusterX + stat_fgbusterX
    var_fgbusterX = stat_fgbusterX**2 + 2 * np.einsum('li, ij, lj -> l',  # S16, Eq. 28
                                                      Cl_xFX['Yy'], sigmaX, Cl_xFX['Yy'])

    #
    # plt.plot(res_xForecast.stat, label='stat xForecast')
    # plt.plot(stat_fgbuster, label='stat fgbuster')
    # plt.plot(stat_fgbusterX, label='stat fgbusterX')
    # # plt.plot(stat_fgcomp[1], label='stat comp')
    # plt.plot(stat_fgcomp1[1], label='stat comp1')
    #
    # plt.xscale('log')
    # plt.yscale('symlog')
    # plt.legend()
    # plt.show()
    #
    # plt.plot(res_xForecast.bias, label='bias xForecast')
    # plt.plot(bias_fgbuster, label='bias fgbuster')
    # plt.plot(bias_fgbusterX, label='bias fgbusterX')
    # # plt.plot(bias_fgcomp[1], label='bias comp')
    # plt.plot(bias_fgcomp1[1], label='bias comp1')
    #
    # plt.xscale('log')
    # plt.yscale('symlog')
    # plt.legend()
    # plt.show()
    #
    lmin_plot = 2
    lmax_plot = 2*nside
    ell = np.arange(lmin_plot, lmax_plot)

    plt.plot(res_xForecast.stat[lmin_plot:lmax_plot]*ell*(ell+1), label='stat xForecast')
    plt.plot(stat_fgbusterX[lmin_plot:lmax_plot]*ell*(ell+1), label='stat à la xForecast')
    plt.plot(stat_fgcomp1[1][lmin_plot:lmax_plot]*ell*(ell+1), label='stat comp')
    plt.xlabel(r'$\ell$')
    plt.ylabel(r'$C_\ell \frac{\ell(\ell+1)}{2\pi} $')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.show()

    plt.plot(res_xForecast.bias[lmin_plot:lmax_plot]*ell*(ell+1), label='bias xForecast')
    plt.plot(bias_fgbusterX[lmin_plot:lmax_plot]*ell*(ell+1), label='bias à la xForecast')
    plt.plot(bias_fgcomp1[1][lmin_plot:lmax_plot]*ell*(ell+1), label='bias comp')
    plt.xlabel(r'$\ell$')
    plt.ylabel(r'$C_\ell \frac{\ell(\ell+1)}{2\pi} $')
    plt.xscale('log')
    plt.yscale('symlog')
    plt.legend()
    plt.show()

    plt.plot(res_xForecast.var[lmin_plot:lmax_plot]*ell*(ell+1), label='var xForecast')
    plt.plot(var_fgbusterX[lmin_plot:lmax_plot]*ell*(ell+1), label='var à la xForecast')
    plt.plot(var_fgcomp1[1][lmin_plot:lmax_plot]*ell*(ell+1), label='var comp')
    plt.xlabel(r'$\ell$')
    plt.ylabel(r'$C_\ell \frac{\ell(\ell+1)}{2\pi} $')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.show()

    delta_stat_xforVSfgX = np.max(
        np.abs(((res_xForecast.stat - stat_fgbusterX)/res_xForecast.stat)[2:]))
    delta_stat_xforVStot = np.max(
        np.abs(((res_xForecast.stat - stat_fgcomp1[1])/res_xForecast.stat)[2:]))
    delta_stat_fgXVStot = np.max(np.abs(((stat_fgbusterX - stat_fgcomp1[1])/stat_fgbusterX)[2:]))

    delta_bias_xforVSfgX = np.max(
        np.abs(((res_xForecast.bias - bias_fgbusterX)/res_xForecast.bias)[2:]))
    delta_bias_xforVStot = np.max(
        np.abs(((res_xForecast.bias - bias_fgcomp1[1])/res_xForecast.bias)[2:]))
    delta_bias_fgXVStot = np.max(np.abs(((bias_fgbusterX - bias_fgcomp1[1])/bias_fgbusterX)[2:]))

    print('')
    print('delta_stat_xforVSfgX = ', delta_stat_xforVSfgX)
    print('delta_stat_xforVStot = ', delta_stat_xforVStot)
    print('delta_stat_fgXVStot = ', delta_stat_fgXVStot)
    print('')
    print('delta_bias_xforVSfgX = ', delta_bias_xforVSfgX)
    print('delta_bias_xforVStot = ', delta_bias_xforVStot)
    print('delta_bias_fgXVStot = ', delta_bias_fgXVStot)

    plt.plot(Clres1[0], label='EE')
    plt.plot(Clres1[1], label='BB')
    plt.plot(Clres1[2], label='EB')
    plt.xscale('log')
    plt.yscale('symlog')
    plt.legend()
    plt.show()

    Clres_xforecast = res_xForecast.bias + res_xForecast.stat

    # plt.plot(ell, Clres1[0][lmin_plot:lmax_plot]*ell*(ell+1), label='residues EE')
    plt.plot(ell, ps_planck[1][lmin_plot:lmax_plot]*ell*(ell+1), label='CMB EE')
    plt.plot(ell, stat[0][lmin_plot:lmax_plot]*ell*(ell+1),
             label='stat EE', color='orange', linestyle='--')
    plt.plot(ell, bias[0][lmin_plot:lmax_plot]*ell*(ell+1),
             label='syst EE', color='orange', linestyle=':')
    plt.plot(ell, var[0][lmin_plot:lmax_plot]*ell*(ell+1),
             label='var EE', color='orange', linestyle='-.')
    plt.plot(ell, stat_noCMB[0][lmin_plot:lmax_plot]*ell*(ell+1),
             label='stat EE noCMB', color='red', linestyle='--')
    plt.plot(ell, bias_noCMB[0][lmin_plot:lmax_plot]*ell*(ell+1),
             label='syst EE noCMB', color='red', linestyle=':')
    plt.plot(ell, var_noCMB[0][lmin_plot:lmax_plot]*ell*(ell+1),
             label='var EE noCMB', color='red', linestyle='-.')
    # plt.plot(ell, stat2[0][lmin_plot:lmax_plot]*ell*(ell+1),
    #          label='stat2 EE', color='green', linestyle='--')
    # plt.plot(ell, bias2[0][lmin_plot:lmax_plot]*ell*(ell+1),
    #          label='syst2 EE', color='green', linestyle=':')
    # plt.plot(ell, var2[0][lmin_plot:lmax_plot]*ell*(ell+1),
    #          label='var2 EE', color='green', linestyle='-.')
    # plt.plot(ell, Clres1[1][lmin_plot:lmax_plot]*ell*(ell+1), label='residues BB')
    plt.plot(ell, ps_planck[2][lmin_plot:lmax_plot]*ell*(ell+1), label='CMB BB')
    plt.plot(ell, stat[1][lmin_plot:lmax_plot]*ell*(ell+1),
             label='stat BB', color='orange', linestyle='--')
    plt.plot(ell, bias[1][lmin_plot:lmax_plot]*ell*(ell+1),
             label='syst BB', color='orange', linestyle=':')
    plt.plot(ell, var[1][lmin_plot:lmax_plot]*ell*(ell+1),
             label='var BB', color='orange', linestyle='-.')
    plt.plot(ell, stat_noCMB[1][lmin_plot:lmax_plot]*ell*(ell+1),
             label='stat BB noCMB', color='red', linestyle='--')
    plt.plot(ell, bias_noCMB[1][lmin_plot:lmax_plot]*ell*(ell+1),
             label='syst BB noCMB', color='red', linestyle=':')
    plt.plot(ell, var_noCMB[1][lmin_plot:lmax_plot]*ell*(ell+1),
             label='var BB noCMB', color='red', linestyle='-.')

    # plt.plot(ell, stat2[1][lmin_plot:lmax_plot]*ell*(ell+1),
    #          label='stat2 BB', color='green', linestyle='--')
    # plt.plot(ell, bias2[1][lmin_plot:lmax_plot]*ell*(ell+1),
    #          label='syst2 BB', color='green', linestyle=':')
    # plt.plot(ell, var2[1][lmin_plot:lmax_plot]*ell*(ell+1),
    #          label='var2 BB', color='green', linestyle='-.')
    plt.plot(ell, ps_planck[4][lmin_plot:lmax_plot]*ell*(ell+1), label='CMB EB')
    plt.plot(ell, stat[2][lmin_plot:lmax_plot]*ell*(ell+1),
             label='stat EB', color='orange', linestyle='--')
    plt.plot(ell, bias[2][lmin_plot:lmax_plot]*ell*(ell+1),
             label='syst EB', color='orange', linestyle=':')
    plt.plot(ell, var[2][lmin_plot:lmax_plot]*ell*(ell+1),
             label='var EB', color='orange', linestyle='-.')
    plt.plot(ell, stat_noCMB[2][lmin_plot:lmax_plot]*ell*(ell+1),
             label='stat EB noCMB', color='red', linestyle='--')
    plt.plot(ell, bias_noCMB[2][lmin_plot:lmax_plot]*ell*(ell+1),
             label='syst EB noCMB', color='red', linestyle=':')
    plt.plot(ell, var_noCMB[2][lmin_plot:lmax_plot]*ell*(ell+1),
             label='var EB noCMB', color='red', linestyle='-.')
    # plt.plot(ell, stat2[2][lmin_plot:lmax_plot]*ell*(ell+1),
    #          label='stat2 EB', color='green', linestyle='--')
    # plt.plot(ell, bias2[2][lmin_plot:lmax_plot]*ell*(ell+1),
    #          label='syst2 EB', color='green', linestyle=':')
    # plt.plot(ell, var2[2][lmin_plot:lmax_plot]*ell*(ell+1),
    #          label='var2 EB', color='green', linestyle='-.')
    # plt.plot(ell, Clres_xforecast[lmin_plot:lmax_plot]*ell *
    #          (ell+1), label='residues xForecast BB', color='m')
    plt.plot(ell, res_xForecast.stat[lmin_plot:lmax_plot]*ell *
             (ell+1), label='stat xForecast BB', color='m', linestyle='--')
    plt.plot(ell, res_xForecast.bias[lmin_plot:lmax_plot]*ell *
             (ell+1), label='syst xForecast BB', color='m', linestyle=':')
    plt.xlabel(r'$\ell$')
    plt.ylabel(r'$C_\ell \frac{\ell(\ell+1)}{2\pi} $')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.show()

    plt.plot(ell, Clcmb[2][lmin_plot:lmax_plot]*ell*(ell+1), label='CMB EB')
    plt.plot(ell, Clres1[2][lmin_plot:lmax_plot]*ell*(ell+1), label='residues EB')

    plt.xlabel(r'$\ell$')
    plt.xscale('log')
    plt.yscale('symlog')
    plt.ylabel(r'$C_\ell \frac{\ell(\ell+1)}{2\pi} $')
    plt.legend()
    plt.show()

    plt.plot(ell, (stat_full+bias_full)[0][lmin_plot:lmax_plot]
             * ell*(ell+1), label='residuals full EE', color='orange')
    plt.plot(ell, (stat+bias)[0][lmin_plot:lmax_plot]*ell*(ell+1),
             label='residuals partial EE', color='orange', linestyle='--')
    plt.plot(ell, (stat_full+bias_full)[1][lmin_plot:lmax_plot]
             * ell*(ell+1), label='residuals full BB', color='red')
    plt.plot(ell, (stat+bias)[1][lmin_plot:lmax_plot]*ell*(ell+1),
             label='residuals partial BB', color='red', linestyle='--')
    # plt.plot(ell, (stat_full+bias_full)[2][lmin_plot:lmax_plot]
    #          * ell*(ell+1), label='residuals full EB', color='g')
    # plt.plot(ell, (stat+bias)[2][lmin_plot:lmax_plot]*ell*(ell+1),
    #          label='residuals partial EB', color='g', linestyle='--')

    plt.plot(ell, (res_xForecast_full.stat + res_xForecast_full.bias)
             [lmin_plot:lmax_plot]*ell*(ell+1), label='xforecast full', color='m')
    plt.plot(ell, (res_xForecast.stat + res_xForecast.bias)
             [lmin_plot:lmax_plot]*ell*(ell+1), label='xforecast partial', color='m', linestyle='--')
    plt.xlabel(r'$\ell$')
    plt.ylabel(r'$C_\ell \frac{\ell(\ell+1)}{2\pi} $')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.show()
    # Bd_grid = np.arange(1.5895, 1.5905, 0.001/100)
    # Bd_grid = np.arange(1.59-1e-3, 1.59+1e-3, 1e-3/1000)
    #
    # L_grid = []
    # angle_array = [0, 0, 0, 0, 0, 0, 0, 1.59, -3]
    # from pixel_based_angle_estimation import get_chi_squared_local
    # for i in Bd_grid:
    #     new_angle_array = angle_array
    #     new_angle_array[-2] = i
    #     L_grid.append(get_chi_squared_local(new_angle_array,
    #                                         ddtPnoise, model, spectral_index=True))

    import bjlib.class_faraday as cf

    ps = cf.power_spectra_operation()
    ps.get_spectra()
    ps.spectra.normalisation = 1

    ps_planck = hp.read_cl(path_BB + '/test_mapbased_param/Cls_Planck2018_lensed_scalar.fits')
    ps_planck_unlensed = hp.read_cl(
        path_BB + '/test_mapbased_param/Cls_Planck2018_unlensed_scalar_and_tensor_r1.fits')

    plt.plot(ell, Clcmb[0][lmin_plot:lmax_plot]*ell*(ell+1)/fsky, label='CMB EE', color='g')
    plt.plot(ell, ps.spectra.spectra[lmin_plot:lmax_plot, 1], label='CAMB EE')
    plt.plot(ell, ps_planck[1, lmin_plot:lmax_plot]*ell*(ell+1), label='planck EE')
    plt.loglog()
    plt.legend()
    plt.show()

    plt.plot(ell, Clcmb[1][lmin_plot:lmax_plot]*ell*(ell+1)/fsky, label='CMB BB', color='g')
    plt.plot(ell, ps.spectra.spectra[lmin_plot:lmax_plot, 2], label='CAMB BB')
    plt.plot(ell, ps_planck[2, lmin_plot:lmax_plot]*ell*(ell+1), label='planck BB')
    plt.loglog()
    plt.legend()
    plt.show()
    # carefull not to use model used for data, here a bias can arise if there is a difference between miscal angles fitted and true ones

    cmb_spectra = ps_planck
    WA_cmb = W[:2].dot(model_data.mix_effectiv[:, :2]) - np.identity(2)
    W_dB_cmb = diff_W[:, 2, :].dot(model_data.mix_effectiv[:, :2])
    W_dBdB_cmb = diff_diff_W[:, :, 2, :].dot(model_data.mix_effectiv[:, :2])

    IPython.embed()
    start = time.time()
    map_list = []
    for i in range(100):
        map_list.append(hp.synfast(cmb_spectra, 128, new=True))
    map_list = np.array(map_list)
    print('time cmb sim = ', time.time() - start)

    start = time.time()
    map_array_reshaped = np.einsum('ijk->jik', map_list)
    mapxmap = np.einsum('i...,j...->ij...', map_array_reshaped[1:], map_array_reshaped[1:])
    mean_overcmb = np.mean(mapxmap, axis=2)
    S_cmb = np.sum(mean_overcmb, axis=-1)
    print('time S computation=', time.time()-start)
    ASAt = model.mix_effectiv[:, :2].dot(S_cmb).dot(model.mix_effectiv[:, :2].T)
    # a = np.einsum('...ij,jk...->...ik', map_list[:, 1:], map_list[:, 1:].T)
    # sigma_cmb = np.mean(a, axis=0)/hp.nside2npix(128)

    ddt_fg = np.einsum('ik...,...kj->ijk', fg_freq_maps_full, fg_freq_maps_full.T)
    ddt_fg *= mask
    F = np.sum(ddt_fg, axis=-1)
    '''


# delta_diff_W_Q = []
# delta_diff_W_U = []
# list_dB_fg = [0, 2]
# list_dB = [-2, -1]
# for comp in range(3):
#     delta_diff_W_Q_list_dB = []
#     delta_diff_W_U_list_dB = []
#
#     for dB in range(2):
#         delta_diff_W_Q_list_freq = []
#         delta_diff_W_U_list_freq = []
#
#         for freq in range(6):
#             delta_diff_W_Q_list_freq.append((diff_W[list_dB[dB]][comp*2][freq*2] -
#                                              W_dB_maxL[list_dB_fg[dB], comp, freq])/W_dB_maxL[list_dB_fg[dB], comp, freq])
#             delta_diff_W_U_list_freq.append((diff_W[list_dB[dB]][comp*2+1][freq*2+1] -
#                                              W_dB_maxL[list_dB_fg[dB], comp, freq])/W_dB_maxL[list_dB_fg[dB], comp, freq])
#         delta_diff_W_Q_list_dB.append(delta_diff_W_Q_list_freq)
#         delta_diff_W_U_list_dB.append(delta_diff_W_U_list_freq)
#     delta_diff_W_Q.append(delta_diff_W_Q_list_dB)
#     delta_diff_W_U.append(delta_diff_W_U_list_dB)
# delta_diff_W_Q = np.array(delta_diff_W_Q)
# delta_diff_W_U = np.array(delta_diff_W_U)

# moved_diff_Q = np.moveaxis(delta_diff_W_Q, 0, 1)
# moved_diff_U = np.moveaxis(delta_diff_W_U, 0, 1)
# print(np.sum(test_dB_Q - moved_diff_Q))
# print(np.sum(test_dB_U - moved_diff_U))
#
# delta_diffdiff_W_Q = []
# delta_diffdiff_W_U = []
#
# for comp in range(3):
#     delta_diff_diff_W_Q_list_dBi = []
#     delta_diff_diff_W_U_list_dBi = []
#
#     for dBi in range(2):
#         delta_diff_diff_W_Q_list_dBii = []
#         delta_diff_diff_W_U_list_dBii = []
#
#         for dBii in range(2):
#             delta_diff_diff_W_Q_list_freq = []
#             delta_diff_diff_W_U_list_freq = []
#
#             for freq in range(6):
#                 delta_diff_diff_W_Q_list_freq.append((diff_diff_W[list_dB[dBi]][list_dB[dBii]][comp*2][freq*2] -
#                                                       W_dBdB_maxL[list_dB_fg[dBi], list_dB_fg[dBii], comp, freq]) / W_dBdB_maxL[list_dB_fg[dBi], list_dB_fg[dBii], comp, freq])
#                 delta_diff_diff_W_U_list_freq.append((diff_diff_W[list_dB[dBi]][list_dB[dBii]][comp*2+1][freq*2+1] -
#                                                       W_dBdB_maxL[list_dB_fg[dBi], list_dB_fg[dBii], comp, freq]) / W_dBdB_maxL[list_dB_fg[dBi], list_dB_fg[dBii], comp, freq])
#             delta_diff_diff_W_Q_list_dBii.append(delta_diff_diff_W_Q_list_freq)
#             delta_diff_diff_W_U_list_dBii.append(delta_diff_diff_W_U_list_freq)
#         delta_diff_diff_W_Q_list_dBi.append(delta_diff_diff_W_Q_list_dBii)
#         delta_diff_diff_W_U_list_dBi.append(delta_diff_diff_W_U_list_dBii)
#     delta_diffdiff_W_Q.append(delta_diff_diff_W_Q_list_dBi)
#     delta_diffdiff_W_U.append(delta_diff_diff_W_U_list_dBi)
# delta_diffdiff_W_Q = np.array(delta_diffdiff_W_Q)
# delta_diffdiff_W_U = np.array(delta_diffdiff_W_U)

'''=========================ploting residuals========================='''
'''
    plt.plot(stat[0], label='stat EE')
    plt.plot(stat[1], label='stat BB')
    plt.plot(stat[2], label='stat EB')
    plt.xscale('log')
    plt.yscale('symlog')
    plt.legend()
    plt.show()
    '''
# plt.plot(bias[0], label='bias EE')
# plt.plot(bias[1], label='bias BB')
# plt.plot(bias[2], label='bias EB')
#
'''
    plt.plot(bias2[0], label='bias2 EE')
    plt.plot(bias2[1], label='bias2 BB')
    plt.plot(bias2[2], label='bias2 EB')
    plt.xscale('log')
    plt.yscale('symlog')
    plt.legend()
    plt.show()
    '''
# plt.plot(var[0], label='var EE')
# plt.plot(var[1], label='var BB')
# plt.plot(var[2], label='var EB')
#
# plt.xscale('log')
# plt.yscale('symlog')
# plt.legend()
# plt.show()
'''
    plt.plot(Clres2[0], label='Clres EE')
    plt.plot(Clres2[1], label='Clres BB')
    plt.plot(Clres2[2], label='Clres EB')
    plt.xscale('log')
    plt.yscale('symlog')
    plt.legend()
    plt.show()
    '''
'''
    # plt.plot(Clres2[0], label='Clres EE')
    plt.plot(Clres1[0], label='Clres1 EE')
    plt.plot(Clcmb[0], label='Clcmb EE')
    plt.xscale('log')
    plt.yscale('symlog')
    plt.legend()
    plt.show()

    # plt.plot(Clres2[1], label='Clres BB')
    plt.plot(Clres1[1], label='Clres1 BB')
    plt.plot(Clcmb[1], label='Clcmb BB')
    plt.xscale('log')
    plt.yscale('symlog')
    plt.legend()
    plt.show()

    # plt.plot(Clres2[2], label='Clres EB')
    plt.plot(Clres1[2], label='Clres1 EB')
    plt.plot(Clcmb[2], label='Clcmb EB')
    plt.xscale('log')
    plt.yscale('symlog')
    plt.legend()
    plt.show()
    '''
#
# plt.plot(Clres[0] - Clres2[0], label='diff Clres EE')
# plt.plot(Clres[1] - Clres2[1], label='diff Clres BB')
# plt.plot(Clres[2] - Clres2[2], label='diff Clres EB')
#
# plt.xscale('log')
# plt.yscale('symlog')
# plt.legend()
# plt.show()

# plt.plot(bias2[2]-bias[2], label='diff bias EB')

# components = [CMB(), Dust(model.dust_freq), Synchrotron(model.synchrotron_freq)]
# instrument_xforecast = {'frequency': model.frequencies, 'Sens_P': V3.so_V3_SA_noise(
#     2, 2, 1, 0.1, nside*3)[2]}
#
# class instrument_xforecast:
#     def __init__(self, nside=128):
#         self.nside = nside
#         self.Frequencies = np.array([27,  39,  93, 145, 225, 280])
#         self.frequency = np.array([27,  39,  93, 145, 225, 280])
#         self.Sens_P = V3.so_V3_SA_noise(2, 2, 1, 0.1, nside*3)[2]
#         self.Beams = [hp.nside2resol(self.nside, arcmin=True)]*6
#         self.fsky = 1
# test_class_instru = instrument_xforecast(nside=nside)
# # fgcos.xForecast(components, instrument_xforecast, d_spectra, lmin=0, lmax=lmax)
# d_spectra1 = copy.deepcopy(d_spectra)
# d_spectra1[0, 0] = np.ones(d_spectra.shape[-1])
# res_xForecast = fgcos.xForecast(components, test_class_instru,
#                                 d_spectra1, lmin=0, lmax=lmax, r=0)
# Cl_xForecast = res_xForecast.stat + res_xForecast.bias

# W_cmbQU = W[0]+W[1]
# W_Q = W_cmbQU[::2]
# W_U = W_cmbQU[1::2]

# diff_W_cmbQU = diff_W[:, 0] + diff_W[:, 1]
# diff_W_Q = diff_W_cmbQU[:, ::2]
# diff_W_U = diff_W_cmbQU[:, 1::2]

# diff_diff_W_cmbQU = diff_diff_W[:, :, 0] + diff_diff_W[:, :, 1]
# diff_diff_W_Q = diff_diff_W_cmbQU[:, :, ::2]
# diff_diff_W_U = diff_diff_W_cmbQU[:, :, 1::2]
# V_Q2 = np.einsum('ij,ji...->...', sigma, diff_diff_W_Q)
# V_U2 = np.einsum('ij,ji...->...', sigma, diff_diff_W_U)
# delta_VQ2 = ((V_U+V_Q)[::2] - V_Q2)/V_Q2
# delta_VU2 = ((V_U+V_Q)[1::2] - V_U2)/V_U2
# print('biggest diff delta VQ def = ', np.max(np.abs(delta_VQ2)))
# print('biggest diff delta VU def = ', np.max(np.abs(delta_VU2)))

# V_comfgQ2 = np.einsum('ij,ji...->...', sigma[-2:, -2:], diff_diff_W_Q[-2:, -2:])
# V_comfgU2 = np.einsum('ij,ji...->...', sigma[-2:, -2:], diff_diff_W_U[-2:, -2:])
# delta_V_Q2 = (V_fg-V_comfgQ2)/V_fg
# delta_V_U2 = (V_fg-V_comfgU2)/V_fg
# print('biggest diff in V Q2 w/ fgbuster = ', np.max(np.abs(delta_V_Q2)))
# print('biggest diff in V U2 w/ fgbuster = ', np.max(np.abs(delta_V_U2)))
# QU_dBdB_delta = []
# for i in range(9):
#     db_list_QU = []
#     for ii in range(9):
#         freq_list_QU = []
#         for freq in range(6):
#             freq_list_QU.append(diff_diff_W[i, ii, 0, freq*2]-diff_diff_W[i, ii, 1, freq*2+1])
#         db_list_QU.append(freq_list_QU)
#     QU_dBdB_delta.append(db_list_QU)
# QU_dBdB_delta = np.array(QU_dBdB_delta)

# y_Q, y_U = get_Wfg_maps(WQ=W_Q, WU=W_U, freq_maps=fg_freq_maps_reshaped)
# Y_Q, Y_U = get_Wfg_maps(WQ=diff_W_Q, WU=diff_W_U, freq_maps=fg_freq_maps_reshaped)
# z_Q, z_U = get_Wfg_maps(WQ=V_Q2, WU=V_U2, freq_maps=fg_freq_maps_reshaped)

# y_alms = get_ys_alms(y_Q=y_Q, y_U=y_U, lmax=lmax)
# Y_alms = get_ys_alms(y_Q=Y_Q, y_U=Y_U, lmax=lmax)
# z_alms = get_ys_alms(y_Q=z_Q, y_U=z_U, lmax=lmax)

# yy = get_ys_Cls(y_alms, y_alms, lmax)
# YY = get_ys_Cls(Y_alms, Y_alms, lmax)
# yz = get_ys_Cls(y_alms, z_alms, lmax)
# zy = get_ys_Cls(z_alms, y_alms, lmax)
#
# Yy = get_ys_Cls(Y_alms, y_alms, lmax)
# Yz = get_ys_Cls(Y_alms, z_alms, lmax)
# stat = np.einsum('ij, ij... -> ...', sigma, YY)
# bias = yy + 2 * yz
# bias2 = yy + yz + zy
# var = stat**2 + 2 * np.einsum('i..., ij, j... -> ...', Yy, sigma, Yy)
# Clres = bias + stat
# Clres2 = bias2 + stat

# Y_alms_fgcomp = get_ys_alms(y_Q=Y_Q[-2:], y_U=Y_U[-2:], lmax=lmax)
# YY_fgcomp = get_ys_Cls(Y_alms_fgcomp, Y_alms_fgcomp, lmax)

# V_Q2_fgcomp = np.einsum('ij,ji...->...', sigmaX, diff_diff_W_Q[-2:, -2:])
# V_U2_fgcomp = np.einsum('ij,ji...->...', sigmaX, diff_diff_W_U[-2:, -2:])
# z_Q_fgcomp, z_U_fgcomp = get_Wfg_maps(
#     WQ=V_Q2_fgcomp, WU=V_U2_fgcomp, freq_maps=fg_freq_maps_reshaped)
# z_alms_fgcomp = get_ys_alms(y_Q=z_Q_fgcomp, y_U=z_U_fgcomp, lmax=lmax)
# yz_fgcomp = get_ys_Cls(y_alms, z_alms_fgcomp, lmax)
# zy_fgcomp = get_ys_Cls(z_alms_fgcomp, y_alms, lmax)

# bias_fgcomp = yy + yz_fgcomp + zy_fgcomp
# stat_fgcomp = np.einsum('ij, ij... -> ...', sigmaX, YY_fgcomp)

# Clres_fgcomp = bias_fgcomp + stat_fgcomp
