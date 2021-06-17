import scipy
from scipy import io
import numpy as np
from scipy.sparse import find


# sub function 1
def phase_link_sub(MM_cpxph, MM_coh, COH, CVM, VM, NSLC, NIFG, pair_index, convertM, ifg_idx):
    for k in range(NIFG):
        COH[pair_index[k, 0], pair_index[k, 1]] = MM_coh[k]
        COH[pair_index[k, 1], pair_index[k, 0]] = MM_coh[k]
        CVM[pair_index[k, 0], pair_index[k, 1]] = MM_coh[k] * np.conj(MM_cpxph[k])
        CVM[pair_index[k, 1], pair_index[k, 0]] = MM_coh[k] * MM_cpxph[k]
        VM[pair_index[k, 0], pair_index[k, 1]] = np.conj(MM_cpxph[k])
        VM[pair_index[k, 1], pair_index[k, 0]] = MM_cpxph[k]

    COH_inv = abs(np.linalg.pinv(COH))
    initial_cpxph = VM[:, 0]
    est_cpxph = initial_cpxph
    Niter = 0
    ph_thre = 0.001
    Niter_thre = 500
    lower_tri_log = np.tril(np.ones(NSLC), -1) > 0

    while np.logical_and(np.logical_or(Niter == 0, sum(abs(np.angle(est_cpxph * np.conj(initial_cpxph)))) > ph_thre),
                         Niter < Niter_thre):
        initial_cpxph = est_cpxph
        est_cpxph = np.dot(convertM * (COH_inv * CVM), initial_cpxph)

        est_cpxph = np.exp(1j * np.angle(est_cpxph * np.conj(est_cpxph[0])))
        Niter = Niter + 1

    est_VM = np.dot(est_cpxph.reshape(-1, 1), np.conj(est_cpxph.reshape(1, -1)))
    res_VM = VM * np.conj(est_VM)
    res_vec = res_VM.ravel('F')[lower_tri_log.ravel('F')]
    est_gamma_temp = abs(sum(res_vec[ifg_idx]) / NIFG)
    est_SMph_temp = np.conj(np.angle(est_cpxph[1:])).T
    est_obs_temp = np.angle(est_VM.ravel('F')[lower_tri_log.ravel('F')]).T

    return est_SMph_temp, est_gamma_temp, est_obs_temp, Niter


# sub function 2
def pair_index2ifg_idx(pair_index, NSLC, NIFG):
    full_matrix = np.zeros((NSLC, NSLC))
    for j in range(NIFG):
        full_matrix[pair_index[j, 0], pair_index[j, 1]] = 1
        full_matrix[pair_index[j, 1], pair_index[j, 0]] = 1

    lower_tri_log = np.tril(np.ones((NSLC, NSLC)), -1) > 0
    ifg_idx_idx = full_matrix.ravel('F')[lower_tri_log.ravel('F')]
    [_, ifg_idx, _] = find(ifg_idx_idx > 0)
    return ifg_idx


def phaselink_MLE4(obs, coh, Input):
    NSLC = int(Input['NSLC'])
    NIFG = int(Input['NIFG'])
    NPT = obs.shape[0]
    pair_index = Input['pair_index']
    convertM = abs(np.ones((NSLC, NSLC)) - np.eye(NSLC))

    ifg_idx = pair_index2ifg_idx(pair_index, NSLC, NIFG)

    # phase link parallel：
    blockstep = 100000
    blocknum = np.ceil(NPT / blockstep)
    est_SMph_cell = {}
    est_gamma_cell = {}
    est_obs_cell = {}
    CoreNum = 12
    COH = np.zeros((NSLC, NSLC))
    CVM = np.zeros((NSLC, NSLC)) + 1j * np.zeros((NSLC, NSLC))
    VM = np.zeros((NSLC, NSLC)) + 1j * np.zeros((NSLC, NSLC))

    #######################################

    for j in range(int(blocknum)):
        print('Process block number ', j + 1, ' of ', blocknum)
        pt_sidx = (j) * blockstep + 1
        pt_eidx = min((j + 1) * blockstep, NPT)
        pt_idx = np.arange(pt_sidx - 1, pt_eidx).T
        nps_blk = len(pt_idx)
        obs_temp = obs[pt_idx, :]
        coh_temp = coh[pt_idx, :]
        est_SMph_blk = np.zeros((nps_blk, NSLC - 1))
        est_gamma_blk = np.zeros((nps_blk, 1))
        est_obs_blk = np.zeros((nps_blk, NIFG))

        for i in range(nps_blk):
            MM_cpxph = np.exp((1j) * (obs_temp[i, 5:]))
            MM_coh = coh_temp[i, 3:]
            [est_SMph_temp, est_gamma_temp, est_obs_temp, Niter_temp] = phase_link_sub(MM_cpxph, MM_coh, COH, CVM, VM,
                                                                                       NSLC, NIFG, pair_index, convertM,
                                                                                       ifg_idx)
            est_SMph_blk[i, :] = est_SMph_temp
            est_gamma_blk[i] = est_gamma_temp
            est_obs_blk[i, :] = est_obs_temp[ifg_idx]

        est_SMph_cell[j, 0] = est_SMph_blk
        est_gamma_cell[j, 0] = est_gamma_blk
        est_obs_cell[j, 0] = est_obs_blk

        if j == 0:
            est_SMph = est_SMph_cell[0, 0]
            est_gamma = est_gamma_cell[0, 0]
            est_obs = est_obs_cell[0, 0]

        if j >= 1:
            est_SMph = np.concatenate((est_SMph, est_SMph_cell[j, 0]), axis=1)
            est_gamma = np.concatenate((est_gamma, est_gamma_cell[j, 0]), axis=1)
            est_obs = np.concatenate((est_obs, est_obs[j, 0]), axis=1)

    est_SMph = np.concatenate((obs[:, 0:5], est_SMph), axis=1)
    est_gamma = np.concatenate((obs[:, 0:5], est_gamma), axis=1)
    est_obs = np.concatenate((obs[:, 0:5], est_obs), axis=1)

    return est_SMph, est_gamma, est_obs
