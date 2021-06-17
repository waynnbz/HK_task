import numpy as np
from scipy.sparse import find

from local_delaunay_tri import local_delaunay_tri
from detect_temporal_triplet import detect_temporal_triplet
from triplet_phase_closure_mcf import triplet_phase_closure_mcf

# def select_point_by_LS_at_arc( obs,Input,est_method=0):
#     '''
#     This function is used to initially select persisent points by LS at arcs
#     :param obs: obs file
#     :param Input: input construct file containing parameters
#     :param est_method:  0 - LS (default)
#                         1 - triplet closure
#     :return: pt_idx_keep:  point index corresponding to points remained
#     '''

#######################################################################
# test input
import scipy.io as sio

obs = sio.loadmat('obs.mat')['obs']
Input = sio.loadmat('Input.mat')
est_method = 1
#######################################################################

NSLC = Input['NSLC']
Nintv = NSLC - 1
B_model = Input['tmatrix']

# step 1 construct network
randn_times = 0
randn_std = np.linspace(1, 10, randn_times)
IDX, _ = local_delaunay_tri(obs[:, 0:5], randn_times, randn_std)
NARC = len(IDX['from'])
PHASE = obs[:, 5:].T
y1 = PHASE[:, IDX['to']] - PHASE[:, IDX['from']]
y1 = np.mod(y1 + np.pi, 2 * np.pi) - np.pi
del PHASE

# step 2 robust estimation at arcs
blockstep = 10000
blocknum = int(np.ceil(NARC / blockstep))
if est_method == 1:
    arc_idx_cell = []
    tri2ifg_matrix, _ = detect_temporal_triplet(Input)

    # TODO: OP with multiprocessing
    y1_cell = []
    for j in range(blocknum):
        pt_sidx = j * blockstep
        pt_eidx = min((j + 1) * blockstep, NARC)
        pt_idx = np.arange(pt_sidx, pt_eidx)
        y1_cell.append(y1[:, pt_idx])

        if np.mod(j, 100) == 0 and j > 0:
            print(f'Process block number {j} of {blocknum}')

        y1_temp = y1_cell[j]
        unwrap_correct_idx_final = triplet_phase_closure_mcf(tri2ifg_matrix, y1_temp, Input, 0)[1]
        arc_idx_cell.append(unwrap_correct_idx_final)

    arc_idx = np.concatenate(arc_idx_cell)
    arc_idx_keep = find(arc_idx)[1]

else:
    # TODO: complete est_method != 1 condition
    pass

#TODO: confirm unique
pt_idx_keep = np.unique(np.vstack((IDX['from'][arc_idx_keep], IDX['to'][arc_idx_keep])))
print(f'{len(pt_idx_keep)} of {obs.shape[0]} points are initially selected.')
#
# # return pt_idx_keep
