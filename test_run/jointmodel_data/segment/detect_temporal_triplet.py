import numpy as np
from scipy.sparse import find

from helper import intersect_mtlb

# def detect_temporal_triplet(Input):

########################################################################
# test input
import scipy.io as sio

Input = sio.loadmat('Input.mat')

########################################################################

pair_index = Input['pair_index']
tmatrix = Input['tmatrix']
NIFG = pair_index.shape[0]

# contruct triplet
triplet_ifg_index = []
for arc in range(NIFG):
    arc1 = find(np.logical_or(pair_index[:, 0] == pair_index[arc, 0], pair_index[:, 1] == pair_index[arc, 0]))[1]
    arc2 = find(np.logical_or(pair_index[:, 0] == pair_index[arc, 1], pair_index[:, 1] == pair_index[arc, 1]))[1]
    IDX_pair1 = pair_index[arc1, :]
    IDX_pair2 = pair_index[arc2, :]
    IDX_pair1[IDX_pair1 == pair_index[arc, 0]] = 0
    IDX_pair2[IDX_pair2 == pair_index[arc, 1]] = 0
    _, ia1, ia2 = intersect_mtlb(np.sum(IDX_pair1, axis=1), np.sum(IDX_pair2, axis=1))
    ia12_length = len(ia1)
    if ia12_length != 0:
        triplet_ifg_index.append(
            np.concatenate([arc1[ia1].reshape(-1, 1), arc2[ia2].reshape(-1, 1), np.tile(arc, (ia12_length, 1))],
                           axis=1))

triplet_ifg_index = np.concatenate(triplet_ifg_index)
triplet_ifg_index = np.unique(np.sort(triplet_ifg_index, axis=1), axis=0)

# convert triplet to design matrix
Ntri = triplet_ifg_index.shape[0]
tri2ifg_matrix = np.zeros((Ntri, NIFG))
for i in range(Ntri):
    tri2ifg_matrix[i, triplet_ifg_index[i, :]] = 1
    intv1 = sum(tmatrix[triplet_ifg_index[i, 0], :])
    intv2 = sum(tmatrix[triplet_ifg_index[i, 1], :])
    intv3 = sum(tmatrix[triplet_ifg_index[i, 2], :])
    max_id = [intv1, intv2, intv3].index(max([intv1, intv2, intv3]))
    tri2ifg_matrix[i, triplet_ifg_index[i, max_id]] = -1

    # return tri2ifg_matrix, triplet_ifg_index
