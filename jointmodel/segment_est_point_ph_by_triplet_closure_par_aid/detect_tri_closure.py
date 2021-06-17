import numpy as np

from helper import ismember_mtlb, intersect_row_mtlb


def detect_tri_closure(allEST_phi,IDX,Tri,res_threshold=0.1):


    Arc = np.vstack((IDX['from'], IDX['to'])).T
    Tri_arc1 = np.vstack((Tri[:, 0], Tri[:, 1])).T
    Tri_arc2 = np.vstack((Tri[:, 1], Tri[:, 2])).T
    Tri_arc3 = np.vstack((Tri[:, 0], Tri[:, 2])).T

    Tri_arc1_idx = ismember_mtlb(Tri_arc1, Arc, 'rows')
    Tri_arc2_idx = ismember_mtlb(Tri_arc2, Arc, 'rows')
    Tri_arc3_idx = ismember_mtlb(Tri_arc3, Arc, 'rows')
    Tri_arc1_phi = allEST_phi[Tri_arc1_idx, :]
    Tri_arc2_phi = allEST_phi[Tri_arc2_idx, :]
    Tri_arc3_phi = allEST_phi[Tri_arc3_idx, :]
    del allEST_phi, IDX
    del Tri_arc1_idx, Tri_arc2_idx, Tri_arc3_idx

    # calculate closure for spatial triplet
    Tri_arc_res = Tri_arc1_phi + Tri_arc2_phi - Tri_arc3_phi
    Tri_arc_res_max = np.max(np.abs(Tri_arc_res), axis=1)

    Tri_idx_remain = Tri_arc_res_max < res_threshold
    Tri_remain = Tri[Tri_idx_remain,:]

    Tri_remain_arc1 = np.hstack((Tri_remain[:, 0], Tri_remain[:, 1]))
    Tri_remain_arc2 = np.hstack((Tri_remain[:, 1], Tri_remain[:, 2]))
    Tri_remain_arc3 = np.hstack((Tri_remain[:, 0], Tri_remain[:, 2]))
    Tri_remain_arc = np.vstack((Tri_remain_arc1, Tri_remain_arc2, Tri_remain_arc3))
    Tri_remain_arc = np.unique(np.sort(Tri_remain_arc, axis=1), axis=0)
    idx_keep = intersect_row_mtlb(Arc, Tri_remain_arc)

    return idx_keep