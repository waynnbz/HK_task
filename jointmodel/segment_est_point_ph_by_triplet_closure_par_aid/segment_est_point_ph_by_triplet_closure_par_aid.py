import numpy as np
from scipy.sparse import find
from itertools import compress
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsmr

from select_point_by_LS_at_arc import *
from detect_tri_closure import *
from spatial_tri_arc_min_res_para import spatial_tri_arc_min_res_para
from subset_detect import *
from helper import intersect_mtlb



def segment_est_point_ph_by_triplet_closure_par_aid(obs,Input,seg_par,seg_index ):
    '''
    :param obs:
    :param Input:
    :param seg_par:
    :param seg_index:
    :return:
    '''

    i_azimuth, i_range = np.unravel_index(seg_index, (seg_par['num_seg_azimuth'], seg_par['num_seg_range']), order='F')
    pt_idx = devide_P(i_range, i_azimuth, obs, seg_par['ra_cor'], seg_par['az_cor'], seg_par['overlap'])

    if len(pt_idx) < 10:
        point_ph = []
        # return point_ph

    obs = obs[pt_idx, :]

    ## further segmentation
    range_length = max(obs[:, 0]) - min(obs[:, 0])
    azimuth_length = max(obs[:, 1]) - min(obs[:, 1])
    range_patch_num = int(np.ceil(range_length / 1000))
    azimuth_patch_num = int(np.ceil(azimuth_length / 1000))
    obs_cell, obs_idx_cell = Divide_Pointset(obs, range_patch_num, azimuth_patch_num)
    Input['N_cell'] = len(obs_cell)

    ## estimate interval phase in each patch
    arc_ph = []
    IDX = []
    for i in range(Input['N_cell']):
        print(f"Processing segment {i} of {Input['N_cell']}")
        obs_temp = obs_cell[i]
        obs_idx_temp = obs_idx_cell[i]
        print(f'In total of {obs_temp.shape[0]} points in this segment')
        arc_ph_temp, IDX_temp = patch_par_est_phi_triplet_closure(obs_temp, Input)
        if arc_ph_temp:
            arc_ph.append(arc_ph_temp)
            IDX.append(np.hstack((obs_idx_temp[IDX_temp[:, 0]], obs_idx_temp[IDX_temp[:, 1]])))

    # del arc_ph_temp, IDX_temp, obs_temp, obs_idx_temp, obs_cell, obs_idx_cell
    IDX = np.concatenate(IDX)
    arc_ph = np.concatenate(arc_ph)
    # TODO: fix sortrows
    st_idx = np.argsort(np.sort(IDX, axis=1)[:, 0])
    IDX = IDX[st_idx]
    st_idx = np.argsort(IDX)
    arc_ph = arc_ph[st_idx, :]

    uni_IDX_kp, ind = np.unique(IDX, return_index=True, axis=0)
    uni_arc_ph_kp = arc_ph[ind, :]
    # del IDX, arc_ph, ind

    ## network subset detection
    NTCP = obs.shape[0]
    TCP_keep = np.unique(uni_IDX_kp)
    uni_NARC = uni_IDX_kp.shape[0]
    # TODO: check TCP_keep shape
    NTCP_kp = len(TCP_keep)
    arc_index = np.arange(uni_NARC)
    arc_tcp1 = csr_matrix(-1, (arc_index, uni_IDX_kp[:, 0]), shape=(uni_NARC, NTCP))
    arc_tcp2 = csr_matrix(1, (arc_index, uni_IDX_kp[:, 1]), shape=(uni_NARC, NTCP))
    arc_tcp = arc_tcp1 + arc_tcp2
    # del arc_tcp1, arc_tcp2, arc_index
    _, st2, _ = subset_detect(uni_IDX_kp[:, 0], uni_IDX_kp[:, 1], obs[TCP_keep, 1], obs[TCP_keep, 0], obs[:, 1],
                              obs[:, 0], 10, 3)
    arc_idx = st2 == 1
    uni_arc_ph_kp_kp = uni_arc_ph_kp[arc_idx, :]
    uni_IDX_kp_kp = uni_IDX_kp[arc_idx, :]
    TCP_kp_kp = np.unique(uni_IDX_kp_kp)
    arc_tcp_kp = arc_tcp[arc_idx, :]
    kp_col = np.any(arc_tcp_kp, axis=0)
    arc_tcp_kp = arc_tcp_kp[:, kp_col]
    IDX_refpnt = 1
    arc_tcp_kp[:, IDX_refpnt] = []
    NTCP_kp_kp = len(TCP_kp_kp)

    print(' ')
    print(f'  Number of arcs:                                      {uni_NARC}')
    print(f'  Number of arcs kept:                                 {sum(arc_idx)}')
    print(f'  Number of original points :                          {NTCP}')
    print(f'  Number of points in the arcs:                        {NTCP_kp}')
    print(f'  Number of points kept:                               {NTCP_kp_kp}')
    print('')
    print('==========================================================================')
    # del uni_arc_ph_kp, uni_IDX_kp, arc_idx, st2, arc_tcp, kp_col

    ## arc to point
    N_arc_ph = uni_arc_ph_kp_kp.shape[1]
    point_ph = np.zeros((NTCP_kp_kp - 1, N_arc_ph))

    for i in range(N_arc_ph):
        print(f'Processing {i} of {N_arc_ph} intervals')
        # TODO: test if lsmr matches
        point_ph[:, i] = lsmr(arc_tcp_kp, uni_arc_ph_kp_kp[:, i])[0]

    point_ph = np.vstack(point_ph[:IDX_refpnt - 1, :], np.zeros((1, N_arc_ph)), point_ph[IDX_refpnt - 1:, :])
    point_ph = np.hstack(obs[TCP_kp_kp, :5], point_ph)

    return point_ph



############################### Subfunctions ###############################

# Subfunction-1 in segment_est_point_ph_by_triplet_closure_par_aid
def Divide_Pointset(P,num_seg_x,num_seg_y):
    '''
    This function dividing big data set into num_set x num_seg segments
    '''
    Mx = (min(P[:, 0]), max(P[:, 0]))
    My = (min(P[:, 1]), max(P[:, 1]))
    Xcor = np.linspace(Mx[0], Mx[1], num_seg_x + 1)
    Ycor = np.linspace(My[0], My[1], num_seg_y + 1)
    overlap = max((Xcor[1] - Xcor[0]), (Ycor[1] - Ycor[0])) / 8
    k = 0
    tD2 = []
    Pk = []
    IDinP = []
    for ix in range(num_seg_x):
        for iy in range(num_seg_y):
            ID2 = devide_P(ix, iy, P, Xcor, Ycor, overlap)
            tD2.append(ID2)
            tD2 = [np.unique(np.concatenate(tD2))]
            if len(tD2[0]) >= 10 or ix == num_seg_x or iy == num_seg_y:
                Pk.append(P[tD2[0], :])
                IDinP.append(tD2[0])
                tD2 = []
                k += 1
    cell_length = list(map(lambda x: len(x) > 10, Pk))
    IDinP = list(compress(IDinP, cell_length))
    Pk = list(compress(Pk, cell_length))

    return Pk, IDinP


# Subfunction-2 in segment_est_point_ph_by_triplet_closure_par_aid
def devide_P(ix,iy,P,Xcor,Ycor,overlap):
    '''
    This function is to extract point ID in the ix_iy segment
    '''
    xl = Xcor[ix]
    xu = Xcor[ix + 1]
    yl = Ycor[iy]
    yu = Ycor[iy + 1]
    kx = find(np.logical_and(P[:, 0] <= xu, P[:, 0] >= xl))[1]
    kxy = np.logical_and(P[kx,1]<=yu, P[kx,1]>=yl)
    ID1 = kx[kxy]
    xl2 = xl - overlap
    xu2 = xu + overlap
    yl2 = yl - overlap
    yu2 = yu + overlap
    kx2 = find(np.logical_and(P[:, 0] <= xu2, P[:, 0] >= xl2))[1]
    kxy2 = np.logical_and(P[kx2, 1] <= yu2, P[kx2, 1] >= yl2)
    ID2 = kx2[kxy2]
    if len(ID2)<len(ID1)+5:     # to insure overlap point >5
        lap2 = 2*overlap
        kx2 = find(np.logical_and(P[:, 0] <= xu + lap2, P[:, 0] >= xl - lap2))[1]
        kxy2 = np.logical_and(P[kx2, 1] <= yu + lap2, P[kx2, 1] >= yl - lap2)
        ID2 = kx2[kxy2]

    return ID2


# Subfunction-3 in segment_est_point_ph_by_triplet_closure_par_aid
def patch_par_est_phi_triplet_closure(obs,Input):
    ## step 1 prepare input and initial point selection
    pt_idx_keep = select_point_by_LS_at_arc(obs, Input, 1)
    if len(pt_idx_keep) < 3:
        allEST_phi = []
        IDX_output = []
    else:
        obs_keep = obs[pt_idx_keep,:]

        ## step 2 construct network
        randn_times = 1
        randn_std = np.linspace(1, 20, randn_times)
        IDX, Tri = local_delaunay_tri(obs_keep[:, :5], randn_times, randn_std)
        NARC = len(IDX['from'])
        PHASE = obs_keep[:, 5:].T
        y0 = PHASE[:, IDX['to']]-PHASE[:, IDX['from']]
        y1 = np.mod(y0 + np.pi, 2 * np.pi) - np.pi
        del PHASE, obs_keep

        ## step 3 robust estimation at arcs
        # TODO: OP with multiprocessing
        blockstep = 10000
        blocknum = int(np.ceil(NARC / blockstep))
        allEST_phi_cell = []
        arc_log_cell = []
        tri2ifg_matrix, triplet_ifg_index = detect_temporal_triplet(Input)
        y1_cell = []
        for j in range(blocknum):
            pt_sidx = j * blockstep
            pt_eidx = min((j+1) * blockstep, NARC)
            pt_idx = np.arange(pt_sidx, pt_eidx)
            y1_cell.append(y1[:, pt_idx])

        # TODO: OP with multiprocessing
        for j in range(blocknum):
            print(f'Process block number {j} of {blocknum}')
            y1_temp = y1_cell[j]
            allEST_phi_blk, unwrap_correct_log_final = triplet_phase_closure_mcf(tri2ifg_matrix, y1_temp, Input, 1)

            allEST_phi_cell.append(allEST_phi_blk)
            arc_log_cell.append(unwrap_correct_log_final)

        # del y1_temp, allEST_phi_blk, y1_cell, y1, allEST_phi_cell
        #TODO: check concat dim
        allEST_phi = np.concatenate(allEST_phi_cell)
        arc_log = np.concatenate(arc_log_cell)

        ## remove arcs by triplet closure in space
        print('Remove arcs with outliers...')
        if Tri.any():
            if allEST_phi.any():
                idx_keep_by_spa_tri = detect_tri_closure(allEST_phi, IDX, Tri)
                idx_keep_by_tem_tri = find(arc_log)[1]
                ind_kp = intersect_mtlb(idx_keep_by_spa_tri, idx_keep_by_tem_tri)[0]
                ## output
                allEST_phi = allEST_phi[ind_kp,:]
                IDX_output = np.hstack((pt_idx_keep[IDX['from'][ind_kp,:]], pt_idx_keep[IDX['to'][ind_kp,:]]))
            else:
                allEST_phi = []
                IDX_output = []
        else:
            if allEST_phi.any():
                ind_kp = find(arc_log)[1]
                allEST_phi = allEST_phi[ind_kp,:]
                IDX_output = np.hstack((pt_idx_keep[IDX['from'][ind_kp,:]], pt_idx_keep[IDX['to'][ind_kp,:]]))
                ## spatial closure detection 1
                NARC_kp = len(ind_kp)
                arc_index = np.arange(NARC_kp)
                arc_tcp1 = csr_matrix(-1, (arc_index, IDX_output[:, 0]), (NARC_kp, obs.shape[0]))
                arc_tcp2 = csr_matrix(1, (arc_index, IDX_output[:, 1]), (NARC_kp, obs.shape[0]))
                arc_tcp = arc_tcp1 + arc_tcp2
                del arc_tcp1, arc_tcp2
                arc_min_res = spatial_tri_arc_min_res_para(IDX_output[:, 0], IDX_output[:, 1], allEST_phi, obs[:, 0],
                                                           obs[:, 1], arc_tcp, 5)
                ind_kp_kp = arc_min_res > 0.1
                allEST_phi = allEST_phi[ind_kp_kp,:]
                IDX_output = IDX_output[ind_kp_kp,:]
            else:
                allEST_phi= []
                IDX_output = []

        print('==========================================================================')
        print(' ')
        print(f'  Number of arcs:                                    {NARC}')
        print(f'  Number of arcs kept:                               {ind_kp}')
        print(f'  Number of original points :                        {obs.shape[0]}')
        print(f'  Number of points initially selected :              {len(pt_idx_keep)}')
        print(f'  Number of points kept:                             {len(np.unique(IDX_output))}')
        print('')
        print('==========================================================================')

    return allEST_phi, IDX_output

# not executed
def patch_par_est_phi_3d_triplet_closure(obs,Input):

    # return allEST_phi, IDX_output
    return None
