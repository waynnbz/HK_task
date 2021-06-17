import numpy as np
from scipy.sparse import find

from helper import intersect_mtlb

# Subfunction-1
def spatial_tri_arc_min_res(IDX1_pair,IDX2_pair,IDX_arc_pair,arc1_phi,arc2_phi,arc_phi):
    arc1_length = arc1_phi.shape[0]
    arc2_length = arc2_phi.shape[0]
    if arc1_length == 0 or arc2_length == 0:
        min_res = 0
    else:
        IDX1_pair[IDX1_pair == IDX_arc_pair[0]] = 0
        IDX2_pair[IDX2_pair == IDX_arc_pair[1]] = 0
        #TODO: confirm intersect function
        _, ia1, ia2 = intersect_mtlb(np.sum(IDX1_pair, axis=1), np.sum(IDX2_pair, axis=1))
        ia12_length = len(ia1)
        if ia12_length == 0:
            min_res = 0
        else:
            Tri_arc1 = IDX1_pair[ia1, :]
            Tri_arc2 = IDX2_pair[ia2, :]
            Tri_arc3 = np.tile(IDX_arc_pair, (ia12_length, 1))
            Tri_arc1_diff = abs(np.diff(Tri_arc1, 1, 1))
            Tri_arc2_diff = abs(np.diff(Tri_arc2, 1, 1))
            Tri_arc3_diff = abs(np.diff(Tri_arc3, 1, 1))
            #TODO: complete index return function
            max_id = np.argmax(np.hstack(Tri_arc1_diff, Tri_arc2_diff, Tri_arc3_diff), axis=1)
            max_id1 = max_id == 0
            max_id2 = max_id == 1
            max_id3 = max_id == 2
            arc1_phi_temp = arc1_phi[ia1,:]
            arc2_phi_temp = arc2_phi[ia2,:]
            arc_phi_temp = np.tile(arc_phi, (ia12_length, 1))
            arc1_phi_temp[max_id1,:] = -arc1_phi_temp[max_id1,:]
            arc2_phi_temp[max_id2,:] = -arc2_phi_temp[max_id2,:]
            arc_phi_temp[max_id3,:] = -arc_phi_temp[max_id3,:]

            arc_res_matrix=arc1_phi_temp+arc2_phi_temp+arc_phi_temp
            min_res = min(np.mean(abs(arc_res_matrix), axis=1))

    return min_res


def spatial_tri_arc_min_res_para( IDX_from,IDX_to,allEST_phi,X,Y,arc_tcp,patch_num_xy=3):

    patch_num_x = patch_num_xy
    patch_num_y = patch_num_xy
    Mx = [min(X), max(X)]
    My = [min(Y), max(Y)]
    Xcor = np.linspace(Mx[0], Mx[1], patch_num_x + 1)
    Ycor = np.linspace(My[0], My[1], patch_num_y + 1)
    overlap = max((Xcor[1] - Xcor[0]), (Ycor[1] - Ycor[0])) / 8
    IDX_cell = []
    allEST_phi_cell = []
    IDX_idx_cell = []
    for ix in range(patch_num_x):
        for iy in range(patch_num_y):
            x_min = Xcor[ix]
            x_max = Xcor[ix + 1]
            y_min = Ycor[iy]
            y_max = Ycor[iy + 1]
            #TODO: confirm X & Y combinable
            idx = np.logical_and.reduce([X >= x_min - overlap, X <= x_max + overlap, Y >= y_min - overlap, Y <= y_max + overlap])
            tmp = arc_tcp[:, idx.T]
            arc_tcp_sum = sum(abs(tmp), axis=1)
            idx_arc = arc_tcp_sum!=0
            IDX_cell.append({'from': IDX_from(idx_arc), 'to': IDX_to(idx_arc)})
            allEST_phi_cell.append(allEST_phi[idx_arc,:])
            IDX_idx_cell.append(find(idx_arc)[1])
    # TODO: check allEST_phi_cell type and how to appply isempty properly
    IDX_cell = IDX_cell(np.logical_not(allEST_phi_cell))




    return arc_min_res_final
