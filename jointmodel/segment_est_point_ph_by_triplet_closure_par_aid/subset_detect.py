import numpy as np
from scipy.sparse import csr_matrix
import time

# from network_conv_v2 import *
from network_merge import *
from helper import intersect_row_mtlb


def subset_detect( IDX_from,IDX_to,X,Y,X_org,Y_org,num_threshold,patch_num_2,CoreNum,disp_flag=1):

    # vector assertations
    assert IDX_from.squeeze().ndim() == 1, 'IDX_from must be a vector'
    assert IDX_to.squeeze().ndim() == 1, 'IDX_to must be a vector'
    assert X.squeeze().ndim() == 1, 'X must be a vector'
    assert Y.squeeze().ndim() == 1, 'Y must be a vector'
    assert X_org.squeeze().ndim() == 1, 'X_org must be a vector'
    assert Y_org.squeeze().ndim() == 1, 'Y_org must be a vector'

    patch_num_2 = max(min(patch_num_2, np.floor(X.shape[0] / 500)), 1)
    patch_num_x = 2 ** patch_num_2
    patch_num_y = 2 ** patch_num_2
    rows = max(X)
    cols = max(Y)
    patch_num = patch_num_x * patch_num_y
    if patch_num_x * patch_num_y == 1:
        print('No need to decomposite the network')
    else:
        if disp_flag == 1:
            print(f'decompose the whole network into {patch_num} patches')
            print(f'patch number in row direction: {patch_num_x}')
            print(f'patch number in col direction: {patch_num_y}')

    total_point = np.unique(np.concatenate((IDX_from, IDX_to)))
    NPS = max(total_point)
    NARC = len(IDX_from)
    arc_index = np.arange(NARC)
    arc_tcp1 = csr_matrix(True, (arc_index, IDX_from), shape=(NARC, NPS))
    arc_tcp2 = csr_matrix(True, (arc_index, IDX_to), shape=(NARC, NPS))
    arc_tcp = arc_tcp1 + arc_tcp2

    row_intv = rows / patch_num_x
    col_intv = cols / patch_num_y
    IDX = []
    ia = intersect_row_mtlb(np.concatenate((X_org.reshape(-1, 1), Y_org.reshape(-1, 1)), axis=1), np.concatenate((X.reshape(-1, 1), Y.reshape(-1, 1)), axis=1))

    for j in range(patch_num_y):
        for i in range(patch_num_x):
            if i == patch_num_x:
                x_max = rows
            else:
                x_max = i * row_intv

            if j == patch_num_y:
                y_max = cols
            else:
                y_max = j * col_intv
            x_min = 1 + j * row_intv
            y_min = 1 + j * col_intv
            idx = np.logical_and.reduce([X>=x_min, X<=x_max, Y>=y_min, Y<=y_max])
            idx = ia[idx]
            tmp = arc_tcp[:,idx.T]
            arc_tcp_sum = np.sum(tmp, axis=1)
            idx_arc = arc_tcp_sum != 0
            IDX.append({'from': IDX_from[idx_arc], 'to': IDX_to[idx_arc]})

    # TODO: OP multi-processing
    subset = []
    num = []
    # t = time.time()
    for iji in range(IDX):
        if disp_flag == 1:
            print(f'Processing the patch: {iji}')
        IDX_from_temp = IDX[iji]['from']
        IDX_to_temp = IDX[iji]['to']
        if not IDX_from_temp or not IDX_to_temp:
            continue
        else:
            ss, nn = network_con_v2(IDX_from_temp,IDX_to_temp)
            subset.append(ss)
            num.append(nn)
    print('Network decompose done !!!')
    subset_ori = subset
    num_ori = num
    for t in range(9,-1,-1):
        pl = 2**t
        pl2 = 2**(t+1)
        subset_final = []
        num_final = []
        if disp_flag == 1:
            print(f'Processing the patch length: {pl2}...')
        # TODO: OP multi-processing
        for k in range(pl**2):
            if disp_flag == 1:
                print(f'Processing the subpatch: {k+1}')
            j = np.floor(k / pl) + 1
            i = np.mod(k, pl) + 1
            sp11 = subset_ori[2 * j * pl2 + 2 * i - 2 * pl2 - 2]
            sp21 = subset_ori[2 * j * pl2 + 2 * i - 2 * pl2 -1]
            sp12 = subset_ori[2 * j * pl2 + 2 * i - pl2 - 2]
            sp22 = subset_ori[2 * j * pl2 + 2 * i - pl2 - 1]
            num11 = num_ori[2 * j * pl2 + 2 * i - 2 * pl2 - 2]
            num21 = num_ori[2 * j * pl2 + 2 * i - 2 * pl2 -1]
            num12 = num_ori[2 * j * pl2 + 2 * i - pl2 - 2]
            num22 = num_ori[2 * j * pl2 + 2 * i - pl2 -1]
            sp_temp12, num_temp12 = network_merge(sp11, sp21, num11, num21, num_threshold)
            sp_temp34, num_temp34 = network_merge(sp12, sp22, num12, num22, num_threshold)
            subset_final[k], num_final[k] = network_merge(sp_temp12, sp_temp34, num_temp12, num_temp34, num_threshold)
        subset_ori = subset_final
        num_ori = num_final
    print('Detection of network subset done!')
    subset_final = subset_final[0]
    num_final = num_final[0]
    flag_final = np.zeros((NARC, 1))
    subset_num = 1
    for i in range(len(subset_final)):
        arc_tcp_tmp = arc_tcp[:, subset_final[i].reshape(-1)]
        arc_tcp_sum = np.sum(arc_tcp_tmp, axis=2)
        idx = arc_tcp_sum != 0
        flag_final[idx] = i
    # elapsed = time.time() - t
    # print(f'')

    return subset_final,flag_final,num_final
