import numpy as np
import time
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import find

def tempcohMST_Est(cpxphresidual,ref_sce,rep_sce,BiasAgr=None):
    '''
    Estiamte temporal coherence by MST
    :param cpxphresidual:
    :param ref_sce:
    :param rep_sce:
    :param BiasAgr:
    :return:
    '''
    ref_sce, rep_sce = np.array(ref_sce).reshape(-1, 1), np.array(rep_sce).reshape(-1, 1)
    SLC12 = np.hstack((ref_sce, rep_sce))
    SLC = np.unique(SLC12)
    NSLC = len(SLC)
    NIFG = len(ref_sce)
    Nintv = NSLC - 1
    image_pair = np.zeros((NIFG, 2))
    for i in range(NSLC):
        n = (SLC12 == SLC[i])
        image_pair[n] = i
    image_pair = np.sort(image_pair, axis=1)

    ## segment for paralell processing
    blockstep = 1000000
    NPS = cpxphresidual.shape[0]
    blocknum = int(np.ceil(NPS / blockstep))
    mst_cpxphresidual = {}
    MST_idx = {}

    for j in range(blocknum):
        print(f'Process block number {j} of {blocknum}')
        pt_sidx = j * blockstep
        pt_eidx = min((j + 1) * blockstep, NPS)
        pt_idx = np.arange(pt_sidx, pt_eidx)
        nps_blk = len(pt_idx)
        cohphresidual_blk = np.abs(cpxphresidual[pt_idx, :])
        inv_cohphresidual_blk = 1 - cohphresidual_blk + np.spacing(1)
        mst_cpxphresidual_blk = np.zeros((nps_blk, Nintv)) + 1j * np.zeros((nps_blk, Nintv))  # complex
        MST_idx_blk = np.zeros((nps_blk, Nintv))
        # TODO: replace with mutliprocessing module
        # matlabpoolchk(CoreNum)
        for i in range(nps_blk):
            SLC_SLC = csr_matrix((inv_cohphresidual_blk[i, :].T.squeeze(),
                                  (image_pair[:, 0].squeeze(), image_pair[:, 1].squeeze())),
                                 shape=(NSLC, NSLC)).toarray()
            UG = np.tril(SLC_SLC + SLC_SLC.T)
            ST = minimum_spanning_tree(UG)
            mst_x, mst_y, _ = find(ST)
            mst_xy = np.hstack((mst_x.reshape(-1, 1), mst_y.reshape(-1, 1)))
            mst_xy = np.sort(mst_xy, axis=1)
            # ia = intersect_rows(image_pair, mst_xy)
            ia = [i for i, v in enumerate(image_pair) if v.tolist() in mst_xy.tolist()]
            MST_idx_blk[i, :] = ia
            mst_cpxphresidual_blk[i, :] = cohphresidual_blk[i, ia]

        mst_cpxphresidual[j] = mst_cpxphresidual_blk
        MST_idx[j] = MST_idx_blk

    print('Estimating temporal coherence...')
    if not BiasAgr:
        tempcoh = np.abs(np.sum(mst_cpxphresidual, 1)) / Nintv
    elif BiasAgr == 'jackknife':
        tempcoh0 = np.abs(np.sum(mst_cpxphresidual, 1)) / Nintv
        # idx is a matrix with incremental values in each row, and remove its diagonal elements
        idx = np.tile(np.arange(Nintv), (Nintv, 1))
        idx = idx[~np.eye(Nintv, dtype=bool)].reshape(Nintv, Nintv - 1)
        tempcoh_jk = []
        for j in range(blocknum):
            print(f'Process block number {j} of {blocknum}')
            pt_sidx = j * blockstep
            pt_eidx = min((j + 1) * blockstep, NPS)
            pt_idx = np.arange(pt_sidx, pt_eidx)
            nps_blk = len(pt_idx)
            mst_cpxphresidual_blk = mst_cpxphresidual[pt_idx, :]
            tempcoh_jk_blk = np.zeros((nps_blk, 1))
            # TODO: OP with multi-processing
            for i in range(nps_blk):
                temp = mst_cpxphresidual_blk[i, :]
                temp_jkmatrix = temp[idx]
                tempcoh_jk_blk[i, 1] = np.mean(np.abs(np.sum(temp_jkmatrix, 2)) / (Nintv - 1))
            tempcoh_jk.append(tempcoh_jk_blk)
        tempcoh_jk = np.concatenate(tempcoh_jk, axis=1)
        tempcoh = Nintv * tempcoh0 - (Nintv - 1) * tempcoh_jk
        tempcoh[tempcoh < 0] = 0

    return tempcoh,MST_idx
