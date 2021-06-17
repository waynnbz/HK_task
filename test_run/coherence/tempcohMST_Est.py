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

    SLC12 = np.hstack(ref_sce, rep_sce)
    SLC =list(dict.fromkeys(SLC12.ravel('F')))
    #TODO: check SLC shape, it's legnth from
    NSLC = max(SLC.shape)
    NIFG = max(ref_sce.shape)
    Nintv = NSLC - 1
    image_pair = np.zeros(NIFG, 2)
    for i in range(NSLC):
        n = SLC12 == SLC[i]
        image_pair[n] = i
    image_pair = np.sort(image_pair, axis=1)

    ## segment for paralell processing
    blockstep = 1000000
    NPS =cpxphresidual.shape[0]
    blocknum = np.ceil(NPS / blockstep)
    #TODO: investigate cell usage; temp(np.zeros)
    mst_cpxphresidual = {}
    MST_idx = {}

    for j in range(blocknum):
        print(f'Process block number {j} of {blocknum}')
        pt_sidx = j * blockstep
        pt_eidx = min((j+1) * blockstep, NPS)
        pt_idx = np.array(np.arange(pt_sidx, pt_eidx))[:, np.newaxis]
        nps_blk = len(pt_idx)
        cohphresidual_blk = np.abs(cpxphresidual[pt_idx,:])
        inv_cohphresidual_blk = 1 - cohphresidual_blk + np.spacing(1)
        mst_cpxphresidual_blk = np.zeros(nps_blk, Nintv) +  1j * np.zeros(nps_blk, Nintv) #complex
        MST_idx_blk = np.zeros(nps_blk, Nintv)
        #TODO: replace with mutliprocessing module
        #matlabpoolchk(CoreNum)
        for i in range(nps_blk):
            SLC_SLC = csr_matrix((inv_cohphresidual_blk[i, :].T, (image_pair[:, 0], image_pair[:, 1])),
                                 shape=(NSLC, NSLC))
            UG = np.tril(SLC_SLC + SLC_SLC.T)
            ST = minimum_spanning_tree(UG)
            mst_x, mst_y, _ = find(ST)
            mst_xy = np.sort(np.hstack(mst_x, mst_y), axis=1)
            #TODO: retrieve the indices of matched image_pair to mst_xy
            [~, ia] = intersect(image_pair, mst_xy, 'rows');
            MST_idx_blk[i,:]=ia.T
            mst_cpxphresidual_blk[i,:]=cohphresidual_blk[i, ia]

        mst_cpxphresidual[j] = mst_cpxphresidual_blk
        MST_idx[j] = MST_idx_blk

    print('Estimating temporal coherence...')
    if not BiasAgr:
        tempcoh = np.abs(np.sum(mst_cpxphresidual, 1)) / Nintv
    elif BiasAgr == 'jackknife':
        tempcoh0 = np.abs(np.sum(mst_cpxphresidual, 1)) / Nintv
        #TODO: replace idx, a row-rep np.arange(Nintv) matrix without diagonal values
        idx = np.zeros(Nintv, Nintv - 1)
        # for i in range(Nintv):
        #     _s1 = np.arange(i)
        #     _s2 =
        #     idx[i, :] = np.hstack(np.arange(i), np.arange(i, Nintv))
        #









    return tempcoh,MST_idx
