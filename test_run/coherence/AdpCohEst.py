import numpy as np
from scipy import signal
import time
from ismember import ismember
import multiprocessing as mp


def AdpCohEst(mlistack, mlilist, infstack, inflist, SHP=None, Btflag=False, Acc=True):
    '''
    Selectively performing three Coherence operation:
        Simple BOXCAR Coherence
        ADP. Coherence (Standard or Accurate Complex)

    Optional: performing Boostrapping bias mitigation on the Coherence result
        Accurate Complex ADP Coherence


    :param mlistack:
    :param mlilist:
    :param infstack:
    :param inflist:
    :param SHP:
    :param Btflag: boolean
    :param Acc:
    :return: Coh,CpxCoh,BtCoh
    '''

    # variables initialization
    BoxCar = False

    if not SHP:
        BoxCar = True

    # TODO: handle invalid input cases, help docs is printed and ends function

    CoreNum = 8
    t = time.time()
    nlines, nwidths, npages = infstack.shape

    # normalize, resulting in a matrix of (-1, 0, 1), search for possible library function
    infstack[infstack != 0] = infstack[infstack != 0] / np.abs(infstack[infstack != 0])
    Coh = np.zeros((nlines, nwidths, npages), 'float32')
    #TODO: need to handle not found case, will throw
    idx = [[list(mlilist.squeeze()).index(c) for c in r] for r in inflist]

    # SHP not provided
    ### BOXCAR COHERENCE ESTIMATE ###
    if BoxCar:
        print('BOXCAR COHERENCE ESTIMATE.')

        # TODO: OP_pack
        h_size = (15, 15)
        h = np.ones(h_size) / (h_size[0] * h_size[1])

        for ii in range(npages):
            m1 = mlistack[:, :, idx[ii, 0]]
            m2 = mlistack[:, :, idx[ii, 1]]

            # numerator, combine real & imaginary #TODO: OP_pack
            nu_r = signal.correlate2d(np.sqrt(m1 * m2) * (infstack[:, :, ii]).real, h)
            nu_i = signal.correlate2d(np.sqrt(m1 * m2) * (infstack[:, :, ii]).real, h)
            nu = complex(nu_r, nu_i)

            # denominator
            de1 = signal.correlate2d(h, m1)
            de2 = signal.correlate2d(h, m2)

            Coh[:, :, ii] = nu / (np.sqrt(de1 * de2) + np.spacing(1))

        CpxCoh = Coh
        Coh = np.abs(Coh)


    # SHP is provided:
    ### ADP. COHERENCE ESTIMATE ###
    else:
        CalWin = SHP.CalWin
        RadiusRow = (CalWin[0] - 1) / 2
        RadiusCol = (CalWin[1] - 1) / 2

        # initialize block and reshape Coh
        if Acc:
            # TODO: double check index conversion & resulting size
            # TODO: try to swap the whole block generation with a pack
            bstep = 1000000
            blocksize = max(np.floor(len(SHP.PixelInd) / bstep), 1)
            lineidx = np.floor(np.linspace(0, len(SHP.PixelInd), blocksize + 1))
            block_size = np.zeros((blocksize, 2))

            for i in range(blocksize):
                if i != blocksize - 1:
                    block_size[i, :] = [lineidx[i], lineidx[i + 1] - 1]
                else:
                    block_size[i, :] = [lineidx[i], len(SHP.PixelInd)]

            Coh = np.reshape(Coh, (nlines * nwidths, npages))

        # Transformation by pages
        for ii in range(npages):

            if not Acc:
                # images are read separately
                m1 = mlistack[:, :, idx[ii, 0]]
                m2 = mlistack[:, :, idx[ii, 1]]
                Intf = np.sqrt(m1 * m2) * infstack[:, :, ii]

                # Edge process (padding)
                m1 = np.pad(m1, (RadiusRow, RadiusCol), 'symmetric')
                m2 = np.pad(m2, (RadiusRow, RadiusCol), 'symmetric')
                Intf = np.pad(Intf, (RadiusRow, RadiusCol), 'symmetric')

                nu = np.zeros((nlines, nwidths), 'float32')
                de1 = nu
                de2 = nu
                num = 1

                # TODO: OP replace looping with parallel computing
                for jj in range(nwidths):
                    for kk in range(nlines):
                        x_global = jj + RadiusCol
                        y_global = kk + RadiusRow
                        MasterValue = m1[y_global - RadiusRow:y_global + RadiusRow,
                                      x_global - RadiusCol:x_global + RadiusCol]
                        SlaveValue = m2[y_global - RadiusRow:y_global + RadiusRow,
                                     x_global - RadiusCol:x_global + RadiusCol]
                        InterfValue = Intf[y_global - RadiusRow:y_global + RadiusRow,
                                      x_global - RadiusCol:x_global + RadiusCol]
                        MasterValue = MasterValue[SHP.PixelInd[:, num]]
                        SlaveValue = SlaveValue[SHP.PixelInd[:, num]]
                        InterfValue = InterfValue[SHP.PixelInd[:, num]]
                        nu[kk, jj] = sum(InterfValue)
                        de1[kk, jj] = sum(MasterValue)
                        de2[kk, jj] = sum(SlaveValue)
                        num = num + 1

                Coh[:, :, ii] = nu / (np.sqrt(de1 * de2) + np.spacing(1))

            # if ACC
            else:
                m1 = mlistack[:, :, idx[ii, 0]]
                m1[:, :, 1] = mlistack[:, :, idx[ii, 1]]
                m1[:, :, 2] = np.sqrt(m1[:, :, 0] * m1[:, :, 1]) * infstack[:, :, ii]
                m1 = np.pad(m1, (RadiusRow, RadiusCol), 'symmetric')

                for kk in range(blocksize):
                    num = block_size[kk, 1] - block_size[kk, 0] + 1
                    # temp is 4D array
                    temp = np.zeros((CalWin[1], CalWin[2], 3, num), 'float32')
                    # TODO: double check unravel_index(), ensure indices are counted from 0
                    X, Y = np.unravel_index(np.arange(block_size[kk, 0], block_size[kk, 1]), (nwidths, nlines))
                    Y = Y + RadiusRow
                    X = X + RadiusCol

                    for jj in range(num):
                        temp[:, :, :, jj] = m1[Y[jj] - RadiusRow:Y[jj] + RadiusRow,
                                            X[jj] - RadiusCol:X[jj] + RadiusCol, :]
                    temp = np.transpose(temp, (0, 1, 3, 2))
                    #TODO: check reshape in Fortran-style
                    temp = np.reshape(temp, (CalWin[0] * CalWin[1], num, 3)) * np.matlib.repmat(
                        SHP.PixelInd[:, block_size[kk, 0]:block_size[kk, 1]], [1, 1, 3])

            print(f'ADP. COHERENCE {ii}/{npages} is finished...')

        if Acc:
            Coh = np.reshape(Coh, (nlines, nwidths, npages))
        CpxCoh = Coh
        Coh = np.abs(Coh)

    # TODO
    ### Boostrapping bias mitigation ###
    if Btflag:
        BtCoh = Coh

        B = 200
        coh_thre = .5  # coherence threshold, coh values lower than 0.5 has larger bias

        for ii in range(npages):
            # read data
            m1 = mlistack[:, :, idx[ii, 0]]
            m2 = mlistack[:, :, idx[ii, 0]]
            Intf = np.sqrt(m1 * m2) * infstack[:, :, ii]

            # Edge process
            m1 = np.pad(m1, (RadiusRow, RadiusCol), 'symmetric')
            m2 = np.pad(m2, (RadiusRow, RadiusCol), 'symmetric')
            Intf = np.pad(Intf, (RadiusRow, RadiusCol), 'symmetric')

            # Bootstrapping estimate
            remove_bias = Coh[:, :, ii] < coh_thre
            num = 1
            for jj in range(nwidths):
                for kk in range(nlines):
                    if remove_bias[kk, jj]:
                        x_global = jj + RadiusCol
                        y_global = kk + RadiusRow
                        MasterValue = m1[y_global - RadiusRow:y_global + RadiusRow,
                                      x_global - RadiusCol: x_global + RadiusCol]
                        SlaveValue = m2[y_global - RadiusRow:y_global + RadiusRow,
                                     x_global - RadiusCol: x_global + RadiusCol]
                        InterfValue = Intf[y_global - RadiusRow:y_global + RadiusRow,
                                      x_global - RadiusCol: x_global + RadiusCol]
                        MasterValue = MasterValue[SHP.PixelInd[:, num]]
                        SlaveValue = SlaveValue[SHP.PixelInd[:, num]]
                        InterfValue = InterfValue[SHP.PixelInd[:, num]]

                        # Non-parametric Bootstrapping
                        Idx = np.random.randint(1, SHP.BroNum[kk, jj] + 1, size=(SHP.BroNum[kk, jj] + 1, B))
                        MasterValue = MasterValue[Idx]
                        SlaveValue = SlaveValue[Idx]
                        InterfValue = InterfValue[Idx]
                        BtCoh[kk, jj, ii] = np.mean(np.abs(np.sum(InterfValue, axis=0))
                                                    / np.sqrt(np.sum(MasterValue, axis=0)
                                                              * np.sum(SlaveValue, axis=0)))
                    num = num + 1
            print(f'BOOTSTRAPPING BIAS MITIGATION: {ii}/{npages} is finished...')

        BtCoh = 2 * Coh - BtCoh
        BtCoh[BtCoh < 0] = 0
    ### return output ###
    # if SHP & var ? not provided, set CpxCoh & BtCoh to null
    # TODO: set returns. no var case -> check what is var
    if not SHP:
        CpxCoh = []
        BtCoh = []

    if not Btflag:
        BtCoh = []

    elapsed = time.time() - t
    print(f'AdpCohEst operation completed in {elapsed / 60} minute(s).')
    print('Done')

    ide = []

    return Coh, CpxCoh, ide, h
