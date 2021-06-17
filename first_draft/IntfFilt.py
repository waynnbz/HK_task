import time

import numpy as np
from scipy.signal import correlate2d


def IntfFilt(intfstack, coh=1, SHP=None, FiltWin=(9, 9), Acc=True):
    '''
    This function filters the phase noise in single-look interferogram using the coherence as a
    weight, returns boxcar filtered phase image without SHP file
    :param intfstack:
    :param coh:
    :param SHP:
    :param FiltWin:
    :param Acc:
    :return:
    '''

    if not SHP:
        BoxFilt = True

    if np.isreal(intfstack):
        intfstack = np.exp(1j * intfstack)
    else:
        intfstack[intfstack != 0] = intfstack[intfstack != 0] / np.abs(intfstack[intfstack != 0])

    t = time.time()

    nlines, nwidths, npages = intfstack.shape
    Ph = intfstack
    if BoxFilt:
        print('BOXCAR filter')
        h = np.ones((5, 5))
        for ii in range(npages):
            Ph[:, :, ii] = correlate2d(intfstack[:, :, ii], h)
            print(f'BOX PHASE FILTERING: {ii}/{npages} is finished...')
        Ph = np.angle(Ph)


    if SHP:
        # Adp
        Cohthre = 0
        BroNumthre = 20
        mask = coh < Cohthre
        mask[np.tile(SHP.BroNum, (1,1,npages)) < BroNumthre] = True

        CalWin = SHP.CalWin
        if FiltWin[0] > CalWin[0]:
            FiltWin[0] = CalWin[0]
        if FiltWin[1] > CalWin[1]:
            FiltWin[1] = FiltWin[1]

        #TODO: dim check
        RadiusRow = (CalWin[0] - 1) / 2
        RadiusCol = (CalWin[1] - 1) / 2
        InitRow = (CalWin[0] + 1) / 2 - 1
        InitCol = (CalWin[1] + 1) / 2 - 1
        Radiusy = (FiltWin[0] - 1) / 2
        Radiusx = (FiltWin[1] - 1) / 2

        if FiltWin[0] == FiltWin[1]:
            # circular window
            x, y = np.meshgrid(np.array(np.arange(CalWin[1])), np.array(np.arange(CalWin[0])))
            WinFT = (x - InitCol)**2 + (y - InitRow)**2 <= np.dot(((FiltWin[0]-1)/2), ((FiltWin[0]-1)/2))
            # rectangular window
            WinFT = np.zeros((CalWin[0], CalWin[1]), dtype='bool')
            WinFT[InitRow-Radiusy:InitRow+Radiusy + 1,InitCol-Radiusx:InitCol+Radiusx + 1] = True

        WinFT = WinFT.ravel('F')
        SHP.PixelInd[np.logical_not(np.tile(WinFT, (1, len(SHP.PixelInd))))] = False
        # read data
        intfstack = np.logical_not(mask)*intfstack
        intfstack = np.pad(intfstack, pad_width=((RadiusRow, RadiusRow), (RadiusCol, RadiusCol), (0, 0)),
                           mode = 'symmetric')
        coh = np.pad(coh, pad_width=((RadiusRow, RadiusRow), (RadiusCol, RadiusCol), (0, 0)),
                           mode = 'symmetric')

        # phase filter
        if Acc:
            # weighted matrix
            Ph = np.reshape(Ph, (nlines*nwidths,npages), 'F')
            mask = np.reshape(mask, (nlines*nwidths,npages), 'F')
            wtinf = intfstack * coh
            #TODO: check if following commented vars need to be cleared in python case
            #clear intfstasck coh
            bstep = 1000000
            blocksize = np.floor(len(SHP.PixelInd) / bstep)
            if blocksize == 0:
                blocksize = 1
            lineidx = floor(linspace(1, length(SHP.PixelInd), blocksize + 1))
            block_size = np.zeros(blocksize, 2)
            for ii in range(blocksize):
                if ii != blocksize - 1:
                    block_size[ii,:]=(lineidx[ii], lineidx[ii + 1] - 1)
                else:
                    block_size[ii,:]=(lineidx[ii], max(SHP.PixelInd.shape))#TODO: check if shape is applicable

            for ii in range(npages):
                tempcoh = wtinf[:,:, ii]
                for jj in range(blocksize):
                    num = block_size[jj, 1] - block_size[jj, 1] + 1
                    temp = np.zeros((CalWin[0], CalWin[1], num), 'float32')
                    #TODO: check the Idx are zero-based
                    Idx = np.arange(block_size[jj, 0],block_size[jj, 1]+1, dtype='uint32')
                    Y, X = np.unravel_index(Idx, (nlines,nwidths), 'F')
                    Y = Y + RadiusRow
                    X = X + RadiusCol
                    for kk in range(num):
                        temp[:,:,kk] = tempcoh[Y[kk]-RadiusRow : Y[kk]+RadiusRow+1, #add 1 for slicing diff
                                       X[kk] - RadiusCol : X[kk] + RadiusCol + 1]
                    temp = np.reshape(temp, (CalWin[0]*CalWin[1],num), 'F')
                    temp[np.logical_not(SHP.PixelInd[:, Idx])] = np.nan
                    temp = np.nansum(temp[:, np.logical_not(mask[Idx, ii])], axis=0)
                    Idx[mask[Idx, ii]] = []
                    Ph[Idx, ii] = temp

                print(f' ADP. PHASE FILTERING: {ii} / {npages} is finished...')

            Ph = np.reshape(np.angle(Ph), (nlines, nwidths, npages), 'F')

        else:
            for ii in range(npages):
                tempintf = intfstack[:,:,ii]
                tempcoh = coh[:,:,ii]
                num = 0
                for jj in range(nwidths):
                    for kk in range(nlines):
                        if np.logical_not(mask[kk,jj,ii]):
                            x_global = jj + RadiusCol
                            y_global = kk + RadiusRow
                            IntfValue = tempintf[y_global - RadiusRow:y_global + RadiusRow + 1, #added 1 for slicing diff
                                        x_global - RadiusCol: x_global + RadiusCol + 1]
                            CohValue  = tempcoh[y_global-RadiusRow:y_global+RadiusRow + 1, #TODO: check if adding 1 is correct
                                        x_global-RadiusCol:x_global+RadiusCol + 1]
                            IntfValue = IntfValue[SHP.PixelInd[:, num]]
                            CohValue = CohValue[SHP.PixelInd[:, num]]
                            Ph[kk, jj, ii] = np.sum(IntfValue * CohValue, axis=0)
                        num += 1
                print(f' ADP. PHASE FILTERING: {ii} / {npages} is finished...')

            Ph = np.angle(Ph)

        elapsed = time.time() - t

        print(f'IntfFilt operation completed in {elapsed/60} minute(s).')
        print('Done!')

    return Ph
