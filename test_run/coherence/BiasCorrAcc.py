import numpy as np

import time



def BiasCorrAcc(CpxCoh, SHP, EstAgr='sec'):
    '''

    :param CpxCoh:
    :param SHP:
    :param EstAgr:
    :return:
    '''

    assert np.isreal(CpxCoh), 'The CpxCoh should be complex data.'

    t = time.time()

    nlines, nwidths, npages = CpxCoh.shape
    CalWin = SHP.CalWin
    RadiusRow = (CalWin[0] - 1) / 2
    RadiusCol = (CalWin[1] - 1) / 2

    bstep = 1000000
    blocksize = np.floor(len(SHP.PixelInd) / bstep)
    if blocksize == 0:
        blocksize = 1
    lineidx = np.floor(np.linspace(1, max(SHP.PixelInd.shape), blocksize + 1))
    block_size = np.zeros((blocksize, 2))
    for i in range(blocksize):
        if i != blocksize - 1:
            block_size[i, :] = (lineidx[i], lineidx[i + 1] - 1)
        else:
            block_size[i, :] = (lineidx[i], len(SHP.PixelInd))

    BiasCorrcoh = np.zeros(nlines * nwidths, npages, 'float32')
    CpxCoh = np.pad(CpxCoh, pad_width=((RadiusRow, RadiusRow), (RadiusCol, RadiusCol), (0, 0)),
           mode='symmetric')

    #TODO: incomplete for loop
    for kk in range(npages):
        if EstAgr == 'sec': #second kind statistic
            tempcoh = np.log(np.abs(CpxCoh[:,:,kk]))
            METHOD = 'LOG-MOMENT'
        else:
            tempcoh = CpxCoh[:,:,kk]
            METHOD = 'REGULAR-MOMENT'

        for ii in range(blocksize):
            num = block_size[ii, 1] - block_size[ii, 0]
            temp = np.zeros(CalWin[0], CalWin[1], num, 'float32')
            Y, X = np.unravel_index(np.arange(block_size[kk, 0], block_size[kk, 1]), (nwidths, nlines))
            Y = Y + RadiusRow
            X = X + RadiusCol
            for jj in range(num):
                temp[:, :, jj] = tempcoh[Y[jj]-RadiusRow:Y[jj]+RadiusRow,X[jj]-RadiusCol:X[jj]+RadiusCol]
            temp = np.reshape(temp, (CalWin[0]*CalWin[1], num), 'F')
            temp[not SHP.PixelInd[:, block_size[ii, 0] : block_size[ii, 1]]] = np.nan #TODO: check tilde equivalent not
            BiasCorrcoh[block_size[ii, 0]: block_size[ii, 1], kk] = np.nanmean(temp, axis=0)

        print(f'{METHOD} BIAS MITIGATION: {kk}/{npages} is finished...')


    BiasCorrcoh = np.exp(BiasCorrcoh) if EstAgr == 'sec' else np.abs(BiasCorrcoh)
    BiasCorrcoh = np.reshape(BiasCorrcoh, (nlines, nwidths, npages), 'F')

    elapsed = time.time() - t
    print(f'BiasCorrAcc operation completed in {elapsed/60} minute(s)')
    print('Done')

    return BiasCorrcoh