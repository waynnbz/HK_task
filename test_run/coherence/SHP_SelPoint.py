import numpy as np
import time
from scipy.stats import f
from scipy.stats import gamma
from scipy.ndimage import label
from collections import namedtuple

from BWStest import BWStest


def SHP_SelPoint(mlistack, CalWin=(15, 15), Alpha=0.05, EstAgr='HTCI'):
    '''

    :param mlistack:
    :param CalWin:
    :param Alpha:
    :param EstAgr:
    :return:
    '''

    t = time.time()

    assert len(mlistack.shape) == 3, 'Please input 3D matrix...'

    nlines, nwidths, npages = mlistack.shape
    mlistack = np.array(mlistack).astype('float')  # correspond to matlab 'single'

    # Parameter prepare:
    RadiusRow = int((CalWin[0] - 1) / 2) #TODO: check if RadiusRow/Col needs shift
    RadiusCol = int((CalWin[1] - 1) / 2)
    InitRow = int((CalWin[0] + 1) / 2) - 1
    InitCol = int((CalWin[1] + 1) / 2) - 1
    LRT_nl = 3  # 2?
    LRT_nw = 3
    if RadiusRow < LRT_nl:
        LRT_nl = 1  # 0?
    if RadiusCol < LRT_nw:
        LRT_nw = 1

    # stats
    CR_lo = f.ppf(Alpha / 2, 2 * npages, 2 * npages)
    CR_up = f.ppf(1 - Alpha / 2, 2 * npages, 2 * npages)
    Galpha_L = gamma.ppf(Alpha / 2, npages, scale=1)
    Galpha_U = gamma.ppf(1 - Alpha / 2, npages, scale=1)

    # Edeg mirror - image
    mlistack = np.pad(mlistack, pad_width=((RadiusRow, RadiusRow), (RadiusCol, RadiusCol), (0,0)),
                      mode='symmetric')
    meanmli = np.mean(mlistack, 2)
    nlines_EP, nwidths_EP = meanmli.shape
    PixelInd = np.zeros((CalWin[0] * CalWin[1], nlines * nwidths), dtype='bool')

    # estimate SHPs
    num = 0
    p = 1
    all_step = np.floor(nlines * nwidths / 10)

    if EstAgr.upper() == 'HTCI':

        # TODO: OP replacing for loop sliding window
        for kk in range(InitCol, nwidths_EP - RadiusCol):
            for ll in range(InitRow, nlines_EP - RadiusRow):
                # Initial estimation(Likelihood - ratiotest)
                temp = meanmli[ll - LRT_nl:ll + LRT_nl + 1, kk - LRT_nw: kk + LRT_nw + 1]
                T = meanmli[ll, kk] / (temp + np.spacing(1))
                T = np.logical_and(T > CR_lo, T < CR_up)
                SeedPoint = np.mean(temp[T])
                # iteration(Gamma Confidence interval)
                MeanMatrix = meanmli[ll - RadiusRow:ll + RadiusRow + 1, kk - RadiusCol: kk + RadiusCol + 1]
                # added 1 for dim diff in matlab
                SeedPoint = np.logical_and(MeanMatrix > Galpha_L * SeedPoint / npages,
                                           MeanMatrix < Galpha_U * SeedPoint / npages)
                SeedPoint[InitRow, InitCol] = True
                # connection
                LL, _ = label(SeedPoint, np.ones((3, 3)))  # using 8 connectivity
                # get components that is connected to the current pixel group
                PixelInd[:, num] = LL.ravel('F') == LL[InitRow, InitCol]  # Fortran-style(column-major)
                num += 1

                # showing progress
                if num == all_step * p:
                    print(f'progress: {10 * p}%')
                    p += 1

    else:

        # TODO: check dim
        for kk in range(InitCol, nwidths_EP - RadiusCol):
            for ll in range(InitRow, nlines_EP - RadiusRow):
                Matrix = mlistack[ll - RadiusRow:ll + RadiusRow + 1, kk - RadiusCol:kk + RadiusCol + 1, :]
                # added 1 for dim diff in matlab
                Ref = Matrix[InitRow, InitCol, :]
                T = BWStest(np.tile(Ref.ravel('F'), (1, CalWin[0] * CalWin[1])),
                            np.reshape(Matrix, (CalWin[0] * CalWin[1], npages), 'F').T, Alpha)
                temp = np.reshape(1 - T, (CalWin[0], CalWin[1]), 'F')
                # connection
                LL, _ = label(temp, np.ones((3, 3)))  # using 8 connectivity
                PixelInd[:, num] = LL.ravel('F') == LL[InitRow, InitCol]  # TODO: check dim
                num += 1

                # showing progress
                if num == all_step * p:
                    print(f'progress: {10 * p}%')
                    p += 1

    BroNum = sum(PixelInd, 0)
    BroNum = np.reshape(BroNum.ravel('F'), (nlines, nwidths), 'F')
    BroNum = np.float32(BroNum - 1)
    elapsed = time.time() - t

    # matlab plot

    print(f'SHP_SelPoint operation completed in {elapsed} second(s).')
    print('Done!')

    SHP = {}
    SHP['PixelInd'] = PixelInd
    SHP['BroNum'] = BroNum
    SHP['CalWin'] = CalWin

    return SHP