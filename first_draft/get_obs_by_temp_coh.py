import time

import numpy as np
import os
import pickle
from scipy.sparse import find

from bash_function import *
from tempcohMST_Est import *


def get_obs_by_temp_coh(ref_sce,rep_sce,an,rn,image_format1,image_format2,cpd,wpd,T,SHP=[]):
    if not SHP:
        SHP_BroNum = np.zeros(an, rn)
    else:
        SHP_BroNum = SHP['BroNum']

    pt_looks = SHP_BroNum.ravel('F') + 1
    NIFG = len(ref_sce)
    os.chdir(cpd)

    if os.path.isfile('tempcoh.pickle') and os.path.isfile('MST_idx.pickle'):
        tempcoh = pickle.load(open('tempcoh.pickle', 'rb'))
        MST_idx = pickle.load(open('MST_idx.pickle', 'rb'))
        retempcoh = np.reshape(tempcoh, (an, rn), 'F')

    else:
        cpxphresidual = np.zeros((an * rn, NIFG)) + 1j * np.zeros((an * rn, NIFG))
        print('Start select coherent point from coherence map...')
        print('Reading coherence map...')
        for i in range(NIFG):
            os.chdir(wpd)
            coh_name = str(ref_sce[i]) + '-' + str(rep_sce[i]) + '.tflt.coh'
            coh, _ = read_file(coh_name, an, image_format1)
            ifgfilt_name = str(ref_sce[i]) + '-' + str(rep_sce[i]) + '.tflt.filt'
            ifgfilt, _ = read_file(ifgfilt_name, an, image_format2)
            ifgcoh = coh * ifgfilt

            os.chdir(cpd)
            phase_name = str(ref_sce[i]) + '-' + str(rep_sce[i]) + '.tflt'
            ifg, _ = read_file(phase_name, an, image_format2)
            cpxphresidual[:, i] = ifgcoh.ravel('F') * np.conj(ifg.ravel('F'))

        BiasAgr = 'jackknife'
        t = time.time()
        tempcoh, MST_idx = tempcohMST_Est(cpxphresidual, ref_sce, rep_sce, BiasAgr)
        elapsed = time.time()
        pickle.dump(tempcoh, open('tempcoh.pickle', 'wb'))
        pickle.dump(MST_idx, open('MST_idx.pickle', 'wb'))
        del cpxphresidual
        retempcoh = np.flipud(np.reshape(tempcoh, (an, rn)))

        ### plot code removed

    ixy = find(tempcoh >= T)
    ix, iy, _ = find(retempcoh >= T)
    ixy_num = len(ixy)
    obs = np.zeros((ixy_num, 5 + NIFG))
    obs[:, 0] = iy
    obs[:, 1] = ix
    coh = np.zeros((ixy_num, 3 + NIFG))
    coh[:, [0, 1]] = np.hstack((iy, ix))
    coh[:, 2] = pt_looks[ixy]
    print('Start retrieving filtered phase observations...')
    os.chdir(wpd)

    for i in range(NIFG):
        phase_name = str(ref_sce[i]) + '-' + str(rep_sce[i]) + '.tflt.filt'
        ifg, _ = read_file(phase_name, an, image_format2)

        ifgph = np.angle(ifg)
        obs[:, i + 5] = ifgph[ixy]
        coh_name = str(ref_sce[i]) + '-' + str(rep_sce[i]) + '.tflt.coh'
        ifgcoh, _ = read_file(coh_name, an, image_format1)
        coh[:, i + 3] = ifgcoh[ixy]

    pickle.dump(ixy, open('ixy.pickle', 'wb'))
    print(f'Totally {NIFG} interferograms are selected...')
    print(f'Totally {ixy_num} points are selected...')
    os.chdir(cpd)
    pickle.dump(coh, open('coh.pickle', 'wb'))

    return obs, coh
