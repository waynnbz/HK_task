import numpy as np
import pandas as pd
from pathlib import Path
import os
import pickle
import glob

from grepfind import *
from AdpCohEst import *
from BiasCorrAcc import *
from IntfFilt import *
from get_obs_by_temp_coh import *
from SHP_SelPoint import *
from bash_function import *


## sub function -- readamp_stack
def readamp_stack(sce,width,lines,sub_name,image_format):

    NSLC = len(sce)
    mlistack = {}
    mlistack['datastack'] = np.zeros((lines, width, NSLC))
    mlistack['filename'] = sce
    for i in range(NSLC):
        amp_name = str(sce[i]) + sub_name
        amp, _ = read_file(amp_name, lines, image_format)
        mlistack['datastack'][:, :, i] = amp

    return mlistack

## sub function readifg_stack
def readifg_stack(ref_sce,rep_sce,width,lines,sub_name,image_format):
    NIFG = len(ref_sce) # check len is applicable
    real_part = np.zeros(lines, width, NIFG)
    imag_part = np.zeros(lines, width, NIFG)
    ifgstack = {}
    ifgstack['datastack'] = complex(real_part, imag_part)
    ifgstack['filename'] = (ref_sce, rep_sce)
    for i in range(NIFG):
        ifg_name = str(ref_sce[i]) + '-' + str(rep_sce[i]) + sub_name
        ifg, _ = read_file(ifg_name, lines, image_format)
        ifgstack['datastack'][:,:,i] = ifg

    return ifgstack

## sub function writecoh_stack
def writecoh_stack(cohstack,ref_sce,rep_sce,sub_name,image_format):
    NIFG = len(ref_sce) # check len applicable
    for i in range(NIFG):
        ifgcoh_name = str(ref_sce[i]) + '-' + str(rep_sce[i]) + sub_name
        ifgcoh = cohstack[:,:,i]
        write_file(ifgcoh, ifgcoh_name, image_format)

## sub function writeifgfilt_stack
def writeifgfilt_stack(ifgfilt_stack,ref_sce,rep_sce,sub_name,image_format):
    lines, width, NIFG = ifgfilt_stack.shape
    amp = np.ones(lines, width)
    for i in range(NIFG):
        ifgfilt_name = str(ref_sce[i]) + '-' + str(rep_sce[i]) + sub_name
        ifgfilt_ph = ifgfilt_stack[:,:, i]
        ifgfilt = amp * np.exp(1j*ifgfilt_ph)
        write_file(ifgfilt, ifgfilt_name, image_format)



#def retrieve_obs_sb_tempcoh(hW = 7,T = 0.15,CoreNum = 4, data_path=Path.cwd()):
'''
all data are expected in data_path directory
pickle module is used for data IO in binary 
'''

### test input ###
data_path = 'data/'
hW = 7
T = 0.15
CoreNum = 4
###

# check number input args
assert np.logical_and(hW>0, hW==np.fix(hW)), 'Half Window size must be a finite, positive integer!'
assert np.logical_and(T<1, T>=0), 'coherence threshold must be a positive real number between 0 and 1!'
assert np.logical_and(CoreNum>0, CoreNum==np.fix(CoreNum)), 'Number of workers must be a finite, positive integer!'

hW = min(hW, 10)
mw = 2*hW+1

#TODO: read shortbaseline & file params
cpd = Path(data_path)
os.chdir(cpd)
cpd = Path.cwd()
if os.path.isfile('shortbaseline'):
    print('Read shortbaseline file')
    C = pd.read_csv('shortbaseline', delimiter='\s+',
                    names=['SCE1', 'SCE2', 'Bp', 'Bt'])
else:
    raise Exception('shortbaseline file doesn''t exist!')

SCE1, SCE2, Bp, Bt = C.SCE1, C.SCE2, C.Bp, C.Bt
SCE = np.unique(np.vstack((SCE1, SCE2)))

mlifile = list(glob.glob('*.mli'))
parfile = list(glob.glob('*.mli.par'))
fltfile = list(glob.glob('*flt'))
# assert len(SCE) < len(mlifile), 'The number of scenes in shortbaseline/baseline file is larger than the number of mli files in current directory!'
# assert len(SCE1) < len(fltfile), 'The number of interferograms in shortbaseline/baseline file is larger than the number of flt files in current directory!'

# grepfind returns None if not find
azimuth_num = int(grepfind(parfile[0], 'azimuth_lines'))
range_num = int(grepfind(parfile[0], 'range_samples'))
azimuth_lk = int(grepfind(parfile[0], 'azimuth_looks'))
range_lk = int(grepfind(parfile[0], 'range_looks'))
looks = azimuth_lk * range_lk
data_format = grepfind(parfile[0], 'image_format')

if data_format == 'SHORT':  # case sensitive?
    bkformat1 = 'int16'
    bkformat2 = 'cpxint16'
elif data_format == 'FLOAT':
    bkformat1 = 'float32'
    bkformat2 = 'cpxfloat32'
else:
    raise Exception('data format is unknown!')

## read mlistack
if os.path.isfile('mlistack.pickle'):
    mlistack = pickle.load(open('mlistack.pickle', 'rb'))
else:
    sub_name = '.mli'
    mlistack = readamp_stack(SCE, range_num, azimuth_num, sub_name, bkformat1)
    pickle.dump(mlistack, open('mlistack.pickle', 'wb'))

## select SHP
if os.path.isfile('SHP.pickle'):
    SHP = pickle.load(open('SHP.pickle', 'rb'))
else:
    CalWin = (mw, mw)
    Alpha = 0.05
    EstAgr = 'HTCI'
    SHP = SHP_SelPoint(mlistack['datastack'], CalWin, Alpha, EstAgr)
    pickle.dump(SHP, open('SHP.pickle', 'wb'))
    # version -v7.3 used in matlab, not sure nocompression applied or not

## coherence estimation and filtering
Path('coherence_phase').mkdir(parents=True, exist_ok=True)
os.chdir('coherence_phase')
wpd = Path.cwd()
NIFG=len(SCE1)
for i in range(NIFG):
    os.chdir(wpd)
    print(f'Processing ifg {i} of {NIFG}...')
    ref_sce = SCE1[i]
    rep_sce = SCE2[i]
    ifgname = str(ref_sce) + '-' + str(rep_sce) + '.tflt'
    ifgcoh_name = ifgname + '.coh'
    ifgfilt_name = ifgname + '.filt'
    #AdpCohEst
    if not os.path.isfile(ifgcoh_name):
        os.chdir(cpd)
        print('Estimate coherence...')
        sub_name = '.tflt'
        ifg = readifg_stack(ref_sce, rep_sce, range_num, azimuth_num, sub_name, bkformat2)
        Btflag = False
        Acc = False
        Coh, Cpxcoh =AdpCohEst(mlistack['datastack'], mlistack['filename'], ifg['datastack'],ifg['filename'], SHP, Btflag, Acc)
        #Coh,Cpxcoh = AdpCohEst(mlistack['datastack'],mlistack['filename'],ifg['datastack'],ifg['filename'])
        Biascorrcoh = BiasCorrAcc(Cpxcoh, SHP, 'sec')
        #TODO: OP chdir?
        os.chdir(wpd)
        sub_name = '.tflt.coh'
        writecoh_stack(Biascorrcoh, ref_sce, rep_sce, sub_name, bkformat1)

    # IntfFilt
    if not os.path.isfile(ifgfilt_name):
        print('Filter the complex interferogram...')
        Acc = False
        ADPPh = IntfFilt(ifg.datastack, Biascorrcoh, SHP, [mw, mw], Acc)
        sub_name = '.tflt.filt'
        writeifgfilt_stack(ADPPh, ref_sce, rep_sce, sub_name, bkformat2)



# %%     get_obs_by_temp_coh

### inputs
ref_sce, rep_sce,an,rn,image_format1,image_format2,cpd,wpd,T,SHP = (SCE1, SCE2, azimuth_num, range_num, bkformat1, bkformat2, cpd, wpd, T, SHP)
###


if not SHP:
    SHP_BroNum = np.zeros(an, rn)
else:
    SHP_BroNum = SHP['BroNum']

pt_looks = SHP_BroNum.ravel('F')+1
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
    # tempcoh, MST_idx = tempcohMST_Est(cpxphresidual, ref_sce, rep_sce, BiasAgr)
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
obs[:, 0]=iy
obs[:, 1]=ix
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

# %%      tempcohMST_Est.py


### test import
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import find
###

## intputs

cpxphresidual,ref_sce,rep_sce,BiasAgr = cpxphresidual, ref_sce, rep_sce, BiasAgr

###

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

# ## segment for paralell processing
# blockstep = 1000000
# NPS = cpxphresidual.shape[0]
# blocknum = int(np.ceil(NPS / blockstep))
# mst_cpxphresidual = {}
# MST_idx = {}
#
# for j in range(blocknum):
#     print(f'Process block number {j} of {blocknum}')
#     pt_sidx = j * blockstep
#     pt_eidx = min((j + 1) * blockstep, NPS)
#     pt_idx = np.arange(pt_sidx, pt_eidx)
#     nps_blk = len(pt_idx)
#     cohphresidual_blk = np.abs(cpxphresidual[pt_idx, :])
#     inv_cohphresidual_blk = 1 - cohphresidual_blk + np.spacing(1)
#     mst_cpxphresidual_blk = np.zeros((nps_blk, Nintv)) + 1j * np.zeros((nps_blk, Nintv))  # complex
#     MST_idx_blk = np.zeros((nps_blk, Nintv))
#     # TODO: replace with mutliprocessing module
#     # matlabpoolchk(CoreNum)
#     for i in range(nps_blk):
#         SLC_SLC = csr_matrix((inv_cohphresidual_blk[i, :].T.squeeze(),
#                               (image_pair[:, 0].squeeze(), image_pair[:, 1].squeeze())),
#                              shape=(NSLC, NSLC)).toarray()
#         UG = np.tril(SLC_SLC + SLC_SLC.T)
#         ST = minimum_spanning_tree(UG)
#         mst_x, mst_y, _ = find(ST)
#         mst_xy = np.hstack((mst_x.reshape(-1, 1), mst_y.reshape(-1, 1)))
#         mst_xy = np.sort(mst_xy, axis=1)
#         # ia = intersect_rows(image_pair, mst_xy)
#         ia = [i for i, v in enumerate(image_pair) if v.tolist() in mst_xy.tolist()]
#         MST_idx_blk[i, :] = ia
#         mst_cpxphresidual_blk[i, :] = cohphresidual_blk[i, ia]
#
#     mst_cpxphresidual[j] = mst_cpxphresidual_blk
#     MST_idx[j] = MST_idx_blk
#
# print('Estimating temporal coherence...')
# if not BiasAgr:
#     tempcoh = np.abs(np.sum(mst_cpxphresidual, 1)) / Nintv
# elif BiasAgr == 'jackknife':
#     tempcoh0 = np.abs(np.sum(mst_cpxphresidual, 1)) / Nintv
#     # idx is a matrix with incremental values in each row, and remove its diagonal elements
#     idx = np.tile(np.arange(Nintv), (Nintv, 1))
#     idx = idx[~np.eye(Nintv, dtype=bool)].reshape(Nintv, Nintv-1)
#     tempcoh_jk = []
#     for j in range(blocknum):
#         print(f'Process block number {j} of {blocknum}')
#         pt_sidx = j * blockstep
#         pt_eidx = min((j + 1) * blockstep, NPS)
#         pt_idx = np.arange(pt_sidx, pt_eidx)
#         nps_blk = len(pt_idx)
#         mst_cpxphresidual_blk = mst_cpxphresidual[pt_idx,:]
#         tempcoh_jk_blk = np.zeros((nps_blk, 1))
#         #TODO: OP with multi-processing
#         for i in range(nps_blk):
#             temp = mst_cpxphresidual_blk[i,:]
#             temp_jkmatrix = temp[idx]
#             tempcoh_jk_blk[i, 1] = np.mean(np.abs(np.sum(temp_jkmatrix, 2)) / (Nintv - 1))
#         tempcoh_jk.append(tempcoh_jk_blk)
#     tempcoh_jk = np.concatenate(tempcoh_jk, axis=1)
#     tempcoh = Nintv * tempcoh0 - (Nintv - 1) * tempcoh_jk
#     tempcoh[tempcoh < 0] = 0




        # cohphresidual_blk = np.abs(cpxphresidual[pt_idx, :])
        # inv_cohphresidual_blk = 1 - cohphresidual_blk + np.spacing(1)
        # mst_cpxphresidual_blk = np.zeros((nps_blk, Nintv)) + 1j * np.zeros((nps_blk, Nintv))  # complex
        # MST_idx_blk = np.zeros((nps_blk, Nintv))






# %%

def intersect_rows(arr1, arr2):
    l = []
    for i in range(len(arr1)):
        for j in range(len(arr2)):
            flag = arr1[i] == arr2[j]
            if len(np.unique(flag)) == 1 and np.unique(flag)[0] == True:
                l.append(i)
    return np.array(list(set(l))).reshape(-1, 1)



# %%
obs = get_obs_by_temp_coh(SCE1, SCE2, azimuth_num, range_num, bkformat1, bkformat2, cpd, wpd, T, SHP)
#obs = get_obs_by_temp_coh(SCE1, SCE2, azimuth_num, range_num, bkformat1, bkformat2, cpd, wpd, T)

pickle.dump(obs, open(cpd/'obs.pickle', 'wb'))
#save cohere_phase_path wpd
output = os.path.isfile('obs.pickle')

