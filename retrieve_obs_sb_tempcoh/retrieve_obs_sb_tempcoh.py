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

    #TODO: confirm sce is 'len-able' (array or list)
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



def retrieve_obs_sb_tempcoh(hW = 7,T = 0.15,CoreNum = 4, data_path=Path.cwd()):
    '''
    all data are expected in data_path directory
    pickle module is used for data IO in binary
    '''

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

    obs = get_obs_by_temp_coh(SCE1, SCE2, azimuth_num, range_num, bkformat1, bkformat2, cpd, wpd, T, SHP)
    # obs = get_obs_by_temp_coh(SCE1, SCE2, azimuth_num, range_num, bkformat1, bkformat2, cpd, wpd, T)

    pickle.dump(obs, open(cpd / 'obs.pickle', 'wb'))
    # save cohere_phase_path wpd
    output = os.path.isfile('obs.pickle')

    return output
