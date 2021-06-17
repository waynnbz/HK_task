import numpy as np
import os
import scipy.io as sio

import time

from parpre_step import *
# from phaselink_MLE4 import *
from prepare_seg_par import *
from phaselink_MLE4 import *
# from segment_est_point_ph_by_triplet_closure_par_aid.segment_est_point_ph_by_triplet_closure_par_aid import *


# sample input
# Output=jointmodel(obs,shortbaseline,80000,9.6000000e+09,667425.0045,760,640,20.1048,11.18145476827794302475,11.313365,300,300,700,4)


# def jointmodel(obs,sb,num,frequency,slantran,width,lines,incangle,spa_r,spa_azi,interval_r,interval_azi,radius):


############################################# test input ##################################################
import pandas as pd
obs = sio.loadmat('obs.mat')['obs']
sb = np.array(pd.read_csv('shortbaseline', delimiter='\s+',
                        names=['SCE1', 'SCE2', 'Bp', 'Bt']))
geo_tcp = np.array(pd.read_csv('geo_tcp', header=None, delimiter='\s+'))
height_tcp = np.array(pd.read_csv('height_tcp', header=None, delimiter='\s+'))
obs[:,2:4] = geo_tcp
obs[:,4] = height_tcp.squeeze()

num,frequency,slantran,width,lines,incangle,spa_r,spa_azi,interval_r,interval_azi,radius = \
    80000,9.6000000e+09,667425.0045,760,640,20.1048,11.18145476827794302475,11.313365,300,300,700
###########################################################################################################


c = 299792458
wavelen = c / frequency
ref_point = 1
ref_slc = 1

Input = parpre_step(obs, sb, num, wavelen, slantran, incangle, spa_r, spa_azi, width, lines, interval_r,
                        interval_azi, radius, ref_point, ref_slc)

# if not os.path.isfile('Input.mat'):
#     Input = parpre_step(obs, sb, num, wavelen, slantran, incangle, spa_r, spa_azi, width, lines, interval_r,
#                         interval_azi, radius, ref_point, ref_slc)
# else:
#     Input = sio.loadmat('Input.mat')
coh = sio.loadmat('coh.mat')['coh']

# est_SMph, est_gamma, est_obs = phaselink_MLE4(obs, coh, Input)
#
# sio.savemat('est_SMph.mat', {'est_SMph':est_SMph})
# sio.savemat('est_gamma.mat', {'est_gamma':est_gamma})
# sio.savemat('est_obs.mat', {'est_obs':est_obs})
# # del est_SMph, est_gamma, est_obs


############################# test output from phaselink ###########################################
est_SMph = sio.loadmat('est_SMph.mat')['est_SMph']
est_gamma = sio.loadmat('est_gamma.mat')['est_gamma']
est_obs = sio.loadmat('est_obs.mat')['est_obs']

#########################################################################################################

T = 0.7
idx = est_gamma[:, -1] >= T
obs_new = est_obs[idx,:]
# del est_obs, est_gamma

seg_par = prepare_seg_par(obs_new, 10000, 10000)

############################# test input for seg_est function ###########################################
# import pickle
# from pathlib import Path
# sf = Path('segment_est_point_ph_by_triplet_closure_par_aid/seg_est_test_data')
# pickle.dump(seg_par, open(sf/'seg_par.pickle', 'wb'))

#########################################################################################################

point_ph = segment_est_point_ph_by_triplet_closure_par_aid(obs_new, Input, seg_par, 0)
# sio.savemat("point_ph.mat", point_ph)
# defo_rate, defo, dem_error = point_ph2par(point_ph, Input)






    # Output = 1
    # print('All finish!')
    # return Output
