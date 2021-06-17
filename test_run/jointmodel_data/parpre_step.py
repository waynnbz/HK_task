import numpy as np
from datetime import datetime
from numpy.linalg import pinv


# def parpre_step(obs,shortbaseline,num,wavelen,slantran,incangle,
#                 spa_r,spa_azi,width,lines,interval_r,interval_azi,radius,
#                 ref_point=1,ref_slc=1):
#     '''
#     PARPRE is used to prepare the general input parameters for the estimators
# 
#     Input:      obs-------------------observation file (range,azimuth,longitude,latitude,ifg1,ifg2,....)
#                 shortbaseline---------baseline file (im1 im2 B_p, B_t)
#                 num-------------------number of sampled points for joint detrend
#                 wavelen---------------the wavelength of the SAR signal(meters)
#                 slantran--------------the slant range to teh center pixel
#                 incangle--------------the incidence angle (center pixel)
#                 spa_r-----------------spatial resolution in range direction
#                 spa_azi---------------spatial resolution in azimuth direction
#                 width-----------------range samples of interferogram
#                 lines-----------------azimuth lines of interferogram
#                 interval_r------------the grid interval in range direction used for networking
#                 interval_azi----------the grid interval in azimuth direction used for networking
#                 radius----------------the searching radius used for networking
#                 ref_point-------------reference point index
#                 ref_slc---------------reference slc index
#
#     Output:     obs            = observation file (range,azimuth,longitude,latitude,ifg1,ifg2,....)
#                 Input.baseline = the spatial baseline of the selected interf.
#                 Input.interval = the time span of the neighboring SLCs
#                 Input.pindex   = the pair index showing the construction of interf.
#                 Input.tmatrix  = the temporal matrix corresponding to the defo. rates
#                 Input.wavelen  = the wavelength of the SAR signal
#                 Input.slantran = the slant range to teh center pixel
#                 Input.incangle = the incidence angle (center pixel)
#                 Input.spa_r    = spatial resolution in range direction
#                 Input.spa_azi  = spatial resolution in azimuth direction
#                 Input.width    = range samples of interferogram
#                 Input.lines    = azimuth lines of interferogram
#                 Input.interval_r=the grid interval in range direction used for networking
#                 Input.interval_azi=the grid interval in azimuth direction used for networking
#                 Input.radius   = the searching radius used for networking
#                 Input.ref_point= reference point index
#                 Input.ref_slc  = reference slc index
#                 Input.date     = SAR data acquisition date
#                 Input.sb       = shortbaseline file
#                 Input.NTCP     = number of TCP points
#                 Input.num      = number of sampled points for joint detrend
#                 Input.NIFG     = number of interferograms
#                 Input.NSLC     = number of SLC
#                 Input.h2p      = design matrix for topographic error
#                 Input.T_year   = vector of temporal interval in unit of year
#                 Input.T_year_matrix= matrix of temporal interval in unit of year
#                 Input.B_t      = sparse matrix of temporal interval in unit of year
#                 Input.B_t2p    = design matrix for vi
#                 Input.B_t2p_sum= design matrix for deformation rate
#                 Input.B_vi     = design matrix for both topographic error and vi
#                 Input.B_v      = design matrix for both topographic error and rate
#                 Input.B_di     = design matrix for both topographic error and di
#                 Input.B_t2p_di = design matrix for di
#                 Input.CoreNum  = number of workers to start on local machine numworks in Matlab
#                 Input.num_threshold= threshold of number of points to be discarded in subset detection
#                 Input.patch_num_2= patch number index in subset detection (total patch number = (patch_num_2^2)*(patch_num_2^2))
#                 Input.arc_threshold=threshold of residual for arcs to be removed
#                 Input.NTCP_SAMP= number of TCP points in sample
#                 Input.NARC     = number of arcs
#         '''

import scipy.io as sio
import pandas as pd
obs = sio.loadmat('obs.mat')['obs']
shortbaseline = np.array(pd.read_csv('shortbaseline', delimiter='\s+',
                        names=['SCE1', 'SCE2', 'Bp', 'Bt']))

num,wavelen,slantran,incangle,spa_r,spa_azi,width,lines,interval_r,interval_azi,radius = \
    80000,0.031228381041666666,667425.0045,20.1048,11.181454768277943,11.313365,760,640,300,300,700

ref_point=1; ref_slc=1

SCE12 = shortbaseline[:, [0, 1]]
SCE = np.unique(SCE12).reshape(-1, 1)
NSLC = len(SCE)
NIFG, imp_size2 = SCE12.shape
input_pair_index = np.zeros((NIFG, imp_size2), dtype='int')
for i in range(NSLC):
    n = (SCE12 == SCE[i])
    input_pair_index[n] = i

# create an interval matrix
Ninterval = NSLC - 1
input_matrix_t = np.zeros((NIFG, Ninterval))
for ii in range(NIFG):
    st = input_pair_index[ii, 0]
    ed = input_pair_index[ii, 1]
    input_matrix_t[ii, list(np.arange(st, ed))] = 1


se_pair_index = np.hstack((np.arange(0, NSLC-1).reshape(-1, 1), np.arange(1, NSLC).reshape(-1, 1)))
Bp = shortbaseline[:, 2].reshape(-1, 1)
input_t_interval = np.zeros((Ninterval, 1))

for iii in range(Ninterval):
    s_d = datetime.strptime(str(int(SCE[iii])), '%Y%m%d')
    e_d = datetime.strptime(str(int(SCE[iii + 1])), '%Y%m%d')
    input_t_interval[iii] = (e_d - s_d).days

## design matrix for dem error
h2p = (-4 * np.pi / wavelen) * Bp / (slantran * np.sin(np.deg2rad(incangle)))

## T_year
T_year = (np.abs(input_t_interval) / 365).T
T_year_matrix = np.tile(T_year, (NIFG, 1))

## design matrix for coseismic deformation
co_index = 1
inv2inv_matrix = np.zeros((Ninterval, Ninterval))
for i in range(Ninterval):
    if i < co_index:
        inv2inv_matrix[i, i] = 1
        inv2inv_matrix[i, i+1] = -1
    elif i == co_index:
        inv2inv_matrix[i, i] = 1
    else:
        inv2inv_matrix[i, i] = 1
        inv2inv_matrix[i, i-1] = -1
B_coseis_t2p_di = (-4*np.pi/wavelen)*(input_matrix_t@inv2inv_matrix)*1e-3

## design matrix for coseismic deformation rate new
co_index = 0
if co_index == 0:
    inv_cosei_matrix = np.zeros((Ninterval, 2))
    inv_cosei_matrix[co_index, 0] = 1
    inv_cosei_matrix[co_index + 1:, 1] = 1
# dead part
# elseif
#     co_index == Ninterval
#     inv_cosei_matrix = zeros(Ninterval, 2);
#     inv_cosei_matrix(Ninterval, 2) = 1;
#     inv_cosei_matrix(1: co_index - 1, 1)=1;
# else
#     inv_cosei_matrix = zeros(Ninterval, 3);
#     inv_cosei_matrix(co_index, 2) = 1;
#     inv_cosei_matrix(1: co_index - 1, 1)=1;
#     inv_cosei_matrix(co_index + 1: end, 3)=1;
#     end

## design matrix for sequential ifg
PROJ_LS = pinv(input_matrix_t.T @ input_matrix_t)@input_matrix_t.T
Bp_se = PROJ_LS @ Bp
h2p_coef = (-4 * np.pi / wavelen) / (slantran * np.sin(np.deg2rad(incangle)))
h2p_se = h2p_coef * Bp_se

## design matrix for deforate
B_t = input_matrix_t * T_year_matrix
B_t_se = np.diag(T_year.squeeze())
B_t_sum = np.sum(B_t, 1).reshape(-1, 1)
B_t_se_sum = np.sum(B_t_se, 1).reshape(-1, 1)
B_t2p = (-4 * np.pi / wavelen) * B_t * 1e-3
B_t2p_se = (-4 * np.pi / wavelen) * B_t_se * 1e-3
B_t2p_sum = (-4 * np.pi / wavelen) * B_t_sum * 1e-3
B_t2p_se_sum = (-4 * np.pi / wavelen) * B_t_se_sum * 1e-3
B_t2p_di = (-4 * np.pi / wavelen) * input_matrix_t * 1e-3
B_t2p_vi_coseis = (-4 * np.pi / wavelen) * B_t @ inv_cosei_matrix * 1e-3

## design matrix combination
B_vi = np.hstack((h2p, B_t2p))
B_vi_se = np.hstack((h2p_se, B_t2p_se))
B_v = np.hstack((h2p, B_t2p_sum))
B_v_se = np.hstack((h2p_se, B_t2p_se_sum))
B_di = np.hstack((h2p, B_t2p_di))
B_coseis_di = np.hstack((h2p, B_coseis_t2p_di))

## temporal low pass deformation model
#TODO: fix matrix indexing is flattened in matlab
T_year_cum = np.cumsum(T_year)
M = np.zeros((Ninterval, 3))
M[:, 0] = 1
M[0, 1] = T_year_cum[0] / 2
M[0, 2] = T_year_cum[0]**2 / 6
for i in range(1, Ninterval):
    M[i, 1] = (T_year_cum[i] + T_year_cum[i-1])/2
    M[i, 2] = (T_year_cum[i]**3 - T_year_cum[i-1]**3)/6 / (T_year.squeeze()[i]+np.spacing(1)) #eps was outside in matlab
B_t2pM = B_t2p @ M
BM = np.hstack((h2p, B_t2pM))
B_t_seM = B_t_se @ M
B_t2p_seM = B_t2p_se @ M
B_seM = np.hstack((h2p_se, B_t2p_seM))


## parameters for subset detection
arc_threshold = 3   # 2.5, 1.5
# CoreNum = 6     # feature('numcores')
num_threshold = 50
patch_num_2 = 3     # (2 ^ 3) * (2 ^ 3) = 8 * 8 = 64
patch_num_x = 3     # 32
patch_num_y = 3     # 32

## unique arc flag
unique_arc_flag = 'par'     # 'par' or 'obs'
merge_par_threshold = 3
merge_obs_threshold = 0.7
patch_min_arcs = 50
n_point_search = 50
min_win_ksize = 3    # unit: 2 km
hight_diff_threshold = 1000     # unit m

## save parameters
Input = {}
Input['baseline'] = Bp
Input['interval'] = input_t_interval
Input['pair_index'] = input_pair_index
Input['se_pair_index'] = se_pair_index
Input['tmatrix'] = input_matrix_t
Input['wavelen'] = wavelen
Input['slantran'] = slantran
Input['incangle'] = incangle
Input['date'] = SCE
Input['sb'] = shortbaseline
Input['NTCP'] = obs.shape[0]
Input['maxheight'] = max(obs[:, 4])
Input['minheight'] = min(obs[:, 4])
Input['num'] = num
Input['NIFG'] = NIFG
Input['NSLC'] = NSLC
Input['Nintv'] = Ninterval
Input['h2p'] = h2p
Input['h2p_se'] = h2p_se
Input['T_year'] = T_year
Input['T_year_matrix'] = T_year_matrix
Input['B_t'] = B_t
Input['B_t_sum'] = B_t_sum
Input['B_t2p_sum'] = B_t2p_sum
Input['B_t2p'] = B_t2p
Input['B_vi'] = B_vi
Input['B_v'] = B_v
Input['B_di'] = B_di
Input['B_t2p_di'] = B_t2p_di
Input['co_index'] = co_index
Input['inv2inv_matrix'] = inv2inv_matrix
Input['B_coseis_t2p_di'] = B_coseis_t2p_di
Input['B_coseis_di'] = B_coseis_di
Input['B_t2p_vi_coseis'] = B_t2p_vi_coseis
Input['B_t2p_se'] = B_t2p_se
Input['B_t2pM'] = B_t2pM
Input['BM'] = BM
Input['B_t_seM'] = B_t_seM
Input['B_t2p_seM'] = B_t2p_seM
Input['B_seM'] = B_seM
Input['B_t2p_se_sum'] = B_t2p_se_sum
Input['B_v_se'] = B_v_se
Input['B_vi_se'] = B_vi_se
# Input['CoreNum'] = CoreNum #CoreNum removed as multi core is replaced with multi-processing
Input['spa_r'] = spa_r
Input['spa_azi'] = spa_azi
Input['width'] = width
Input['lines'] = lines
Input['interval_r'] = interval_r
Input['interval_azi'] = interval_azi
Input['radius'] = radius
Input['ref_point'] = ref_point
Input['ref_slc'] = ref_slc
Input['num_threshold'] = num_threshold
Input['patch_num_2'] = patch_num_2
Input['patch_num_x'] = patch_num_x
Input['patch_num_y'] = patch_num_y
Input['arc_threshold'] = arc_threshold
Input['unique_arc_flag'] = unique_arc_flag
Input['merge_par_threshold'] = merge_par_threshold
Input['merge_obs_threshold'] = merge_obs_threshold
Input['patch_min_arcs'] = patch_min_arcs
Input['n_point_search'] = n_point_search
Input['minw_ksize'] = min_win_ksize
Input['hdiff_T'] = hight_diff_threshold

    # return Input
