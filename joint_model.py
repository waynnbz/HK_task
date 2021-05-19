import os
import math
import time
from glob import glob
import numpy as np
import scipy.io as sio
import multiprocessing as mp
import prepare_paras as pp
from phaselink import phase_link_main, combine_est_blk
from ini_point_selection_by_seg import prepare_seg_par, est_pointph_by_triplet_closure_par_aid
# from sample_PS import samp_PS_quadtree

if __name__ == '__main__':
    ## obtain folder path in windows, need modifications in linux system
    cwd = os.getcwd()
    inner_pro_name = glob(cwd + '/*/*/DEM_GEO')[0].split('\\')[-2]
    postprocessing_dir = cwd + '/example_data/postprocessing'
    product_dir = cwd + '/example_data/' + inner_pro_name + '/PRODUCT'

    ## read geo_tcp & hgt_tcp file to substitude cols in obs
    geo_tcp_file = open(postprocessing_dir + '/geo_tcp', 'r')
    geo_tcp_list = geo_tcp_file.read().splitlines()
    geo_tcp = np.zeros((len(geo_tcp_list), 2))
    for i in range(len(geo_tcp_list)):
        [geo_tcp[i, 0], geo_tcp[i, 1]] = geo_tcp_list[i].split()

    hgt_tcp_file = open(postprocessing_dir + '/height_tcp', 'r')
    hgt_tcp = np.array(hgt_tcp_file.read().splitlines())

    obs_mat = sio.loadmat(postprocessing_dir + '/obs.mat')
    obs = obs_mat['obs']
    [ntcp, obs_col] = obs.shape
    obs[:, 2:4] = geo_tcp
    obs[:, 4] = hgt_tcp

    coh_mat = sio.loadmat(postprocessing_dir + '/coh.mat')
    coh = coh_mat['coh']

    ## read short baseline file
    short_bs_file = open(postprocessing_dir + '/shortbaseline', 'r')
    short_bs = []
    for line in short_bs_file.readlines():
        short_bs.append(list(line.split()))
    short_bs_file.close()
    short_bs = np.array(short_bs)

    ## get parameters
    ref_image_file = open('./example_data/ref_image', 'r')
    ref_image = ref_image_file.read().splitlines()[0]
    ref_image_file.close()
    num_sampled_point = 80000  # the number of sampled points for joint model
    range_g_interv = 300  # the grid interval in range direction used for networking
    azi_g_interv = 300  # the grid interval in azimuth direction used for networking
    network_radius = 700  # the searching radius used for networking
    blockstep = 50000  # limit of number of points
    est_threshold = 0.6
    range_step = azimuth_step = 1000
    # cal_core_num = 8  # the number of CPU for calculation

    ## prepare parameters for the joint model
    input_paras = {
        'ref_image': ref_image,
        'num_sampled_point': num_sampled_point,
        'range_g_interv': range_g_interv,
        'azi_g_interv': azi_g_interv,
        'network_radius': network_radius
    }

    input_paras['ntcp'] = ntcp
    input_paras['max_height'] = max(hgt_tcp)
    input_paras['min_height'] = min(hgt_tcp)
    input_paras = pp.read_mli_par(product_dir, ref_image, input_paras)
    input_paras = pp.prepare_input(obs, short_bs, input_paras)

    nslc = input_paras['nslc']
    nifg = input_paras['nifg']

    blocknum = int(math.ceil(ntcp / blockstep))
    # start_time = time.time()
    os.chdir(postprocessing_dir)
    # phase link
    phase_link_main(input_paras, obs, coh, blockstep)
    # print('phase link finished in {:.4f} s'.format(time.time() - start_time))
    [est_SMph, est_gamma, est_obs] = combine_est_blk(blocknum)
    sb_obs = est_obs[np.where(est_obs[:, -1] > est_threshold)]
    seg_par = prepare_seg_par(sb_obs, range_step, azimuth_step)
    for i in range(seg_par['num_segs']):
        est_pointph_by_triplet_closure_par_aid(sb_obs, input_paras, seg_par, i)
    # TODO verify the functinality of the quadtree function
    # samp_PS_quadtree(obs, num_sampled_point) # quadtree
