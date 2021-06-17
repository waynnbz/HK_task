import numpy as np

def prepare_seg_par(obs,range_step,azimuth_step):

    max_range = max(obs[:, 0])
    min_range = min(obs[:, 0])
    max_azimuth = max(obs[:, 1])
    min_azimuth = min(obs[:, 1])
    num_seg_range = int(np.ceil((max_range - min_range) / range_step))
    num_seg_azimuth = int(np.ceil((max_azimuth - min_azimuth) / azimuth_step))
    ra_cor = np.linspace(min_range, max_range, num_seg_range + 1)
    az_cor = np.linspace(min_azimuth, max_azimuth, num_seg_azimuth + 1)
    overlap = max((ra_cor[1] - ra_cor[0]), (az_cor[1] - az_cor[0])) / 8
    max_seg_num = num_seg_range * num_seg_azimuth

    seg_par = {}
    seg_par['num_seg_range'] = num_seg_range
    seg_par['num_seg_azimuth'] = num_seg_azimuth
    seg_par['ra_cor'] = ra_cor
    seg_par['az_cor'] = az_cor
    seg_par['overlap'] = overlap
    seg_par['max_seg_num'] = max_seg_num

    return seg_par
