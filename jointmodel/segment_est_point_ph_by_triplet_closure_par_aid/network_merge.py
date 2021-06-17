import numpy as np
from scipy.sparse import find

from helper import intersect_mtlb


def network_merge( subset1,subset2,num1,num2,num_threshold=20):

    idx1 = find(num1 > num_threshold)[1]
    idx2 = find(num2 > num_threshold)[1]
    if idx1:
        subset1 = subset1[idx1]
        num1 = num1[idx1]
    else:
        subset1 = []
        num1 = []

    if idx2:
        subset2 = subset2[idx2]
        num2 = num2[idx2]
    else:
        subset2 = []
        num2 = []

    subset_final_tmp = []

    for i in range(len(idx1)):
        for j in range(len(idx2)):
            if num2[j]==0:
                continue
            else:
                tmp1 = subset1[i].reshape(-1, 1)
                tmp2 = subset2[j].rehsape(-1, 1)
                inter = intersect_mtlb(tmp1, tmp2)[1]
                if inter:
                    tmp = np.unique(np.vstack((tmp1,tmp2)), axis=0)
                    subset_final_tmp = np.unique(np.vstack((subset_final_tmp, tmp)), axis=0)
                    #TODO: check what is subset2 and how to remove element properly
                    subset2[j] = []
                    num2[j] = 0
                    subset1[i] = subset_final_tmp
        num1[i] = len(subset1[i])
        subset_final_tmp = []

    for i in range(len(idx1)-1):
        for j in range(i+1, len(idx1)):
            if num1[j]==0:
                continue
            else:
                tmp1 = subset1[i].reshape(-1, 1)
                tmp2 = subset1[j].reshape(-1, 1)
                inter = intersect_mtlb(tmp1, tmp2)[1]
                if inter:
                    tmp = np.unique(np.vstack((tmp1,tmp2)), axis=0)
                    subset_final_tmp = np.unique(np.vstack((subset_final_tmp,tmp)), axis=0)
                    #TODO: check what is subset2 and how to remove element properly
                    subset1[j] = []
                    num1[j] = 0
                    subset1[i] = subset_final_tmp
        subset_final_tmp = []
        num1[i] = len(subset1[i])
    idx1_final = num1!=0
    idx2_final = num2!=0

    subset_final = np.concatenate((subset1[idx1_final], subset2[idx2_final]))
    idx_sort = np.argsort(np.concatenate((num1[idx1_final], num2[idx2_final]))[::-1])
    flag = np.concatenate((num1[idx1_final], num2[idx2_final]))[::-1].sort()
    subset_final = subset_final[idx_sort]

    return subset_final, flag
