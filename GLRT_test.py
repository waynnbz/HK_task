# -*- coding: utf-8 -*-
"""
###################################################
###         									###
###				最大似然比检测				    ###
###												###			
###		原理									###
###		===========								###
###		根据最大似然比检测方法编写，用以对两个  ###
###     独立的瑞利分布进行检验                  ###
###												###
###		参数									###
###		===========								###
###						                        ###
###												###
###		信息									###
###		==========								###
###		作者 	:  陈先龙						###
###		邮箱 	:  yinzhe@uni@cgnpc.com.cn	    ###
###		版本 	:  0.9							###
###		创建日期:  2019-12-13					###
###												###
###												###
###												###
###################################################
"""

import scipy.stats as st
import numpy as np
import pandas as pd
from scipy import signal, misc
import matplotlib.pyplot as plt


alpha =0.05
num = 50
sigma = 200.0/np.sqrt(2);
rep = 10000


# 似然比函数版本（两个瑞利分布的独立样本）
def lamda(sigma,num):
    r1 = np.random.rayleigh(sigma,num)
    r2 = np.random.rayleigh(sigma,num)
    theta_r1 = sum(r1**2)/num
    theta_r2 = sum(r2**2)/num
    theta = 0.5*(theta_r1 + theta_r2)
    lamda = 4*num*np.log(theta) - 2*num*np.log(theta_r1) - 2*num*np.log(theta_r2)
    
    return lamda

L = [lamda(sigma, num) for i in range(rep)]
Lambda = pd.Series(L)
C = Lambda.quantile(1-alpha)
print(C)





