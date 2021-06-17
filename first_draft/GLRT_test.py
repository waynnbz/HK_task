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


def BWS_test(X,Y,alpha=0.05):
    ''' 
    功能
    ===========
    应用Baumgartnerdeng提出的非参数的秩检验方法
    -BWS检验方法检验两组独立样本是否均为某种分布，
    当返回的H值为0时认为零假设成立，即该样本统计上
    可认为是相同的分布，否则返回值1。

    原理
    ===========
    BWS检验属于秩和检验方法,是KS检验的加权版本，
    其基本思想是对经验分布差的平方进行加权，
    而权的确定依赖于真实分布，考虑到真实分布是
    未知的，检验统计量B用秩近似加权。参见文献
    《A Nonparametric Test for the General 
    Two-Sample Problem》, W. Baumgartner, P. Weiß 
    and H. Schindler%  Biometrics Vol. 54,
    No. 3 (Sep., 1998), pp. 1129-1135。
    应用了Pandas数据分析库。
    
        
    参数
    ===========
    X    : 待检验的样本数列1;
    Y    : 待检验的样本数列2;
    alpha: 显著性水平，可以取为0.05或0.01，
           这里默认为0.05。   
    
    信息
    ===========
    作者    :  陈先龙
    邮箱    :  yinzhe@uni-space.net.cn
    版本    :  0.5
    创建日期:  2019-11-15
        
    示例
    ===========
    >>> fenbu1 = np.random.normal(7,5,1000000)
    >>> fenbu2 = np.random.normal(7,5,100000)
    >>> fenbu3 = np.random.normal(7,3,1000000)
    >>> BWS_test(X1,X2)
    不能拒绝这两个样本数列为同一种分布假设。
    0.4095055372889697
    >>> ks_2(X1,X3)
    不能认这两个样本数列为同一种分布。
    0.0
    '''

    n = len(X)
    m = len(Y)

    # 确定判别标准 
    biaozhun = defaultdict(lambda:2.493)
    biaozhun.update({5:2.533,6:2.552,7:2.62,8:2.564,9:2.575,10:2.583})  
    if alpha==0.01: b =3.88
    else: 
        b = biaozhun[n]
    
    # 秩和检验
    fb = list(X) + list(Y)
    s = pd.Series(fb)
    s1 =s.rank()
    Xrank = s1[:n]
    Yrank = s1[-m:]
    
    x = np.arange(1,n+1)
    fenzi = 1.0/n*(Xrank-(m+n)*x/n)**2
    fenmu = x/(n+1)*(1-x/(n+1))*m*1.0*(m+n)/n
    Bx = sum(fenzi/fenmu)
    
    y = np.arange(1,m+1)
    fenziy = 1.0/m*(Yrank-(m+n)*y/m)**2
    fenmuy = y/(m+1)*(1-y/(m+1))*n*1.0*(m+n)/m
    By = sum(fenziy/fenmuy)

    B = 0.5*(Bx + By)
        
    if B>=b:
        H = 0
        print('不能拒绝这两个样本数列为同一种分布假设。')
    else:
        H = 1
        print('不能认这两个样本数列为同一种分布。')
    
    return H,Bx,By,b

print(BWS_test(fenbu1,fenbu2))
print(BWS_test(fenbu1,fenbu3))
print(BWS_test(fenbu1,fenbu4))

f1 = [0.0166,0.0247,0.0295,0.0588,0.0642]
f2 = [0.0178,0.0182,0.0202,0.0393,0.0906]

print(BWS_test(f1,f2))




