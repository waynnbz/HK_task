_all_= ['is_power2',
        'sm_2Dmean',
        'CurveFitting_3',
        'leastsq_3',
        'corr_complex',
        'corr_ampli',
        'ks_fenbu',
        'ks_2',
        'CM_test',
        'AD_test',
        'BWS_test',
        'means_filter',
        'KNN_i',
        'KNN_d',
        'KNN_ni',
        'KNN_nd',
        'Match1D',
        'Match2D',
        '',
        ''
       ]



import numpy as np
import pandas as pd
from scipy.optimize import leastsq
from scipy import interpolate
from sympy import Poly
import math
from scipy import optimize
from scipy.stats import kstest, ks_2samp
from collections import defaultdict
from sklearn.neighbors import KDTree
import bisect
import scipy.integrate as sci


def is_power2(num):
    '''   
    功能
    ===========
    判断是否为2的乘方数，是返回True。否返回False。

    原理
    ===========
    参见http://code.activestate.com/recipes/577514-chek-if-a-number-is-a-power-of-two/。
        
    参数
    ===========
    num : 待判断的整数。   
           
    示例
    ===========
    >>> is_power2(32)
    True
    >>> is_power2(24)
    False
    '''
    
    return  num != 0 and ((num & (num-1)) == 0)


def sm_2Dmean(my_Dataframe, meta):
    '''
    对笛卡尔坐标网格的某一属性值进行二维平均处理；
    即一点处的值变为该点前后左右四个点的平均值，
    边界点的值保持不变。
    输入的是带有水平编号和垂直编号作为层次索引的
    DataFrame格式数据和要进行平均处理的属性的名称。
    '''
    
    my_Df = my_Dataframe
    idxname = my_Df.index.names
    list_key1 =  my_Df.index.levels[0]
    list_key2 =  my_Df.index.levels[1]
    
    for idx in list_key1[1:-1]:
        xulie = my_Df.query(idxname[0]+'=='+str(idx))
        chulihou = pjchuli(list(xulie[meta]))
        my_Df.loc[idx,:] = chulihou
        
    for ix in list_key2[1:-1]:
        xlie = my_Df.query(idxname[1]+'=='+str(ix))
        chulihou = pjchuli(list(xlie[meta])) 
        my_Df.loc[(slice(None),ix),meta] = chulihou     
        
    return my_Df   


def CurveFitting_3(X,Y):
    ''' 
    功能
    ===========
    返回优化拟合三次多项式的系数，幂次按照
    从高到低顺序排列，最后为常数项。

    原理
    ===========
    应用scipy库的optimize函数包curve_fit函数。
        
    参数
    ===========
    X : 自变量数组;
    Y : 因变量数组，注意数目应与因变量数目一致。   
    
    信息
    ===========
    作者    :  陈先龙
    邮箱    :  yinzhe@uni-space.net.cn
    版本    :  0.9
    创建日期:  2019-11-14
        
    示例
    ===========
    >>> X = np.array([i*100 for i in range(1,11)])
    >>> Y = np.array([13.5, 15.4, 17.3, 19.1, 21.0, 22.9, 24.8, 26.6, 28.5, 30.1])
    >>> CurveFitting_3(X,Y)
    (-2.175604912877237e-09, 2.9079301046952396e-06, 0.017676182645133577, 11.736666940942648)
    '''
    
    def f_3(x, C3, C2, C1, C0):
        return C3*x*x*x + C2*x*x + C1*x + C0

    C3, C2, C1, C0 = optimize.curve_fit(f_3, X, Y)[0]
    para = C3, C2, C1, C0

    return para
    

def leastsq_3(X,Y):
    ''' 
    功能
    ===========
    返回最小二乘拟合三次多项式的系数，幂次按照
    从高到低顺序排列，最后为常数项。

    原理
    ===========
    应用scipy库的optimize函数包leastsq函数。
        
    参数
    ===========
    X : 自变量数组;
    Y : 因变量数组，注意数目应与因变量数目一致。   
    
    信息
    ===========
    作者    :  陈先龙
    邮箱    :  yinzhe@uni-space.net.cn
    版本    :  0.9
    创建日期:  2019-11-15
        
    示例
    ===========
    >>> X = np.array([i*100 for i in range(1,11)])
    >>> Y = np.array([13.5, 15.4, 17.3, 19.1, 21.0, 22.9, 24.8, 26.6, 28.5, 30.1])
    >>> leastsq_3(X,Y)
    (-2.1756027045085527e-09, 2.9079263388067203e-06, 0.017676184460398568, 11.736666724310911)
    '''
           
    def res(p):
        C3, C2, C1, C0 = p
        return(Y - (C3*X**3 + C2*X**2 + C1*X + C0))

    r = optimize.leastsq(res, [1,1,1,1])
    C3, C2, C1, C0 = r[0]
    para = C3, C2, C1, C0
    
    return para


def corr_complex(X,Y):
    ''' 
    功能
    ===========
    计算两个复数二维（包括一维）复数数组的皮尔逊相干系数，
    相干系数值为复数。

    原理
    ===========
    应用复数数组相干系数的基本定义计算，使用到numpy库。
        
    参数
    ===========
    X : 二维复数数组，注意格式为numpy二维array;
    Y : 二维复数数组，注意两个数组形状相同。   
    
    信息
    ===========
    作者    :  陈先龙
    邮箱    :  yinzhe@uni-space.net.cn
    版本    :  0.9
    创建日期:  2019-11-18
        
    示例
    ===========
    >>> a = np.array([[1 + 1j, 2+3j , 3+3j], 
             [3 - 1j, 4+3j,   1+2j]])
    >>> b = np.array([[5 +0.7j,1+3j, 2], 
             [33, 4-3j, 1-2j]])
    >>> corr_complex(a,b)
    (0.5011991419086997+0.026359353974957422j)
    '''
           
    # 共轭相乘计算相干系数
    fenzi = sum((abs(X)*Y.conj()).flatten())
    fenmu = (sum((abs(X)**2).flatten()) * sum((abs(Y)**2).flatten()))**0.5
    gamma = fenzi / fenmu

    return gamma


def corr_ampli(X,Y):
    ''' 
    功能
    ===========
    计算两个复数二维（包括一维）复数数组振幅信息的
    皮尔逊相干系数，相干系数值为实数。

    原理
    ===========
    应用复数数组相干系数的振幅计算公式计算，使用到numpy库。
        
    参数
    ===========
    X : 二维复数数组，注意格式为numpy二维array;
    Y : 二维复数数组，注意两个数组形状相同。   
    
    信息
    ===========
    作者    :  陈先龙
    邮箱    :  yinzhe@uni-space.net.cn
    版本    :  0.9
    创建日期:  2019-11-18
        
    示例
    ===========
    >>> a = np.array([[1 + 1j, 2+3j , 3+3j], 
             [3 - 1j, 4+3j,   1+2j]])
    >>> b = np.array([[5 +0.7j,1+3j, 2], 
             [33, 4-3j, 1-2j]])
    >>> corr_ampli(a,b)
    0.5549426809729129
    '''
           
    # 应用振幅信息计算相干系数
    fenzi = sum((abs(X)*abs(Y)).flatten())
    fenmu = (sum((abs(X)**2).flatten()) * sum((abs(Y)**2).flatten()))**0.5
    gamma = fenzi / fenmu

    return gamma


def ks_fenbu(X,type_fb='norm',alpha=0.05):
    ''' 
    功能
    ===========
    Kolmogorov-Smirnov test拟合优度检验 (KS检验），
    检验一组样本是否为某种分布，当返回的H值为1时
    认为零假设成立即该样本统计上可认为指定的分布,
    否则返回值0。

    原理
    ===========
    依据总体分布状况，计算出分类变量中各类别的期望频数，
    与分布的观察频数进行对比，判断期望频数与观察频数
    是否有显著差异，从而达到从分类变量进行分析的目的,
    应用scipy库的stats函数包。
    
        
    参数
    ===========
    X      : 待检验的样本数列;
    type_fb: 检验的分布类型，默认为正态分布'norm';
    alpha  : 显著性水平，通常取为0.05或0.01，
             这里默认为0.05。   
    
    信息
    ===========
    作者    :  陈先龙
    邮箱    :  yinzhe@uni-space.net.cn
    版本    :  0.9
    创建日期:  2019-11-14
        
    示例
    ===========
    >>> X = np.random.normal(0,1,1000)
    >>> ks_fenbu(X)
    不能拒绝该样本数列服从指定分布假设。
    0
    >>> R = np.random.rayleigh(1,1000)
    >>> ks_fenbu(R, 'rayleigh')
    不能拒绝该样本数列服从指定分布假设。
    1
    '''
    
    # 检验一个数列是否服从某种分布   
    test_stat = kstest(X, type_fb)
    pvalue = test_stat[1]

    if test_stat[1]>=alpha:
        H = 1
        print('不能拒绝该样本数列服从指定分布假设。')
    else:
        H = 0
        print('不能认为该数列服从指定分布。')
    
    return H,pvalue


def ks_2(X1,X2,alpha=0.05):
    ''' 
    功能
    ===========
    Kolmogorov-Smirnov test拟合优度检验 (KS检验），
    检验两组独立样本是否来自于同一总体，当返回的H
    值为1时认为零假设成立，即该两组独立样本统计上
    可认为是来自于同一总体，否则返回值0。

    原理
    ===========
    依据总体分布状况，计算出分类变量中各类别的期望频数，
    与分布的观察频数进行对比，判断期望频数与观察频数
    是否有显著差异，从而达到从分类变量进行分析的目的,
    应用scipy库的stats函数包。
    
        
    参数
    ===========
    X1   : 待检验的样本数列1;
    X2   : 待检验的样本数列2;
    alpha: 显著性水平，通常取为0.05或0.01，
           这里默认为0.05。   
    
    信息
    ===========
    作者    :  陈先龙
    邮箱    :  yinzhe@uni-space.net.cn
    版本    :  0.9
    创建日期:  2019-11-14
        
    示例
    ===========
    >>> fenbu1 = np.random.normal(7,5,1000000)
    >>> fenbu2 = np.random.normal(7,5,100000)
    >>> fenbu3 = np.random.normal(7,3,1000000)
    >>> ks_2(fenbu1,fenbu2)
    不能拒绝这两个样本来自于同一总体假设。
    0
    >>> ks_2(fenbu1,fenbu3)
    不能认这两个样本来自同一总体。
    1
    '''
    
    # 检验两个独立样本数列是否为同一种分布   
    test_stat = ks_2samp(X1,X2)
    pvalue = test_stat[1]
    
    if pvalue>=alpha:
        H = 1
#        print('不能拒绝这两个样本来自于同一总体假设。')
    else:
        H = 0
#       print('不能认这两个样本来自同一总体。')
    
    return H


def CM_test(X1,X2,alpha=0.05):
    ''' 
    功能
    ===========
    CM拟合优度检验，检验两组样本数相同的独立样本
    是否来自于同一总体，当返回的H值为1时认为
    零假设成立，即该两组独立样本统计上可认为是
    来自于同一总体，否则返回值0。

    原理
    ===========
    参见文献：
    - ANDERSON T W,DARLING D A.Asymptotic theory 
      of certain “goodness of fit”criteria based on 
      stochastic processes [J].The Annals of  
      Mathematical Statistics,1952,23(2):193-212。
    - ANDERSON T W.On the distribution of the 
      two-sample cramer-von mises criterion [J ].
      Annals of Mathematical Statistics, 
      1962,34(1):1148-1159.
    
        
    参数
    ===========
    X1   : 待检验的样本数列1;
    X2   : 待检验的样本数列2，注意X1与X2样本数需相同;
    alpha: 显著性水平，通常取为0.05或0.01，
           这里默认为0.05。   
    
    信息
    ===========
    作者    :  陈先龙
    邮箱    :  yinzhe@uni-space.net.cn
    版本    :  0.9
    创建日期:  2019-12-20
        
    示例
    ===========
    >>> fenbu1 = np.random.normal(7,5,1000000)
    >>> fenbu2 = np.random.normal(7,5,100000)
    >>> fenbu3 = np.random.normal(7,3,1000000)
    >>> CM_test(fenbu1,fenbu2)
    不能拒绝这两个样本来自于同一总体假设。
    0
    >>> CM_test(fenbu1,fenbu3)
    不能认这两个样本来自同一总体。
    1
    '''
    
    # 计算检验统计量   
    N = len(X1)       # 注意X1与X2样本数需相同
    
    fb = list(X1) + list(X2)
    s  = pd.Series(fb)
    s1 =  s.rank()
    X1rank = s1[:N]
    X2rank = list(s1[-N:])
    
    tmpX1 = sum([(X1rank[i]-2*(i+1))**2 for i in range(N)])
    tmpX2 = sum([(X2rank[i]-2*(i+1))**2 for i in range(N)])
    test_stat = (tmpX1 +tmpX2)/(4.0*N**2)
        
    if test_stat>=alpha:
        H = 1
#        print('不能拒绝这两个样本来自于同一总体假设。')
    else:
        H = 0
#        print('不能认这两个样本来自同一总体。')
    
    return H


def AD_test(X1,X2,alpha=0.05):
    ''' 
    功能
    ===========
    AD拟合优度检验，检验两组样本数相同的独立样本
    是否来自于同一总体，当返回的H值为1时认为零假设
    成立，即该两组独立样本统计上可认为是来自于同一
    总体，否则返回值0。

    原理
    ===========
    参见文献：
    - PETTITT A N.A two-sample anderson-darling rank 
      statistic[J].Biometrika.Biometrika,1976,63(1):161-168.
    - GOEL K,ADAM N.A Distribution scatterer interferometry 
      approach for precision monitoring of known surface
      deformation phenomena[J].IEEE Transactions on Geoscience 
      & Remote Sensing,2014,52(9):5454-5468.
    
        
    参数
    ===========
    X1   : 待检验的样本数列1;
    X2   : 待检验的样本数列2，注意X1与X2样本数需相同;
    alpha: 显著性水平，通常取为0.05或0.01，
           这里默认为0.05。   
    
    信息
    ===========
    作者    :  陈先龙
    邮箱    :  yinzhe@uni-space.net.cn
    版本    :  0.9
    创建日期:  2019-12-20
        
    示例
    ===========
    >>> fenbu1 = np.random.normal(7,5,1000000)
    >>> fenbu2 = np.random.normal(7,5,100000)
    >>> fenbu3 = np.random.normal(7,3,1000000)
    >>> AD_test(fenbu1,fenbu2)
    不能拒绝这两个样本来自于同一总体假设。
    0
    >>> AD_test(fenbu1,fenbu3)
    不能认这两个样本来自同一总体。
    1
    '''
    
    # 计算检验统计量   
    N = len(X1)       # 注意X1与X2样本数需相同
    
    fb  = list(X1) + list(X2)
    sf1 = pd.Series(X1)
    sf2 = pd.Series(X2)
    s   = pd.Series(fb)
    
    somme = 0
    for x in fb:
        F1 = len(sf1[sf1<=x])/(N+1) 
        F2 = len(sf2[sf2<=x])/(N+1)
        F  = len(s[s<=x])/(N+1)
        tmp= (F1 - F2)**2/max(F*(1-F),0.001)
        somme = somme + tmp
    
    test_stat = somme*N / 2.0
        
    if test_stat>=alpha:
        H = 1
#        print('不能拒绝这两个样本来自于同一总体假设。')
    else:
        H = 0
#        print('不能认这两个样本来自同一总体。')
    
    return H


def BWS_test(X,Y,alpha=0.05):
    ''' 
    功能
    ===========
    应用Baumgartnerdeng提出的非参数的秩检验方法
    -BWS检验方法检验两组独立样本是否来自于同一总体，
    当返回的H值为1时认为零假设成立，即该两组独立样本
    统计上可认为是来自于同一总体，否则返回值0。

    原理
    ===========
    BWS检验属于秩和检验方法,是KS检验的加权版本，
    其基本思想是对经验分布差的平方进行加权，
    而权的确定依赖于真实分布，考虑到真实分布是
    未知的，检验统计量B用秩近似加权。参见文献
    《A Nonparametric Test for the General 
    Two-Sample Problem》, W. Baumgartner, P. Wei? 
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
    版本    :  0.9
    创建日期:  2019-11-15
        
    示例
    ===========
    >>> fenbu1 = np.random.normal(7,5,1000000)
    >>> fenbu2 = np.random.normal(7,5,100000)
    >>> fenbu3 = np.random.normal(7,3,1000000)
    >>> BWS_test(X1,X2)
    不能拒绝这两个样本来自于同一总体假设。
    0
    >>> ks_2(X1,X3)
    不能拒绝这两个样本来自于同一总体假设。
    0
    >>> f1 = [0.0166,0.0247,0.0295,0.0588,0.0642]
    >>> f2 = [0.0178,0.0182,0.0202,0.0393,0.0906]
    >>> BWS_test(f1,f2)
    不能认这两个样本来自同一总体。
    1
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
    s  = pd.Series(fb)
    s1 =  s.rank()
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

    B = 0.5*(Bx+By)
    
    if B>=b:
        H = 1
#        print('不能拒绝这两个样本来自于同一总体假设。')
    else:
        H = 0
#        print('不能认这两个样本来自同一总体。')
    
    return H


def means_filter(input_image, filter_size):
    '''
    均值滤波器
    :param input_image: 输入图像
    :param filter_size: 滤波器大小
    :return: 输出图像

    注：此实现滤波器大小必须为奇数且 >= 3
    '''
    input_image_cp = np.copy(input_image)  # 输入图像的副本
    filter_template = np.ones((filter_size, filter_size))  # 空间滤波器模板
    pad_num = int((filter_size - 1) / 2)  # 输入图像需要填充的尺寸
    # 填充输入图像
    input_image_cp = np.pad(input_image_cp, (pad_num, pad_num), mode="constant", constant_values=0)  
    m, n = input_image_cp.shape  # 获取填充后的输入图像的大小

    output_image = np.copy(input_image_cp)  # 输出图像

    # 空间滤波
    for i in range(pad_num, m - pad_num):
        for j in range(pad_num, n - pad_num):
            somme = np.sum(filter_template*input_image_cp[i-pad_num:i+pad_num+1, j-pad_num:j+pad_num+1])
            output_image[i, j] = somme / (filter_size ** 2)

    output_image = output_image[pad_num:m - pad_num, pad_num:n - pad_num]  # 裁剪

    return output_image


def KNN_i(X,jz,num=1):
    '''   
    功能
    ===========
    从一个点集中给出到指定点最相邻的几个点和距离。

    原理
    ===========
    应用sklearn库中的neighbors模块的KDTree。
        
    参数
    ===========
    X : 点集坐标矩阵,numpy格式;
    jz: 指定点索引值(从1开始计数);
    k : 近邻点数目,缺省时默认只取最近的一个点。   
           
    信息
    ===========
    作者    :  陈先龙
    邮箱    :  yinzhe@uni-space.net.cn
    版本    :  0.9
    创建日期:  2020-11-24

    示例
    ===========
    >>> np.random.seed(0)
    >>> X = np.random.random((10, 2))
    >>> X
    [[0.5488135  0.71518937]
     [0.60276338 0.54488318]
     [0.4236548  0.64589411]
     [0.43758721 0.891773  ]
     [0.96366276 0.38344152]
     [0.79172504 0.52889492]
     [0.56804456 0.92559664]
     [0.07103606 0.0871293 ]
     [0.0202184  0.83261985]
     [0.77815675 0.87001215]]
    >>> KNN_i(X,5,3)
    array([5, 1, 9], dtype=int64)
    >>> KNN_i(X,5)
    array([5], dtype=int64)
    '''
    
    from sklearn.neighbors import KDTree

    tree = KDTree(X)
    n = num+1  # 算法认为最近的点是其本身
    nearest_dis, nearest_ind = tree.query(X, k=n) 

    KNN_i = nearest_ind[jz-1,1:]
    
    return KNN_i


def KNN_d(X,jz,num=1):
    '''   
    功能
    ===========
    从一个点集中给出到指定点最相邻的几个点和距离。

    原理
    ===========
    应用sklearn库中的neighbors模块的KDTree。
        
    参数
    ===========
    X : 点集坐标矩阵,numpy格式;
    jz: 指定点索引值(从1开始计数);
    k : 近邻点数目,缺省时默认只取最近的一个点。   
           
    信息
    ===========
    作者    :  陈先龙
    邮箱    :  yinzhe@uni-space.net.cn
    版本    :  0.9
    创建日期:  2020-11-24

    示例
    ===========
    >>> np.random.seed(0)
    >>> X = np.random.random((10, 2))
    >>> X
    [[0.5488135  0.71518937]
     [0.60276338 0.54488318]
     [0.4236548  0.64589411]
     [0.43758721 0.891773  ]
     [0.96366276 0.38344152]
     [0.79172504 0.52889492]
     [0.56804456 0.92559664]
     [0.07103606 0.0871293 ]
     [0.0202184  0.83261985]
     [0.77815675 0.87001215]]
    >>> KNN_d(X,5,3)
    array([0.2252094 , 0.39536284, 0.52073358])
    >>> KNN_d(X,5)
    array([0.2252094])
    '''
    
    from sklearn.neighbors import KDTree

    tree = KDTree(X)
    n = num+1  # 算法认为最近的点是其本身
    nearest_dis, nearest_ind = tree.query(X, k=n) 

    KNN_d = nearest_dis[jz-1,1:]

    return KNN_d


def KNN_ni(X,num=1):
    '''   
    功能
    ===========
    从一个点集每个点相邻最近的几个点。

    原理
    ===========
    应用sklearn库中的neighbors模块的KDTree。
        
    参数
    ===========
    X : 点集坐标矩阵,numpy格式;
    k : 近邻点数目,缺省时默认只取最近的一个点。   
           
    信息
    ===========
    作者    :  陈先龙
    邮箱    :  yinzhe@uni-space.net.cn
    版本    :  0.9
    创建日期:  2020-11-24

    示例
    ===========
    >>> np.random.seed(0)
    >>> X = np.random.random((10, 2))
    >>> X
    [[0.5488135  0.71518937]
     [0.60276338 0.54488318]
     [0.4236548  0.64589411]
     [0.43758721 0.891773  ]
     [0.96366276 0.38344152]
     [0.79172504 0.52889492]
     [0.56804456 0.92559664]
     [0.07103606 0.0871293 ]
     [0.0202184  0.83261985]
     [0.77815675 0.87001215]]
    >>> KNN_ni(X,3)
    array([[2, 1, 3],
          [0, 5, 2],
          [0, 1, 3],
          [6, 0, 2],
          [5, 1, 9],
          [1, 4, 0],
          [3, 0, 9],
          [2, 1, 8],
          [3, 2, 0],
          [6, 0, 3]], dtype=int64)
    >>> KNN_ni(X)
    array([[2],
           [0],
           [0],
           [6],
           [5],
           [1],
           [3],
           [2],
           [3],
           [6]], dtype=int64)
    '''
    
    from sklearn.neighbors import KDTree

    tree = KDTree(X)
    n = num+1  # 算法认为最近的点是其本身
    nearest_dis, nearest_ind = tree.query(X, k=n) 

    KNN_k_ind = nearest_ind[:,1:]
    
    return KNN_k_ind


def KNN_nd(X,num=1):
    '''   
    功能
    ===========
    从一个点集每个点相邻最近的几个到该点距离。

    原理
    ===========
    应用sklearn库中的neighbors模块的KDTree。
        
    参数
    ===========
    X : 点集坐标矩阵,numpy格式;
    k : 近邻点数目,缺省时默认只取最近的一个点。   
           
    信息
    ===========
    作者    :  陈先龙
    邮箱    :  yinzhe@uni-space.net.cn
    版本    :  0.9
    创建日期:  2020-11-24

    示例
    ===========
    >>> np.random.seed(0)
    >>> X = np.random.random((10, 2))
    >>> X
    [[0.5488135  0.71518937]
     [0.60276338 0.54488318]
     [0.4236548  0.64589411]
     [0.43758721 0.891773  ]
     [0.96366276 0.38344152]
     [0.79172504 0.52889492]
     [0.56804456 0.92559664]
     [0.07103606 0.0871293 ]
     [0.0202184  0.83261985]
     [0.77815675 0.87001215]]
    >>> KNN_nd(X,3)
    array([[0.14306129, 0.1786471 , 0.20869372],
           [0.1786471 , 0.18963685, 0.20562852],
           [0.14306129, 0.20562852, 0.2462733 ],
           [0.13477076, 0.20869372, 0.2462733 ],
           [0.2252094 , 0.39536284, 0.52073358],
           [0.18963685, 0.2252094 , 0.30612356],
           [0.13477076, 0.2112843 , 0.21734021],
           [0.66072543, 0.70162138, 0.74722058],
           [0.42153982, 0.44455307, 0.54148195],
           [0.21734021, 0.27670999, 0.34126404]])
    >>> KNN_nd(X)       
    array([[0.14306129],
           [0.1786471 ],
           [0.14306129],
           [0.13477076],
           [0.2252094 ],
           [0.18963685],
           [0.13477076],
           [0.66072543],
           [0.42153982],
           [0.21734021]])
    '''
    
    from sklearn.neighbors import KDTree

    tree = KDTree(X)
    n = num+1  # 算法认为最近的点是其本身
    nearest_dis, nearest_ind = tree.query(X, k=n) 

    KNN_k_dis = nearest_dis[:,1:]

    return KNN_k_dis


def Match1D(X,Y):
    '''
    功能
    ===========
    一维观测值样本匹配预处理，便于后续进行拟合回归
    等各种操作,注意返回的是匹配和排序后的自变量和
    观测值两个数组组成的元组。

    原理
    ===========
    自变量与观测值长度匹配及按照自变量顺序对应排序。
        
    参数
    ===========
    X : 点集坐标数组,numpy格式;
    Y : 观测值数组,numpy格式。   
           
    信息
    ===========
    作者    :  陈先龙
    邮箱    :  yinzhe@uni-space.net.cn
    版本    :  0.9
    创建日期:  2020-12-29

    示例
    ===========
    >>> x1 = np.array([2,4,5,3,9,7,6])
    >>> y1 = np.array([0.08820262,0.02000786,0.0489369,
                       0.11204466,0.0933779,-0.04886389,
                       0.04750442,-0.00756786,-0.00516094,
                       0.02052993])
    >>> Match1D(x1,y1)
   (array([2, 3, 4, 5, 6, 7, 9]),
    array([ 0.08820262,  0.11204466,  0.02000786,  0.0489369 ,  
            0.04750442, -0.04886389,  0.0933779 ]))
    ''' 

    # 样本长度匹配
    if len(X)!=len(Y):
        l = min((len(X),len(Y)))
        X = X[:l]
        Y = Y[:l]

    # 对样本和自变量对应排序
    order = np.argsort(X)
    X = X[order]
    Y = Y[order] 

    return X,Y


def Match2D(pts,val):
    '''
    功能
    ===========
    二维观测值样本匹配预处理，便于后续进行拟合回归
    等各种操作,注意返回的是匹配和排序后的自变量坐标
    和观测值两个数组组成的元组。

    原理
    ===========
    自变量与观测值长度匹配及按照自变量顺序先X坐标再
    Y坐标对应排序。
        
    参数
    ===========
    pts : 点集坐标二维数组,numpy格式;
    val : 观测值数组,numpy格式。   
           
    信息
    ===========
    作者    :  陈先龙
    邮箱    :  yinzhe@uni-space.net.cn
    版本    :  0.9
    创建日期:  2020-12-30

    示例
    ===========
    >>> pts = np.array([[3,4],[1,2],[5,6],[1,1],[2,2],[1,5],[1,4],[2,3],[3,3]])
    >>> val = np.array([0.08820262,0.02000786,0.0489369,
                       0.11204466,0.0933779,-0.04886389,
                       0.04750442,-0.00756786,-0.00516094,
                       0.02052993])
    >>> Match2D(pts,val)
    (array([[1, 1],
            [1, 2],
            [1, 4],
            [1, 5],
            [2, 2],
            [2, 3],
            [3, 3],
            [3, 4],
            [5, 6]]),
    array([ 0.11204466,  0.02000786,  0.04750442, -0.04886389,  0.0933779 ,
           -0.00756786, -0.00516094,  0.08820262,  0.0489369 ]))
    ''' 

    # 样本长度匹配
    if len(pts)!=len(val):
        l = min((len(pts),len(val)))
        pts = pts[:l]
        val = val[:l]

    # 二维坐标排序，先x坐标再y坐标。
    idx = np.lexsort((pts[:,1],pts[:,0]))

    # 对样本和自变量对应排序
    pts = pts[idx]
    val = val[idx] 

    return pts,val


def chabiao(jd, Y, x):
    '''
    功能
    ===========
    查表求出对应的值。
  
    原理
    ===========
    相邻的两个节点值进行线性插值，
    越界超过节点时，按照线性进行进行外推，
    需要用到数据分析pandas库和标准库bisect。
      
    参数
    ===========
    jd : 节点值序列，应是已经正向排序且无重复的序列；
    Y  : 待求函数值序列；   
    x  : 待求的自变量值。
  
    信息
    ==========
    作者    :  陈先龙
    邮箱    :  chenxianlong@cgnpc.com.cn
    版本    :  0.9
    创建日期:  2017-08-01
    
    示例
    ==========
    >>> X = np.array([ 100,  200,  300,  400,  500,  600,  700,  800,  900, 1000])
    >>> Y = np.array([13.5, 15.4, 17.3, 19.1, 21.0, 22.9, 24.8, 26.6, 28.5, 30.1])
    >>> chabiao(X, Y, 250)
    16.350000000000001
    >>> chabiao(X, Y, 750)
    25.699999999999999
    >>> chabiao(X, Y, 50)
    12.550000000000001
    >>> chabiao(X, Y, 1250)
    34.100000000000009
    '''

    place = bisect.bisect_right(jd,x)
    
    # 当待查的值为节点值时，直接给出对应的函数值。
    if x==jd[place-1]: 
        value = Y[place-1]

    elif place==0:              # 小于最小节点,外推
        xielv = (Y[1]-Y[0]) / (jd[1]-jd[0])
        value = Y[0] - xielv*(jd[0]-x)
    
    elif place==len(jd):          # 大于最小节点,外推
        xielv = (Y[-1]-Y[-2]) / (jd[-1]-jd[-2])
        value = Y[-1] + xielv*(x-jd[-1])

    else: 
        xielv = (Y[place]-Y[place-1]) / (jd[place]-jd[place-1])
        value = Y[place-1] + xielv*(x-jd[place-1])

    return value


def inter2(a, b):
    ''' 
    功能
    ===========
    二维样条插值范例。
    
    原理
    ===========
    x内层，y外层，
    即插值基准列表中的第一个元素为y取第一个y基准值时的
    各个x基准状况下的函数值列表。
        
    参数
    ===========
    a :  要求值点得x坐标；
    b ： 要求值点的y坐标。
    
    信息
    ==========
    作者  :  陈先龙
    邮箱  :  chenxianlong@cgnpc.com.cn
    版本  :  0.9
    创建日期 :  2014-04-2
        
    示例
    ==========
    >>> inter2(1.5, 2.5)
    7.4999999999999964
    >>> inter2(2.5, 1.5)
    6.4999999999999956
    '''
        
    xData = [1, 2, 3, 4]
    yData = [1, 2, 3, 4]
    value = [[4, 5, 6, 7], [6, 7, 8, 9], [8, 9, 10, 11], [10, 11, 12, 13]]
    
    inter = interpolate.interp2d(xData, yData, value, kind='cubic')
    yy = inter(a, b) + [0]
    return yy[0]


def jiange(t1, t2, value1, value2):
    ''' 
    功能
    ===========
    由起始时刻的物理量值和终止时刻的物理量值给出
    每个时间间隔内的物理量均值序列。
                
    原理
    ===========
    应用numpy库进行分区。
          
    参数
    ===========
    t1    : 起始时刻,注意是整数;
    t2    : 终止时刻,注意是整数,且必须大于起始时刻;
    value1: 起始时刻物理量值;
    value2: 终止时刻物理量值。
     
    信息
    ===========
    作者    :  陈先龙
    邮箱    :  chenxianlong@cgnpc.com.cn
    版本    :  0.9
    创建日期:  2017-11-18
          
    示例
    ===========
    >>> jiange(0,10,100,120)
    array([ 101.,  103.,  105.,  107.,  109.,  111.,  113.,  115.,  117.,  119.])
    >>> jiange(0,5,35,20)
    array([ 33.5,  30.5,  27.5,  24.5,  21.5])
    '''
          
    dt = t2 - t1
    n  = dt*2 + 1
    vl = np.linspace(value1, value2, n)   
    
    vlist = vl[1::2]
         
    return vlist


def extrefilter(signal):
    '''
    功能
    ===========
    筛选出一组离散数据信号中的波峰和波谷。
  
    原理
    ===========
    
      
    参数
    ===========
    signal : 信号序列。
  
    信息
    ==========
    作者    :  陈先龙
    邮箱    :  chenxianlong@cgnpc.com.cn
    版本    :  0.9
    创建日期:  2017-02-08
    
    示例
    ==========
    >>> list1 = 
    np.array([ 700. ,  636.4,  678.5,  729.8,  694.3,  632.9,  767.7,  677.8,
        768.1,  640.9,  740.4,  721.8,  779.7,  668.3,  625. ,  604.4,
        709.1,  670.3,  616.8,  719.5,  750.2,  734.4,  652.2,  770.1,
        708.7,  708.7,  764.9,  656.1,  790.3,  634.4,  678.4,  749.8,
        734.8,  791.8,  736.3,  641.3,  744.3,  717.1,  778.5,  635.2])
    >>> extrefilter(list1)
    array([ 700. ,  636.4,  729.8,  632.9,  767.7,  677.8,  768.1,  640.9,
        740.4,  721.8,  779.7,  604.4,  709.1,  616.8,  750.2,  652.2,
        770.1,  708.7,  764.9,  656.1,  790.3,  634.4,  749.8,  734.8,
        791.8,  641.3,  744.3,  717.1,  778.5,  635.2])
    '''

    a = np.array([signal[0]])
    for i,v in enumerate(signal[1:]):
        a = np.append(a,[signal[i],signal[i+1]])
        if a[-2]>a[-3] and a[-2]>a[-1]:
            a = np.delete(a, -1)
        elif a[-2]<a[-3] and a[-2]<a[-1]:
            a = np.delete(a, -1)    
            
        else:
            a = np.delete(a, np.s_[-2:])

    a = np.append(a, signal[-1])

    return a


def integral(X, Y, x_start, x_end):
    '''
    功能
    ===========
    根据已知的自变量和函数值的对应表，求在自变量范围内给定
    起点的积分值。
  
    原理
    ===========
    采用辛普森积分法则计算，
    需要用到scipy.integrate库和标准库bisect。
      
    参数
    ===========
    X       : 已知的自变量系列，应是已经正向排序且无重复的序列；
    Y       : 待求函数值序列；   
    x_start : 待求积分的下限；
    X_end   : 待求积分的上限
    
    信息
    ==========
    作者    :  陈先龙
    邮箱    :  chenxianlong@cgnpc.com.cn
    版本    :  0.9
    创建日期:  2018-04-19
    
    示例
    ==========
    >>> X = [ 100,  200,  300,  400,  500,  600,  700,  800,  900, 1000]
    >>> Y = [13.5, 15.4, 17.3, 19.1, 21.0, 22.9, 24.8, 26.6, 28.5, 30.1]
    >>> integral(X, Y, 150, 350)
    3341.25
    >>> integral(X, Y, 100, 350)
    3968.125
    >>> integral(X, Y, 100, 300)
    3080.0
    '''

    # 备份自变量和函数值列表
    XX = X
    YY = Y
    
    # 起始点在自变量序列中的插入位置
    place_start = bisect.bisect_right(XX, x_start)
    place_end   = bisect.bisect_right(XX, x_end)
    
    # 上下限及对应的函数值加入进已知系列中。
    if x_start not in XX: 
        xielv = (YY[place_start]-YY[place_start-1]) / (XX[place_start]-XX[place_start-1])
        y_start = YY[place_start-1] + xielv*(x_start-XX[place_start-1])
        bisect.insort_right(X, x_start)
        YY.insert(place_start, y_start)

    if x_end not in XX: 
        xielv1 = (YY[place_end]-YY[place_end-1]) / (XX[place_end]-XX[place_end-1])
        y_end  = YY[place_end-1] + xielv1*(x_end-XX[place_end-1])
        bisect.insort_right(X, x_end)
        YY.insert(place_end, y_end)

    # 应用辛普森积分法则求积分
    istart = XX.index(x_start)
    iend   = XX.index(x_end)
    xi = XX[istart:iend+1]
    yi = YY[istart:iend+1]
    integral = sci.simps(yi, xi)
    
    return integral