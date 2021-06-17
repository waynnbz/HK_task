import numpy as np
from scipy.stats import rankdata

def BWStest(Xarray,Yarray,Alpha=.05):


    n, m = Xarray.shape
    ranks = rankdata(np.concatenate((Xarray, Yarray)), axis=1)
    xrank = np.sort(ranks[0:n, :])
    yrank = np.sort(ranks[n:, :])
    temp = np.arange(1., n+1)[:, np.newaxis] @ np.ones((1, m))
    tempx = (xrank - 2. * temp)**2
    tempy = (yrank - 2. * temp)**2
    temp = temp / (n + 1) * (1 - temp / (n + 1)) * 2 * n
    BX = 1 / n * sum(tempx / temp, 0)
    BY = 1 / n * sum(tempy / temp, 0)

    # test statistic
    B = 1 / 2 * (BX + BY)

    if Alpha == .05:
        if n == 5:
            b = 2.533
        elif n == 6:
            b = 2.552
        elif n == 7:
            b = 2.620
        elif n == 8:
            b = 2.564
        elif n == 9:
            b = 2.575
        elif n == 10:
            b = 2.583
        else:
            b = 2.493
    else:
        b = 3.880

    H = B >= b

    return H