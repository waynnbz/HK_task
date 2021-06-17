import numpy as np


def OutputMat(M, filename):

    if len(M.shape) == 3:
        M_t = np.transpose(M, (2, 1, 0))
    np.savetxt(filename, np.ravel(M_t, 'F'), delimiter=",")

    return