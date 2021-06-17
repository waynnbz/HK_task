import numpy as np
from scipy.spatial import Delaunay

from helper import edges


def local_delaunay_tri( obs,randn_times=0,randn_std=1,tri_dimension = 2 ):
    '''
    This function is used to generate arc list of points by randomly delaunay
    network
    Input parameters:
    obs             : observation file
    randn_times     : repeat time of randomization
    randn_std       : std of normal distrition
    '''


    if len(randn_std) == 1 and randn_times > 0:
        randn_std = np.tile(randn_std, (randn_times, 1))
    NPS = obs.shape[0]
    Xcor = obs[:, 0].reshape(-1, 1)
    Ycor = obs[:, 1].reshape(-1, 1)
    Zcor = obs[:, 4].reshape(-1, 1) + 0.1*np.random.randn(NPS, 1) #TODO: check why column 4 used here

    if tri_dimension == 3:
        # options = 'Qt Qbb Qc Qx'
        DT = Delaunay(np.concatenate([Xcor, Ycor, Zcor], axis=1))#, qhull_options=options)
    else:
        # options = 'Qt Qbb Qc'
        DT = Delaunay(np.concatenate([Xcor, Ycor], axis=1))#, qhull_options=options)

    Arc = edges(DT.simplices)
    tri = DT.simplices

    if randn_times > 0:
        Arc_cell = []
        tri_cell = []
        for i in range(randn_times):
            print(f'{i} of {randn_times}, constuct delaunay network with STD of {randn_std[i]}')
            Xcor = obs[:, 0].reshape(-1, 1) + randn_std[i] * np.random.randn(NPS, 1)
            Ycor = obs[:, 1].reshape(-1, 1) + randn_std[i] * np.random.randn(NPS, 1)
            Zcor = obs[:, 2].reshape(-1, 1) + randn_std[i] * np.random.randn(NPS, 1)
            if tri_dimension==3:
                DT_temp=Delaunay(np.concatenate([Xcor, Ycor, Zcor], axis=1))#, qhull_options=options)
            else:
                DT_temp=Delaunay(np.concatenate([Xcor, Ycor], axis=1))#, qhull_options=options)
            Arc_cell.append(edges(DT_temp.simplices))
            tri_cell.append(DT_temp.simplices)
        Arc_cell = np.concatenate(Arc_cell)
        tri_cell = np.concatenate(tri_cell)
        Arc = np.vstack((Arc, Arc_cell))
        tri = np.vstack((tri, tri_cell))

    Arc = np.unique(np.sort(Arc, axis=1), axis=0)

    if tri.shape[1] == 3:
        Tri = np.unique(np.sort(tri, 1), axis=0)
    #TODO: confirm dims
    elif tri.shape[1] == 4:
        Tri1 = np.vstack(tri[:,0],tri[:,1],tri[:,2])
        Tri2 = np.vstack(tri[:,0],tri[:,1],tri[:,3])
        Tri3 = np.vstack(tri[:,0],tri[:,2],tri[:,3])
        Tri4 = np.vstack(tri[:,1],tri[:,2],tri[:,3])
        Tri = np.vstack(Tri1, Tri2, Tri3, Tri4)
        Tri = np.unique(np.sort(Tri, axis=1), axis=0)
    else:
        raise Exception('Triangle dimension larger will be considered in furture ...')

    IDX = {}
    IDX['to'] = Arc[:, 1]
    IDX['from'] = Arc[:, 0]

    return IDX, Tri
