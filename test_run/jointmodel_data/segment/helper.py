import numpy as np

# called in local_delaunay_tri
def edges(simplices):
    '''
    接受矩阵simps，其每一行为三边形的顶点标签[v1, v2, v3]，返回所有独特的边（[v1, v4]]）
    '''
    if simplices.shape[1] == 3:
        edg = [[[x, y], [x, z], [y, z]] for x, y, z in np.sort(simplices)]
        edg = np.unique(np.concatenate(edg), axis=0)
    else:
        raise Exception('Wrong number of vertices in computing triangulation edges')

    return edg

# called in detect_temporal_triplet, detect_tri_closure
def intersect_mtlb(a, b):
    '''
    replicate of matlab intersect, returns matched values & indices
    '''
    a1, ia = np.unique(a, return_index=True)
    b1, ib = np.unique(b, return_index=True)
    aux = np.concatenate((a1, b1))
    aux.sort()
    c = aux[:-1][aux[1:] == aux[:-1]]

    return c, ia[np.isin(a1, c)], ib[np.isin(b1, c)]


# called in detect_tri_closure
def ismember_mtlb(arr1, arr2, mtd='ele'):
    '''
    判断数组元素是否为集数组成员，并返回其位置
    mtd:    - 'ele'元素级别比较
            - ‘row’整行做比较
    '''
    if mtd == 'ele':
        # try:
        #     idx = np.array([[list(arr2.ravel('F')).index(c) for c in r] for r in arr1])
        # except ValueError:
        #     raise Exception('ismember_mtlb match not find')
        idx = np.array([[list(arr2.ravel('F')).index(c) for c in r] for r in arr1])

    elif mtd == 'rows':
        idx = np.array([arr2.tolist().index(r.tolist()) for r in arr1])
    else:
        raise Exception('Wrong input dimension')

    return idx

# # called in detect tri_closure
# def ismember(a, b):
#     '''
#     解决上面自定义ismember_mtlb无法处理无匹配项的问题 但无法进行2D row比较
#     参考： https://stackoverflow.com/questions/15864082/python-equivalent-of-matlabs-ismember-function
#     '''
#     bind = {}
#     for i, elt in enumerate(b):
#         if elt.tolist() not in bind:
#             bind[elt.tolist()] = i
#     return [bind.get(itm, None) for itm in a]


# called in detect_tri_closure
def intersect_row_mtlb(arr1, arr2):
    '''
    return index of matched rows in arr2
    '''

    idx = [i for i, v in enumerate(arr1) if v.tolist() in arr2.tolist()]

    return idx


def lscov(A, B, V):
    '''
    Computing General Least Squares
    参考 https://ggf.readthedocs.io/en/stable/_modules/ggf/matlab_funcs.html
    '''
    W = np.sqrt(np.linalg.inv(V))
    Aw = np.dot(W, A)
    Bw = np.dot(B.T, W)

    # set rcond=1e-10 to prevent diverging odd indices in x
    x, residuals, rank, s = np.linalg.lstsq(Aw, Bw.T, rcond=1e-10)

    return x
