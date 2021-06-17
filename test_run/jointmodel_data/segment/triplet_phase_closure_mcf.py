import numpy as np
from scipy.sparse import find
import cplex
from cplex.exceptions import CplexError


# def triplet_phase_closure_mcf(tri2ifg_matrix,ifg_ph_wrap,Input,correct_idx=0):

#################################################################################
# test input
import pickle

tri2ifg_matrix = pickle.load(open('test_data/tri2ifg_matrix.pickle', 'rb'))
ifg_ph_wrap = pickle.load(open('test_data/y1_temp.pickle', 'rb'))
import scipy.io as sio

Input = sio.loadmat('Input.mat')
correct_idx = 1
#################################################################################

NIFG, NARC = ifg_ph_wrap.shape
Nintv = Input['NSLC'].squeeze() - 1
Ntri = tri2ifg_matrix.shape[0]
B = Input['tmatrix']
# TODO: check the necessity for nan matrix
arc_interval_ph = np.zeros((Nintv, NARC))

# form triplet_closure for all arcs
triplet_closure_integer = np.round(tri2ifg_matrix @ ifg_ph_wrap / (2 * np.pi))  # banker rounding, diff from matlab
# detect arc with zero closure
triplet_closure_log = triplet_closure_integer == 0
unwrap_correct_log = np.sum(triplet_closure_log, 0) == Ntri
# derive intv phase for arc with zero closure, np.round diff
arc_interval_ph[:, unwrap_correct_log] = np.linalg.lstsq(B, ifg_ph_wrap[:, unwrap_correct_log], rcond=None)[0]
unwrap_correct_log_final = unwrap_correct_log

# TODO: complete correct_idx == 1 implementation
if correct_idx == 1:
    ## detect arc with non-zero closure
    unwrap_partial_idx=find(np.logical_not(unwrap_correct_log))[1]
    NARC_unwrap_partial=len(unwrap_partial_idx)

    ## find dataset for arc with non-zero closure
    ifg_ph_wrap_partial=ifg_ph_wrap[:,unwrap_partial_idx]

    ## correct the ifg_ph of arc with non-zero closure
    cost = np.ones(NIFG * 2)
    L = np.zeros(NIFG * 2)
    U = np.ones(NIFG * 2) * cplex.infinity
    beq = -np.round((tri2ifg_matrix @ ifg_ph_wrap_partial) / (2 * np.pi))
    Aeq = np.hstack((tri2ifg_matrix, -tri2ifg_matrix))
    intcon = 'I'*Aeq.shape[1]
    sense = 'E'*Aeq.shape[0]
    col_names = list(map(str, np.arange(Aeq.shape[1])))
    row_names = list(map(str, np.arange(Aeq.shape[0])))

    prob = cplex.Cplex()
    prob.objective.set_sense(prob.objective.sense.minimize)
    prob.variables.add(obj=cost, lb=L, ub=U, types=intcon,
                       names=col_names)
    rows = [[name, row] for name, row in zip(np.tile(col_names, (Aeq.shape[0], 1)), Aeq)]
    xpm = []
    for i in range(NARC_unwrap_partial):

        rhs = beq[:, i]
        prob.linear_constraints.add(lin_expr=rows, senses=sense, rhs = rhs, names=row_names)
        prob.set_warning_stream(None)
        prob.set_results_stream(None)
        prob.solve()
        xpm_temp = prob.solution.get_values()
        prob.linear_constraints.delete()
        if xpm_temp:
            xpm.append(np.array(xpm_temp).reshape(-1, 1))

    xpm = np.concatenate(xpm, axis=1)
    xpm = np.round(xpm)
    xp = xpm[:NIFG,:]
    xm = xpm[NIFG:,:]
    K = xp - xm
    ifg_ph_partial = K * (2 * np.pi) + ifg_ph_wrap_partial
    # form triplet_closure again
    triplet_closure_partial_integer = np.round(tri2ifg_matrix @ ifg_ph_partial / (2 * np.pi))
    # update intv phase for arc with zero cloosure
    triplet_closure_partial_correct_idx = triplet_closure_partial_integer == 0
    unwrap_correct_log_partial_1 = np.sum(triplet_closure_partial_correct_idx, axis=0) == Ntri
    unwrap_correct_idx_partial_2 = unwrap_partial_idx[unwrap_correct_log_partial_1]
    unwrap_correct_log_partial_3 = np.zeros((NARC), dtype=bool)     #false(1, NARC);
    unwrap_correct_log_partial_3[unwrap_correct_idx_partial_2] = True
    arc_interval_ph[:, unwrap_correct_idx_partial_2]= \
        np.linalg.lstsq(B, ifg_ph_partial[:, unwrap_correct_log_partial_1], rcond=None)[0] #B\ifg_ph_partial[:, unwrap_correct_log_partial_1]
    ## update IDX and intv phase
    unwrap_correct_log_final = np.logical_or(unwrap_correct_log, unwrap_correct_log_partial_3)

unwrap_correct_log_final = unwrap_correct_log_final.conj().T
arc_interval_ph = arc_interval_ph.T

if correct_idx ==1:
    print(f'Number of arcs:                                     {NARC}')
    print(f'Number of arcs recovered without ambiguity:         {sum(unwrap_correct_log)}')
    print(f'Number of arcs recovered with ambiguity:            {sum(unwrap_correct_log_partial_1)}')
    print(f'Number of arcs unrecovered with ambiguity:          {NARC_unwrap_partial-sum(unwrap_correct_log_partial_1)}')

# return arc_interval_ph, unwrap_correct_log_final
