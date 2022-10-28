from .newton import newton

import numpy as np

def get_tangent(X, eq, prev_tangent=None, solver=np.linalg.solve):
    eq.set_cont_param(X[-1])
    Y = X[:-1]

    F_param = eq.F_cont_param(Y)
    F_x = eq.jac(Y)

    # TODO change to the default solver
    tx = np.linalg.solve(F_x, -F_param)
    t = np.append(tx, 1)

    if prev_tangent is not None:
        sign = np.dot(t, prev_tangent)
        if sign < 0:
            t = -t 
    
    return t / np.linalg.norm(t)

def swipePALC(X, ds, eq, direction='f', save_scalars_every=100, 
                save_state_every=10):
    pass