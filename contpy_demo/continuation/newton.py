import numpy as np
from threadpoolctl import threadpool_limits

def newton(X, func, jac, max_steps=50, atol=1e-6, max_threads=1, 
            solver=np.linalg.solve, test_function=None):

    success = False
    test = 0

    for step in range(max_steps):

        if step == 0:
            Y = func(X)
            err = np.abs(Y).sum()

        J = jac(X)

        with threadpool_limits(limits=max_threads):
            dX = solver(J, Y)

        X -= dX

        Y = func(X)
        err = np.abs(Y).sum()

        if err < atol:
            success = True
            
            if test_function is not None:
                test = test_function(J)
            break

    return X, success, test
