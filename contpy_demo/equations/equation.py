import numpy as np


class Equation: # define as abstart class??
    def __init__(self, name, params, rhs=None, jac=None, cont_param_name=None):
        self.params = params
        self.cont_param_name = cont_param_name

        if rhs is not None:
            self.rhs = rhs
        if jac is not None:
            self.jac = jac

        # more stuff

    def set_param(self, pname, pval):
        # TODO check existance of pname
        self.params[pname] = pval

    def get_param(self, pname):
        return self.params[pname]

    def get_params(self, pnames):
        return [self.get_param(pname) for pname in pnames.split(' ')]

    def set_cont_param(self, pval):
        self.set_param(self.cont_param_name, pval)

    def rhs(self, X):
        pass

    def jac(self, X):
        pass

    def pre_palc_step(self, X0, ds, tangent):
        self.X0 = X0
        self.ds = ds
        self.tangent = tangent

    def rhs_palc(self, X):
        cont_param_val = X[-1]
        self.set_cont_param(cont_param_val)

        Y = self.rhs(X[:-1])
        s = np.dot(self.tangent, (X - self.X0)) - self.ds

        return np.append(Y, s)

    def F_cont_param(self, X):
        pass

    def jac_palc(self, X):
        cont_param_val = X[-1]
        Y = X[:-1]

        self.set_cont_param(cont_param_val)

        N = len(X) - 1
        jac = np.zeros((N+1, N+1))

        jac[:-1, :-1] = self.jac(Y)

        # PALC
        jac[:N, -1] = self.F_cont_param(Y)
        jac[-1, :] = self.tangent

        return jac
