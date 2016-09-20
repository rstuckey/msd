#!/usr/bin/env python

import math
import sys

import numpy as np
import pymc as mc
import scipy as sp

pyublas_exists = True
try:
    from msd import MSD_BOOST
except ImportError:
    pyublas_exists = False

cython_exists = True
try:
    from msdc import MSD_CYTHON
except ImportError:
    cython_exists = False

from msd import MSD


if __name__ == '__main__':

    # Set some defaults
    if ('VERBOSE' not in locals()):
        VERBOSE = True
    if ('PLOT_MAP' not in locals()):
        PLOT_MAP = False
    if ('PLOT_MCMC' not in locals()):
        PLOT_MCMC = False
    if (PLOT_MAP or PLOT_MCMC):
        from plot import plot, updateplot, clearplot, addplot

    if ('MODEL' not in locals()):
        MODEL = 'python'
    if ((MODEL == 'boost') and (not pyublas_exists)):
        print "Warning: pyublas does not exist! Setting MODEL = 'python'"
        MODEL = 'python'
    if ((MODEL == 'cython') and (not cython_exists)):
        print "Warning: cython does not exist! Setting MODEL = 'python'"
        MODEL = 'python'

    if ('ZERO_SEED' not in locals()):
        ZERO_SEED = True
    if ZERO_SEED:
        np.random.seed(1)

    if ('BERNOULLI_MASK' not in locals()):
        BERNOULLI_MASK = False
    if ('TRANSFORM_PARAMS' not in locals()):
        TRANSFORM_PARAMS = False

    # True parameter list
    CT = [ msd.get_coeffs()[ck] for ck in c_idx ]

    if (MODEL == 'boost'):
        # Create the model (Boost extension)
        msd_best = MSD_BOOST("Mass-Spring-Damper_BAYES_EST (Boost)", N)
        msd_best.set_external_forces(T, E, 'linear_uniform')
    elif (MODEL == 'cython'):
        # Create the model (Boost extension)
        msd_best = MSD_CYTHON("Mass-Spring-Damper_BAYES_EST (Cython)")
        msd_best.set_external_forces(T, E, 'linear_uniform')
    else:
        # Create the model (pure Python)
        msd_best = MSD("Mass-Spring-Damper_BAYES_EST")
        msd_best.set_external_forces(T, E, 'linear')

    if ADD_NOISE:
        sdnu_0 = sdnu
        sdnu_L = [ 0.0, sdnu_0*2.0 ]
    else:
        sdx = np.fix((np.max(X, axis=0) - np.min(X, axis=0))*100.0 + 1.0)/100.0*0.01
        sdnu_0 = sdx[:,1]
        sdnu_L = [ 0.0, sdnu_0*2.0 ]

    # Intermediate parameter dict
    CD = { ck : None for ck in c_idx }

    # Stochastic variable dict
    P = { }

    if ('C0' not in locals()):
        C0 = [ 0.5*CT[i] for i in range(len(c_idx)) ]

    if ('CL' not in locals()):
        CL = [ [ 0, 0 ] for ck in c_idx ]
        for i in range(len(c_idx)):
            ck = c_idx[i]
            if (np.abs(C0[i]) < 0.1):
                cl = C0[i] - 1.0
                cu = C0[i] + 1.0
            else:
                cl = C0[i] - 2.0*np.abs(CT[i])
                cu = C0[i] + 2.0*np.abs(CT[i])
            CL[i] = [ cl, cu ]

    for i in range(len(c_idx)):
        ck = c_idx[i]
        P[ck] = mc.Uniform(ck, lower=CL[i][0], upper=CL[i][1], value=C0[i])

# ------------------------------------------------------------------------------

    class Meanfcn(object):

        def __init__(self, z0, T, G, kws, FF):
            self.z0 = z0
            self.T = T
            self.G = G
            self.kws = kws
            self.FF = FF
            self.fopt_max = None
            self.it = 0;
            self.CD = CD
            self.C0 = C0
            self.C = C0

        def __call__(self, **kwargs):
            for i in range(len(c_idx)):
                ck = c_idx[i]
                self.CD[ck] = self.C0[i]

            for ck in kwargs.keys():
                self.CD[ck] = kwargs[ck].item()
                i = c_idx.index(ck)
                self.C[i] = kwargs[ck].item()

            msd_best.set_coeffs(self.CD)

            # Compute the response
            Xe, Xedot, Fe = msd_best.integrate(z0, T)

            dF = F - Fe
            fopt_sum = np.sum(dF*dF)

            if (PLOT_MAP or PLOT_MCMC):
                if (self.it < np.size(FF, 0)):
                    self.FF[self.it, 0] = math.log(fopt_sum)
                else:
                    self.FF = np.roll(self.FF, -1)
                    self.FF[-1, 0] = math.log(fopt_sum)

                f_max = None
                if ((self.fopt_max is None) or (self.fopt_max < math.log(fopt_sum))):
                    f_max = math.log(fopt_sum) * 1.1
                    self.fopt_max = math.log(fopt_sum)
                    # rescale = True
                f_txt = '{:.4f}'.format(fopt_sum)

                updateplot(self.kws['fig'], self.kws['Axes'], self.kws['Lines'], self.kws['Text'], Xe, Fe, self.FF, f_max=f_max, f_txt=f_txt, c_txt=self.C)

            if VERBOSE:
                sys.stdout.write("[")
                for i in range(len(c_idx)):
                    sys.stdout.write(" %s: %.4f" % (c_idx[i], C[i]))
                sys.stdout.write(" ] fopt = %.6e\n" % fopt)
                sys.stdout.flush()

            self.it += 1

            return Xe[:,1]

# ------------------------------------------------------------------------------

    def eval_prec(**kwargs):
        std = kwargs.values()[0]
        return 1.0/std**2.0

    meanfcn = Meanfcn(z0, T, G, kws, FF)

    pk = { ck: P[ck] for ck in c_idx } # parents
    mean_nu = mc.Deterministic(meanfcn, "Deterministic mean function", 'mean', pk, dtype=np.float64, trace=True, cache_depth=2, plot=False, verbose=0)
    std_nu = mc.Uniform('std', lower=sdnu_L[0], upper=sdnu_L[1], value=sdnu_0)
    pk = { 'std': std_nu } # parents
    prec_nu = mc.Deterministic(eval_prec, "Deterministic precision", 'prec', pk, dtype=np.float64, trace=True, cache_depth=2, plot=False, verbose=0)
    obs_nu = mc.Normal('obs', mean_nu, prec_nu, value=Nu, observed=True)

    P['obs'] = obs_nu

    model = mc.Model(P)

    stoch_list = list(c_idx)
    stoch_list.append('std')
