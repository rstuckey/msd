#!/usr/bin/env python

import math
import sys

import numpy as np
import pymc as mc
import scipy as sp

from pymc.Node import ZeroProbability

#HEAD = '\033[95m'
HEAD = '\033[1m'
OKBL = '\033[94m'
OKGR = '\033[92m'
WARN = '\033[93m'
FAIL = '\033[91m'
ENDC = '\033[0m'
#BOLD = '\033[1m'


def calc_bpic(mcmc):
    """
    Calculates Bayesian Predictive Information Criterion

    From: MCMC._calc_dic

    See also: Ando, T., 2011 - "Predictive Bayesian Model Selection"
    """
    # Find mean deviance
    mean_deviance = np.mean(mcmc.db.trace('deviance')(), axis=0)

    # Set values of all parameters to their mean
    for stochastic in mcmc.stochastics:

        # Calculate mean of paramter
        try:
            mean_value = np.mean(mcmc.db.trace(stochastic.__name__)(), axis=0)

            # Set current value to mean
            stochastic.value = mean_value

        except KeyError:
            print(FAIL, "No trace available for {:s}. DIC value may not be valid.".format(stochastic.__name__), ENDC)

    # Return twice deviance minus deviance at means
    return 3*mean_deviance - 2*mcmc.deviance

def writeout(c_idxj, CTj, Cj, coeff_scales=None, Cej=None, C0j=None, DIC=None, BPIC=None, PD=None, title="", outpath=""):
    """
    Write output results to stdout and file.
    """
    if (coeff_scales is None):
        coeff_scales = { ck : 1.0 for ck in c_idxj }

        #CTj = CTj.copy()
        #Cj = Cj.copy()
        #if (Cej is not None):
        #    Cej = Cej.copy()
        #if (C0j is not None):
        #    C0j[i] /= coeff_scales[ck]
        #for i in range(len(c_idxj)):
        #    ck = c_idxj[i]
        #    CTj[i] /= coeff_scales[ck]
        #    Cj[i] /= coeff_scales[ck]
        #    if (Cej is not None):
        #        Cej[i] /= coeff_scales[ck]
        #    if (C0j is not None):
        #        C0j[i] /= coeff_scales[ck]

    if outpath:
        out_file = open(outpath, 'a+')

    #title += "\n"
    sys.stdout.write(HEAD + title + ENDC + "\n")
    if outpath:
        out_file.write(title + "\n")

    header = "            TRUE      EST"
    if (Cej is not None):
        header+= "    STD     "
    if (C0j is not None):
        header+= "      INIT"
    header += "\n"
    sys.stdout.write(header)
    if outpath:
        out_file.write(header)

    ctj_minw = -2
    for i in range(len(c_idxj)):
        ck = c_idxj[i]
        if (abs(CTj[i]) > 0.0):
            ctj_w = int(math.log10(abs(CTj[i]/coeff_scales[ck])))
            if (ctj_w < ctj_minw):
                ctj_minw = ctj_w
    ctj_minw_s = "%d" % (abs(ctj_minw) + 1)

    if (Cej is not None):
        cej_maxw = 0
        for i in range(len(c_idxj)):
            ck = c_idxj[i]
            if (abs(Cej[i]) > 0.0):
                cej_w = int(math.log10(abs(Cej[i]/coeff_scales[ck]))) + 1 + 1 + 2 + 2
                if (Cej[i] < 0.0):
                    cej_w += 1
                if (cej_w > cej_maxw):
                    cej_maxw = cej_w
        cej_maxw_s = "%d" % cej_maxw

    for i in range(len(c_idxj)):
        ck = c_idxj[i]
        Cej_s = ""
        if (Cej is not None):
            cej_s = ("(%." + ctj_minw_s + "f)") % (Cej[i]/coeff_scales[ck])
            Cej_s = ("%" + cej_maxw_s + "s %3.0f%%") % (cej_s, np.abs(Cej[i]/Cj[i])*100.0)
        C0j_s = ""
        if (C0j is not None):
            C0j_s = ("%8." + ctj_minw_s + "f") % (C0j[i]/coeff_scales[ck])
        outline = (":%5s: %8." + ctj_minw_s + "f %8." + ctj_minw_s + "f %s %s\n") % (ck, CTj[i]/coeff_scales[ck], Cj[i]/coeff_scales[ck], Cej_s, C0j_s)
        sys.stdout.write(outline)
        if outpath:
            out_file.write(outline)

    if ((DIC is not None) and (BPIC is not None) and (PD is not None)):
        header = "\n        DIC        BPIC      PD\n"
        sys.stdout.write(header)
        if outpath:
            out_file.write(header)
        outline = " %11.1f %11.1f %6.1f\n" % (DIC, BPIC, PD)
        sys.stdout.write(outline)
        if outpath:
            out_file.write(outline)

    if outpath:
        out_file.close()


class MyMAP(mc.MAP):
    """
    :SeeAlso: mcmc.MAP
    """
    def __init__(self, input=None, direc_list=None, eps=.001, diff_order = 5, verbose=-1):

        mc.MAP.__init__(self, input=input, eps=eps, diff_order=diff_order, verbose=verbose)

        self.direc_list = direc_list
        self.ccount = 0

    def fit(self, method='fmin', iterlim=1000, tol=.0001, verbose=0):
        """
        N.fit(method='fmin', iterlim=1000, tol=.001):

        Causes the normal approximation object to fit itself.

        method: May be one of the following, from the scipy.optimize package:
            -fmin_l_bfgs_b
            -fmin_ncg
            -fmin_cg
            -fmin_powell
            -fmin
        """
        self.tol = tol
        self.method = method
        self.verbose = verbose

        # print self.stochastics

        p = np.zeros(self.len, dtype=float)
        # d = np.zeros(self.len, dtype=float)
        d = np.zeros((self.len, self.len), dtype=float)
        # d = np.ones((self.len, self.len), dtype=float)
        # i = 0
        for stochastic in self.stochastics:
            p[self._slices[stochastic]] = np.ravel(stochastic.value)
            if self.ccount == 0:
                print(stochastic.__name__,)
            if stochastic.__name__ in self.direc_list:
                i = self.direc_list.index(stochastic.__name__)
                j = self._slices[stochastic]
                d[i, j] = 1.0
            # i += 1
        if self.ccount == 0:
            print()
            # print self.stochastics
            print(d)
            print(p)

        def callback(p):
            pass

        if self.method == 'fmin_powell':
            (p, fopt, d, iter, funcalls, warnflag) = sp.optimize.fmin_powell(func=self.func,
                            x0=p,
                            callback=callback,
                            direc=d,
                            maxiter=iterlim,
                            ftol=tol,
                            disp=verbose,
                            full_output=True)

        else:
            raise ValueError('Method unknown.')

        if self.ccount == 0:
            print(d)

        self.ccount += 1

        self._set_stochastics(p)
        self._mu = p

        try:
            self.logp_at_max = self.logp
        except:
            raise RuntimeError('Posterior probability optimization converged to value with zero probability.')

        lnL = sum([x.logp for x in self.observed_stochastics]) # log-likelihood of observed stochastics
        self.AIC = 2. * (self.len - lnL) # 2k - 2 ln(L)
        try:
            self.BIC = self.len * np.log(self.data_len) - 2. * lnL # k ln(n) - 2 ln(L)
        except FloatingPointError:
            self.BIC = -np.Inf

        self.fitted = True

class MyNormApprox(mc.NormApprox):
    """
    :SeeAlso: mcmc.NormApprox
    """
    def __init__(self, input=None, db='ram', eps=.001, diff_order=5, **kwds):
        mc.MAP.__init__(self, input, eps, diff_order)
        mc.Sampler.__init__(self, input, db, reinit_model=False, **kwds)
        self.C = mc.NormApproxC(self)
        # mc.NormApprox.__init__(self, input, eps, diff_order)

    def i_logp(self, index):
        """
        Evaluates the log-probability of the Markov blanket of
        a stochastic owning a particular index.
        """
        all_relevant_stochastics = set()
        p,i = self.stochastic_indices[index]
        try:
            return p.logp + mc.utils.logp_of_set(p.extended_children)
        except ZeroProbability:
            return -1.0e12 # np.finfo(np.float64).min is too negative!

