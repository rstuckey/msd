#!/usr/bin/env python

import os
import sys
import time

import numpy as np
import numpy.matlib as ml
import pymc as mc

# import matplotlib.pyplot as pp

pyublas_exists = True
try:
    from msd import MSD_BOOST
except ImportError:
    pyublas_exists = False

cython_exists = True
try:
    from msd.msdc import MSD_CYTHON
except ImportError:
    cython_exists = False

from msd import MSD

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from _modelposterior import weight
from _utils import calc_bpic, writeout, MyMAP, MyNormApprox


if __name__ == '__main__':

    #HEAD = '\033[95m'
    HEAD = '\033[1m'
    OKBL = '\033[94m'
    OKGR = '\033[92m'
    WARN = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    #BOLD = '\033[1m'

    # Set some defaults
    if ('ADD_NOISE' not in locals()):
        ADD_NOISE = True
    if ('VERBOSE' not in locals()):
        VERBOSE = False
    VERBOSE_ORIG = VERBOSE
    if ('PLOT_MAP' not in locals()):
        PLOT_MAP = False
    if ('PLOT_MCMC' not in locals()):
        PLOT_MCMC = False
    if (PLOT_MAP or PLOT_MCMC):
        from plot import plot, updateplot, clearplot, addplot

    if ('DO_PYMC_MAP' not in locals()):
        DO_PYMC_MAP = False
    if ('DO_PYMC_MCMC' not in locals()):
        DO_PYMC_MCMC = True

    if ('MODEL' not in locals()):
        MODEL = 'python'
    if ((MODEL == 'boost') and (not pyublas_exists)):
        print(WARN, "Warning: pyublas does not exist! Setting MODEL = 'python'", ENDC)
        MODEL = 'python'
    if ((MODEL == 'cython') and (not cython_exists)):
        print(WARN, "Warning: cython does not exist! Setting MODEL = 'python'", ENDC)
        MODEL = 'python'
    if ('ITER_LIM' not in locals()):
        ITER_LIM = 1000

    if ('NUM_SAMPLES' in locals()):
        Nc = NUM_SAMPLES
    else:
        if ((MODEL == 'boost') or (MODEL == 'cython')):
            Nc = [ 50000, 50000 ]
        else:
            Nc = [ 50, 50 ]

    if ('ZERO_SEED' not in locals()):
        ZERO_SEED = True
    if ZERO_SEED:
        np.random.seed(1)
    else:
        print(WARN, "Warning: Random seed will be automatically set.", ENDC)

    if ('PYMC_DB' not in locals()):
        PYMC_DB = 'ram' # [ 'ram, 'pickle', 'hdf5']
    if ('CLOSE_PYMC_DB' not in locals()):
        CLOSE_PYMC_DB = True
    if ('ADAPTIVE_MCMC_STEP' not in locals()):
        ADAPTIVE_MCMC_STEP = True

    if PYMC_DB != 'ram':
        if ('OUTPUT_ID' not in locals()):
            OUTPUT_ID = time.strftime("%y%m%d", time.localtime())
            print(WARN, "Warning: No OUTPUT_ID specified. Using '{:s}' instead.".format(OUTPUT_ID), ENDC)
        OUTPUT_DIR = os.path.join("..", "output", OUTPUT_ID)
        if (not os.path.exists(OUTPUT_DIR)):
            os.makedirs(OUTPUT_DIR)

    if ('msd' not in locals()): # sim has not been run
        print(HEAD, "SIMULATING", ENDC)
        with open('sim.py') as f:
            exec(f.read())

    print(HEAD, "BUILDING PROBABILITY DISTRIBUTION MODEL", ENDC)

    if ('c_idx' not in locals()):
        c_idx = [ 'k', 'b', 'd' ]

    FF = ml.repmat(None, 50, 1)

    print()

    if PLOT_MAP or PLOT_MCMC:
        if (('fig' not in locals()) or (fig is None)):
            fig, Axes, Lines, Text = plot(msd.name, T, E, Z, G, Xe=np.zeros(X.shape), Fe=np.zeros(F.shape), FF=FF)
            fig.canvas.draw()
    else:
        fig, Axes, Lines, Text = ( None, None, None, None )

    kws = { 'fig': fig, 'Axes': Axes, 'Lines': Lines, 'Text': Text }

    with open('bms_model.py') as f:
        exec(f.read())

    if DO_PYMC_MAP:

        print(HEAD, "BAYES ESTIMATE (MAXIMUM A-PRIORI):", ENDC)
        if VERBOSE_ORIG:
            print(WARN, "Switching VERBOSE to False for this...", ENDC)
            VERBOSE = False
        PLOT = PLOT_MAP

        from scipy.optimize import fmin_ncg, fmin, fmin_powell, fmin_cg, fmin_bfgs, fmin_ncg, fmin_l_bfgs_b
        from scipy.misc import derivative

        mmap = mc.MAP(model)

        tic = time.time()

        mmap.fit(method='fmin_powell', iterlim=ITER_LIM, tol=0.001, verbose=0)
        mmap.revert_to_max()

        print()

        toc = time.time() - tic
        print(time.strftime("Time elapsed: %Hh %Mm %Ss", time.gmtime(toc)))

        print()

        mmap_vars_value = { ck : mmap.get_node(ck).value.item() for ck in c_idx }

        C = [ None for i in range(len(c_idx)) ]
        for i in range(len(c_idx)):
            ck = c_idx[i]
            C[i] = mmap_vars_value[ck]

        print("            TRUE      M_EST")
        for i in range(len(c_idx)):
            print("%5s: %10.4f %10.4f".format(c_idx[i], msd.get_coeffs()[c_idx[i]], C[i]))

        for i in range(len(c_idx)):
            ck = c_idx[i]
            msd_best.set_coeffs({ ck : C[i] })

        # Compute the response
        Xe, Xedot, Fe = msd_best.integrate(z0, T)

        if VERBOSE_ORIG:
            VERBOSE = True

    if DO_PYMC_MCMC:

        print(HEAD, "BAYES ESTIMATE (MARKOV CHAIN MONTE CARLO):", ENDC)

        if VERBOSE:
            print(WARN, "Switching VERBOSE to False for this...", ENDC)
            VERBOSE = False
        if PLOT_MCMC:
            print(WARN, "Switch PLOT_MCMC to False for this...", ENDC)
            PLOT_MCMC = False

        PLOT = PLOT_MCMC

        Nccs = [ 0 ] + Nc
        for k in range(1, len(Nccs)):
            Nccs[k] += Nccs[k - 1]
        mcmc_trace_idx = [ range(Nccs[k], Nccs[k + 1]) for k in range(len(Nc)) ]

        mcmc_stats_alpha = 0.05
        mcmc_stats_conf = int((1.0 - mcmc_stats_alpha)*100.0)
        mcmc_stats_hpd_key = '%d%% HPD interval' % mcmc_stats_conf
        mcmc_stats_batches = 100
        mcmc_stats_quantiles = ( 2.5, 25, 50, 75, 97.5 )

        if PYMC_DB != 'ram':
            dbname = "msd-temp." + PYMC_DB
            dbpath = os.path.join(OUTPUT_DIR, dbname)

        if PYMC_DB == 'ram':
            mcmc = mc.MCMC(model)
        elif ((PYMC_DB == 'pickle') or (PYMC_DB == 'sqlite')):
            mcmc = mc.MCMC(model, db=PYMC_DB, dbname=dbpath)
        elif PYMC_DB == 'hdf5':
            mcmc = mc.MCMC(model, db=PYMC_DB, dbname=dbpath, dbmode='w', dbcomplevel=5, dbcomplib='bzip2')

        if ADAPTIVE_MCMC_STEP:
            mcmc.use_step_method(mc.AdaptiveMetropolis, [ P[ck] for ck in c_idx ])

        tic = time.time()

        for k in range(len(Nc)):
            print()
            print(OKBL, "Chain {:d}:".format(k), ENDC)
            mcmc.sample(Nc[k], burn=0, thin=1, tune_interval=1000, save_interval=None, progress_bar=True)

        mcmc_stats = [ mcmc.stats(chain=k) for k in range(len(Nc)) ]
        mcmc_trace = [ { ck : np.ravel(mcmc.trace(ck, chain=k)[:]) for ck in stoch_list } for k in range(len(Nc)) ]
        mcmc_deviance = [ mcmc.trace('deviance', chain=k)[:] for k in range(len(Nc)) ]

        # Update Nc if sampling halted prematurely
        Nc = [ len(mcmc_trace[k][stoch_list[0]]) for k in range(len(Nc)) ]
        Nccs = [ 0 ] + Nc
        for k in range(1, len(Nccs)):
            Nccs[k] += Nccs[k - 1]
        mcmc_trace_idx = [ range(Nccs[k], Nccs[k + 1]) for k in range(len(Nc)) ]

        mcmc_trace_all = { ck : np.hstack((mcmc_trace[k][ck] for k in range(len(Nc)))) for ck in stoch_list }

        print()

        toc = time.time() - tic
        print(time.strftime("Time elapsed: %Hh %Mm %Ss", time.gmtime(toc)))

        A_mcmc = np.vstack(( mcmc_trace[-1][ck] for ck in c_idx ))
        Cove_mcmc = np.cov(A_mcmc)
        if np.all(np.diag(Cove_mcmc) > 0.0):
            Vare_mcmc = np.sqrt(np.diag(Cove_mcmc))
            Corre_mcmc = ((Cove_mcmc.T/Vare_mcmc).T)/Vare_mcmc

        C_MC = [ None for i in range(len(c_idx)) ]
        Ce_MC = [ None for i in range(len(c_idx)) ]

        for i in range(len(c_idx)):
            ck = c_idx[i]
            C_MC[i] = np.mean(mcmc_trace[-1][ck])
            Ce_MC[i] = np.sqrt(np.cov(mcmc_trace[-1][ck]))

        DIC = mcmc.DIC
        BPIC = calc_bpic(mcmc)

        mean_deviance = np.mean(mcmc_deviance[-1])
        PD = mcmc.DIC - mean_deviance

        print("            TRUE      B_EST")
        for i in range(len(c_idx)):
            ck = c_idx[i]
            print("{:5s}: {:10.4f} {:10.4f}".format(ck, msd.get_coeffs()[ck], C_MC[i]))

        msd_best.set_coeffs({ 'k': C_MC[0], 'b': C_MC[1], 'd': C_MC[2] })

        if VERBOSE_ORIG:
            VERBOSE = True

        # Compute the response
        Xe, Xedot, Fe = msd_best.integrate(z0, T)
