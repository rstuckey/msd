#!/usr/bin/env python

import os
import sys
import time

import numpy as np
import pymc as mc

# import matplotlib.pyplot as pp

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
        # VERBOSE = True
        VERBOSE = False
    VERBOSE_ORIG = VERBOSE
    if ('PLOT_MAP' not in locals()):
        PLOT_MAP = False
    if ('PLOT_MCMC' not in locals()):
        PLOT_MCMC = False
    if (PLOT_MAP or PLOT_MCMC):
        from plot import plot, updateplot, clearplot, addplot

    if ('DO_PYMC_MAP' not in locals()):
        #DO_PYMC_MAP = True
        DO_PYMC_MAP = False
    if ('DO_PYMC_MCMC' not in locals()):
        DO_PYMC_MCMC = False
        # DO_PYMC_MCMC = False
    if ('DO_PYMC_BMS' not in locals()):
        DO_PYMC_BMS = False
        # DO_PYMC_BMS = True

    if ('MODEL' not in locals()):
        MODEL = 'python'
    if ((MODEL == 'boost') and (not pyublas_exists)):
        print WARN, "Warning: pyublas does not exist! Setting MODEL = 'python'", ENDC
        MODEL = 'python'
    if ((MODEL == 'cython') and (not cython_exists)):
        print WARN, "Warning: cython does not exist! Setting MODEL = 'python'", ENDC
        MODEL = 'python'

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
        print WARN, "Warning: Random seed will be automatically set.", ENDC

    if ('PYMC_DB' not in locals()):
        # PYMC_DB = 'ram'
        # PYMC_DB = 'pickle'
        PYMC_DB = 'hdf5'
    if ('CLOSE_PYMC_DB' not in locals()):
        CLOSE_PYMC_DB = True
    if ('ADAPTIVE_MCMC_STEP' not in locals()):
        ADAPTIVE_MCMC_STEP = True

    # if (PLOT_BMS and (not pp.isinteractive())):
    #     pp.ion()

    if ('msd' not in locals()): # sim has not been run
        print HEAD, "SIMULATING", ENDC
        # sys.stdout.write("Running sim...")
        # sys.stdout.flush()
        execfile("sim.py")
        # sys.stdout.write("\n")

    print HEAD, "BUILDING PROBABILITY DISTRIBUTION MODEL", ENDC

    if ('c_idx' not in locals()):
        c_idx = [ 'k', 'b', 'd' ]

    FF = ml.repmat(None, 50, 1)

    print

    if PLOT_MAP or PLOT_MCMC:
        if (('fig' not in locals()) or (fig is None)):
            fig, Axes, Lines, Text = plot(msd.name, T, E, Z, G, Xe=np.zeros(X.shape), Fe=np.zeros(F.shape), FF=FF)
            # pp.show()
            fig.canvas.draw()
    else:
        fig, Axes, Lines, Text = ( None, None, None, None )

    kws = { 'fig': fig, 'Axes': Axes, 'Lines': Lines, 'Text': Text }

    execfile("bms_model.py")

    if DO_PYMC_MAP:

        print HEAD, "BAYES ESTIMATE (MAXIMUM A-PRIORI):", ENDC
        if VERBOSE_ORIG:
            print WARN, "Switching VERBOSE to False for this...", ENDC
            VERBOSE = False
        # if PLOT_MAP:
        #     print WARN, "Switch PLOT_MAP to False for this...", ENDC
        #     PLOT_MAP = False
        PLOT = PLOT_MAP

        from scipy.optimize import fmin_ncg, fmin, fmin_powell, fmin_cg, fmin_bfgs, fmin_ncg, fmin_l_bfgs_b
        from scipy.misc import derivative

        # Compute MAP prior to MCMC, so as to start with good initial values
        mmap = mc.MAP(model)
        #mmap = MyMAP(model, direc_list=stoch_list)

        tic = time.time()

        # mmap.fit(method='fmin')
        mmap.fit(method='fmin_powell', iterlim=1000, tol=.0001, verbose=0)
        # mmap.fit(method='fmin_powell', iterlim=10, tol=.01, verbose=0)
        mmap.revert_to_max()

        print

        toc = time.time() - tic
        print time.strftime("Time elapsed: %Hh %Mm %Ss", time.gmtime(toc))

        print

        # mmap_vars_value = { ck : mmap.__dict__[ck].value.item() for ck in c_idx }
        mmap_vars_value = { ck : mmap.get_node(ck).value.item() for ck in c_idx }

        C = [ None for i in range(len(c_idx)) ]
        for i in range(len(c_idx)):
            ck = c_idx[i]
            C[i] = mmap_vars_value[ck]
            # C1[i] = P[ck].value.item()

        # writeout(c_idx1, CT1, C1, coeff_scales=coeff_scales, C0j=C01, title="MAXIMUM A-PRIORI", outpath=outpath)

        print "            TRUE      M_EST"
        for i in range(len(c_idx)):
            print "%5s: %10.4f %10.4f" % (c_idx[i], msd.get_coeffs()[c_idx[i]], C[i])

        for i in range(len(c_idx)):
            ck = c_idx[i]
            msd_best.set_coeffs({ ck : C[i] })

        # Compute the response
        Xe, Xedot, Fe = msd_best.integrate(z0, T)

        if VERBOSE_ORIG:
            VERBOSE = True

    # pd_stZ = mc.Uniform('std', lower=0.0, upper=sdz[1]*2.0, value=sdz[1])

    # @mc.deterministic
    # def prec(S=pd_stZ):
    #     return 1.0/S**2.0

    # P = { }
    # for i in range(len(c_idx)):
    #     ck = c_idx[i]
    #     C = msd.get_coeffs()
    #     P[ck] = mc.Uniform(ck, lower=min(0.0, C[ck]*2.0), upper=max(0.0, C[ck]*2.0), value=C[ck])
    #     msd.get_coeffs()

    # meanfcn = Meanfcn(z0, T, G, kws, FF)

    # obs = mc.Normal('obs', meanfcn.mean, prec, value=Z[:,1], observed=True)

    # model = mc.Model({ 'obs':obs, 'k':P['k'], 'b':P['b'], 'd':P['d'], 'prec':prec })

    # stoch_list = list(c_idx)
    # stoch_list.append('std')

    if DO_PYMC_MCMC:

        print HEAD, "BAYES ESTIMATE (MARKOV CHAIN MONTE CARLO):", ENDC

        if VERBOSE:
            print WARN, "Switching VERBOSE to False for this...", ENDC
            VERBOSE = False
        if PLOT_MCMC:
            print WARN, "Switch PLOT_MCMC to False for this...", ENDC
            PLOT_MCMC = False

        PLOT = PLOT_MCMC

        # if ((MODEL == 'boost') or (MODEL == 'cython')):
        #     Nc = [ 50000, 50000 ]
        # else:
        #     Nc = [ 50, 50 ]

        Nccs = [ 0 ] + Nc
        for k in range(1, len(Nccs)):
            Nccs[k] += Nccs[k - 1]
        mcmc_trace_idx = [ range(Nccs[k], Nccs[k + 1]) for k in range(len(Nc)) ]

        mcmc_stats_alpha = 0.05
        mcmc_stats_conf = int((1.0 - mcmc_stats_alpha)*100.0)
        mcmc_stats_hpd_key = '%d%% HPD interval' % mcmc_stats_conf
        mcmc_stats_batches = 100
        mcmc_stats_quantiles = ( 2.5, 25, 50, 75, 97.5 )

        # if PYMC_DB == 'ram':
        mcmc = mc.MCMC(model)
        # elif ((PYMC_DB == 'pickle') or (PYMC_DB == 'sqlite')):
        #     mcmc = mc.MCMC(model, db=PYMC_DB, dbname=dbpath)
        # elif PYMC_DB == 'hdf5':
        #     mcmc = mc.MCMC(model, db=PYMC_DB, dbname=dbpath, dbmode='w', dbcomplevel=5, dbcomplib='bzip2')

        # C_sca1 = dict(zip(c_idx, [ 1.0/(ct*ct) for ct in CT ]))

        if ADAPTIVE_MCMC_STEP:
            mcmc.use_step_method(mc.AdaptiveMetropolis, [ P[ck] for ck in c_idx ])

        tic = time.time()

        # mcmc.sample(Nsb[0], Nsb[1], progress_bar=True)
        for k in range(len(Nc)):
            print
            print OKBL, "Chain %d:" % k, ENDC
            mcmc.sample(Nc[k], burn=0, thin=1, tune_interval=1000, save_interval=None, progress_bar=True)
            # mcmc.sample(Nc[0])
            # mcmc.save_state()

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

        print

        # if PYMC_DB != 'ram':
        #     mcmc.db.commit()

        toc = time.time() - tic
        print time.strftime("Time elapsed: %Hh %Mm %Ss", time.gmtime(toc))

        A_mcmc = np.vstack(( mcmc_trace[-1][ck] for ck in c_idx ))
        Cove_mcmc = np.cov(A_mcmc)
        if np.all(np.diag(Cove_mcmc) > 0.0):
            Vare_mcmc = np.sqrt(np.diag(Cove_mcmc))
            Corre_mcmc = ((Cove_mcmc.T/Vare_mcmc).T)/Vare_mcmc
            # from corrplot import plotccorr
            # plotccorr(remus_best.name, c_idx1, Corre_mcmc)

        C = [ None for i in range(len(c_idx)) ]
        Ce = [ None for i in range(len(c_idx)) ]

        for i in range(len(c_idx)):
            ck = c_idx[i]
            C[i] = np.mean(mcmc_trace[-1][ck])
            Ce[i] = np.sqrt(np.cov(mcmc_trace[-1][ck]))

        DIC = mcmc.DIC
        BPIC = calc_bpic(mcmc)
        # From mcmc._calc_dic:
        mean_deviance = np.mean(mcmc_deviance[-1])
        PD = mcmc.DIC - mean_deviance

        # writeout(c_idx1, CT1, C1, coeff_scales=coeff_scales, Cej=Ce1, C0j=C01, DIC=DIC, BPIC=BPIC, PD=PD, title="MARKOV CHAIN MONTE CARLO", outpath=outpath)

        print "            TRUE      B_EST"
        for i in range(len(c_idx)):
            ck = c_idx[i]
            print "%5s: %10.4f %10.4f" % (ck, msd.C[ck], C[i])

        for i in range(len(c_idx)):
            ck = c_idx[i]
            msd_best.set_coeffs({ ck : C[i] })

        # if ((PYMC_DB != 'ram') and CLOSE_PYMC_DB):
        #     mcmc.db.close()

        if VERBOSE_ORIG:
            VERBOSE = True

        # Function to interpolate over external force input at integration time
        # e_func = interpolate.interp1d(T, E, kind='linear', axis=0, bounds_error=False)

        # Compute the response
        # if ((MODEL == 'boost') or (MODEL == 'cython')):
        #     Xe, Xedot, Fe = msd_best.integrate(x0, t0, dt)
        # else:
        Xe, Xedot, Fe = msd_best.integrate(z0, T)

        # if PLOT:
        #     addplot(Axes, jr_idx, T, Xe, Fe, color='indianred')
        #     clearplot(figopt, Lines)

        # Compute the response
        # Xe, Xedot, Fe = msd_best.integrate(z0, T, e_func)
        # Xe, Xedot, Fe = msd_best.integrate(z0, T)

        # if PLOT_BMS:
        #     addplot(Axes, T, Xe, Fe, color='indianred')

    # if DO_PYMC_BMS:

    #     c_idx_ext = ['k', 'b', 'd', 'z']

    #     pd_stF_ext = mc.Uniform('std', lower=0.0, upper=sdz[1]*2.0, value=sdz[1])

    #     @mc.deterministic
    #     def prec_ext(S=pd_stF_ext):
    #         return 1.0/S**2.0

    #     P_ext = { }
    #     for i in range(len(c_idx)):
    #         ck = c_idx[i]
    #         P_ext[ck] = mc.Uniform(ck, lower=min(0.0, msd.C[ck]*2.0), upper=max(0.0, msd.C[ck]*2.0), value=msd.C[ck])

    #     P_ext['z'] = mc.Uniform('z', lower=0.0, upper=1.0)

    #     @mc.deterministic
    #     # def mean(Z=Z, Zdot=Zdot, k=pd_k, b=pd_b, d=pd_d):
    #     def mean_ext(k=P_ext['k'], b=P_ext['b'], d=P_ext['d'], z=P_ext['z']):
    #     # def mean(**P):
    #         # Fe = k*Z[:,0] + b*Zdot[:,0] + d*E[:,0]

    #         C = [ k.item(), b.item(), d.item(), z.item() ]

    #         # sys.stdout.write(".")
    #         # sys.stdout.flush()

    #         # for i in range(len(c_idx)):
    #         #     msd_best.C[c_idx[i]] = C[i]
    #         msd_best.set_coeffs({ 'k': C[0], 'b': C[1], 'd': C[2], 'z': C[3] })

    #         # Compute the response
    #         # Xe, Xedot, Fe = msd_best.integrate(z0, T, e_func)
    #         Xe, Xedot, Fe = msd_best.integrate(z0, T)

    #         if PLOT_BMS:
    #             updateplot(Lines, Xe, Fe)

    #         dF = F - Fe
    #         fopt = np.sum(dF*dF)
    #         if VERBOSE:
    #             sys.stdout.write("[")
    #             for i in range(len(c_idx_ext)):
    #                 sys.stdout.write(" %s: %.4f" % (c_idx_ext[i], C[i]))
    #             sys.stdout.write(" ] fopt = %.6e\n" % fopt)
    #             sys.stdout.flush()

    #         #return Fe
    #         return Xe[:,1]

    #     #obs_ext = mc.Normal('obs', mean_ext, prec_ext, value=G, observed=True)
    #     obs_ext = mc.Normal('obs', mean_ext, prec_ext, value=Z[:,1], observed=True)

    #     model_ext = mc.Model({ 'obs':obs_ext, 'k':P_ext['k'], 'b':P_ext['b'], 'd':P_ext['d'], 'z':P_ext['z'], 'prec':prec_ext })

    #     stoch_list_ext = list(c_idx_ext)
    #     stoch_list_ext.append('std')

    #     mcmc_ext = mc.MCMC(model_ext)

    #     for k in range(len(Nc)):
    #         print
    #         print "Chain %d:" % k
    #         mcmc_ext.sample(Nc[k], burn=0, thin=1, tune_interval=1000, save_interval=None, progress_bar=True)

    #     mcmc_ext_stats = [ mcmc_ext.stats(chain=k) for k in range(len(Nc)) ]
    #     mcmc_ext_trace = [ { ck : mcmc_ext.trace(ck, chain=k)[:].ravel() for ck in stoch_list_ext } for k in range(len(Nc)) ]

    #     mcmc_ext_trace_all = { ck : np.hstack((mcmc_ext_trace[k][ck] for k in range(len(Nc)))) for ck in stoch_list_ext }

    #     C_ext = [ None for i in range(len(c_idx_ext)) ]

    #     for i in range(len(c_idx_ext)):
    #         ck = c_idx_ext[i]
    #         C_ext[i] = mcmc_ext_stats[-1][ck]['mean']

    #     print "            TRUE      B_EST (EXT)"
    #     for i in range(len(c_idx_ext)):
    #         ck = c_idx_ext[i]
    #         print "%5s: %10.4f %10.4f" % (ck, msd.C[ck], C_ext[i])

    #     msd_best.set_coeffs({ 'k': C_ext[0], 'b': C_ext[1], 'd': C_ext[2], 'z': C_ext[3] })

    #     Xe_temp = Xe.copy()
    #     Xedot_temp = Xedot.copy()
    #     Fe_temp = Fe.copy()

    #     # Compute the response
    #     # Xe, Xedot, Fe = msd_best.integrate(z0, T, e_func)
    #     Xe_ext, Xedot_ext, Fe_ext = msd_best.integrate(z0, T)

    #     #p, ll, lp = weight(models=[ model, model_ext ], iter=Nc[-1], priors={ model: 0.9, model_ext: 0.1 })
    #     #p, ll, lp = weight(models=[ model, model_ext ], iter=Nc[-1])

    #     print
    #     print "DIC:     MSD        MSD (EXT)"
    #     print "       %10.4f %10.4f" % (mcmc.dic, mcmc_ext.dic)

    #     def calc_bpic(mcmc):
    #         """Calculates Bayesian Predictive Information Criterion"""

    #         # Find mean deviance
    #         mean_deviance = np.mean(mcmc.db.trace('deviance')(), axis=0)

    #         # Set values of all parameters to their mean
    #         for stochastic in mcmc.stochastics:

    #             # Calculate mean of paramter
    #             try:
    #                 mean_value = np.mean(mcmc.db.trace(stochastic.__name__)(), axis=0)

    #                 # Set current value to mean
    #                 stochastic.value = mean_value

    #             except KeyError:
    #                 print "No trace available for %s. DIC value may not be valid." % stochastic.__name__

    #         # Return twice deviance minus deviance at means
    #         return 3*mean_deviance - 2*mcmc.deviance

    #     print "BPIC:    MSD        MSD (EXT)"
    #     print "       %10.4f %10.4f" % (calc_bpic(mcmc), calc_bpic(mcmc_ext))

    # if PLOT_BMS:
    #     clearplot(Lines)
