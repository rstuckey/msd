#!/usr/bin/env python

import sys

import matplotlib.pyplot as pp
import numpy as np
from scipy import linalg, stats

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
    if ('PLOT_REG' not in locals()):
        # PLOT_REG = True
        PLOT_REG = False
    if PLOT_REG:
        from plot import plot, updateplot, clearplot, addplot

    if ('MODEL' not in locals()):
        MODEL = 'python'
    if ((MODEL == 'boost') and (not pyublas_exists)):
        print "Warning: pyublas does not exist! Setting MODEL = 'python'"
        MODEL = 'python'
    if ((MODEL == 'cython') and (not cython_exists)):
        print "Warning: cython does not exist! Setting MODEL = 'python'"
        MODEL = 'python'

    if ('msd' not in locals()): # sim has not been run
        sys.stdout.write("Running sim...")
        sys.stdout.flush()
        execfile("sim.py")
        sys.stdout.write("\n")

    FF = ml.repmat(None, 50, 1)

    if PLOT_REG:
        if (('fig' not in locals()) or (fig is None)):
            fig, Axes, Lines, Text = plot(msd.name, T, E, Z, G, Xe=np.zeros(X.shape), Fe=np.zeros(F.shape), FF=FF)
            # pp.show()
            fig.canvas.draw()
    else:
        fig, Axes, Lines, Text = ( None, None, None, None )

    kws = { 'fig': fig, 'Axes': Axes, 'Lines': Lines, 'Text': Text }

    print "LINEAR REGRESSION:"

    if (MODEL == 'boost'):
        # Create the model (Boost extension)
        msd_est = MSD_BOOST("Mass-Spring-Damper_FMIN_EST (Boost)", N)
        msd_est.set_external_forces(T, E, 'linear_uniform')
    elif (MODEL == 'cython'):
        # Create the model (Boost extension)
        msd_est = MSD_CYTHON("Mass-Spring-Damper_FMIN_EST (Cython)")
        msd_est.set_external_forces(T, E, 'linear_uniform')
    else:
        # Create the model (pure Python)
        msd_est = MSD("Mass-Spring-Damper_FMIN_EST")
        msd_est.set_external_forces(T, E, 'linear')

    c_idx = ['k', 'b', 'd']

    A = np.c_[Z[:,0],Zdot[:,0],E[:,0]]

    # Least squares solution
    ( C, resid, rank, sigma ) = linalg.lstsq(A, G)

    sigma2 = np.sum((G - np.dot(A, C))**2.0)/(N - len(c_idx)) # RMSE

    cov = sigma2*np.linalg.inv(np.dot(A.T, A)) # covariance matrix
    se = np.sqrt(np.diag(cov)) # standard error

    alpha = 0.05
    conf = 100.0*(1.0 - alpha) # confidence level

    sT = stats.distributions.t.ppf(1.0 - alpha/2.0, N - len(c_idx)) # student T multiplier
    CI = sT*se

    SS_tot = np.sum((G - np.mean(G))**2.0)
    SS_err = np.sum((np.dot(A, C) - G)**2)

    #  http://en.wikipedia.org/wiki/Coefficient_of_determination
    Rsq = 1.0 - SS_err/SS_tot

    print "            TRUE       EST    {:2.0f}% CONF".format(conf)
    for i in range(len(c_idx)):
        print "{:5s}: {:10.4f} {:10.4f} +/-{:-.4f}".format(c_idx[i], msd.C[c_idx[i]], C[i, 0], CI[i])

    print "R^2 = %.4f" % Rsq

    for i in range(len(c_idx)):
        msd_est.C[c_idx[i]] = C[i, 0]

    # Estimated force matrix
    H = np.dot(A, C)

    # Function to interpolate over external force input at integration time
    # e_func = interpolate.interp1d(T, E, kind='linear', axis=0, bounds_error=False)

    # Compute the response
    # Xe, Xedot, Fe = msd_est.integrate(z0, T, e_func)
    Xe, Xedot, Fe = msd_est.integrate(z0, T)

    if PLOT_REG:
        updateplot(fig, Axes, Lines, Text, Xe, Fe, FF, f_max=None, f_txt=None, c_txt=np.ravel(C))

    # if PLOT_REG:
    #     if ('Axes' not in locals()):
    #         Axes, Lines = msd_plot(msd.name, T, E, Z, G, Xe=np.zeros(X.shape), Fe=np.zeros(F.shape))
    #         pp.show()

    #     msd_addplot(Axes, T, Xe, Fe, color='yellowgreen')
