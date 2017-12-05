#!/usr/bin/env python

import math
import sys
import time

import numpy as np
import numpy.matlib as ml
from scipy import optimize

pyublas_exists = True
try:
    from msd.msdu import MSD_PYUBLAS
except ImportError:
    pyublas_exists = False

cython_exists = True
try:
    from msd.msdc import MSD_CYTHON
except ImportError:
    cython_exists = False

boost_exists = True
try:
    from msd.msdb import MSD_BOOST
except ImportError:
    boost_exists = False

lmfit_exists = True
try:
    import lmfit as lm
except ImportError:
    lmfit_exists = False

from msd import MSD


if __name__ == '__main__':

    # Set some defaults
    if ('ADD_NOISE' not in locals()):
        ADD_NOISE = True
    if ('VERBOSE' not in locals()):
        VERBOSE = False
    if ('PLOT_ESTIM' not in locals()):
        PLOT_ESTIM = False
    if PLOT_ESTIM:
        from plot import plot, updateplot, clearplot, addplot

    if ('MODEL' not in locals()):
        MODEL = 'python'
    if ((MODEL == 'pyublas') and (not pyublas_exists)):
        print("Warning: pyublas does not exist! Setting MODEL = 'python'")
        MODEL = 'python'
    if ((MODEL == 'cython') and (not cython_exists)):
        print("Warning: cython does not exist! Setting MODEL = 'python'")
        MODEL = 'python'
    if ((MODEL == 'boost') and (not boost_exists)):
        print("Warning: boost does not exist! Setting MODEL = 'python'")
        MODEL = 'python'

    if ('OPTFUN' not in locals()):
        OPTFUN = 'optimize'
    if ((OPTFUN == 'lmfit') and (not lmfit_exists)):
        print("Warning: lmfit does not exist! Setting OPTFUN = 'optimize'")
        OPTFUN = 'optimize'
    if ('MAXFUN' not in locals()):
        MAXFUN = 100

    if ('msd' not in locals()): # sim has not been run
        sys.stdout.write("Running sim...")
        sys.stdout.flush()
        execfile("sim.py")
        sys.stdout.write("\n")

    FF = ml.repmat(None, 50, 1)

    if PLOT_ESTIM:
        if (('fig' not in locals()) or (fig is None)):
            fig, Axes, Lines, Text = plot(msd.name, T, E, Z, G, Xe=np.zeros(X.shape), Fe=np.zeros(F.shape), FF=FF)
            fig.canvas.draw()
    else:
        fig, Axes, Lines, Text = ( None, None, None, None )

    kws = { 'fig': fig, 'Axes': Axes, 'Lines': Lines, 'Text': Text }

    # Create the simulation model
    if (MODEL == 'python'):
        # Pure Python
        msd_fest = MSD("Mass-Spring-Damper_FMIN_EST")
        msd_fest.set_external_forces(T, E, 'linear_unifom')
    elif (MODEL == 'cython'):
        # Cython
        msd_fest = MSD_CYTHON("Mass-Spring-Damper_FMIN_EST (Cython)")
        msd_fest.set_external_forces(T, E, 'linear_uniform')
    elif (MODEL == 'pyublas'):
        # PyUblas extension
        msd_fest = MSD_PYUBLAS("Mass-Spring-Damper_FMIN_EST (PyUblas)", N)
        msd_fest.set_external_forces(T, E, 'linear_unifom')
    elif (MODEL == 'numba'):
        # Numba JIT
        msd_fest = MSD_NUMBA("Mass-Spring-Damper_FMIN_EST (Numba)", N)
        msd_fest.set_external_forces(T, E, 'linear_unifom')
    elif (MODEL == 'numba_jc'):
        # Numba JIT
        msd_fest = MSD_NUMBA_JC(N)
        msd_fest.set_external_forces(T, E, 1)
    elif (MODEL == 'boost'):
        # Boost extension
        msd_fest = MSD_BOOST("Mass-Spring-Damper_FMIN_EST (Boost)", N)
        msd_fest.set_external_forces(T, E, 'linear_uniform')

    c_idx = ['k', 'b', 'd']

    if (OPTFUN == 'optimize'):

        print("POWELL'S MINIMIZATION:")

        class Objfun(object):

            def __init__(self, z0, T, G, FF):
                self.z0 = z0
                self.T = T
                self.G = G
                self.FF = FF
                self.fopt_max = None
                self.it = 0;

            def __call__(self, C, fig, Axes, Lines, Text):
                msd_fest.set_coeffs({ 'k': C[0], 'b': C[1], 'd': C[2] })

                # Compute the response
                if (MODEL in ['python', 'cython', 'pyublas', 'numba', 'boost']):
                    Xe, Xedot, Fe = msd_fest.integrate(self.z0, self.T)
                elif (MODEL == 'numba_jc'):
                    Xe, Xedot, Fe = msd_integrate(msd_fest, self.z0, self.T)

                # For fmin, fmin_powell, fmin_bfgs, fmin_l_bfgs_b
                dF = G - Fe
                fopt_sum = np.sum(dF*dF)

                if PLOT_ESTIM:
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

                    updateplot(fig, Axes, Lines, Text, Xe, Fe, self.FF, f_max=f_max, f_txt=f_txt, c_txt=C)

                if VERBOSE:
                    sys.stdout.write("[")
                    for i in range(len(c_idx)):
                        sys.stdout.write(" {}: {:.4f}".format(c_idx[i], C[i]))
                    sys.stdout.write(" ] fopt = {:.6e}\n".format(fopt_sum))
                    sys.stdout.flush()

                self.it += 1

                return fopt_sum

                # For leastsq
                # fopt = np.ravel(F - Fe)
                # sys.stdout.write("[")
                # for i in range(len(c_idx)):
                #     sys.stdout.write(" %s: %.4f" % (c_idx[i], C[i]))
                # sys.stdout.write("] fopt = %.6e\n" % np.sum(fopt*fopt))
                # sys.stdout.flush()
                # return fopt

        tic = time.clock()

        objfun = Objfun(z0, T, G, FF)

        # Need to start with a nontrivial parameter set to avoid getting stuck in a local minima straight away...
        C = optimize.fmin_powell(objfun, C0, args=( fig, Axes, Lines, Text ), maxiter=100, maxfun=MAXFUN)

        toc = time.clock() - tic
        print("Time elapsed: {:f} seconds").format(toc)

        # print("Number of iterations: {:f}").format(iter)
        # print("Number of function evaluations: {:f}").format(funcalls)

        C_PM = C.tolist()

        print
        print("            TRUE      F_EST")
        for i in range(len(c_idx)):
            ck = c_idx[i]
            print("{:5s}: {:10.4f} {:10.4f}").format(ck, msd.get_coeffs()[ck], C_PM[i])

        msd_fest.set_coeffs({ 'k': C_PM[0], 'b': C_PM[1], 'd': C_PM[2] })

        # Compute the response
        Xe, Xedot, Fe = msd_fest.integrate(z0, T)

        if PLOT_ESTIM:
            addplot(fig, Axes, T, Xe, Fe, color='cornflowerblue')

    # ------------------------------------------------------------------------------

    if (OPTFUN == 'lmfit'):

        print("LEVENBERG-MARQUARDT OPTIMIZATION:")

        # Define objective function: returns the array to be minimized
        class Fcn2min(object):
            def __init__(self, z0, T, G, FF):
                self.z0 = z0
                self.T = T
                self.G = G
                self.FF = FF
                self.fopt_max = None
                self.it = 0;

            def __call__(self, P, **kws):
                C = [ P[c_idx[i]].value for i in range(len(c_idx)) ]
                msd_fest.set_coeffs({ 'k': C[0], 'b': C[1], 'd': C[2] })

                # Compute the response
                Xe, Xedot, Fe = msd_fest.integrate(self.z0, self.T)

                fopt = np.ravel(self.G - Fe)
                fopt_sum = np.sum(fopt*fopt)

                if PLOT_ESTIM:
                    if (self.it < np.size(FF, 0)):
                        self.FF[self.it, 0] = fopt_sum
                    else:
                        self.FF = np.roll(self.FF, -1)
                        self.FF[-1, 0] = fopt_sum

                    f_max = None
                    if ((self.fopt_max is None) or (self.fopt_max < fopt_sum)):
                        f_max = fopt_sum * 1.1
                        self.fopt_max = fopt_sum
                        # rescale = True
                    f_txt = '{:.4f}'.format(fopt_sum)

                    updateplot(kws['fig'], kws['Axes'], kws['Lines'], kws['Text'], Xe, Fe, self.FF, f_max=f_max, f_txt=f_txt, c_txt=C)

                # For leastsq
                # fopt = np.ravel(F - Fe)
                # fopt = np.ravel(G - Fe)
                if VERBOSE:
                    sys.stdout.write("[")
                    for i in range(len(c_idx)):
                        sys.stdout.write(" {}: {:.4f}".format(c_idx[i], C[i]))
                    sys.stdout.write(" ] fopt = {:.6e}\n".format(fopt_sum))
                    sys.stdout.flush()

                self.it += 1

                return fopt

        # Create a set of Parameters
        P = lm.Parameters()
        for i in range(len(c_idx)):
            ck = c_idx[i]
            P.add(ck, value=C0[i])

        tic = time.clock()

        fcn2min = Fcn2min(z0, T, G, FF)

        # Do fit, here with leastsq model
        res = lm.minimize(fcn2min, P, kws=kws, method='leastsq', epsfcn=0.1)

        toc = time.clock() - tic
        print("Time elapsed: {:f} seconds").format(toc)

        C_LM = [ res.params[c_idx[i]].value for i in range(len(c_idx)) ]

        print
        print("            TRUE      F_EST")
        for i in range(len(c_idx)):
            ck = c_idx[i]
            print("{:5s}: {:10.4f} {:10.4f}").format(ck, msd.get_coeffs()[ck], C_LM[i])

        msd_fest.set_coeffs({ 'k': C_LM[0], 'b': C_LM[1], 'd': C_LM[2] })

        # Compute the response
        Xe, Xedot, Fe = msd_fest.integrate(z0, T)

        if PLOT_ESTIM:
            clearplot(fig, Lines)
            addplot(fig, Axes, T, Xe, Fe, color='goldenrod')
