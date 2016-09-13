#!/usr/bin/env python

import sys

import numpy as np
import numpy.matlib as ml

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

    #HEAD = '\033[95m'
    HEAD = '\033[1m'
    OKBL = '\033[94m'
    OKGR = '\033[92m'
    WARN = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    #BOLD = '\033[1m'

    # Set some defaults
    if ('NOISE_SD' not in locals()):
        NOISE_SD = 0.01
    if isinstance(NOISE_SD, (int, float)):
        NOISE_SD = [ NOISE_SD for _ in range(3) ]
    if ('STATE_NOISE_SD' not in locals()):
        STATE_NOISE_SD = 0.0
    # if isinstance(STATE_NOISE_SD, (int, float)):
    #     STATE_NOISE_SD = [ STATE_NOISE_SD for _ in range(2) ]

    # Set some defaults
    if ('VERBOSE' not in locals()):
        # VERBOSE = True
        VERBOSE = False
    if ('PLOT_SIM' not in locals()):
        # PLOT_SIM = True
        PLOT_SIM = False

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
    else:
        print "Warning: Random seed will be automatically set."

    # Initial system state and external force input
    x0 = np.zeros((2, ))
    d0 = 0.0

    # Create the predefined external force vectors

    if ('D' not in locals()): # no external force defined
        # Create the time vector
        dt = 0.01
        T = np.arange(0.0, 15.0, dt)
        N = T.shape[0]

        D = ml.repmat(d0, N, 1)

        T_S = [ t_s + t_ds for t_s in np.arange(0.0, 15.0, 3.0) for t_ds in [ 0.0, 1.0 ] ]
        T_S.append(T[N - 1])
        N_S = len(T_S)
        D_S = [ d0 + d_ds for t_s in np.arange(0.0, 15.0, 3.0) for d_ds in [ 1.0, 0.0 ] ]
        D_S.append(d0)

        # for t_s in np.arange(0.0, 15.0, 1.0):
        # for t_s in T_S:
        for t_i in range(0, N_S - 1, 2):
            # D[np.logical_and((T >= t_s), (T < t_s + 1.0)), 0] = d0 + np.random.randint(-2, 3)*1.0
            D[np.logical_and((T >= T_S[t_i]), (T < T_S[t_i + 1])), 0] = d0 + 1.0

    if (MODEL == 'boost'):
        # Create the model (Boost extension)
        msd = MSD_BOOST("Mass-Spring-Damper_FMIN_EST", N)
        msd.set_external_forces(T_S, D_S, 'zero')
    elif (MODEL == 'cython'):
        # Create the model (Boost extension)
        msd = MSD_CYTHON("Mass-Spring-Damper_FMIN_EST")
        msd.set_external_forces(T_S, D_S, 'zero')
    else:
        # Create the model (pure Python)
        msd = MSD("Mass-Spring-Damper_FMIN_EST")
        msd.set_external_forces(T_S, D_S, 'zero')

    c_idx = ['k', 'b', 'd']

    # True parameter set
    CT = np.array([ msd.get_coeffs()[ck] for ck in c_idx ])

    # Initial parameter set
    C0 = np.array([ 0.5*msd.get_coeffs()[ck] for ck in c_idx ])

    # Initial parameter dict
    CD = dict(zip(c_idx, C0))

    # Function to interpolate over external force input at integration time
    # d_func = interpolate.interp1d(T, D, kind='linear', axis=0, bounds_error=False)
    # d_func = interpolate.interp1d(T_S, D_S, kind='zero', axis=0, bounds_error=False)

    # msd.set_external_forces(T_S, D_S, 'zero')

    if (STATE_NOISE_SD > 0.0):
        # sdw = np.array(STATE_NOISE_SD)
        # W = np.random.randn(N, len(x0))*STATE_NOISE_SD
        sdw = STATE_NOISE_SD
        W = np.random.randn(N, 1)*sdw
        msd.add_state_noise(T, W)

    # Compute the response
    # X, Xdot, F = msd.integrate(x0, T, d_func)
    X, Xdot, F = msd.integrate(x0, T)

    sys.stdout.write("\n")

    # Measured state, state rate, external force input and force matrices

    sdz = np.zeros((len(x0),))

    if any(w > 0.0 for w in NOISE_SD[:2]):
        sdz = np.array(NOISE_SD[:2])
        # sdz = np.array([ 0.1, 0.1 ])*0.01
    Z = X + np.random.randn(N, len(x0))*sdz
    z0 = x0

    Nu = Z[:,1]
    sdnu = sdz[1]

    sdzdot = np.zeros((len(x0),))
    if any(w > 0.0 for w in NOISE_SD[:2]):
        sdzdot = np.array(NOISE_SD[:2])
        # sdzdot = np.array([ 0.1, 0.1 ])*0.01
    Zdot = Xdot + np.random.randn(N, len(x0))*sdzdot

    # Re-compute state rates from measured system state?
    # for n in range(N):
    #     if (n == 0):
    #         zdot = (Z[1] - Z[0])/dt
    #     elif (n == N - 1):
    #         zdot = (Z[N - 1] - Z[N - 2])/dt
    #     else:
    #         zdot = (Z[n + 1] - Z[n - 1])/dt/2.0
    #     Zdot[n] = zdot

        # sde = 0.1
    sde = 0.0
    if (NOISE_SD[2] > 0.0):
        sde = NOISE_SD[2]
    E = D + np.random.randn(N, 1)*sde
    e0 = d0

    G = F.copy()
    sdg = 0.0
    if any(w > 0.0 for w in NOISE_SD):
        # Forces are calculated from (measured) accelerations, not measured directly
        for n in range(N):
            G[n] = msd.forces(Zdot[n], Z[n])
        sdg = np.std(F - G)

    # Initial measured system state
    # z0 = Z[0,:]
    # z0 = X[0,:]

    if PLOT_SIM:
        execfile("simplot.py")
