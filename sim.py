#!/usr/bin/env python

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

    # Colors for terminal output
    HEAD = '\033[1m'
    OKBL = '\033[94m'
    OKGR = '\033[92m'
    WARN = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'

    # Measurement and state noise standard deviations
    if ('NOISE_SD' not in locals()):
        NOISE_SD = 0.01
    if isinstance(NOISE_SD, (int, float)):
        NOISE_SD = [ NOISE_SD for _ in range(3) ]
    if ('STATE_NOISE_SD' not in locals()):
        STATE_NOISE_SD = 0.0

    # Output verbosity
    if ('VERBOSE' not in locals()):
        VERBOSE = False

    # Plot flag
    if ('PLOT_SIM' not in locals()):
        PLOT_SIM = False
    if PLOT_SIM:
        from plot import plot

    # Simulation model
    if ('MODEL' not in locals()):
        MODEL = 'python'
    if ((MODEL == 'boost') and (not pyublas_exists)):
        print "Warning: pyublas does not exist! Setting MODEL = 'python'"
        MODEL = 'python'
    if ((MODEL == 'cython') and (not cython_exists)):
        print "Warning: cython does not exist! Setting MODEL = 'python'"
        MODEL = 'python'

    # Zero the RNG seed
    if ('ZERO_SEED' not in locals()):
        ZERO_SEED = True
    if ZERO_SEED:
        np.random.seed(1)
    else:
        print "Warning: Random seed will be automatically set."

    # Initial system state and external force input
    x0 = np.zeros((2, ))
    d0 = 0.0

    if ('D' not in locals()):
        # Sample period
        dt = 0.01
        # Create the time vector
        T = np.arange(0.0, 15.0, dt)
        N = T.shape[0]

        # Create the predefined external force vector
        D = ml.repmat(d0, N, 1)

        T_S = [ t_s + t_ds for t_s in np.arange(0.0, 15.0, 3.0) for t_ds in [ 0.0, 1.0 ] ]
        T_S.append(T[N - 1])
        N_S = len(T_S)
        D_S = [ d0 + d_ds for t_s in np.arange(0.0, 15.0, 3.0) for d_ds in [ 1.0, 0.0 ] ]
        D_S.append(d0)

        for t_i in range(0, N_S - 1, 2):
            # D[np.logical_and((T >= t_s), (T < t_s + 1.0)), 0] = d0 + np.random.randint(-2, 3)*1.0
            D[np.logical_and((T >= T_S[t_i]), (T < T_S[t_i + 1])), 0] = d0 + 1.0

    # Create the simulation model
    if (MODEL == 'boost'):
        # Boost extension
        msd = MSD_BOOST("Mass-Spring-Damper (Boost)", N)
        msd.set_external_forces(T_S, D_S, 'zero')
    elif (MODEL == 'cython'):
        # Cython
        msd = MSD_CYTHON("Mass-Spring-Damper (Cython)")
        msd.set_external_forces(T_S, D_S, 'zero')
    else:
        # Pure Python
        msd = MSD("Mass-Spring-Damper (Python)")
        msd.set_external_forces(T_S, D_S, 'zero')

    # Identification keys
    c_idx = ['k', 'b', 'd']

    # True parameter set
    CT = np.array([ msd.get_coeffs()[ck] for ck in c_idx ])

    # Initial parameter set
    C0 = np.array([ 0.5*msd.get_coeffs()[ck] for ck in c_idx ])

    # Initial parameter dict
    CD = dict(zip(c_idx, C0))

    # Add any state noise
    if (STATE_NOISE_SD > 0.0):
        sdw = STATE_NOISE_SD
        W = np.random.randn(N, 1)*sdw
        msd.add_state_noise(T, W)

    # Compute the response
    X, Xdot, F = msd.integrate(x0, T)

    print

    # State noise standard deviation vector
    sdz = np.zeros((len(x0),))
    if any(w > 0.0 for w in NOISE_SD[:2]):
        sdz = np.array(NOISE_SD[:2])

    # Measured state matrix
    Z = X + np.random.randn(N, len(x0))*sdz

    # Set the initial measured state equal to the initial true state
    z0 = x0

    Nu = Z[:,1]
    sdnu = sdz[1]

    # State rate noise standard deviation vector
    sdzdot = np.zeros((len(x0),))
    if any(w > 0.0 for w in NOISE_SD[:2]):
        sdzdot = np.array(NOISE_SD[:2])

    # Measured state rate matrix
    Zdot = Xdot + np.random.randn(N, len(x0))*sdzdot

    # External force noise standard deviation vector
    sde = 0.0
    if (NOISE_SD[2] > 0.0):
        sde = NOISE_SD[2]

    # Measured external force matrix
    E = D + np.random.randn(N, 1)*sde

    # Set the initial measured external force equal to the initial true external force
    e0 = d0

    # Compute the inertial force and noise standard deviation
    G = F.copy()
    sdg = 0.0
    if any(w > 0.0 for w in NOISE_SD):
        # Forces are calculated from (measured) accelerations, not measured directly
        for n in range(N):
            G[n] = msd.forces(Zdot[n], Z[n])
        sdg = np.std(F - G)

    if PLOT_SIM:
        FF = ml.repmat(None, 50, 1)
        fig, Axes, Lines, Text = plot(msd.name, T, E, Z, G, Xe=np.zeros(X.shape), Fe=np.zeros(F.shape), FF=FF)
        fig.canvas.draw()
