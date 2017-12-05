#!/usr/bin/env python

import numpy as np
import numpy.matlib as ml
from scipy import interpolate, integrate

# Simulation model
MODEL = 'numba_jc' # ['python', 'cython', 'pyublas', 'numba', 'numba_jc', boost']

if (MODEL == 'python'):
    # Pure Python
    from msd import MSD
elif (MODEL == 'cython'):
    # Cython
    from msd.msdc import MSD_CYTHON
elif (MODEL == 'pyublas'):
    # PyUblas extension
    from msd.msdu import MSD_PYUBLAS
elif (MODEL == 'numba'):
    # Numba JIT
    from msd.msdn import MSD_NUMBA
elif (MODEL == 'numba_jc'):
    # Numba JIT
    from msd.msdn import MSD_NUMBA_JC, msd_integrate
elif (MODEL == 'boost'):
    # Boost extension
    from msd.msdb import MSD_BOOST


# Measurement and state noise standard deviations
NOISE_SD = [ 0.001 for _ in range(3) ]
NOISE_SD = [ 0.0 for _ in range(3) ]
STATE_NOISE_SD = 0.0

VERBOSE = False

# Zero the RNG seed
np.random.seed(1)

# Initial system state and external force input
x0 = np.zeros((2, ))
d0 = 0.0

# Sample period
dt = 0.01

# Start and end time
t0 = 0.0
tN = 15.0

# Create the time vector
T = np.arange(t0, tN, dt)
N = T.shape[0]

# Create the predefined external force vector
T_S0 = np.hstack((t0, np.arange(t0 + 1.0, tN + 1.0, 1.0), tN))
D_S0 = np.hstack((d0, np.array([ d0 + ((_ % 2)*2 - 1) * 1.0 for _ in range(T_S0.shape[0] - 2) ]), d0))
interpfun = interpolate.interp1d(T_S0, D_S0, kind='zero', axis=0, bounds_error=False)
D0 = np.array([ [ interpfun(t) ] for t in T ])
T_S = T_S0.copy()
D_S = D_S0.copy()
D = D0.copy()

# Create the simulation model
if (MODEL == 'python'):
    # Pure Python
    msd = MSD("Mass-Spring-Damper (Python)")
    msd.set_external_forces(T_S, D_S, 'zero')
elif (MODEL == 'cython'):
    # Cython
    msd = MSD_CYTHON("Mass-Spring-Damper (Cython)")
    msd.set_external_forces(T_S, D_S, 'zero')
elif (MODEL == 'pyublas'):
    # PyUblas extension
    msd = MSD_PYUBLAS("Mass-Spring-Damper (PyUblas)", N)
    msd.set_external_forces(T_S, D_S, 'zero')
elif (MODEL == 'numba'):
    # Numba JIT
    msd = MSD_NUMBA("Mass-Spring-Damper (Numba)", N)
    msd.set_external_forces(T_S, D_S, 'zero')
elif (MODEL == 'numba_jc'):
    # Numba JIT
    msd = MSD_NUMBA_JC(N)
    msd.set_external_forces(T_S, D_S, 0)
elif (MODEL == 'boost'):
    # Boost extension
    msd = MSD_BOOST("Mass-Spring-Damper (Boost)", N)
    msd.set_external_forces(T_S, D_S, 'zero')

# Identification keys
c_idx = ['k', 'b', 'd']

if (MODEL in ['python', 'cython', 'pyublas', 'numba', 'boost']):
	# True parameter set
	CT = [ msd.get_coeffs()[ck] for ck in c_idx ]
	# Initial parameter set
	C0 = [ 0.5*msd.get_coeffs()[ck] for ck in c_idx ]
elif (MODEL == 'numba_jc'):
	CT = msd.get_coeffs()
	C0 = 0.5*msd.get_coeffs()

# Add any state noise
if (STATE_NOISE_SD > 0.0):
    sdw = STATE_NOISE_SD
    W = np.random.randn(N, 1)*sdw
    msd.add_state_noise(T, W)

# Compute the response
if (MODEL in ['python', 'cython', 'pyublas', 'numba', 'boost']):
    X, Xdot, F = msd.integrate(x0, T)
elif (MODEL == 'numba_jc'):
    X, Xdot, F = msd_integrate(msd, x0, T)

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
