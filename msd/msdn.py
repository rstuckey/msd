#!/usr/bin/env python

import numpy as np
from scipy import integrate

from numba import jit, jitclass
from numba import f8, i1, b1


# ------------------------------------------------------------------------------
# MSD_NUMBA class
# ------------------------------------------------------------------------------
spec = [
    ('m', f8),
    ('interp_enum', i1),
    ('xa', f8[:]),
    ('xdota', f8[:]),
    ('Ca', f8[:]),
    ('T_Sa', f8[:]),
    ('D_Sa', f8[:]),
    ('state_noise', b1),
    ('Ta', f8[:]),
    ('Wa', f8[:])
]
@jitclass(spec)
class MSD_NUMBA(object):
    """
    The MSD_NUMBA class represents a Mass-Spring-Damper system.
    """
    def __init__(self):
        """
        Initialise the msd object.
        """
        self.m = 30.48

        self.interp_enum = 2

        self.xa = np.array([ 0.0, 0.0 ])
        self.xdota = np.array([ 0.0, 0.0 ])

        # Model coefficients
        self.Ca = np.array([ -50.0, -10.0, 1.0 ])
        self.T_Sa = np.array([ 0.0 ])
        self.D_Sa = np.array([ 0.0 ])

        self.state_noise = False
        self.Ta = np.array([ 0.0 ])
        self.Wa = np.array([ 0.0 ])

    def init(self):
        pass

    def get_coeffs(self):
        """
        Get the model coefficient array.
        """
        return self.Ca

    def set_coeffs(self, Ca):
        """
        Set the model coefficient array.
        """
        for i in range(len(Ca)):
            self.Ca[i] = Ca[i]

    def set_external_forces(self, T_S, D_S, interp_enum):
        """
        Set the external force interpolant points.
        """
#         self.T_S = T_S.copy()
#         self.D_S = D_S.copy()
        self.interp_enum = interp_enum

#         self.T_Sa = np.array(T_S)
#         self.D_Sa = np.array(D_S)
        for i in range(len(T_S)):
            self.T_Sa[i] = T_S[i]
        for i in range(len(D_S)):
            self.D_Sa[i] = D_S[i]

    def add_state_noise(self, T, W):
        """
        Set the state noise interpolant points.
        """
#         self.T = T.copy()
#         self.W = W.copy()

#         self.Ta = np.array(T)
#         self.Wa = np.array(W)
        for i in range(len(T)):
            self.Ta[i] = T[i]
        for i in range(len(W)):
            self.Wa[i] = W[i]

        self.state_noise = True

    def rates(self, x, t):
        """
        Calculate the system state-rate for the current state x.

        :param: x = current system state [ xp, xpd ]
        :param: t = current time

        :returns: xdot = system state-rate
        """
        self.xa[0] = x[0]
        self.xa[1] = x[1]

        self.xdota = self.get_rates(self.xa, t, self.T_Sa, self.D_Sa, self.interp_enum, self.m, self.Ca, self.state_noise, self.Ta, self.Wa)

        return [ self.xdota[0], self.xdota[1] ]

    def forces(self, xdot, x):
        """
        Calculate the forces from recorded state data.

        :param: xdot = system state rate
        :param: x = system state [ xp, xpd ]

        :returns: f = state forces
        """
        xpddot = xdot[1]

        f = self.m*xpddot

        return f

#     @jit('f8(f8, f8[:], f8[:])')
    def interp1d_zero(self, t, _T_S, _D_S):
        # Zero-order interpolation of the external force vector

        n = 0
        _N_S = len(_T_S)

        if (t <= _T_S[0]):
            return _D_S[0]
        elif (_T_S[_N_S- 1] <= t):
            return _D_S[_N_S - 1]

        while ((n < _N_S - 1) and (_T_S[n] < t)):
            n += 1

        return _D_S[n - 1]

#     @jit('f8(f8, f8[:], f8[:])')
    def interp1d_linear(self, t, _T_S, _D_S):
        # First-order interpolation of the external force vector, with non-uniform sampling frequency

        n = 0
        _N_S = len(_T_S)

        if (t <= _T_S[0]):
            return _D_S[0]
        elif (_T_S[_N_S - 1] <= t):
            return _D_S[_N_S - 1]

        while ((n < _N_S - 1) and (_T_S[n] < t)):
            n += 1

        dddt = (_D_S[n] - _D_S[n - 1])/(_T_S[n] - _T_S[n - 1])

        return _D_S[n - 1] + dddt*(t - _T_S[n - 1])

#     @jit('f8(f8, f8[:], f8[:])')
    def interp1d_linear_uniform(self, t, _T_S, _D_S):
        # First-order interpolation of the external force vector, with uniform sampling frequency

        n = 0
        _N_S = len(_T_S)

        if (t <= _T_S[0]):
            return _D_S[0]
        elif (_T_S[_N_S - 1] <= t):
            return _D_S[_N_S - 1]

        n = int((t - _T_S[0])/(_T_S[1] - _T_S[0])) + 1

        dddt = (_D_S[n] - _D_S[n - 1])/(_T_S[n] - _T_S[n - 1])

        return _D_S[n - 1] + dddt*(t - _T_S[n - 1])

#     @jit('f8[:](f8[:], f8, f8[:], f8[:], i1, f8, f8[:], b1, f8[:], f8[:])')
    def get_rates(self, x, t, T_Sa, D_Sa, interp_enum, m, C, state_noise, Ta, Wa):
        # Calculate the state rate from the state, external force and system parameters

        if (interp_enum == 0):
            d = self.interp1d_zero(t, T_Sa, D_Sa)
        elif (interp_enum == 1):
            d = self.interp1d_linear_uniform(t, T_Sa, D_Sa)
        else:
            d = self.interp1d_linear(t, T_Sa, D_Sa)

        w = 0.0
        if state_noise:
            w = self.interp1d_linear_uniform(t, Ta, Wa)

        xdot = np.zeros((2, ))

        xdot[0] = 1.0/m*(0.0*x[0] +   m*x[1] +   0.0)
        xdot[1] = 1.0/m*(C[0]*x[0] + C[1]*x[1] + C[2]*d) + w

        return xdot


@jit(nopython=False)
def msd_integrate(msd, x0, T):
    """
    Integrate the differential equations and calculate the resulting rates and forces.

    :param: x0 = initial system state
    :param: T = sequence of time points for which to solve for x

    :returns: X = system state array
    :returns: Xdot = state rates array
    :returns: F = state force array
    """
    N = T.shape[0]

    dt = T[1] - T[0]

    # Initialise the model
    msd.init()

    # Perform the integration
    X = integrate.odeint(msd.rates, x0, T, rtol=1.0e-6, atol=1.0e-6)

    Xdot = np.zeros((N, len(x0)))
    for n in range(N):
        Xdot[n] = msd.rates(X[n], T[n])

    # Force and moment matrix
    F = np.zeros((N, 1))
    for n in range(N):
        F[n] = msd.forces(Xdot[n], X[n])

    return X, Xdot, F
