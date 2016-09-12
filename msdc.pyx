#!/usr/bin/env python

import array
# from libcpp cimport bool as bool_t
from cpython cimport bool
import numpy as np
import numpy.matlib as ml
from scipy import interpolate, integrate

cython_exists = True
try:
    from cpython cimport array
except ImportError:
    cython_exists = False


cdef double interp1d_zero(double t, double[:] _T_S, double[:] _D_S):

    cdef int n = 0
    cdef int _N_S = len(_T_S)

    if (t <= _T_S[0]):
        return _D_S[0]
    elif (_T_S[_N_S- 1] <= t):
        return _D_S[_N_S - 1]

    while ((n < _N_S - 1) and (_T_S[n] < t)):
        n += 1

    return _D_S[n - 1]

cdef double interp1d_linear(double t, double[:] _T_S, double[:] _D_S):

    cdef int n = 0
    cdef int _N_S = len(_T_S)
    cdef double dddt

    if (t <= _T_S[0]):
        return _D_S[0]
    elif (_T_S[_N_S - 1] <= t):
        return _D_S[_N_S - 1]

    while ((n < _N_S - 1) and (_T_S[n] < t)):
        n += 1

    dddt = (_D_S[n] - _D_S[n - 1])/(_T_S[n] - _T_S[n - 1])

    return _D_S[n - 1] + dddt*(t - _T_S[n - 1])

cdef double interp1d_linear_uniform(double t, double[:] _T_S, double[:] _D_S):

    cdef int n = 0
    cdef int _N_S = len(_T_S)
    cdef double dddt

    if (t <= _T_S[0]):
        return _D_S[0]
    elif (_T_S[_N_S - 1] <= t):
        return _D_S[_N_S - 1]

    n = int((t - _T_S[0])/(_T_S[1] - _T_S[0])) + 1

    dddt = (_D_S[n] - _D_S[n - 1])/(_T_S[n] - _T_S[n - 1])

    return _D_S[n - 1] + dddt*(t - _T_S[n - 1])

cdef get_rates(double[:] x, double t, double[:] T_Sa, double[:] D_Sa, bytes interp_kind, double m, double[:] C, bool state_noise, double[:] Ta, double[:] Wa, double[:] xdot):

    cdef double d
    if (interp_kind == 'zero'):
        d = interp1d_zero(t, T_Sa, D_Sa)
    elif (interp_kind == 'linear_uniform'):
        d = interp1d_linear_uniform(t, T_Sa, D_Sa)
    else:
        d = interp1d_linear(t, T_Sa, D_Sa)

    w = 0.0
    if state_noise:
        w = interp1d_linear_uniform(t, Ta, Wa)

    xdot[0] = 1.0/m*(0.0*x[0] +   m*x[1] +   0.0)
    xdot[1] = 1.0/m*(C[0]*x[0] + C[1]*x[1] + C[2]*d) + w


# ------------------------------------------------------------------------------
# MSD_CYTHON class
# ------------------------------------------------------------------------------
class MSD_CYTHON(object):
    """
    The MSD_CYTHON class represents a Mass-Spring-Damper system.
    """
    # System parameters
    m = 30.48

    def __init__(self, name, **kwargs):
        """
        Initialise the msd object.

        :param: name  = system name
        """
        self.name = name

        # Pass through any other keyword arguments
        for key in kwargs:
            self.__dict__[key] = kwargs[key]

        self.c_idx = [ 'k', 'b', 'd' ]

        # Model coefficients
        self.C = { 'k': -50.0, 'b': -10.0, 'd': 1.0, 'z': 0.0 }

        self.interp_kind = 'linear'

        self.xa = array.array('d', [ 0.0, 0.0 ])
        self.xdota = array.array('d', [ 0.0, 0.0 ])

        self.Ca = array.array('d', [ self.C[self.c_idx[i]] for i in range(len(self.c_idx)) ])
        self.T_Sa = array.array('d', [ ])
        self.D_Sa = array.array('d', [ ])

        # cdef object state_noise = <bool_t> False
        # self.state_noise = state_noise
        self.state_noise = False
        self.Ta = array.array('d', [ ])
        self.Wa = array.array('d', [ ])

    def __str__(self):
        return self.name

    def init(self):
        pass

    def get_coeffs(self):
        """
        Get the model coefficients.
        """
        return self.C

    def set_coeffs(self, C):
        """
        Set the model coefficients.
        """
        # for i in range(len(self.c_idx)):
        #     ck = self.c_idx[i]
        #     self.C[ck] = C[ck]
        #     self.Ca[i] = C[ck]
        for ck, c in C.iteritems():
            if ck in self.c_idx:
                self.C[ck] = C[ck]
                i = self.c_idx.index(ck)
                self.Ca[i] = C[ck]

    def set_external_forces(self, T_S, D_S, interp_kind):
        """
        Set the external force interpolant points.
        """
        self.T_S = T_S
        self.D_S = D_S
        self.interp_kind = interp_kind

        self.T_Sa = array.array('d', T_S)
        self.D_Sa = array.array('d', D_S)

    def add_state_noise(self, T, W):
        """
        Set the state noise interpolant points.
        """
        self.T = T
        self.W = W

        self.Ta = array.array('d', T)
        self.Wa = array.array('d', W)

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

        # self.xa = get_arr(self.T_Sa, self.D_Sa, t, self.x)
        get_rates(self.xa, t, self.T_Sa, self.D_Sa, self.interp_kind, self.m, self.Ca, self.state_noise, self.Ta, self.Wa, self.xdota)

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

    def integrate(self, x0, T):
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
        self.init()

        # Test rates function
        # self.rates(x0, T[0])

        # cdef array.array T_S = array.array('d', self.T_S)
        # cdef array.array D_S = array.array('d', self.D_S)

        # cdef int[:] ca = a


        # Perform the integration
        # X = integrate.odeint(self.rates, x0, T, args=(d_func,), rtol=1.49012e0, atol=1.49012e0, mxstep=4)
        # X = integrate.odeint(self.rates, x0, T, rtol=1.0e-6, atol=1.0e-6)
        X = integrate.odeint(self.rates, x0, T, rtol=1.0e-6, atol=1.0e-6)

        # X = integrate.odeint(self.rates, x0, T)

        Xdot = np.zeros((N, len(x0)))
        for n in range(N):
            Xdot[n] = self.rates(X[n], T[n])

        # Force and moment matrix
        F = np.zeros((N, 1))
        for n in range(N):
            F[n] = self.forces(Xdot[n], X[n])

        return X, Xdot, F
