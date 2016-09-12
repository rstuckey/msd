#!/usr/bin/env python

import numpy as np
import numpy.matlib as ml
from scipy import interpolate, integrate

import array

pyublas_exists = True
try:
    import pyublas
except ImportError:
    pyublas_exists = False

if pyublas_exists:
    import msde


# ------------------------------------------------------------------------------
# MSD class
# ------------------------------------------------------------------------------
class MSD(object):
    """
    The MSD class represents a Mass-Spring-Damper system.
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

        # External force function
        self.d_func = None

        # State noise function
        self.w_func = None

        self.init()

    def __str__(self):
        return self.name

    def init(self):
        """
        Construct the force and moment matrices.
        """
        # Rigid body mass matrix
        self.C_M_I = 1.0/self.m

        self.C_SD = ml.zeros((2, 2))
        self.M_EF = ml.zeros((2, 1))

    def get_coeffs(self):
        """
        Get the model coefficients.
        """
        return self.C

    def set_coeffs(self, C):
        """
        Set the model coefficients.
        """
        for ck in C.keys():
            self.C[ck] = C[ck]

    def set_external_forces(self, T_S, D_S, interp_kind):
        """
        Set the external force interpolant points.
        """
        self.d_func = interpolate.interp1d(T_S, D_S, kind=interp_kind, axis=0, bounds_error=False)

    def add_state_noise(self, T_S, W_S):
        """
        Set the state noise interpolant points.
        """
        self.w_func = interpolate.interp1d(T_S, W_S, kind='linear', axis=0, bounds_error=False)

    def rates(self, x, t):
        """
        Calculate the system state-rate for the current state x.

        :param: x = current system state [ xp, xpd ]
        :param: t = current time

        :returns: xdot = system state-rate
        """

        # Spring-damper forces
        # C_SD = np.mat([[    0.0,      self.m     ],
        #                [ self.C['k'], self.C['b']]])
        self.C_SD[0,0] = 0.0
        self.C_SD[0,1] = self.m
        self.C_SD[1,0] = self.C['k']
        self.C_SD[1,1] = self.C['b']

        M_SD = self.C_SD*x.reshape((-1, 1))
        M_SD[1,0] = M_SD[1,0] + self.C['z']*x[1]*x[1] # z is a dummy coefficient

        d = np.nan_to_num(self.d_func(t))

        # External force
        # M_EF = np.mat([[           0.0 ],
        #                [ self.C['d']*f ]])
        self.M_EF[0,0] = 0.0
        self.M_EF[1,0] = self.C['d']*d

        xdot = np.ravel(self.C_M_I*(M_SD + self.M_EF))

        if (self.w_func is not None):
            # xdot += np.nan_to_num(self.w_func(t))
            xdot[1] += np.nan_to_num(self.w_func(t))

        return xdot

    def rrates(self, t, x):
        """
        Rates method with arguments reversed.
        """
        return self.rates(x, t, self.d_func)

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

        # Initialise the model
        self.init()

        # Test rates function
        # self.rates(x0, T[0])

        # Perform the integration
        # X = integrate.odeint(self.rates, x0, T, args=(d_func,), rtol=1.49012e0, atol=1.49012e0, mxstep=4)
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


# ------------------------------------------------------------------------------
# MSD_BOOST class
# ------------------------------------------------------------------------------
class MSD_BOOST(object):
    """
    The MSD_BOOST class utilises the msde Boost Python C++ extension
    to simulate a a Mass-Spring-Damper system.
    """
    # System parameters
    m = 30.48 # this is duplicated in msde.cpp

    def __init__(self, name, N, **kwargs):
        """
        Initialise the MSD_BOOST object.

        :param: name  = system name
        """
        self.name = name
        self.N = N

        # Pass through any other keyword arguments
        for key in kwargs:
            self.__dict__[key] = kwargs[key]

        self.plant = msde.Plant()
        self.observer = msde.Observer(self.N)

    def __str__(self):
        return self.name

    def get_coeffs(self):
        """
        Get the model coefficients.
        """
        return self.plant.get_coeffs()

    def set_coeffs(self, C):
        """
        Set the model coefficients.
        """
        self.plant.set_coeffs(C)

    def set_external_forces(self, T_S, D_S, kind):
        """
        Set the external force interpolant points.
        """
        self.plant.set_external_forces(np.array(T_S), np.array(D_S), kind)

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
        dt = T[1] - T[0]
        # N = T.shape[0]
        self.plant.set_initial_state(x0)
        # N = msde.integrate(self.plant, self.observer, T_S[0], T_S[N_S - 1], dt)
        # msde.integrate(self.plant, self.observer, T[0], dt, self.N - 1)
        msde.integrate(self.plant, self.observer, T[0], dt, self.N)

        X = self.observer.X
        Xdot = self.observer.Xdot
        F = self.observer.F.reshape((-1, 1))

        return X, Xdot, F
