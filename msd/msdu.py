#!/usr/bin/env python

import numpy as np
from scipy import integrate

pyublas_exists = True
try:
    import pyublas
except ImportError:
    pyublas_exists = False

if pyublas_exists:
    from msd import msdux


# ------------------------------------------------------------------------------
# MSD_PYUBLAS class
# ------------------------------------------------------------------------------
class MSD_PYUBLAS(object):
    """
    The MSD_PYUBLAS class utilises the msdux PyUblas C++ extension
    to simulate a a Mass-Spring-Damper system.
    """
    # System parameters
    m = 30.48 # this is duplicated in msdux.cpp

    def __init__(self, name, N, **kwargs):
        """
        Initialise the MSD_PYUBLAS object.

        :param: name  = system name
        """
        self.name = name
        self.N = N

        # Pass through any other keyword arguments
        for key in kwargs:
            self.__dict__[key] = kwargs[key]

        self.plant = msdux.Plant()
        self.observer = msdux.Observer(self.N)

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
        # N = msdux.integrate(self.plant, self.observer, T_S[0], T_S[N_S - 1], dt)
        # msdux.integrate(self.plant, self.observer, T[0], dt, self.N - 1)
        msdux.integrate(self.plant, self.observer, T[0], dt, self.N)

        X = self.observer.X
        Xdot = self.observer.Xdot
        F = self.observer.F.reshape((-1, 1))

        return X, Xdot, F
