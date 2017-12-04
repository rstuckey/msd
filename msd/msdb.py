#!/usr/bin/env python

import numpy as np
from scipy import integrate

from msd import msdbx


# ------------------------------------------------------------------------------
# MSD_BOOST class
# ------------------------------------------------------------------------------
class MSD_BOOST(object):
    """
    The MSD_BOOST class utilises the msdbx Boost Python C++ extension
    to simulate a a Mass-Spring-Damper system.
    """
    # System parameters
    m = 30.48 # this is duplicated in msdbx.cpp

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

        self.plant = msdbx.Plant()
        self.observer = msdbx.Observer(self.N)

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
        # Squeeze out any superfluous dimensions and convert to lists
        T_S = T_S.tolist()
        D_S = D_S.squeeze().tolist()
        self.plant.set_external_forces(T_S, D_S, kind)

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
        # print("Setting initial state!")
        self.plant.set_initial_state(x0.tolist())
        # N = msdbx.integrate(self.plant, self.observer, T_S[0], T_S[N_S - 1], dt)
        # msdbx.integrate(self.plant, self.observer, T[0], dt, self.N - 1)
        # print("Integrating!")
        msdbx.integrate(self.plant, self.observer, T[0], dt, self.N)

        # print("Copying!")
        # This conversion is slow!
        X = np.array(self.observer.X)
        Xdot = np.array(self.observer.Xdot)
        F = np.array(self.observer.F).reshape((-1, 1))
        # X = self.observer.X
        # Xdot = self.observer.Xdot
        # F = self.observer.F

        return X, Xdot, F
