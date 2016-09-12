#!/usr/bin/env python

import matplotlib.pyplot as pp


def run_from_ipython():
    try:
        __IPYTHON__
        return True
    except NameError:
        return False


if __name__ == '__main__':

    if ('msd' not in locals()): # sim has not been run
        raise("Please run sim first!")

    x_str = [ r'$x (m)$', r'$\dot x (m/s)$' ]
    x_fac = [ 1.0, 1.0 ]
    xd_str = [ r'$\dot x (m/s)$', r'$\ddot x (m/s^2)$' ]
    xd_fac = [ 1.0, 1.0 ]
    d_str = [ r'$\delta_F (N)$' ]
    d_fac = [ 1.0 ]
    f_str = [ r'$X (N)$' ]
    f_fac = [ 1.0 ]

    if (not pp.isinteractive()):
        pp.ion()

    newfig = True
    figs = pp.get_fignums()
    for fignum in figs:
        if (pp.figure(fignum).get_label() == msd.name + " Simulation"):
            newfig = False
            break

    nc = 1 # number of columns
    nr = 6 # number of rows

    if (newfig):

        figsim = pp.figure(msd.name + " Simulation", figsize=(8.0, 15.0))
        figsimnum = figsim.number

        ax = pp.subplot(nr, nc, 1)
        ax.grid(color='lightgrey', linestyle=':')
        pp.plot(T, G*f_fac[0], color='mediumpurple', linestyle='-', linewidth=1.5)
        pp.plot(T, F*f_fac[0], color='purple', linestyle='-', linewidth=1.5)
        pp.xlim(T[0], T[N - 1])
        pp.ylabel(f_str[0])

        ax = pp.subplot(nr, nc, 2)
        ax.grid(color='lightgrey', linestyle=':')
        pp.plot(T, Zdot[:,1]*xd_fac[0], color='dodgerblue', linestyle='-', linewidth=1.5)
        pp.xlim(T[0], T[N - 1])
        pp.ylabel(xd_str[1])

        for j in range(2):
            ax = pp.subplot(nr, nc, j + 3)
            ax.grid(color='lightgrey', linestyle=':')
            pp.plot(T, Z[:,j]*x_fac[j], color='lightseagreen', linestyle='-', linewidth=1.5)
            pp.xlim(T[0], T[N - 1])
            pp.ylabel(x_str[j])

        ax = pp.subplot(nr, nc, 5)
        ax.grid(color='lightgrey', linestyle=':')
        pp.plot(T, E*d_fac[0], color='forestgreen', linestyle='-', linewidth=1.5)
        pp.xlim(T[0], T[N - 1])
        pp.ylabel(d_str[0])

        pp.xlabel(r'$time (s)$')

        axes = figsim.get_axes()

        pp.draw()

    if ('Xe' in locals()): # reg, estim or bms has been run
        figsim = pp.figure(figsimnum)

        axes = figsim.get_axes()

        pp.axes(axes[0])
        if ('H' in locals()):
            pp.plot(T, H*f_fac[0], color='red', linestyle='-', linewidth=1.5)
        pp.plot(T, Fe*f_fac[0], color='red', linestyle='-', linewidth=1.5)

        pp.axes(axes[1])
        pp.plot(T, Xedot[:,1]*xd_fac[1], color='red', linestyle='-', linewidth=1.5)

        for j in range(2):
            pp.axes(axes[j + 2])
            pp.plot(T, Xe[:,j]*x_fac[j], color='red', linestyle='-', linewidth=1.5)

            pp.xlabel(r'$time (s)$')

    pp.show()
