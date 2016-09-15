#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as pp


x_str = [ r'$x (m)$', r'$\dot x (m/s)$' ]
x_fac = [ 1.0, 1.0 ]
# xd_str = [ r'$\dot x (m/s)$', r'$\ddot x (m/s^2)$' ]
# xd_fac = [ 1.0, 1.0 ]
d_str = [ r'$\delta_F (N)$' ]
d_fac = [ 1.0 ]
f_str = [ r'$X (N)$' ]
f_fac = [ 1.0 ]
ff_str = [ r'ln ${\Delta X}^2 (N^2)$' ]

c_idx = [ 'k', 'b', 'd' ]

def plot(name, T, D, X, F, Xe=None, Fe=None, FF=None):
    """
    Plot the system response.
    """
    N = T.shape[0]

    nc = 2 # number of columns
    # nr = 2 # number of rows
    if (FF is None):
        nr = 2 # number of rows
        figsize = (10.0, 4.0)
    else:
        nr = 3
        figsize = (10.0, 6.0)

    # fig = pp.figure(name + " Optimisation", figsize=(12.0, 4.0))

    fig, AxesArr = pp.subplots(nr, nc, figsize=figsize)

    # Axes = [ Axarr[0, 0], Axarr[0, 1], Axarr[1, 0], Axarr[1, 1] ]
    # Axes = [ ]
    Axes = np.ravel(AxesArr)
    Lines = [ ]
    Text = [ ]

    # ax = pp.subplot(nr, nc, 1)
    # Axes.append(ax)
    ax = Axes[0]
    ax.grid(color='lightgrey', linestyle=':')
    ax.plot(T, F*f_fac[0], color='#DFD4F4', linestyle='-', linewidth=1.5)
    if (Fe is not None):
        lines = ax.plot(T, Fe*f_fac[0], color='blue', linestyle='-', linewidth=1.5)
        Lines.append(lines)
    ax.set_xlim(T[0], T[N - 1])
    ax.autoscale(enable=False)
    ax.set_ylabel(f_str[0])

    for j in range(2):
        # ax = pp.subplot(nr, nc, j + 2)
        # Axes.append(ax)
        ax = Axes[j + 1]
        ax.grid(color='lightgrey', linestyle=':')
        ax.plot(T, X[:,j]*x_fac[j], color='#BCE8E6', linestyle='-', linewidth=1.5)
        if (Xe is not None):
            lines = ax.plot(T, Xe[:,j]*x_fac[j], color='blue', linestyle='-', linewidth=1.5)
            Lines.append(lines)
        ax.set_xlim(T[0], T[N - 1])
        ax.autoscale(enable=False)
        ax.set_ylabel(x_str[j])

    # ax = pp.subplot(nr, nc, 4)
    # Axes.append(ax)
    ax = Axes[3]
    ax.grid(color='lightgrey', linestyle=':')
    ax.plot(T, D*d_fac[0], color='#BDDCBD', linestyle='-', linewidth=1.5)
    ax.set_xlim(T[0], T[N - 1])
    ax.autoscale(enable=False)
    ax.set_ylabel(d_str[0])

    if (FF is not None):
        ax = Axes[4]
        ax.grid(color='lightgrey', linestyle=':')
        lines = ax.plot(range(0, np.size(FF, 0)), FF, color='red', linestyle='-', linewidth=1.5)
        Lines.append(lines)
        ax.set_xlim(0, np.size(FF, 0))
        ax.set_ylim(bottom=0.0)
        ax.autoscale(enable=False)
        # ax.autoscale(enable=True, axis='y')
        ax.set_ylabel(ff_str[0])
        lim = ax.set_ylim(bottom=0.0)

        text = ax.annotate("{:.4f}".format(0.0), xy=(0.98, 0.06), xycoords='axes fraction', backgroundcolor='black', color='white', fontsize=12, horizontalalignment='right', verticalalignment='bottom', weight='bold')
        Text.append(text)

    if (FF is not None):
        ax = Axes[5]
        ax.set_xticks([ ])
        ax.set_xticklabels([ ])
        ax.set_yticks([ ])
        ax.set_yticklabels([ ])
        for j in range(3):
            text = ax.annotate("{} = {:.4f}".format(c_idx[j], 0.0), xy=(0.1, 1.0 - 0.2*(j + 1)), xycoords='axes fraction', backgroundcolor='white', color='black', fontsize=12, horizontalalignment='left', verticalalignment='bottom', weight='bold')
            Text.append(text)

    # pp.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.2)
    pp.subplots_adjust(left=0.10, wspace=0.3)

    # pp.show()
    fig.canvas.show()

    return fig, Axes, Lines, Text

def updateplot(fig, Axes, Lines, Text, Xe, Fe, FF, f_max=None, f_txt=None, c_txt=None): #, ylim=None):
    """
    Update the system response plots with new data.
    """
    Lines[0][0].set_ydata(Fe*f_fac[0])
    for j in range(2):
        Lines[j + 1][0].set_ydata(Xe[:,j]*x_fac[j])
    Lines[3][0].set_ydata(FF)
    if (f_max is not None):
        Axes[4].set_ylim(top=f_max)
    if (f_txt is not None):
        Text[0].set_text(f_txt)

    if (c_txt is not None):
        for j in range(3):
            Text[j + 1].set_text("{} = {:.4f}".format(c_idx[j], c_txt[j]))

    # if rescale:
    #     Axes[4].set_ylim(ylim)
        # Axes[4].relim(visible_only=True)
        # Axes[4].autoscale_view(scalex=False)
    # pp.draw()
    fig.canvas.draw()

def clearplot(fig, Lines):
    """
    Clear the system response plots (set visible to False).
    """
    Lines[0][0].set_visible(False)
    for j in range(2):
        Lines[j + 1][0].set_visible(False)
    # pp.draw()
    fig.canvas.draw()

def addplot(fig, Axes, T, Xe, Fe, **axis_props):
    """
    Add data to the system response plots.
    """
    # pp.axes(Axes[0])
    ax = Axes[0]
    ax.plot(T, Fe*f_fac[0], linestyle='-', linewidth=1.5, **axis_props)
    for j in range(2):
        # pp.axes(Axes[j + 1])
        ax = Axes[j + 1]
        ax.plot(T, Xe[:,j]*x_fac[j], linestyle='-', linewidth=1.5, **axis_props)
    # pp.draw()
    fig.canvas.draw()

