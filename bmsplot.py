#!/usr/bin/env python

import copy

import numpy as np
import matplotlib.pyplot as pp
import matplotlib.pylab as pl

from pymc.utils import quantiles as calc_quantiles, hpd as calc_hpd


def run_from_ipython():
    try:
        __IPYTHON__
        return True
    except NameError:
        return False

if __name__ == '__main__':

    if ('Xedot' not in locals()): # bms has not been run
        raise("Please run bms first!")

    C_str = { 'k' : r'$k$', 'b' : r'$b$', 'd' : r'$d$' }

    if DO_PYMC_BMS:
        C_str.update({ 'z' : r'$z$' })

    stoch_list_plot = [ c_idx ]

    if DO_PYMC_BMS:
        stoch_list_plot.append(c_idx_ext)

    if DO_PYMC_BMS:
        mcmc_trace_temp = copy.deepcopy(mcmc_trace)

    for j in range(len(stoch_list_plot)):

        nc = 3 # number of columns
        nr = len(stoch_list_plot[j]) # number of rows

        if (j == 0):
            figname = msd_best.name + " Coefficient Traces, Autocorrelations & Histograms"
        elif (j == 1):
            figname = msd_best.name + " Extended Coefficient Traces, Autocorrelations & Histograms"
            mcmc_trace = mcmc_ext_trace

        fig, AxesArr = pp.subplots(nr, nc, figsize=(10.0, 6.0))

        acorr_maxlags = 100
        hist_num_bins = min(50, max(10, Nc[-1]/250))
        hist_hpd_alpha = 0.05

        Nccs = [ 0 ] + Nc
        for k in range(1, len(Nccs)):
            Nccs[k] += Nccs[k - 1]

        for i in range(nr):
            ck = stoch_list_plot[j][i]

            assert (len(mcmc_trace[-1][ck]) == Nc[-1]), "Trace length for %s (%d) is not equal to %d!" % (ck, len(mcmc_trace[-1][ck]), Nc[-1])

            c = np.mean(mcmc_trace[-1][ck])

            ax = AxesArr[i, 0]
            ax.grid(color='lightgrey', linestyle=':')
            ax.tick_params(axis='both', which='major', labelsize=10)
            for k in range(len(Nc)):
                alpha = 0.5
                if (k == len(Nc) - 1):
                    alpha = 1.0
                ax.plot(range(Nccs[k], Nccs[k + 1]), mcmc_trace[k][ck], alpha=alpha, color='seagreen', linestyle='-', linewidth=1.0, zorder=2)
                if (k > 0):
                    ax.axvline(Nccs[k], alpha=0.75, linestyle='--', linewidth=1.5, color='darkgreen')
            ax.set_xlim(0, sum(Nc))
            ax.set_ylabel(C_str[ck], rotation='horizontal')
            if (i == 0):
                ax.set_title("Trace")

            ax = AxesArr[i, 1]
            ax.grid(color='lightgrey', linestyle=':')
            # Calculate the autocorrelation (raw and detrended)
            (acorr_lags, acorr_c, acorr_line, acorr_b) = ax.acorr(mcmc_trace[-1][ck], detrend=pl.mlab.detrend_none, linewidth=0.0, markersize=0.0, maxlags=acorr_maxlags, usevlines=False)
            ax.fill_between(acorr_lags, acorr_line.get_ydata(), alpha=0.25, color='crimson', linewidth=0.0)
            ax.acorr(mcmc_trace[-1][ck], color='crimson', detrend=pl.mlab.detrend_mean, linestyle='-', linewidth=1.5, maxlags=acorr_maxlags)
            ax.set_xlim(-acorr_maxlags, acorr_maxlags)
            ax.set_ylim(-0.1, 1.1)
            ax.set_ylabel(C_str[ck], rotation='horizontal')
            if (i == 0):
                ax.set_title("Autocorrelation (detrended)")

            ax = AxesArr[i, 2]
            ax.grid(color='lightgrey', linestyle=':')
            # Calculate the median and 95% Highest Probability Density (HPD) or minimum width Bayesian Confidence (BCI) interval
            hist_quant = calc_quantiles(mcmc_trace[-1][ck])
            hist_hpd = calc_hpd(mcmc_trace[-1][ck], hist_hpd_alpha)
            (hist_n, hist_bins, hist_patches) = ax.hist(mcmc_trace[-1][ck], bins=hist_num_bins, color='steelblue', histtype='stepfilled', linewidth=0.0, normed=True, zorder=2)
            ax.set_ylim(0.0, max(hist_n)*1.1)
            ax.axvspan(hist_hpd[0], hist_hpd[1], alpha=0.25, facecolor='darkslategray', linewidth=1.5)
            ax.axvline(hist_quant[50], linestyle='-', linewidth=1.5, color='darkslategray')
            ax.set_ylabel(C_str[ck], rotation='horizontal')
            if (i == 0):
                ax.set_title("Posterior (%%%2.0f HPD)" % ((1.0 - hist_hpd_alpha)*100.0))

        pp.subplots_adjust(left=0.1, wspace=0.3)

        fig.canvas.show()

    if DO_PYMC_BMS:
        mcmc_trace = mcmc_trace_temp
