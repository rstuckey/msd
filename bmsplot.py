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

        figtrace = pp.figure(figname, figsize=(15.0, 9.0))

        #Nso = Nsb[0] - Nsb[1]
        acorr_maxlags = 100
        # acorr_maxlags = 50
        hist_num_bins = min(50, max(5, Nc[-1]/1000))
        hist_hpd_alpha = 0.05

        Nccs = [ 0 ] + Nc
        for k in range(1, len(Nccs)):
            Nccs[k] += Nccs[k - 1]

        for i in range(nr):
            ck = stoch_list_plot[j][i]

            assert (len(mcmc_trace[-1][ck]) == Nc[-1]), "Trace length for %s (%d) is not equal to %d!" % (ck, len(mcmc_trace[-1][ck]), Nc[-1])

            c = np.mean(mcmc_trace[-1][ck])

            ax = pp.subplot(nr, nc, i*nc + 1)
            ax.grid(color='lightgrey', linestyle=':')
            ax.tick_params(axis='both', which='major', labelsize=10)
            for k in range(len(Nc)):
                alpha = 0.5
                if (k == len(Nc) - 1):
                    alpha = 1.0
                pp.plot(range(Nccs[k], Nccs[k + 1]), mcmc_trace[k][ck], alpha=alpha, color='seagreen', linestyle='-', linewidth=1.0, zorder=2)
                if (k > 0):
                    pp.axvline(Nccs[k], alpha=0.75, linestyle='--', linewidth=1.5, color='darkgreen')
            pp.xlim(0, sum(Nc))
            pp.ylabel(C_str[ck], rotation='horizontal')
            if (i == 0):
                pp.title("Trace")

            ax = pp.subplot(nr, nc, i*nc + 2)
            ax.grid(color='lightgrey', linestyle=':')
            # Calculate the autocorrelation (raw and detrended)
            (acorr_lags, acorr_c, acorr_line, acorr_b) = pp.acorr(mcmc_trace[-1][ck], detrend=pl.mlab.detrend_none, linewidth=0.0, markersize=0.0, maxlags=acorr_maxlags, usevlines=False)
            pp.fill_between(acorr_lags, acorr_line.get_ydata(), alpha=0.25, color='crimson', linewidth=0.0)
            pp.acorr(mcmc_trace[-1][ck], color='crimson', detrend=pl.mlab.detrend_mean, linestyle='-', linewidth=1.5, maxlags=acorr_maxlags)
            pp.xlim(-acorr_maxlags, acorr_maxlags)
            pp.ylim(-0.1, 1.1)
            pp.ylabel(C_str[ck], rotation='horizontal')
            if (i == 0):
                pp.title("Autocorrelation (detrended)")

            ax = pp.subplot(nr, nc, i*nc + 3)
            ax.grid(color='lightgrey', linestyle=':')
            # Calculate the median and 95% Highest Probability Density (HPD) or minimum width Bayesian Confidence (BCI) interval
            hist_quant = calc_quantiles(mcmc_trace[-1][ck])
            hist_hpd = calc_hpd(mcmc_trace[-1][ck], hist_hpd_alpha)
            (hist_n, hist_bins, hist_patches) = pp.hist(mcmc_trace[-1][ck], bins=hist_num_bins, color='steelblue', histtype='stepfilled', linewidth=0.0, normed=True, zorder=2)
            pp.ylim(0.0, max(hist_n)*1.1)
            pp.axvspan(hist_hpd[0], hist_hpd[1], alpha=0.25, facecolor='darkslategray', linewidth=1.5)
            # ax.set_autoscaley_on(False)
            # pp.plot([c, c], ax.get_ylim(), color='darkslategray', linewidth=1.5)
            pp.axvline(hist_quant[50], linestyle='-', linewidth=1.5, color='darkslategray')
            pp.ylabel(C_str[ck], rotation='horizontal')
            if (i == 0):
                pp.title("Posterior (%%%2.0f HPD)" % ((1.0 - hist_hpd_alpha)*100.0))

    if DO_PYMC_BMS:
        mcmc_trace = mcmc_trace_temp

    pp.show()
