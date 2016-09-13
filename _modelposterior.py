#!/usr/bin/env python

# From: https://github.com/pymc-devs/pymc/blob/2.2/pymc/sandbox/MultiModelInference/ModelPosterior.py
#
# See also: https://github.com/pymc-devs/pymc/wiki/BayesFactor
#           http://stronginference.com/post/bayes-factors-pymc
#           http://healthyalgorithms.com/2009/08/25/mcmc-in-python-pymc-for-bayesian-model-selection/
#
# Anand Patil says:
# "MultiModelInference is in the sandbox because its really hard to guarantee reasonable performance in most practical
#  cases. Estimates of Bayes factors by Monte Carlo integration just get really bad as model complexity increases. The
#  mode of failure is that one sampled parameter set ends up with a vastly higher likelihood than any other. This is the
#  main reason for the popularity of alternatives to Bayes factors, such as DIC."

# March 30 07 AP: This can work with any Model subclass, not just Sampler.

from pymc import *
from numpy import mean, exp, Inf, zeros
from tqdm import *

from pymc import six
#xrange = six.moves.xrange


def sample_likelihood(model, iter, verbose=False):
    """
    Returns iter samples of

    log p(data|self.stochastics, self) * sum(self.potentials)
        and
    sum(self.potentials),

    where 'sample' means that self.stochastics are drawn from their joint prior and then these
    quantities are evaluated. See documentation.

    Exponentiating, averaging and dividing the return values gives an estimate of the model
    likelihood, p(data|self).
    """

    model._generations = find_generations(model)

    loglikes = zeros(iter)

    if len (model.potentials) > 0:
        logpots = zeros(iter)
    else:
        logpots = zeros(1, dtype=float)

    try:
        for i in trange(iter):

            model.draw_from_prior()

            for datum in model.observed_stochastics | model.potentials:
                loglikes[i] += datum.logp
            if verbose:
                print "Loglike = %.6e" % loglikes[i]

            if len (model.potentials) > 0:
                for pot in model.potentials:
                    logpots[i] += pot.logp

    except KeyboardInterrupt:
        print 'Halted at sample ', i, ' of ', iter

    return loglikes[:i], logpots[:i]

def weight(models, iter, priors = None, verbose=False):
    """
    posteriors, loglikes, logpots = weight(models, iter, priors = None)

    models is a list of Models, iter is the number of samples to use, and
    priors is a dictionary of prior weights keyed by model.

    Example:

    M1 = Model(model_1)
    M2 = Model(model_2)
    p, ll, lp = weight(models = [M1,M2], iter = 100000, priors = {M1: .8, M2: .2})

    Returns a dictionary keyed by model of the model posterior probabilities,
    and two similar dictionaries containing the log-likelihoods and log-potentials
    sampled over the course of the estimation.

    WARNING: the weight() function will usually not work well unless
    the dimension of the parameter space is small. Please do not trust
    its output unless you check that it has weighted a large number of
    samples more or less evenly.
    """

    # TODO: Need to attach a standard error to the return values.
    loglikes = {}
    logpots = {}
    i=0
    for model in models:
        if verbose:
            print 'Model ', i
        loglikes[model], logpots[model] = sample_likelihood(model, iter, verbose)
        i+=1

    # Find max log-likelihood for regularization purposes
    max_loglike = -Inf
    max_logpot = -Inf
    for model in models:
        max_loglike = max((max_loglike,loglikes[model].max()))
        max_logpot = max((max_logpot,logpots[model].max()))

    posteriors = {}
    sumpost = 0
    for model in models:

        # Regularize
        loglikes[model] -= max_loglike
        logpots[model] -= max_logpot

        # Exponentiate and average
        posteriors[model] = mean(exp(loglikes[model])) / mean(exp(logpots[model]))

        # Multiply in priors
        if priors is not None:
            posteriors[model] *= priors[model]

        # Count up normalizing constant
        sumpost += posteriors[model]

    # Normalize
    for model in models:
        posteriors[model] /= sumpost

    return posteriors, loglikes, logpots

# From: https://github.com/pymc-devs/pymc/blob/2.2/pymc/utils.py

def find_generations(container, with_data = False):
    """
    A generation is the set of stochastic variables that only has parents in
    previous generations.
    """

    generations = []

    # Find root generation
    generations.append(set())
    all_children = set()
    if with_data:
        stochastics_to_iterate = container.stochastics | container.observed_stochastics
    else:
        stochastics_to_iterate = container.stochastics
    for s in stochastics_to_iterate:
        all_children.update(s.extended_children & stochastics_to_iterate)
    generations[0] = stochastics_to_iterate - all_children

    # Find subsequent _generations
    children_remaining = True
    gen_num = 0
    while children_remaining:
        gen_num += 1

        # Find children of last generation
        generations.append(set())
        for s in generations[gen_num-1]:
            generations[gen_num].update(s.extended_children & stochastics_to_iterate)

        # Take away stochastics that have parents in the current generation.
        thisgen_children = set()
        for s in generations[gen_num]:
            thisgen_children.update(s.extended_children & stochastics_to_iterate)
        generations[gen_num] -= thisgen_children

        # Stop when no subsequent _generations remain
        if len(thisgen_children) == 0:
            children_remaining = False
    return generations
