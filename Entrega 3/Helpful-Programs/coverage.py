# -*- coding: utf-8 -*-
"""
Created on Mon May 18 19:12:12 2026

@author: lucas
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson, chi2

# confidence level # this code makes a plot of the confi
conf = 0.95

# alpha = 1 - cl (prob outside interval)
def interval(k, alpha=0.05):
    """Exact confidence interval for a Poisson count k."""
    
    # ppf is the percent point function (inverse of CDF)
    # For a given probability, gives k value to get that prob
    lo = chi2.ppf(alpha / 2, 2 * k) / 2 if k > 0 else 0.0
    hi = chi2.ppf(1 - alpha / 2, 2 * (k + 1)) / 2
    return lo, hi

# k_max_sigma tells the simulation to stop at these amount of sigmas
# this function calculates an interval [low, high] for each k possible
# and sees if the interval contains mu. Repeating over all values
# of k gives the probability of finding mu inside the interval
# (which is the coverage)
def coverage(mu, cl=0.95, k_max_sigma=8):
    """
    P(low(K) <= mu <= high(K)) where K ~ Poisson(mu).
    Sum over k from 0 to mu + k_max_sigma * sqrt(mu).
    """
    alpha = 1 - cl
    k_max = int(mu + k_max_sigma * max(np.sqrt(mu), 1)) + 1
    ks = np.arange(0, k_max + 1)
    probs = poisson.pmf(ks, mu)
    total = 0.0
    for k, p in zip(ks, probs):
        low, high = interval(k, alpha)
        if low <= mu <= high:
            total += p
    return total

# mu values
mu_vals = np.linspace(0.1, 100, 1000)
# coverage values
cov_vals = np.array([coverage(mu, conf) for mu in mu_vals])


plt.figure(1)
plt.clf()
plt.plot(mu_vals, cov_vals, label=f"Coverage (CL={conf})")
plt.axhline(conf, color="tab:orange", label=f"{conf}")
plt.xlabel(r"$\mu$  (true Poisson mean)", fontsize=13)
plt.ylabel("Coverage probability", fontsize=13)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.5)
plt.tight_layout()
plt.show()

#%%

""" Difference between likelihood and CIs for estimation

Likelihood asks: given k (data), which μ fits best? (point estimate)
CI asks: given k (data), which μ values are not ruled out at level α?"

"""




