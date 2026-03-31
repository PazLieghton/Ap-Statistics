#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 15:07:53 2026

@author: finazzi
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom

#%% This part to compare the distribution with generated samples

# Parameters
N = 10          # number of trials in each experiment
p = 0.3         # probability of success
N_tries = 10    # number of generated experiments

# Generate binomial samples
samples = np.random.binomial(N, p, N_tries)

# Empirical distribution
values, counts = np.unique(samples, return_counts=True)
empirical_probs = counts / N_tries

# Theoretical distribution
x = np.arange(0, N+1)
theoretical_probs = binom.pmf(x, N, p)

# Plot comparison
plt.figure(1)
plt.clf()
plt.bar(values, empirical_probs, alpha=0.6, label="Measured")
plt.plot(x, theoretical_probs, 'o', label="Theoretical")
plt.xlabel("Number of successes")
plt.ylabel("Probability")
plt.legend()
plt.title(f"Binomial Distribution (N={N}, p={p})")
plt.show()


#%% Light bulb example

p = 0.02
N = 50
N_tries = 10
N_max = 7

# Generate binomial samples
samples = np.random.binomial(N, p, N_tries)

# Empirical distribution
values, counts = np.unique(samples, return_counts=True)
empirical_probs = counts / N_tries

# Theoretical distribution
x = np.arange(0, N_max+1)
theoretical_probs = binom.pmf(x, N, p)

# Plot comparison
plt.figure(2)
plt.clf()
plt.bar(values, empirical_probs, alpha=0.6, label="Measured")
plt.plot(x, theoretical_probs, 'o', label="Theoretical")
plt.xlabel("Number of successes")
plt.ylabel("Probability")
plt.legend()
plt.title(f"Binomial Distribution (N={N}, p={p})")
plt.show()








