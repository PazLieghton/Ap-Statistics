#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 15:07:53 2026

@author: finazzi
"""

import numpy as np
import matplotlib.pyplot as plt

# Parameters
N = 10000     # total number of flips
p = 0.5       # true probability of heads

# Simulate coin flips (1 = heads, 0 = tails)
flips = np.random.binomial(1, p, N)

# Running estimate of probability of heads as P = #A/#Omega
running_prob = np.cumsum(flips) / np.arange(1, N+1)

# Plot
plt.figure(1)
plt.clf()
plt.plot(running_prob, label="Measured probability")
plt.axhline(p, linestyle="--", color="black", label="True probability (p)")
plt.xlabel("Number of flips")
plt.ylabel("Estimated P(heads)")
plt.title("Convergence of Measured Probability (Single Experiment)")
plt.legend()
plt.show()

#%%