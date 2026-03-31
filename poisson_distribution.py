#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 12:27:00 2026

@author: finazzi
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

np.random.seed(0)

lambda1 = 3
lambda2 = 5
N = 10000000

# Generate samples
X = poisson.rvs(lambda1, size=N)
Y = poisson.rvs(lambda2, size=N)

Z = X + Y

lambda_sum = lambda1 + lambda2

# Range for plotting
k = np.arange(0, 25)

# Theoretical distribution
theoretical = poisson.pmf(k, lambda_sum)

# Empirical histogram
hist, bins = np.histogram(Z, bins=np.arange(-0.5, 25.5, 1), density=True)
centers = (bins[:-1] + bins[1:]) / 2

plt.figure(figsize=(7,5))
plt.bar(centers, hist, alpha=0.6, label="Monte Carlo (X+Y)")
plt.plot(k, theoretical, 'o-', label=f"Poisson($\lambda$={lambda_sum}) theory")
plt.xlabel("k")
plt.ylabel("Probability")
plt.title("Sum of two Poisson variables")
plt.legend()
plt.show()


#%% train example

lam = 8.3

k = np.arange(0, 20)

pmf = poisson.pmf(k, lam)

# Exact probability
p_exact_10 = poisson.pmf(10, lam)

# Probability more than 7
p_more_7 = 1 - poisson.cdf(7, lam)

print("Train example")
print("P(N = 10) =", p_exact_10)
print("P(N > 7)  =", p_more_7)

###########################################
# Graphical representation
###########################################

plt.figure(figsize=(7,5))

plt.bar(k, pmf, alpha=0.6, label="Poisson PMF")

# highlight k=10
plt.bar(10, poisson.pmf(10, lam), color='red', label="P(N=10)")

# highlight region N>7
mask = k > 7
plt.bar(k[mask], pmf[mask], color='orange', alpha=0.7, label="N > 7")

plt.xlabel("Number of trains")
plt.ylabel("Probability")
plt.title("Poisson distribution ($\lambda$=8.3 trains/hour)")
plt.legend()
plt.show()

#%%