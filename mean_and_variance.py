#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 16:43:52 2026

@author: finazzi
"""

import numpy as np
import matplotlib.pyplot as plt

# Parameters
n_dice = 10
n_experiments = 500

# Monte Carlo experiment
results = []

for _ in range(n_experiments):
    dice = np.random.randint(1, 7, n_dice)
    total = np.sum(dice)
    results.append(total)

# Sample mean 
def manual_mean(data):
    s = 0
    for x in data:
        s += x
    return s / len(data)

# Sample variance
def manual_variance(data):
    mu = manual_mean(data)
    s = 0
    for x in data:
        s += (x - mu) ** 2
    return s / (len(data) - 1)

mean_est = manual_mean(results)
var_est = manual_variance(results)

print("Manual mean:", mean_est)
print("Manual variance:", var_est)

mu = mean_est
sigma = np.sqrt(var_est)

# Plot
x = np.arange(n_experiments)

plt.figure(1)
plt.clf()

plt.scatter(x, results, s=10, label="Experiment outcome")

plt.axhline(mu, label="Expected value", linewidth=2, color="tab:orange")

plt.axhline(mu + sigma, linestyle="--", label="1σ", color="black")
plt.axhline(mu - sigma, linestyle="--", color="black")

plt.axhline(mu + 2*sigma, linestyle=":", color="red")
plt.axhline(mu - 2*sigma, linestyle=":", label="2σ", color="red")

plt.axhline(mu + 3*sigma, linestyle="-.", color="tab:green")
plt.axhline(mu - 3*sigma, linestyle="-.", label="3σ", color="tab:green")

plt.xlabel("Experiment number")
plt.ylabel("Sum of 10 dice")
plt.title("Sum of 10 six-sided dice")
plt.ylim(10, 60)
plt.legend()
plt.grid(True)

plt.show()

#%% chebyshev's theorem

def Count(sigma):
    cnt = 0
    for i in range(len(results)):
        if results[i] <= (mu + sigma) and results[i] > (mu - sigma):
            cnt += 1
    return cnt

total = len(results)

sigmas = 1.5
print(f"Theoretical probability expected inside {mu} ± {sigmas*sigma}: >= {1 - 1/sigmas**2}")
print(f"Measured probability inside {mu} ± {sigmas*sigma} = {Count(sigmas*sigma)/total}")





