# # -*- coding: utf-8 -*-
# """
# Created on Tue Mar 31 11:49:36 2026

# @author: pazli
# """
# import numpy as np
# import matplotlib.pyplot as plt
# import scipy.stats as stats

# N = 100000
# a = 0.1
# b = 4

# def Linear(x):
#     return a*x + b

# samples = []
# samples_cv=[]
# for i in range(N):
#     u = stats.norm.rvs(loc = 0, scale = 1)
#     u_cv = Linear (u)
#     samples.append(u)
#     samples_cv.append(u_cv)

# bins = np.linspace (-2,b+2,400)
# plt.figure(1)
# plt.clf()
# plt.hist(samples,bins=bins)
# plt.hist(samples_cv,bins=bins)

#Uniform#
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 11:49:36 2026

@author: pazli
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

N = 10000000
a = 0.1
b = 4

def Uniform (x):
    return np.log(2*x+0.000000001)

samples = []
samples_cv=[]
for i in range(N):
    u = stats.norm.rvs()
    u_cv = Uniform (u)
    samples.append(u)
    samples_cv.append(u_cv)

bins = np.linspace (-2,b+2,400)
plt.figure(1)
plt.clf()
plt.hist(samples,bins=bins)
plt.hist(samples_cv,bins=100)
