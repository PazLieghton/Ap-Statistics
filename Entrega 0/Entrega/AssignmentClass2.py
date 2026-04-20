# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 17:56:42 2026

@author: lucas -> Paz Lieghton.
"""

import numpy as np
import matplotlib.pyplot as plt

# 12 students study for a test and each of them record the 
# following parameters

# hours studied
x = [2, 4, 6, 5, 7, 3, 8, 5, 6, 4, 7, 3]

# problems solved in those hours
y = [8, 15, 22, 18, 27, 10, 30, 17, 21, 13, 25, 11]

# hours slept
z = [8, 7, 5, 6, 4, 7, 4, 6, 5, 7, 4, 8]

# Ex: The first student solves 8 problems in 2 hours and sleeps 8 hours


#%% 

# 1. Do you expect any of these variables to have correlation? With
# what sign?
#I expect more hours studied -> More problems slved + Correlation (X->Y) (+)
#I expect more studied hours -> Less sleep - Correlation (X-Z) (-)
#I expect more hours slept -> Less problems solved - Correlation (Z->Y)  (-)

# 2. Calculate the covariance matrix and interpret the results
data = np.vstack([x, y, z])#Docuentacion
#Variable explorer
cov_matrix = np.cov(data)#Covariance
corr_matrix = np.corrcoef(data)#Correlation
#Discussion done in class: [+ study,- sleep],[+ study, + problems solved],[+ problems, -sleep]

# 3. Plot all variables against each other. Does the result make sense?

#Done post class. yeah it does, the slopes for (X->Y is psitive) as expected
#Same for the negative slope of (X->Z) and (Y->Z|)
####################################################################################
#####################################################################################
plt.figure(figsize=(12, 4))
plt.subplot(131); plt.scatter(x, y); plt.xlabel('Study hours(X)'); plt.ylabel('Problems solved(Y)')
plt.subplot(132); plt.scatter(x, z); plt.xlabel('Study hours(X)'); plt.ylabel('Hours slept(Z)')
plt.subplot(133); plt.scatter(y, z); plt.xlabel('Problems solved(Y)'); plt.ylabel('Hours slept(Z)')
plt.show()

