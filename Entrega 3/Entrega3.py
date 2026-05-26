# -*- coding: utf-8 -*-
"""
Assignment 3 Paz Lieghton
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.odr import ODR, Model, RealData
from scipy.stats import chi2 as chi2_dist


x = np.array([2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0])
y = np.array([2.78, 3.29, 3.29, 3.33, 3.23, 3.69, 3.46, 3.87, 3.62, 3.40, 3.99])
sigma = 0.3 #For A.3 Use Sigma 0.2 to see the chi2 going over the degees of freedom
n  = len(x) #11
nu = n - 2  # degrees of freedom 2: Parametters a1 and a2 Its (9 = nu)

# Each yi is a Gaussian random variable centered on the true value A(xi)
# This is maximized by finding the values of the parameters 8 that minimize the quantity
#chi2 = sum_i(yi - A(xi))^2 / sigma_i^2 (Cowan 7.3)
#I will reference during the assignment the linear model as: A(x) = a1 + a2*x 

#A.1 Functions in the assignemt
w   = 1.0 / sigma**2   #Weights of every point
S   = n * w            #Weeight sum
Sx  = w * np.sum(x)    #Weights of poins * sum of (xi)
Sy  = w * np.sum(y)    #Weights of poins * sum of (yi)
Sxx = w * np.sum(x**2) 
Sxy = w * np.sum(x*y)  

Delta = S * Sxx - Sx**2  #Determinant of 2x2
#Thank you cowann... Math DELETE LATER
#S*a1  + Sx*a2  = Sy     
#Sx*a1 + Sxx*a2 = Sxy 
#In the matrix form
#|S   Sx | |a1|   |Sy |
#|Sx  Sxx| |a2| = |Sxy|
#Thus we can calculate a1 ===> C and a2 -> Slope/Modulo
#We apply determinant
#det = Sxx*Sy - Sx*Sxy =>  a1 = (Sxx*Sy - Sx*Sxy) / Delta
# For a2: replace column 2 with [Sy, Sxy]:
# det = S*Sxy - Sx*Sy     -->  a2 = (S*Sxy - Sx*Sy) / Delta

a1 = (Sxx * Sy - Sx * Sxy) / Delta   #C
a2 = (S   * Sxy- Sx * Sy ) / Delta   #Slope 

cov = (sigma**2 / Delta) * np.array([[Sxx, -Sx],
                                    [-Sx,   S]]) #PDF
#cov = (0.0467591-> uncertainty C) 	-0.0184091 DELETE THIS
#      -0.0184091	(0.00736364 -> Uncertainty Slope)

sigma_a1 = np.sqrt(cov[0, 0])   #uncertainty on C
sigma_a2 = np.sqrt(cov[1, 1])   #uncertainty on slope
cov_a1a2 = cov[1, 0]            #diagonal #[0,1 is also good]

#A.2
print("[PART A.1 & A.2] -- Linear Least Squares Fit")
print(f"  a1 (C) = {a1:.3} +/- {sigma_a1:.3f}")
print(f"  a2 (slope)   = {a2:.3f} +/- {sigma_a2:.3f}")
print(f"  Cov(a1, a2) = {cov_a1a2:.3f} Anti correlated between slope and C")

#A.3
#X^2 = chi2 measures total agreement between data and model thee Goodness - of - firt
# If the model is correct and errors are right, the minimum chi2 follows
#a chi2 distribution with nu = N - m degrees of freedom (Cowan 7.5) SEE:line 15
chi2    = np.sum(((y - (a1 + a2 * x)) / sigma)**2)
p_value = 1 - chi2_dist.cdf(chi2, nu) #the probability that a random Chi2 
            #variable with 9 degrees of freedom is below value observed

print(f"[A.3]\n  chi2 = {chi2:.3f}  (expected nu = {nu})")
print(f"  p-value = {p_value:.3f}")
print("\nchi2 < nu: residue smaller than expected. Good precision")
print("If i put sigma 0.2 it is chi2 Bigger than nu -> Sigma is too big??.")
#ASK this what changes when changing sigma, what is the critical point I guess

#A.4
def linear(p, x):
    a1, a2 = p
    return a1 + a2 * x

model = Model(linear) #Using the class problem reference
data  = RealData(x, y, sy=sigma)
odr_fit = ODR(data, model, beta0=[1.0, 1.0])
odr_fit.set_job(fit_type=2)   #fit_type=2 - Docuemntation ordinary least squares (y only)
result  = odr_fit.run()

a1_odr = result.beta[0]   #Odr C
a2_odr = result.beta[1]   #Odr Slope
#ODR normalizes cov by sigma^2 we discard that by multiplying to get true values
cov_odr = result.cov_beta * sigma**2 #<-- CONFUSING documentation thing


print("\n [A.4] ODR")
print(f"a1 = {a1_odr:.6f}  (analytic: {a1:.6f}) :D")
print(f"a2 ={a2_odr:.6f}  (analytic: {a2:.6f}) :D")
print(f"Cov(a1,a2) ODR = {cov_odr[0,1]:.6f}  (analytic: {cov_a1a2:.6f}) :D")

#A.5
xa = np.linspace(0, 5, 300)
plt.figure(1)
plt.clf()
plt.errorbar(x, y, yerr=sigma, fmt=".k", capsize=3, label="Data")
plt.plot(xa, a1 + a2 * xa, label=f"Best Fit Line Extened: y = {a1:.3f} + {a2:.3f} x")
plt.xlabel("x", fontsize=12)
plt.ylabel("y", fontsize=12)
plt.title("[A.5] Linear fit", fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.4)
plt.show()
################################################################################'
# In matrix form: Var(ya) = |1 xa| * cov * |1 | #DELETE
#                                          |xa|
xa_vals = np.linspace(0, 5, 300)
#Var(ya) = sigma_a1^2 + xa^2 * sigma_a2^2 + 2*xa*Cov(a1,a2) -> Error propagation
var_full = np.zeros(len(xa_vals))
for i, xa in enumerate(xa_vals):
    var_full[i] = cov[0,0] + xa**2 * cov[1,1] + 2 * xa * cov_a1a2

#Diagonal only: same formula but pretending Cov(a1,a2) = 0 for no variation

var_diag = np.zeros(len(xa_vals))

for i, xa in enumerate(xa_vals):
    var_diag[i] = cov[0,0] + xa**2 * cov[1,1]


    
# B.2
# Where is Var(ya) smallest? Take d/dxa of the full variance and set to zero:
# d/dxa [sigma_a1^2 + xa^2*sigma_a2^2 + 2*xa*Cov] = 0
# 2*xa*sigma_a2^2 + 2*Cov(a1,a2) = 0
# xa_min = -Cov(a1,a2) / sigma_a2^2
# This should equal x_bar — the fit is most constrained at the center of the data

x_bar  = np.mean(x)
xa_min = -cov_a1a2 / cov[1, 1]   # cov[1,1] = sigma_a2^2

print("[PART B.2] -- Minimum of Var(ya)")
print(f"  x_bar  = {x_bar:.4f}")
print(f"  xa_min = {xa_min:.4f}  (must match x_bar)")
print(f"  sigma(ya) at minimum = {np.sqrt(var_full.min()):.4f}")

plt.figure(2)
plt.clf()
plt.plot(xa_vals, var_full,       label="Var(ya) full covariance")
plt.plot(xa_vals, var_diag, "--", label="Var(ya) diagonal only (wrong)")
# Mark the minimum — should land exactly on x_bar = 2.5
plt.axvline(xa_min, color="tab:green", linestyle=":", label=f"Minimum at x_bar = {xa_min:.1f}")
plt.xlabel("xa", fontsize=13)
plt.ylabel("Var(ya)", fontsize=13)
plt.title("Part B.2 — Variance of predicted value", fontsize=13)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.show()

# B.3
# The confidence band at each xa is just the fit line +/- one sigma of ya
# sigma(ya) = sqrt(Var(ya)), so the band is:
#   upper = (a1 + a2*xa) + sqrt(Var(ya))
#   lower = (a1 + a2*xa) - sqrt(Var(ya))
# We plot both bands: the correct one (full cov) and the wrong one (diagonal)
# The wrong band is always wider because it ignores the negative Cov term

y_bar    = np.mean(y)
ya_fit   = a1 + a2 * xa_vals   # the best-fit line at every xa
band_full = np.sqrt(var_full)   # correct sigma(ya)
band_diag = np.sqrt(var_diag)   # wrong sigma(ya), always >= band_full

plt.figure(3)
plt.clf()
plt.errorbar(x, y, yerr=sigma, fmt=".k", capsize=3, label="Data")
plt.plot(xa_vals, ya_fit, label=f"Best fit: y = {a1:.3f} + {a2:.3f} x")
plt.axhline(y_bar, color="gray", linestyle="--", linewidth=1, label=f"y_bar = {y_bar:.3f}")
plt.fill_between(xa_vals, ya_fit - band_full, ya_fit + band_full,
                 alpha=0.25, label="Band full cov (correct)")
plt.fill_between(xa_vals, ya_fit - band_diag, ya_fit + band_diag,
                 alpha=0.2,  label="Band diagonal only (wrong)")
plt.xlabel("x", fontsize=13)
plt.ylabel("y", fontsize=13)
plt.title("Part B.3 — Confidence bands", fontsize=13)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.show()