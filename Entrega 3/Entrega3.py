# -*- coding: utf-8 -*-
"""
Assignment 3 - Paz Lieghton
The aim of this assignment is to be more comment heavy thats why its longer
than the previous ones I sent, I Realized the assignment.py did not stand alone as a
competent representation of the models. Plus the second assignement was below my standards.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.odr import ODR,Model,RealData
from scipy.stats import chi2 as chi2_dist

x = np.array([2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0])
y = np.array([2.78, 3.29, 3.29, 3.33, 3.23, 3.69, 3.46, 3.87, 3.62, 3.40, 3.99])
sigma = 0.3 #For A.3 Use Sigma 0.2 to see the chi2 going over the degees of freedom to have fun
n  = 11 #Length of the interval
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

cov = (1 / Delta) * np.array([[Sxx, -Sx],[-Sx,   S]]) #Fixed after asking
#cov = (0.5195451-> uncertainty C) 	--0.204545 DELETE THIS
#      --0.204545	(0.0818182 -> Uncertainty Slope)
# f = Slope * X + C
sigma_a1 = np.sqrt(cov[0, 0])   #uncertainty on C
sigma_a2 = np.sqrt(cov[1, 1])   #uncertainty on slope
cov_a1a2 = cov[1, 0]            #diagonal #[0,1 is also good]

#A.2
print("[PART A.1 & A.2] -- Linear Least Squares Fit")
print(f"\n a1 (C) = {a1:.3} +/- {sigma_a1:.3f}")
print(f"\n a2 (slope)   = {a2:.3f} +/- {sigma_a2:.3f}")
print(f"\n Cov(a1, a2) = {cov_a1a2:.3f} Anti correlated between slope and C")
#Anti correlated thus: if a1 increases, a2 decreases.

#A.3
#X^2 = chi2 measures total agreement between data and model thee Goodness - of - firt
# If the model is correct and errors are right, the minimum chi2 follows
#a chi2 distribution with nu = N - m degrees of freedom (Cowan 7.5) SEE:line 15
chi2    = np.sum(((y - (a1 + a2 * x)) / sigma)**2)
p_value = 1 - chi2_dist.cdf(chi2, nu) #the probability that a random Chi2 
            #variable with 9 degrees of freedom is below value observed
#Class said that if it is bigger than 0.5 then the model is correct terminal says 0.864 so :D

print(f"[A.3]\n  chi2 = {chi2:.3f}  (expected nu = {nu})")
print(f"  p-value = {p_value:.3f}")
#print("\nchi2 < nu: residue smaller than expected. Good precision p is biger than 0.05?")
#ASK this what changes when changing sigma, what is the critical point I guess
print("A larger slope forces a smaller intercept to keep the line through the data,\n") 
print("thus the parameters are anti-correlated\n")

#A.4
def linear(p, x):
    a1, a2 = p
    return a1 + a2 * x

model = Model(linear) #Using the class problem reference
data  = RealData(x, y, sy=sigma)
odr_fit = ODR(data, model, beta0=[1.0, 1.0])
odr_fit.set_job(fit_type=2)#fit_type=2 - Docuemntation ordinary least squares (y only)
result  = odr_fit.run()

a1_odr = result.beta[0]   #Odr C
a2_odr = result.beta[1]   #Odr Slope
#ODR normalizes cov by sigma^2 we discard that by multiplying to get true values
cov_odr = result.cov_beta * sigma**2 #<-- CONFUSING documentation thing we have to multiply sigma ** 2

print("\n [A.4] ODR")
print(f"\na1 = {a1_odr:.6f}  (analytic: {a1:.6f}) :D")
print(f"\na2 ={a2_odr:.6f}  (analytic: {a2:.6f}) :D")
print(f"\nCov(a1,a2) ODR = {cov_odr[0,1]:.6f}  (analytic: {cov_a1a2:.6f}) :D")

#A.5
xa = xa_vals = np.linspace(0, 5, 300)
plt.figure(1)
plt.clf()
plt.errorbar(x, y, yerr=sigma, fmt=".k", capsize=3, label="Data")
plt.plot(xa, a1 + a2 * xa, label=f"Best Fit Line Extened: y = {a1:.3f} + {a2:.3f} x")
plt.xlabel("x")
plt.ylabel("y")
plt.title("[A.5] Linear fit")
plt.legend(fontsize=11)
plt.grid(True)
plt.show()
#%%
#In matrix form: Var(ya) = |1 xa| * cov * |1 | #DELETE BEFORE SENDING
#                                         |xa|
#[B.1] Variance of predicted value at every xa
#Var(ya) = Var(a1) + xa²·Var(a2) + 2·xa·Cov(a1,a2)
#cov[0,0] = Var(a1), cov[1,1] = Var(a2), cov_a1a2 = Cov(a1,a2)
var_full = cov[0,0] + xa_vals**2 * cov[1,1] +2 * xa_vals * cov_a1a2 #correct way
var_diag = cov[0,0] + xa_vals**2 * cov[1,1]   #incorrect (ignores Cov)

#[B.2]. Where is Var(ya) smallest?
#I minimise by setting d/dxa = 0  =>  xa_min = -Cov(a1,a2) /sigma_a2^2 = x_bar
x_bar  = np.mean(x)
xa_min = -cov_a1a2 / cov[1, 1] #diagonal

print("\n\n[PART B.2]  Analytical minimum of Var(ya)")
print(f" x_bar  = {x_bar:.3f}")
print(f" xa_min = {xa_min:.3f} X_bar and x_min are the same\n\n\n\n")

plt.figure(2)
plt.plot(xa_vals, var_full,       label="Var(ya) full covariance")
plt.plot(xa_vals, var_diag, "--", label="Var(ya) diagonal only (wrong)")
plt.axvline(xa_min, color="green", linestyle=":", label=f"Minimum at x = {xa_min:.1f}")
plt.xlabel("xa"); plt.ylabel("Var(ya)")
plt.title("[B.2] Variance of predicted value")
plt.legend()
plt.grid(True)
plt.show()

#[B.3]. Confidence bands: fit line(ya) +/- sigma(ya), correct vs incorrect
#About 68% of variance which is standar deviation
#band_full: correct uses full covariance including the negative Cov term
#band_diag: wrong  ignores Cov, always wider than necessary
y_bar  = np.mean(y)
ya_fit = a1 + a2 * xa_vals
band_full = np.sqrt(var_full) #correct way  fully covariance small band
band_diag = np.sqrt(var_diag)#wrong, wide band 
#By the way thank you so much fot teaching this stuff in the mathematical background
#I had to do plots of this sort for confidence belts for Physics 4
plt.figure(3)
plt.errorbar(x, y, yerr=sigma, fmt=".k", capsize=3, label="Data")
plt.plot(xa_vals, ya_fit, label=f"Best fit: y = {a1:.3f} + {a2:.3f} x")
plt.axhline(y_bar, color="gray", linestyle="--", label=f"y_bar = {y_bar:.3f}")
plt.fill_between(xa_vals, ya_fit -band_full, ya_fit+ band_full, alpha=0.7, label="Full cov (Correct)")
plt.fill_between(xa_vals, ya_fit-band_diag, ya_fit +band_diag, alpha=0.5,  label="Diagonal only (Incrrect)")
plt.xlabel("x")
plt.ylabel("y")
plt.title("[B.3] Confidence bands")
plt.legend(); plt.grid(True)
plt.show()
#%%
#Generated 1000 synthetic datasets, fit each, record chi2
#If model is  correct chi 2 should follow chi2(nu=9)

N_MC  = 1000
#Each column is one synthetic experiment: y_mc shape is (11, 1000)
y_mc  = np.random.normal(a1 + a2 * x[:, None], sigma, size=(n, N_MC))

#For the sake of brevity I reuse the functions o f [A.1]
Sy_mc  = w * y_mc.sum(axis=0)
Sxy_mc = w * (x[:, None] * y_mc).sum(axis=0)

a1_mc  = (Sxx * Sy_mc - Sx  * Sxy_mc) / Delta
a2_mc  = (S   * Sxy_mc - Sx * Sy_mc)  / Delta

chi2_mc = np.sum(((y_mc - (a1_mc + a2_mc * x[:, None])) / sigma)**2, axis=0)

#[C.1].Normalised histogram with Poisson errors vs theoretical pdf
counts, edges = np.histogram(chi2_mc, bins=100)
bw  = edges[1] - edges[0]
bc  = 0.5 * (edges[:-1] + edges[1:])
plt.figure(4)
plt.bar(bc, counts/(N_MC*bw), width=bw, alpha=0.6, label="MC chi2")
plt.errorbar(bc, counts/(N_MC*bw), yerr=np.sqrt(counts)/(N_MC*bw), fmt="none", color="k", capsize=3)
plt.plot(np.linspace(0,30,300), chi2_dist.pdf(np.linspace(0,30,300), nu), "r", label=f"chi2(nu={nu})")
plt.xlabel("chi2"); plt.ylabel("Density")
plt.title("[C.1] MC chi2 vs theoretical"); plt.legend(); plt.grid(True)
plt.show()

print(f"[C] MC mean = {np.mean(chi2_mc):.3f} (expected {nu}), std = {np.std(chi2_mc):.3f} (expected {np.sqrt(2*nu):.3f})")
print("Wrong model -> histogram shifts right")

#Comment on what  Does the simulation confirm that the chi-square statistic follows the expected distribution
#################################################################################
#%%
# C.2 Bonus: Wrong model Monte Carlo we gottoreuse the dataset
#Instead of fitting a line (correct), we fit a constant  y = c  (wrong model).
#A constant ignores the slope entirely, so there is a contribtion
#contribution from a2*(xi - x_bar)
#The chi2 of the wrong fit should NOT follow chi2(nu_wrong = 10)
#it should be shifted to the right by the non-centrality 
#lambda = a2^2 * sum((xi-x_bar)^2) / sigma^2

nu_wrong = n - 1   #n = 11 

c_mc       = y_mc.mean(axis=0)                             #shape (N_MC,) one c per dataset
chi2_wrong = np.sum(((y_mc - c_mc) / sigma)**2, axis=0)    # shape (N_MC,)

#Analytical prediction for the mean of chi2 wrong (non central chi2)
#E[chi2_wrong] = nu_wrong + lambda
#lambda = a2^2 * sum((xi - x_bar)^2) / sigma^2 -> See Report
lam_theory = a2**2 * np.sum((x - np.mean(x))**2) / sigma**2
mean_theory = nu_wrong + lam_theory

print("\n[C.2 Bonus] Wrong model (constant fit)")
print(f"MC mean chi2 = {np.mean(chi2_wrong):.2f}  (theory: nu + lambda = {mean_theory:.2f})")
print(f"MC std  chi2 ={np.std(chi2_wrong):.2f}")
print(f"Correct model mean = {np.mean(chi2_mc):.2f} (expected nu = {nu})")

#Plot [3.2]: two histograms side by side + their theoretical reference curves
#Bins are going to be different than the pdf
counts_w, edges_w = np.histogram(chi2_wrong, bins=50, range=(0, 55))
bw_w = edges_w[1] - edges_w[0]
bc_w = 0.5 * (edges_w[:-1] + edges_w[1:])

xr = np.linspace(0, 55, 500)
plt.figure(5)
#The following plos is a huge conundrum of graphing magic and i hat to get aid for this.
#Reminds me of doing large scale microcontroller control without an rtos.
#Wrong model histogram
plt.bar(bc_w, counts_w / (N_MC * bw_w), width=bw_w, alpha=0.55,
        color="steelblue", label="MC chi2 Wrong model (constant fit)")
plt.errorbar(bc_w, counts_w / (N_MC * bw_w),
             yerr=np.sqrt(counts_w) / (N_MC * bw_w),
             fmt="none", color="k", capsize=3)
#Correct model histogram (replotted for comparison from previous)
counts_c, edges_c = np.histogram(chi2_mc, bins=50, range=(0, 30))
bw_c = edges_c[1] - edges_c[0]
bc_c = 0.5 * (edges_c[:-1] + edges_c[1:])
plt.bar(bc_c, counts_c / (N_MC * bw_c), width=bw_c, alpha=0.45,
        color="orange", label="MC chi2 Correct model (linear fit)")

#Theoretical chi2(nu_wrong) what the wrong histogram WOULD look like if the model were correct
#Honestly unnecessary to plot this but it looks cool.
plt.plot(xr, chi2_dist.pdf(xr, nu_wrong), "r--",
         label=rf"$\chi^2(\nu={nu_wrong})$ Reference for constant fit")

#Theoretical chi2(nu) for the correct linear model
plt.plot(xr, chi2_dist.pdf(xr, nu), "darkorange", linestyle=":",
         label=rf"$\chi^2(\nu={nu})$ Correct linear model")
#The means are both marked
plt.axvline(np.mean(chi2_wrong), color="steelblue", linestyle="--", alpha=0.8,
            label=f"Wrong model mean = {np.mean(chi2_wrong):.1f}")
plt.axvline(nu_wrong, color="red", linestyle="--", alpha=0.6,
            label=f"Expected mean if model correct = {nu_wrong}")

plt.xlabel(r"$\chi^2$")
plt.ylabel("Density")
plt.title("[C.2 Bonus] Wrong model: chi2 shifts right relative to chi2(10)")
plt.legend(fontsize=9)
plt.grid(True)
plt.tight_layout()
plt.savefig("fig_c2.png", dpi=150) #This is for the Latex Formatting DELETE
plt.show()