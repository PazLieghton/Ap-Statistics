# -*- coding: utf-8 -*-
"@author: pazli"
import numpy as np
import matplotlib.pyplot as plt
###############################################################################
print("[A.1] - Generate n = 200 samples from Exp(λ0 = 2.5) DONE \n")
np.random.seed(69696969)
lambda0 = 2.5
n = 200
t = np.random.exponential(scale=1 / lambda0, size=n)

print("[A.2] - Log‑likelihood for exponential distribution. DONE \n")
sum_t = np.sum(t)
def log_likelihood(lam):
    return n * np.log(lam) - lam * sum_t

print("[A.3] - Simple plot. DONE\n")
lam_vals = np.linspace(1, 4)
ell_vals = [log_likelihood(l) for l in lam_vals]
lambda_hat = 1 / np.mean(t)
plt.figure()
plt.plot(lam_vals, ell_vals)
plt.axvline(lambda0, color='red', linestyle='--') #reference line for lambda = 2.5
plt.axvline(lambda_hat, color='green', linestyle='--')
plt.xlabel('λ'); plt.ylabel('log‑likelihood')
plt.title(label="Part A.3")
plt.show()
print("The distance between the MLE lambda hat and lambda zero is:\n")
print('Distance λ MLE − λ0:', abs(lambda_hat - lambda0))
print(lambda_hat)
print("[A.4] - Analytical result and numerical verification")
#lambda_hat = exact closed‑form MLE = 1/t_hat
#lambda_grid = best λ on the discrete grid useful tool to verify it without analytical analysis apparetly
#If the grid is fine enough, lambda_grid should be very close to lambda_hat.
lambda_hat = 1 / np.mean(t)
lambda_grid = lam_vals[np.argmax(ell_vals)]
print(f"λ = {lambda_hat:.6f} (grid: {lambda_grid:.6f}) | Is it close to 0.001?: {abs(lambda_hat - lambda_grid) < 1e-4}")

print("[A.5] Comment on the curvature of l(λ) at its maximum.")
print(f"Curvature = {-n/lambda_hat**2:.2f} | More negative = lower variance=More precise")
#%%
print("\n\n\n\n[B.1] - Fisher information for one observation: I(λ) = 1/λ², thus I(λ) = n/λ². for n obs.\n")
print("Cramer‑Rao bound: Var(λ_HAT) ≥ λ²/n.")#This is a math review
print("[B.2] Numerical Fisher information at λ")
h = 1e-5 #Steps, equation is in the problem pdf
I_hat = -(log_likelihood(lambda_hat + h) - 2*log_likelihood(lambda_hat) + log_likelihood(lambda_hat - h)) / h**2
I_analytic = n / lambda_hat**2
#Some ai use for the presentation of values like the following long print, I hate loops.
print(f"I_hat (numeric) = {I_hat:.2f}, I_analytic = {I_analytic:.2f},"f"relative error = {abs(I_hat/I_analytic - 1):.2e}")

print("[B.3–B.4] - Repeating experiment M = 2000 times Storing it and computing the sample variance.")
M = 2000
lambda_hats = 1.0 / np.mean(np.random.exponential(scale=1/lambda0, size=(M, n)), axis=1)
sample_var = np.var(lambda_hats, ddof=1)
CR_bound = lambda0**2 / n
print(f"Sample variance = {sample_var:.6f}........(CR bound = {CR_bound:.6f})")

print("[B.5] - Histogram + errors + normal overlay.")
#Plotting a normal over the histogram required robot help
plt.figure(figsize=(8, 5))
bins = 25
counts, edges = np.histogram(lambda_hats, bins=bins)
centers = (edges[:-1] + edges[1:]) / 2
width = edges[1] - edges[0]
plt.bar(centers, counts, width=width, alpha=0.7, color='skyblue', edgecolor='black', label='MLE estimates')
plt.errorbar(centers, counts, yerr=np.sqrt(counts), fmt='none', ecolor='blue', capsize=2)
#The previous if for the error in:Y
x = np.linspace(lambda0 - 4*np.sqrt(CR_bound), lambda0 + 4*np.sqrt(CR_bound), 200)
pdf = np.exp(-(x - lambda0)**2 / (2*CR_bound)) / np.sqrt(2*np.pi*CR_bound)
plt.plot(x, pdf * M * width, 'r-', lw=2, label=r'$\mathcal{N}(\lambda_0,\ \lambda_0^2/n)$')#This is notation for math cuteness
plt.xlabel(r'$\hat{\lambda}$')
plt.ylabel('Counts')
plt.title('Distribution of MLE over 2000 repetitios')
plt.legend()
plt.tight_layout()
plt.show()
print("Is the histogram compatible with a Normal distribution?\n")
print("Yes it is, specially if yu put more bins for the simulations instead of just 25, its very close")
#%%
#The following part was very difficult due to my flu last week.... thus there is heavy ai use in this item.
from scipy.optimize import minimize
from scipy.stats import gamma
print("\n\n\n[C.1] - Generate n = 150 samples from Gamma(α = 3, β = 1.5)")
alpha_true, beta_true = 3.0, 1.5
n = 1500
t = np.random.gamma(shape=alpha_true, scale=1/beta_true, size=n)
# Pre‑computing two sums that never change
S_t  = np.sum(t)        # Σt_i
S_ln = np.sum(np.log(t))      # Σln(t_i)
print("[C.2] - Write the two-parameter log-likelihood")
def neg_loglikelihood(p):
    return -np.sum(gamma.logpdf(t, p[0], scale=1/p[1]))
print("[C.3] - Find the MLE using scipy.optimize.minimize. Use the starting values")
mean_t, var_t = np.mean(t), np.var(t, ddof=1)
start_params = [mean_t**2 / var_t, mean_t / var_t]

res = minimize(neg_loglikelihood, x0=start_params, method='L-BFGS-B', bounds=[(1e-5, None), (1e-5, None)])#check documentation
a_hat, b_hat = res.x
print(f"MLE Estimates: α={a_hat:.4f}, β={b_hat:.4f}")

print("[C.4] - Approximate the observed Fisher information matrix using central finite differencest o compute all entries of the Hessian of l(α, β) at the MLE:")
h = 1e-4
def loglik(a, b): return -neg_loglikelihood([a, b])
#Computing the Hessian (Second Derivatives) matrix entries
Haa = (loglik(a_hat+h, b_hat)- 2*loglik(a_hat, b_hat) + loglik(a_hat-h, b_hat)) / h**2
Hbb = (loglik(a_hat, b_hat+h) -2*loglik(a_hat, b_hat) + loglik(a_hat, b_hat-h)) / h**2
Hab = (loglik(a_hat+h, b_hat+h) -loglik(a_hat+h, b_hat-h)-loglik(a_hat-h, b_hat+h)+loglik(a_hat-h, b_hat-h)) / (4*h**2)
Hessian = np.array([[Haa, Hab], [Hab, Hbb]])
cov_matrix = np.linalg.inv(-Hessian) # Σ = -H⁻¹ [the pdf asks MINUS Hessian]
print("[C.5] - Print Σ and compute the correlation")
corr = cov_matrix[0,1] / np.sqrt(cov_matrix[0,0] * cov_matrix[1,1]) #Beautiful
print(f"Covariance Matrix:\n{cov_matrix}")
print(f"Correlation: {corr:.4f}")
##############################################################################
#%%
##############################################################################
print("\n\n\n[Part D] - The Multivariate Normal and Confidence Ellipses")
from scipy.stats import chi2

def neg_loglik(p, data):
    return -np.sum(gamma.logpdf(data, p[0], scale=1/p[1]))
print("[D.1]---2000 repetitions....(it takes time)\n")
#2000 repetitions of part C this makes it run vert slow and needed eavy ai use, this code is INSANE
M = 2000
alpha_hats, beta_hats = np.zeros(M), np.zeros(M)
for m in range(M):
    t_m = np.random.gamma(shape=alpha_true, scale=1/beta_true, size=n)
    mu_m, v_m = np.mean(t_m), np.var(t_m, ddof=1)
    res_m = minimize(neg_loglik, x0=[mu_m**2/v_m, mu_m/v_m], args=(t_m,),
                     method='L-BFGS-B', bounds=[(1e-5,None),(1e-5,None)])
    alpha_hats[m], beta_hats[m] = res_m.x
pairs = np.column_stack([alpha_hats, beta_hats])

print("[D.2] Compute the sample covariance matrix of the M pairs and compare it entry-by-entry to Σ from Part C\n - M Pairs:", np.cov(pairs.T).round(6))
print("Theoretical Σ:\n", cov_matrix.round(6))

print("[D.3] - Produce a scatter plot of all M pairs. Overlay three nested confidence ellipses at levels 68%, 90%, and 95% using equation")
#For this I referenced the plots from class HEAVILY
vals, vecs = np.linalg.eigh(cov_matrix)
arc = np.array([np.cos(t := np.linspace(0,2*np.pi,500)), np.sin(t)])

fig, ax = plt.subplots(figsize=(8,6))
ax.scatter(alpha_hats, beta_hats, s=5, alpha=0.25, color='steelblue', label='MLE pairs')
ax.scatter(*[alpha_true,beta_true], color='black', s=80, marker='*', zorder=6, label='True θ₀')
ax.scatter(a_hat, b_hat, color='limegreen', s=60, marker='D', zorder=6, label='MLE (single run)')

for p, col in zip([0.68,0.90,0.95], ['gold','darkorange','crimson']):
    pts = np.array([a_hat,b_hat])[:,None] + vecs @ np.diag(np.sqrt(vals*chi2.ppf(p,df=2))) @ arc
    ax.plot(pts[0], pts[1], color=col, lw=2, label=f'{int(p*100)}% ellipse')

ax.set(xlabel=r'$\hat{\alpha}$', ylabel=r'$\hat{\beta}$', title='MLE Scatter + Confidence Ellipses')
ax.legend(fontsize=9); plt.tight_layout(); plt.show()

print("[D.4] - Count the fraction of the M estimates that fall inside each ellips compare to the nominal levels")
d2 = np.einsum('mi,ij,mj->m', pairs - [a_hat,b_hat], np.linalg.inv(cov_matrix), pairs - [a_hat,b_hat])
for p in [0.68,0.90,0.95]:
    print(f"[D.4] {int(p*100)}%: empirical={np.mean(d2<=chi2.ppf(p,df=2)):.4f}  nominal={p:.2f}")

print("[D.5] - Ellipses tilt positively: α and β are strongly correlated because the gamma mean is α/β. α>β ")

print("[D.6] - Additionally, produce two marginal histograms one for α and one for β")
print("with Poisson error bars and the theoretical normal curve N overlaid for each parameter")

#heavy ai use for the plotting otherwise it would have been impossible on time.
def marginal_hist(ax, estimates, true, var, label, color):
    M = len(estimates)
    counts, edges = np.histogram(estimates, bins=30)
    centers = (edges[:-1] + edges[1:]) / 2
    width = edges[1] - edges[0]

    ax.bar(centers, counts, width=width, alpha=0.7, color=color,
           edgecolor='black', label='MLE estimates')
    ax.errorbar(centers, counts, yerr=np.sqrt(counts), fmt='none',
                ecolor='navy', capsize=2, alpha=0.6)

    std = np.sqrt(var)
    x = np.linspace(true - 4*std, true + 4*std, 300)
    normal_curve = M * width * np.exp(-(x - true)**2 / (2*var)) / np.sqrt(2*np.pi*var)
    ax.plot(x, normal_curve, 'r-', lw=2, label=rf'$\mathcal{{N}}(\theta_0,\hat{{\Sigma}}_{{ii}})$')
    ax.axvline(true, color='black', ls='--', lw=1.2, label=f'True={true}')

    ax.set_xlabel(label)
    ax.set_ylabel('Counts')
    ax.legend(fontsize=9)

fig, axes = plt.subplots(1, 2, figsize=(12,5))
fig.suptitle('Marginal distributions over 2000 repetitions', fontweight='bold')

data = [
    (alpha_hats, alpha_true, cov_matrix[0,0], r'$\hat{\alpha}$', 'skyblue'),
    (beta_hats,  beta_true,  cov_matrix[1,1], r'$\hat{\beta}$',  'mediumpurple')
]
for ax, (est, true, var, lbl, col) in zip(axes, data):
    marginal_hist(ax, est, true, var, lbl, col)

plt.tight_layout()
plt.show()