# -*- coding: utf-8 -*-
"""
Programming Assignment 4 - Paz Lieghton
AP-Stats FINAL BOSS
219 plates in dataset!!!!
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import chi2, t as t_dist
############################################
ALPHA    = 0.05
ALPHABET = [chr(ord("A") + i) for i in range(26)] #Alphabet range tuple
df = pd.read_csv("plates.csv", dtype=str)
old, new = df[df["format"] == "old"], df[df["format"] == "new"]#plate_id,digits,letters,format
print(f"Plates loaded: {len(df)}  (old: {len(old)}, new: {len(new)})")
###############################################################################
#Functions for part A and C to be used
#Function A: The Chi-Square "Goodness‑of‑Fit" test
def chi2_gof(O, E, nu): #observed counts,e expected counts, nu degs of freedom
    stat = np.sum((O -E) ** 2 / E) #(observed - expected)^2/expected
    p    = stats.chi2.sf(stat, nu)   #reject the null, if p is less thn alpha reject null hypothesis
    return stat, p
#Function B: Drawing te distribution plot
#Function B: Drawing te distribution plot
def plot_null(dist, nu, observed_stats, xlabel, title, fname, two_tailed=False):
    """
    Plot the null distribution (chi2 or t student) with rejection region shaded,
    dist: scipy.stats distribution object(chi,t-student), nu:degrees of freedom
    observed_stats: list of tuples, xlabel, title, fname: plot labels and output filename
    !!!!!!two_tailed: if True, shade both tails (for t-test), tails the fox
    """
    # For two-tailed, split alpha into two halves
    half = ALPHA / 2 if two_tailed else ALPHA #Call 0.05
    crit = dist.ppf(1 - half, nu)   # critical value for rejection
    
    # Scale the x-axis relative to the critical value so the distribution curve stays visible
    xhi = crit * 2.5
    
    # Generate x values; for chi2 start at 0, for t start at -xhi
    x = np.linspace(-xhi if two_tailed else 0, xhi, 400)
    pdf = dist.pdf(x, nu)           # density at each x

    plt.figure(); plt.clf()
    plt.plot(x, pdf, "k-", linewidth=1.8, label=f"null ($\\nu$={nu:.3g})")
    
    # Shade right tail (and left tail if two-tailed)
    plt.fill_between(x[x >= crit], pdf[x >= crit], color="red", alpha=0.18,
                     label=f"rejection region ($\\alpha$={ALPHA})")
    if two_tailed:
        plt.fill_between(x[x <= -crit], pdf[x <= -crit], color="red", alpha=0.18)
        
    # Draw vertical lines or edge flags for each observed statistic
    for value, p, label, color in observed_stats:
        if value > xhi:
            #If the statistic is massively off-screen to the right
            #This will be useful for the later part of the report
            plt.axvline(xhi * 0.98, color=color, linestyle=":", linewidth=1.8)
            plt.text(xhi * 0.95, np.max(pdf) * 0.5, 
                     f"{label} stat={value:.2f}\n(Off-screen)\np={p:.3g}", 
                     color=color, fontweight='bold', ha='right',
                     bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
        elif two_tailed and value < -xhi:
            # If the statistic is massively off-screen to the left (for two-tailed t-tests)
            plt.axvline(-xhi * 0.98, color=color, linestyle=":", linewidth=1.8)
            plt.text(-xhi * 0.95, np.max(pdf) * 0.5, 
                     f"{label} stat={value:.2f}\n(Off-screen)\np={p:.3g}", 
                     color=color, fontweight='bold', ha='left',
                     bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
        else:
            # Normal placement
            plt.axvline(value, color=color, linestyle="--", linewidth=1.8,
                        label=f"{label}: stat={value:.2f}, p={p:.3g}")
            
    plt.xlabel(xlabel); plt.ylabel("Density"); plt.title(title)
    plt.legend(fontsize=9); plt.tight_layout()
    plt.savefig(fname, dpi=150); plt.show()
#%%
# ============================================================================
# PART A - Are the digits uniformly distributed?
# ============================================================================
# If plates were assigned at random, every digit 0-9 has the same probability
# p_k = 1/10. Under H0 the expected count for digit k is E_k = n/10.
# Each plate contributes 3 digit characters (treated as separate observations).
# 10 categories with 1 linear constraint (total n is fixed) -> nu = 9.

def digit_freq(group):
    digits = np.array([int(c) for s in group["digits"] for c in str(s)])
    return np.bincount(digits, minlength=10), len(digits)

O_old, n_old = digit_freq(old)
O_new, n_new = digit_freq(new)
E_old, E_new = np.full(10, n_old / 10), np.full(10, n_new / 10)
k = np.arange(10)

print(f"\n[A.1] O_old = {O_old.tolist()}")
print(f"[A.1] O_new = {O_new.tolist()}")

chi2_old, p_old = chi2_gof(O_old, E_old, nu=9)
chi2_new, p_new = chi2_gof(O_new, E_new, nu=9)

for label, stat, p in [("Old", chi2_old, p_old), ("New", chi2_new, p_new)]:
    print(f"[A.3] {label}: chi2={stat:.3f}, p={p:.4f}  ->  "
          f"{'REJECT H0' if p < ALPHA else 'fail to reject H0'}")
print("[A.3] Same conclusion?", "Yes" if (p_old < ALPHA) == (p_new < ALPHA) else "No")

# Fig A2: observed vs expected counts, both formats side by side.
# Error bars are Poisson: each O_k is a count, so its uncertainty is sqrt(O_k).
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, O, E, n, stat, p, title in [
    (axes[0], O_old, E_old, n_old, chi2_old, p_old, "Old (ABC 123)"),
    (axes[1], O_new, E_new, n_new, chi2_new, p_new, "New (AB 123 CD)"),
]:
    ax.bar(k - 0.2, O, 0.4, color="steelblue", edgecolor="black", linewidth=0.5,
           label="Observed $O_k$")
    ax.errorbar(k - 0.2, O, yerr=np.sqrt(O), fmt="none", color="black",
                capsize=3, linewidth=1, label=r"$\pm\sqrt{O_k}$")
    ax.bar(k + 0.2, E, 0.4, color="tomato", edgecolor="black", linewidth=0.5,
           alpha=0.85, label=f"Expected $n/10={n/10:.1f}$")
    ax.text(0.02, 0.96,
            f"$\\chi^2$={stat:.2f}, $p$={p:.3f}\n"
            f"{'reject' if p < ALPHA else 'fail to reject'} $H_0$",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))
    ax.set_xlabel("Digit $k$"); ax.set_ylabel("Count")
    ax.set_title(f"Part A - {title}  (n={n})"); ax.set_xticks(k)
    ax.legend(fontsize=8); ax.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout(); plt.savefig("fig_A2_digits.png", dpi=150); plt.show()

# Fig A3: where do the two chi2 statistics land on the null chi2(9)?
plot_null(chi2, 9,
          [(chi2_old, p_old, "Old", "steelblue"),
           (chi2_new, p_new, "New", "darkorange")],
          r"$\chi^2$", r"Part A - $\chi^2$ vs null $\chi^2(\nu=9)$",
          "fig_A3_chi2.png")


#%%
# ============================================================================
# PART B - Do old and new format plates differ in mean digit sum?
# ============================================================================
# Digit sum = sum of the 3 digit characters. Each sum is roughly normal by
# the CLT (sum of 3 ~ uniform-{0..9} variables) - we verify this with a
# histogram and a Q-Q plot before running the t-test. We do NOT assume equal
# variances -> Welch's t-test.

sums_old = [sum(int(c) for c in str(s)) for s in old["digits"]]
sums_new = [sum(int(c) for c in str(s)) for s in new["digits"]]
n1, n2   = len(sums_old), len(sums_new)
m1, m2   = np.mean(sums_old), np.mean(sums_new)
v1, v2   = np.var(sums_old, ddof=1), np.var(sums_new, ddof=1)

print(f"\n[B.1] Old: n={n1}, mean={m1:.3f}, var={v1:.3f}")
print(f"[B.1] New: n={n2}, mean={m2:.3f}, var={v2:.3f}")

# Fig B1: histogram with normal fit + Q-Q plot for each format.
# Two figures, one per format - simpler to read than a 2x2 grid.
xx = np.linspace(0, 27, 200)
for sums, mean, var, n, label, color in [
    (sums_old, m1, v1, n1, "Old", "steelblue"),
    (sums_new, m2, v2, n2, "New", "tomato"),
]:
    fig, (ax_h, ax_q) = plt.subplots(1, 2, figsize=(11, 4))
    ax_h.hist(sums, bins=np.arange(0, 29, 2), density=True,
              color=color, edgecolor="black", alpha=0.75)
    ax_h.plot(xx, stats.norm.pdf(xx, mean, np.sqrt(var)), "k-", linewidth=1.8,
              label=f"Normal fit  $\\mu$={mean:.2f}, $\\sigma$={np.sqrt(var):.2f}")
    ax_h.set_title(f"{label} - histogram (n={n})")
    ax_h.set_xlabel("Digit sum"); ax_h.set_ylabel("Density")
    ax_h.legend(fontsize=8); ax_h.grid(axis="y", linestyle="--", alpha=0.4)

    (osm, osr), (slope, intercept, r) = stats.probplot(sums, dist="norm")
    ax_q.scatter(osm, osr, s=18, color=color, edgecolor="black", linewidth=0.4)
    ax_q.plot(osm, slope * osm + intercept, "k-", linewidth=1.5)
    ax_q.text(0.05, 0.92, f"$R^2$={r**2:.3f}", transform=ax_q.transAxes, fontsize=9,
              bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
    ax_q.set_title(f"{label} - Q-Q plot")
    ax_q.set_xlabel("Theoretical quantiles"); ax_q.set_ylabel("Ordered digit sums")
    ax_q.grid(linestyle="--", alpha=0.4)
    plt.tight_layout(); plt.savefig(f"fig_B1_normality_{label}.png", dpi=150); plt.show()

# Welch's t-test: H0: mu_old = mu_new
t_stat, p_t = stats.ttest_ind(sums_old, sums_new, equal_var=False)
print(f"\n[B.2] Welch t-test: t={t_stat:.3f}, p={p_t:.4f}  ->  "
      f"{'REJECT H0' if p_t < ALPHA else 'fail to reject H0'}")

# Welch-Satterthwaite degrees of freedom and 95% CI for mu_old - mu_new.
# The CI is (diff) +/- t*(nu, 0.975) * se. Zero inside CI <=> fail to reject.
se   = np.sqrt(v1 / n1 + v2 / n2)
nu_w = (v1 / n1 + v2 / n2) ** 2 / (
       v1**2 / (n1**2 * (n1 - 1)) + v2**2 / (n2**2 * (n2 - 1)))
diff = m1 - m2
ci   = diff + np.array([-1, 1]) * t_dist.ppf(0.975, nu_w) * se
print(f"[B.3] 95% CI for mu_old - mu_new: ({ci[0]:.3f}, {ci[1]:.3f})  "
      f"({'contains 0' if ci[0] <= 0 <= ci[1] else 'excludes 0'}) "
      f"-> consistent with test: {(ci[0] <= 0 <= ci[1]) == (p_t >= ALPHA)}")

# Fig B2: observed t vs null t(nu_w). Two-tailed: H0 can fail in either
# direction (mu_old could be larger or smaller than mu_new).
plot_null(t_dist, nu_w, [(t_stat, p_t, "observed", "purple")],
          "$t$", "Part B - $t$ vs null (Welch's t-test)",
          "fig_B2_t.png", two_tailed=True)


#%%
# ============================================================================
# PART C - Do old and new format plates draw letters from the same distribution?
# ============================================================================
# Same chi-square machinery as Part A, but instead of a uniform reference
# we use new-format letter frequencies as the empirical null distribution.
# H0: old-format plates draw letters from the same distribution as new-format.

def letter_counts(group):
    idx  = [ord(c) - ord("A") for s in group["letters"] for c in str(s) if c.isalpha()]
    cnts = np.bincount(idx, minlength=26)
    return cnts, cnts.sum()

O_old_l, n_old_l = letter_counts(old)
O_new_l, n_new_l = letter_counts(new)
x26 = np.arange(26)

# Fig C1: normalised letter frequencies side by side.
# Poisson error on a relative frequency: sqrt(O) / n.
fig, ax = plt.subplots(figsize=(14, 5))
ax.bar(x26 - 0.2, O_old_l / n_old_l, 0.4, color="steelblue", edgecolor="black",
       label=f"Old (n={n_old_l})")
ax.errorbar(x26 - 0.2, O_old_l / n_old_l, yerr=np.sqrt(O_old_l) / n_old_l,
            fmt="none", color="black", capsize=2)
ax.bar(x26 + 0.2, O_new_l / n_new_l, 0.4, color="tomato", edgecolor="black",
       label=f"New / reference (n={n_new_l})")
ax.errorbar(x26 + 0.2, O_new_l / n_new_l, yerr=np.sqrt(O_new_l) / n_new_l,
            fmt="none", color="black", capsize=2)
ax.set_xticks(x26); ax.set_xticklabels(ALPHABET)
ax.set_xlabel("Letter"); ax.set_ylabel("Relative frequency")
ax.set_title("Part C.1 - Letter distribution: old (observed) vs new (reference)")
ax.legend(fontsize=9); ax.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout(); plt.savefig("fig_C1_letters.png", dpi=150); plt.show()

# E_l = n_old * p_hat_new_l  (scale reference frequencies to old-format total)
E_l = n_old_l * (O_new_l / n_new_l)

# The chi2 approximation requires E_l >= 5 for every bin. Letters that fail
# are pooled into an "other" bin (or dropped entirely if E=0 in both pools,
# which happens for letters excluded by regulation, e.g. I/O/Q).
keep = E_l >= 5
if (~keep).any():
    dropped = [ALPHABET[i] for i in range(26) if not keep[i]]
    p_other, o_other = E_l[~keep].sum(), O_old_l[~keep].sum()
    if p_other > 0:
        O_pooled = np.append(O_old_l[keep], o_other)
        E_pooled = np.append(E_l[keep], p_other)
        print(f"\n[C.2] Pooled {(~keep).sum()} low-count letters into 'other': {dropped}")
    else:
        O_pooled, E_pooled = O_old_l[keep], E_l[keep]
        print(f"\n[C.2] Dropped {(~keep).sum()} letters absent from both pools: {dropped}")
else:
    O_pooled, E_pooled = O_old_l, E_l

nu_c = len(O_pooled) - 1
chi2_c, p_c = chi2_gof(O_pooled, E_pooled, nu_c)
print(f"[C.2] chi2={chi2_c:.3f}, nu={nu_c}, p={p_c:.4f}  ->  "
      f"{'REJECT H0' if p_c < ALPHA else 'fail to reject H0'}")

plot_null(chi2, nu_c, [(chi2_c, p_c, "observed", "purple")],
          r"$\chi^2$", r"Part C - $\chi^2$ vs null distribution", "fig_C2_chi2.png")

# C.3: which letters drive the statistic most?
# Computed on unpooled bins for interpretation - not valid standalone tests.
terms = np.where(E_l > 0, (O_old_l - E_l) ** 2 / np.where(E_l > 0, E_l, 1), 0.0)
order = np.argsort(terms)[::-1]
print("\n[C.3] Top 5 letters by chi2 contribution:")
for i in order[:5]:
    print(f"  {ALPHABET[i]}: O={O_old_l[i]:>3d}, E={E_l[i]:6.2f}, term={terms[i]:.2f}"
          f"  ({'over' if O_old_l[i] > E_l[i] else 'under'}-represented)")

# Fig C3: signed standardised residuals. Top-5 contributors in red.
residuals = (O_old_l - E_l) / np.sqrt(np.where(E_l > 0, E_l, np.nan))
fig, ax = plt.subplots(figsize=(14, 4.5))
ax.bar(x26, np.nan_to_num(residuals), edgecolor="black", linewidth=0.5,
       color=["tomato" if i in set(order[:5]) else "steelblue" for i in range(26)])
ax.axhline(0, color="black", linewidth=1.0)
ax.axhline( 2, color="red", linestyle="--", linewidth=1.0)
ax.axhline(-2, color="red", linestyle="--", linewidth=1.0)
ax.set_xticks(x26); ax.set_xticklabels(ALPHABET)
ax.set_xlabel("Letter"); ax.set_ylabel(r"$(O_\ell - E_\ell)/\sqrt{E_\ell}$")
ax.set_title("Part C.3 - Standardised residuals  (red = top-5 contributors)")
ax.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout(); plt.savefig("fig_C3_residuals.png", dpi=150); plt.show()


#%%
# ============================================================================
# PART D - Are certain consecutive letter pairs suppressed?
# ============================================================================
# If consecutive letters were drawn independently, P(j,k) = p_hat_j * p_hat_k,
# so the expected pair count is E_jk = N * p_hat_j * p_hat_k. Systematic
# suppression of offensive combinations would push specific cells far from
# that product. We restrict to letters with >= 10 individual occurrences so
# every E_jk >= 5 (approximately), giving K letters and nu = K^2 - 1.

pairs = [(s[i], s[i+1]) for s in df["letters"].astype(str)
         for i in range(len(s) - 1) if s[i].isalpha() and s[i+1].isalpha()]
N = len(pairs)
print(f"\n[D.1] Total consecutive letter pairs: {N}")

# Observed pair matrix and marginal letter frequencies
P_jk = np.zeros((26, 26), dtype=int)
for a, b in pairs:
    P_jk[ord(a) - ord("A"), ord(b) - ord("A")] += 1

all_letters = [c for s in df["letters"].astype(str) for c in s if c.isalpha()]
cnt_all = np.bincount([ord(c) - ord("A") for c in all_letters], minlength=26)
p_hat   = cnt_all / cnt_all.sum()

# Expected pair matrix under independence: outer product of marginals
E_jk = N * np.outer(p_hat, p_hat)

# Keep only letters with >= 10 individual occurrences
idx          = np.where(cnt_all >= 10)[0]
K            = len(idx)
letters_kept = [ALPHABET[i] for i in idx]
P_sub, E_sub = P_jk[np.ix_(idx, idx)], E_jk[np.ix_(idx, idx)]

print(f"[D.3] K={K} letters retained: {letters_kept}")
print(f"[D.3] Min expected count in restricted set: {E_sub.min():.2f}")

chi2_d, p_d = chi2_gof(P_sub, E_sub, K**2 - 1)
print(f"[D.3] chi2={chi2_d:.3f}, nu={K**2-1}, p={p_d:.4f}  ->  "
      f"{'REJECT H0' if p_d < ALPHA else 'fail to reject H0'}")

plot_null(chi2, K**2 - 1, [(chi2_d, p_d, "observed", "purple")],
          r"$\chi^2$", "Part D - $\\chi^2$ vs null (pair independence)",
          "fig_D1_chi2.png")

# D.4: standardised residuals matrix. Rank all pairs, report top-5 each way.
R    = (P_sub - E_sub) / np.sqrt(E_sub)
flat = sorted([(letters_kept[i], letters_kept[j], R[i, j])
               for i in range(K) for j in range(K)], key=lambda t: t[2])
under, over = flat[:5], flat[-5:][::-1]
print("\n[D.4] Top 5 under-represented:", [f"{a}{b}:{r:.2f}" for a, b, r in under])
print("[D.4] Top 5 over-represented: ",  [f"{a}{b}:{r:.2f}" for a, b, r in over])

# Heatmap: cap colour scale at 98th percentile so one extreme outlier does
# not erase contrast everywhere else. Box and label the top-10 pairs.
vmax = max(3.0, np.percentile(np.abs(R), 98))
fig, ax = plt.subplots(figsize=(max(8, K * 0.42), max(7, K * 0.42)))
im = ax.imshow(R, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
fig.colorbar(im, ax=ax, label=r"$(P_{jk}-E_{jk})/\sqrt{E_{jk}}$", shrink=0.85)
ax.set_xticks(range(K)); ax.set_xticklabels(letters_kept, fontsize=8)
ax.set_yticks(range(K)); ax.set_yticklabels(letters_kept, fontsize=8)
ax.set_xlabel("Second letter"); ax.set_ylabel("First letter")
ax.set_title("Part D.4 - Standardised residuals of letter-pair frequencies")
for a, b, r in over + under:
    i, j = letters_kept.index(a), letters_kept.index(b)
    ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=False,
                                edgecolor="black", linewidth=1.6))
    ax.text(j, i, f"{a}{b}\n{r:+.1f}", ha="center", va="center", fontsize=6.5,
            fontweight="bold", color="white" if abs(r) > vmax * 0.6 else "black")
plt.tight_layout(); plt.savefig("fig_D2_heatmap.png", dpi=150); plt.show()