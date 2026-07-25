# -*- coding: utf-8 -*-
"""
Programming Assignment 4 - Paz Lieghton
AP-Stats FINAL BOSS
219 plates in dataset!!!!
  Part A - chi-square goodness-of-fit: are digits 0-9 uniformly distributed?
  Part B - Welch's t-test: do old/new plates have the same mean digit sum?
  Part C - chi-square goodness-of-fit (empirical reference): do old/new
           plates draw their letters from the same distribution?
  Part D - chi-square test of independence: are consecutive letter pairs
           suppressed relative to what independence would predict?
  Part E - KS test: can we estimate the registration year of a plate from
           its letter prefix, using Argentine plate sequence history?
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import chi2, t as t_dist

############################################
ALPHA    = 0.05
ALPHABET = [chr(ord("A") + i) for i in range(26)]  # Alphabet range tuple

df = pd.read_csv("plates.csv", dtype=str)
old, new = df[df["format"] == "old"], df[df["format"] == "new"]  # plate_id,digits,letters,format
print(f"Plates loaded: {len(df)}  (old: {len(old)}, new: {len(new)})")

###############################################################################
# SHARED HELPER FUNCTIONS
# chi2_gof is the workhorse: both A and C reduce to summing (O-E)^2/E over
# bins. The only thing that changes between parts is how E is constructed.
# plot_null draws the null pdf, shades the rejection tail, and drops a
# vertical line where the observed statistic landed. It works for any
# scipy distribution that takes a single shape parameter (chi2, t, etc.).
# The off-screen box catches the rare case where the statistic is so extreme
# it would fall outside the visible plot area - useful for Part D where
# chi2 can be enormous with many pairs and a large K.
###############################################################################

def chi2_gof(O, E, nu):
    # Pearson chi-square: sum of squared standardised residuals.
    # sf (survival function) = 1 - CDF, which is exactly P(chi2_nu >= stat).
    stat = np.sum((O - E) ** 2 / E)
    p    = stats.chi2.sf(stat, nu)
    return stat, p


def plot_null(dist, nu, observed_stats, xlabel, title, fname, two_tailed=False):
    # For two-tailed tests (t-test) alpha is split equally across both tails.
    # crit * 2.5 gives enough room to see the full bell/chi2 curve.
    half = ALPHA / 2 if two_tailed else ALPHA
    crit = dist.ppf(1 - half, nu)
    xhi  = crit * 2.5
    x    = np.linspace(-xhi if two_tailed else 0, xhi, 400)
    pdf  = dist.pdf(x, nu)

    plt.figure(); plt.clf()
    plt.plot(x, pdf, "k-", linewidth=1.8, label=f"null ($\\nu$={nu:.3g})")
    # Shade the rejection region - right tail always, left tail if two-tailed
    plt.fill_between(x[x >= crit], pdf[x >= crit], color="red", alpha=0.18,
                     label=f"rejection region ($\\alpha$={ALPHA})")
    if two_tailed:
        plt.fill_between(x[x <= -crit], pdf[x <= -crit], color="red", alpha=0.18)

    for value, p, label, color in observed_stats:
        if value > xhi:
            # Stat is off the right edge of the plot: draw a dotted line at the
            # border and put a text box with the exact value so it's not lost.
            plt.axvline(xhi * 0.98, color=color, linestyle=":", linewidth=1.8)
            plt.text(xhi * 0.95, np.max(pdf) * 0.5,
                     f"{label} stat={value:.2f}\n(Off-screen)\np={p:.3g}",
                     color=color, fontweight="bold", ha="right",
                     bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
        else:
            plt.axvline(value, color=color, linestyle="--", linewidth=1.8,
                        label=f"{label}: stat={value:.2f}, p={p:.3g}")

    plt.xlabel(xlabel); plt.ylabel("Density"); plt.title(title)
    plt.legend(fontsize=9); plt.tight_layout()
    plt.savefig(fname, dpi=150); plt.show()


#%%
###############################################################################
# PART A - Are the digits uniformly distributed?
# H0: every digit 0-9 is equally likely (p_k = 1/10 for all k).
# Each plate gives us 3 digit characters, each treated as an independent
# observation. Expected count per digit: E_k = n/10.
# 10 bins, 1 constraint (n fixed) -> nu = 9 degrees of freedom.
# We test old and new formats separately because they span different eras
# and could have been issued under different administrative policies.
###############################################################################

def digit_freq(group):
    # Flatten all digit strings into individual characters and count 0-9.
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

# Fig A2: side-by-side bar chart of O_k vs E_k for each format.
# Poisson error bars: a count O_k has uncertainty sqrt(O_k) by definition.
# E_k is a fixed number (n/10), so it carries no statistical error bar.
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
###############################################################################
# PART B - Do old and new format plates differ in mean digit sum?
# H0: mu_old = mu_new (same population mean digit sum).
# Digit sum = sum of the 3 digit characters (e.g. "473" -> 4+7+3 = 14).
# By the CLT, the sum of 3 roughly-uniform variables is approximately normal.
# We DON'T assume equal variances between old and new -> Welch's t-test,
# which estimates its own effective degrees of freedom via Welch-Satterthwaite.
# We verify normality with a histogram + Q-Q plot before trusting the test.
###############################################################################

sums_old = [sum(int(c) for c in str(s)) for s in old["digits"]]
sums_new = [sum(int(c) for c in str(s)) for s in new["digits"]]
n1, n2 = len(sums_old), len(sums_new)
m1, m2 = np.mean(sums_old), np.mean(sums_new)
v1, v2 = np.var(sums_old, ddof=1), np.var(sums_new, ddof=1)  # ddof=1: unbiased sample variance

print(f"\n[B.1] Old: n={n1}, mean={m1:.3f}, var={v1:.3f}")
print(f"[B.1] New: n={n2}, mean={m2:.3f}, var={v2:.3f}")

# Fig B1: one figure per format, histogram + Q-Q side by side.
# If the histogram tracks the normal curve and the Q-Q hugs the line (R^2 ~ 1),
# normality holds well enough for the t-test to be valid.
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

    # probplot returns (theoretical quantiles, ordered sample values) and
    # the (slope, intercept, r) of the best-fit line through the Q-Q cloud.
    (osm, osr), (slope, intercept, r) = stats.probplot(sums, dist="norm")
    ax_q.scatter(osm, osr, s=18, color=color, edgecolor="black", linewidth=0.4)
    ax_q.plot(osm, slope * osm + intercept, "k-", linewidth=1.5)
    ax_q.text(0.05, 0.92, f"$R^2$={r**2:.3f}", transform=ax_q.transAxes, fontsize=9,
              bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
    ax_q.set_title(f"{label} - Q-Q plot")
    ax_q.set_xlabel("Theoretical quantiles"); ax_q.set_ylabel("Ordered digit sums")
    ax_q.grid(linestyle="--", alpha=0.4)
    plt.tight_layout(); plt.savefig(f"fig_B1_normality_{label}.png", dpi=150); plt.show()

# Welch's t-test: equal_var=False tells scipy NOT to pool the variances.
t_stat, p_t = stats.ttest_ind(sums_old, sums_new, equal_var=False)
print(f"\n[B.2] Welch t-test: t={t_stat:.3f}, p={p_t:.4f}  ->  "
      f"{'REJECT H0' if p_t < ALPHA else 'fail to reject H0'}")

# Welch-Satterthwaite effective degrees of freedom and 95% CI for mu_old - mu_new.
# Zero inside the CI is equivalent to failing to reject H0 - these two must agree.
se   = np.sqrt(v1 / n1 + v2 / n2)
nu_w = (v1 / n1 + v2 / n2) ** 2 / (
       v1**2 / (n1**2 * (n1 - 1)) + v2**2 / (n2**2 * (n2 - 1)))
ci   = (m1 - m2) + np.array([-1, 1]) * t_dist.ppf(0.975, nu_w) * se
print(f"[B.3] 95% CI for mu_old - mu_new: ({ci[0]:.3f}, {ci[1]:.3f})  "
      f"({'contains 0' if ci[0] <= 0 <= ci[1] else 'excludes 0'}) "
      f"-> consistent with test: {(ci[0] <= 0 <= ci[1]) == (p_t >= ALPHA)}")

# Fig B2: two-tailed because mu_old could be larger or smaller than mu_new.
plot_null(t_dist, nu_w, [(t_stat, p_t, "observed", "purple")],
          "$t$", "Part B - $t$ vs null (Welch's t-test)",
          "fig_B2_t.png", two_tailed=True)


#%%
###############################################################################
# PART C - Do old and new format plates draw letters from the same distribution?
# H0: old-format plates draw letters from the same distribution as new-format.
# Same chi-square machinery as Part A, but E comes from the new-format pool
# acting as an empirical reference (not a uniform 1/26 prior).
# E_l = n_old * p_hat_new_l: scale the new-format relative frequencies up
# to the old-format sample size so O and E are in the same units (counts).
# Letters with E_l < 5 must be pooled - the chi2 approximation breaks down
# for very sparse bins. Letters with E=0 in BOTH pools (e.g. I, O, Q which
# Argentine traffic law excludes entirely) get dropped as uninformative 0/0.
###############################################################################

def letter_counts(group):
    # Convert each letter character to its 0-25 index and count via bincount.
    idx  = [ord(c) - ord("A") for s in group["letters"] for c in str(s) if c.isalpha()]
    cnts = np.bincount(idx, minlength=26)
    return cnts, cnts.sum()

O_old_l, n_old_l = letter_counts(old)
O_new_l, n_new_l = letter_counts(new)
x26 = np.arange(26)

# Fig C1: relative frequency bars with Poisson error bars (sqrt(O)/n).
# The error bar on a relative frequency f = O/n is sqrt(O)/n because
# O is Poisson-distributed and the division by n just rescales the uncertainty.
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

E_l = n_old_l * (O_new_l / n_new_l)  # expected old counts under new-format reference

keep = E_l >= 5
if (~keep).any():
    dropped = [ALPHABET[i] for i in range(26) if not keep[i]]
    p_other, o_other = E_l[~keep].sum(), O_old_l[~keep].sum()
    if p_other > 0:
        # Rare letters with E < 5 but E > 0: fold into a single "other" bin.
        # This preserves their observed counts rather than throwing them away.
        O_pooled = np.append(O_old_l[keep], o_other)
        E_pooled = np.append(E_l[keep], p_other)
        print(f"\n[C.2] Pooled {(~keep).sum()} low-count letters into 'other': {dropped}")
    else:
        # E=0 AND O=0 for these letters: they never appear in either pool
        # (e.g. I, O, Q excluded by DNRPA regulation). A 0/0 bin would give
        # nan in the chi2 sum, so we drop them entirely - they carry no info.
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

# C.3: rank letters by their individual contribution to the total chi2 stat.
# These are computed on the UNPOOLED bins just for interpretation - some bins
# have E < 5 so they aren't valid standalone tests, only a ranking for the report.
terms = np.where(E_l > 0, (O_old_l - E_l) ** 2 / np.where(E_l > 0, E_l, 1), 0.0)
order = np.argsort(terms)[::-1]
print("\n[C.3] Top 5 letters by chi2 contribution:")
for i in order[:5]:
    print(f"  {ALPHABET[i]}: O={O_old_l[i]:>3d}, E={E_l[i]:6.2f}, term={terms[i]:.2f}"
          f"  ({'over' if O_old_l[i] > E_l[i] else 'under'}-represented)")

# Fig C3: signed standardised residuals (O-E)/sqrt(E) per letter.
# +2/-2 dashed lines are the ~95% Poisson bands for a single bin.
# Top-5 contributors highlighted in red as a visual guide for the report.
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
###############################################################################
# PART D - Are certain consecutive letter pairs suppressed?
# H0: consecutive letters within a plate are drawn independently, so the
# probability of pair (j,k) is just the product of marginals: p_j * p_k.
# Expected count: E_jk = N * p_hat_j * p_hat_k (outer product).
# We restrict the test to letters with >= 10 individual occurrences so that
# every E_jk in the sub-table clears ~5 (Pearson's rule of thumb).
# nu = K^2 - 1: K^2 cells with one constraint (pair counts sum to N).
###############################################################################

# Each plate gives 2 pairs (old ABC: AB and BC) or 3 pairs (new ABCD: AB, BC, CD).
pairs = [(s[i], s[i+1]) for s in df["letters"].astype(str)
         for i in range(len(s) - 1) if s[i].isalpha() and s[i+1].isalpha()]
N = len(pairs)
print(f"\n[D.1] Total consecutive letter pairs: {N}")

# Build the 26x26 observed pair matrix and marginal letter frequencies.
P_jk = np.zeros((26, 26), dtype=int)
for a, b in pairs:
    P_jk[ord(a) - ord("A"), ord(b) - ord("A")] += 1

all_letters = [c for s in df["letters"].astype(str) for c in s if c.isalpha()]
cnt_all = np.bincount([ord(c) - ord("A") for c in all_letters], minlength=26)
p_hat   = cnt_all / cnt_all.sum()   # marginal relative frequency of each letter

E_jk = N * np.outer(p_hat, p_hat)  # expected under independence

# Trim to the K letters with >= 10 individual occurrences.
# This ensures every E_sub[i,j] = N * p_i * p_j >= 5 approximately.
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

# D.4: standardised residuals R_jk = (P-E)/sqrt(E). Rank all K^2 pairs and
# report the 5 most under- and over-represented for the report commentary.
R    = (P_sub - E_sub) / np.sqrt(E_sub)
flat = sorted([(letters_kept[i], letters_kept[j], R[i, j])
               for i in range(K) for j in range(K)], key=lambda t: t[2])
under, over = flat[:5], flat[-5:][::-1]
print("\n[D.4] Top 5 under-represented:", [f"{a}{b}:{r:.2f}" for a, b, r in under])
print("[D.4] Top 5 over-represented: ",  [f"{a}{b}:{r:.2f}" for a, b, r in over])

# Heatmap: cap the colour scale at the 98th percentile of |R| so one extreme
# cell doesn't compress all the contrast elsewhere. Boxed+labelled top-10 pairs.
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


#%% Part E - Interactive version with adjustable years

import numpy as np

# 1. Wikipedia Anchor Data (Prefix, Decimal Year)
# Decimal Year = Year + (Month-1)/12
OLD_ANCHORS = [
    ("AB000", 1995.00), ("AN000", 1996.00), ("BD000", 1997.00), ("BU000", 1998.00),
    ("CN000", 1999.00), ("DC000", 2000.00), ("DP000", 2001.00), ("DX000", 2002.00),
    ("ED000", 2003.00), ("EJ000", 2004.00), ("EU000", 2005.00), ("FI000", 2006.00),
    ("GB000", 2007.00), ("GV000", 2008.00), ("HU000", 2009.00), ("IM000", 2010.00),
    ("JN000", 2011.00), ("KV000", 2012.00), ("MC000", 2013.00), ("NM000", 2014.00),
    ("ON000", 2015.00), ("PM000", 2016.00), ("PZ999", 2016.25)
]

NEW_ANCHORS = [
    ("AA000AA", 2016.25), ("AB000AA", 2017.08), ("AC000AA", 2017.83),
    ("AD000AA", 2018.50), ("AE000AA", 2019.75), ("AF000AA", 2021.58),
    ("AG000AA", 2023.25), ("AH000AA", 2024.83), ("AI000AA", 2025.91),
    ("ZZ999ZZ", 2035.00) # Buffer for future interpolation
]

# 2. Math to convert a plate into a sequential number
def rank_old(p): return (ord(p[0])-65)*676000 + (ord(p[1])-65)*26000 + (ord(p[2])-65)*1000 + int(p[3:])
def rank_new(p): return (ord(p[0])-65)*17576000 + (ord(p[1])-65)*676000 + int(p[2:5])*676 + (ord(p[5])-65)*26 + (ord(p[6])-65)

# Split anchors into X (ranks) and Y (years) for numpy to process
OLD_X, OLD_Y = [rank_old(m[0]) for m in OLD_ANCHORS], [m[1] for m in OLD_ANCHORS]
NEW_X, NEW_Y = [rank_new(m[0]) for m in NEW_ANCHORS], [m[1] for m in NEW_ANCHORS]

# 3. The Core Estimator Logic
def estimate_year(plate):
    p = plate.upper().replace(" ", "").replace("-", "")
    
    # Old Format (e.g., ABC 123)
    if len(p) == 6 and p[:3].isalpha() and p[3:].isdigit():
        # Handle the re-patented 1995 legacy car rule
        if p[0] in "RSTUVWXYZ":
            return f"[{p}] -> Old Vehicle (Pre-1995), Re-patented in 1995"
            
        # Interpolate between Wikipedia dates
        year = np.interp(rank_old(p), OLD_X, OLD_Y)
        month = int((year % 1) * 12) + 1
        return f"[{p}] -> Estimated Issue Date: {month:02d}/{int(year)}"
        
    # New Format (e.g., AB 123 CD)
    elif len(p) == 7 and p[:2].isalpha() and p[2:5].isdigit() and p[5:].isalpha():
        year = np.interp(rank_new(p), NEW_X, NEW_Y)
        month = int((year % 1) * 12) + 1
        return f"[{p}] -> Estimated Issue Date: {month:02d}/{int(year)}"
        
    return "Error: Invalid plate structure."

# 4. Interactive Console Loop
print("--- Wikipedia-Backed Plate Estimator ---")
print("Type 'exit' to quit.\n")

while True:
    user_input = input("Enter a plate (e.g., AD 123 ZZ): ")
    if user_input.lower() == 'exit':
        print("Done!")
        break
    print(estimate_year(user_input) + "\n")