# -*- coding: utf-8 -*-
"""
Assignment 4 - Part A - Paz Lieghton
Are Argentine license plate digits uniformly distributed?

The big idea here: if plates are assigned "at random", every digit 0-9
should appear equally often. We test that claim formally with a
chi-square goodness-of-fit test, separately for old (ABC 123) and
new (AB 123 CD) format plates.

Data collected by hand from real parked cars. Going out in the cold
to stare at license plates is, apparently, statistics fieldwork.

NOTE: if plates.csv is not present, a small synthetic dataset is generated
so the code can be tested before data collection is complete.
Replace it with your real CSV when ready.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import chi2 as chi2_dist   # same import pattern as test_examples

# ============================================================
# Load data — keep dtype=str so "028" stays "028" and not 28
# ============================================================

import os

if os.path.exists("plates.csv"):
    df = pd.read_csv("plates.csv", dtype=str)
    print("Loaded real data from plates.csv")
else:
    # -------------------------------------------------------------
    # Synthetic fallback — DELETE or ignore when real data is ready.
    # Generates a uniform-ish digit distribution so the test should
    # FAIL to reject H0 (p > 0.05), which is the expected result.
    # -------------------------------------------------------------
    print("plates.csv not found — generating synthetic test data.")
    rng = np.random.default_rng(42)

    def random_plate(fmt):
        digits = "".join(rng.choice(list("0123456789"), 3))
        if fmt == "old":
            letters = "".join(rng.choice(list("ABCDEFGHJKLMNOPQRSTUVWXYZ"), 3))
        else:
            letters = "".join(rng.choice(list("ABCDEFGHJKLMNOPQRSTUVWXYZ"), 4))
        return digits, letters

    rows = []
    for i in range(1, 101):
        fmt = "old" if i <= 60 else "new"
        d, l = random_plate(fmt)
        rows.append({"plate_id": i, "digits": d, "letters": l, "format": fmt})
    for i in range(101, 151):
        fmt = "new"
        d, l = random_plate(fmt)
        rows.append({"plate_id": i, "digits": d, "letters": l, "format": fmt})

    df = pd.DataFrame(rows)
    df.to_csv("plates.csv", index=False)  # save so other parts can reuse it


# Separate by plate format right away
old_df = df[df["format"] == "old"].copy().reset_index(drop=True)
new_df = df[df["format"] == "new"].copy().reset_index(drop=True)

print(f"\nTotal plates : {len(df)}")
print(f"  Old (ABC 123)    : {len(old_df)}")
print(f"  New (AB 123 CD)  : {len(new_df)}")

# ============================================================
# [PART A.1] — Frequency table O_k for digits 0-9
# ============================================================
# Each plate has exactly 3 digit characters.
# We treat every character in the string as a separate observation:
#   "473" -> three events: 4, 7, 3
#
# So the total digit count is:
#   n_old = 3 * (number of old plates)
#   n_new = 3 * (number of new plates)
#
# O_k = number of times digit k (k = 0,1,...,9) appears in the group.
# We will compute these separately for old and new format.

def extract_digits(group_df):
    """
    Flatten all digit characters from a DataFrame group into an int array.
    Uses str() to be safe against any weird column types.
    """
    all_digits = []
    for digits_str in group_df["digits"]:
        for ch in str(digits_str).strip():
            if ch.isdigit():               # skip any accidental spaces or NaN artefacts
                all_digits.append(int(ch))
    return np.array(all_digits, dtype=int)

digits_old = extract_digits(old_df)
digits_new = extract_digits(new_df)

# np.bincount with minlength=10 guarantees all 10 bins even if a digit is absent
# (unlikely in a real dataset of 150+ plates, but let's be defensive)
O_old = np.bincount(digits_old, minlength=10)   # shape (10,)
O_new = np.bincount(digits_new, minlength=10)

n_old = len(digits_old)   # total digit observations, old
n_new = len(digits_new)   # total digit observations, new

print(f"\n[A.1] Digit observations")
print(f"  Old : {n_old} digits from {len(old_df)} plates (= {len(old_df)} × 3 = {3*len(old_df)})")
print(f"  New : {n_new} digits from {len(new_df)} plates (= {len(new_df)} × 3 = {3*len(new_df)})")

print(f"\n  Digit   : {list(range(10))}")
print(f"  O (old) : {O_old.tolist()}")
print(f"  O (new) : {O_new.tolist()}")

# ============================================================
# [PART A.2] — Expected counts and histograms with Poisson error bars
# ============================================================
# Under H0 (uniform distribution), all 10 digits are equally likely.
# The expected count for digit k is simply:
#
#   E_k = n / 10
#
# because P(digit = k) = 1/10 for all k, and E_k = n * P(digit = k).
#
# The Poisson error bar on a count C is sqrt(C).
# Counts are Poisson random variables with mean E_k under H0,
# so the standard uncertainty on O_k is sqrt(O_k).
# (This is the same Poisson error bar convention used throughout the course.)
#
# We plot observed (O_k, blue) and expected (E_k, red) side by side.
# If the bars agree within the error bars, data is consistent with H0.

E_old = np.full(10, n_old / 10.0)
E_new = np.full(10, n_new / 10.0)

k = np.arange(10)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, O, E, label, n_total in [
    (axes[0], O_old, E_old, "Old format  (ABC 123)",   n_old),
    (axes[1], O_new, E_new, "New format  (AB 123 CD)", n_new),
]:
    # Observed counts — offset left by 0.2 so bars don't overlap
    ax.bar(k - 0.2, O, width=0.35, alpha=0.75, color="steelblue",
           label="Observed $O_k$")
    # Poisson error bars: uncertainty on each observed count = sqrt(O_k)
    ax.errorbar(k - 0.2, O, yerr=np.sqrt(O),
                fmt="none", color="black", capsize=4, linewidth=1.2)
    # Expected counts — constant horizontal line would also work, but bars
    # make the per-digit comparison much easier to read visually
    ax.bar(k + 0.2, E, width=0.35, alpha=0.75, color="tomato",
           label=f"Expected $E_k = n/10 = {n_total/10:.1f}$")

    ax.set_xlabel("Digit  $k$", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"[A.2]  Digit distribution — {label}", fontsize=12)
    ax.set_xticks(k)
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", linestyle="--", alpha=0.45)

plt.tight_layout()
plt.savefig("fig_A2_digit_distribution.png", dpi=150)
plt.show()

# ============================================================
# [PART A.3] — Chi-square goodness-of-fit test
# ============================================================
# The chi-square statistic is (Cowan 7.3 style):
#
#   chi2 = sum_{k=0}^{9}  (O_k - E_k)^2 / E_k
#
# which is the same as sum (O_k - E_k)^2 / (sqrt(E_k))^2 as written
# in the assignment. Under H0 this follows a chi2 distribution with:
#
#   nu = 10 - 1 = 9  degrees of freedom
#
# Why 9? We have 10 categories (the 10 digits), and 1 linear constraint
# (the total count n is fixed), so we lose 1 degree of freedom.
# This is exactly the "nu = N_bins - 1" rule for goodness-of-fit.
#
# The p-value answers: "if H0 were true, how often would we see a
# chi2 this large or larger just by chance?"
#
#   p = P(chi2_nu >= chi2_obs) = 1 - CDF(chi2_obs, nu)
#
# Decision rule at alpha = 0.05:
#   p < 0.05  -> reject H0  (digits are NOT uniformly distributed)
#   p >= 0.05 -> fail to reject H0  (data is consistent with uniform)

nu = 9  # 10 categories - 1 constraint, same for both formats

def chi2_gof_test(O, E, nu, label):
    """
    Chi-square goodness-of-fit test.
    O: observed counts (array, length = number of bins)
    E: expected counts (same shape as O)
    nu: degrees of freedom
    label: string for printing
    Returns: (chi2_stat, p_value)
    """
    # The statistic: sum of squared standardised residuals
    chi2_stat = np.sum((O - E)**2 / E)

    # p-value: right-tail probability of the chi2(nu) distribution
    # sf = survival function = 1 - CDF, so this is P(X >= chi2_stat)
    # same trick used in test_examples.py for the runs test
    p_value = chi2_dist.sf(chi2_stat, nu)

    print(f"\n[A.3]  Chi-square goodness-of-fit — {label}")
    print(f"  H0: digits {0}-{9} are uniformly distributed")
    print(f"  chi2_obs = {chi2_stat:.4f}    (expected mean under H0 = nu = {nu})")
    print(f"  p-value  = {p_value:.4f}")

    # Decision: same pattern as test_examples.py
    if p_value < 0.05:
        print(f"  p < 0.05  -->  REJECT H0: digits are NOT uniformly distributed")
    else:
        print(f"  p >= 0.05 -->  Fail to reject H0: consistent with uniform distribution")

    return chi2_stat, p_value

chi2_old, p_old = chi2_gof_test(O_old, E_old, nu, "Old format")
chi2_new, p_new = chi2_gof_test(O_new, E_new, nu, "New format")

# Do both formats lead to the same conclusion?
print("\n[A.3] Summary: same conclusion for both formats?")
reject_old = p_old < 0.05
reject_new = p_new < 0.05
if reject_old == reject_new:
    verdict = "REJECT H0 for both" if reject_old else "Fail to reject H0 for both"
    print(f"  Yes — {verdict}. Consistent story across formats.")
else:
    print("  No — formats disagree. One rejects, the other does not. Worth investigating.")


# ============================================================
# [A.3 - Bonus plot] Where do our chi2 values land on chi2(nu=9)?
# ============================================================
# This is a visual sanity check: we overlay the theoretical pdf
# and mark where our observed chi2 values fall.
# The shaded red region is the 5% rejection tail (right side).
# If a vertical line falls in the red zone, we reject H0.
# Same style as the chi2 MC plot in Entrega 3.

x_range = np.linspace(0, 30, 400)
critical_val = chi2_dist.ppf(0.95, nu)  # threshold for alpha = 0.05

fig, ax = plt.subplots(figsize=(9, 4))

ax.plot(x_range, chi2_dist.pdf(x_range, nu), "k-", linewidth=1.8,
        label=rf"$\chi^2(\nu={nu})$ theoretical")

# Shade the 5% rejection region
x_tail = x_range[x_range >= critical_val]
ax.fill_between(x_tail, chi2_dist.pdf(x_tail, nu),
                alpha=0.25, color="red",
                label=f"Rejection region  (α=0.05,  critical={critical_val:.2f})")

# Mark our observed values
ax.axvline(chi2_old, color="steelblue", linestyle="--", linewidth=1.8,
           label=f"Old:  $\\chi^2$ = {chi2_old:.2f},  p = {p_old:.3f}")
ax.axvline(chi2_new, color="darkorange", linestyle="--", linewidth=1.8,
           label=f"New:  $\\chi^2$ = {chi2_new:.2f},  p = {p_new:.3f}")

ax.set_xlabel(r"$\chi^2$", fontsize=12)
ax.set_ylabel("Density", fontsize=12)
ax.set_title(r"[A.3]  Observed $\chi^2$ vs $\chi^2(\nu=9)$  —  digit uniformity test",
             fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("fig_A3_chi2_test.png", dpi=150)
plt.show()

# ============================================================
# [A.3 - Residuals] Standardised residuals (O_k - E_k) / sqrt(E_k)
# ============================================================
# A complementary view: the signed residual per digit.
# Values outside [-2, +2] are the main contributors to chi2.
# If H0 holds, we expect about 68% of residuals within [-1, +1]
# and 95% within [-2, +2]. Any digit jumping outside that range
# is a candidate for a "weird" frequency — could be a real effect
# or just statistical noise at n ~ 150 plates.

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

for ax, O, E, label in [
    (axes[0], O_old, E_old, "Old format"),
    (axes[1], O_new, E_new, "New format"),
]:
    residuals = (O - E) / np.sqrt(E)
    colors = ["tomato" if abs(r) > 2 else "steelblue" for r in residuals]
    ax.bar(k, residuals, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)
    ax.axhline(0,  color="black",  linewidth=1.0)
    ax.axhline( 2, color="red",    linestyle="--", linewidth=1.0, label="±2 sigma")
    ax.axhline(-2, color="red",    linestyle="--", linewidth=1.0)
    ax.set_xlabel("Digit $k$", fontsize=11)
    ax.set_ylabel(r"$(O_k - E_k)\,/\,\sqrt{E_k}$", fontsize=11)
    ax.set_title(f"[A.3]  Standardised residuals — {label}", fontsize=11)
    ax.set_xticks(k)
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig("fig_A3_residuals.png", dpi=150)
plt.show()

print("\n\n[DONE] Part A complete. Figures saved:")
print("  fig_A2_digit_distribution.png")
print("  fig_A3_chi2_test.png")
print("  fig_A3_residuals.png")
#%%
# -*- coding: utf-8 -*-
# Assignment 4 - Part A - Paz Lieghton
# Chi-square goodness-of-fit: are Argentine plate digits uniformly distributed?
# Tested separately for old (ABC 123) and new (AB 123 CD) format plates.

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import chi2 as chi2_dist

# Load data — dtype=str keeps "028" from silently becoming 28
if os.path.exists("plates.csv"):
    df = pd.read_csv("plates.csv", dtype=str)
else:
    rng = np.random.default_rng(42)  # synthetic fallback — swap for real CSV
    fmts = ["old"]*60 + ["new"]*90
    rows = [{"plate_id": i+1, "format": f,
             "digits":  "".join(rng.choice(list("0123456789"), 3)),
             "letters": "".join(rng.choice(list("ABCDEFGHJKLMNOPQRSTUVWXYZ"),
                                            3 if f == "old" else 4))}
            for i, f in enumerate(fmts)]
    df = pd.DataFrame(rows); df.to_csv("plates.csv", index=False)

old_df, new_df = df[df["format"] == "old"], df[df["format"] == "new"]
print(f"Plates: {len(df)} total  |  Old: {len(old_df)}  |  New: {len(new_df)}")

# [A.1] Frequency table O_k for digits k=0..9, separately by format.
# Each plate has 3 digit chars: "473" contributes observations 4, 7, 3.
# Under H0 (uniform), E_k = n/10 for all k.
def extract_digits(grp):
    return np.array([int(c) for s in grp["digits"] for c in str(s) if c.isdigit()])

digits_old, digits_new = extract_digits(old_df), extract_digits(new_df)
O_old, O_new = np.bincount(digits_old, minlength=10), np.bincount(digits_new, minlength=10)
E_old, E_new = np.full(10, len(digits_old)/10.0), np.full(10, len(digits_new)/10.0)
k = np.arange(10)
print(f"\n[A.1] O_old = {O_old.tolist()}\n[A.1] O_new = {O_new.tolist()}")

# [A.2] Histograms: observed vs expected with Poisson error bars sqrt(O_k)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, O, E, lbl in zip(axes, [O_old, O_new], [E_old, E_new],
                          ["Old (ABC 123)", "New (AB 123 CD)"]):
    ax.bar(k-0.2, O, 0.35, alpha=0.75, color="steelblue", label="Observed $O_k$")
    ax.errorbar(k-0.2, O, yerr=np.sqrt(O), fmt="none", color="k", capsize=4)
    ax.bar(k+0.2, E, 0.35, alpha=0.75, color="tomato", label=f"Expected $E_k={E[0]:.1f}$")
    ax.set(xlabel="Digit $k$", ylabel="Count", title=f"[A.2] Digit distribution — {lbl}", xticks=k)
    ax.legend(); ax.grid(True, axis="y", ls="--", alpha=0.45)
plt.tight_layout(); plt.savefig("fig_A2.png", dpi=150); plt.show()

# [A.3] Chi-square goodness-of-fit test
# chi2 = sum_k (O_k - E_k)^2 / E_k  ~  chi2(nu=9) under H0
# nu = 10 categories - 1 constraint (total n is fixed) = 9
# p = P(chi2_nu >= chi2_obs)  — reject H0 at alpha=0.05 if p < 0.05
nu = 9
results = {}
for O, E, lbl in zip([O_old, O_new], [E_old, E_new], ["Old", "New"]):
    chi2_stat = np.sum((O - E)**2 / E)
    p = chi2_dist.sf(chi2_stat, nu)   # sf = 1-CDF, same trick as test_examples
    results[lbl] = (chi2_stat, p)
    print(f"\n[A.3] {lbl}: chi2={chi2_stat:.4f}, p={p:.4f}  ->  "
          f"{'REJECT H0' if p < 0.05 else 'Fail to reject H0'}")

# Residuals plot: bars outside +-2 are the main chi2 contributors
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
for ax, O, E, lbl in zip(axes, [O_old, O_new], [E_old, E_new], ["Old", "New"]):
    chi2_stat, p = results[lbl]
    res = (O - E) / np.sqrt(E)
    ax.bar(k, res, color=["tomato" if abs(r) > 2 else "steelblue" for r in res],
           alpha=0.8, edgecolor="k", lw=0.5)
    ax.axhline(0, color="k", lw=1.0)
    for h in [2, -2]: ax.axhline(h, color="red", ls="--", lw=1.0)
    ax.set(xlabel="Digit $k$", ylabel=r"$(O_k-E_k)/\sqrt{E_k}$",
           title=f"[A.3] Residuals — {lbl}  |  chi2={chi2_stat:.2f}, p={p:.3f}", xticks=k)
    ax.grid(True, axis="y", ls="--", alpha=0.4)
plt.tight_layout(); plt.savefig("fig_A3_residuals.png", dpi=150); plt.show()

# Bonus: chi2(nu=9) pdf with rejection region and observed values marked
x  = np.linspace(0, 30, 400)
crit = chi2_dist.ppf(0.95, nu)
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(x, chi2_dist.pdf(x, nu), "k-", lw=1.8, label=rf"$\chi^2(\nu={nu})$")
ax.fill_between(x[x>=crit], chi2_dist.pdf(x[x>=crit], nu),
                alpha=0.25, color="red", label=f"Rejection region (crit={crit:.2f})")
for lbl, (c2, p), clr in zip(["Old","New"], results.values(), ["steelblue","darkorange"]):
    ax.axvline(c2, color=clr, ls="--", lw=1.8, label=f"{lbl}: chi2={c2:.2f}, p={p:.3f}")
ax.set(xlabel=r"$\chi^2$", ylabel="Density", title=r"[A.3] $\chi^2$ observed vs $\chi^2(\nu=9)$")
ax.legend(); ax.grid(True, ls="--", alpha=0.4)
plt.tight_layout(); plt.savefig("fig_A3_chi2.png", dpi=150); plt.show()