import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

# Set random seed for reproducibility
np.random.seed(42)

# ----------------------------
# Part A: Exact Calculation with Bayes' Theorem
# ----------------------------
print("=" * 50)
print("Part A: Theoretical Bayes' Theorem")
print("=" * 50)

# Production shares
P_A = 0.20
P_B = 0.50
P_C = 0.30

# Defect rates
P_def_given_A = 0.10
P_def_given_B = 0.02
P_def_given_C = 0.05

# A.1 Overall probability of defect
P_def = (P_A * P_def_given_A + P_B * P_def_given_B + P_C * P_def_given_C)
print(f"A.1 P(defect) = {P_def:.4f}")

# A.2 Bayes for defective item
P_A_given_def = (P_def_given_A * P_A) / P_def
P_B_given_def = (P_def_given_B * P_B) / P_def
P_C_given_def = (P_def_given_C * P_C) / P_def
print("\nA.2 Conditional probabilities given defective:")
print(f"P(A|defect) = {P_A_given_def:.4f}")
print(f"P(B|defect) = {P_B_given_def:.4f}")
print(f"P(C|defect) = {P_C_given_def:.4f}")
print(f"Sum = {P_A_given_def + P_B_given_def + P_C_given_def:.4f}")

# A.3 For non-defective items
P_ndef_given_A = 1 - P_def_given_A
P_ndef_given_B = 1 - P_def_given_B
P_ndef_given_C = 1 - P_def_given_C

P_ndef = (P_A * P_ndef_given_A + P_B * P_ndef_given_B + P_C * P_ndef_given_C)

P_A_given_ndef = (P_ndef_given_A * P_A) / P_ndef
P_B_given_ndef = (P_ndef_given_B * P_B) / P_ndef
P_C_given_ndef = (P_ndef_given_C * P_C) / P_ndef
print("\nA.3 Conditional probabilities given non-defective:")
print(f"P(A|non-defect) = {P_A_given_ndef:.4f}")
print(f"P(B|non-defect) = {P_B_given_ndef:.4f}")
print(f"P(C|non-defect) = {P_C_given_ndef:.4f}")
print(f"Sum = {P_A_given_ndef + P_B_given_ndef + P_C_given_ndef:.4f}")

# A.4 Explanation
print("\nA.4 Explanation: Machine B has the smallest defect rate (2%), so despite producing 50% of items, it contributes only a small fraction of defects. Hence, given a defective item, it is less likely to have come from B.")

# ----------------------------
# Part B: Monte Carlo Simulation with N = 100,000
# ----------------------------
print("\n" + "=" * 50)
print("Part B: Monte Carlo Simulation (N = 100,000)")
print("=" * 50)

N = 100000
machines = ['A', 'B', 'C']
machine_probs = [P_A, P_B, P_C]
defect_rates = [P_def_given_A, P_def_given_B, P_def_given_C]

# B.1 Generate items
machine_choices = np.random.choice(machines, size=N, p=machine_probs)
defect_status = np.zeros(N, dtype=bool)
for i, mach in enumerate(machines):
    mask = (machine_choices == mach)
    n_mach = np.sum(mask)
    defect_status[mask] = np.random.rand(n_mach) < defect_rates[i]

# B.2 & B.3 Compute fractions among defective and non-defective
defective_mask = defect_status
nondefective_mask = ~defect_status

defective_counts = {m: np.sum((machine_choices == m) & defective_mask) for m in machines}
nondefective_counts = {m: np.sum((machine_choices == m) & nondefective_mask) for m in machines}

total_defective = np.sum(defective_mask)
total_nondefective = np.sum(nondefective_mask)

sim_P_A_def = defective_counts['A'] / total_defective
sim_P_B_def = defective_counts['B'] / total_defective
sim_P_C_def = defective_counts['C'] / total_defective

sim_P_A_ndef = nondefective_counts['A'] / total_nondefective
sim_P_B_ndef = nondefective_counts['B'] / total_nondefective
sim_P_C_ndef = nondefective_counts['C'] / total_nondefective

print("B.2 Simulated P(machine | defect):")
print(f"P(A|defect) = {sim_P_A_def:.4f} (theoretical: {P_A_given_def:.4f})")
print(f"P(B|defect) = {sim_P_B_def:.4f} (theoretical: {P_B_given_def:.4f})")
print(f"P(C|defect) = {sim_P_C_def:.4f} (theoretical: {P_C_given_def:.4f})")

print("\nB.3 Simulated P(machine | non-defect):")
print(f"P(A|non-defect) = {sim_P_A_ndef:.4f} (theoretical: {P_A_given_ndef:.4f})")
print(f"P(B|non-defect) = {sim_P_B_ndef:.4f} (theoretical: {P_B_given_ndef:.4f})")
print(f"P(C|non-defect) = {sim_P_C_ndef:.4f} (theoretical: {P_C_given_ndef:.4f})")

# B.4 Bar chart comparison
theoretical_def = [P_A_given_def, P_B_given_def, P_C_given_def]
simulated_def = [sim_P_A_def, sim_P_B_def, sim_P_C_def]
theoretical_ndef = [P_A_given_ndef, P_B_given_ndef, P_C_given_ndef]
simulated_ndef = [sim_P_A_ndef, sim_P_B_ndef, sim_P_C_ndef]

x = np.arange(len(machines))
width = 0.35

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.bar(x - width/2, theoretical_def, width, label='Theoretical')
ax1.bar(x + width/2, simulated_def, width, label='Monte Carlo')
ax1.set_xticks(x)
ax1.set_xticklabels(machines)
ax1.set_ylabel('Probability')
ax1.set_title('Given Defective')
ax1.legend()

ax2.bar(x - width/2, theoretical_ndef, width, label='Theoretical')
ax2.bar(x + width/2, simulated_ndef, width, label='Monte Carlo')
ax2.set_xticks(x)
ax2.set_xticklabels(machines)
ax2.set_ylabel('Probability')
ax2.set_title('Given Non-defective')
ax2.legend()

plt.tight_layout()
plt.show()
print("B.4 The bar chart shows excellent agreement; Monte Carlo estimates are very close to the theoretical values due to the large sample size.")

# ----------------------------
# Part C: Convergence as a function of N
# ----------------------------
print("\n" + "=" * 50)
print("Part C: Convergence of P(A|defect) with increasing N")
print("=" * 50)

N_list = [100, 500, 1000, 5000, 20000, 100000]
estimates = []
errors = []

for n in N_list:
    # Simulate
    mach_choices = np.random.choice(machines, size=n, p=machine_probs)
    defects = np.zeros(n, dtype=bool)
    for i, mach in enumerate(machines):
        mask = (mach_choices == mach)
        defects[mask] = np.random.rand(np.sum(mask)) < defect_rates[i]
    # Estimate P(A|defect)
    def_mask = defects
    if np.sum(def_mask) == 0:
        est = 0.0
    else:
        est = np.sum((mach_choices == 'A') & def_mask) / np.sum(def_mask)
    estimates.append(est)
    errors.append(abs(est - P_A_given_def))

# C.2 Plot estimate vs N
plt.figure(figsize=(10, 5))
plt.plot(N_list, estimates, 'o-', label='Monte Carlo estimate')
plt.axhline(y=P_A_given_def, color='r', linestyle='--', label=f'Exact value = {P_A_given_def:.4f}')
plt.xscale('log')
plt.xlabel('Number of samples N (log scale)')
plt.ylabel('Estimated P(A|defect)')
plt.title('Convergence of Monte Carlo estimate for P(A|defect)')
plt.legend()
plt.grid(True)
plt.show()

# C.3 Find N where error <= 0.01
print("\nC.3 Absolute errors:")
for n, err in zip(N_list, errors):
    print(f"N = {n:6d}: error = {err:.4f}")
    if err <= 0.01:
        print(f"      -> Agreement within 1% achieved at N = {n}")
        break

# ----------------------------
# Part D: Poisson distribution extension
# ----------------------------
print("\n" + "=" * 50)
print("Part D: Poisson-distributed flaws")
print("=" * 50)

# D.1 Poisson PMF
def poisson_pmf(k, mu):
    return np.exp(-mu) * (mu**k) / np.math.factorial(k)

# D.2 Analytical expression for P(A|k flaws)
# P(A) = 0.4, P(B) = 0.6, mu_A = 3, mu_B = 1
P_A_pois = 0.4
P_B_pois = 0.6
mu_A = 3
mu_B = 1

def P_A_given_k(k):
    num = P_A_pois * poisson_pmf(k, mu_A)
    denom = num + P_B_pois * poisson_pmf(k, mu_B)
    return num / denom

def P_B_given_k(k):
    return 1 - P_A_given_k(k)

# D.3 Evaluate for k = 0..5
k_vals = np.arange(0, 6)
probs_A = [P_A_given_k(k) for k in k_vals]
probs_B = [P_B_given_k(k) for k in k_vals]

print("\nD.3 P(A|k flaws) and P(B|k flaws) for k = 0,...,5:")
for k, pA, pB in zip(k_vals, probs_A, probs_B):
    print(f"k = {k}: P(A|k) = {pA:.4f}, P(B|k) = {pB:.4f}")

# Plot
plt.figure(figsize=(8, 5))
plt.plot(k_vals, probs_A, 'o-', label='P(A|k flaws)')
plt.plot(k_vals, probs_B, 's-', label='P(B|k flaws)')
plt.xlabel('Number of flaws (k)')
plt.ylabel('Posterior probability')
plt.title('Posterior probability of machine given k flaws (Poisson)')
plt.legend()
plt.grid(True)
plt.show()
print("Trend: P(A|k) increases with k because machine A has a higher average flaw rate (μ=3 vs μ=1). More flaws make it more likely the item came from A.")

# D.4 Monte Carlo verification for k = 2
print("\nD.4 Monte Carlo verification for k = 2 (N = 200,000)")
N_pois = 200000
mach_pois = np.random.choice(['A', 'B'], size=N_pois, p=[P_A_pois, P_B_pois])
flaws = np.zeros(N_pois, dtype=int)
for i, mach in enumerate(['A', 'B']):
    mask = (mach_pois == mach)
    mu = mu_A if mach == 'A' else mu_B
    flaws[mask] = np.random.poisson(mu, size=np.sum(mask))

# Select items with exactly 2 flaws
mask_k2 = (flaws == 2)
if np.sum(mask_k2) > 0:
    from_A_k2 = np.sum((mach_pois == 'A') & mask_k2)
    total_k2 = np.sum(mask_k2)
    sim_P_A_given_k2 = from_A_k2 / total_k2
    theo_P_A_given_k2 = P_A_given_k(2)
    print(f"Simulated P(A|k=2) = {sim_P_A_given_k2:.4f}")
    print(f"Theoretical P(A|k=2) = {theo_P_A_given_k2:.4f}")
    print(f"Absolute error = {abs(sim_P_A_given_k2 - theo_P_A_given_k2):.4f}")
else:
    print("No items with exactly 2 flaws found – try increasing N or rerun with different seed.")

print("\nAll parts completed.")