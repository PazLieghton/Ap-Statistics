# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 17:39:42 2026
@author: pazli
"""

import numpy as np
import matplotlib.pyplot as plt
#Part A: Exact Calculation with Bayes Theorem

#KEY VALUES
#Shares of Total Production

P_A = 0.20
P_B = 0.50
P_C = 0.30

#-----Defect ratess------
P_defect_given_A = 0.10 #P(defect|A)
P_defect_given_B = 0.02
P_defect_given_C = 0.05


###############################################################################
print("\n ---------PART A: Exact Calculation with Bays Theorem----------\n")

print("[A.1]: Probability of a random selected item is defective:")
P_defect = (P_defect_given_A * P_A + P_defect_given_B * P_B + P_defect_given_C * P_C)
print(f"       P(defect) = {P_defect:} = {P_defect:5f}\n")


print("[A.2]: Apply Bayes theorem to compute conditional probabilities:")
#Pass from P(defect|A) to P(A|defect)

P_A_given_defect = (P_defect_given_A * P_A) / P_defect
P_B_given_defect = (P_defect_given_B * P_B) / P_defect
P_C_given_defect = (P_defect_given_C * P_C) / P_defect


print(f"A.2: P(A|defect) = {P_A_given_defect:.4f}")
print(f"P(B|defect) = {P_B_given_defect:.4f}")
print(f"P(C|defect) = {P_C_given_defect:.4f}")

# Verify they sum to 1
print(f"Sum = {P_A_given_defect + P_B_given_defect + P_C_given_defect:.4f}")


# [A.3]: Repeat for non-defective items (AKA - the inferse and bayes again)
P_not_defect = 1 - P_defect
P_defect_given_A_not =0.9 #Same Logic from the class 1 - P(a)
P_defect_given_B_not =0.98 # 1 - P(defect|a,b,c)
P_defect_given_C_not =0.95

P_A_given_not_defect = (P_defect_given_A_not * P_A) / P_not_defect
P_B_given_not_defect = (P_defect_given_B_not * P_B) / P_not_defect
P_C_given_not_defect = (P_defect_given_C_not * P_C) / P_not_defect

#DEBUG
# print(P_defect_given_A_not) 
# print(P_defect_given_B_not) 
# print(P_defect_given_C_not) 

print(f"A.3: P(A|not defect) = {P_A_given_not_defect:.4f}")
print(f"P(B|not defect) = {P_B_given_not_defect:.4f}")
print(f"P(C|not defect) = {P_C_given_not_defect:.4f}")


#[A.4]
print("\nA.4: Intuitive explanation:(Why is machine B, despite being the largest producer the smallest P(B|defect)\n")
print("Machine B has the smallest P(B|defect) because even though  it produces 50% of all the items,")
print("it only has a 2% defect rate. In contrast, machine A has a 10% defect rate but")
print("only produces 20% of items. Therefore, even though the machine B produces more items,")
print("its low defect rate means that when we observe a defective item, it's less likely to have come from machine B.")

###############################################################################

#ENDIFA
# ----------------------------
print ("\n\n-----------------------------------")
print("Part B: Monte Carlo Simulation (N = 100,000)")

N = 100000
machines = ['A', 'B', 'C']
machine_probs = [P_A, P_B, P_C]
defect_rates = [P_defect_given_A, P_defect_given_B, P_defect_given_C]

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
print(f"P(A|defect) = {sim_P_A_def:.4f} (theoretical: {P_A_given_defect:.4f})")
print(f"P(B|defect) = {sim_P_B_def:.4f} (theoretical: {P_B_given_defect:.4f})")
print(f"P(C|defect) = {sim_P_C_def:.4f} (theoretical: {P_C_given_defect:.4f})")

print("\nB.3 Simulated P(machine | non-defect):")
print(f"P(A|non-defect) = {sim_P_A_ndef:.4f} (theoretical: {P_A_given_not_defect:.4f})")
print(f"P(B|non-defect) = {sim_P_B_ndef:.4f} (theoretical: {P_B_given_not_defect:.4f})")
print(f"P(C|non-defect) = {sim_P_C_ndef:.4f} (theoretical: {P_C_given_not_defect:.4f})")

# B.4 Bar chart comparison
theoretical_def = [P_A_given_defect, P_B_given_defect, P_C_given_defect]
simulated_def = [sim_P_A_def, sim_P_B_def, sim_P_C_def]
theoretical_ndef = [P_A_given_not_defect, P_B_given_not_defect, P_C_given_not_defect]
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