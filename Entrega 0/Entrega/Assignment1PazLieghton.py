# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 17:39:42 2026
@author: Paz Lieghton
"""
import numpy as np
import matplotlib.pyplot as plt #For B,C,D
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
print("\n --------PART A: Exact Calculation with Bays Theorem--------\n")

print("[A.1]: Probability of a random selected item is defective:")
P_defect = (P_defect_given_A * P_A + P_defect_given_B * P_B + P_defect_given_C * P_C)
print(f"    P(defect) = {P_defect:} = {P_defect:5f}\n")


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


#[A.4] Intuivive explanation
print("\nA.4: Intuitive explanation:(Why is machine B, despite being the largest producer the smallest rate P(B|defect)\n")
print("Machine B has the smallest P(B|defect) because even though  it produces 50% of all the items,")
print("it only has a 2% defect rate. In contrast, machine A has a 10% defect rate but")
print("only produces 20% of items. even tough the machine B produces more items,")
print("its low defect rate means that when we observe a defective item, it's less likely to have come from machine B.")
###############################################################################
###############################################################################
print("\n-------- PART B: Monte Carlo Simulation --------\n")

np.random.seed(420696969)
N = 100000

print("[B.1]: Two step Simulation to creeate N= 100000 items, and store it in a long vector: DONE!")

#We create the tuples for the machines and the probabilities
machines = ['A', 'B', 'C']
probs = [P_A, P_B, P_C] #Referring initial date
machine_samples = np.random.choice(machines, size=N, p=probs)
#We create thus the long tuple of the emachines and of the N size with the probabiliies given

#Now with the machines samples, we create the array of defect probabilities for each items machine product
defect_probs = np.zeros(N) 
#The logic : If content of tuple is 'A,B,C' we add to the nul tuple the p_defect_given
defect_probs[machine_samples == 'A'] = P_defect_given_A
defect_probs[machine_samples == 'B'] = P_defect_given_B
defect_probs[machine_samples == 'C'] = P_defect_given_C

#Random module, array of floats ig its less than the defect_probs it gives true/false
#Giving a long tuple of true false defect rates true = 1 --> use this later
defective = np.random.rand(N) < defect_probs

print("\n[B.2]:In All the defective items, compute the fraction that came from each machine.")

#We masking, if I use If/For loops this is going to be a bible.
defective_mask = defective
non_defective_mask = ~defective 

count_defective = np.sum(defective_mask)#We sum the: True = 1.
#Too many variables kekw we go to the machine_samples tuple and we make a mask of each ITEM + Defective_mask to make
A_def = np.sum((machine_samples == 'A') & defective_mask)
B_def = np.sum((machine_samples == 'B') & defective_mask)
C_def = np.sum((machine_samples == 'C') & defective_mask)

#And now tge simulated estumates of the P(A,B,C|defect)
sim_P_A_given_defect = A_def / count_defective #
sim_P_B_given_defect = B_def / count_defective #
sim_P_C_given_defect = C_def / count_defective #

print(f"Simulated P(A|defect) = {sim_P_A_given_defect:.3f}")
print(f"Simulated P(B|defect) = {sim_P_B_given_defect:.3f}")
print(f"Simulated P(C|defect) = {sim_P_C_given_defect:.3f}")

print("\n[B.3]:In All the NON-defective items, compute the fraction that came from each machine.")
#Same logic of iteming the samples and now just use non_defective_mask 
count_non_defective = np.sum(non_defective_mask)
A_nondef = np.sum((machine_samples == 'A') & non_defective_mask)
B_nondef = np.sum((machine_samples == 'B') & non_defective_mask)
C_nondef = np.sum((machine_samples == 'C') & non_defective_mask)

sim_P_A_given_nondef = A_nondef / count_non_defective
sim_P_B_given_nondef = B_nondef / count_non_defective
sim_P_C_given_nondef = C_nondef / count_non_defective

print(f"Simulated P(A|not defect) = {sim_P_A_given_nondef:.3f}")
print(f"Simulated P(B|not defect) = {sim_P_B_given_nondef:.3f}")
print(f"Simulated P(C|not defect) = {sim_P_C_given_nondef:.3f}")

print("\n[B.4]:Bar Chart DONE!.")

labels = ['A', 'B', 'C']

theoretical_def = [P_A_given_defect, P_B_given_defect, P_C_given_defect]
simulated_def = [sim_P_A_given_defect, sim_P_B_given_defect, sim_P_C_given_defect]
x = np.arange(len(labels))
width = 0.3
fig, (ax1, ax2) = plt.subplots(1,2,figsize=(12, 5))
#we colocate 2 values in x axis together for bar chart
ax1.bar(x - width/2, theoretical_def, width, label='Theoretical', color='green')
ax1.bar(x + width/2, simulated_def, width, label='Simulated', color='lightgreen')
ax1.set_ylabel('Probability')
ax1.set_title('P(machine|defective)')
ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.legend()
ax1.grid(axis='y', linestyle='--', alpha=0.7)

# Non-defective subplot
theoretical_nondef = [P_A_given_not_defect, P_B_given_not_defect, P_C_given_not_defect]
simulated_nondef = [sim_P_A_given_nondef, sim_P_B_given_nondef, sim_P_C_given_nondef]

ax2.bar(x - width/2, theoretical_nondef, width, label='Theoretical', color='blue')
ax2.bar(x + width/2, simulated_nondef, width, label='Simulated', color='skyblue')
ax2.set_ylabel('Probability')
ax2.set_title('P(machine|non-defective)')
ax2.set_xticks(x)
ax2.set_xticklabels(labels)
ax2.legend()
ax2.grid(axis='y', linestyle='--', alpha=0.7)
#For the LaTex
#plt.savefig('Histograms, dpi=150')
plt.show()

######################################################################################
###############################################################################
###############################################################################
print("\n--------PART C: Convergence as a Function of N --------\n")

N_values = [ 100, 500, 1000, 5000, 20000, 100000]

# Store estimates and absolute errors in a tuples
estimates = []
errors = []

print("[C.1] Simulating for different N = [100,500,1000,5000,20000,100000]")
#Six loops
for N in N_values:
    #Re-run the B part many times (6) Basically same
    machine_samples = np.random.choice(machines, size=N, p=probs)
    defect_probs = np.zeros(N)
    defect_probs[machine_samples == 'A'] = P_defect_given_A
    defect_probs[machine_samples == 'B'] = P_defect_given_B
    defect_probs[machine_samples == 'C'] = P_defect_given_C
    #Is defective.
    defective = np.random.rand(N) < defect_probs
    #Cunt defective items and those from machine A
    count_defective = np.sum(defective)
    #Just in case we divide in zero (I tried with tuples of 5,10 
    if count_defective == 0:
        sim_P_A_given_defect = 0.0
    else:
        A_def = np.sum((machine_samples == 'A') & defective)
        sim_P_A_given_defect = A_def / count_defective
    estimates.append(sim_P_A_given_defect)
    #Equation is in Assignment
    error = abs(sim_P_A_given_defect - P_A_given_defect)
    errors.append(error)
    
    print(f"N = {N:6d}  ->  P(A|defect) = {sim_P_A_given_defect:.4f}  |  Error = {error:.4f}")
    #Wonderful way to learn how to use print statements multiple times no loops. ^^^
    
print("\n[C.2]:Line Chart DONE!.")
  
plt.figure(figsize=(10, 5))
plt.plot(N_values, estimates, 'bo-', label='Simulated estimate')
plt.axhline(y=P_A_given_defect, color='red', linestyle='--', label=f'Exact = {P_A_given_defect:.3f}')
plt.xscale('log')  #Important
plt.xlabel('Sample size N (log Scale)')
plt.ylabel('P(A |defect)')
plt.title('Convergence of Monte Carlo estimate for P(A|defect)')
plt.legend()
plt.show()

print("\n[C.3] For each N, computing the absolute error")

# Target: error smaller than 1% of the exact P(A|defect)
target_error = 0.01*P_A_given_defect
print(f"Target absolute error: {target_error:.4f}")

reliable_N = None
#Loop N and finf the first one that is less than the target #It rarely workd in n = 1000000
for i in range(len(N_values)):
    current_N = N_values[i]
    current_error = errors[i]
    
    if current_error < target_error:
        reliable_N = current_N
        break   #Stop if it is smaller than target

if reliable_N is not None:
    #Loop again
    for i in range(len(N_values)):
        if N_values[i] == reliable_N:
            final_error = errors[i]
            break
#This last C.3 item was very AI heavy... sorry.
    print(f"The estimate is at 1% error range at N = {reliable_N} (error = {final_error:.3f}).")
else:
    print("None of the tested N values achieved < 1% error")
########################################################################################

from scipy.stats import poisson #Can use

print("\n----------PART D: Extension to the Poisson Distribution-----------\n")

#Data for the Machines
P_A = 0.4          #Share of P_A \ P_B production
P_B = 0.6
mu_A = 3.0         # Poisson mean flaws for A and B
mu_B = 1.0

print("[D.1]\n")
print("P(X = k | μ) = (μ^k * e^{-μ}) / k!\n")
print(" Applying this to the two machines:")
print(f"P(k flaws | A) = (μ_A^k * e^(-μ_A)) / k! = ({mu_A}^k * e^(-{mu_A})) / k!")
print(f"P(k flaws | B) = (μ_B^k * e^(-μ_B)) / k! = ({mu_B}^k * e^(-{mu_B})) / k!")
# D.2 Bayes' theorem expression (analytical) This is theory... right

print("[D.2] Bayes' theorem gives:")
print("                   P(k|A) * P(A)")
print("      P(A|k) = ---------------------------")
print("                P(k|A)*P(A) + P(k|B)*P(B)")
print("P(B|k) = 1 - P(A|k)\n")

print("[D.3] Evaluating P(A|k) for k = 0..5 and plotting... DONE!")

k_values = np.arange(0, 6)          # 0,1,2,3,4,5
# Poisson PMF for each k
pmf_A = poisson.pmf(k_values, mu_A)   # P(k|A)
pmf_B = poisson.pmf(k_values, mu_B)   # P(k|B)

# Bayes' theorem numerator and denominator
num = pmf_A * P_A
den = pmf_A * P_A + pmf_B * P_B
P_A_given_k = num / den
P_B_given_k = 1 - P_A_given_k

#Display items each
print("  k | P(A|k)   | P(B|k)")
for k, pa, pb in zip(k_values, P_A_given_k, P_B_given_k):
    print(f"  {k} | {pa:.4f}   | {pb:.4f}")

#Heavy Ai use on this.
plt.figure(figsize=(8, 5))
plt.plot(k_values, P_A_given_k, 'o-', label='P(A | k flaws)', color='darkorange')
plt.plot(k_values, P_B_given_k, 's-', label='P(B | k flaws)', color='purple')
plt.xlabel('Number of flaws k')
plt.ylabel('Posterior Probability')
plt.title('Probability that item came from machine A or B given k flaws')
plt.xticks(k_values)
plt.ylim(0, 1)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.show()

print("\nTrend explanation:")
print("Machine A has a higher mean number of flaws (μA=3) than machine B (μB=1).")
print("The more the flaws count increases the more likeley it is the first machine")
print("P(A|k) increases with k, while P(B|k) decreases the more k increases.")

# ---------- D.4 Monte Carlo verification for k = 2 ---------- 
print("\n[D.4] Monte Carlo verification for k = 2 with N = 200,000")

N_sim = 200_000
np.random.seed(69696969)   # for reproducibility

#Dsmplr yhr probabilities again
machines_D = np.random.choice(['A', 'B'], size=N_sim, p=[P_A, P_B])

#Now instead of masking we apply the poisson
flaws = np.zeros(N_sim, dtype=int)
mask_A = (machines_D == 'A')
mask_B = (machines_D == 'B')
flaws[mask_A] = poisson.rvs(mu=mu_A, size=np.sum(mask_A))
flaws[mask_B] = poisson.rvs(mu=mu_B, size=np.sum(mask_B))

#Now exactly count the  = 2 flaws.
mask_k2 = (flaws == 2)
total_k2 = np.sum(mask_k2)
A_k2 = np.sum((machines_D == 'A') & mask_k2)#Key

sim_P_A_given_2 = A_k2 / total_k2 if total_k2 > 0 else 0.0  #Only way

# Analytical value for k=2
analytical_P_A_given_2 = P_A_given_k[2]   # from array computed above

print(f"Total items with exactly 2 flaws: {total_k2}")
print(f"Items with 2 flaws from machine A: {A_k2}")
print(f"Simulated P(A | k=2) = {sim_P_A_given_2:.3f}")
print(f"Analytical P(A | k=2) = {analytical_P_A_given_2:.3f}")

print(f"Absolute error = {abs(sim_P_A_given_2 - analytical_P_A_given_2):.3f}")