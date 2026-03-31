# Applied Probability and Statistics - Programming Assignment 2 Solution
#By Paz Lieghton - Version 0.1


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson
import seaborn as sns #-> CHECK IF NEEDED USE THE PYTHON GRAPH Galery 



#Plt Style document
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Part A: Exact Calculation with Bayes' Theorem

# Given data
P_A = 0.20
P_B = 0.50
P_C = 0.30
P_defect_given_A = 0.10
P_defect_given_B = 0.02
P_defect_given_C = 0.05

print("=== PART A: Exact Calculation with Bayes' Theorem ===")

# A.1: Law of Total Probability to compute P(defect)
P_defect = (P_defect_given_A * P_A + 
           P_defect_given_B * P_B + 
           P_defect_given_C * P_C)
print(f"A.1: P(defect) = {P_defect:.4f}")

# A.2: Apply Bayes' theorem to compute conditional probabilities
P_A_given_defect = (P_defect_given_A * P_A) / P_defect
P_B_given_defect = (P_defect_given_B * P_B) / P_defect
P_C_given_defect = (P_defect_given_C * P_C) / P_defect

print(f"A.2: P(A|defect) = {P_A_given_defect:.4f}")
print(f"      P(B|defect) = {P_B_given_defect:.4f}")
print(f"      P(C|defect) = {P_C_given_defect:.4f}")

# Verify they sum to 1
print(f"      Sum = {P_A_given_defect + P_B_given_defect + P_C_given_defect:.4f}")

# A.3: Repeat for non-defective items
P_not_defect = 1 - P_defect
P_defect_given_A_not = 1 - P_defect_given_A
P_defect_given_B_not = 1 - P_defect_given_B
P_defect_given_C_not = 1 - P_defect_given_C

P_A_given_not_defect = (P_defect_given_A_not * P_A) / P_not_defect
P_B_given_not_defect = (P_defect_given_B_not * P_B) / P_not_defect
P_C_given_not_defect = (P_defect_given_C_not * P_C) / P_not_defect

print(f"A.3: P(A|not defect) = {P_A_given_not_defect:.4f}")
print(f"      P(B|not defect) = {P_B_given_not_defect:.4f}")
print(f"      P(C|not defect) = {P_C_given_not_defect:.4f}")

# A.4: Intuitive explanation
print("\nA.4: Intuitive explanation:")
print("Machine B has the smallest P(B|defect) because although it produces 50% of items,")
print("it only has a 2% defect rate. In contrast, machine A has a 10% defect rate but")
print("only produces 20% of items. Therefore, even though machine B produces more items,")
print("its low defect rate means that when we observe a defective item, it's less likely")
print("to have come from machine B.")

# Part B: Monte Carlo Verification

print("\n=== PART B: Monte Carlo Verification ===")

def monte_carlo_simulation(N=100000):
    # Step 1: Sample which machine produced each item
    machines = np.random.choice(['A', 'B', 'C'], size=N, p=[P_A, P_B, P_C])
    
    # Step 2: For each machine, sample whether the item is defective
    defects = []
    for machine in machines:
        if machine == 'A':
            defect_prob = P_defect_given_A
        elif machine == 'B':
            defect_prob = P_defect_given_B
        else:  # machine == 'C'
            defect_prob = P_defect_given_C
            
        defects.append(np.random.binomial(1, defect_prob))
    
    defects = np.array(defects)
    
    # Calculate simulated probabilities for defective items
    defective_indices = np.where(defects == 1)[0]
    A_count_defective = np.sum(machines[defective_indices] == 'A')
    B_count_defective = np.sum(machines[defective_indices] == 'B')
    C_count_defective = np.sum(machines[defective_indices] == 'C')
    
    total_defective = len(defective_indices)
    
    # Calculate probabilities
    P_A_given_defect_sim = A_count_defective / total_defective if total_defective > 0 else 0
    P_B_given_defect_sim = B_count_defective / total_defective if total_defective > 0 else 0
    P_C_given_defect_sim = C_count_defective / total_defective if total_defective > 0 else 0
    
    # Calculate probabilities for non-defective items
    non_defective_indices = np.where(defects == 0)[0]
    A_count_non_defective = np.sum(machines[non_defective_indices] == 'A')
    B_count_non_defective = np.sum(machines[non_defective_indices] == 'B')
    C_count_non_defective = np.sum(machines[non_defective_indices] == 'C')
    
    total_non_defective = len(non_defective_indices)
    
    # Calculate probabilities
    P_A_given_not_defect_sim = A_count_non_defective / total_non_defective if total_non_defective > 0 else 0
    P_B_given_not_defect_sim = B_count_non_defective / total_non_defective if total_non_defective > 0 else 0
    P_C_given_not_defect_sim = C_count_non_defective / total_non_defective if total_non_defective > 0 else 0
    
    return (P_A_given_defect_sim, P_B_given_defect_sim, P_C_given_defect_sim,
            P_A_given_not_defect_sim, P_B_given_not_defect_sim, P_C_given_not_defect_sim)

# Run simulation with N=100000
sim_results = monte_carlo_simulation(100000)
print(f"B.2: Simulated P(A|defect) = {sim_results[0]:.4f}")
print(f"      Simulated P(B|defect) = {sim_results[1]:.4f}")
print(f"      Simulated P(C|defect) = {sim_results[2]:.4f}")

print(f"B.3: Simulated P(A|not defect) = {sim_results[3]:.4f}")
print(f"      Simulated P(B|not defect) = {sim_results[4]:.4f}")
print(f"      Simulated P(C|not defect) = {sim_results[5]:.4f}")

# Compare with theoretical values
print("\nB.4: Comparison between theoretical and simulated results")
print("For defective items:")
print(f"Theoretical  P(A|defect) = {P_A_given_defect:.4f} | Simulated = {sim_results[0]:.4f}")
print(f"Theoretical  P(B|defect) = {P_B_given_defect:.4f} | Simulated = {sim_results[1]:.4f}")
print(f"Theoretical  P(C|defect) = {P_C_given_defect:.4f} | Simulated = {sim_results[2]:.4f}")

# Part C: Convergence as a Function of N

print("\n=== PART C: Convergence Analysis ===")

def convergence_analysis():
    N_values = [100, 500, 1000, 5000, 20000, 100000]
    P_A_given_defect_estimates = []
    
    for N in N_values:
        # Run simulation
        sim_results = monte_carlo_simulation(N)
        P_A_given_defect_estimates.append(sim_results[0])
        
    # Calculate absolute errors
    errors = [abs(est - P_A_given_defect) for est in P_A_given_defect_estimates]
    
    # Plot results
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.semilogx(N_values, P_A_given_defect_estimates, 'o-', label='Simulated')
    plt.axhline(y=P_A_given_defect, color='r', linestyle='--', label='Theoretical')
    plt.xlabel('N (sample size)')
    plt.ylabel(r'$\hat{P}(A|\text{defect})$')
    plt.title('Convergence of P(A|defect) vs N')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.semilogx(N_values, errors, 'o-', color='green')
    plt.xlabel('N (sample size)')
    plt.ylabel(r'$|\hat{P}(A|\text{defect}) - P(A|\text{defect})|$')
    plt.title('Absolute Error vs N')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Find N where error is within 1%
    target_error = 0.01 * P_A_given_defect
    for i, (N, error) in enumerate(zip(N_values, errors)):
        if error <= target_error:
            print(f"Convergence achieved to within 1% at N = {N} (error = {error:.6f})")
            break
    else:
        print("Convergence to 1% not achieved within tested range")
        
    return N_values, P_A_given_defect_estimates

# Run convergence analysis
N_vals, estimates = convergence_analysis()

# Part D: Extension to the Poisson Distribution

print("\n=== PART D: Poisson Distribution Extension ===")

# Given data for Poisson extension
P_A_poisson = 0.40  # P(A)
P_B_poisson = 0.60  # P(B)
mu_A = 3
mu_B = 1

def poisson_pmf(k, mu):
    """Calculate Poisson probability mass function"""
    return poisson.pmf(k, mu)

# D.1: Write down the Poisson PMF and express conditional probabilities
print("D.1: Poisson PMF is P(X=k|μ) = μ^k * e^(-μ) / k!")
print(f"P(k flaws | A) = {mu_A}^k * e^(-{mu_A}) / k!")
print(f"P(k flaws | B) = {mu_B}^k * e^(-{mu_B}) / k!")

# D.2: Apply Bayes' theorem to derive P(A|k flaws)
def bayes_poisson_A(k):
    """Calculate P(A|k flaws) using Bayes' theorem"""
    # P(k flaws | A) * P(A)
    numerator = poisson_pmf(k, mu_A) * P_A_poisson
    
    # P(k flaws | B) * P(B)
    denominator = (poisson_pmf(k, mu_A) * P_A_poisson + 
                  poisson_pmf(k, mu_B) * P_B_poisson)
    
    return numerator / denominator

# D.3: Evaluate numerically and plot
k_values = list(range(6))
P_A_given_k = [bayes_poisson_A(k) for k in k_values]
P_B_given_k = [1 - p for p in P_A_given_k]  # Since only A or B

print("D.3: Numerical evaluation")
for k, p_A in enumerate(P_A_given_k):
    print(f"k = {k}: P(A|k flaws) = {p_A:.4f}")

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(k_values, P_A_given_k, 'o-', label='P(A|k flaws)')
plt.plot(k_values, P_B_given_k, 's-', label='P(B|k flaws)')
plt.xlabel('Number of flaws (k)')
plt.ylabel('Probability')
plt.title('Bayesian Probabilities for Machines A and B with Poisson Flaw Counts')
plt.legend()
plt.grid(True)
plt.show()

# Explain the trend
print("\nIntuitive explanation:")
print("As k increases, the probability P(A|k flaws) increases because:")
print("- Machine A has higher mean number of flaws (μ_A = 3 vs μ_B = 1)")
print("- Observing a large number of flaws makes it more likely that the item came from machine A")
print("- The Poisson distribution with higher mean is more likely to produce larger k values")

# D.4: Monte Carlo verification for k=2
def monte_carlo_poisson_verification():
    N = 200000
    k_2_count_A = 0
    k_2_count_B = 0
    total_k_2 = 0
    
    # Generate items according to machine probabilities
    for _ in range(N):
        # Sample which machine produced it
        if np.random.rand() < P_A_poisson:
            # Machine A: sample from Poisson with μ=3
            flaws = np.random.poisson(mu_A)
            if flaws == 2:
                k_2_count_A += 1
                total_k_2 += 1
        else:
            # Machine B: sample from Poisson with μ=1
            flaws = np.random.poisson(mu_B)
            if flaws == 2:
                k_2_count_B += 1
                total_k_2 += 1
    
    # Estimate P(A|k=2) from simulation
    if total_k_2 > 0:
        simulated_A_given_2 = k_2_count_A / total_k_2
        theoretical_A_given_2 = bayes_poisson_A(2)
        print(f"\nD.4: Monte Carlo verification for k=2:")
        print(f"Theoretical P(A|k=2) = {theoretical_A_given_2:.6f}")
        print(f"Simulated P(A|k=2) = {simulated_A_given_2:.6f}")
        print(f"Difference = {abs(simulated_A_given_2 - theoretical_A_given_2):.6f}")
    else:
        print("No items with k=2 found in simulation")

# Run Monte Carlo verification
import numpy as np

monte_carlo_poisson_verification()

print("\nAll calculations completed successfully!")