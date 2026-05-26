import numpy as np
from scipy.odr import ODR, Model, RealData
import scipy.stats as stats
import matplotlib.pyplot as plt

# Generate data 
x = np.linspace(-3, 3, 50)
true_coeffs = [1, -2, 0.5, 3] # a, b, c, d  →  a·x³ + b·x² + c·x + d
sigma = 0.2

# Define model for ODR
def cubic(p, x):
    a, b, c, d = p
    return  a*x**3 + b*x**2 + c*x + d

def chi2(model, y, y_err):
    _sum = 0
    for i in range(len(y)):
        _sum += (model[i] - y[i])**2 / y_err[i]**2
    return _sum

# generate the data x, y, y_err
y_true = np.polyval(true_coeffs, x) 
y = [ stats.norm.rvs(cubic(true_coeffs, i), sigma) for i in x ]
y_err = [ sigma for i in y ]


model = Model(cubic)
data  = RealData(x, y, sy=sigma) # sy = known σ on y

# beta0 are initial parameters
odr = ODR(data, model, beta0=[1, 1, 1, 1]) #recommended use for the minimm vaue for s numericallt
odr.set_job(fit_type=2) # 0 = full ODR, 2 = Least-squares like (errors in y only)
result = odr.run() # thats all the information with the initial parameters here pretty simple



x_fine = np.linspace(x[0], x[-1],50)

plt.figure(1)
plt.clf()
plt.errorbar(x, y, yerr=y_err, fmt=".k")
plt.plot(x_fine, cubic(result.beta, x_fine)) # the way to plot al this bullshit


chi2 = chi2(cubic(result.beta, x), y, y_err)
print(f"Chi2: {chi2}")
print(f"Expected: {len(x) - len(true_coeffs)}")
# The distance chi square will fall outside thing 
# we gotta do a interval of the plus minus of the expected and print the interval and 
# Chi outside interval about fifty percent 


