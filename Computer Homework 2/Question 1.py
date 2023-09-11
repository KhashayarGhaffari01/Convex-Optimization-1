# %%
import numpy as np
import cvxpy as cp
from matplotlib import pyplot as plt

N = np.array([0, 4, 2, 2, 3, 0, 4, 5, 6, 6, 4, 1, 4, 4, 0, 1, 3, 4, 2, 0, 3, 2, 0, 1])
N_test = (0, 1, 3, 2, 3, 1, 4, 5, 3, 1, 4, 3, 5, 5, 2, 1, 1, 1, 2, 0, 1, 2, 1, 0)
x = []
plt.plot(np.arange(24), N)
rho_set = [0.1, 1, 10, 100]
for rho in rho_set:
    lamb = cp.Variable(24)
    objective = cp.Minimize(
        cp.sum(lamb) - cp.sum(cp.multiply(N, cp.log(lamb))) + rho * cp.sum_squares(lamb[:-1] - lamb[1:]))
    constraints = []
    prob = cp.Problem(objective, constraints)
    result = prob.solve()
    plt.plot(np.arange(24), lamb.value)
    x.append(np.sum(lamb.value) - np.sum(N_test * np.log(lamb.value)))
plt.show()

plt.plot(rho_set, x, '.')
plt.show()
