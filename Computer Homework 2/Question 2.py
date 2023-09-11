#%%
import numpy as np
import cvxpy as cp

A = np.array([(1, 2, 0, 1),
              (0, 0, 3, 1),
              (0, 3, 1, 1),
              (2, 1, 2, 5),
              (1, 0, 3, 2)])
c_max = np.ones(5) * 100
p = np.array([3, 2, 7, 6])
p_disc = np.array([2, 1, 4, 2])
q = np.array([4, 10, 5, 10])

x = cp.Variable(4)
y = cp.Variable(4)
objective = cp.Maximize(cp.sum(y))
constraints = [cp.multiply(p, x) >= y,
               cp.multiply(p,q)+cp.multiply(p_disc, x-q) >= y,
               x >= np.zeros(4),
               cp.matmul(A, x) <= c_max]
prob = cp.Problem(objective, constraints)
result = prob.solve()

print(x.value)
print(-y.value/x.value)
