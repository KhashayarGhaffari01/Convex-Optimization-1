# %%
import cvxpy as cp
from blend_design_data import *

cof = cp.Variable(k)
constraints = [np.log(P) @ cof <= np.log(P_spec),
               np.log(D) @ cof <= np.log(D_spec),
               np.log(A) @ cof <= np.log(A_spec),
               cp.sum(cof) == 1, cof >= 0]

problem = cp.Problem(cp.Minimize(0), constraints)
problem.solve()

w = np.exp(np.log(W) @ cof.value)

print("W:")
print(w)
