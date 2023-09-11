# %%
import numpy as np
import cvxpy as cp

S = np.ones((2, 2))
T = np.ones((2, 2))
C = cp.Variable((3, 3), symmetric=True)
constraints = [C >> 0]

# Blocks of Matrix C
C1 = C[0:2, 0:2]
C2 = C[1:3, 1:3]
C_13 = C[0, 2]

objective = cp.Minimize(
    cp.norm(C1 - S, "fro") ** 2 + cp.norm(C2 - T, "fro") ** 2 +
    cp.norm(C_13, "fro") ** 2)

problem = cp.Problem(objective, constraints)
problem.solve()

print("Optimal value:")
print(problem.value)
print("Covariance matrix:")
print(C.value)
print("Eigenvalues:")
print(np.linalg.eig(C.value)[0])
