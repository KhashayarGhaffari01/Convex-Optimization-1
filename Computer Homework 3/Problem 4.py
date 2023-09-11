# %%
import numpy as np
import cvxpy as cvx
import matplotlib.pyplot as plt
from various_obj_regulator_data import *

u_ = cvx.Variable((m, T))
x_ = cvx.Variable((n, T + 1))
objs = [
    (cvx.Minimize(cvx.sum_squares(u_)), "$\\sum||u_t||_2^2$"),
    (cvx.Minimize(cvx.sum(cvx.norm(u_, 2, axis=0))), "$\\sum||u_t||_2$"),
    (cvx.Minimize(cvx.max(cvx.norm(u_, axis=0))), "$\\max||u_t||_2$"),
    (cvx.Minimize(cvx.sum(cvx.norm(u_, 1, axis=0))), "$\\sum||u_t||_1$")
]
plt.figure(figsize=(15, 5))
for i, obj in enumerate(objs):

    constraints = [x_[:, -1] == np.zeros(n)]
    constraints.append(x_[:, 0] == x_init)
    for t in range(1, T + 1):
        constraints.append(x_[:, t] == A @ x_[:, t - 1] + B @ u_[:, t - 1])
    prob = cvx.Problem(obj[0], constraints)
    prob.solve()
    plt.plot(u_.value.T)
    if i == 0:
        plt.ylabel("$u_t$")
    plt.title(obj[1])
    plt.xlabel("t")
    plt.savefig(f"fig_{i}_uvalue.jpg")
    plt.show()
    plt.xlabel("t")
    plt.plot(np.linalg.norm(u_.value, axis=0), label="$||u||_2$")
    if i == 2:
        plt.ylim(ymax=.12, ymin=0)
    if i == 0:
        plt.ylabel("$||u_t||$")
    plt.savefig(f"u_norm_{i}.jpg")
    plt.show()
plt.tight_layout()
plt.show()
