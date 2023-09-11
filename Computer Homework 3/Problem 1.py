#%%
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from opt_evac_data import *


def optimize_evacuation(A, Q, F, q_initial, r, s, r_tilde, s_tilde, T):
    n, m = A.shape
    q = cp.Variable((n, T))
    f = cp.Variable((m, T - 1))

    node_risk = cp.reshape(q.T @ r + cp.square(q).T @ s, (T, 1))
    edge_risk = cp.vstack((cp.reshape(cp.abs(f).T @ r_tilde + cp.square(f).T @ s_tilde, (T - 1, 1)), np.array([[0]])))
    risk = node_risk + edge_risk

    constraints = [
        q[:, 0] == q_initial,
        q[:, 1:] == A @ f + q[:, :-1],
        0 <= q,
        q <= np.tile(Q, (T, 1)).T,
        cp.abs(f) <= np.tile(F, (T - 1, 1)).T
    ]

    problem = cp.Problem(cp.Minimize(sum(risk)), constraints)
    problem.solve(solver=cp.ECOS)

    q_values = np.array(q.value)
    f_values = np.array(f.value)
    node_risk_values = np.array(node_risk.value)
    total_risk = problem.value

    print("Total risk:")
    print(total_risk)
    print("Evacuated at t =", (node_risk_values <= 1e-4).nonzero()[0][0] + 1)

    plt.rc('text', usetex=True)

    plt.plot(np.arange(1, T + 1), risk.value)
    plt.xlabel('$t$')
    plt.ylabel('$R_t$')
    plt.savefig('p11.jpg')
    plt.show()

    plt.plot(np.arange(1, T + 1), q_values.T)
    plt.xlabel('$t$')
    plt.ylabel('$q_t$')
    plt.savefig('p12.jpg')
    plt.show()


    plt.plot(np.arange(1, T), f_values.T)
    plt.xlabel('$t$')
    plt.ylabel('$f_t$')
    plt.savefig('p13.jpg')
    plt.show()

    return


optimize_evacuation(A, Q, F, q1, r, s, rtild, stild, T)
