# %%
import cvxpy as cp
from multi_risk_portfolio_data import *

w = cp.Variable(n)
t = cp.Variable()
Sigma = [Sigma_1, Sigma_2, Sigma_3, Sigma_4, Sigma_5, Sigma_6]
risks = [cp.quad_form(w, Sigma[i]) for i in range(5)]
risk_constraints = [risk <= t for risk in risks]
prob = cp.Problem(cp.Maximize(w.T @ mu - gamma * t),
                  risk_constraints + [cp.sum(w) == 1])
prob.solve()
print("weights:", w.value)
print("gamma values:", [(risk.dual_value[0]) for risk in risk_constraints])
print("risk values:", [(risk.value) for risk in risks])
print('worst case risk:', t.value)
