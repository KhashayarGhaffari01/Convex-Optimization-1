#%%
import cvxpy as cp
x = cp.Variable(1)
y = cp.Variable(1)
objective = cp.Maximize(0)
constraints = [cp.inv_pos(x) + cp.inv_pos(y) <= 1]
prob = cp.Problem(objective, constraints)
print(prob.is_dcp())

#%%
import cvxpy as cp
x = cp.Variable(1)
y = cp.Variable(1)
objective = cp.Maximize(0)
constraints = [x >= cp.inv_pos(y)]
prob = cp.Problem(objective, constraints)
print(prob.is_dcp())

#%%
x = cp.Variable(1)
y = cp.Variable(1)
objective = cp.Maximize(0)
constraints = [cp.quad_over_lin(x + y, cp.sqrt(y)) <= x - y + 5]
prob = cp.Problem(objective, constraints)
print(prob.is_dcp())

#%%
x = cp.Variable(1)
y = cp.Variable(1)
z = cp.Variable(1)
objective = cp.Maximize(0)
constraints = [x + z <= 1 + cp.geo_mean(cp.vstack([x - cp.quad_over_lin(z,y), y]))]
prob = cp.Problem(objective, constraints)
print(prob.is_dcp())
