"""Mean-variance optimizer with L1/L2 regularization using CVX"""
import cvxpy as cp
import numpy as np




def mean_variance_opt(mu, Sigma, target_return=None, l2=0.0, l1=0.0, max_weight=1.0):
    n = len(mu)
    w = cp.Variable(n)
    ret = mu @ w
    risk = cp.quad_form(w, Sigma)
    obj = cp.Minimize(risk + l2 * cp.sum_squares(w) + l1 * cp.norm1(w))
    constraints = [cp.sum(w) == 1, w >= -max_weight, w <= max_weight]
    if target_return is not None:
        constraints.append(ret >= target_return)
    prob = cp.Problem(obj, constraints)
    prob.solve(solver=cp.SCS)
    return w.value, prob.status