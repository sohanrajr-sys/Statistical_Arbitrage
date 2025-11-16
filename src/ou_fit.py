"""Fit OU parameters using discrete AR(1) approx and MLE for continuous-time OU."""
import numpy as np
import statsmodels.api as sm
from scipy.optimize import minimize




def fit_ou_discrete(spread):
    s = spread.dropna().values
    s_tm = s[:-1]
    s_tp = s[1:]
    X = sm.add_constant(s_tm)
    res = sm.OLS(s_tp, X).fit()
    a, phi = res.params
    resid = res.resid
    sigma_eps = np.std(resid, ddof=1)
    dt = 1.0
    if phi <= 0:
        kappa = np.nan
        mu = np.nan
        sigma = np.nan
    else:
        kappa = -np.log(phi) / dt
        mu = a / (1 - phi)
        sigma = sigma_eps * np.sqrt( -2 * np.log(phi) / (dt*(1-phi**2)) )
    return dict(phi=phi, a=a, sigma_eps=sigma_eps, kappa=kappa, mu=mu, sigma=sigma)




def ou_log_likelihood(params, s, dt=1.0):
    kappa, mu, sigma = params
    # transition density of OU is Normal with mean m = mu + (s_t - mu) * exp(-kappa dt)
    # variance v = sigma^2/(2 kappa) * (1 - exp(-2 kappa dt))
    s_tm = s[:-1]
    s_tp = s[1:]
    m = mu + (s_tm - mu) * np.exp(-kappa*dt)
    v = (sigma**2)/(2*kappa) * (1 - np.exp(-2*kappa*dt))
    ll = -0.5 * np.sum(np.log(2*np.pi*v) + (s_tp - m)**2 / v)
    return -ll




def fit_ou_mle(spread, initial=(0.5, 0.0, 0.1), dt=1.0):
    s = spread.dropna().values
    bounds = [(1e-6, 10.0), (-10.0, 10.0), (1e-6, 5.0)]
    res = minimize(ou_log_likelihood, x0=initial, args=(s, dt), bounds=bounds)
    kappa, mu, sigma = res.x
    return dict(kappa=kappa, mu=mu, sigma=sigma, success=res.success)