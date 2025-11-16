import numpy as np
import pandas as pd




def sharpe_ratio(returns, annual_factor=252):
    r = returns.mean() * annual_factor
    vol = returns.std() * np.sqrt(annual_factor)
    return r / vol if vol>0 else np.nan




def sortino_ratio(returns, annual_factor=252, target=0.0):
    neg = returns[returns<target]
    downside = np.sqrt((neg**2).mean()) * np.sqrt(annual_factor) if len(neg)>0 else 0.0
    excess = returns.mean()*annual_factor - target
    return excess / downside if downside>0 else np.nan




def historical_var_cvar(returns, alpha=0.01):
    r = returns.dropna().values
    if len(r)==0:
        return np.nan, np.nan
    sorted_r = np.sort(r)
    idx = max(1, int(np.floor(alpha * len(sorted_r))))
    var = - sorted_r[idx-1]
    cvar = - sorted_r[:idx].mean() if idx>0 else var
    return var, cvar




def max_drawdown(cum_returns):
    running_max = np.maximum.accumulate(cum_returns)
    dd = (running_max - cum_returns)
    return dd.max()