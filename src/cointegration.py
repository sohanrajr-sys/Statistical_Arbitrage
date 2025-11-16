"""Cointegration utilities: Engle-Granger and helper functions."""
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import numpy as np




def engle_granger_test(y, x, maxlag=10):
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    resid = model.resid
    adf_res = adfuller(resid, maxlag=maxlag, autolag='AIC')
    return dict(model=model, resid=resid, adf_stat=adf_res[0], adf_pvalue=adf_res[1])