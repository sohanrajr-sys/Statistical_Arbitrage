"""Kalman-based dynamic hedge ratio with improvements (Fix 2)."""
import numpy as np
import pandas as pd
from pykalman import KalmanFilter


import numpy as np
import pandas as pd
from pykalman import KalmanFilter

import numpy as np
import pandas as pd
from pykalman import KalmanFilter
from scipy.signal import savgol_filter

import numpy as np
import pandas as pd
from pykalman import KalmanFilter
from scipy.signal import savgol_filter

import numpy as np
import pandas as pd
from pykalman import KalmanFilter
from scipy.signal import savgol_filter

def kf_hedge_ratio_pykalman(
    y,
    x,
    n_iter=0,
    transition_cov=1e-5,
    observation_cov=1e-2,
    smooth_window=21,       # longer window for smoother beta (must be odd)
    smooth_poly=2,          # lower polynomial degree to suppress noise
    clip_beta=(0.5, 2.0)
):
    obs = np.column_stack([x.values, np.ones(len(x))])
    obs_mats = obs.reshape(len(x), 1, 2)

    kf = KalmanFilter(
        transition_matrices=np.eye(2),
        observation_matrices=obs_mats,
        initial_state_mean=np.zeros(2),
        initial_state_covariance=np.eye(2),
        transition_covariance=np.eye(2) * transition_cov,
        observation_covariance=observation_cov
    )

    smoothed_state_means = None
    if n_iter and int(n_iter) > 0:
        try:
            kf_em = kf.em(y.values, n_iter=int(n_iter))
            smoothed_state_means, _ = kf_em.smooth(y.values)
        except Exception:
            smoothed_state_means = None

    if smoothed_state_means is None:
        smoothed_state_means, _ = kf.smooth(y.values)

    beta = pd.Series(smoothed_state_means[:, 0], index=y.index)
    intercept = pd.Series(smoothed_state_means[:, 1], index=y.index)

    # Savitzky–Golay smoothing, fallback if not enough data
    if len(beta) >= smooth_window:
        beta_smooth = savgol_filter(beta.values, window_length=smooth_window, polyorder=smooth_poly)
        beta = pd.Series(beta_smooth, index=beta.index)
    else:
        beta = beta.ewm(alpha=0.1, adjust=False).mean()

    # Final touch: light EMA for micro-smoothing
    beta = beta.ewm(alpha=0.1, adjust=False).mean()

    # Clipping and lag
    beta = beta.clip(lower=clip_beta[0], upper=clip_beta[1])
    beta = beta.shift(1).bfill()

    return beta, intercept









def kalman_recursive_ols(y, x, delta=1e-5, R=0.02, clip=(0.9, 1.2), lag_steps=2):
    n = len(y)
    beta = np.zeros(n)
    P = np.zeros(n)
    beta[0] = 0.0
    P[0] = 1.0

    Q = delta / (1 - delta)

    for t in range(1, n):
        beta_pred = beta[t - 1]
        P_pred = P[t - 1] + Q
        xt = x.iloc[t]
        yt = y.iloc[t]

        denom = (xt ** 2) * P_pred + R
        K = (P_pred * xt) / denom if denom != 0 else 0.0

        beta[t] = beta_pred + K * (yt - beta_pred * xt)
        P[t] = (1 - K * xt) * P_pred

    beta_series = pd.Series(beta, index=y.index)
    P_series = pd.Series(P, index=y.index)

    beta = beta.ewm(alpha=0.01, adjust=False).mean()      # Moderate smoothing
    beta = beta.rolling(window=3, min_periods=1).mean()   # Small rolling window
    beta = beta.clip(lower=0.5, upper=2.0)                # Default clipping
    beta = beta.shift(1).bfill()                          # 1 period lag


    return beta_series, P_series
