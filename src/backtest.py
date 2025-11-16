"""Backtesting engine for pairs (vectorized)"""
import numpy as np
import pandas as pd
from .metrics import sharpe_ratio, historical_var_cvar, max_drawdown


def generate_signals(df, zcol='zscore', entry=2.0, exit=0.5, capital_per_trade=1e5):
    """
    Generate signals and compute PnL.
    - df: expects at least columns ['X','Y','beta_kf_used','zscore']
      where beta_kf_used is the lagged beta used for constructing spread/positions (to avoid lookahead)
    - entry/exit: z-score thresholds
    - capital_per_trade: scaling factor to compute returns from raw pnl (not hard-coded)
    Returns (df_with_signals, perf_dict)
    """
    df = df.copy()
    # Position logic: pos_spread in {-1,0,1}
    pos = np.zeros(len(df))
    position = 0
    for t in range(len(df)):
        z = df[zcol].iloc[t]
        if position == 0:
            if z > entry:
                position = -1
            elif z < -entry:
                position = 1
        elif position == 1:
            # currently long spread (long Y short X)
            if z > -exit:
                position = 0
        elif position == -1:
            # currently short spread
            if z < exit:
                position = 0
        pos[t] = position

    df['pos_spread'] = pos
    # Use provided lagged beta for hedging; fallback to 'beta_kf' if not provided
    if 'beta_kf_used' in df.columns:
        beta_for_hedge = df['beta_kf_used']
    else:
        beta_for_hedge = df.get('beta_kf', pd.Series(0.0, index=df.index))

    # h_Y = -pos_spread (we use 1 unit of Y per spread), h_X scaled by beta
    df['h_Y'] = - df['pos_spread']
    df['h_X'] = beta_for_hedge * df['pos_spread']

    # PnL: use previous bar's holdings times price change (no lookahead)
    df['pnl'] = df['h_Y'].shift(1) * df['Y'].diff() + df['h_X'].shift(1) * df['X'].diff()
    df['pnl'].fillna(0, inplace=True)
    df['cum_pnl'] = df['pnl'].cumsum()

    # Normalize returns by capital_per_trade to get meaningful Sharpe/Sortino etc.
    df['ret'] = df['pnl'] / float(capital_per_trade)
    df['cum_ret'] = (1 + df['ret']).cumprod()

    perf = {
        'sharpe': sharpe_ratio(df['ret']),
        'var_cvar': historical_var_cvar(df['ret']),
        'max_dd': max_drawdown(df['cum_ret'].values)
    }
    return df, perf
