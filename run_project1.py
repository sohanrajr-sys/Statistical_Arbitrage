import argparse
import numpy as np
import pandas as pd
import yfinance as yf
from src.cointegration import engle_granger_test
from src.kalman import kf_hedge_ratio_pykalman, kalman_recursive_ols
from src.ou_fit import fit_ou_discrete, fit_ou_mle
from src.backtest import generate_signals
from src.metrics import historical_var_cvar, sharpe_ratio
from src.optimizer import mean_variance_opt
import matplotlib.pyplot as plt

def load_stock_pair(ticker_x, ticker_y, start="2020-01-01", end="2023-01-01"):
    """
    Downloads stock price data for a pair from yfinance.
    Returns a dataframe with columns ['X', 'Y'], indexed by datetime.
    """
    df_x = yf.download(ticker_x, start=start, end=end, progress=False)
    df_y = yf.download(ticker_y, start=start, end=end, progress=False)
    print(df_x['Close'].dtypes)
    # Preferred: Use concat to guarantee the right structure
    ko_close = df_x[['Close']].copy()
    ko_close.columns = ['X']
    pep_close = df_y[['Close']].copy()
    pep_close.columns = ['Y']

    df = ko_close.merge(pep_close, left_index=True, right_index=True, how='inner')

    df = df.dropna()
    return df

def demo_pipeline():
    # --- Pick pair of commonly traded stocks, e.g. Coke and Pepsi ---
    ticker_x = 'BAC'   # Bank of America
    ticker_y = 'WFC'  # Wells Fargo

    # Load historical stock prices from yfinance
    df = load_stock_pair(ticker_x, ticker_y, start="2011-01-01", end="2020-12-31")
    print(f"Loaded {len(df)} daily bars for {ticker_x}/{ticker_y}")

    # Cointegration
    eg = engle_granger_test(df['Y'], df['X'])
    print('EG ADF p:', eg['adf_pvalue'])

    # Kalman: estimate dynamic beta & intercept
    beta, intercept = kf_hedge_ratio_pykalman(df['Y'], df['X'], n_iter=3,
                                              transition_cov=1e-6,
                                              observation_cov=1e-1)
    # Smooth beta to reduce high-frequency noise and avoid overtrading:
    beta_smoothed = beta.rolling(window=50, min_periods=1).mean()
    df['beta_kf'] = beta_smoothed

    # To avoid lookahead bias: use lagged beta 
    df['beta_kf_used'] = df['beta_kf'].shift(1).bfill()

    # Tradable spread
    df['spread'] = df['Y'] - df['beta_kf_used'] * df['X']

    oud = fit_ou_discrete(df['spread'])
    print('OU discrete map:', oud)
    kappa = oud.get('kappa', np.nan)
    if np.isfinite(kappa) and kappa > 0:
        half_life = np.log(2) / kappa
    else:
        half_life = np.nan
    print('Implied half-life (bars):', half_life)

    oum = fit_ou_mle(df['spread'], initial=(0.5, 0.0, 0.2))
    print('OU MLE:', oum)
    if oum['success']:
        hl = np.log(2) / oum['kappa'] if oum['kappa'] > 0 else np.nan
        print('OU MLE half-life (bars):', hl)

    roll_win = 200
    df['zscore'] = (df['spread'] - df['spread'].rolling(roll_win).mean()) / df['spread'].rolling(roll_win).std()
    df = df.dropna()

    bt, perf = generate_signals(df, zcol='zscore', entry=2.0, exit=0.5, capital_per_trade=1e5)
    print('Backtest perf (demo):', perf)

    plt.figure(figsize=(10, 4))
    plt.title('Spread (tradable): Y - beta_lag*X')
    bt['spread'].plot()
    plt.tight_layout()

    plt.figure(figsize=(10, 4))
    plt.title('Smoothed beta (kf)')
    bt['beta_kf'].plot()
    plt.tight_layout()

    plt.figure(figsize=(10, 4))
    plt.title('Cumulative PnL')
    bt['cum_pnl'].plot()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo', action='store_true')
    args = parser.parse_args()
    if args.demo:
        demo_pipeline()
    else:
        print('Options: --demo')
