"""Data loader utilities.
Supports: CSV ingestion, simple resampling, basic cleaning. Add connectors (Polygon, Bloomberg, WRDS) in your environment.
"""
import pandas as pd
import os




def load_csv_pair(path_x, path_y, ts_col='timestamp', price_col='price', parse_dates=True):
    x = pd.read_csv(path_x, parse_dates=[ts_col])
    y = pd.read_csv(path_y, parse_dates=[ts_col])
    x = x.set_index(ts_col)[[price_col]].rename(columns={price_col: 'X'})
    y = y.set_index(ts_col)[[price_col]].rename(columns={price_col: 'Y'})
    df = x.join(y, how='inner')
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df




def resample_to_minutes(df, rule='1T'):
    return df.resample(rule).last().ffill()