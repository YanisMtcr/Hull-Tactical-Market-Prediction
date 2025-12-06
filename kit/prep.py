import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

liste_log = ["E11","E12","E13","E14","E4","E6","M16","S10","S11","S4","V1","V11","V12","V6","V8"]

def fillna(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].ffill().bfill().fillna(0)
    return df


def columns_drop(df, droplist, y_field):
    y = df[y_field].values
    cols_to_drop = [c for c in droplist + [y_field] if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    return df, y


def apply_log(df,method='log'):
    df_new = df.copy()
    for col in liste_log:
        if method == 'log1p':
            df_new[col] = np.log1p(df_new[col])

        elif method == 'log':
            df_new[col] = np.log(df_new[col] + 1e-9) 
    return df_new



def create_features(df, price_col='forward_returns', lags=[1, 2, 3, 5, 21]):
    df_feat = df.copy()
    past_returns = df_feat[price_col].shift(1)
    for lag in lags:
        df_feat[f'ret_lag_{lag}'] = past_returns.shift(lag - 1)

    df_feat['volatility_5'] = past_returns.rolling(window=5).std()
    df_feat['volatility_21'] = past_returns.rolling(window=21).std()
    

    df_feat['vola_ratio'] = df_feat['volatility_5'] / (df_feat['volatility_21'] + 1e-9)

    rolling_mean_21 = past_returns.rolling(window=21).mean()
    df_feat['z_score_21'] = (past_returns - rolling_mean_21) / (df_feat['volatility_21'] + 1e-9)

    df_feat['skew_10'] = past_returns.rolling(window=10).skew()

    df_feat['momentum_5'] = past_returns.rolling(window=5).sum()
    df_feat['momentum_21'] = past_returns.rolling(window=21).sum()

    df_feat['mom_diff'] = df_feat['momentum_5'] - df_feat['momentum_21']

    return df_feat



def create_target(df, return_col='forward_returns', threshold=0.015, horizon=5):
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=horizon)
    future_cum_return = df[return_col].rolling(window=indexer).sum()
    
    conditions = [
        (future_cum_return > threshold),  
        (future_cum_return < -threshold)  
    ]
    choices = [2, 0] 

    df['target'] = np.select(conditions, choices, default=1)
    
    return df

class Scaler(BaseEstimator, TransformerMixin):
    def __init__(self, limits=(0.01, 0.99)):
        self.limits = limits
        self.scaler = StandardScaler()
        self.lower_bounds = {}
        self.upper_bounds = {}
        self.numeric_cols = []

    def fit(self, X, y=None):
        if not hasattr(X, "columns"):
            X = pd.DataFrame(X)
        self.numeric_cols = X.select_dtypes(include=[np.number]).columns
        
        for col in self.numeric_cols:
            self.lower_bounds[col] = X[col].quantile(self.limits[0])
            self.upper_bounds[col] = X[col].quantile(self.limits[1])
        
        X_temp = X.copy()
        for col in self.numeric_cols:
            X_temp[col] = X_temp[col].clip(lower=self.lower_bounds[col], upper=self.upper_bounds[col])
            
        self.scaler.fit(X_temp[self.numeric_cols])
        return self

    def transform(self, X):
        if not hasattr(X, "columns"):
            X = pd.DataFrame(X)
    
        X_new = X.copy()
        for col in self.numeric_cols:
            if col in X_new.columns:
                X_new[col] = X_new[col].clip(lower=self.lower_bounds[col], upper=self.upper_bounds[col])
        
        return self.scaler.transform(X_new[self.numeric_cols])

def categorize_y(y, bin=0.003): 
    y = pd.to_numeric(y, errors='coerce')
    y_class = np.where(y < -bin, 0,
                       np.where(y <= bin, 1, 2))
    if isinstance(y, pd.Series):
        y_class = y_class.astype(float)
        y_class[y.isna()] = np.nan
        
    return y_class


def drop_flat_rows(df, threshold=0.7):
    numeric_df = df.select_dtypes(include=[np.number])
    zeros_count = (numeric_df == 0).sum(axis=1)
    zeros_pct = zeros_count / numeric_df.shape[1]
    df_clean = df[zeros_pct < threshold].copy()
    return df_clean