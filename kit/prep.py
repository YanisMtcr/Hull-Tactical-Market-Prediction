import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

liste_log = ["E11","E12","E13","E14","E4","E6","M16","S10","S11","S4","V1","V11","V12","V6","V8"] # used the distribution seen with the profiling to select 

def fillna(df):
    """
    Fill missing values in numeric columns.

    Uses forward fill, then backward fill, then replaces remaining NaN with 0.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with numeric NaNs filled.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].ffill().bfill().fillna(0)
    return df


def columns_drop(df, droplist, y_field):
    """
    Drop selected columns and extract the target.

    Args:
        df: Input DataFrame.
        droplist: Columns to remove if present.
        y_field: Target column name.

    Returns:
        Tuple of (features_df, y_array).
    """
    y = df[y_field].values
    cols_to_drop = [c for c in droplist + [y_field] if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    return df, y


def apply_log(df,method='log'):
    """
    Apply a log transform to predefined columns.

    Args:
        df: Input DataFrame.
        method: 'log' or 'log1p'.

    Returns:
        A copy of df with transformed columns when available.
    """
    df_new = df.copy()
    for col in liste_log:
        if method == 'log1p':
            df_new[col] = np.log1p(df_new[col])

        elif method == 'log':
            df_new[col] = np.log(df_new[col] + 1e-9) 
    return df_new



def create_features(df, price_col='forward_returns', lags=[1, 2, 3, 5, 21]):
    """
    Create lagged return and basic time-series features.

    Adds lagged returns, short/long volatility, volatility ratio,
    z-score, skewness, and simple momentum features.

    Args:
        df: Input DataFrame.
        price_col: Column used to compute past returns.
        lags: List of lag values.

    Returns:
        DataFrame with new feature columns.
    """

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


class Scaler(BaseEstimator, TransformerMixin):
    """
    Clip numeric features to quantile bounds and apply standard scaling.

    Limits are learned during fit and applied during transform.
    """
    def __init__(self, limits=(0.01, 0.99)):
        self.limits = limits
        self.scaler = StandardScaler()
        self.lower_bounds = {}
        self.upper_bounds = {}
        self.numeric_cols = []

    def fit(self, X, y=None):
        """
        Fit clipping bounds and the internal StandardScaler.

        Args:
            X: Feature matrix (DataFrame or array-like).
            y: Ignored.

        Returns:
            Self.
        """
        if not hasattr(X, "columns"):
            X = pd.DataFrame(X)
        self.numeric_cols = X.select_dtypes(include=[np.number]).columns
        
        for col in self.numeric_cols: 
            self.lower_bounds[col] = X[col].quantile(self.limits[0])
            self.upper_bounds[col] = X[col].quantile(self.limits[1])
        
        X_temp = X.copy()
        for col in self.numeric_cols:
            X_temp[col] = X_temp[col].clip(lower=self.lower_bounds[col], upper=self.upper_bounds[col]) # Learn per-column quantile bounds for clipping before scaling
            
        self.scaler.fit(X_temp[self.numeric_cols]) 
        return self

    def transform(self, X):
        """
        Clip numeric columns using learned bounds and scale them.

        Args:
            X: Feature matrix (DataFrame or array-like).

        Returns:
            Numpy array of scaled numeric features.
        """
        if not hasattr(X, "columns"):
            X = pd.DataFrame(X)
    
        X_new = X.copy()
        for col in self.numeric_cols:
            if col in X_new.columns:
                X_new[col] = X_new[col].clip(lower=self.lower_bounds[col], upper=self.upper_bounds[col]) # Apply the same clipping bounds learned in fit, then scale
        
        return self.scaler.transform(X_new[self.numeric_cols])


def categorize_y(y, bin=0.003): 
    """
    Convert continuous returns into 3 classes.

    Args:
        y: Array-like or Series of returns.
        bin: Neutral band size around zero.

    Returns:
        Array of classes (0, 1, 2), preserving NaN for Series input.
    """
    y = pd.to_numeric(y, errors='coerce')
    y_class = np.where(y < -bin, 0,
                       np.where(y <= bin, 1, 2))
    if isinstance(y, pd.Series):
        y_class = y_class.astype(float)
        y_class[y.isna()] = np.nan
        
    return y_class


def drop_flat_rows(df, threshold=0.7):
    """
    Remove rows with too many zeros across numeric columns.

    Args:
        df: Input DataFrame.
        threshold: Max allowed fraction of zeros per row.

    Returns:
        Filtered DataFrame.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    zeros_count = (numeric_df == 0).sum(axis=1)
    zeros_pct = zeros_count / numeric_df.shape[1]
    df_clean = df[zeros_pct < threshold].copy()
    return df_clean