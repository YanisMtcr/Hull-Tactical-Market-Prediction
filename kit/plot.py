import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display 
import shap # type: ignore
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    make_scorer
)
import seaborn as sns
from sklearn.inspection import permutation_importance


def set_global_plot_style():
    """Sets a unified style for all matplotlib and seaborn plots"""

    plt.style.use('seaborn-v0_8-whitegrid')
    
 
    plt.rcParams.update({
        'figure.figsize': (10, 6),       
        'figure.dpi': 100,
        'axes.titlesize': 18,            
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'grid.alpha': 0.6,               
        'grid.linestyle': '-',           
        'legend.fontsize': 12,
        'font.family': 'sans-serif',    
        'axes.edgecolor': '0.3',        
        'axes.linewidth': 1.2,
        'lines.linewidth': 2.0          
    })


    sns.set_context("talk") 
    sns.set_palette('deep')

set_global_plot_style()

def calculate_max_drawdown(cumulative_returns):
    """
    Compute the maximum drawdown from a cumulative return series.

    Args:
        cumulative_returns: Series of cumulative wealth values.

    Returns:
        The minimum drawdown value.
    """
    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - running_max) / running_max
    return drawdown.min()

def calculate_full_metrics(returns_series, cum_returns, name, risk_free_rate_series):
    """
    Compute basic performance metrics for a strategy.

    Args:
        returns_series: Period returns of the strategy.
        cum_returns: Cumulative wealth series (base 100).
        name: Strategy name.
        risk_free_rate_series: Risk-free rate aligned with returns.

    Returns:
        Dict with total return, Sharpe, annualized volatility and max drawdown.
    """
    total_return = cum_returns.iloc[-1] / 100 - 1
    vol_ann = returns_series.std() * np.sqrt(252)
    max_dd = calculate_max_drawdown(cum_returns)
    excess_ret = returns_series - risk_free_rate_series.values
    sharpe = excess_ret.mean() / (returns_series.std() + 1e-9) * np.sqrt(252)

    return {
        'Strategy': name,
        'Total Return': total_return,
        'Sharpe Ratio': sharpe,
        'Annualized Volatility': vol_ann,
        'Max Drawdown': max_dd
    }

def run_multi_threshold_backtest(model, X_test, df_full, thresholds=[0.32, 0.33, 0.34, 0.35]):
    """
    Run a simple backtest using multiple probability thresholds.

    Builds wealth curves for each threshold and compares against buy & hold.
    Displays a formatted metrics table and saves results to CSV.

    Args:
        model: Fitted classifier with predict_proba.
        X_test: Test features.
        df_full: Full DataFrame containing forward returns and risk-free rate.
        thresholds: List of thresholds to test.
    Returns:
        None.
    """
    subset = df_full.loc[X_test.index]
    market_ret = subset['forward_returns'].astype(float)
    risk_free = subset['risk_free_rate'].astype(float)

    probas = model.predict_proba(X_test)
    p_baisse = probas[:, 0]

    wealth_curves = {}
    metrics_all = []

    for thr in thresholds:
        pos = np.where(p_baisse > thr, 0.0, 1.0)
        ret = (1 - pos) * risk_free.values + pos * market_ret.values

        ret_series = pd.Series(ret, index=X_test.index)
        cum = ((1 + ret_series).cumprod() * 100)

        wealth_curves[f"Model (thr={thr})"] = cum

        metrics_all.append(
            calculate_full_metrics(ret_series, cum, f"Model (thr={thr})", risk_free)
        )

    cum_market = ((1 + market_ret).cumprod() * 100)
    wealth_curves["Market (Buy & Hold)"] = cum_market

    metrics_all.append(
        calculate_full_metrics(market_ret, cum_market, "Market (Buy & Hold)", risk_free)
    )

    df_metrics = pd.DataFrame(metrics_all).set_index("Strategy")

    styled_df = df_metrics.style.format({
        'Total Return': '{:.2%}',
        'Annualized Volatility': '{:.2%}',
        'Max Drawdown': '{:.2%}',
        'Sharpe Ratio': '{:.2f}'
    }).background_gradient(cmap='RdYlGn', subset=['Sharpe Ratio'])

    display(styled_df)

    df_metrics.to_csv("results_trading.csv")

    plt.figure() 

    plt.plot(cum_market.index, cum_market.values, label="Market (Buy & Hold)",
             color="black", linestyle="--", linewidth=2, alpha=0.7)

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for (name, curve), color in zip(wealth_curves.items(), colors):
        if "Market" not in name:
            plt.plot(curve.index, curve.values, label=name, linewidth=2.2, color=color)

    plt.title("Wealth Curves — Multi-Threshold Voting Model") 
    plt.ylabel("Wealth (Base 100)")
    plt.legend()
    plt.show()



def shap_catboost(vote_model, X_test, classe ,sample_size=500, top_n=15 ):
        """
        Compute and plot SHAP summary for the CatBoost part of a voting model.

        Args:
            vote_model: VotingClassifier containing a CatBoost pipeline named "cat".
            X_test: Test features.
            classe: Class index to explain.
            sample_size: Max number of rows to sample.
            top_n: Number of top features to display.

        Returns:
            Tuple of (all_class_shap_values, X_sample).
        """
        cat = vote_model.named_estimators_["cat"].named_steps["model"]

        feature_names = cat.feature_names_
        X_aligned = X_test[feature_names]  # Align X_test to CatBoost feature order

        sample_size = min(sample_size, len(X_aligned))
        X_sample = X_aligned.sample(sample_size, random_state=42)

        explainer = shap.TreeExplainer(cat)
        shap_raw = explainer.shap_values(X_sample)  

        shap_fixed = np.transpose(shap_raw, (2, 0, 1))  # Reorder SHAP output to (class, samples, features) for easier slicing

        sv = shap_fixed[classe] 

        importance = np.abs(sv).mean(axis=0)

        idx = np.argsort(importance)[-top_n:][::-1]
        top_features = X_sample.columns[idx]

        top_values = sv[:, idx]

        plt.figure()
        shap.summary_plot(
            top_values,
            X_sample[top_features],
            feature_names=top_features,
            plot_type="dot",
            max_display=top_n,
            color=plt.cm.coolwarm,
            alpha=0.8,
            show=False
        )
        plt.xlabel("SHAP Value")
        plt.show()
        return shap_fixed, X_sample


def display_confusion_matrix(y_true, y_pred, title, labels, cmap):
    """
    Plot a confusion matrix with counts and percentages.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        title: Plot title.
        labels: Class names for axes.
        cmap: Matplotlib/seaborn colormap.

    Returns:
        None.
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
    group_percentages = ["{0:.1%}".format(value) for value in cm_norm.flatten()]
    box_labels = [f"{c}\n({p})" for c, p in zip(group_counts, group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cm.shape)

    plt.figure() 
    sns.heatmap(
        cm_norm, 
        annot=box_labels, 
        fmt='', 
        cmap=cmap, 
        xticklabels=labels, 
        yticklabels=labels,
        cbar=False
    )
    plt.title(title)
    plt.ylabel('True Label (Reality)')
    plt.xlabel('Predicted Label')
    plt.show()



def calculate_and_plot_permutation_importance(model, X_test, y_test, num_features=10):
    """
    Compute and plot permutation importance for a fitted model.
    Args:
        model: Fitted estimator.
        X_test: Test features.
        y_test: Test labels.
        num_features: Number of top features to display.

    Returns:
        DataFrame of top permutation importances.
    """
    result = permutation_importance(
        model, 
        X_test, y_test, 
        n_repeats=20, 
        random_state=42, 
        n_jobs=-1,
        scoring=make_scorer(accuracy_score)
    )

    df_importance = pd.DataFrame({
        'Feature': X_test.columns,
        'Mean Drop': result.importances_mean,
        'Std': result.importances_std
    })

    df_importance = df_importance.sort_values(
        by='Mean Drop', ascending=False
    ).head(num_features)
    
    plt.figure()

    plt.barh(
        df_importance['Feature'],
        df_importance['Mean Drop'],
        color="steelblue"
    )

    plt.gca().invert_yaxis()
    plt.title("Permutation Importance") 
    plt.xlabel("Mean Decrease") 
    plt.show()
    
    return df_importance