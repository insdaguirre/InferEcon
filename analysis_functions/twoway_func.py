from __future__ import annotations

from typing import List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import io
from scipy import stats
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

display_name = "Twoway"

def _create_twoway_plot(x: pd.Series, y: pd.Series, x_label: str, y_label: str, 
                        title: str = "Twoway Plot") -> str:
    """Create a twoway plot with scatter and regression line."""
    plt.figure(figsize=(12, 10))
    
    # Create scatter plot
    plt.scatter(x, y, alpha=0.6, s=60, edgecolors='black', linewidth=0.5, 
                color='steelblue', label='Data Points')
    
    # Fit regression line
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    
    # Get regression line
    x_range = np.linspace(x.min(), x.max(), 100)
    X_range = sm.add_constant(x_range)
    y_pred = model.predict(X_range)
    
    # Plot regression line
    plt.plot(x_range, y_pred, 'r-', linewidth=3, alpha=0.8, 
             label=f'OLS: y = {model.params[1]:.3f}x + {model.params[0]:.3f}')
    
    # Add confidence intervals
    pred_ols = model.get_prediction(X_range)
    conf_int = pred_ols.conf_int()
    plt.fill_between(x_range, conf_int[:, 0], conf_int[:, 1], 
                     alpha=0.2, color='red', label='95% CI')
    
    # Add prediction intervals
    pred_int = model.get_prediction(X_range)
    pred_conf_int = pred_int.conf_int(alpha=0.32)  # 68% PI (roughly ±1 SD)
    plt.fill_between(x_range, pred_conf_int[:, 0], pred_conf_int[:, 1], 
                     alpha=0.1, color='blue', label='68% Prediction Interval')
    
    # Add statistics
    r_squared = model.rsquared
    adj_r_squared = model.rsquared_adj
    f_stat = model.fvalue
    f_pval = model.f_pvalue
    
    stats_text = f'R² = {r_squared:.4f}\nAdj. R² = {adj_r_squared:.4f}\nF = {f_stat:.2f}\np = {f_pval:.4f}'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, fontsize=11,
             verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", 
             facecolor="lightyellow", alpha=0.9))
    
    # Add correlation info
    corr, corr_p = stats.pearsonr(x, y)
    corr_text = f'r = {corr:.3f}\np = {corr_p:.4f}'
    plt.text(0.98, 0.98, corr_text, transform=plt.gca().transAxes, fontsize=11,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.9))
    
    # Add sample info
    sample_text = f'n = {len(x)}\nMean x = {x.mean():.3f}\nMean y = {y.mean():.3f}'
    plt.text(0.02, 0.02, sample_text, transform=plt.gca().transAxes, fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))
    
    # Customize plot
    plt.xlabel(x_label, fontsize=14, fontweight='bold')
    plt.ylabel(y_label, fontsize=14, fontweight='bold')
    plt.title(title, fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left', fontsize=11)
    
    # Add trend line equation
    slope = model.params[1]
    intercept = model.params[0]
    equation_text = f'y = {slope:.3f}x + {intercept:.3f}'
    plt.text(0.98, 0.02, equation_text, transform=plt.gca().transAxes, fontsize=12,
             horizontalalignment='right', bbox=dict(boxstyle="round,pad=0.3", 
             facecolor="white", alpha=0.9))
    
    plt.tight_layout()
    
    # Convert to base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
    buffer.seek(0)
    plt.close()
    
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return img_str

def _create_residual_plot(x: pd.Series, y: pd.Series, x_label: str, y_label: str) -> str:
    """Create residual plot for regression diagnostics."""
    plt.figure(figsize=(12, 10))
    
    # Fit regression
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    
    # Get residuals and fitted values
    residuals = model.resid
    fitted = model.fittedvalues
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Residuals vs Fitted
    ax1.scatter(fitted, residuals, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.8)
    ax1.set_xlabel('Fitted Values', fontweight='bold')
    ax1.set_ylabel('Residuals', fontweight='bold')
    ax1.set_title('Residuals vs Fitted Values', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 2. Residuals vs X
    ax2.scatter(x, residuals, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.8)
    ax2.set_xlabel(x_label, fontweight='bold')
    ax2.set_ylabel('Residuals', fontweight='bold')
    ax2.set_title('Residuals vs X Variable', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. Q-Q plot
    from scipy.stats import probplot
    probplot(residuals, dist="norm", plot=ax3)
    ax3.set_title('Normal Q-Q Plot of Residuals', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. Histogram of residuals
    ax4.hist(residuals, bins=20, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax4.axvline(residuals.mean(), color='red', linestyle='--', alpha=0.8, 
                 label=f'Mean: {residuals.mean():.3f}')
    ax4.set_xlabel('Residuals', fontweight='bold')
    ax4.set_ylabel('Frequency', fontweight='bold')
    ax4.set_title('Histogram of Residuals', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Convert to base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
    buffer.seek(0)
    plt.close()
    
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return img_str

def _create_regression_summary_table(x: pd.Series, y: pd.Series) -> pd.DataFrame:
    """Create regression summary table."""
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    
    # Basic statistics
    n = len(x)
    r_squared = model.rsquared
    adj_r_squared = model.rsquared_adj
    f_stat = model.fvalue
    f_pval = model.f_pvalue
    
    # Coefficient statistics
    slope = model.params[1]
    slope_se = model.bse[1]
    slope_t = model.tvalues[1]
    slope_p = model.pvalues[1]
    
    intercept = model.params[0]
    intercept_se = model.bse[0]
    intercept_t = model.tvalues[0]
    intercept_p = model.pvalues[0]
    
    # Create summary table
    summary_data = [
        {'Statistic': 'Observations', 'Value': n},
        {'Statistic': 'R-squared', 'Value': r_squared},
        {'Statistic': 'Adjusted R-squared', 'Value': adj_r_squared},
        {'Statistic': 'F-statistic', 'Value': f_stat},
        {'Statistic': 'F P-value', 'Value': f_pval},
        {'Statistic': 'Intercept', 'Value': intercept},
        {'Statistic': 'Intercept Std. Error', 'Value': intercept_se},
        {'Statistic': 'Intercept t-statistic', 'Value': intercept_t},
        {'Statistic': 'Intercept P-value', 'Value': intercept_p},
        {'Statistic': 'Slope', 'Value': slope},
        {'Statistic': 'Slope Std. Error', 'Value': slope_se},
        {'Statistic': 'Slope t-statistic', 'Value': slope_t},
        {'Statistic': 'Slope P-value', 'Value': slope_p}
    ]
    
    summary_df = pd.DataFrame(summary_data)
    summary_df['Value'] = summary_df['Value'].round(4)
    
    return summary_df

def apply(df: pd.DataFrame) -> List[dict]:
    """Create twoway plots with regression lines and diagnostics."""
    outputs = []
    
    # Find numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        return [{"type": "text", "title": "Twoway", "data": "Need at least 2 numeric variables for twoway plots."}]
    
    # Use first column as y, others as x variables
    y_col = numeric_cols[0]
    x_cols = numeric_cols[1:min(5, len(numeric_cols))]  # Limit to 4 x variables
    
    # Create twoway plots for each x variable
    for i, x_col in enumerate(x_cols):
        # Remove missing values
        mask = ~(df[y_col].isna() | df[x_col].isna())
        x_clean = df[x_col][mask]
        y_clean = df[y_col][mask]
        
        if len(x_clean) > 0:
            # 1. Main twoway plot
            twoway_plot = _create_twoway_plot(
                x_clean, y_clean, 
                x_col, y_col, 
                f"Twoway: {y_col} vs {x_col}"
            )
            
            outputs.append({
                "type": "image", 
                "title": f"Twoway Plot {i+1}: {y_col} vs {x_col}", 
                "data": twoway_plot
            })
            
            # 2. Residual diagnostics plot
            residual_plot = _create_residual_plot(x_clean, y_clean, x_col, y_col)
            outputs.append({
                "type": "image", 
                "title": f"Residual Diagnostics: {y_col} vs {x_col}", 
                "data": residual_plot
            })
            
            # 3. Regression summary table
            summary_table = _create_regression_summary_table(x_clean, y_clean)
            outputs.append({
                "type": "table", 
                "title": f"Regression Summary: {y_col} vs {x_col}", 
                "data": summary_table
            })
    
    # 4. Overall interpretation guide
    interpretation = f"""
**Twoway Plot Analysis for {y_col} vs X variables:**

**What the plots show:**
1. **Scatter Plot**: Raw data points showing the relationship
2. **Regression Line**: Best-fit OLS line with equation y = mx + b
3. **Confidence Intervals**: 95% confidence bands around the regression line
4. **Prediction Intervals**: 68% prediction bands for individual observations

**Regression Diagnostics:**
- **Residuals vs Fitted**: Check for homoskedasticity (constant variance)
- **Residuals vs X**: Check for systematic patterns
- **Q-Q Plot**: Check for normality of residuals
- **Histogram**: Distribution of residuals

**Key Statistics:**
- **R²**: Proportion of variance explained (higher is better)
- **F-statistic**: Overall model significance
- **t-statistics**: Individual coefficient significance
- **Standard Errors**: Precision of coefficient estimates

**Interpretation Guidelines:**
- **Slope**: Change in y per unit change in x
- **Intercept**: Expected y when x = 0
- **R² ≥ 0.7**: Good fit, **R² ≥ 0.9**: Excellent fit
- **P < 0.05**: Statistically significant relationship
    """
    
    outputs.append({
        "type": "text", 
        "title": "Twoway Plot Interpretation Guide", 
        "data": interpretation
    })
    
    return outputs
