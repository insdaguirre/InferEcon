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

display_name = "Avplot"

def _create_avplot(x: pd.Series, y: pd.Series, other_x: pd.DataFrame, 
                   x_label: str, y_label: str, title: str = "Added-Variable Plot") -> str:
    """Create an added-variable plot (partial regression plot)."""
    plt.figure(figsize=(12, 10))
    
    # Step 1: Regress y on other X variables (excluding x)
    if len(other_x.columns) > 0:
        X_other = sm.add_constant(other_x)
        model_y_other = sm.OLS(y, X_other).fit()
        y_resid = model_y_other.resid
    else:
        y_resid = y - y.mean()
    
    # Step 2: Regress x on other X variables
    if len(other_x.columns) > 0:
        X_other = sm.add_constant(other_x)
        model_x_other = sm.OLS(x, X_other).fit()
        x_resid = model_x_other.resid
    else:
        x_resid = x - x.mean()
    
    # Step 3: Create the added-variable plot
    plt.scatter(x_resid, y_resid, alpha=0.6, s=60, edgecolors='black', linewidth=0.5,
                color='darkgreen', label='Partial Residuals')
    
    # Step 4: Fit regression line through the partial residuals
    X_partial = sm.add_constant(x_resid)
    model_partial = sm.OLS(y_resid, X_partial).fit()
    
    # Get regression line
    x_range = np.linspace(x_resid.min(), x_resid.max(), 100)
    X_range = sm.add_constant(x_range)
    y_pred = model_partial.predict(X_range)
    
    # Plot regression line
    plt.plot(x_range, y_pred, 'r-', linewidth=3, alpha=0.8,
             label=f'Partial: y = {model_partial.params[1]:.3f}x + {model_partial.params[0]:.3f}')
    
    # Add confidence intervals
    pred_ols = model_partial.get_prediction(X_range)
    conf_int = pred_ols.conf_int()
    plt.fill_between(x_range, conf_int[:, 0], conf_int[:, 1],
                     alpha=0.2, color='red', label='95% CI')
    
    # Add zero lines
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
    
    # Add statistics
    partial_r_squared = model_partial.rsquared
    partial_f_stat = model_partial.fvalue
    partial_f_pval = model_partial.f_pvalue
    
    # Partial correlation
    partial_corr, partial_corr_p = stats.pearsonr(x_resid, y_resid)
    
    stats_text = f'Partial R² = {partial_r_squared:.4f}\nPartial F = {partial_f_stat:.2f}\np = {partial_f_pval:.4f}'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, fontsize=11,
             verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3",
             facecolor="lightyellow", alpha=0.9))
    
    # Partial correlation info
    corr_text = f'Partial r = {partial_corr:.3f}\np = {partial_corr_p:.4f}'
    plt.text(0.98, 0.98, corr_text, transform=plt.gca().transAxes, fontsize=11,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.9))
    
    # Sample info
    sample_text = f'n = {len(x_resid)}\nMean x_resid = {x_resid.mean():.3f}\nMean y_resid = {y_resid.mean():.3f}'
    plt.text(0.02, 0.02, sample_text, transform=plt.gca().transAxes, fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))
    
    # Customize plot
    plt.xlabel(f'Residuals of {x_label} | Other Variables', fontsize=14, fontweight='bold')
    plt.ylabel(f'Residuals of {y_label} | Other Variables', fontsize=14, fontweight='bold')
    plt.title(title, fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left', fontsize=11)
    
    # Add partial regression equation
    slope = model_partial.params[1]
    intercept = model_partial.params[0]
    equation_text = f'Partial: y = {slope:.3f}x + {intercept:.3f}'
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

def _create_partial_correlation_table(x: pd.Series, y: pd.Series, other_x: pd.DataFrame) -> pd.DataFrame:
    """Create table showing partial correlations."""
    if len(other_x.columns) == 0:
        # Simple correlation if no other variables
        corr, p_value = stats.pearsonr(x, y)
        return pd.DataFrame({
            'Variable': [x.name],
            'Type': ['Simple Correlation'],
            'Correlation': [corr],
            'P-value': [p_value],
            'Significance': ['Yes' if p_value < 0.05 else 'No']
        })
    
    # Calculate partial correlations
    partial_corrs = []
    
    for col in other_x.columns:
        # Regress y on other variables (excluding current col)
        other_cols = [c for c in other_x.columns if c != col]
        if other_cols:
            X_other = sm.add_constant(other_x[other_cols])
            model_y = sm.OLS(y, X_other).fit()
            y_resid = model_y.resid
        else:
            y_resid = y - y.mean()
        
        # Regress x on other variables (excluding current col)
        if other_cols:
            X_other = sm.add_constant(other_x[other_cols])
            model_x = sm.OLS(x, X_other).fit()
            x_resid = model_x.resid
        else:
            x_resid = x - x.mean()
        
        # Calculate partial correlation
        partial_corr, partial_p = stats.pearsonr(x_resid, y_resid)
        
        partial_corrs.append({
            'Variable': col,
            'Type': 'Partial Correlation',
            'Correlation': partial_corr,
            'P-value': partial_p,
            'Significance': 'Yes' if partial_p < 0.05 else 'No'
        })
    
    # Add the main variable
    if len(other_x.columns) > 0:
        X_other = sm.add_constant(other_x)
        model_y = sm.OLS(y, X_other).fit()
        y_resid = model_y.resid
        
        model_x = sm.OLS(x, X_other).fit()
        x_resid = model_x.resid
        
        partial_corr, partial_p = stats.pearsonr(x_resid, y_resid)
        
        partial_corrs.insert(0, {
            'Variable': x.name,
            'Type': 'Partial Correlation',
            'Correlation': partial_corr,
            'P-value': partial_p,
            'Significance': 'Yes' if partial_p < 0.05 else 'No'
        })
    
    partial_df = pd.DataFrame(partial_corrs)
    partial_df['Correlation'] = partial_df['Correlation'].round(4)
    partial_df['P-value'] = partial_df['P-value'].round(4)
    
    return partial_df

def apply(df: pd.DataFrame) -> List[dict]:
    """Create added-variable plots (partial regression plots)."""
    outputs = []
    
    # Find numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        return [{"type": "text", "title": "Avplot", "data": "Need at least 2 numeric variables for added-variable plots."}]
    
    # Use first column as y, others as x variables
    y_col = numeric_cols[0]
    x_cols = numeric_cols[1:min(6, len(numeric_cols))]  # Limit to 5 x variables
    
    # Create added-variable plots for each x variable
    for i, x_col in enumerate(x_cols):
        # Remove missing values
        mask = ~(df[y_col].isna() | df[x_col].isna())
        y_clean = df[y_col][mask]
        x_clean = df[x_col][mask]
        
        # Get other x variables
        other_x_cols = [col for col in x_cols if col != x_col]
        other_x_clean = df[other_x_cols][mask] if other_x_cols else pd.DataFrame()
        
        if len(x_clean) > 0:
            try:
                # 1. Added-variable plot
                avplot_img = _create_avplot(
                    x_clean, y_clean, other_x_clean,
                    x_col, y_col,
                    f"Added-Variable Plot: {y_col} vs {x_col}"
                )
            
                outputs.append({
                    "type": "image", 
                    "title": f"Added-Variable Plot {i+1}: {y_col} vs {x_col}", 
                    "data": avplot_img
                })
                
                # 2. Partial correlation table
                partial_corr_table = _create_partial_correlation_table(x_clean, y_clean, other_x_clean)
                outputs.append({
                    "type": "table", 
                    "title": f"Partial Correlations: {y_col} vs {x_col}", 
                    "data": partial_corr_table
                })
            except Exception as e:
                outputs.append({
                    "type": "text",
                    "title": f"Error in {x_col} analysis",
                    "data": f"Could not create added-variable plot for {x_col}: {str(e)}"
                })
    
    # 3. Interpretation guide
    interpretation = f"""
**Added-Variable Plot (Partial Regression Plot) Analysis:**

**What these plots show:**
1. **Partial Residuals**: Residuals after controlling for other variables
2. **Partial Regression Line**: Relationship between x and y, holding other variables constant
3. **Zero Lines**: Reference lines showing no partial relationship
4. **Confidence Bands**: Uncertainty around the partial relationship

**Key Concepts:**
- **Partial Correlation**: Correlation between x and y after removing effects of other variables
- **Partial R²**: Proportion of variance in y_resid explained by x_resid
- **Partial F-test**: Significance of the partial relationship

**Interpretation:**
- **Slope**: Change in y per unit change in x, holding other variables constant
- **R²**: How much of the remaining variance in y is explained by x
- **Significance**: Whether the partial relationship is statistically significant

**What to look for:**
1. **Linear Pattern**: Points should follow a straight line
2. **Constant Variance**: Spread should be roughly uniform
3. **No Outliers**: Points far from the pattern may be influential
4. **Slope Direction**: Positive/negative indicates direction of partial relationship

**Advantages over simple scatter plots:**
- Controls for confounding variables
- Shows unique contribution of each variable
- Reveals relationships masked by other variables
- More accurate coefficient interpretation
    """
    
    outputs.append({
        "type": "text", 
        "title": "Added-Variable Plot Interpretation Guide", 
        "data": interpretation
    })
    
    return outputs
