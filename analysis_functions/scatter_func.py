from __future__ import annotations

from typing import List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import io
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

display_name = "Scatter"

def _create_scatter_plot(x: pd.Series, y: pd.Series, x_label: str, y_label: str, 
                         title: str = "Scatter Plot") -> str:
    """Create a scatter plot and return as base64 string."""
    plt.figure(figsize=(10, 8))
    
    # Create scatter plot
    plt.scatter(x, y, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    
    # Add trend line
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    plt.plot(x, p(x), "r--", alpha=0.8, linewidth=2, label=f'Trend: y = {z[0]:.3f}x + {z[1]:.3f}')
    
    # Add correlation coefficient
    corr, p_value = stats.pearsonr(x, y)
    plt.text(0.05, 0.95, f'r = {corr:.3f}\np = {p_value:.4f}', 
             transform=plt.gca().transAxes, fontsize=12, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Customize plot
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add statistics text
    stats_text = f'n = {len(x)}\nMean x = {x.mean():.3f}\nMean y = {y.mean():.3f}'
    plt.text(0.95, 0.05, stats_text, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
    
    plt.tight_layout()
    
    # Convert to base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
    buffer.seek(0)
    plt.close()
    
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return img_str

def _create_multiple_scatters(df: pd.DataFrame, y_col: str, x_cols: List[str]) -> List[str]:
    """Create multiple scatter plots for different x variables."""
    plots = []
    
    for x_col in x_cols:
        # Remove missing values
        mask = ~(df[y_col].isna() | df[x_col].isna())
        x_clean = df[x_col][mask]
        y_clean = df[y_col][mask]
        
        if len(x_clean) > 0:
            plot_str = _create_scatter_plot(
                x_clean, y_clean, 
                x_col, y_col, 
                f"Scatter: {y_col} vs {x_col}"
            )
            plots.append(plot_str)
    
    return plots

def _create_correlation_matrix_plot(df: pd.DataFrame, numeric_cols: List[str]) -> str:
    """Create a correlation matrix heatmap."""
    plt.figure(figsize=(10, 8))
    
    # Calculate correlation matrix
    corr_matrix = df[numeric_cols].corr()
    
    # Create heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": .8})
    
    plt.title("Correlation Matrix", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Convert to base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
    buffer.seek(0)
    plt.close()
    
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return img_str

def _create_pairplot(df: pd.DataFrame, numeric_cols: List[str]) -> str:
    """Create a pairplot for multiple variables."""
    plt.figure(figsize=(12, 10))
    
    # Create pairplot
    pair_data = df[numeric_cols].dropna()
    if len(pair_data) > 0:
        sns.pairplot(pair_data, diag_kind='kde', plot_kws={'alpha': 0.6, 's': 30})
        plt.suptitle("Pairwise Relationships", y=1.02, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        plt.close()
        
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return img_str
    
    return ""

def apply(df: pd.DataFrame) -> List[dict]:
    """Create scatter plots and related visualizations."""
    outputs = []
    
    # Find numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        return [{"type": "text", "title": "Scatter", "data": "Need at least 2 numeric variables for scatter plots."}]
    
    # Use first column as y, others as x variables
    y_col = numeric_cols[0]
    x_cols = numeric_cols[1:min(6, len(numeric_cols))]  # Limit to 5 x variables
    
    # 1. Individual scatter plots
    scatter_plots = _create_multiple_scatters(df, y_col, x_cols)
    
    for i, plot_str in enumerate(scatter_plots):
        outputs.append({
            "type": "image", 
            "title": f"Scatter Plot {i+1}: {y_col} vs {x_cols[i]}", 
            "data": plot_str
        })
    
    # 2. Correlation matrix heatmap
    if len(numeric_cols) >= 2:
        corr_plot = _create_correlation_matrix_plot(df, numeric_cols)
        outputs.append({
            "type": "image", 
            "title": "Correlation Matrix Heatmap", 
            "data": corr_plot
        })
    
    # 3. Pairplot for first few variables
    if len(numeric_cols) >= 3:
        pair_plot = _create_pairplot(df, numeric_cols[:4])  # Limit to 4 variables for pairplot
        if pair_plot:
            outputs.append({
                "type": "image", 
                "title": "Pairwise Relationships", 
                "data": pair_plot
            })
    
    # 4. Summary statistics for scatter plots
    summary_data = []
    for x_col in x_cols:
        mask = ~(df[y_col].isna() | df[x_col].isna())
        x_clean = df[x_col][mask]
        y_clean = df[y_col][mask]
        
        if len(x_clean) > 0:
            corr, p_value = stats.pearsonr(x_clean, y_clean)
            summary_data.append({
                'X Variable': x_col,
                'N': len(x_clean),
                'Correlation (r)': corr,
                'P-value': p_value,
                'Significance': 'Yes' if p_value < 0.05 else 'No'
            })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_df['Correlation (r)'] = summary_df['Correlation (r)'].round(4)
        summary_df['P-value'] = summary_df['P-value'].round(4)
        
        outputs.append({
            "type": "table", 
            "title": "Scatter Plot Summary Statistics", 
            "data": summary_df
        })
    
    # 5. Interpretation guide
    interpretation = f"""
**Scatter Plot Analysis for {y_col} vs X variables:**

**What to look for:**
1. **Linear vs Non-linear**: Check if points follow a straight line or curve
2. **Strength**: Points closer to line = stronger relationship
3. **Direction**: Positive slope = positive correlation, negative slope = negative correlation
4. **Outliers**: Points far from the main pattern
5. **Clustering**: Groups or patterns in the data

**Correlation Interpretation:**
- |r| ≥ 0.9: Very strong
- |r| ≥ 0.7: Strong  
- |r| ≥ 0.5: Moderate
- |r| ≥ 0.3: Weak
- |r| < 0.3: Very weak

**Significance**: P < 0.05 indicates statistically significant correlation
    """
    
    outputs.append({
        "type": "text", 
        "title": "Scatter Plot Interpretation Guide", 
        "data": interpretation
    })
    
    return outputs
