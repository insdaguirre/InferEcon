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

display_name = "Margins"

def _create_marginal_effects_plot(df: pd.DataFrame, y_col: str, x_cols: List[str]) -> str:
    """Create marginal effects plot showing how y changes with each x variable."""
    plt.figure(figsize=(15, 12))
    
    # Remove missing values
    mask = ~(df[y_col].isna() | df[x_cols].isna().any(axis=1))
    y_clean = df[y_col][mask]
    X_clean = df[x_cols][mask]
    
    if len(y_clean) == 0:
        return ""
    
    # Fit full model
    X_with_const = sm.add_constant(X_clean)
    model = sm.OLS(y_clean, X_with_const).fit()
    
    # Calculate marginal effects (partial derivatives)
    n_points = 50
    marginal_effects = {}
    
    for col in x_cols:
        # Create range of values for this variable
        x_range = np.linspace(X_clean[col].min(), X_clean[col].max(), n_points)
        
        # Hold other variables at their means
        X_means = X_clean.mean().copy()
        marginal_y = []
        
        for x_val in x_range:
            X_means[col] = x_val
            # Create prediction array with correct dimensions
            # We need to ensure the order matches the original X_clean columns
            X_pred_array = np.array([[X_means[col_name] for col_name in x_cols]])
            # Manual constant addition since sm.add_constant has issues with numpy arrays
            X_pred = np.column_stack([np.ones(1), X_pred_array])
            y_pred = model.predict(X_pred)[0]
            marginal_y.append(y_pred)
        
        marginal_effects[col] = {
            'x_values': x_range,
            'y_values': marginal_y,
            'coefficient': model.params[col],
            'std_error': model.bse[col]
        }
    
    # Create subplots
    n_vars = len(x_cols)
    cols = min(3, n_vars)
    rows = (n_vars + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    if n_vars == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.reshape(1, -1)
    else:
        axes = axes.reshape(rows, cols)
    
    for i, col in enumerate(x_cols):
        row = i // cols
        col_idx = i % cols
        ax = axes[row, col_idx]
        
        # Plot marginal effect
        x_vals = marginal_effects[col]['x_values']
        y_vals = marginal_effects[col]['y_values']
        coef = marginal_effects[col]['coefficient']
        se = marginal_effects[col]['std_error']
        
        ax.plot(x_vals, y_vals, 'b-', linewidth=2, label=f'β = {coef:.3f}')
        
        # Add confidence bands
        y_upper = np.array(y_vals) + 1.96 * se
        y_lower = np.array(y_vals) - 1.96 * se
        ax.fill_between(x_vals, y_lower, y_upper, alpha=0.3, color='blue', label='95% CI')
        
        # Add actual data points
        ax.scatter(X_clean[col], y_clean, alpha=0.4, s=20, color='red', label='Data')
        
        ax.set_xlabel(col, fontweight='bold')
        ax.set_ylabel(f'Predicted {y_col}', fontweight='bold')
        ax.set_title(f'Marginal Effect: {col}', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide empty subplots
    for i in range(n_vars, rows * cols):
        row = i // cols
        col_idx = i % cols
        axes[row, col_idx].set_visible(False)
    
    plt.tight_layout()
    
    # Convert to base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
    buffer.seek(0)
    plt.close()
    
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return img_str

def _create_interaction_plot(df: pd.DataFrame, y_col: str, x_cols: List[str]) -> str:
    """Create interaction plot if we have at least 2 variables."""
    if len(x_cols) < 2:
        return ""
    
    plt.figure(figsize=(12, 10))
    
    # Remove missing values
    mask = ~(df[y_col].isna() | df[x_cols[:2]].isna().any(axis=1))
    y_clean = df[y_col][mask]
    x1_clean = df[x_cols[0]][mask]
    x2_clean = df[x_cols[1]][mask]
    
    if len(y_clean) == 0:
        return ""
    
    # Create interaction model
    X_interact = pd.DataFrame({
        'x1': x1_clean,
        'x2': x2_clean,
        'x1_x2': x1_clean * x2_clean
    })
    X_with_const = sm.add_constant(X_interact)
    
    # Fit model
    model = sm.OLS(y_clean, X_with_const).fit()
    
    # Create interaction plot
    x1_range = np.linspace(x1_clean.min(), x1_clean.max(), 20)
    x2_values = [x2_clean.quantile(0.25), x2_clean.median(), x2_clean.quantile(0.75)]
    x2_labels = ['Low (25th)', 'Medium (50th)', 'High (75th)']
    
    colors = ['blue', 'red', 'green']
    
    for i, (x2_val, x2_label) in enumerate(zip(x2_values, x2_labels)):
        y_pred = []
        for x1_val in x1_range:
            X_pred = pd.DataFrame({
                'const': [1],
                'x1': [x1_val],
                'x2': [x2_val],
                'x1_x2': [x1_val * x2_val]
            })
            y_pred.append(model.predict(X_pred)[0])
        
        plt.plot(x1_range, y_pred, color=colors[i], linewidth=2, 
                label=f'{x2_label} {x_cols[1]} = {x2_val:.3f}')
    
    plt.xlabel(x_cols[0], fontsize=14, fontweight='bold')
    plt.ylabel(f'Predicted {y_col}', fontsize=14, fontweight='bold')
    plt.title(f'Interaction Effect: {x_cols[0]} × {x_cols[1]}', fontsize=16, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add interaction coefficient info
    interaction_coef = model.params['x1_x2']
    interaction_p = model.pvalues['x1_x2']
    
    info_text = f'Interaction Coefficient: {interaction_coef:.4f}\nP-value: {interaction_p:.4f}'
    plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes, fontsize=11,
             verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3",
             facecolor="lightyellow", alpha=0.9))
    
    plt.tight_layout()
    
    # Convert to base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
    buffer.seek(0)
    plt.close()
    
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return img_str

def _create_predicted_values_table(df: pd.DataFrame, y_col: str, x_cols: List[str]) -> pd.DataFrame:
    """Create table of predicted values at different levels of x variables."""
    # Remove missing values
    mask = ~(df[y_col].isna() | df[x_cols].isna().any(axis=1))
    y_clean = df[y_col][mask]
    X_clean = df[x_cols][mask]
    
    if len(y_clean) == 0:
        return pd.DataFrame()
    
    # Fit model
    X_with_const = sm.add_constant(X_clean)
    model = sm.OLS(y_clean, X_with_const).fit()
    
    # Create prediction scenarios
    scenarios = []
    
    # Scenario 1: All variables at their means
    X_means = X_clean.mean()
    X_pred_means = np.column_stack([np.ones(1), X_means.values.reshape(1, -1)])
    y_pred_means = model.predict(X_pred_means)[0]
    
    scenarios.append({
        'Scenario': 'All Variables at Mean',
        'Predicted Y': y_pred_means,
        'Description': 'Baseline prediction'
    })
    
    # Scenario 2: Each variable at mean ± 1 SD, others at mean
    for col in x_cols:
        # High scenario (+1 SD)
        X_high = X_means.copy()
        X_high[col] = X_clean[col].mean() + X_clean[col].std()
        X_pred_high = np.column_stack([np.ones(1), X_high.values.reshape(1, -1)])
        y_pred_high = model.predict(X_pred_high)[0]
        
        scenarios.append({
            'Scenario': f'{col} at Mean + 1 SD',
            'Predicted Y': y_pred_high,
            'Description': f'Effect of high {col}'
        })
        
        # Low scenario (-1 SD)
        X_low = X_means.copy()
        X_low[col] = X_clean[col].mean() - X_clean[col].std()
        X_pred_low = np.column_stack([np.ones(1), X_low.values.reshape(1, -1)])
        y_pred_low = model.predict(X_pred_low)[0]
        
        scenarios.append({
            'Scenario': f'{col} at Mean - 1 SD',
            'Predicted Y': y_pred_low,
            'Description': f'Effect of low {col}'
        })
    
    # Scenario 3: Interaction effects (if we have 2+ variables)
    if len(x_cols) >= 2:
        # High-High scenario
        X_high_high = X_means.copy()
        X_high_high[x_cols[0]] = X_clean[x_cols[0]].mean() + X_clean[x_cols[0]].std()
        X_high_high[x_cols[1]] = X_clean[x_cols[1]].mean() + X_clean[x_cols[1]].std()
        X_pred_high_high = np.column_stack([np.ones(1), X_high_high.values.reshape(1, -1)])
        y_pred_high_high = model.predict(X_pred_high_high)[0]
        
        scenarios.append({
            'Scenario': f'{x_cols[0]} & {x_cols[1]} High',
            'Predicted Y': y_pred_high_high,
            'Description': 'Interaction effect'
        })
    
    scenarios_df = pd.DataFrame(scenarios)
    scenarios_df['Predicted Y'] = scenarios_df['Predicted Y'].round(4)
    
    return scenarios_df

def apply(df: pd.DataFrame) -> List[dict]:
    """Create marginal effects plots and predicted values tables."""
    outputs = []
    
    # Find numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        return [{"type": "text", "title": "Margins", "data": "Need at least 2 numeric variables for marginal effects analysis."}]
    
    # Use first column as y, others as x variables
    y_col = numeric_cols[0]
    x_cols = numeric_cols[1:min(6, len(numeric_cols))]  # Limit to 5 x variables
    
    # 1. Marginal effects plot
    marginal_plot = _create_marginal_effects_plot(df, y_col, x_cols)
    if marginal_plot:
        outputs.append({
            "type": "image", 
            "title": "Marginal Effects Plot", 
            "data": marginal_plot
        })
    
    # 2. Interaction plot (if we have 2+ variables)
    if len(x_cols) >= 2:
        interaction_plot = _create_interaction_plot(df, y_col, x_cols)
        if interaction_plot:
            outputs.append({
                "type": "image", 
                "title": "Interaction Effects Plot", 
                "data": interaction_plot
            })
    
    # 3. Predicted values table
    pred_table = _create_predicted_values_table(df, y_col, x_cols)
    if not pred_table.empty:
        outputs.append({
            "type": "table", 
            "title": "Predicted Values at Different Scenarios", 
            "data": pred_table
        })
    
    # 4. Marginal effects summary table
    # Remove missing values
    mask = ~(df[y_col].isna() | df[x_cols].isna().any(axis=1))
    y_clean = df[y_col][mask]
    X_clean = df[x_cols][mask]
    
    if len(y_clean) > 0:
        X_with_const = sm.add_constant(X_clean)
        model = sm.OLS(y_clean, X_with_const).fit()
        
        marginal_summary = []
        for col in x_cols:
            marginal_summary.append({
                'Variable': col,
                'Coefficient': model.params[col],
                'Std. Error': model.bse[col],
                't-statistic': model.tvalues[col],
                'P-value': model.pvalues[col],
                'Marginal Effect': f'{model.params[col]:.4f} units per unit change in {col}'
            })
        
        marginal_df = pd.DataFrame(marginal_summary)
        numeric_cols_to_round = ['Coefficient', 'Std. Error', 't-statistic', 'P-value']
        for col in numeric_cols_to_round:
            marginal_df[col] = marginal_df[col].round(4)
        
        outputs.append({
            "type": "table", 
            "title": "Marginal Effects Summary", 
            "data": marginal_df
        })
    
    # 5. Interpretation guide
    interpretation = f"""
**Marginal Effects Analysis for {y_col}:**

**What these plots show:**
1. **Marginal Effects Plot**: How y changes when each x variable changes, holding others constant
2. **Interaction Plot**: How the effect of one variable depends on the level of another
3. **Predicted Values**: Expected y values under different scenarios

**Key Concepts:**
- **Marginal Effect**: Change in y per unit change in x, ceteris paribus
- **Interaction Effect**: How the effect of one variable varies with another
- **Predicted Values**: Expected outcomes under specific conditions

**Interpretation:**
- **Slope**: Steeper slope = stronger marginal effect
- **Curvature**: Non-linear relationships show curved lines
- **Confidence Bands**: Wider bands = more uncertainty
- **Interaction**: Lines not parallel = interaction present

**What to look for:**
1. **Linear vs Non-linear**: Straight lines vs curves
2. **Effect Size**: Steepness of the relationship
3. **Interactions**: Lines crossing or diverging
4. **Uncertainty**: Width of confidence bands

**Practical Use:**
- **Policy Analysis**: How changes in x affect y
- **Risk Assessment**: Predict outcomes under different scenarios
- **Optimization**: Find optimal levels of x variables
- **Sensitivity Analysis**: Which variables matter most
    """
    
    outputs.append({
        "type": "text", 
        "title": "Marginal Effects Interpretation Guide", 
        "data": interpretation
    })
    
    return outputs


def apply_with_config(df: pd.DataFrame, config: dict) -> List[dict]:
    """Run marginal effects using selected y and x columns if provided."""
    try:
        y_col = config.get('y_col')
        x_cols = config.get('x_cols', [])
        # If config is incomplete, fallback to default behavior
        if not y_col or not x_cols:
            return apply(df)
        outputs: List[dict] = []
        # 1. Marginal effects plot
        marginal_plot = _create_marginal_effects_plot(df, y_col, x_cols)
        if marginal_plot:
            outputs.append({"type": "image", "title": "Marginal Effects Plot", "data": marginal_plot})
        # 2. Interaction plot
        if len(x_cols) >= 2:
            interaction_plot = _create_interaction_plot(df, y_col, x_cols)
            if interaction_plot:
                outputs.append({"type": "image", "title": "Interaction Effects Plot", "data": interaction_plot})
        # 3. Predicted values table
        pred_table = _create_predicted_values_table(df, y_col, x_cols)
        if not pred_table.empty:
            outputs.append({"type": "table", "title": "Predicted Values at Different Scenarios", "data": pred_table})
        # 4. Marginal effects summary
        mask = ~(df[y_col].isna() | df[x_cols].isna().any(axis=1))
        y_clean = df[y_col][mask]
        X_clean = df[x_cols][mask]
        if len(y_clean) > 0:
            X_with_const = sm.add_constant(X_clean)
            model = sm.OLS(y_clean, X_with_const).fit()
            marginal_summary = []
            for col in x_cols:
                marginal_summary.append({
                    'Variable': col,
                    'Coefficient': model.params[col],
                    'Std. Error': model.bse[col],
                    't-statistic': model.tvalues[col],
                    'P-value': model.pvalues[col],
                    'Marginal Effect': f'{model.params[col]:.4f} units per unit change in {col}'
                })
            marginal_df = pd.DataFrame(marginal_summary)
            for c in ['Coefficient', 'Std. Error', 't-statistic', 'P-value']:
                marginal_df[c] = marginal_df[c].round(4)
            outputs.append({"type": "table", "title": "Marginal Effects Summary", "data": marginal_df})
        # Interpretation section reused
        interpretation = f"""
**Marginal Effects Analysis for {y_col}:**
"""
        outputs.append({"type": "text", "title": "Marginal Effects Interpretation Guide", "data": interpretation})
        return outputs
    except Exception as exc:
        return [{"type": "text", "title": "Margins Error", "data": f"Error computing marginal effects: {str(exc)}"}]
