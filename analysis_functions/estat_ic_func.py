from __future__ import annotations

from typing import List
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

display_name = "Estat IC"

def _fit_ols_model(X: pd.DataFrame, y: pd.Series) -> dict:
    """Fit OLS model and return model statistics."""
    try:
        # Add constant
        X_with_const = sm.add_constant(X)
        model = sm.OLS(y, X_with_const).fit()
        
        return {
            'aic': model.aic,
            'bic': model.bic,
            'r_squared': model.rsquared,
            'adj_r_squared': model.rsquared_adj,
            'f_stat': model.fvalue,
            'f_pvalue': model.f_pvalue,
            'nobs': model.nobs,
            'df_resid': model.df_resid,
            'ssr': model.ssr,
            'mse': model.mse_resid
        }
    except:
        return None

def _fit_polynomial_models(X: pd.DataFrame, y: pd.Series, max_degree: int = 3) -> List[dict]:
    """Fit polynomial models of different degrees."""
    models = []
    
    for degree in range(1, max_degree + 1):
        try:
            if degree == 1:
                # Linear model
                model_stats = _fit_ols_model(X, y)
                if model_stats:
                    model_stats['degree'] = degree
                    model_stats['model_type'] = 'Linear'
                    models.append(model_stats)
            else:
                # Polynomial model
                X_poly = X.copy()
                for col in X.columns:
                    X_poly[f'{col}_squared'] = X[col] ** 2
                    if degree >= 3:
                        X_poly[f'{col}_cubed'] = X[col] ** 3
                
                model_stats = _fit_ols_model(X_poly, y)
                if model_stats:
                    model_stats['degree'] = degree
                    model_stats['model_type'] = f'Polynomial (degree {degree})'
                    models.append(model_stats)
        except:
            continue
    
    return models

def _fit_interaction_models(X: pd.DataFrame, y: pd.Series) -> List[dict]:
    """Fit models with interaction terms."""
    models = []
    
    if len(X.columns) >= 2:
        try:
            # Add interaction terms
            X_interact = X.copy()
            for i, col1 in enumerate(X.columns):
                for j, col2 in enumerate(X.columns[i+1:], i+1):
                    X_interact[f'{col1}_x_{col2}'] = X[col1] * X[col2]
            
            model_stats = _fit_ols_model(X_interact, y)
            if model_stats:
                model_stats['degree'] = 1
                model_stats['model_type'] = 'With Interactions'
                models.append(model_stats)
        except:
            pass
    
    return models

def apply(df: pd.DataFrame) -> List[dict]:
    """Calculate and compare AIC, BIC, and other information criteria for different model specifications."""
    outputs = []
    
    # Find numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        return [{"type": "text", "title": "Estat IC", "data": "Need at least 2 numeric variables for model comparison."}]
    
    # Use first column as dependent variable, others as independent
    y_col = numeric_cols[0]
    X_cols = numeric_cols[1:min(6, len(numeric_cols))]  # Limit to 5 independent variables
    
    # Prepare data
    y = df[y_col].dropna()
    X = df[X_cols].dropna()
    
    # Align indices
    common_idx = y.index.intersection(X.index)
    if len(common_idx) < 10:
        return [{"type": "text", "title": "Estat IC", "data": "Insufficient overlapping data for model comparison."}]
    
    y = y.loc[common_idx]
    X = X.loc[common_idx]
    
    # Fit different model specifications
    all_models = []
    
    # 1. Base linear model
    base_model = _fit_ols_model(X, y)
    if base_model:
        base_model['degree'] = 1
        base_model['model_type'] = 'Base Linear'
        all_models.append(base_model)
    
    # 2. Polynomial models
    poly_models = _fit_polynomial_models(X, y, max_degree=3)
    all_models.extend(poly_models)
    
    # 3. Interaction models
    interaction_models = _fit_interaction_models(X, y)
    all_models.extend(interaction_models)
    
    if not all_models:
        return [{"type": "text", "title": "Estat IC", "data": "Could not fit any models for comparison."}]
    
    # Create comparison table
    comparison_data = []
    for model in all_models:
        comparison_data.append({
            'Model': model['model_type'],
            'AIC': model['aic'],
            'BIC': model['bic'],
            'R²': model['r_squared'],
            'Adj. R²': model['adj_r_squared'],
            'F-stat': model['f_stat'],
            'F P-value': model['f_pvalue'],
            'N': model['nobs'],
            'SSR': model['ssr'],
            'MSE': model['mse']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Round numeric columns
    numeric_cols_to_round = ['AIC', 'BIC', 'R²', 'Adj. R²', 'F-stat', 'F P-value', 'SSR', 'MSE']
    for col in numeric_cols_to_round:
        if col in comparison_df.columns:
            comparison_df[col] = comparison_df[col].round(4)
    
    outputs.append({
        "type": "table", 
        "title": "Model Comparison - Information Criteria", 
        "data": comparison_df
    })
    
    # Find best models by different criteria
    best_aic_idx = comparison_df['AIC'].idxmin()
    best_bic_idx = comparison_df['BIC'].idxmin()
    best_r2_idx = comparison_df['R²'].idxmin()
    
    best_models = []
    best_models.append(f"**Best AIC**: {comparison_df.loc[best_aic_idx, 'Model']} (AIC = {comparison_df.loc[best_aic_idx, 'AIC']:.4f})")
    best_models.append(f"**Best BIC**: {comparison_df.loc[best_bic_idx, 'Model']} (BIC = {comparison_df.loc[best_bic_idx, 'BIC']:.4f})")
    best_models.append(f"**Best R²**: {comparison_df.loc[best_r2_idx, 'Model']} (R² = {comparison_df.loc[best_r2_idx, 'R²']:.4f})")
    
    # AIC and BIC differences
    aic_diffs = []
    bic_diffs = []
    
    base_aic = comparison_df.loc[0, 'AIC'] if len(comparison_df) > 0 else 0
    base_bic = comparison_df.loc[0, 'BIC'] if len(comparison_df) > 0 else 0
    
    for idx, row in comparison_df.iterrows():
        if idx > 0:  # Skip base model
            aic_diff = row['AIC'] - base_aic
            bic_diff = row['BIC'] - base_bic
            
            aic_diffs.append({
                'Model': row['Model'],
                'ΔAIC': aic_diff,
                'Interpretation': 'Better' if aic_diff < 0 else 'Worse'
            })
            
            bic_diffs.append({
                'Model': row['Model'],
                'ΔBIC': bic_diff,
                'Interpretation': 'Better' if bic_diff < 0 else 'Worse'
            })
    
    if aic_diffs:
        aic_diff_df = pd.DataFrame(aic_diffs)
        aic_diff_df['ΔAIC'] = aic_diff_df['ΔAIC'].round(4)
        outputs.append({
            "type": "table", 
            "title": "AIC Differences from Base Model", 
            "data": aic_diff_df
        })
    
    if bic_diffs:
        bic_diff_df = pd.DataFrame(bic_diffs)
        bic_diff_df['ΔBIC'] = bic_diff_df['ΔBIC'].round(4)
        outputs.append({
            "type": "table", 
            "title": "BIC Differences from Base Model", 
            "data": bic_diff_df
        })
    
    # Model selection guidance
    guidance = """
**Model Selection Guidelines:**

**AIC (Akaike Information Criterion):**
- Lower is better
- Penalizes complexity less than BIC
- Good for prediction-focused models

**BIC (Bayesian Information Criterion):**
- Lower is better  
- Penalizes complexity more than AIC
- Good for explanatory models
- Tends to select simpler models

**R² vs Adjusted R²:**
- R² always increases with more variables
- Adjusted R² penalizes complexity
- Use Adjusted R² for model comparison

**General Rule:** Choose model with lowest AIC/BIC that doesn't overfit
    """
    
    outputs.append({
        "type": "text", 
        "title": "Model Selection Guidance", 
        "data": guidance
    })
    
    # Best models summary
    best_models_text = "\n".join(best_models)
    outputs.append({
        "type": "text", 
        "title": "Best Models by Criterion", 
        "data": best_models_text
    })
    
    return outputs
