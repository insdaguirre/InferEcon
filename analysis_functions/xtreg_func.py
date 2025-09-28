from __future__ import annotations

from typing import List, Optional
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

display_name = "Xtreg"

def _fit_pooled_ols(y: pd.Series, X: pd.DataFrame) -> dict:
    """Fit pooled OLS model."""
    X_with_const = sm.add_constant(X)
    model = sm.OLS(y, X_with_const).fit()
    
    return {
        'model': model,
        'coefficients': model.params,
        'std_errors': model.bse,
        't_values': model.tvalues,
        'p_values': model.pvalues,
        'r_squared': model.rsquared,
        'adj_r_squared': model.rsquared_adj,
        'f_stat': model.fvalue,
        'f_pval': model.f_pvalue,
        'nobs': model.nobs,
        'residuals': model.resid
    }

def _fit_fixed_effects(y: pd.Series, X: pd.DataFrame, entity_col: str, df_panel: pd.DataFrame) -> dict:
    """Fit fixed effects model using entity demeaning."""
    # Group by entity and demean
    y_demeaned = y.groupby(df_panel[entity_col]).transform(lambda x: x - x.mean())
    X_demeaned = X.groupby(df_panel[entity_col]).transform(lambda x: x - x.mean())
    
    # Remove constant term (absorbed by FE)
    if 'const' in X_demeaned.columns:
        X_demeaned = X_demeaned.drop('const', axis=1)
    
    # Fit OLS on demeaned data
    model = sm.OLS(y_demeaned, X_demeaned).fit()
    
    # Calculate within R-squared
    y_within = y.groupby(df_panel[entity_col]).transform(lambda x: x - x.mean())
    ss_res = np.sum(model.resid ** 2)
    ss_tot = np.sum(y_within ** 2)
    within_r2 = 1 - (ss_res / ss_tot)
    
    return {
        'model': model,
        'coefficients': model.params,
        'std_errors': model.bse,
        't_values': model.tvalues,
        'p_values': model.pvalues,
        'r_squared': model.rsquared,
        'within_r_squared': within_r2,
        'nobs': model.nobs,
        'residuals': model.resid,
        'model_type': 'Fixed Effects'
    }

def _fit_random_effects(y: pd.Series, X: pd.DataFrame, entity_col: str, df_panel: pd.DataFrame) -> dict:
    """Fit random effects model using GLS."""
    # Calculate variance components
    n_entities = df_panel[entity_col].nunique()
    n_per_entity = len(y) / n_entities
    
    # Group by entity
    y_grouped = y.groupby(df_panel[entity_col])
    X_grouped = X.groupby(df_panel[entity_col])
    
    # Calculate within and between variation
    y_within = y_grouped.transform(lambda x: x - x.mean())
    X_within = X_grouped.transform(lambda x: x - x.mean())
    
    y_between = y_grouped.mean()
    X_between = X_grouped.mean()
    
    # Fit OLS on between variation
    if 'const' in X_between.columns:
        X_between = X_between.drop('const', axis=1)
    
    model_between = sm.OLS(y_between, X_between).fit()
    
    # Calculate theta (weight for RE)
    sigma_u = model_between.mse_resid  # Between variance
    sigma_e = np.var(y_within)  # Within variance
    
    theta = 1 - np.sqrt(sigma_e / (sigma_e + n_per_entity * sigma_u))
    
    # Transform data for RE
    y_re = y - theta * y_grouped.transform('mean')
    X_re = X - theta * X_grouped.transform('mean')
    
    # Fit RE model
    if 'const' in X_re.columns:
        X_re = X_re.drop('const', axis=1)
    
    model = sm.OLS(y_re, X_re).fit()
    
    return {
        'model': model,
        'coefficients': model.params,
        'std_errors': model.bse,
        't_values': model.tvalues,
        'p_values': model.pvalues,
        'r_squared': model.rsquared,
        'theta': theta,
        'sigma_u': sigma_u,
        'sigma_e': sigma_e,
        'nobs': model.nobs,
        'residuals': model.resid,
        'model_type': 'Random Effects'
    }

def _fit_between_effects(y: pd.Series, X: pd.DataFrame, entity_col: str, df_panel: pd.DataFrame) -> dict:
    """Fit between effects model using entity means."""
    # Calculate entity means
    y_between = y.groupby(df_panel[entity_col]).mean()
    X_between = X.groupby(df_panel[entity_col]).mean()
    
    # Add constant
    X_between = sm.add_constant(X_between)
    
    # Fit OLS on between variation
    model = sm.OLS(y_between, X_between).fit()
    
    return {
        'model': model,
        'coefficients': model.params,
        'std_errors': model.bse,
        't_values': model.tvalues,
        'p_values': model.pvalues,
        'r_squared': model.rsquared,
        'adj_r_squared': model.rsquared_adj,
        'nobs': model.nobs,
        'residuals': model.resid,
        'model_type': 'Between Effects'
    }

def _fit_first_differences(y: pd.Series, X: pd.DataFrame, entity_col: str, time_col: str, df_panel: pd.DataFrame) -> dict:
    """Fit first differences model."""
    # Sort by entity and time
    df_sorted = df_panel.sort_values([entity_col, time_col])
    
    # Calculate first differences
    y_diff = df_sorted[y.name].groupby(df_sorted[entity_col]).diff()
    X_diff = df_sorted[X.columns].groupby(df_sorted[entity_col]).diff()
    
    # Remove NaN values
    mask = ~(y_diff.isna() | X_diff.isna().any(axis=1))
    y_diff_clean = y_diff[mask]
    X_diff_clean = X_diff[mask]
    
    if len(y_diff_clean) == 0:
        return None
    
    # Add constant
    X_diff_clean = sm.add_constant(X_diff_clean)
    
    # Fit OLS on differenced data
    model = sm.OLS(y_diff_clean, X_diff_clean).fit()
    
    return {
        'model': model,
        'coefficients': model.params,
        'std_errors': model.bse,
        't_values': model.tvalues,
        'p_values': model.pvalues,
        'r_squared': model.rsquared,
        'adj_r_squared': model.rsquared_adj,
        'nobs': model.nobs,
        'residuals': model.resid,
        'model_type': 'First Differences'
    }

def _calculate_panel_statistics(df_panel: pd.DataFrame, entity_col: str, time_col: str) -> dict:
    """Calculate panel data statistics."""
    n_entities = df_panel[entity_col].nunique()
    n_time_periods = df_panel[time_col].nunique()
    total_obs = len(df_panel)
    
    # Check for balanced panel
    obs_per_entity = df_panel.groupby(entity_col).size()
    is_balanced = obs_per_entity.nunique() == 1
    
    # Calculate time variation
    time_variation = df_panel.groupby(entity_col).size().describe()
    
    return {
        'n_entities': n_entities,
        'n_time_periods': n_time_periods,
        'total_observations': total_obs,
        'is_balanced': is_balanced,
        'observations_per_entity': obs_per_entity.to_dict(),
        'time_variation_summary': time_variation.to_dict()
    }

def apply(df: pd.DataFrame) -> List[dict]:
    """Perform panel data regressions with various estimation methods."""
    outputs = []
    
    # Find numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 3:
        return [{"type": "text", "title": "Xtreg", "data": "Need at least 3 numeric variables for panel data analysis."}]
    
    # Create panel structure
    n_entities = min(10, len(df) // 5)  # Create up to 10 entities
    if n_entities < 2:
        return [{"type": "text", "title": "Xtreg", "data": "Need more observations to create panel structure."}]
    
    # Create entity and time identifiers
    entity_id = np.repeat(range(n_entities), len(df) // n_entities)
    if len(entity_id) < len(df):
        entity_id = np.append(entity_id, [entity_id[-1]] * (len(df) - len(entity_id)))
    
    time_id = np.tile(range(len(df) // n_entities), n_entities)
    if len(time_id) < len(df):
        time_id = np.append(time_id, [time_id[-1]] * (len(df) - len(time_id)))
    
    # Add to dataframe
    df_panel = df.copy()
    df_panel['entity_id'] = entity_id
    df_panel['time_id'] = time_id
    
    # Use first column as dependent variable, others as independent
    y_col = numeric_cols[0]
    x_cols = numeric_cols[1:min(6, len(numeric_cols))]  # Limit to 5 x variables
    
    # Prepare data
    mask = ~(df_panel[y_col].isna() | df_panel[x_cols].isna().any(axis=1))
    y = df_panel[y_col][mask]
    X = df_panel[x_cols][mask]
    entity_col = 'entity_id'
    time_col = 'time_id'
    
    if len(y) < 20:
        return [{"type": "text", "title": "Xtreg", "data": "Insufficient overlapping data for panel analysis."}]
    
    # Calculate panel statistics
    panel_stats = _calculate_panel_statistics(df_panel, entity_col, time_col)
    
    # 1. Panel Data Summary
    panel_summary = pd.DataFrame({
        'Statistic': ['Number of Entities', 'Time Periods', 'Total Observations', 'Balanced Panel'],
        'Value': [
            panel_stats['n_entities'],
            panel_stats['n_time_periods'],
            panel_stats['total_observations'],
            'Yes' if panel_stats['is_balanced'] else 'No'
        ]
    })
    
    outputs.append({
        "type": "table", 
        "title": "Panel Data Structure", 
        "data": panel_summary
    })
    
    # 2. Pooled OLS
    try:
        pooled_results = _fit_pooled_ols(y, X)
        
        pooled_coef_data = []
        for var, coef, se, t_val, p_val in zip(
            pooled_results['coefficients'].index,
            pooled_results['coefficients'].values,
            pooled_results['std_errors'].values,
            pooled_results['t_values'].values,
            pooled_results['p_values'].values
        ):
            pooled_coef_data.append({
                'Variable': var,
                'Coefficient': coef,
                'Std. Error': se,
                't-statistic': t_val,
                'P-value': p_val
            })
        
        pooled_coef_table = pd.DataFrame(pooled_coef_data)
        numeric_cols_to_round = ['Coefficient', 'Std. Error', 't-statistic', 'P-value']
        for col in numeric_cols_to_round:
            pooled_coef_table[col] = pooled_coef_table[col].round(4)
        
        outputs.append({
            "type": "table", 
            "title": "Pooled OLS Results", 
            "data": pooled_coef_table
        })
        
        # Pooled model summary
        pooled_summary = pd.DataFrame({
            'Statistic': ['R-squared', 'Adjusted R-squared', 'F-statistic', 'F P-value', 'Observations'],
            'Value': [
                pooled_results['r_squared'],
                pooled_results['adj_r_squared'],
                pooled_results['f_stat'],
                pooled_results['f_pval'],
                pooled_results['nobs']
            ]
        })
        pooled_summary['Value'] = pooled_summary['Value'].round(4)
        
        outputs.append({
            "type": "table", 
            "title": "Pooled OLS Model Summary", 
            "data": pooled_summary
        })
        
    except Exception as e:
        outputs.append({
            "type": "text", 
            "title": "Pooled OLS Error", 
            "data": f"Error fitting pooled OLS: {str(e)}"
        })
    
    # 3. Fixed Effects
    try:
        fe_results = _fit_fixed_effects(y, X, entity_col, df_panel)
        
        fe_coef_data = []
        for var, coef, se, t_val, p_val in zip(
            fe_results['coefficients'].index,
            fe_results['coefficients'].values,
            fe_results['std_errors'].values,
            fe_results['t_values'].values,
            fe_results['p_values'].values
        ):
            fe_coef_data.append({
                'Variable': var,
                'Coefficient': coef,
                'Std. Error': se,
                't-statistic': t_val,
                'P-value': p_val
            })
        
        fe_coef_table = pd.DataFrame(fe_coef_data)
        numeric_cols_to_round = ['Coefficient', 'Std. Error', 't-statistic', 'P-value']
        for col in numeric_cols_to_round:
            fe_coef_table[col] = fe_coef_table[col].round(4)
        
        outputs.append({
            "type": "table", 
            "title": "Fixed Effects Results", 
            "data": fe_coef_table
        })
        
        # FE model summary
        fe_summary = pd.DataFrame({
            'Statistic': ['R-squared', 'Within R-squared', 'Observations'],
            'Value': [
                fe_results['r_squared'],
                fe_results['within_r_squared'],
                fe_results['nobs']
            ]
        })
        fe_summary['Value'] = fe_summary['Value'].round(4)
        
        outputs.append({
            "type": "table", 
            "title": "Fixed Effects Model Summary", 
            "data": fe_summary
        })
        
    except Exception as e:
        outputs.append({
            "type": "text", 
            "title": "Fixed Effects Error", 
            "data": f"Error fitting fixed effects: {str(e)}"
        })
    
    # 4. Random Effects
    try:
        re_results = _fit_random_effects(y, X, entity_col, df_panel)
        
        re_coef_data = []
        for var, coef, se, t_val, p_val in zip(
            re_results['coefficients'].index,
            re_results['coefficients'].values,
            re_results['std_errors'].values,
            re_results['t_values'].values,
            re_results['p_values'].values
        ):
            re_coef_data.append({
                'Variable': var,
                'Coefficient': coef,
                'Std. Error': se,
                't-statistic': t_val,
                'P-value': p_val
            })
        
        re_coef_table = pd.DataFrame(re_coef_data)
        numeric_cols_to_round = ['Coefficient', 'Std. Error', 't-statistic', 'P-value']
        for col in numeric_cols_to_round:
            re_coef_table[col] = re_coef_table[col].round(4)
        
        outputs.append({
            "type": "table", 
            "title": "Random Effects Results", 
            "data": re_coef_table
        })
        
        # RE model summary
        re_summary = pd.DataFrame({
            'Statistic': ['R-squared', 'Theta', 'Sigma_u', 'Sigma_e', 'Observations'],
            'Value': [
                re_results['r_squared'],
                re_results['theta'],
                re_results['sigma_u'],
                re_results['sigma_e'],
                re_results['nobs']
            ]
        })
        re_summary['Value'] = re_summary['Value'].round(4)
        
        outputs.append({
            "type": "table", 
            "title": "Random Effects Model Summary", 
            "data": re_summary
        })
        
    except Exception as e:
        outputs.append({
            "type": "text", 
            "title": "Random Effects Error", 
            "data": f"Error fitting random effects: {str(e)}"
        })
    
    # 5. Between Effects
    try:
        between_results = _fit_between_effects(y, X, entity_col, df_panel)
        
        between_coef_data = []
        for var, coef, se, t_val, p_val in zip(
            between_results['coefficients'].index,
            between_results['coefficients'].values,
            between_results['std_errors'].values,
            between_results['t_values'].values,
            between_results['p_values'].values
        ):
            between_coef_data.append({
                'Variable': var,
                'Coefficient': coef,
                'Std. Error': se,
                't-statistic': t_val,
                'P-value': p_val
            })
        
        between_coef_table = pd.DataFrame(between_coef_data)
        numeric_cols_to_round = ['Coefficient', 'Std. Error', 't-statistic', 'P-value']
        for col in numeric_cols_to_round:
            between_coef_table[col] = between_coef_table[col].round(4)
        
        outputs.append({
            "type": "table", 
            "title": "Between Effects Results", 
            "data": between_coef_table
        })
        
    except Exception as e:
        outputs.append({
            "type": "text", 
            "title": "Between Effects Error", 
            "data": f"Error fitting between effects: {str(e)}"
        })
    
    # 6. First Differences
    try:
        fd_results = _fit_first_differences(y, X, entity_col, time_col, df_panel)
        
        if fd_results:
            fd_coef_data = []
            for var, coef, se, t_val, p_val in zip(
                fd_results['coefficients'].index,
                fd_results['coefficients'].values,
                fd_results['std_errors'].values,
                fd_results['t_values'].values,
                fd_results['p_values'].values
            ):
                fd_coef_data.append({
                    'Variable': var,
                    'Coefficient': coef,
                    'Std. Error': se,
                    't-statistic': t_val,
                    'P-value': p_val
                })
            
            fd_coef_table = pd.DataFrame(fd_coef_data)
            numeric_cols_to_round = ['Coefficient', 'Std. Error', 't-statistic', 'P-value']
            for col in numeric_cols_to_round:
                fd_coef_table[col] = fd_coef_table[col].round(4)
            
            outputs.append({
                "type": "table", 
                "title": "First Differences Results", 
                "data": fd_coef_table
            })
        
    except Exception as e:
        outputs.append({
            "type": "text", 
            "title": "First Differences Error", 
            "data": f"Error fitting first differences: {str(e)}"
        })
    
    # 7. Model Comparison
    try:
        comparison_data = []
        
        if 'pooled_results' in locals():
            comparison_data.append({
                'Model': 'Pooled OLS',
                'R²': pooled_results['r_squared'],
                'Observations': pooled_results['nobs'],
                'Notes': 'No panel structure'
            })
        
        if 'fe_results' in locals():
            comparison_data.append({
                'Model': 'Fixed Effects',
                'R²': fe_results['within_r_squared'],
                'Observations': fe_results['nobs'],
                'Notes': 'Controls time-invariant unobservables'
            })
        
        if 're_results' in locals():
            comparison_data.append({
                'Model': 'Random Effects',
                'R²': re_results['r_squared'],
                'Observations': re_results['nobs'],
                'Notes': 'Efficient if uncorrelated unobservables'
            })
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            comparison_df['R²'] = comparison_df['R²'].round(4)
            
            outputs.append({
                "type": "table", 
                "title": "Model Comparison", 
                "data": comparison_df
            })
        
    except Exception as e:
        outputs.append({
            "type": "text", 
            "title": "Model Comparison Error", 
            "data": f"Error creating model comparison: {str(e)}"
        })
    
    # 8. Interpretation Guide
    interpretation = f"""
**Panel Data Regression Analysis Results:**

**Data Structure:**
- **Entities**: {panel_stats['n_entities']} (e.g., firms, countries, individuals)
- **Time Periods**: {panel_stats['n_time_periods']} (e.g., years, quarters)
- **Total Observations**: {panel_stats['total_observations']}
- **Balanced Panel**: {'Yes' if panel_stats['is_balanced'] else 'No'}

**Estimation Methods:**

**1. Pooled OLS:**
- Ignores panel structure
- Assumes no entity-specific effects
- Use when entities are homogeneous

**2. Fixed Effects (FE):**
- Controls for time-invariant unobservables
- More robust but less efficient
- Use when you suspect correlation between unobservables and regressors

**3. Random Effects (RE):**
- Assumes uncorrelated unobservables
- More efficient but less robust
- Use when unobservables are uncorrelated with regressors

**4. Between Effects:**
- Uses cross-sectional variation only
- Ignores within-entity variation
- Use for cross-sectional analysis

**5. First Differences:**
- Controls for time-invariant effects
- Alternative to FE
- Use when you prefer differencing over demeaning

**Model Selection:**
- **Hausman Test**: Compare FE vs RE
- **Economic Theory**: Consider data generating process
- **Data Structure**: Check for balanced/unbalanced panel
- **Research Question**: What variation are you interested in?
        """
    
    outputs.append({
        "type": "text", 
        "title": "Panel Data Analysis Interpretation", 
        "data": interpretation
    })
    
    return outputs


def apply_with_config(df: pd.DataFrame, config: dict) -> List[dict]:
    """Perform panel regressions using user-provided columns and panel ids.

    Config keys:
    - y_col: dependent variable
    - x_cols: list of regressors
    - entity_col/time_col: panel identifiers if present
    - create_panel: bool; if true, auto-create ids using n_entities
    - n_entities: number of entities to synthesize when creating panel ids
    """
    outputs: List[dict] = []
    try:
        y_col = config.get('y_col')
        x_cols = config.get('x_cols', [])
        entity_col = config.get('entity_col')
        time_col = config.get('time_col')
        create_panel = config.get('create_panel', False)
        n_entities = config.get('n_entities')
        if not y_col or not x_cols:
            return apply(df)
        # Build df_panel
        df_panel = df.copy()
        if create_panel or not (entity_col and time_col and entity_col in df.columns and time_col in df.columns):
            # synthesize
            n = len(df)
            if not n_entities:
                n_entities = max(2, min(10, n // 5))
            ent = np.repeat(range(n_entities), max(1, n // n_entities))
            if len(ent) < n:
                ent = np.append(ent, [ent[-1]] * (n - len(ent)))
            tvals = np.tile(range(max(1, n // max(1, n_entities))), n_entities)
            if len(tvals) < n:
                tvals = np.append(tvals, [tvals[-1]] * (n - len(tvals)))
            df_panel['entity_id'] = ent[:n]
            df_panel['time_id'] = tvals[:n]
            entity_col = 'entity_id'
            time_col = 'time_id'
        else:
            # ensure columns exist
            df_panel['entity_id'] = df_panel[entity_col]
            df_panel['time_id'] = df_panel[time_col]
            entity_col = 'entity_id'
            time_col = 'time_id'
        # Prepare data
        mask = ~(df_panel[y_col].isna() | df_panel[x_cols].isna().any(axis=1))
        y = df_panel[y_col][mask]
        X = df_panel[x_cols][mask]
        if len(y) < 10:
            return [{"type": "text", "title": "Xtreg", "data": "Insufficient overlapping data for panel analysis."}]
        # Reuse same pipeline as apply()
        panel_stats = _calculate_panel_statistics(df_panel, entity_col, time_col)
        panel_summary = pd.DataFrame({
            'Statistic': ['Number of Entities', 'Time Periods', 'Total Observations', 'Balanced Panel'],
            'Value': [panel_stats['n_entities'], panel_stats['n_time_periods'], panel_stats['total_observations'], 'Yes' if panel_stats['is_balanced'] else 'No']
        })
        outputs.append({"type": "table", "title": "Panel Data Structure", "data": panel_summary})
        # Pooled OLS
        try:
            pooled_results = _fit_pooled_ols(y, X)
            pooled_coef_table = pd.DataFrame({
                'Variable': pooled_results['coefficients'].index,
                'Coefficient': pooled_results['coefficients'].values,
                'Std. Error': pooled_results['std_errors'].values,
                't-statistic': pooled_results['t_values'].values,
                'P-value': pooled_results['p_values'].values
            })
            for c in ['Coefficient', 'Std. Error', 't-statistic', 'P-value']:
                pooled_coef_table[c] = pooled_coef_table[c].round(4)
            outputs.append({"type": "table", "title": "Pooled OLS Results", "data": pooled_coef_table})
        except Exception as exc:
            outputs.append({"type": "text", "title": "Pooled OLS Error", "data": f"Error: {str(exc)}"})
        # Fixed Effects
        try:
            fe_results = _fit_fixed_effects(y, X, entity_col, df_panel)
            fe_table = pd.DataFrame({
                'Variable': fe_results['coefficients'].index,
                'Coefficient': fe_results['coefficients'].values,
                'Std. Error': fe_results['std_errors'].values,
                't-statistic': fe_results['t_values'].values,
                'P-value': fe_results['p_values'].values
            })
            for c in ['Coefficient', 'Std. Error', 't-statistic', 'P-value']:
                fe_table[c] = fe_table[c].round(4)
            outputs.append({"type": "table", "title": "Fixed Effects Results", "data": fe_table})
        except Exception as exc:
            outputs.append({"type": "text", "title": "Fixed Effects Error", "data": f"Error: {str(exc)}"})
        # Random Effects
        try:
            re_results = _fit_random_effects(y, X, entity_col, df_panel)
            re_table = pd.DataFrame({
                'Variable': re_results['coefficients'].index,
                'Coefficient': re_results['coefficients'].values,
                'Std. Error': re_results['std_errors'].values,
                't-statistic': re_results['t_values'].values,
                'P-value': re_results['p_values'].values
            })
            for c in ['Coefficient', 'Std. Error', 't-statistic', 'P-value']:
                re_table[c] = re_table[c].round(4)
            outputs.append({"type": "table", "title": "Random Effects Results", "data": re_table})
        except Exception as exc:
            outputs.append({"type": "text", "title": "Random Effects Error", "data": f"Error: {str(exc)}"})
        return outputs
    except Exception as exc:
        return [{"type": "text", "title": "Xtreg Error", "data": f"Error: {str(exc)}"}]
