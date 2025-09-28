from __future__ import annotations

from typing import List, Optional
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

display_name = "Hausman"

def _fit_fixed_effects_model(y: pd.Series, X: pd.DataFrame, entity_col: str, df_panel: pd.DataFrame) -> dict:
    """Fit fixed effects model using entity demeaning."""
    # Group by entity and demean
    y_demeaned = y.groupby(df_panel[entity_col]).transform(lambda x: x - x.mean())
    X_demeaned = X.groupby(df_panel[entity_col]).transform(lambda x: x - x.mean())
    
    # Remove constant term (absorbed by FE)
    if 'const' in X_demeaned.columns:
        X_demeaned = X_demeaned.drop('const', axis=1)
    
    # Fit OLS on demeaned data
    model_fe = sm.OLS(y_demeaned, X_demeaned).fit()
    
    return {
        'model': model_fe,
        'coefficients': model_fe.params,
        'std_errors': model_fe.bse,
        'cov_matrix': model_fe.cov_params(),
        'residuals': model_fe.resid,
        'r_squared': model_fe.rsquared,
        'nobs': model_fe.nobs
    }

def _fit_random_effects_model(y: pd.Series, X: pd.DataFrame, entity_col: str, df_panel: pd.DataFrame) -> dict:
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
    
    model_re = sm.OLS(y_re, X_re).fit()
    
    return {
        'model': model_re,
        'coefficients': model_re.params,
        'std_errors': model_re.bse,
        'cov_matrix': model_re.cov_params(),
        'theta': theta,
        'sigma_u': sigma_u,
        'sigma_e': sigma_e,
        'r_squared': model_re.rsquared,
        'nobs': model_re.nobs
    }

def _perform_hausman_test(fe_results: dict, re_results: dict) -> dict:
    """Perform Hausman test comparing FE vs RE estimates."""
    # Extract coefficients and covariance matrices
    beta_fe = fe_results['coefficients']
    beta_re = re_results['coefficients']
    
    # Get common variables
    common_vars = beta_fe.index.intersection(beta_re.index)
    if len(common_vars) == 0:
        return {'error': 'No common variables between FE and RE models'}
    
    beta_fe_common = beta_fe[common_vars]
    beta_re_common = beta_re[common_vars]
    
    # Get covariance matrices for common variables
    cov_fe = fe_results['cov_matrix'].loc[common_vars, common_vars]
    cov_re = re_results['cov_matrix'].loc[common_vars, common_vars]
    
    # Calculate difference in coefficients
    beta_diff = beta_fe_common - beta_re_common
    
    # Calculate variance of difference
    var_diff = cov_fe - cov_re
    
    # Check if variance matrix is positive definite
    try:
        # Calculate Hausman statistic
        hausman_stat = beta_diff.T @ np.linalg.inv(var_diff) @ beta_diff
        df = len(common_vars)
        hausman_pval = 1 - stats.chi2.cdf(hausman_stat, df)
        
        return {
            'hausman_statistic': hausman_stat,
            'p_value': hausman_pval,
            'degrees_of_freedom': df,
            'beta_diff': beta_diff,
            'var_diff': var_diff
        }
    except np.linalg.LinAlgError:
        # If matrix is not invertible, use alternative approach
        # Calculate simple t-test for each coefficient
        t_stats = []
        p_vals = []
        
        for var in common_vars:
            diff = beta_fe[var] - beta_re[var]
            se_diff = np.sqrt(cov_fe.loc[var, var] + cov_re.loc[var, var])
            t_stat = diff / se_diff
            p_val = 2 * (1 - stats.norm.cdf(abs(t_stat)))
            
            t_stats.append(t_stat)
            p_vals.append(p_val)
        
        return {
            'alternative_test': 'Individual t-tests',
            'variables': common_vars.tolist(),
            't_statistics': t_stats,
            'p_values': p_vals
        }

def apply(df: pd.DataFrame) -> List[dict]:
    """Perform Hausman test comparing fixed vs random effects models."""
    outputs = []
    
    # Find numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 3:
        return [{"type": "text", "title": "Hausman", "data": "Need at least 3 numeric variables for panel data analysis."}]
    
    # For demonstration, we'll create a simple panel structure
    # In practice, you'd want user input for entity and time identifiers
    
    # Create a simple entity identifier (assuming data is ordered)
    n_entities = min(10, len(df) // 5)  # Create up to 10 entities
    if n_entities < 2:
        return [{"type": "text", "title": "Hausman", "data": "Need more observations to create panel structure."}]
    
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
    
    if len(y) < 20:
        return [{"type": "text", "title": "Hausman", "data": "Insufficient overlapping data for panel analysis."}]
    
    # Perform panel data analysis
    try:
        # 1. Fit Fixed Effects model
        fe_results = _fit_fixed_effects_model(y, X, entity_col, df_panel)
        
        # 2. Fit Random Effects model
        re_results = _fit_random_effects_model(y, X, entity_col, df_panel)
        
        # 3. Perform Hausman test
        hausman_results = _perform_hausman_test(fe_results, re_results)
        
        # 4. Fixed Effects Results
        fe_coef_data = []
        for var, coef, se in zip(
            fe_results['coefficients'].index,
            fe_results['coefficients'].values,
            fe_results['std_errors'].values
        ):
            fe_coef_data.append({
                'Variable': var,
                'Coefficient': coef,
                'Std. Error': se
            })
        
        fe_coef_table = pd.DataFrame(fe_coef_data)
        fe_coef_table['Coefficient'] = fe_coef_table['Coefficient'].round(4)
        fe_coef_table['Std. Error'] = fe_coef_table['Std. Error'].round(4)
        
        outputs.append({
            "type": "table", 
            "title": "Fixed Effects Model Results", 
            "data": fe_coef_table
        })
        
        # 5. Random Effects Results
        re_coef_data = []
        for var, coef, se in zip(
            re_results['coefficients'].index,
            re_results['coefficients'].values,
            re_results['std_errors'].values
        ):
            re_coef_data.append({
                'Variable': var,
                'Coefficient': coef,
                'Std. Error': se
            })
        
        re_coef_table = pd.DataFrame(re_coef_data)
        re_coef_table['Coefficient'] = re_coef_table['Coefficient'].round(4)
        re_coef_table['Coefficient'] = re_coef_table['Coefficient'].round(4)
        
        outputs.append({
            "type": "table", 
            "title": "Random Effects Model Results", 
            "data": re_coef_table
        })
        
        # 6. Model Comparison
        comparison_data = [
            {'Model': 'Fixed Effects', 'R²': fe_results['r_squared'], 'Observations': fe_results['nobs']},
            {'Model': 'Random Effects', 'R²': re_results['r_squared'], 'Observations': re_results['nobs']}
        ]
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df['R²'] = comparison_df['R²'].round(4)
        
        outputs.append({
            "type": "table", 
            "title": "Model Comparison", 
            "data": comparison_df
        })
        
        # 7. Hausman Test Results
        if 'error' not in hausman_results:
            if 'hausman_statistic' in hausman_results:
                hausman_table = pd.DataFrame({
                    'Test': ['Hausman Statistic', 'P-value', 'Degrees of Freedom'],
                    'Value': [
                        hausman_results['hausman_statistic'],
                        hausman_results['p_value'],
                        hausman_results['degrees_of_freedom']
                    ]
                })
                hausman_table['Value'] = hausman_table['Value'].round(4)
                
                outputs.append({
                    "type": "table", 
                    "title": "Hausman Test Results", 
                    "data": hausman_table
                })
                
                # Coefficient differences
                diff_data = []
                for var, diff in hausman_results['beta_diff'].items():
                    diff_data.append({
                        'Variable': var,
                        'FE Coefficient': fe_results['coefficients'].get(var, np.nan),
                        'RE Coefficient': re_results['coefficients'].get(var, np.nan),
                        'Difference': diff
                    })
                
                diff_df = pd.DataFrame(diff_data)
                numeric_cols_to_round = ['FE Coefficient', 'RE Coefficient', 'Difference']
                for col in numeric_cols_to_round:
                    diff_df[col] = diff_df[col].round(4)
                
                outputs.append({
                    "type": "table", 
                    "title": "Coefficient Differences (FE - RE)", 
                    "data": diff_df
                })
            else:
                # Alternative test results
                alt_data = []
                for var, t_stat, p_val in zip(
                    hausman_results['variables'],
                    hausman_results['t_statistics'],
                    hausman_results['p_values']
                ):
                    alt_data.append({
                        'Variable': var,
                        't-statistic': t_stat,
                        'P-value': p_val
                    })
                
                alt_df = pd.DataFrame(alt_data)
                alt_df['t-statistic'] = alt_df['t-statistic'].round(4)
                alt_df['P-value'] = alt_df['P-value'].round(4)
                
                outputs.append({
                    "type": "table", 
                    "title": "Alternative Test Results (Individual t-tests)", 
                    "data": alt_df
                })
        
        # 8. Random Effects Parameters
        re_params = pd.DataFrame({
            'Parameter': ['Theta', 'Sigma_u (Between)', 'Sigma_e (Within)'],
            'Value': [
                re_results['theta'],
                re_results['sigma_u'],
                re_results['sigma_e']
            ]
        })
        re_params['Value'] = re_params['Value'].round(4)
        
        outputs.append({
            "type": "table", 
            "title": "Random Effects Model Parameters", 
            "data": re_params
        })
        
        # 9. Interpretation
        if 'hausman_statistic' in hausman_results:
            hausman_pval = hausman_results['p_value']
            interpretation = f"""
**Hausman Test Results:**

**Test Statistic**: {hausman_results['hausman_statistic']:.4f}
**P-value**: {hausman_pval:.4f}
**Degrees of Freedom**: {hausman_results['degrees_of_freedom']}

**Interpretation:**
- **H₀**: Random effects model is appropriate (coefficients are consistent)
- **H₁**: Fixed effects model is needed (coefficients are inconsistent)

**Decision Rule:**
- **P < 0.05**: Reject H₀ → Use Fixed Effects model
- **P ≥ 0.05**: Fail to reject H₀ → Random Effects model is acceptable

**Your Result**: {'⚠️ Use Fixed Effects (RE is inconsistent)' if hausman_pval < 0.05 else '✅ Random Effects is acceptable (FE not needed)'}

**Model Choice:**
- **Fixed Effects**: More robust, controls for time-invariant unobservables
- **Random Effects**: More efficient, assumes uncorrelated unobservables
- **Hausman Test**: Helps choose between efficiency and robustness
            """
        else:
            interpretation = """
**Alternative Test Results:**

Since the Hausman test covariance matrix was not positive definite, 
we performed individual t-tests comparing FE vs RE coefficients.

**Interpretation:**
- **Significant differences**: Suggest using Fixed Effects
- **No significant differences**: Random Effects may be appropriate

**Model Recommendations:**
- **Fixed Effects**: When you suspect correlation between unobservables and regressors
- **Random Effects**: When unobservables are uncorrelated with regressors
- **Always check**: Economic theory and data structure
            """
        
        outputs.append({
            "type": "text", 
            "title": "Hausman Test Interpretation", 
            "data": interpretation
        })
        
    except Exception as e:
        outputs.append({
            "type": "text", 
            "title": "Hausman Test Error", 
            "data": f"Error performing Hausman test: {str(e)}"
        })
    
    return outputs


def apply_with_config(df: pd.DataFrame, config: dict) -> List[dict]:
    """Run Hausman test using selected y/x and either provided or synthesized panel ids."""
    try:
        y_col = config.get('y_col')
        x_cols = config.get('x_cols', [])
        entity_col = config.get('entity_col')
        time_col = config.get('time_col')
        create_panel = config.get('create_panel', False)
        n_entities = config.get('n_entities')
        if not y_col or not x_cols:
            return apply(df)
        df_panel = df.copy()
        if create_panel or not (entity_col and time_col and entity_col in df.columns and time_col in df.columns):
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
            df_panel['entity_id'] = df_panel[entity_col]
            df_panel['time_id'] = df_panel[time_col]
            entity_col = 'entity_id'
            time_col = 'time_id'
        mask = ~(df_panel[y_col].isna() | df_panel[x_cols].isna().any(axis=1))
        y = df_panel[y_col][mask]
        X = df_panel[x_cols][mask]
        if len(y) < 20:
            return [{"type": "text", "title": "Hausman", "data": "Insufficient overlapping data for panel analysis."}]
        fe_results = _fit_fixed_effects_model(y, X, entity_col, df_panel)
        re_results = _fit_random_effects_model(y, X, entity_col, df_panel)
        hausman_results = _perform_hausman_test(fe_results, re_results)
        # Reuse tables from default apply()
        out = []
        fe_table = pd.DataFrame({
            'Variable': fe_results['coefficients'].index,
            'Coefficient': fe_results['coefficients'].values,
            'Std. Error': fe_results['std_errors'].values
        })
        fe_table['Coefficient'] = fe_table['Coefficient'].round(4)
        fe_table['Std. Error'] = fe_table['Std. Error'].round(4)
        out.append({"type": "table", "title": "Fixed Effects Model Results", "data": fe_table})
        re_table = pd.DataFrame({
            'Variable': re_results['coefficients'].index,
            'Coefficient': re_results['coefficients'].values,
            'Std. Error': re_results['std_errors'].values
        })
        re_table['Coefficient'] = re_table['Coefficient'].round(4)
        out.append({"type": "table", "title": "Random Effects Model Results", "data": re_table})
        if 'hausman_statistic' in hausman_results:
            haus_tbl = pd.DataFrame({'Test': ['Hausman Statistic', 'P-value', 'Degrees of Freedom'], 'Value': [hausman_results['hausman_statistic'], hausman_results['p_value'], hausman_results['degrees_of_freedom']]})
            haus_tbl['Value'] = haus_tbl['Value'].round(4)
            out.append({"type": "table", "title": "Hausman Test Results", "data": haus_tbl})
        return out
    except Exception as exc:
        return [{"type": "text", "title": "Hausman Test Error", "data": f"Error performing Hausman test: {str(exc)}"}]
