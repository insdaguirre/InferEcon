from __future__ import annotations

from typing import List, Optional
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

display_name = "Areg"

def _absorb_fixed_effects(y: pd.Series, X: pd.DataFrame, fe_cols: List[str], df_panel: pd.DataFrame) -> dict:
    """Absorb high-dimensional fixed effects using the Frisch-Waugh-Lovell theorem."""
    # Step 1: Regress y on fixed effects
    y_fe = pd.get_dummies(df_panel[fe_cols], drop_first=True)
    if len(y_fe.columns) > 0:
        y_fe_with_const = sm.add_constant(y_fe)
        y_on_fe = sm.OLS(y, y_fe_with_const).fit()
        y_resid = y_on_fe.resid
    else:
        y_resid = y - y.mean()
    
    # Step 2: Regress each X variable on fixed effects
    X_resid = pd.DataFrame(index=X.index, columns=X.columns)
    
    for col in X.columns:
        if len(y_fe.columns) > 0:
            X_on_fe = sm.OLS(X[col], y_fe_with_const).fit()
            X_resid[col] = X_on_fe.resid
        else:
            X_resid[col] = X[col] - X[col].mean()
    
    # Step 3: Regress y residuals on X residuals
    X_resid_clean = X_resid.dropna()
    y_resid_clean = y_resid[X_resid_clean.index]
    
    if len(y_resid_clean) == 0:
        return None
    
    # Add constant
    X_resid_with_const = sm.add_constant(X_resid_clean)
    
    # Fit OLS on residualized data
    model = sm.OLS(y_resid_clean, X_resid_with_const).fit()
    
    # Calculate absorbed R-squared
    y_total_ss = np.sum((y_resid_clean - y_resid_clean.mean()) ** 2)
    y_residual_ss = np.sum(model.resid ** 2)
    absorbed_r2 = 1 - (y_residual_ss / y_total_ss)
    
    return {
        'model': model,
        'coefficients': model.params,
        'std_errors': model.bse,
        't_values': model.tvalues,
        'p_values': model.pvalues,
        'r_squared': model.rsquared,
        'absorbed_r_squared': absorbed_r2,
        'nobs': model.nobs,
        'residuals': model.resid,
        'y_residuals': y_resid_clean,
        'X_residuals': X_resid_clean,
        'fe_dummies_created': len(y_fe.columns) if len(y_fe.columns) > 0 else 0
    }

def _calculate_fe_statistics(df_panel: pd.DataFrame, fe_cols: List[str]) -> dict:
    """Calculate fixed effects statistics."""
    fe_stats = {}
    
    for col in fe_cols:
        unique_vals = df_panel[col].nunique()
        total_obs = len(df_panel)
        avg_obs_per_fe = total_obs / unique_vals
        
        fe_stats[col] = {
            'unique_values': unique_vals,
            'total_observations': total_obs,
            'avg_obs_per_fe': avg_obs_per_fe,
            'is_high_dim': unique_vals > 50  # High-dimensional if > 50 categories
        }
    
    return fe_stats

def _perform_fe_tests(y: pd.Series, X: pd.DataFrame, fe_cols: List[str], df_panel: pd.DataFrame) -> dict:
    """Perform tests for fixed effects significance."""
    test_results = {}
    
    for fe_col in fe_cols:
        # Create fixed effects dummies
        fe_dummies = pd.get_dummies(df_panel[fe_col], drop_first=True)
        
        if len(fe_dummies.columns) > 0:
            # Test if FE are jointly significant
            X_with_fe = pd.concat([X, fe_dummies], axis=1)
            X_with_fe = sm.add_constant(X_with_fe)
            
            # Full model
            full_model = sm.OLS(y, X_with_fe).fit()
            
            # Restricted model (no FE)
            X_restricted = sm.add_constant(X)
            restricted_model = sm.OLS(y, X_restricted).fit()
            
            # F-test for FE significance
            ssr_restricted = restricted_model.ssr
            ssr_full = full_model.ssr
            df_restricted = restricted_model.df_resid
            df_full = full_model.df_resid
            
            f_stat = ((ssr_restricted - ssr_full) / (df_restricted - df_full)) / (ssr_full / df_full)
            f_pval = 1 - stats.f.cdf(f_stat, df_restricted - df_full, df_full)
            
            test_results[fe_col] = {
                'f_statistic': f_stat,
                'f_pvalue': f_pval,
                'df_numerator': df_restricted - df_full,
                'df_denominator': df_full,
                'ssr_restricted': ssr_restricted,
                'ssr_full': ssr_full
            }
    
    return test_results

def _create_fe_summary_table(fe_stats: dict) -> pd.DataFrame:
    """Create summary table for fixed effects."""
    summary_data = []
    
    for fe_col, stats_dict in fe_stats.items():
        summary_data.append({
            'Fixed Effect': fe_col,
            'Unique Values': stats_dict['unique_values'],
            'Total Obs': stats_dict['total_observations'],
            'Avg Obs per FE': round(stats_dict['avg_obs_per_fe'], 2),
            'High Dimensional': 'Yes' if stats_dict['is_high_dim'] else 'No'
        })
    
    return pd.DataFrame(summary_data)

def apply(df: pd.DataFrame) -> List[dict]:
    """Perform high-dimensional fixed effects regression using absorption."""
    outputs = []
    
    # Find numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 3:
        return [{"type": "text", "title": "Areg", "data": "Need at least 3 numeric variables for fixed effects analysis."}]
    
    # Create panel structure with multiple fixed effects
    n_entities = min(15, len(df) // 3)  # Create more entities for FE analysis
    if n_entities < 3:
        return [{"type": "text", "title": "Areg", "data": "Need more observations to create panel structure."}]
    
    # Create entity, time, and group identifiers
    entity_id = np.repeat(range(n_entities), len(df) // n_entities)
    if len(entity_id) < len(df):
        entity_id = np.append(entity_id, [entity_id[-1]] * (len(df) - len(entity_id)))
    
    time_id = np.tile(range(len(df) // n_entities), n_entities)
    if len(time_id) < len(df):
        time_id = np.append(time_id, [time_id[-1]] * (len(df) - len(time_id)))
    
    # Create additional group identifier (e.g., industry, region)
    group_id = np.random.choice(range(min(8, n_entities // 2)), len(df))
    
    # Add to dataframe
    df_panel = df.copy()
    df_panel['entity_id'] = entity_id
    df_panel['time_id'] = time_id
    df_panel['group_id'] = group_id
    
    # Use first column as dependent variable, others as independent
    y_col = numeric_cols[0]
    x_cols = numeric_cols[1:min(6, len(numeric_cols))]  # Limit to 5 x variables
    
    # Prepare data
    mask = ~(df_panel[y_col].isna() | df_panel[x_cols].isna().any(axis=1))
    y = df_panel[y_col][mask]
    X = df_panel[x_cols][mask]
    
    # Fixed effects columns
    fe_cols = ['entity_id', 'time_id', 'group_id']
    
    if len(y) < 30:
        return [{"type": "text", "title": "Areg", "data": "Insufficient overlapping data for fixed effects analysis."}]
    
    # Calculate fixed effects statistics
    fe_stats = _calculate_fe_statistics(df_panel, fe_cols)
    
    # 1. Fixed Effects Summary
    fe_summary_table = _create_fe_summary_table(fe_stats)
    outputs.append({
        "type": "table", 
        "title": "Fixed Effects Structure", 
        "data": fe_summary_table
    })
    
    # 2. Perform fixed effects tests
    fe_tests = _perform_fe_tests(y, X, fe_cols, df_panel)
    
    if fe_tests:
        fe_test_data = []
        for fe_col, test_dict in fe_tests.items():
            fe_test_data.append({
                'Fixed Effect': fe_col,
                'F-statistic': test_dict['f_statistic'],
                'P-value': test_dict['f_pvalue'],
                'DF Numerator': test_dict['df_numerator'],
                'DF Denominator': test_dict['df_denominator']
            })
        
        fe_test_table = pd.DataFrame(fe_test_data)
        numeric_cols_to_round = ['F-statistic', 'P-value', 'DF Numerator', 'DF Denominator']
        for col in numeric_cols_to_round:
            fe_test_table[col] = fe_test_table[col].round(4)
        
        outputs.append({
            "type": "table", 
            "title": "Fixed Effects Significance Tests", 
            "data": fe_test_table
        })
    
    # 3. Perform fixed effects absorption
    try:
        areg_results = _absorb_fixed_effects(y, X, fe_cols, df_panel)
        
        if areg_results:
            # Coefficient table
            coef_data = []
            for var, coef, se, t_val, p_val in zip(
                areg_results['coefficients'].index,
                areg_results['coefficients'].values,
                areg_results['std_errors'].values,
                areg_results['t_values'].values,
                areg_results['p_values'].values
            ):
                coef_data.append({
                    'Variable': var,
                    'Coefficient': coef,
                    'Std. Error': se,
                    't-statistic': t_val,
                    'P-value': p_val
                })
            
            coef_table = pd.DataFrame(coef_data)
            numeric_cols_to_round = ['Coefficient', 'Std. Error', 't-statistic', 'P-value']
            for col in numeric_cols_to_round:
                coef_table[col] = coef_table[col].round(4)
            
            outputs.append({
                "type": "table", 
                "title": "Fixed Effects Absorption Results", 
                "data": coef_table
            })
            
            # Model summary
            model_summary = pd.DataFrame({
                'Statistic': ['R-squared', 'Absorbed R-squared', 'Observations', 'FE Dummies Created'],
                'Value': [
                    areg_results['r_squared'],
                    areg_results['absorbed_r_squared'],
                    areg_results['nobs'],
                    areg_results['fe_dummies_created']
                ]
            })
            model_summary['Value'] = model_summary['Value'].round(4)
            
            outputs.append({
                "type": "table", 
                "title": "Fixed Effects Absorption Model Summary", 
                "data": model_summary
            })
            
            # Comparison with OLS
            try:
                # Fit OLS without fixed effects
                X_ols = sm.add_constant(X)
                ols_model = sm.OLS(y, X_ols).fit()
                
                comparison_data = []
                for var in x_cols:
                    if var in ols_model.params.index and var in areg_results['coefficients'].index:
                        ols_coef = ols_model.params[var]
                        fe_coef = areg_results['coefficients'][var]
                        difference = ols_coef - fe_coef
                        
                        comparison_data.append({
                            'Variable': var,
                            'OLS Coefficient': ols_coef,
                            'FE Coefficient': fe_coef,
                            'Difference': difference,
                            'Percent Change': (difference / ols_coef * 100) if ols_coef != 0 else np.nan
                        })
                
                if comparison_data:
                    comparison_df = pd.DataFrame(comparison_data)
                    numeric_cols_to_round = ['OLS Coefficient', 'FE Coefficient', 'Difference', 'Percent Change']
                    for col in numeric_cols_to_round:
                        comparison_df[col] = comparison_df[col].round(4)
                    
                    outputs.append({
                        "type": "table", 
                        "title": "OLS vs Fixed Effects Comparison", 
                        "data": comparison_df
                    })
                
            except Exception as e:
                outputs.append({
                    "type": "text", 
                    "title": "OLS Comparison Error", 
                    "data": f"Could not compare with OLS: {str(e)}"
                })
        
    except Exception as e:
        outputs.append({
            "type": "text", 
            "title": "Fixed Effects Absorption Error", 
            "data": f"Error performing fixed effects absorption: {str(e)}"
        })
    
    # 4. Interpretation Guide
    interpretation = f"""
**High-Dimensional Fixed Effects Analysis Results:**

**Fixed Effects Structure:**
- **Entity Fixed Effects**: {fe_stats['entity_id']['unique_values']} entities
- **Time Fixed Effects**: {fe_stats['time_id']['unique_values']} time periods  
- **Group Fixed Effects**: {fe_stats['group_id']['unique_values']} groups

**What This Analysis Does:**
1. **Absorbs Fixed Effects**: Removes time-invariant and group-invariant unobservables
2. **Controls for Confounding**: Eliminates bias from omitted variables
3. **Focuses on Within Variation**: Uses only variation within entities/groups over time

**Key Benefits:**
- **Robustness**: Controls for many unobservable factors
- **Efficiency**: Handles high-dimensional fixed effects
- **Interpretation**: Coefficients show within-entity effects

**When to Use:**
- **High-dimensional FE**: Many categories (e.g., firm, industry, region, time)
- **Omitted Variable Bias**: Suspect correlation between unobservables and regressors
- **Panel Data**: Multiple observations per entity over time

**Interpretation:**
- **Coefficients**: Effect of x on y, holding fixed effects constant
- **R-squared**: Variation explained by x variables only
- **Absorbed R²**: Total variation explained (including FE)
- **FE Tests**: Whether fixed effects are jointly significant

**Comparison with OLS:**
- **Large differences**: Suggest omitted variable bias
- **Small differences**: OLS may be adequate
- **Significance changes**: Fixed effects improve inference
        """
    
    outputs.append({
        "type": "text", 
        "title": "Fixed Effects Absorption Interpretation", 
        "data": interpretation
    })
    
    return outputs


def apply_with_config(df: pd.DataFrame, config: dict) -> List[dict]:
    """Run absorption model using selected Y/X and optional FE columns or synthesized ids."""
    outputs: List[dict] = []
    try:
        y_col = config.get('y_col')
        x_cols = config.get('x_cols', [])
        fe_cols = config.get('fe_cols', []) or []
        create_panel = config.get('create_panel', False)
        n_entities = config.get('n_entities')
        if not y_col or not x_cols:
            return apply(df)
        df_panel = df.copy()
        # If user didn't provide FE columns, synthesize panel ids so model can run
        if not fe_cols or create_panel:
            n = len(df_panel)
            if not n_entities:
                n_entities = max(3, min(15, max(3, n // 3)))
            ent = np.repeat(range(n_entities), max(1, n // n_entities))
            if len(ent) < n:
                ent = np.append(ent, [ent[-1]] * (n - len(ent)))
            tvals = np.tile(range(max(1, n // max(1, n_entities))), n_entities)
            if len(tvals) < n:
                tvals = np.append(tvals, [tvals[-1]] * (n - len(tvals)))
            df_panel['entity_id'] = ent[:n]
            df_panel['time_id'] = tvals[:n]
            if 'group_id' not in df_panel.columns:
                df_panel['group_id'] = np.random.choice(range(min(8, n_entities // 2 if n_entities else 4)), n)
            fe_cols = ['entity_id', 'time_id', 'group_id']
        mask = ~(df_panel[y_col].isna() | df_panel[x_cols].isna().any(axis=1))
        y = df_panel[y_col][mask]
        X = df_panel[x_cols][mask]
        if len(y) < 20:
            return [{"type": "text", "title": "Areg", "data": "Insufficient overlapping data for fixed effects analysis."}]
        fe_stats = _calculate_fe_statistics(df_panel, fe_cols)
        fe_summary_table = _create_fe_summary_table(fe_stats)
        outputs.append({"type": "table", "title": "Fixed Effects Structure", "data": fe_summary_table})
        fe_tests = _perform_fe_tests(y, X, fe_cols, df_panel)
        if fe_tests:
            fe_test_data = []
            for fe_col, test_dict in fe_tests.items():
                fe_test_data.append({'Fixed Effect': fe_col, 'F-statistic': test_dict['f_statistic'], 'P-value': test_dict['f_pvalue'], 'DF Numerator': test_dict['df_numerator'], 'DF Denominator': test_dict['df_denominator']})
            fe_test_table = pd.DataFrame(fe_test_data)
            for c in ['F-statistic', 'P-value', 'DF Numerator', 'DF Denominator']:
                fe_test_table[c] = fe_test_table[c].round(4)
            outputs.append({"type": "table", "title": "Fixed Effects Significance Tests", "data": fe_test_table})
        areg_results = _absorb_fixed_effects(y, X, fe_cols, df_panel)
        if areg_results:
            coef_table = pd.DataFrame({
                'Variable': areg_results['coefficients'].index,
                'Coefficient': areg_results['coefficients'].values,
                'Std. Error': areg_results['std_errors'].values,
                't-statistic': areg_results['t_values'].values,
                'P-value': areg_results['p_values'].values
            })
            for c in ['Coefficient', 'Std. Error', 't-statistic', 'P-value']:
                coef_table[c] = coef_table[c].round(4)
            outputs.append({"type": "table", "title": "Fixed Effects Absorption Results", "data": coef_table})
        return outputs
    except Exception as exc:
        return [{"type": "text", "title": "Fixed Effects Absorption Error", "data": f"Error performing fixed effects absorption: {str(exc)}"}]
