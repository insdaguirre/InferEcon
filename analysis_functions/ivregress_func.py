from __future__ import annotations

from typing import List, Tuple
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

display_name = "Ivregress"

def _first_stage_regression(x: pd.Series, z: pd.Series, other_x: pd.DataFrame = None) -> dict:
    """First stage regression: regress endogenous variable on instruments."""
    if other_x is not None and len(other_x.columns) > 0:
        X_first = pd.concat([z, other_x], axis=1)
    else:
        X_first = z.to_frame()
    
    X_first = sm.add_constant(X_first)
    model_first = sm.OLS(x, X_first).fit()
    
    return {
        'model': model_first,
        'fitted_values': model_first.fittedvalues,
        'r_squared': model_first.rsquared,
        'f_stat': model_first.fvalue,
        'f_pval': model_first.f_pvalue
    }

def _second_stage_regression(y: pd.Series, x_fitted: pd.Series, other_x: pd.DataFrame = None) -> dict:
    """Second stage regression: regress dependent variable on fitted endogenous variable."""
    if other_x is not None and len(other_x.columns) > 0:
        X_second = pd.concat([x_fitted, other_x], axis=1)
    else:
        X_second = x_fitted.to_frame()
    
    X_second = sm.add_constant(X_second)
    model_second = sm.OLS(y, X_second).fit()
    
    return {
        'model': model_second,
        'coefficients': model_second.params,
        'std_errors': model_second.bse,
        't_values': model_second.tvalues,
        'p_values': model_second.pvalues,
        'r_squared': model_second.rsquared,
        'adj_r_squared': model_second.rsquared_adj
    }

def _calculate_iv_statistics(y: pd.Series, x: pd.Series, z: pd.Series, 
                           other_x: pd.DataFrame = None) -> dict:
    """Calculate IV-specific statistics."""
    # First stage
    first_stage = _first_stage_regression(x, z, other_x)
    
    # Second stage
    second_stage = _second_stage_regression(y, first_stage['fitted_values'], other_x)
    
    # Overidentification test (if we have more instruments than endogenous variables)
    if z.shape[1] > 1:  # Multiple instruments
        # Calculate residuals from second stage
        y_pred = second_stage['model'].fittedvalues
        residuals = y - y_pred
        
        # Regress residuals on instruments
        if other_x is not None and len(other_x.columns) > 0:
            X_overid = pd.concat([z, other_x], axis=1)
        else:
            X_overid = z
        
        X_overid = sm.add_constant(X_overid)
        overid_model = sm.OLS(residuals, X_overid).fit()
        
        # J-statistic (Hansen's J)
        j_stat = overid_model.nobs * overid_model.rsquared
        j_pval = 1 - stats.chi2.cdf(j_stat, z.shape[1] - 1)
        
        overid_test = {
            'j_statistic': j_stat,
            'j_pvalue': j_pval,
            'df': z.shape[1] - 1
        }
    else:
        overid_test = None
    
    # Weak instruments test (F-statistic from first stage)
    weak_instruments = {
        'f_statistic': first_stage['f_stat'],
        'f_pvalue': first_stage['f_pval'],
        'r_squared': first_stage['r_squared']
    }
    
    # Endogeneity test (Durbin-Wu-Hausman test)
    # Compare OLS vs IV estimates
    if other_x is not None and len(other_x.columns) > 0:
        X_ols = pd.concat([x, other_x], axis=1)
    else:
        X_ols = x.to_frame()
    
    X_ols = sm.add_constant(X_ols)
    ols_model = sm.OLS(y, X_ols).fit()
    
    # Get coefficient on endogenous variable
    ols_coef = ols_model.params[1]  # Assuming x is the first variable
    iv_coef = second_stage['coefficients'][1]
    
    # Calculate test statistic
    ols_var = ols_model.bse[1] ** 2
    iv_var = second_stage['std_errors'][1] ** 2
    
    if iv_var > ols_var:
        dwh_stat = (ols_coef - iv_coef) ** 2 / (iv_var - ols_var)
        dwh_pval = 1 - stats.chi2.cdf(dwh_stat, 1)
        
        endogeneity_test = {
            'dwh_statistic': dwh_stat,
            'dwh_pvalue': dwh_pval,
            'ols_coefficient': ols_coef,
            'iv_coefficient': iv_coef,
            'difference': ols_coef - iv_coef
        }
    else:
        endogeneity_test = None
    
    return {
        'first_stage': first_stage,
        'second_stage': second_stage,
        'overidentification': overid_test,
        'weak_instruments': weak_instruments,
        'endogeneity': endogeneity_test
    }

def apply(df: pd.DataFrame) -> List[dict]:
    """Perform instrumental variables estimation with 2SLS."""
    outputs = []
    
    # Find numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 3:
        return [{"type": "text", "title": "Ivregress", "data": "Need at least 3 numeric variables for IV estimation (y, x, z)."}]
    
    # Use first column as dependent variable, second as endogenous, third as instrument
    y_col = numeric_cols[0]
    x_col = numeric_cols[1]  # Endogenous variable
    z_col = numeric_cols[2]  # Instrument
    
    # Additional control variables (if any)
    control_cols = numeric_cols[3:min(8, len(numeric_cols))]  # Limit to 5 control variables
    
    # Prepare data
    mask = ~(df[y_col].isna() | df[x_col].isna() | df[z_col].isna())
    if control_cols:
        mask = mask & ~df[control_cols].isna().any(axis=1)
    
    y = df[y_col][mask]
    x = df[x_col][mask]
    z = df[z_col][mask]
    controls = df[control_cols][mask] if control_cols else pd.DataFrame()
    
    if len(y) < 10:
        return [{"type": "text", "title": "Ivregress", "data": "Insufficient overlapping data for IV estimation."}]
    
    # Perform IV estimation
    try:
        iv_results = _calculate_iv_statistics(y, x, z, controls)
        
        # 1. First Stage Results
        first_stage = iv_results['first_stage']
        first_stage_table = pd.DataFrame({
            'Statistic': ['R-squared', 'F-statistic', 'F P-value', 'Observations'],
            'Value': [
                first_stage['r_squared'],
                first_stage['f_stat'],
                first_stage['f_pval'],
                len(y)
            ]
        })
        first_stage_table['Value'] = first_stage_table['Value'].round(4)
        
        outputs.append({
            "type": "table", 
            "title": "First Stage Results", 
            "data": first_stage_table
        })
        
        # 2. Second Stage Results
        second_stage = iv_results['second_stage']
        
        # Coefficient table
        coef_data = []
        for var, coef, se, t_val, p_val in zip(
            second_stage['coefficients'].index,
            second_stage['coefficients'].values,
            second_stage['std_errors'].values,
            second_stage['t_values'].values,
            second_stage['p_values'].values
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
            "title": "Second Stage Results - Coefficient Estimates", 
            "data": coef_table
        })
        
        # 3. Model Summary
        model_summary = pd.DataFrame({
            'Statistic': ['R-squared', 'Adjusted R-squared', 'Observations'],
            'Value': [
                second_stage['r_squared'],
                second_stage['adj_r_squared'],
                len(y)
            ]
        })
        model_summary['Value'] = model_summary['Value'].round(4)
        
        outputs.append({
            "type": "table", 
            "title": "IV Model Summary", 
            "data": model_summary
        })
        
        # 4. Weak Instruments Test
        weak_instruments = iv_results['weak_instruments']
        weak_instruments_table = pd.DataFrame({
            'Test': ['F-statistic', 'P-value', 'R-squared'],
            'Value': [
                weak_instruments['f_statistic'],
                weak_instruments['f_pvalue'],
                weak_instruments['r_squared']
            ]
        })
        weak_instruments_table['Value'] = weak_instruments_table['Value'].round(4)
        
        outputs.append({
            "type": "table", 
            "title": "Weak Instruments Test", 
            "data": weak_instruments_table
        })
        
        # 5. Overidentification Test (if applicable)
        if iv_results['overidentification']:
            overid = iv_results['overidentification']
            overid_table = pd.DataFrame({
                'Test': ['J-statistic', 'P-value', 'Degrees of Freedom'],
                'Value': [
                    overid['j_statistic'],
                    overid['j_pvalue'],
                    overid['df']
                ]
            })
            overid_table['Value'] = overid_table['Value'].round(4)
            
            outputs.append({
                "type": "table", 
                "title": "Overidentification Test (Hansen's J)", 
                "data": overid_table
            })
        
        # 6. Endogeneity Test
        if iv_results['endogeneity']:
            endogeneity = iv_results['endogeneity']
            endogeneity_table = pd.DataFrame({
                'Test': ['DWH Statistic', 'P-value', 'OLS Coef', 'IV Coef', 'Difference'],
                'Value': [
                    endogeneity['dwh_statistic'],
                    endogeneity['dwh_pvalue'],
                    endogeneity['ols_coefficient'],
                    endogeneity['iv_coefficient'],
                    endogeneity['difference']
                ]
            })
            endogeneity_table['Value'] = endogeneity_table['Value'].round(4)
            
            outputs.append({
                "type": "table", 
                "title": "Endogeneity Test (Durbin-Wu-Hausman)", 
                "data": endogeneity_table
            })
        
        # 7. Interpretation and Diagnostics
        interpretation = f"""
**Instrumental Variables (2SLS) Analysis Results:**

**Model Specification:**
- **Dependent Variable**: {y_col}
- **Endogenous Variable**: {x_col}
- **Instrument**: {z_col}
- **Control Variables**: {', '.join(control_cols) if control_cols else 'None'}

**Key Diagnostics:**

**1. Weak Instruments Test:**
- F-statistic: {weak_instruments['f_statistic']:.4f}
- **Rule of thumb**: F > 10 suggests instruments are not weak
- **Your result**: {'✅ Instruments appear strong' if weak_instruments['f_statistic'] > 10 else '⚠️ Instruments may be weak'}

**2. Overidentification Test:**
- J-statistic: {overid['j_statistic']:.4f if iv_results['overidentification'] else 'N/A'}
- **Interpretation**: {'✅ Instruments are valid' if iv_results['overidentification'] and overid['j_pvalue'] > 0.05 else '⚠️ Some instruments may be invalid' if iv_results['overidentification'] else 'N/A (exactly identified)'}

**3. Endogeneity Test:**
- DWH statistic: {endogeneity['dwh_statistic']:.4f if iv_results['endogeneity'] else 'N/A'}
- **Interpretation**: {'✅ Variable is exogenous (OLS is consistent)' if iv_results['endogeneity'] and endogeneity['dwh_pvalue'] > 0.05 else '⚠️ Variable is endogenous (IV is needed)' if iv_results['endogeneity'] else 'N/A'}

**When to use IV:**
- When you suspect endogeneity (reverse causality, omitted variables)
- When you have valid instruments (exogenous, relevant, not weak)
- When OLS estimates are biased and inconsistent

**Instrument Validity Requirements:**
1. **Relevance**: Correlated with endogenous variable (F > 10)
2. **Exogeneity**: Not correlated with error term
3. **Exclusion**: Only affects y through x
        """
        
        outputs.append({
            "type": "text", 
            "title": "IV Analysis Interpretation", 
            "data": interpretation
        })
        
    except Exception as e:
        outputs.append({
            "type": "text", 
            "title": "IV Estimation Error", 
            "data": f"Error performing IV estimation: {str(e)}"
        })
    
    return outputs


def apply_with_config(df: pd.DataFrame, config: dict) -> List[dict]:
    """Perform IV estimation using user-selected columns from configuration.

    Expected config keys:
    - y_col: dependent variable name
    - x_col: endogenous regressor name
    - z_col: instrument column name (or multiple instruments list)
    - control_vars: optional list of control variable names
    """
    outputs: List[dict] = []
    try:
        y_col = config.get('y_col')
        x_col = config.get('x_col')
        z_col = config.get('z_col')
        control_cols = config.get('control_vars', []) or []
        if not y_col or not x_col or not z_col:
            return apply(df)
        # Support multiple instruments if user provided a list
        z_df = df[[z_col]] if isinstance(z_col, str) else df[z_col]
        mask = ~(df[y_col].isna() | df[x_col].isna() | z_df.isna().any(axis=1))
        if control_cols:
            mask = mask & ~df[control_cols].isna().any(axis=1)
        y = df.loc[mask, y_col]
        x = df.loc[mask, x_col]
        z = z_df.loc[mask]
        controls = df.loc[mask, control_cols] if control_cols else pd.DataFrame()
        if len(y) < 10:
            return [{"type": "text", "title": "Ivregress", "data": "Insufficient overlapping data for IV estimation."}]
        iv_results = _calculate_iv_statistics(y, x, z, controls)
        # Reuse reporting from default apply by building minimal tables
        first_stage = iv_results['first_stage']
        first_stage_table = pd.DataFrame({
            'Statistic': ['R-squared', 'F-statistic', 'F P-value', 'Observations'],
            'Value': [first_stage['r_squared'], first_stage['f_stat'], first_stage['f_pval'], len(y)]
        })
        first_stage_table['Value'] = first_stage_table['Value'].round(4)
        outputs.append({"type": "table", "title": "First Stage Results", "data": first_stage_table})
        second_stage = iv_results['second_stage']
        coef_table = pd.DataFrame({
            'Variable': second_stage['coefficients'].index,
            'Coefficient': second_stage['coefficients'].values,
            'Std. Error': second_stage['std_errors'].values,
            't-statistic': second_stage['t_values'].values,
            'P-value': second_stage['p_values'].values
        })
        for c in ['Coefficient', 'Std. Error', 't-statistic', 'P-value']:
            coef_table[c] = coef_table[c].round(4)
        outputs.append({"type": "table", "title": "Second Stage Results - Coefficient Estimates", "data": coef_table})
        model_summary = pd.DataFrame({
            'Statistic': ['R-squared', 'Adjusted R-squared', 'Observations'],
            'Value': [second_stage['r_squared'], second_stage['adj_r_squared'], len(y)]
        })
        model_summary['Value'] = model_summary['Value'].round(4)
        outputs.append({"type": "table", "title": "IV Model Summary", "data": model_summary})
        weak_instruments = iv_results['weak_instruments']
        weak_tbl = pd.DataFrame({'Test': ['F-statistic', 'P-value', 'R-squared'], 'Value': [weak_instruments['f_statistic'], weak_instruments['f_pvalue'], weak_instruments['r_squared']]})
        weak_tbl['Value'] = weak_tbl['Value'].round(4)
        outputs.append({"type": "table", "title": "Weak Instruments Test", "data": weak_tbl})
        if iv_results['overidentification']:
            overid = iv_results['overidentification']
            over_tbl = pd.DataFrame({'Test': ['J-statistic', 'P-value', 'Degrees of Freedom'], 'Value': [overid['j_statistic'], overid['j_pvalue'], overid['df']]})
            over_tbl['Value'] = over_tbl['Value'].round(4)
            outputs.append({"type": "table", "title": "Overidentification Test (Hansen's J)", "data": over_tbl})
        if iv_results['endogeneity']:
            end = iv_results['endogeneity']
            end_tbl = pd.DataFrame({'Test': ['DWH Statistic', 'P-value', 'OLS Coef', 'IV Coef', 'Difference'], 'Value': [end['dwh_statistic'], end['dwh_pvalue'], end['ols_coefficient'], end['iv_coefficient'], end['difference']]})
            end_tbl['Value'] = end_tbl['Value'].round(4)
            outputs.append({"type": "table", "title": "Endogeneity Test (Durbin-Wu-Hausman)", "data": end_tbl})
        interpretation = f"""
**Instrumental Variables (2SLS) Analysis Results:**

**Model Specification:**
- **Dependent Variable**: {y_col}
- **Endogenous Variable**: {x_col}
- **Instrument(s)**: {z_col if isinstance(z_col, str) else ', '.join(z_col)}
- **Control Variables**: {', '.join(control_cols) if control_cols else 'None'}
"""
        outputs.append({"type": "text", "title": "IV Analysis Interpretation", "data": interpretation})
        return outputs
    except Exception as exc:
        return [{"type": "text", "title": "IV Estimation Error", "data": f"Error performing IV estimation: {str(exc)}"}]
