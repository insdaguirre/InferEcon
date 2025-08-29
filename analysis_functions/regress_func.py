from __future__ import annotations

from typing import List, Optional
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
warnings.filterwarnings('ignore')

display_name = "Regress"

def _get_numeric_columns(df: pd.DataFrame, exclude_cols: List[str] = None) -> List[str]:
    """Get numeric columns excluding specified columns."""
    if exclude_cols is None:
        exclude_cols = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [col for col in numeric_cols if col not in exclude_cols]

def _calculate_vif(X: pd.DataFrame) -> pd.DataFrame:
    """Calculate Variance Inflation Factors."""
    vif_data = []
    for i, col in enumerate(X.columns):
        try:
            vif = variance_inflation_factor(X.values, i)
            vif_data.append({'Variable': col, 'VIF': vif})
        except:
            vif_data.append({'Variable': col, 'VIF': np.nan})
    
    vif_df = pd.DataFrame(vif_data)
    vif_df['VIF'] = vif_df['VIF'].round(3)
    return vif_df

def _heteroskedasticity_tests(model, X: pd.DataFrame, y: pd.Series) -> dict:
    """Perform heteroskedasticity tests."""
    results = {}
    
    # Breusch-Pagan test
    try:
        bp_stat, bp_pval, bp_fstat, bp_fpval = het_breuschpagan(model.resid, X)
        results['Breusch-Pagan'] = {
            'Statistic': bp_stat,
            'P-value': bp_pval,
            'F-statistic': bp_fstat,
            'F P-value': bp_fpval
        }
    except:
        results['Breusch-Pagan'] = {'Error': 'Could not compute'}
    
    # White test
    try:
        white_stat, white_pval, white_fstat, white_fpval = het_white(model.resid, X)
        results['White'] = {
            'Statistic': white_stat,
            'P-value': white_pval,
            'F-statistic': white_fstat,
            'F P-value': white_fpval
        }
    except:
        results['White'] = {'Error': 'Could not compute'}
    
    return results

def _ramsey_reset_test(model, X: pd.DataFrame, y: pd.Series) -> dict:
    """Perform Ramsey RESET test for functional form."""
    try:
        # Add squared and cubed fitted values
        fitted = model.fittedvalues
        X_with_fitted = X.copy()
        X_with_fitted['fitted_sq'] = fitted**2
        X_with_fitted['fitted_cubed'] = fitted**3
        
        # Test if these terms are significant
        test_model = sm.OLS(y, sm.add_constant(X_with_fitted)).fit()
        
        # F-test for the additional terms
        from scipy import stats
        rss_restricted = model.ssr
        rss_unrestricted = test_model.ssr
        df_restricted = len(X.columns) + 1
        df_unrestricted = len(X_with_fitted.columns) + 1
        
        f_stat = ((rss_restricted - rss_unrestricted) / (df_unrestricted - df_restricted)) / (rss_unrestricted / (len(y) - df_unrestricted))
        p_value = 1 - stats.f.cdf(f_stat, df_unrestricted - df_restricted, len(y) - df_unrestricted)
        
        return {
            'F-statistic': f_stat,
            'P-value': p_value,
            'Interpretation': 'Reject H0: functional form is adequate' if p_value < 0.05 else 'Fail to reject H0: functional form appears adequate'
        }
    except Exception as e:
        return {'Error': f'Could not compute: {str(e)}'}

def apply(df: pd.DataFrame) -> List[dict]:
    """Perform OLS regression with comprehensive diagnostics."""
    outputs = []
    
    # Find numeric columns
    numeric_cols = _get_numeric_columns(df)
    
    if len(numeric_cols) < 2:
        return [{"type": "text", "title": "Regress", "data": "Need at least 2 numeric variables for regression analysis."}]
    
    # For demonstration, use first column as dependent variable, others as independent
    # In a real app, you'd want user input for variable selection
    y_col = numeric_cols[0]
    X_cols = numeric_cols[1:min(6, len(numeric_cols))]  # Limit to 5 independent variables
    
    # Prepare data
    y = df[y_col].dropna()
    X = df[X_cols].dropna()
    
    # Align indices
    common_idx = y.index.intersection(X.index)
    if len(common_idx) < 10:
        return [{"type": "text", "title": "Regress", "data": "Insufficient overlapping data for regression analysis."}]
    
    y = y.loc[common_idx]
    X = X.loc[common_idx]
    
    # Add constant
    X_with_const = sm.add_constant(X)
    
    # Fit OLS model
    try:
        model = sm.OLS(y, X_with_const).fit()
        
        # 1. Basic OLS Results
        summary_stats = pd.DataFrame({
            'Statistic': [
                'R-squared', 'Adjusted R-squared', 'F-statistic', 'F P-value',
                'AIC', 'BIC', 'Observations', 'Degrees of Freedom Residuals'
            ],
            'Value': [
                model.rsquared, model.rsquared_adj, model.fvalue, model.f_pvalue,
                model.aic, model.bic, model.nobs, model.df_resid
            ]
        })
        summary_stats['Value'] = summary_stats['Value'].round(4)
        
        outputs.append({
            "type": "table", 
            "title": "Regression Summary Statistics", 
            "data": summary_stats
        })
        
        # 2. Coefficient Table
        coef_table = pd.DataFrame({
            'Variable': ['Constant'] + X_cols,
            'Coefficient': [model.params[0]] + list(model.params[1:]),
            'Std. Error': [model.bse[0]] + list(model.bse[1:]),
            't-statistic': [model.tvalues[0]] + list(model.tvalues[1:]),
            'P-value': [model.pvalues[0]] + list(model.pvalues[1:]),
            '95% CI Lower': [model.conf_int().iloc[0, 0]] + list(model.conf_int().iloc[1:, 0]),
            '95% CI Upper': [model.conf_int().iloc[0, 1]] + list(model.conf_int().iloc[1:, 1])
        })
        
        # Round numeric columns
        numeric_cols_to_round = ['Coefficient', 'Std. Error', 't-statistic', 'P-value', '95% CI Lower', '95% CI Upper']
        for col in numeric_cols_to_round:
            coef_table[col] = coef_table[col].round(4)
        
        outputs.append({
            "type": "table", 
            "title": "Coefficient Estimates", 
            "data": coef_table
        })
        
        # 3. VIF Analysis (estat vif)
        vif_df = _calculate_vif(X)
        outputs.append({
            "type": "table", 
            "title": "Variance Inflation Factors (VIF)", 
            "data": vif_df
        })
        
        # VIF interpretation
        vif_interpretation = []
        high_vif_vars = vif_df[vif_df['VIF'] > 10]['Variable'].tolist()
        if high_vif_vars:
            vif_interpretation.append(f"⚠️ **High VIF (>10) detected for**: {', '.join(high_vif_vars)}")
            vif_interpretation.append("This suggests potential multicollinearity issues.")
        else:
            vif_interpretation.append("✅ **VIF values look reasonable** (all < 10)")
        
        outputs.append({
            "type": "text", 
            "title": "VIF Interpretation", 
            "data": "\n".join(vif_interpretation)
        })
        
        # 4. Heteroskedasticity Tests (estat hettest)
        het_tests = _heteroskedasticity_tests(model, X_with_const, y)
        
        het_results = []
        for test_name, test_results in het_tests.items():
            if 'Error' not in test_results:
                het_results.append({
                    'Test': test_name,
                    'Statistic': test_results['Statistic'],
                    'P-value': test_results['P-value'],
                    'F-statistic': test_results['F-statistic'],
                    'F P-value': test_results['F P-value']
                })
        
        if het_results:
            het_df = pd.DataFrame(het_results)
            het_df = het_df.round(4)
            outputs.append({
                "type": "table", 
                "title": "Heteroskedasticity Tests", 
                "data": het_df
            })
            
            # Interpretation
            het_interpretation = []
            for _, row in het_df.iterrows():
                if row['P-value'] < 0.05:
                    het_interpretation.append(f"⚠️ **{row['Test']}**: Reject H0 - Heteroskedasticity detected (p = {row['P-value']:.4f})")
                else:
                    het_interpretation.append(f"✅ **{row['Test']}**: Fail to reject H0 - No heteroskedasticity (p = {row['P-value']:.4f})")
            
            outputs.append({
                "type": "text", 
                "title": "Heteroskedasticity Interpretation", 
                "data": "\n".join(het_interpretation)
            })
        
        # 5. Ramsey RESET Test (estat ovtest)
        reset_test = _ramsey_reset_test(model, X_with_const, y)
        if 'Error' not in reset_test:
            reset_df = pd.DataFrame([reset_test])
            reset_df = reset_df.round(4)
            outputs.append({
                "type": "table", 
                "title": "Ramsey RESET Test (Functional Form)", 
                "data": reset_df
            })
            
            reset_interpretation = reset_test['Interpretation']
            outputs.append({
                "type": "text", 
                "title": "RESET Test Interpretation", 
                "data": reset_interpretation
            })
        
        # 6. Residuals Summary
        residuals = model.resid
        fitted = model.fittedvalues
        
        resid_stats = pd.DataFrame({
            'Statistic': ['Mean', 'Std Dev', 'Min', 'Max', 'Skewness', 'Kurtosis'],
            'Value': [
                residuals.mean(),
                residuals.std(),
                residuals.min(),
                residuals.max(),
                stats.skew(residuals),
                stats.kurtosis(residuals)
            ]
        })
        resid_stats['Value'] = resid_stats['Value'].round(4)
        
        outputs.append({
            "type": "table", 
            "title": "Residuals Summary Statistics", 
            "data": resid_stats
        })
        
        # 7. Residuals vs Fitted Values (rvfplot equivalent)
        rvf_data = pd.DataFrame({
            'Fitted Values': fitted,
            'Residuals': residuals
        })
        
        outputs.append({
            "type": "table", 
            "title": "Residuals vs Fitted Values (First 20 observations)", 
            "data": rvf_data.head(20)
        })
        
        # Model specification
        model_spec = f"""
**Model Specification:**
- **Dependent Variable**: {y_col}
- **Independent Variables**: {', '.join(X_cols)}
- **Sample Size**: {len(y)}
- **Model**: OLS (Ordinary Least Squares)
        """
        
        outputs.append({
            "type": "text", 
            "title": "Model Information", 
            "data": model_spec
        })
        
    except Exception as e:
        outputs.append({
            "type": "text", 
            "title": "Regression Error", 
            "data": f"Error performing regression: {str(e)}"
        })
    
    return outputs
