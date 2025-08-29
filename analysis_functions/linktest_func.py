from __future__ import annotations

from typing import List
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

display_name = "Linktest"

def _perform_linktest(y: pd.Series, X: pd.DataFrame) -> dict:
    """Perform linktest to detect model specification errors."""
    try:
        # Step 1: Fit the original model
        X_with_const = sm.add_constant(X)
        original_model = sm.OLS(y, X_with_const).fit()
        
        # Step 2: Get fitted values and their squares
        fitted_values = original_model.fittedvalues
        fitted_squared = fitted_values ** 2
        
        # Step 3: Create new design matrix with fitted values
        X_linktest = pd.DataFrame({
            'fitted': fitted_values,
            'fitted_squared': fitted_squared
        })
        X_linktest = sm.add_constant(X_linktest)
        
        # Step 4: Fit the linktest model
        linktest_model = sm.OLS(y, X_linktest).fit()
        
        # Step 5: Extract test results
        fitted_coef = linktest_model.params['fitted']
        fitted_sq_coef = linktest_model.params['fitted_squared']
        
        fitted_tstat = linktest_model.tvalues['fitted']
        fitted_sq_tstat = linktest_model.tvalues['fitted_squared']
        
        fitted_pval = linktest_model.pvalues['fitted']
        fitted_sq_pval = linktest_model.pvalues['fitted_squared']
        
        # Step 6: Calculate test statistics
        linktest_stat = fitted_sq_coef ** 2 / linktest_model.bse['fitted_squared'] ** 2
        linktest_pval = 1 - stats.chi2.cdf(linktest_stat, 1)
        
        return {
            'fitted_coef': fitted_coef,
            'fitted_tstat': fitted_tstat,
            'fitted_pval': fitted_pval,
            'fitted_sq_coef': fitted_sq_coef,
            'fitted_sq_tstat': fitted_sq_tstat,
            'fitted_sq_pval': fitted_sq_pval,
            'linktest_stat': linktest_stat,
            'linktest_pval': linktest_pval,
            'r_squared': linktest_model.rsquared,
            'adj_r_squared': linktest_model.rsquared_adj,
            'f_stat': linktest_model.fvalue,
            'f_pval': linktest_model.f_pvalue
        }
        
    except Exception as e:
        return {'error': str(e)}

def _perform_ovtest(y: pd.Series, X: pd.DataFrame) -> dict:
    """Perform Ramsey's RESET test (ovtest equivalent)."""
    try:
        # Fit original model
        X_with_const = sm.add_constant(X)
        original_model = sm.OLS(y, X_with_const).fit()
        
        # Get fitted values
        fitted_values = original_model.fittedvalues
        
        # Create augmented model with fitted^2 and fitted^3
        X_augmented = X_with_const.copy()
        X_augmented['fitted_squared'] = fitted_values ** 2
        X_augmented['fitted_cubed'] = fitted_values ** 3
        
        # Fit augmented model
        augmented_model = sm.OLS(y, X_augmented).fit()
        
        # F-test for the additional terms
        rss_restricted = original_model.ssr
        rss_unrestricted = augmented_model.ssr
        df_restricted = len(X.columns) + 1
        df_unrestricted = len(X_augmented.columns)
        
        f_stat = ((rss_restricted - rss_unrestricted) / (df_unrestricted - df_restricted)) / (rss_unrestricted / (len(y) - df_unrestricted))
        p_value = 1 - stats.f.cdf(f_stat, df_unrestricted - df_restricted, len(y) - df_unrestricted)
        
        return {
            'f_statistic': f_stat,
            'p_value': p_value,
            'df_numerator': df_unrestricted - df_restricted,
            'df_denominator': len(y) - df_unrestricted
        }
        
    except Exception as e:
        return {'error': str(e)}

def _perform_heteroskedasticity_test(y: pd.Series, X: pd.DataFrame) -> dict:
    """Perform heteroskedasticity test using fitted values."""
    try:
        # Fit model
        X_with_const = sm.add_constant(X)
        model = sm.OLS(y, X_with_const).fit()
        
        # Get residuals and fitted values
        residuals = model.resid
        fitted_values = model.fittedvalues
        
        # Test heteroskedasticity against fitted values
        # Regress squared residuals on fitted values
        squared_residuals = residuals ** 2
        het_model = sm.OLS(squared_residuals, sm.add_constant(fitted_values)).fit()
        
        # F-test for heteroskedasticity
        f_stat = het_model.fvalue
        f_pval = het_model.f_pvalue
        
        # R-squared of heteroskedasticity regression
        het_r2 = het_model.rsquared
        
        return {
            'f_statistic': f_stat,
            'p_value': f_pval,
            'r_squared': het_r2,
            'interpretation': 'Heteroskedasticity detected' if f_pval < 0.05 else 'No heteroskedasticity'
        }
        
    except Exception as e:
        return {'error': str(e)}

def apply(df: pd.DataFrame) -> List[dict]:
    """Perform linktest and related specification error detection tests."""
    outputs = []
    
    # Find numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        return [{"type": "text", "title": "Linktest", "data": "Need at least 2 numeric variables for linktest analysis."}]
    
    # Use first column as dependent variable, others as independent
    y_col = numeric_cols[0]
    X_cols = numeric_cols[1:min(6, len(numeric_cols))]  # Limit to 5 independent variables
    
    # Prepare data
    y = df[y_col].dropna()
    X = df[X_cols].dropna()
    
    # Align indices
    common_idx = y.index.intersection(X.index)
    if len(common_idx) < 10:
        return [{"type": "text", "title": "Linktest", "data": "Insufficient overlapping data for linktest analysis."}]
    
    y = y.loc[common_idx]
    X = X.loc[common_idx]
    
    # 1. Linktest Results
    linktest_results = _perform_linktest(y, X)
    
    if 'error' not in linktest_results:
        # Linktest coefficients table
        linktest_coefs = pd.DataFrame({
            'Variable': ['Constant', 'Fitted', 'Fitted²'],
            'Coefficient': [
                linktest_results.get('fitted_coef', 0) - linktest_results.get('fitted_sq_coef', 0),
                linktest_results.get('fitted_coef', 0),
                linktest_results.get('fitted_sq_coef', 0)
            ],
            't-statistic': [
                np.nan,  # Constant t-stat not directly available
                linktest_results.get('fitted_tstat', 0),
                linktest_results.get('fitted_sq_tstat', 0)
            ],
            'P-value': [
                np.nan,  # Constant p-value not directly available
                linktest_results.get('fitted_pval', 0),
                linktest_results.get('fitted_sq_pval', 0)
            ]
        })
        
        # Round numeric columns
        numeric_cols_to_round = ['Coefficient', 't-statistic', 'P-value']
        for col in numeric_cols_to_round:
            linktest_coefs[col] = linktest_coefs[col].round(4)
        
        outputs.append({
            "type": "table", 
            "title": "Linktest Results - Coefficient Estimates", 
            "data": linktest_coefs
        })
        
        # Linktest summary statistics
        linktest_summary = pd.DataFrame({
            'Statistic': [
                'Linktest Statistic', 'Linktest P-value', 'R-squared', 'Adjusted R-squared',
                'F-statistic', 'F P-value'
            ],
            'Value': [
                linktest_results.get('linktest_stat', 0),
                linktest_results.get('linktest_pval', 0),
                linktest_results.get('r_squared', 0),
                linktest_results.get('adj_r_squared', 0),
                linktest_results.get('f_stat', 0),
                linktest_results.get('f_pval', 0)
            ]
        })
        linktest_summary['Value'] = linktest_summary['Value'].round(4)
        
        outputs.append({
            "type": "table", 
            "title": "Linktest Summary Statistics", 
            "data": linktest_summary
        })
        
        # Linktest interpretation
        fitted_sq_pval = linktest_results.get('fitted_sq_pval', 1)
        linktest_pval = linktest_results.get('linktest_pval', 1)
        
        interpretation = []
        
        if fitted_sq_pval < 0.05:
            interpretation.append("⚠️ **Fitted² coefficient is significant** (p < 0.05)")
            interpretation.append("This suggests potential model specification problems.")
        else:
            interpretation.append("✅ **Fitted² coefficient is not significant** (p ≥ 0.05)")
            interpretation.append("This suggests the model specification is adequate.")
        
        if linktest_pval < 0.05:
            interpretation.append("⚠️ **Linktest is significant** (p < 0.05)")
            interpretation.append("Model specification may need improvement.")
        else:
            interpretation.append("✅ **Linktest is not significant** (p ≥ 0.05)")
            interpretation.append("Model specification appears adequate.")
        
        interpretation.append(f"\n**Key Test**: Fitted² coefficient p-value = {fitted_sq_pval:.4f}")
        interpretation.append(f"**Linktest**: Chi² p-value = {linktest_pval:.4f}")
        
        interpretation_text = "\n".join(interpretation)
        outputs.append({
            "type": "text", 
            "title": "Linktest Interpretation", 
            "data": interpretation_text
        })
        
    else:
        outputs.append({
            "type": "text", 
            "title": "Linktest Error", 
            "data": f"Could not perform linktest: {linktest_results['error']}"
        })
    
    # 2. Ramsey RESET Test (ovtest)
    ovtest_results = _perform_ovtest(y, X)
    
    if 'error' not in ovtest_results:
        ovtest_df = pd.DataFrame([ovtest_results])
        ovtest_df = ovtest_df.round(4)
        outputs.append({
            "type": "table", 
            "title": "Ramsey RESET Test (ovtest)", 
            "data": ovtest_df
        })
        
        ovtest_interpretation = "Reject H0: functional form is adequate" if ovtest_results['p_value'] < 0.05 else "Fail to reject H0: functional form appears adequate"
        outputs.append({
            "type": "text", 
            "title": "RESET Test Interpretation", 
            "data": ovtest_interpretation
        })
    
    # 3. Heteroskedasticity Test
    het_results = _perform_heteroskedasticity_test(y, X)
    
    if 'error' not in het_results:
        het_df = pd.DataFrame([het_results])
        het_df = het_df.round(4)
        outputs.append({
            "type": "table", 
            "title": "Heteroskedasticity Test vs Fitted Values", 
            "data": het_df
        })
        
        het_interpretation = het_results.get('interpretation', 'Test completed')
        outputs.append({
            "type": "text", 
            "title": "Heteroskedasticity Interpretation", 
            "data": het_interpretation
        })
    
    # 4. Specification Error Summary
    summary = """
**Linktest Summary:**

**Purpose**: Detects model specification errors by testing if fitted values and their squares are significant predictors of the dependent variable.

**Interpretation**:
- **Fitted coefficient**: Should be significant (p < 0.05) - indicates the model captures the relationship
- **Fitted² coefficient**: Should NOT be significant (p ≥ 0.05) - indicates adequate specification
- **Linktest**: Overall test of specification adequacy

**What to do if problems detected**:
1. Check for omitted variables
2. Consider non-linear transformations
3. Test for interaction effects
4. Verify functional form assumptions
    """
    
    outputs.append({
        "type": "text", 
        "title": "Specification Error Detection Guide", 
        "data": summary
    })
    
    return outputs
