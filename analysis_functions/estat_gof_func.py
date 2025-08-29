from __future__ import annotations

from typing import List
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

display_name = "Estat GOF"

def _calculate_pseudo_r2(y_true: pd.Series, y_pred: pd.Series, model_type: str = 'linear') -> dict:
    """Calculate pseudo R-squared for different model types."""
    results = {}
    
    if model_type == 'linear':
        # Standard R-squared
        r2 = r2_score(y_true, y_pred)
        results['R²'] = r2
        
        # Adjusted R-squared
        n = len(y_true)
        p = 1  # Number of parameters (simplified)
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        results['Adjusted R²'] = adj_r2
        
        # McFadden's R-squared (pseudo R²)
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - y_true.mean()) ** 2)
        mcfadden_r2 = 1 - (ss_res / ss_tot)
        results['McFadden R²'] = mcfadden_r2
        
    else:
        # For non-linear models, use pseudo R-squared measures
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - y_true.mean()) ** 2)
        
        # McFadden's R-squared
        mcfadden_r2 = 1 - (ss_res / ss_tot)
        results['McFadden R²'] = mcfadden_r2
        
        # Cox & Snell R-squared
        n = len(y_true)
        cox_snell_r2 = 1 - (ss_res / ss_tot) ** (2/n)
        results['Cox & Snell R²'] = cox_snell_r2
        
        # Nagelkerke R² (adjusted Cox & Snell)
        nagelkerke_r2 = cox_snell_r2 / (1 - (ss_res / ss_tot) ** (2/n))
        results['Nagelkerke R²'] = nagelkerke_r2
    
    return results

def _calculate_error_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    """Calculate various error metrics."""
    # Remove any NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    
    if len(y_true_clean) == 0:
        return {}
    
    # Mean Squared Error
    mse = mean_squared_error(y_true_clean, y_pred_clean)
    
    # Root Mean Squared Error
    rmse = np.sqrt(mse)
    
    # Mean Absolute Error
    mae = mean_absolute_error(y_true_clean, y_pred_clean)
    
    # Mean Absolute Percentage Error
    mape = np.mean(np.abs((y_true_clean - y_pred_clean) / y_true_clean)) * 100
    
    # Median Absolute Error
    medae = np.median(np.abs(y_true_clean - y_pred_clean))
    
    # Maximum Error
    max_error = np.max(np.abs(y_true_clean - y_pred_clean))
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE (%)': mape,
        'MedAE': medae,
        'Max Error': max_error
    }

def _calculate_distribution_tests(y_true: pd.Series, y_pred: pd.Series) -> dict:
    """Calculate distribution and normality tests for residuals."""
    residuals = y_true - y_pred
    
    # Remove NaN values
    residuals = residuals.dropna()
    
    if len(residuals) < 3:
        return {}
    
    results = {}
    
    # Normality test (Shapiro-Wilk)
    try:
        shapiro_stat, shapiro_p = stats.shapiro(residuals)
        results['Shapiro-Wilk Statistic'] = shapiro_stat
        results['Shapiro-Wilk P-value'] = shapiro_p
        results['Normality (Shapiro-Wilk)'] = 'Normal' if shapiro_p > 0.05 else 'Not Normal'
    except:
        results['Normality (Shapiro-Wilk)'] = 'Could not compute'
    
    # Jarque-Bera test
    try:
        jb_stat, jb_p = stats.jarque_bera(residuals)
        results['Jarque-Bera Statistic'] = jb_stat
        results['Jarque-Bera P-value'] = jb_p
        results['Normality (Jarque-Bera)'] = 'Normal' if jb_p > 0.05 else 'Not Normal'
    except:
        results['Normality (Jarque-Bera)'] = 'Could not compute'
    
    # Residual statistics
    results['Residual Mean'] = residuals.mean()
    results['Residual Std'] = residuals.std()
    results['Residual Skewness'] = stats.skew(residuals)
    results['Residual Kurtosis'] = stats.kurtosis(residuals)
    
    return results

def _calculate_prediction_accuracy(y_true: pd.Series, y_pred: pd.Series) -> dict:
    """Calculate prediction accuracy metrics."""
    # Remove NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    
    if len(y_true_clean) == 0:
        return {}
    
    # Direction accuracy (for time series or ordered data)
    if len(y_true_clean) > 1:
        true_direction = np.diff(y_true_clean)
        pred_direction = np.diff(y_pred_clean)
        
        # Count correct direction predictions
        correct_direction = np.sum((true_direction > 0) == (pred_direction > 0))
        direction_accuracy = correct_direction / len(true_direction) * 100
    else:
        direction_accuracy = np.nan
    
    # Hit rate (predictions within certain percentage of actual)
    percentage_errors = np.abs((y_true_clean - y_pred_clean) / y_true_clean) * 100
    
    hit_rates = {}
    for threshold in [5, 10, 15, 20]:
        hit_rate = np.mean(percentage_errors <= threshold) * 100
        hit_rates[f'Hit Rate (±{threshold}%)'] = hit_rate
    
    results = {
        'Direction Accuracy (%)': direction_accuracy
    }
    results.update(hit_rates)
    
    return results

def apply(df: pd.DataFrame) -> List[dict]:
    """Calculate comprehensive goodness of fit statistics for regression models."""
    outputs = []
    
    # Find numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        return [{"type": "text", "title": "Estat GOF", "data": "Need at least 2 numeric variables for goodness of fit analysis."}]
    
    # Use first column as dependent variable, others as independent
    y_col = numeric_cols[0]
    X_cols = numeric_cols[1:min(6, len(numeric_cols))]  # Limit to 5 independent variables
    
    # Prepare data
    y = df[y_col].dropna()
    X = df[X_cols].dropna()
    
    # Align indices
    common_idx = y.index.intersection(X.index)
    if len(common_idx) < 10:
        return [{"type": "text", "title": "Estat GOF", "data": "Insufficient overlapping data for goodness of fit analysis."}]
    
    y = y.loc[common_idx]
    X = X.loc[common_idx]
    
    # Fit OLS model
    try:
        X_with_const = sm.add_constant(X)
        model = sm.OLS(y, X_with_const).fit()
        
        # Get predictions
        y_pred = model.fittedvalues
        
        # 1. R-squared and Pseudo R-squared measures
        r2_measures = _calculate_pseudo_r2(y, y_pred, 'linear')
        r2_df = pd.DataFrame(list(r2_measures.items()), columns=['Measure', 'Value'])
        r2_df['Value'] = r2_df['Value'].round(4)
        
        outputs.append({
            "type": "table", 
            "title": "R-squared and Pseudo R-squared Measures", 
            "data": r2_df
        })
        
        # 2. Error Metrics
        error_metrics = _calculate_error_metrics(y, y_pred)
        error_df = pd.DataFrame(list(error_metrics.items()), columns=['Metric', 'Value'])
        error_df['Value'] = error_df['Value'].round(4)
        
        outputs.append({
            "type": "table", 
            "title": "Error Metrics", 
            "data": error_df
        })
        
        # 3. Distribution Tests
        dist_tests = _calculate_distribution_tests(y, y_pred)
        dist_df = pd.DataFrame(list(dist_tests.items()), columns=['Test', 'Value'])
        
        # Round numeric values
        numeric_mask = dist_df['Value'].apply(lambda x: isinstance(x, (int, float)) and not pd.isna(x))
        dist_df.loc[numeric_mask, 'Value'] = dist_df.loc[numeric_mask, 'Value'].round(4)
        
        outputs.append({
            "type": "table", 
            "title": "Residual Distribution Tests", 
            "data": dist_df
        })
        
        # 4. Prediction Accuracy
        pred_accuracy = _calculate_prediction_accuracy(y, y_pred)
        pred_df = pd.DataFrame(list(pred_accuracy.items()), columns=['Metric', 'Value'])
        pred_df['Value'] = pred_df['Value'].round(2)
        
        outputs.append({
            "type": "table", 
            "title": "Prediction Accuracy Metrics", 
            "data": pred_df
        })
        
        # 5. Model Fit Summary
        fit_summary = pd.DataFrame({
            'Statistic': [
                'Observations', 'Parameters', 'Degrees of Freedom',
                'Sum of Squared Residuals', 'Mean Squared Error',
                'Root Mean Squared Error', 'Mean Absolute Error'
            ],
            'Value': [
                model.nobs, len(model.params), model.df_resid,
                model.ssr, model.mse_resid, np.sqrt(model.mse_resid),
                mean_absolute_error(y, y_pred)
            ]
        })
        fit_summary['Value'] = fit_summary['Value'].round(4)
        
        outputs.append({
            "type": "table", 
            "title": "Model Fit Summary", 
            "data": fit_summary
        })
        
        # 6. Goodness of Fit Interpretation
        interpretation = []
        
        # R-squared interpretation
        r2_val = r2_measures.get('R²', 0)
        if r2_val >= 0.9:
            r2_quality = "Excellent"
        elif r2_val >= 0.7:
            r2_quality = "Good"
        elif r2_val >= 0.5:
            r2_quality = "Moderate"
        else:
            r2_quality = "Poor"
        
        interpretation.append(f"**R² = {r2_val:.4f}**: {r2_quality} fit")
        
        # Error interpretation
        rmse_val = error_metrics.get('RMSE', 0)
        mae_val = error_metrics.get('MAE', 0)
        interpretation.append(f"**RMSE = {rmse_val:.4f}**: Average prediction error")
        interpretation.append(f"**MAE = {mae_val:.4f}**: Average absolute prediction error")
        
        # Normality interpretation
        normality = dist_tests.get('Normality (Shapiro-Wilk)', 'Unknown')
        interpretation.append(f"**Residual Normality**: {normality}")
        
        # Hit rate interpretation
        hit_rate_10 = pred_accuracy.get('Hit Rate (±10%)', 0)
        interpretation.append(f"**Prediction Accuracy**: {hit_rate_10:.1f}% of predictions within ±10% of actual values")
        
        interpretation_text = "\n".join(interpretation)
        outputs.append({
            "type": "text", 
            "title": "Goodness of Fit Interpretation", 
            "data": interpretation_text
        })
        
    except Exception as e:
        outputs.append({
            "type": "text", 
            "title": "Goodness of Fit Error", 
            "data": f"Error calculating goodness of fit statistics: {str(e)}"
        })
    
    return outputs
