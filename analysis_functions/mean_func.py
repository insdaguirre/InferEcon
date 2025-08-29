from __future__ import annotations

from typing import List
import pandas as pd
import numpy as np
from scipy import stats

display_name = "Mean"

def apply(df: pd.DataFrame) -> List[dict]:
    """Calculate means with standard errors and confidence intervals for numeric variables."""
    outputs = []
    
    # Find numeric variables
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_cols:
        return [{"type": "text", "title": "Mean", "data": "No numeric variables found for mean analysis."}]
    
    # Calculate means with SEs and CIs
    mean_results = []
    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) == 0:
            continue
            
        mean_val = series.mean()
        std_err = series.std() / np.sqrt(len(series))
        
        # 95% confidence interval
        ci_lower, ci_upper = stats.t.interval(0.95, len(series)-1, loc=mean_val, scale=std_err)
        
        mean_results.append({
            'Variable': col,
            'Mean': mean_val,
            'Std. Err.': std_err,
            '95% CI Lower': ci_lower,
            '95% CI Upper': ci_upper,
            'N': len(series)
        })
    
    if mean_results:
        mean_df = pd.DataFrame(mean_results)
        # Round numeric columns
        numeric_cols_to_round = ['Mean', 'Std. Err.', '95% CI Lower', '95% CI Upper']
        for col in numeric_cols_to_round:
            if col in mean_df.columns:
                mean_df[col] = mean_df[col].round(4)
        
        outputs.append({
            "type": "table", 
            "title": "Means with Standard Errors and 95% CIs", 
            "data": mean_df
        })
        
        # Summary of confidence intervals
        ci_summary = []
        for _, row in mean_df.iterrows():
            ci_width = row['95% CI Upper'] - row['95% CI Lower']
            ci_summary.append({
                'Variable': row['Variable'],
                'Mean': row['Mean'],
                'CI Width': ci_width,
                'Precision': f"±{ci_width/2:.4f}"
            })
        
        ci_summary_df = pd.DataFrame(ci_summary)
        ci_summary_df['CI Width'] = ci_summary_df['CI Width'].round(4)
        
        outputs.append({
            "type": "table", 
            "title": "Confidence Interval Summary", 
            "data": ci_summary_df
        })
    else:
        outputs.append({
            "type": "text", 
            "title": "Mean", 
            "data": "No valid numeric data found for mean analysis."
        })
    
    return outputs
