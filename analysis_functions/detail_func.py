from __future__ import annotations

from typing import List
import pandas as pd
import numpy as np
from scipy import stats

display_name = "Detail"

def apply(df: pd.DataFrame) -> List[dict]:
    """Provide detailed summary with percentiles, skewness, and kurtosis like Stata's detail command."""
    outputs = []
    
    # Find numeric variables
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_cols:
        return [{"type": "text", "title": "Detail", "data": "No numeric variables found for detailed analysis."}]
    
    # Calculate detailed statistics for each variable
    for col in numeric_cols[:5]:  # Limit to first 5 to avoid overwhelming output
        series = df[col].dropna()
        if len(series) == 0:
            continue
            
        # Basic statistics
        n = len(series)
        mean_val = series.mean()
        std_val = series.std()
        min_val = series.min()
        max_val = series.max()
        
        # Percentiles
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        percentile_values = [series.quantile(p/100) for p in percentiles]
        
        # Shape statistics
        skewness = stats.skew(series)
        kurtosis = stats.kurtosis(series)
        
        # Create detailed stats table
        detail_stats = pd.DataFrame({
            'Statistic': [
                'N', 'Mean', 'Std. Dev.', 'Min', 'Max',
                '1st Percentile', '5th Percentile', '10th Percentile',
                '25th Percentile', 'Median', '75th Percentile',
                '90th Percentile', '95th Percentile', '99th Percentile',
                'Skewness', 'Kurtosis'
            ],
            'Value': [
                n, mean_val, std_val, min_val, max_val,
                *percentile_values,
                skewness, kurtosis
            ]
        })
        
        # Round numeric values
        detail_stats['Value'] = detail_stats['Value'].round(4)
        
        outputs.append({
            "type": "table", 
            "title": f"Detailed Statistics: {col}", 
            "data": detail_stats
        })
        
        # Additional insights
        insights = []
        
        # Skewness interpretation
        if abs(skewness) < 0.5:
            skew_interpretation = "Approximately symmetric"
        elif skewness > 0.5:
            skew_interpretation = "Right-skewed (positive skew)"
        else:
            skew_interpretation = "Left-skewed (negative skew)"
        
        # Kurtosis interpretation
        if abs(kurtosis) < 0.5:
            kurt_interpretation = "Approximately normal"
        elif kurtosis > 0.5:
            kurt_interpretation = "Heavy-tailed (leptokurtic)"
        else:
            kurt_interpretation = "Light-tailed (platykurtic)"
        
        # Outlier detection using IQR method
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = series[(series < lower_bound) | (series > upper_bound)]
        
        insights.append(f"**Skewness**: {skewness:.4f} - {skew_interpretation}")
        insights.append(f"**Kurtosis**: {kurtosis:.4f} - {kurt_interpretation}")
        insights.append(f"**IQR**: {iqr:.4f}")
        insights.append(f"**Outlier Bounds**: [{lower_bound:.4f}, {upper_bound:.4f}]")
        insights.append(f"**Potential Outliers**: {len(outliers)} observations")
        
        if len(outliers) > 0:
            insights.append(f"**Outlier Values**: {outliers.values[:5].tolist()}")
        
        # Normality test
        if n >= 3:
            try:
                _, p_value = stats.normaltest(series)
                if p_value < 0.05:
                    insights.append(f"**Normality Test**: p = {p_value:.4f} - Data may not be normally distributed")
                else:
                    insights.append(f"**Normality Test**: p = {p_value:.4f} - Data appears approximately normal")
            except:
                insights.append("**Normality Test**: Could not compute")
        
        insights_text = "\n".join(insights)
        outputs.append({
            "type": "text", 
            "title": f"Insights: {col}", 
            "data": insights_text
        })
    
    return outputs
