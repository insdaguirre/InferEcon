from __future__ import annotations

from typing import List
import pandas as pd
import numpy as np

display_name = "Tabstat"

def apply(df: pd.DataFrame) -> List[dict]:
    """Create custom summary statistics table like Stata's tabstat command."""
    outputs = []
    
    # Find numeric variables
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_cols:
        return [{"type": "text", "title": "Tabstat", "data": "No numeric variables found for tabstat analysis."}]
    
    # Calculate various statistics
    stats_data = []
    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) == 0:
            continue
            
        stats_data.append({
            'Variable': col,
            'N': len(series),
            'Mean': series.mean(),
            'Std. Dev.': series.std(),
            'Min': series.min(),
            'Max': series.max(),
            'Median': series.median(),
            '25th Pct.': series.quantile(0.25),
            '75th Pct.': series.quantile(0.75),
            'Range': series.max() - series.min(),
            'IQR': series.quantile(0.75) - series.quantile(0.25),
            'CV': (series.std() / series.mean()) if series.mean() != 0 else np.nan
        })
    
    if stats_data:
        stats_df = pd.DataFrame(stats_data)
        
        # Round numeric columns
        numeric_cols_to_round = ['Mean', 'Std. Dev.', 'Min', 'Max', 'Median', '25th Pct.', '75th Pct.', 'Range', 'IQR', 'CV']
        for col in numeric_cols_to_round:
            if col in stats_df.columns:
                stats_df[col] = stats_df[col].round(4)
        
        outputs.append({
            "type": "table", 
            "title": "Summary Statistics (All Variables)", 
            "data": stats_df
        })
        
        # Transposed view (variables as columns, stats as rows)
        stats_transposed = stats_df.set_index('Variable').T
        outputs.append({
            "type": "table", 
            "title": "Summary Statistics (Transposed View)", 
            "data": stats_transposed
        })
        
        # Summary by variable type (if we can categorize)
        if len(numeric_cols) > 1:
            # Group variables by their characteristics
            var_groups = []
            for _, row in stats_df.iterrows():
                cv = row['CV']
                if pd.isna(cv):
                    group = "Unknown"
                elif cv < 0.1:
                    group = "Low Variation (CV < 0.1)"
                elif cv < 0.5:
                    group = "Medium Variation (CV 0.1-0.5)"
                else:
                    group = "High Variation (CV > 0.5)"
                var_groups.append(group)
            
            stats_df_with_group = stats_df.copy()
            stats_df_with_group['Variation Group'] = var_groups
            
            group_summary = stats_df_with_group.groupby('Variation Group').agg({
                'N': 'count',
                'Mean': 'mean',
                'Std. Dev.': 'mean',
                'CV': 'mean'
            }).round(4)
            
            outputs.append({
                "type": "table", 
                "title": "Summary by Variation Group", 
                "data": group_summary.reset_index()
            })
    else:
        outputs.append({
            "type": "text", 
            "title": "Tabstat", 
            "data": "No valid numeric data found for tabstat analysis."
        })
    
    return outputs
