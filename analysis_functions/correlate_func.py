from __future__ import annotations

from typing import List
import pandas as pd
import numpy as np

display_name = "Correlate"

def apply(df: pd.DataFrame) -> List[dict]:
    """Create correlation matrices for numeric variables."""
    outputs = []
    
    # Find numeric variables
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        return [{"type": "text", "title": "Correlate", "data": "Need at least 2 numeric variables for correlation analysis."}]
    
    # Create correlation matrix
    corr_matrix = df[numeric_cols].corr().round(4)
    
    # Add significance indicators (simple approach)
    corr_with_p = corr_matrix.copy()
    for i, col1 in enumerate(numeric_cols):
        for j, col2 in enumerate(numeric_cols):
            if i != j:
                # Simple significance indicator based on correlation strength
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) >= 0.7:
                    corr_with_p.iloc[i, j] = f"{corr_val:.4f}***"
                elif abs(corr_val) >= 0.5:
                    corr_with_p.iloc[i, j] = f"{corr_val:.4f}**"
                elif abs(corr_val) >= 0.3:
                    corr_with_p.iloc[i, j] = f"{corr_val:.4f}*"
                else:
                    corr_with_p.iloc[i, j] = f"{corr_val:.4f}"
    
    outputs.append({
        "type": "table", 
        "title": "Correlation Matrix", 
        "data": corr_with_p
    })
    
    # Summary statistics for correlations
    corr_values = []
    for i in range(len(numeric_cols)):
        for j in range(i+1, len(numeric_cols)):
            corr_values.append(corr_matrix.iloc[i, j])
    
    if corr_values:
        summary_stats = pd.DataFrame({
            'Statistic': ['Count', 'Mean', 'Std Dev', 'Min', 'Max'],
            'Value': [
                len(corr_values),
                np.mean(corr_values),
                np.std(corr_values),
                np.min(corr_values),
                np.max(corr_values)
            ]
        })
        summary_stats['Value'] = summary_stats['Value'].round(4)
        
        outputs.append({
            "type": "table", 
            "title": "Correlation Summary", 
            "data": summary_stats
        })
    
    return outputs
