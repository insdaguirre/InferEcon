from __future__ import annotations

from typing import List
import pandas as pd
import numpy as np

display_name = "Tabulate"

def apply(df: pd.DataFrame) -> List[dict]:
    """Create frequency tables and cross-tabulations for categorical variables."""
    outputs = []
    
    # Find categorical variables (object type or low cardinality numeric)
    categorical_cols = []
    for col in df.columns:
        if df[col].dtype == 'object' or df[col].nunique() <= 20:
            categorical_cols.append(col)
    
    if not categorical_cols:
        return [{"type": "text", "title": "Tabulate", "data": "No categorical variables found for tabulation."}]
    
    # Single variable frequency tables
    for col in categorical_cols[:3]:  # Limit to first 3 to avoid overwhelming output
        freq_table = df[col].value_counts().reset_index()
        freq_table.columns = [col, 'Frequency']
        freq_table['Percent'] = (freq_table['Frequency'] / freq_table['Frequency'].sum() * 100).round(2)
        freq_table['Cumulative'] = freq_table['Percent'].cumsum().round(2)
        
        outputs.append({
            "type": "table", 
            "title": f"Frequency table: {col}", 
            "data": freq_table
        })
    
    # Cross-tabulation for first two categorical variables
    if len(categorical_cols) >= 2:
        col1, col2 = categorical_cols[0], categorical_cols[1]
        crosstab = pd.crosstab(df[col1], df[col2], margins=True, margins_name='Total')
        
        # Add percentages
        crosstab_pct = pd.crosstab(df[col1], df[col2], normalize='index', margins=True, margins_name='Total') * 100
        
        outputs.append({
            "type": "table", 
            "title": f"Cross-tabulation: {col1} × {col2} (Counts)", 
            "data": crosstab
        })
        
        outputs.append({
            "type": "table", 
            "title": f"Cross-tabulation: {col1} × {col2} (Row %)",
            "data": crosstab_pct.round(2)
        })
    
    return outputs
