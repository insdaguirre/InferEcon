from __future__ import annotations

from typing import List
import pandas as pd
import numpy as np

display_name = "Describe"

def apply(df: pd.DataFrame) -> List[dict]:
    """Provide detailed variable information like Stata's describe command."""
    outputs = []
    
    # Variable properties
    var_info = []
    for col in df.columns:
        series = df[col]
        dtype = str(series.dtype)
        non_null = series.count()
        null_count = series.isnull().sum()
        null_pct = (null_count / len(df) * 100) if len(df) > 0 else 0
        
        # For numeric variables, add more details
        if np.issubdtype(series.dtype, np.number):
            var_type = "numeric"
            if series.nunique() <= 20:
                var_type += " (categorical)"
        else:
            var_type = "string"
            if series.nunique() <= 20:
                var_type += " (categorical)"
        
        var_info.append({
            'Variable': col,
            'Type': var_type,
            'Storage': dtype,
            'Display': dtype,
            'Non-Null': non_null,
            'Null': null_count,
            'Null %': f"{null_pct:.1f}%",
            'Unique': series.nunique()
        })
    
    var_info_df = pd.DataFrame(var_info)
    outputs.append({
        "type": "table", 
        "title": "Variable Properties", 
        "data": var_info_df
    })
    
    # Memory usage
    memory_usage = df.memory_usage(deep=True)
    total_memory = memory_usage.sum()
    memory_df = pd.DataFrame({
        'Variable': memory_usage.index,
        'Memory (bytes)': memory_usage.values,
        'Memory (MB)': (memory_usage.values / 1024 / 1024).round(3)
    })
    memory_df = memory_df.sort_values('Memory (bytes)', ascending=False)
    
    outputs.append({
        "type": "table", 
        "title": "Memory Usage", 
        "data": memory_df
    })
    
    # Dataset summary
    summary_data = {
        'Property': [
            'Observations',
            'Variables', 
            'Total Memory (MB)',
            'Numeric Variables',
            'String Variables',
            'Missing Values',
            'Missing %'
        ],
        'Value': [
            len(df),
            len(df.columns),
            f"{total_memory / 1024 / 1024:.3f}",
            len(df.select_dtypes(include=[np.number]).columns),
            len(df.select_dtypes(include=['object']).columns),
            df.isnull().sum().sum(),
            f"{df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100:.1f}%"
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    outputs.append({
        "type": "table", 
        "title": "Dataset Summary", 
        "data": summary_df
    })
    
    return outputs
