from __future__ import annotations

from typing import List
import pandas as pd

display_name = "List"

def apply(df: pd.DataFrame) -> List[dict]:
    """Show data rows like Stata's list command."""
    outputs = []
    
    # Show first 10 rows
    first_rows = df.head(10)
    outputs.append({
        "type": "table", 
        "title": "First 10 Rows", 
        "data": first_rows
    })
    
    # Show last 10 rows
    last_rows = df.tail(10)
    outputs.append({
        "type": "table", 
        "title": "Last 10 Rows", 
        "data": last_rows
    })
    
    # Show random sample of 10 rows
    if len(df) > 10:
        random_sample = df.sample(n=min(10, len(df)), random_state=42)
        outputs.append({
            "type": "table", 
            "title": "Random Sample (10 rows)", 
            "data": random_sample
        })
    
    # Data info summary
    info_data = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        non_null = df[col].count()
        null_count = df[col].isnull().sum()
        unique_count = df[col].nunique()
        
        info_data.append({
            'Variable': col,
            'Type': dtype,
            'Non-Null': non_null,
            'Null': null_count,
            'Unique': unique_count
        })
    
    info_df = pd.DataFrame(info_data)
    outputs.append({
        "type": "table", 
        "title": "Data Information", 
        "data": info_df
    })
    
    return outputs
