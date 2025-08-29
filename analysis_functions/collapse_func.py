from __future__ import annotations

from typing import List, Dict, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import io
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

display_name = "Collapse"

def _collapse_by_group(df: pd.DataFrame, group_cols: List[str], agg_cols: List[str], 
                      agg_functions: List[str]) -> pd.DataFrame:
    """Collapse dataset by grouping variables with specified aggregation functions."""
    # Remove missing values in grouping columns
    mask = ~df[group_cols].isna().any(axis=1)
    df_clean = df[mask]
    
    if len(df_clean) == 0:
        return pd.DataFrame()
    
    # Create aggregation dictionary
    agg_dict = {}
    for col in agg_cols:
        agg_dict[col] = agg_functions
    
    # Perform groupby aggregation
    collapsed = df_clean.groupby(group_cols).agg(agg_dict).reset_index()
    
    # Flatten column names if multiple aggregation functions
    if len(agg_functions) > 1:
        collapsed.columns = [f"{col[1]}_{col[0]}" if col[1] != '' else col[0] 
                           for col in collapsed.columns]
    
    return collapsed

def _create_collapse_summary(df_original: pd.DataFrame, df_collapsed: pd.DataFrame, 
                           group_cols: List[str], agg_cols: List[str]) -> pd.DataFrame:
    """Create summary comparing original vs collapsed dataset."""
    summary_data = [
        {
            'Metric': 'Original Observations',
            'Value': len(df_original),
            'Description': 'Total rows in original dataset'
        },
        {
            'Metric': 'Collapsed Observations',
            'Value': len(df_collapsed),
            'Description': 'Total rows after collapsing'
        },
        {
            'Metric': 'Reduction Factor',
            'Value': f"{len(df_original) / len(df_collapsed):.1f}x",
            'Description': 'How much the dataset was reduced'
        },
        {
            'Metric': 'Grouping Variables',
            'Value': len(group_cols),
            'Description': f"Number of grouping variables: {', '.join(group_cols)}"
        },
        {
            'Metric': 'Aggregated Variables',
            'Value': len(agg_cols),
            'Description': f"Number of variables aggregated: {', '.join(agg_cols)}"
        }
    ]
    
    return pd.DataFrame(summary_data)

def _create_group_size_analysis(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    """Analyze group sizes and distribution."""
    # Count observations per group
    group_counts = df.groupby(group_cols).size().reset_index(name='count')
    
    # Calculate statistics
    count_stats = group_counts['count'].describe()
    
    # Create summary table
    stats_data = [
        {'Statistic': 'Total Groups', 'Value': len(group_counts)},
        {'Statistic': 'Min Group Size', 'Value': int(count_stats['min'])},
        {'Statistic': 'Max Group Size', 'Value': int(count_stats['max'])},
        {'Statistic': 'Mean Group Size', 'Value': round(count_stats['mean'], 2)},
        {'Statistic': 'Median Group Size', 'Value': int(count_stats['50%'])},
        {'Statistic': 'Std Dev Group Size', 'Value': round(count_stats['std'], 2)},
        {'Statistic': 'Q25 Group Size', 'Value': int(count_stats['25%'])},
        {'Statistic': 'Q75 Group Size', 'Value': int(count_stats['75%'])}
    ]
    
    return pd.DataFrame(stats_data)

def _create_group_size_histogram(df: pd.DataFrame, group_cols: List[str]) -> str:
    """Create histogram of group sizes."""
    # Count observations per group
    group_counts = df.groupby(group_cols).size()
    
    plt.figure(figsize=(12, 8))
    
    # Create histogram
    plt.hist(group_counts.values, bins=min(30, len(group_counts.unique())), 
             alpha=0.7, edgecolor='black', linewidth=0.5, color='steelblue')
    
    # Customize plot
    plt.xlabel('Group Size (Number of Observations)', fontsize=14, fontweight='bold')
    plt.ylabel('Frequency (Number of Groups)', fontsize=14, fontweight='bold')
    plt.title(f'Distribution of Group Sizes by {", ".join(group_cols)}', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Add statistics
    stats_text = f'Total Groups: {len(group_counts)}\nMean Size: {group_counts.mean():.1f}\nMedian Size: {group_counts.median():.1f}'
    plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=11,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
    
    plt.tight_layout()
    
    # Convert to base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
    buffer.seek(0)
    plt.close()
    
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return img_str

def _create_aggregation_comparison(df_original: pd.DataFrame, df_collapsed: pd.DataFrame, 
                                 group_cols: List[str], agg_cols: List[str]) -> List[pd.DataFrame]:
    """Compare original vs collapsed statistics for aggregated variables."""
    comparison_tables = []
    
    for col in agg_cols:
        if col in df_original.columns:
            # Original statistics
            orig_stats = df_original[col].describe()
            
            # Collapsed statistics (assuming mean aggregation)
            if f"mean_{col}" in df_collapsed.columns:
                collapsed_col = f"mean_{col}"
            elif col in df_collapsed.columns:
                collapsed_col = col
            else:
                continue
            
            collapsed_stats = df_collapsed[collapsed_col].describe()
            
            # Create comparison table
            comparison_data = [
                {'Statistic': 'Count', 'Original': int(orig_stats['count']), 'Collapsed': int(collapsed_stats['count'])},
                {'Statistic': 'Mean', 'Original': round(orig_stats['mean'], 4), 'Collapsed': round(collapsed_stats['mean'], 4)},
                {'Statistic': 'Std Dev', 'Original': round(orig_stats['std'], 4), 'Collapsed': round(collapsed_stats['std'], 4)},
                {'Statistic': 'Min', 'Original': round(orig_stats['min'], 4), 'Collapsed': round(collapsed_stats['min'], 4)},
                {'Statistic': '25%', 'Original': round(orig_stats['25%'], 4), 'Collapsed': round(collapsed_stats['25%'], 4)},
                {'Statistic': '50%', 'Original': round(orig_stats['50%'], 4), 'Collapsed': round(collapsed_stats['50%'], 4)},
                {'Statistic': '75%', 'Original': round(orig_stats['75%'], 4), 'Collapsed': round(collapsed_stats['75%'], 4)},
                {'Statistic': 'Max', 'Original': round(orig_stats['max'], 4), 'Collapsed': round(collapsed_stats['max'], 4)}
            ]
            
            comparison_df = pd.DataFrame(comparison_data)
            comparison_tables.append(comparison_df)
    
    return comparison_tables

def _create_group_heatmap(df: pd.DataFrame, group_cols: List[str], agg_cols: List[str]) -> str:
    """Create heatmap showing aggregated values by groups."""
    if len(group_cols) < 2 or len(agg_cols) == 0:
        return ""
    
    # Select first two grouping variables and first aggregated variable
    group1, group2 = group_cols[0], group_cols[1]
    agg_var = agg_cols[0]
    
    # Find the aggregated column name
    if f"mean_{agg_var}" in df.columns:
        agg_col = f"mean_{agg_var}"
    elif agg_var in df.columns:
        agg_col = agg_var
    else:
        return ""
    
    # Create pivot table for heatmap
    pivot_data = df.pivot_table(values=agg_col, index=group1, columns=group2, aggfunc='mean')
    
    if pivot_data.empty or pivot_data.isna().all().all():
        return ""
    
    plt.figure(figsize=(12, 10))
    
    # Create heatmap
    sns.heatmap(pivot_data, annot=True, cmap='coolwarm', center=pivot_data.mean().mean(),
                square=True, linewidths=0.5, cbar_kws={"shrink": .8}, fmt='.3f')
    
    plt.title(f'Heatmap: {agg_var} by {group1} and {group2}', fontsize=16, fontweight='bold')
    plt.xlabel(group2, fontsize=14, fontweight='bold')
    plt.ylabel(group1, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Convert to base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
    buffer.seek(0)
    plt.close()
    
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return img_str

def apply(df: pd.DataFrame) -> List[dict]:
    """Collapse dataset to group-level averages."""
    outputs = []
    
    # Find columns
    all_cols = df.columns.tolist()
    
    if len(all_cols) < 2:
        return [{"type": "text", "title": "Collapse", "data": "Need at least 2 variables for collapsing."}]
    
    # For demonstration, create grouping variables
    # In practice, you'd want user input for group and variable selection
    
    # Find numeric columns for aggregation
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        return [{"type": "text", "title": "Collapse", "data": "Need at least 2 numeric variables for collapsing."}]
    
    # Create grouping variables (e.g., based on first numeric column)
    df_copy = df.copy()
    
    # Create discrete groups for first numeric column
    if len(numeric_cols) >= 2:
        group_col = numeric_cols[0]
        df_copy['group_var'] = pd.qcut(df_copy[group_col], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop')
        
        # Create second grouping variable if possible
        if len(numeric_cols) >= 3:
            group_col2 = numeric_cols[1]
            df_copy['group_var2'] = pd.qcut(df_copy[group_col2], q=3, labels=['Low', 'Medium', 'High'], duplicates='drop')
            group_cols = ['group_var', 'group_var2']
        else:
            group_cols = ['group_var']
    else:
        return [{"type": "text", "title": "Collapse", "data": "Insufficient variables for grouping."}]
    
    # Remove rows with NaN groups
    df_copy = df_copy.dropna(subset=group_cols)
    
    if len(df_copy) == 0:
        return [{"type": "text", "title": "Collapse", "data": "Insufficient data for collapsing."}]
    
    # Select variables to aggregate (exclude grouping variables)
    if 'group_var2' in locals():
        agg_cols = [col for col in numeric_cols if col not in [group_col, group_col2]]
    else:
        agg_cols = [col for col in numeric_cols if col != group_col]
    agg_cols = agg_cols[:3]  # Limit to 3 variables for performance
    
    # Define aggregation functions
    agg_functions = ['mean', 'std', 'count']
    
    # Perform collapsing
    try:
        collapsed_df = _collapse_by_group(df_copy, group_cols, agg_cols, agg_functions)
        
        if collapsed_df.empty:
            return [{"type": "text", "title": "Collapse", "data": "Error in collapsing operation."}]
        
        # 1. Collapse Summary
        collapse_summary = _create_collapse_summary(df_copy, collapsed_df, group_cols, agg_cols)
        outputs.append({
            "type": "table", 
            "title": "Collapse Summary", 
            "data": collapse_summary
        })
        
        # 2. Collapsed Dataset Preview
        preview_df = collapsed_df.head(20).copy()
        # Round numeric columns
        for col in preview_df.columns:
            if preview_df[col].dtype in ['float64', 'float32']:
                preview_df[col] = preview_df[col].round(4)
        
        outputs.append({
            "type": "table", 
            "title": "Collapsed Dataset Preview (First 20 Rows)", 
            "data": preview_df
        })
        
        # 3. Group Size Analysis
        group_size_analysis = _create_group_size_analysis(df_copy, group_cols)
        outputs.append({
            "type": "table", 
            "title": "Group Size Analysis", 
            "data": group_size_analysis
        })
        
        # 4. Group Size Distribution Plot
        group_size_plot = _create_group_size_histogram(df_copy, group_cols)
        if group_size_plot:
            outputs.append({
                "type": "image", 
                "title": "Group Size Distribution", 
                "data": group_size_plot
            })
        
        # 5. Aggregation Comparison
        comparison_tables = _create_aggregation_comparison(df_copy, collapsed_df, group_cols, agg_cols)
        
        for i, comp_table in enumerate(comparison_tables):
            if i < len(agg_cols):
                outputs.append({
                    "type": "table", 
                    "title": f"Original vs Collapsed: {agg_cols[i]}", 
                    "data": comp_table
                })
        
        # 6. Group Heatmap (if 2+ grouping variables)
        if len(group_cols) >= 2:
            heatmap_img = _create_group_heatmap(collapsed_df, group_cols, agg_cols)
            if heatmap_img:
                outputs.append({
                    "type": "image", 
                    "title": "Group Heatmap", 
                    "data": heatmap_img
                })
        
        # 7. Collapsed Dataset Statistics
        collapsed_stats = collapsed_df.describe()
        collapsed_stats = collapsed_stats.round(4)
        
        outputs.append({
            "type": "table", 
            "title": "Collapsed Dataset Statistics", 
            "data": collapsed_stats
        })
        
        # 8. Interpretation Guide
        interpretation = f"""
**Dataset Collapse Results:**

**What Happened:**
- **Original Dataset**: {len(df_copy)} observations
- **Collapsed Dataset**: {len(collapsed_df)} observations
- **Reduction**: {len(df_copy) / len(collapsed_df):.1f}x smaller

**Grouping Structure:**
- **Primary Group**: {group_cols[0]} (4 quartiles)
- **Secondary Group**: {group_cols[1] if len(group_cols) > 1 else 'None'} (3 levels)
- **Total Groups**: {len(collapsed_df)}

**Aggregated Variables:**
- **Variables**: {', '.join(agg_cols)}
- **Functions**: {', '.join(agg_functions)}
- **Output**: Each group now has one value per variable per function

**Key Benefits:**

**1. Data Reduction:**
- Smaller dataset for analysis
- Faster computation
- Easier visualization

**2. Group-Level Analysis:**
- Compare groups directly
- Identify patterns across categories
- Reduce noise in individual observations

**3. Summary Statistics:**
- Group means, standard deviations, counts
- Between-group variation
- Overall patterns

**Use Cases:**
- **Panel Data**: Average across time periods
- **Survey Data**: Aggregate by demographic groups
- **Geographic Data**: Summarize by regions
- **Experimental Data**: Group by treatment conditions

**Interpretation:**
- **Mean**: Average value within each group
- **Std**: Variation within each group
- **Count**: Number of observations per group
- **Group Differences**: Compare means across groups

**Next Steps:**
- Use collapsed data for group comparisons
- Create visualizations by group
- Perform statistical tests between groups
- Export collapsed dataset for further analysis
        """
        
        outputs.append({
            "type": "text", 
            "title": "Collapse Analysis Interpretation", 
            "data": interpretation
        })
        
    except Exception as e:
        outputs.append({
            "type": "text", 
            "title": "Collapse Error", 
            "data": f"Error performing collapse operation: {str(e)}"
        })
    
    return outputs
