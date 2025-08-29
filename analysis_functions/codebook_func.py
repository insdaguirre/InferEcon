from __future__ import annotations

from typing import List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import io
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

display_name = "Codebook"

def _create_variable_summary(df: pd.DataFrame, col: str) -> dict:
    """Create comprehensive summary for a single variable."""
    series = df[col]
    dtype = series.dtype
    
    # Basic info
    n_total = len(series)
    n_missing = series.isna().sum()
    n_valid = n_total - n_missing
    missing_pct = (n_missing / n_total) * 100
    
    # Type-specific analysis
    if np.issubdtype(dtype, np.number):
        # Numeric variable
        valid_series = series.dropna()
        
        if len(valid_series) > 0:
            summary = {
                'type': 'Numeric',
                'n_total': n_total,
                'n_missing': n_missing,
                'n_valid': n_valid,
                'missing_pct': missing_pct,
                'min': valid_series.min(),
                'max': valid_series.max(),
                'mean': valid_series.mean(),
                'median': valid_series.median(),
                'std': valid_series.std(),
                'unique_values': valid_series.nunique(),
                'mode': valid_series.mode().iloc[0] if len(valid_series.mode()) > 0 else np.nan,
                'skewness': valid_series.skew(),
                'kurtosis': valid_series.kurtosis(),
                'q25': valid_series.quantile(0.25),
                'q75': valid_series.quantile(0.75),
                'iqr': valid_series.quantile(0.75) - valid_series.quantile(0.25)
            }
        else:
            summary = {
                'type': 'Numeric',
                'n_total': n_total,
                'n_missing': n_missing,
                'n_valid': n_valid,
                'missing_pct': missing_pct,
                'min': np.nan, 'max': np.nan, 'mean': np.nan, 'median': np.nan,
                'std': np.nan, 'unique_values': 0, 'mode': np.nan,
                'skewness': np.nan, 'kurtosis': np.nan, 'q25': np.nan, 'q75': np.nan, 'iqr': np.nan
            }
    else:
        # Categorical/string variable
        valid_series = series.dropna()
        
        if len(valid_series) > 0:
            value_counts = valid_series.value_counts()
            unique_values = valid_series.nunique()
            
            summary = {
                'type': 'Categorical',
                'n_total': n_total,
                'n_missing': n_missing,
                'n_valid': n_valid,
                'missing_pct': missing_pct,
                'unique_values': unique_values,
                'most_common': value_counts.index[0] if len(value_counts) > 0 else np.nan,
                'most_common_count': value_counts.iloc[0] if len(value_counts) > 0 else 0,
                'most_common_pct': (value_counts.iloc[0] / n_valid * 100) if len(value_counts) > 0 else 0,
                'least_common': value_counts.index[-1] if len(value_counts) > 0 else np.nan,
                'least_common_count': value_counts.iloc[-1] if len(value_counts) > 0 else 0,
                'least_common_pct': (value_counts.iloc[-1] / n_valid * 100) if len(value_counts) > 0 else 0,
                'max_length': valid_series.astype(str).str.len().max() if valid_series.dtype == 'object' else np.nan,
                'min_length': valid_series.astype(str).str.len().min() if valid_series.dtype == 'object' else np.nan
            }
        else:
            summary = {
                'type': 'Categorical',
                'n_total': n_total,
                'n_missing': n_missing,
                'n_valid': n_valid,
                'missing_pct': missing_pct,
                'unique_values': 0, 'most_common': np.nan, 'most_common_count': 0,
                'most_common_pct': 0, 'least_common': np.nan, 'least_common_count': 0,
                'least_common_pct': 0, 'max_length': np.nan, 'min_length': np.nan
            }
    
    return summary

def _create_value_distribution_plot(df: pd.DataFrame, col: str) -> str:
    """Create distribution plot for a variable."""
    series = df[col].dropna()
    
    if len(series) == 0:
        return ""
    
    plt.figure(figsize=(12, 8))
    
    if np.issubdtype(series.dtype, np.number):
        # Numeric variable - histogram and box plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Histogram
        ax1.hist(series, bins=min(30, len(series.unique())), alpha=0.7, edgecolor='black', linewidth=0.5)
        ax1.set_xlabel(col, fontweight='bold')
        ax1.set_ylabel('Frequency', fontweight='bold')
        ax1.set_title(f'Distribution of {col}', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Add statistics
        stats_text = f'Mean: {series.mean():.3f}\nStd: {series.std():.3f}\nN: {len(series)}'
        ax1.text(0.95, 0.95, stats_text, transform=ax1.transAxes, fontsize=10,
                 verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
        
        # Box plot
        ax2.boxplot(series, vert=False)
        ax2.set_xlabel(col, fontweight='bold')
        ax2.set_title(f'Box Plot of {col}', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Add outlier info
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        outliers = series[(series < Q1 - 1.5*IQR) | (series > Q3 + 1.5*IQR)]
        
        outlier_text = f'Outliers: {len(outliers)}\nQ1: {Q1:.3f}\nQ3: {Q3:.3f}\nIQR: {IQR:.3f}'
        ax2.text(0.95, 0.95, outlier_text, transform=ax2.transAxes, fontsize=10,
                 verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
        
    else:
        # Categorical variable - bar plot
        value_counts = series.value_counts()
        
        # Limit to top 20 categories for readability
        if len(value_counts) > 20:
            top_values = value_counts.head(20)
            other_count = value_counts.iloc[20:].sum()
            if other_count > 0:
                top_values['Other'] = other_count
            value_counts = top_values
        
        plt.figure(figsize=(max(12, len(value_counts) * 0.8), 8))
        bars = plt.bar(range(len(value_counts)), value_counts.values, alpha=0.7, edgecolor='black', linewidth=0.5)
        
        plt.xlabel('Categories', fontweight='bold')
        plt.ylabel('Count', fontweight='bold')
        plt.title(f'Distribution of {col}', fontweight='bold')
        plt.xticks(range(len(value_counts)), value_counts.index, rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{height}', ha='center', va='bottom', fontsize=9)
        
        # Add statistics
        stats_text = f'Total: {len(series)}\nUnique: {series.nunique()}\nMost Common: {value_counts.index[0]}'
        plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=10,
                 verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))
    
    plt.tight_layout()
    
    # Convert to base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
    buffer.seek(0)
    plt.close()
    
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return img_str

def _create_unique_values_table(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Create table showing unique values and their frequencies."""
    series = df[col].dropna()
    
    if len(series) == 0:
        return pd.DataFrame()
    
    value_counts = series.value_counts()
    
    # Limit to reasonable number of rows
    if len(value_counts) > 50:
        top_values = value_counts.head(49)
        other_count = value_counts.iloc[49:].sum()
        if other_count > 0:
            top_values['Other'] = other_count
        value_counts = top_values
    
    # Create table
    unique_table = pd.DataFrame({
        'Value': value_counts.index,
        'Count': value_counts.values,
        'Percentage': (value_counts.values / len(series) * 100).round(2),
        'Cumulative Count': value_counts.values.cumsum(),
        'Cumulative Percentage': (value_counts.values.cumsum() / len(series) * 100).round(2)
    })
    
    return unique_table

def apply(df: pd.DataFrame) -> List[dict]:
    """Create comprehensive codebook for variables."""
    outputs = []
    
    # Get all columns
    all_cols = df.columns.tolist()
    
    if len(all_cols) == 0:
        return [{"type": "text", "title": "Codebook", "data": "No variables found in dataset."}]
    
    # Limit to first 10 variables for performance
    cols_to_analyze = all_cols[:10]
    
    # 1. Overall dataset summary
    dataset_summary = pd.DataFrame({
        'Statistic': ['Total Variables', 'Total Observations', 'Memory Usage (MB)', 'Data Types'],
        'Value': [
            len(all_cols),
            len(df),
            round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
            ', '.join([str(dtype).split('.')[-1] for dtype in df.dtypes.value_counts().to_dict().keys()])
        ]
    })
    
    outputs.append({
        "type": "table", 
        "title": "Dataset Overview", 
        "data": dataset_summary
    })
    
    # 2. Variable-by-variable codebook
    for col in cols_to_analyze:
        # Variable summary
        summary = _create_variable_summary(df, col)
        
        # Create summary table
        if summary['type'] == 'Numeric':
            summary_data = [
                {'Statistic': 'Variable Type', 'Value': summary['type']},
                {'Statistic': 'Total Observations', 'Value': summary['n_total']},
                {'Statistic': 'Missing Values', 'Value': f"{summary['n_missing']} ({summary['missing_pct']:.1f}%)"},
                {'Statistic': 'Valid Observations', 'Value': summary['n_valid']},
                {'Statistic': 'Unique Values', 'Value': summary['unique_values']},
                {'Statistic': 'Minimum', 'Value': f"{summary['min']:.4f}" if not np.isnan(summary['min']) else 'N/A'},
                {'Statistic': 'Maximum', 'Value': f"{summary['max']:.4f}" if not np.isnan(summary['max']) else 'N/A'},
                {'Statistic': 'Mean', 'Value': f"{summary['mean']:.4f}" if not np.isnan(summary['mean']) else 'N/A'},
                {'Statistic': 'Median', 'Value': f"{summary['median']:.4f}" if not np.isnan(summary['median']) else 'N/A'},
                {'Statistic': 'Standard Deviation', 'Value': f"{summary['std']:.4f}" if not np.isnan(summary['std']) else 'N/A'},
                {'Statistic': 'Skewness', 'Value': f"{summary['skewness']:.4f}" if not np.isnan(summary['skewness']) else 'N/A'},
                {'Statistic': 'Kurtosis', 'Value': f"{summary['kurtosis']:.4f}" if not np.isnan(summary['kurtosis']) else 'N/A'},
                {'Statistic': '25th Percentile', 'Value': f"{summary['q25']:.4f}" if not np.isnan(summary['q25']) else 'N/A'},
                {'Statistic': '75th Percentile', 'Value': f"{summary['q75']:.4f}" if not np.isnan(summary['q75']) else 'N/A'},
                {'Statistic': 'Interquartile Range', 'Value': f"{summary['iqr']:.4f}" if not np.isnan(summary['iqr']) else 'N/A'}
            ]
        else:
            summary_data = [
                {'Statistic': 'Variable Type', 'Value': summary['type']},
                {'Statistic': 'Total Observations', 'Value': summary['n_total']},
                {'Statistic': 'Missing Values', 'Value': f"{summary['n_missing']} ({summary['missing_pct']:.1f}%)"},
                {'Statistic': 'Valid Observations', 'Value': summary['n_valid']},
                {'Statistic': 'Unique Values', 'Value': summary['unique_values']},
                {'Statistic': 'Most Common Value', 'Value': str(summary['most_common'])},
                {'Statistic': 'Most Common Count', 'Value': summary['most_common_count']},
                {'Statistic': 'Most Common %', 'Value': f"{summary['most_common_pct']:.1f}%"},
                {'Statistic': 'Least Common Value', 'Value': str(summary['least_common'])},
                {'Statistic': 'Least Common Count', 'Value': summary['least_common_count']},
                {'Statistic': 'Least Common %', 'Value': f"{summary['least_common_pct']:.1f}%"}
            ]
            
            if not np.isnan(summary['max_length']):
                summary_data.extend([
                    {'Statistic': 'Max String Length', 'Value': summary['max_length']},
                    {'Statistic': 'Min String Length', 'Value': summary['min_length']}
                ])
        
        summary_df = pd.DataFrame(summary_data)
        outputs.append({
            "type": "table", 
            "title": f"Codebook: {col}", 
            "data": summary_df
        })
        
        # Distribution plot
        plot_img = _create_value_distribution_plot(df, col)
        if plot_img:
            outputs.append({
                "type": "image", 
                "title": f"Distribution: {col}", 
                "data": plot_img
            })
        
        # Unique values table (for categorical or if few unique numeric values)
        if summary['unique_values'] <= 50:
            unique_table = _create_unique_values_table(df, col)
            if not unique_table.empty:
                outputs.append({
                    "type": "table", 
                    "title": f"Unique Values: {col}", 
                    "data": unique_table
                })
    
    # 3. Data quality summary
    quality_data = []
    for col in cols_to_analyze:
        missing_pct = (df[col].isna().sum() / len(df)) * 100
        # Handle different dtype formats safely
        dtype_str = str(df[col].dtype)
        if 'int' in dtype_str:
            type_name = 'Integer'
        elif 'float' in dtype_str:
            type_name = 'Float'
        elif 'object' in dtype_str or 'string' in dtype_str:
            type_name = 'String'
        elif 'datetime' in dtype_str:
            type_name = 'DateTime'
        else:
            type_name = dtype_str.split('.')[-1] if '.' in dtype_str else dtype_str
        
        quality_data.append({
            'Variable': col,
            'Type': type_name,
            'Missing %': round(missing_pct, 1),
            'Quality': 'Good' if missing_pct < 5 else 'Fair' if missing_pct < 20 else 'Poor'
        })
    
    quality_df = pd.DataFrame(quality_data)
    outputs.append({
        "type": "table", 
        "title": "Data Quality Summary", 
        "data": quality_df
    })
    
    # 4. Interpretation guide
    interpretation = f"""
**Codebook Analysis Results:**

**Dataset Overview:**
- **Total Variables**: {len(all_cols)} columns
- **Total Observations**: {len(df)} rows
- **Memory Usage**: {round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2)} MB

**What the Codebook Shows:**

**1. Variable Types:**
- **Numeric**: Continuous or discrete numerical variables
- **Categorical**: String, object, or categorical variables

**2. Data Quality:**
- **Missing Values**: Percentage of missing data
- **Unique Values**: Count of distinct values
- **Data Distribution**: Shape and spread of data

**3. Key Statistics:**
- **Central Tendency**: Mean, median, mode
- **Variability**: Standard deviation, range, IQR
- **Shape**: Skewness, kurtosis
- **Outliers**: Extreme values beyond 1.5*IQR

**4. Categorical Analysis:**
- **Frequency Tables**: Count and percentage of each category
- **Most/Least Common**: Dominant and rare categories
- **String Analysis**: Length patterns for text variables

**Data Quality Assessment:**
- **Good**: < 5% missing data
- **Fair**: 5-20% missing data  
- **Poor**: > 20% missing data

**Recommendations:**
- Check for data entry errors in high-missing variables
- Consider imputation for moderate missing data
- Investigate outliers in numeric variables
- Validate categorical coding schemes
        """
    
    outputs.append({
        "type": "text", 
        "title": "Codebook Interpretation Guide", 
        "data": interpretation
    })
    
    return outputs
