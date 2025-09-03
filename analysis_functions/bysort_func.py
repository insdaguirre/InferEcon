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

display_name = "Bysort"

def _create_grouped_summary(df: pd.DataFrame, group_col: str, var_col: str) -> pd.DataFrame:
    """Create grouped summary statistics for a variable."""
    # Remove missing values
    mask = ~(df[group_col].isna() | df[var_col].isna())
    df_clean = df[mask]
    
    if len(df_clean) == 0:
        return pd.DataFrame()
    
    # Group by the grouping variable
    grouped = df_clean.groupby(group_col)[var_col]
    
    # Calculate summary statistics
    summary_stats = grouped.agg([
        'count', 'mean', 'std', 'min', 'max', 'median'
    ]).round(4)
    
    # Add additional statistics
    summary_stats['q25'] = grouped.quantile(0.25).round(4)
    summary_stats['q75'] = grouped.quantile(0.75).round(4)
    summary_stats['iqr'] = (summary_stats['q75'] - summary_stats['q25']).round(4)
    summary_stats['cv'] = (summary_stats['std'] / summary_stats['mean'] * 100).round(2)
    
    # Rename columns for clarity
    summary_stats.columns = [
        'N', 'Mean', 'Std Dev', 'Min', 'Max', 'Median', 
        'Q25', 'Q75', 'IQR', 'CV (%)'
    ]
    
    # Add group sizes
    group_sizes = df_clean[group_col].value_counts().sort_index()
    summary_stats['Group Size'] = group_sizes
    
    # Reorder columns
    col_order = ['Group Size', 'N', 'Mean', 'Std Dev', 'Min', 'Max', 'Median', 'Q25', 'Q75', 'IQR', 'CV (%)']
    summary_stats = summary_stats[col_order]
    
    return summary_stats

def _create_grouped_boxplot(df: pd.DataFrame, group_col: str, var_col: str) -> str:
    """Create grouped box plot."""
    # Remove missing values
    mask = ~(df[group_col].isna() | df[var_col].isna())
    df_clean = df[mask]
    
    if len(df_clean) == 0:
        return ""
    
    plt.figure(figsize=(12, 8))
    
    # Create box plot
    sns.boxplot(data=df_clean, x=group_col, y=var_col, palette='Set3')
    
    # Customize plot
    plt.xlabel(group_col, fontsize=14, fontweight='bold')
    plt.ylabel(var_col, fontsize=14, fontweight='bold')
    plt.title(f'Distribution of {var_col} by {group_col}', fontsize=16, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add statistics text
    group_stats = df_clean.groupby(group_col)[var_col].agg(['count', 'mean', 'std'])
    stats_text = f'Total Groups: {len(group_stats)}\nTotal Observations: {group_stats["count"].sum()}'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, fontsize=11,
             verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", 
             facecolor="lightblue", alpha=0.8))
    
    plt.tight_layout()
    
    # Convert to base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
    buffer.seek(0)
    plt.close()
    
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return img_str

def _create_grouped_histogram(df: pd.DataFrame, group_col: str, var_col: str) -> str:
    """Create grouped histogram."""
    # Remove missing values
    mask = ~(df[group_col].isna() | df[var_col].isna())
    df_clean = df[mask]
    
    if len(df_clean) == 0:
        return ""
    
    # Get unique groups
    groups = df_clean[group_col].unique()
    n_groups = len(groups)
    
    # Determine subplot layout
    cols = min(3, n_groups)
    rows = (n_groups + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    if n_groups == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.reshape(1, -1)
    else:
        axes = axes.reshape(rows, cols)
    
    for i, group in enumerate(groups):
        row = i // cols
        col_idx = i % cols
        ax = axes[row, col_idx]
        
        # Get data for this group
        group_data = df_clean[df_clean[group_col] == group][var_col]
        
        # Create histogram
        ax.hist(group_data, bins=min(20, len(group_data.unique())), alpha=0.7, 
                edgecolor='black', linewidth=0.5)
        ax.set_xlabel(var_col, fontweight='bold')
        ax.set_ylabel('Frequency', fontweight='bold')
        ax.set_title(f'{group_col} = {group}', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        stats_text = f'n = {len(group_data)}\nMean = {group_data.mean():.3f}\nStd = {group_data.std():.3f}'
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
    
    # Hide empty subplots
    for i in range(n_groups, rows * cols):
        row = i // cols
        col_idx = i % cols
        axes[row, col_idx].set_visible(False)
    
    plt.suptitle(f'Distribution of {var_col} by {group_col}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Convert to base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
    buffer.seek(0)
    plt.close()
    
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return img_str

def _create_grouped_barplot(df: pd.DataFrame, group_col: str, var_col: str) -> str:
    """Create grouped bar plot showing means by group."""
    # Remove missing values
    mask = ~(df[group_col].isna() | df[var_col].isna())
    df_clean = df[mask]
    
    if len(df_clean) == 0:
        return ""
    
    # Calculate group means and standard errors
    group_stats = df_clean.groupby(group_col)[var_col].agg(['count', 'mean', 'std']).reset_index()
    group_stats['se'] = group_stats['std'] / np.sqrt(group_stats['count'])
    
    plt.figure(figsize=(12, 8))
    
    # Create bar plot
    bars = plt.bar(range(len(group_stats)), group_stats['mean'], 
                   yerr=group_stats['se'], capsize=5, alpha=0.7, 
                   edgecolor='black', linewidth=0.5)
    
    # Customize plot
    plt.xlabel(group_col, fontsize=14, fontweight='bold')
    plt.ylabel(f'Mean {var_col}', fontsize=14, fontweight='bold')
    plt.title(f'Mean {var_col} by {group_col}', fontsize=16, fontweight='bold')
    plt.xticks(range(len(group_stats)), group_stats[group_col], rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + group_stats.iloc[i]['se'] + height*0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Add statistics text
    total_n = group_stats['count'].sum()
    overall_mean = df_clean[var_col].mean()
    stats_text = f'Total N = {total_n}\nOverall Mean = {overall_mean:.3f}'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, fontsize=11,
             verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", 
             facecolor="lightgreen", alpha=0.8))
    
    plt.tight_layout()
    
    # Convert to base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
    buffer.seek(0)
    plt.close()
    
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return img_str

def _perform_group_comparison_tests(df: pd.DataFrame, group_col: str, var_col: str) -> dict:
    """Perform statistical tests comparing groups."""
    # Remove missing values
    mask = ~(df[group_col].isna() | df[var_col].isna())
    df_clean = df[mask]
    
    if len(df_clean) == 0:
        return {}
    
    groups = df_clean[group_col].unique()
    
    if len(groups) < 2:
        return {}
    
    # One-way ANOVA
    group_data = [df_clean[df_clean[group_col] == group][var_col].values for group in groups]
    f_stat, anova_p = stats.f_oneway(*group_data)
    
    # Kruskal-Wallis test (non-parametric alternative)
    h_stat, kw_p = stats.kruskal(*group_data)
    
    # Effect size (eta-squared for ANOVA)
    ss_between = 0
    ss_total = 0
    grand_mean = df_clean[var_col].mean()
    
    for group in groups:
        group_values = df_clean[df_clean[group_col] == group][var_col]
        group_mean = group_values.mean()
        group_n = len(group_values)
        
        ss_between += group_n * (group_mean - grand_mean) ** 2
        ss_total += np.sum((group_values - grand_mean) ** 2)
    
    eta_squared = ss_between / ss_total if ss_total > 0 else 0
    
    # Pairwise comparisons (if only 2 groups)
    pairwise_results = {}
    if len(groups) == 2:
        group1_data = df_clean[df_clean[group_col] == groups[0]][var_col]
        group2_data = df_clean[df_clean[group_col] == groups[1]][var_col]
        
        # t-test
        t_stat, t_p = stats.ttest_ind(group1_data, group2_data)
        
        # Mann-Whitney U test
        u_stat, u_p = stats.mannwhitneyu(group1_data, group2_data, alternative='two-sided')
        
        # Cohen's d
        pooled_std = np.sqrt(((len(group1_data) - 1) * group1_data.var() + 
                             (len(group2_data) - 1) * group2_data.var()) / 
                            (len(group1_data) + len(group2_data) - 2))
        cohens_d = (group1_data.mean() - group2_data.mean()) / pooled_std if pooled_std > 0 else 0
        
        pairwise_results = {
            't_statistic': t_stat,
            't_pvalue': t_p,
            'u_statistic': u_stat,
            'u_pvalue': u_p,
            'cohens_d': cohens_d
        }
    
    return {
        'anova_f': f_stat,
        'anova_p': anova_p,
        'kw_h': h_stat,
        'kw_p': kw_p,
        'eta_squared': eta_squared,
        'pairwise': pairwise_results
    }

def apply(df: pd.DataFrame, config: dict = None) -> List[dict]:
    """Perform grouped summary analysis."""
    outputs = []
    
    # Find columns
    all_cols = df.columns.tolist()
    
    if len(all_cols) < 2:
        return [{"type": "text", "title": "Bysort", "data": "Need at least 2 variables for grouped analysis."}]
    
    # Use configuration if provided, otherwise use defaults
    if config and 'group_col' in config and 'var_col' in config:
        group_col = config['group_col']
        var_col = config['var_col']
        grouping_method = config.get('grouping_method', 'Quartiles')
        n_bins = config.get('n_bins', 4)
        
        # Create groups based on configuration
        df_copy = df.copy()
        
        if grouping_method == "Categories":
            # Use categorical variable as-is
            df_copy['group'] = df_copy[group_col].astype(str)
        else:
            # Create numeric groups
            if grouping_method == "Custom bins":
                df_copy['group'] = pd.cut(df_copy[group_col], bins=n_bins, labels=[f'Bin_{i+1}' for i in range(n_bins)], duplicates='drop')
            else:
                # Quartiles, Quintiles, Deciles
                df_copy['group'] = pd.qcut(df_copy[group_col], q=n_bins, labels=[f'Q{i+1}' for i in range(n_bins)], duplicates='drop')
        
        # Remove rows with NaN groups
        df_copy = df_copy.dropna(subset=['group'])
        
        if len(df_copy) < 10:
            return [{"type": "text", "title": "Bysort", "data": "Insufficient data for grouped analysis after grouping."}]
        
        # Perform grouped analysis
        try:
            # 1. Grouped Summary Statistics
            grouped_stats = _create_grouped_summary_table(df_copy, 'group', var_col)
            outputs.append({
                "type": "table", 
                "title": f"Grouped Summary: {var_col} by {group_col}", 
                "data": grouped_stats
            })
            
            # 2. Grouped Box Plot
            boxplot_img = _create_grouped_boxplot(df_copy, 'group', var_col)
            outputs.append({
                "type": "image", 
                "title": f"Box Plot: {var_col} by {group_col}", 
                "data": boxplot_img
            })
            
            # 3. Grouped Histogram
            hist_img = _create_grouped_histogram(df_copy, 'group', var_col)
            outputs.append({
                "type": "image", 
                "title": f"Histogram: {var_col} by {group_col}", 
                "data": hist_img
            })
            
            # 4. Group Comparison Tests
            comparison_tests = _perform_group_comparison_tests(df_copy, 'group', var_col)
            outputs.append({
                "type": "table", 
                "title": f"Group Comparison Tests: {var_col}", 
                "data": comparison_tests
            })
            
        except Exception as e:
            outputs.append({
                "type": "text", 
                "title": "Bysort Error", 
                "data": f"Error performing grouped analysis: {str(e)}"
            })
    
    else:
        # Fallback to default behavior
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            return [{"type": "text", "title": "Bysort", "data": "Need at least 2 numeric variables for grouped analysis."}]
        
        # Use first numeric column as grouping variable, second as analysis variable
        group_col = numeric_cols[0]
        var_col = numeric_cols[1]
    
    # Create discrete groups (e.g., quartiles)
    df_copy = df.copy()
    df_copy['group_quartile'] = pd.qcut(df_copy[group_col], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop')
    
    # Remove rows with NaN groups
    df_copy = df_copy.dropna(subset=['group_quartile'])
    
    if len(df_copy) == 0:
        return [{"type": "text", "title": "Bysort", "data": "Insufficient data for grouped analysis."}]
    
    # 1. Grouped Summary Statistics
    grouped_summary = _create_grouped_summary(df_copy, 'group_quartile', var_col)
    
    if not grouped_summary.empty:
        outputs.append({
            "type": "table", 
            "title": f"Grouped Summary: {var_col} by {group_col} Quartiles", 
            "data": grouped_summary
        })
    
    # 2. Grouped Box Plot
    boxplot_img = _create_grouped_boxplot(df_copy, 'group_quartile', var_col)
    if boxplot_img:
        outputs.append({
            "type": "image", 
            "title": f"Grouped Box Plot: {var_col} by {group_col} Quartiles", 
            "data": boxplot_img
        })
    
    # 3. Grouped Histograms
    hist_img = _create_grouped_histogram(df_copy, 'group_quartile', var_col)
    if hist_img:
        outputs.append({
            "type": "image", 
            "title": f"Grouped Histograms: {var_col} by {group_col} Quartiles", 
            "data": hist_img
        })
    
    # 4. Grouped Bar Plot
    barplot_img = _create_grouped_barplot(df_copy, 'group_quartile', var_col)
    if barplot_img:
        outputs.append({
            "type": "image", 
            "title": f"Grouped Bar Plot: {var_col} by {group_col} Quartiles", 
            "data": barplot_img
        })
    
    # 5. Statistical Tests
    test_results = _perform_group_comparison_tests(df_copy, 'group_quartile', var_col)
    
    if test_results:
        # ANOVA and Kruskal-Wallis results
        test_table = pd.DataFrame({
            'Test': ['One-way ANOVA', 'Kruskal-Wallis H'],
            'Statistic': [test_results['anova_f'], test_results['kw_h']],
            'P-value': [test_results['anova_p'], test_results['kw_p']],
            'Effect Size': [f"η² = {test_results['eta_squared']:.4f}", 'N/A']
        })
        
        test_table['Statistic'] = test_table['Statistic'].round(4)
        test_table['P-value'] = test_table['P-value'].round(4)
        
        outputs.append({
            "type": "table", 
            "title": "Group Comparison Tests", 
            "data": test_table
        })
        
        # Pairwise comparisons if applicable
        if test_results['pairwise']:
            pairwise = test_results['pairwise']
            pairwise_table = pd.DataFrame({
                'Test': ['Independent t-test', 'Mann-Whitney U', "Cohen's d"],
                'Statistic': [pairwise['t_statistic'], pairwise['u_statistic'], pairwise['cohens_d']],
                'P-value': [pairwise['t_pvalue'], pairwise['u_pvalue'], 'N/A']
            })
            
            pairwise_table['Statistic'] = pairwise_table['Statistic'].round(4)
            pairwise_table['P-value'] = pairwise_table['P-value'].round(4)
            
            outputs.append({
                "type": "table", 
                "title": "Pairwise Group Comparisons", 
                "data": pairwise_table
            })
    
    # 6. Group Summary by Original Variable
    # Create additional grouping based on the original variable
    df_copy['var_group'] = pd.qcut(df_copy[var_col], q=3, labels=['Low', 'Medium', 'High'], duplicates='drop')
    
    # Remove rows with NaN groups
    df_copy = df_copy.dropna(subset=['var_group'])
    
    if len(df_copy) > 0:
        reverse_summary = _create_grouped_summary(df_copy, 'var_group', group_col)
        
        if not reverse_summary.empty:
            outputs.append({
                "type": "table", 
                "title": f"Reverse Grouping: {group_col} by {var_col} Groups", 
                "data": reverse_summary
            })
    
    # 7. Interpretation Guide
    interpretation = f"""
**Grouped Summary Analysis Results:**

**Analysis Setup:**
- **Grouping Variable**: {group_col} (divided into quartiles)
- **Analysis Variable**: {var_col}
- **Grouping Method**: Quartiles (4 equal-sized groups)

**What the Analysis Shows:**

**1. Grouped Summary Statistics:**
- **N**: Number of observations in each group
- **Mean**: Average value in each group
- **Std Dev**: Standard deviation within each group
- **Min/Max**: Range of values in each group
- **Median**: Middle value in each group
- **Q25/Q75**: 25th and 75th percentiles
- **IQR**: Interquartile range (Q75 - Q25)
- **CV (%)**: Coefficient of variation (Std/Mean × 100)

**2. Visualizations:**
- **Box Plot**: Shows distribution shape and outliers by group
- **Histograms**: Individual distribution plots for each group
- **Bar Plot**: Mean values with standard error bars

**3. Statistical Tests:**
- **One-way ANOVA**: Tests if group means are significantly different
- **Kruskal-Wallis**: Non-parametric alternative to ANOVA
- **Effect Size**: η² (eta-squared) measures strength of group differences

**4. Pairwise Comparisons:**
- **t-test**: Compares two groups (if applicable)
- **Mann-Whitney U**: Non-parametric group comparison
- **Cohen's d**: Standardized difference between groups

**Interpretation Guidelines:**
- **P < 0.05**: Groups are significantly different
- **η² > 0.06**: Small effect, **η² > 0.14**: Medium effect, **η² > 0.26**: Large effect
- **|d| > 0.2**: Small difference, **|d| > 0.5**: Medium difference, **|d| > 0.8**: Large difference

**Use Cases:**
- Compare performance across categories
- Analyze regional differences
- Study time period effects
- Investigate demographic patterns
        """
    
    outputs.append({
        "type": "text", 
        "title": "Grouped Analysis Interpretation", 
        "data": interpretation
    })
    
    return outputs

def apply_with_config(df: pd.DataFrame, config: dict) -> List[dict]:
    """Apply grouped analysis with configuration parameters."""
    return apply(df, config)
