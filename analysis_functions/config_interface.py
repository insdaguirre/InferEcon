"""
Configuration interface for analysis functions that require user input.
This module provides a standardized way to collect user configuration
for functions that need variable selection or parameter specification.
"""

from typing import Dict, List, Any, Optional, Tuple
import streamlit as st
import pandas as pd
import numpy as np

class FunctionConfig:
    """Base class for function configuration."""
    
    def __init__(self, function_name: str, df: pd.DataFrame):
        self.function_name = function_name
        self.df = df
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        self.all_cols = df.columns.tolist()
    
    def get_config(self) -> Dict[str, Any]:
        """Override this method to define configuration UI."""
        raise NotImplementedError("Subclasses must implement get_config()")

class RegressionConfig(FunctionConfig):
    """Configuration for regression-based functions."""
    
    def get_config(self) -> Dict[str, Any]:
        st.subheader(f"Configure {self.function_name}")
        
        if len(self.numeric_cols) < 2:
            st.error("Need at least 2 numeric variables for regression analysis.")
            return {}
        
        # Dependent variable selection
        y_col = st.selectbox(
            "Select dependent variable (Y):",
            self.numeric_cols,
            key=f"{self.function_name}_y"
        )
        
        # Independent variables selection
        remaining_cols = [col for col in self.numeric_cols if col != y_col]
        x_cols = st.multiselect(
            "Select independent variables (X):",
            remaining_cols,
            default=remaining_cols[:min(5, len(remaining_cols))],
            key=f"{self.function_name}_x"
        )
        
        if not x_cols:
            st.warning("Please select at least one independent variable.")
            return {}
        
        return {
            'y_col': y_col,
            'x_cols': x_cols
        }

class PanelDataConfig(FunctionConfig):
    """Configuration for panel data functions."""
    
    def get_config(self) -> Dict[str, Any]:
        st.subheader(f"Configure {self.function_name}")
        
        if len(self.numeric_cols) < 3:
            st.error("Need at least 3 numeric variables for panel data analysis.")
            return {}
        
        # Dependent variable selection
        y_col = st.selectbox(
            "Select dependent variable (Y):",
            self.numeric_cols,
            key=f"{self.function_name}_y"
        )
        
        # Independent variables selection
        remaining_cols = [col for col in self.numeric_cols if col != y_col]
        x_cols = st.multiselect(
            "Select independent variables (X):",
            remaining_cols,
            default=remaining_cols[:min(5, len(remaining_cols))],
            key=f"{self.function_name}_x"
        )
        
        if not x_cols:
            st.warning("Please select at least one independent variable.")
            return {}
        
        # Panel structure options
        st.markdown("**Panel Structure:**")
        col1, col2 = st.columns(2)
        
        with col1:
            n_entities = st.slider(
                "Number of entities:",
                min_value=2,
                max_value=min(20, len(self.df) // 5),
                value=min(10, len(self.df) // 5),
                key=f"{self.function_name}_entities"
            )
        
        with col2:
            create_panel = st.checkbox(
                "Create panel structure automatically",
                value=True,
                key=f"{self.function_name}_create_panel"
            )
        
        return {
            'y_col': y_col,
            'x_cols': x_cols,
            'n_entities': n_entities,
            'create_panel': create_panel
        }

class IVConfig(FunctionConfig):
    """Configuration for instrumental variables functions."""
    
    def get_config(self) -> Dict[str, Any]:
        st.subheader(f"Configure {self.function_name}")
        
        if len(self.numeric_cols) < 3:
            st.error("Need at least 3 numeric variables for IV estimation (Y, X, Z).")
            return {}
        
        # Dependent variable selection
        y_col = st.selectbox(
            "Select dependent variable (Y):",
            self.numeric_cols,
            key=f"{self.function_name}_y"
        )
        
        # Endogenous variable selection
        remaining_cols = [col for col in self.numeric_cols if col != y_col]
        x_col = st.selectbox(
            "Select endogenous variable (X):",
            remaining_cols,
            key=f"{self.function_name}_x"
        )
        
        # Instrument selection
        instrument_cols = [col for col in remaining_cols if col != x_col]
        z_col = st.selectbox(
            "Select instrument (Z):",
            instrument_cols,
            key=f"{self.function_name}_z"
        )
        
        # Control variables selection
        control_cols = [col for col in instrument_cols if col != z_col]
        control_vars = st.multiselect(
            "Select control variables (optional):",
            control_cols,
            default=control_cols[:min(3, len(control_cols))],
            key=f"{self.function_name}_controls"
        )
        
        return {
            'y_col': y_col,
            'x_col': x_col,
            'z_col': z_col,
            'control_vars': control_vars
        }

class GroupedAnalysisConfig(FunctionConfig):
    """Configuration for grouped analysis functions."""
    
    def get_config(self) -> Dict[str, Any]:
        st.subheader(f"Configure {self.function_name}")
        
        if len(self.all_cols) < 2:
            st.error("Need at least 2 variables for grouped analysis.")
            return {}
        
        # Grouping variable selection
        group_col = st.selectbox(
            "Select grouping variable:",
            self.all_cols,
            key=f"{self.function_name}_group"
        )
        
        # Analysis variable selection
        remaining_cols = [col for col in self.all_cols if col != group_col]
        var_col = st.selectbox(
            "Select variable to analyze:",
            remaining_cols,
            key=f"{self.function_name}_var"
        )
        
        # Grouping method for numeric variables
        if pd.api.types.is_numeric_dtype(self.df[group_col]):
            grouping_method = st.selectbox(
                "Grouping method:",
                ["Quartiles", "Quintiles", "Deciles", "Custom bins"],
                key=f"{self.function_name}_method"
            )
            
            if grouping_method == "Custom bins":
                n_bins = st.slider(
                    "Number of bins:",
                    min_value=2,
                    max_value=10,
                    value=4,
                    key=f"{self.function_name}_bins"
                )
            else:
                n_bins = {"Quartiles": 4, "Quintiles": 5, "Deciles": 10}[grouping_method]
        else:
            grouping_method = "Categories"
            n_bins = None
        
        return {
            'group_col': group_col,
            'var_col': var_col,
            'grouping_method': grouping_method,
            'n_bins': n_bins
        }

class CollapseConfig(FunctionConfig):
    """Configuration for collapse/aggregation functions."""
    
    def get_config(self) -> Dict[str, Any]:
        st.subheader(f"Configure {self.function_name}")
        
        if len(self.all_cols) < 2:
            st.error("Need at least 2 variables for collapse analysis.")
            return {}
        
        # Grouping variables selection
        group_cols = st.multiselect(
            "Select grouping variables:",
            self.all_cols,
            default=self.all_cols[:min(2, len(self.all_cols))],
            key=f"{self.function_name}_groups"
        )
        
        if not group_cols:
            st.warning("Please select at least one grouping variable.")
            return {}
        
        # Variables to aggregate
        remaining_cols = [col for col in self.all_cols if col not in group_cols]
        agg_cols = st.multiselect(
            "Select variables to aggregate:",
            remaining_cols,
            default=remaining_cols[:min(5, len(remaining_cols))],
            key=f"{self.function_name}_agg"
        )
        
        if not agg_cols:
            st.warning("Please select at least one variable to aggregate.")
            return {}
        
        # Aggregation functions
        agg_functions = st.multiselect(
            "Select aggregation functions:",
            ["mean", "sum", "count", "std", "min", "max", "median"],
            default=["mean", "count"],
            key=f"{self.function_name}_funcs"
        )
        
        if not agg_functions:
            st.warning("Please select at least one aggregation function.")
            return {}
        
        return {
            'group_cols': group_cols,
            'agg_cols': agg_cols,
            'agg_functions': agg_functions
        }

class AvplotConfig(FunctionConfig):
    """Configuration for added-variable plots."""
    
    def get_config(self) -> Dict[str, Any]:
        st.subheader(f"Configure {self.function_name}")
        
        if len(self.numeric_cols) < 2:
            st.error("Need at least 2 numeric variables for added-variable plots.")
            return {}
        
        # Dependent variable selection
        y_col = st.selectbox(
            "Select dependent variable (Y):",
            self.numeric_cols,
            key=f"{self.function_name}_y"
        )
        
        # Variable to plot
        remaining_cols = [col for col in self.numeric_cols if col != y_col]
        x_col = st.selectbox(
            "Select variable to plot (X):",
            remaining_cols,
            key=f"{self.function_name}_x"
        )
        
        # Control variables
        control_cols = [col for col in remaining_cols if col != x_col]
        control_vars = st.multiselect(
            "Select control variables (optional):",
            control_cols,
            default=control_cols[:min(5, len(control_cols))],
            key=f"{self.function_name}_controls"
        )
        
        return {
            'y_col': y_col,
            'x_col': x_col,
            'control_vars': control_vars
        }

# Configuration mapping for each function
FUNCTION_CONFIGS = {
    'Regress': RegressionConfig,
    'Xtreg': PanelDataConfig,
    'Areg': PanelDataConfig,
    'Hausman': PanelDataConfig,
    'Ivregress': IVConfig,
    'Bysort': GroupedAnalysisConfig,
    'Collapse': CollapseConfig,
    'Avplot': AvplotConfig,
    'Margins': RegressionConfig,  # Can reuse regression config
}

def get_function_config(function_name: str, df: pd.DataFrame) -> Dict[str, Any]:
    """Get configuration for a specific function."""
    if function_name in FUNCTION_CONFIGS:
        config_class = FUNCTION_CONFIGS[function_name]
        config_instance = config_class(function_name, df)
        return config_instance.get_config()
    else:
        return {}  # No configuration needed
