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
        
        if len(self.numeric_cols) < 2:
            st.error("Need at least 2 numeric variables to configure panel models.")
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
        id_guess = next((c for c in self.all_cols if c.lower() in ["id", "entity", "panel", "firm", "country"]), None)
        time_guess = next((c for c in self.all_cols if c.lower() in ["time", "year", "date", "period", "t"]), None)
        has_existing = id_guess is not None and time_guess is not None
        
        use_existing = st.checkbox(
            "Use existing panel identifiers (entity/time)",
            value=has_existing,
            key=f"{self.function_name}_use_existing"
        )
        
        entity_col: Optional[str] = None
        time_col: Optional[str] = None
        n_entities: Optional[int] = None
        create_panel = not use_existing
        
        if use_existing:
            col1, col2 = st.columns(2)
            with col1:
                entity_col = st.selectbox(
                    "Entity ID column:",
                    self.all_cols,
                    index=self.all_cols.index(id_guess) if id_guess in self.all_cols else 0,
                    key=f"{self.function_name}_entity"
                )
            with col2:
                time_col = st.selectbox(
                    "Time column:",
                    self.all_cols,
                    index=self.all_cols.index(time_guess) if time_guess in self.all_cols else 0,
                    key=f"{self.function_name}_time"
                )
        else:
            col1, col2 = st.columns(2)
            with col1:
                n_entities = st.slider(
                    "Number of entities to create:",
                    min_value=2,
                    max_value=max(2, min(50, max(2, len(self.df) // 3))),
                    value=min(10, max(2, len(self.df) // 5)),
                    key=f"{self.function_name}_entities"
                )
            with col2:
                st.caption("Panel identifiers will be auto-generated as `entity_id` and `time_id`.")
        
        # Optional: additional fixed effects selection (useful for Areg)
        fe_cols = st.multiselect(
            "Fixed effects to absorb (optional):",
            [c for c in self.all_cols if c not in [y_col] + x_cols],
            default=[],
            key=f"{self.function_name}_fe_cols"
        )
        
        return {
            'y_col': y_col,
            'x_cols': x_cols,
            'entity_col': entity_col,
            'time_col': time_col,
            'n_entities': n_entities,
            'create_panel': create_panel,
            'fe_cols': fe_cols,
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


class ScatterConfig(FunctionConfig):
    """Configuration for scatter plots and simple y~x visuals."""
    def get_config(self) -> Dict[str, Any]:
        st.subheader(f"Configure {self.function_name}")
        if len(self.numeric_cols) < 2:
            st.error("Need at least 2 numeric variables for visualization.")
            return {}
        y_col = st.selectbox(
            "Select dependent variable (Y):",
            self.numeric_cols,
            key=f"{self.function_name}_y"
        )
        remaining = [c for c in self.numeric_cols if c != y_col]
        x_cols = st.multiselect(
            "Select X variables to plot against Y:",
            remaining,
            default=remaining[:min(5, len(remaining))],
            key=f"{self.function_name}_xs"
        )
        if not x_cols:
            st.warning("Please select at least one X variable.")
            return {}
        return { 'y_col': y_col, 'x_cols': x_cols }

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
    'Scatter': ScatterConfig,
    'Twoway': ScatterConfig,
}

def get_function_config(function_name: str, df: pd.DataFrame) -> Dict[str, Any]:
    """Get configuration for a specific function."""
    if function_name in FUNCTION_CONFIGS:
        config_class = FUNCTION_CONFIGS[function_name]
        config_instance = config_class(function_name, df)
        return config_instance.get_config()
    else:
        return {}  # No configuration needed
