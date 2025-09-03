from __future__ import annotations

import base64
import io
from typing import List, Dict, Any

import pandas as pd
import streamlit as st

# Import should work now since this script is launched by run_app.py
from analysis_functions import load_functions
from analysis_functions.config_interface import get_function_config, FUNCTION_CONFIGS

st.set_page_config(page_title="Econometrics Toolkit", layout="wide")
st.title("Econometrics Toolkit")

# Load available functions
available_funcs = load_functions()

st.sidebar.header("1) Upload CSV")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"]) 

st.sidebar.header("2) Choose functions")
selected = []
for func in available_funcs:
    if st.sidebar.checkbox(func.display_name, value=(func.display_name == "Summarize")):
        selected.append(func)

# Show configuration for selected functions that need it
configurations: Dict[str, Dict[str, Any]] = {}
if uploaded_file is not None and selected:
    try:
        df = pd.read_csv(uploaded_file)
        
        # Check if any selected functions need configuration
        config_needed = [func for func in selected if func.display_name in FUNCTION_CONFIGS]
        
        if config_needed:
            st.header("3) Configure Functions")
            st.markdown("Some functions require additional configuration. Please specify the variables and parameters below:")
            
            for func in config_needed:
                with st.expander(f"Configure {func.display_name}", expanded=True):
                    config = get_function_config(func.display_name, df)
                    if config:  # Only store if configuration was successful
                        configurations[func.display_name] = config
                    else:
                        st.warning(f"Configuration incomplete for {func.display_name}")
    except Exception as exc:
        st.error(f"Failed to read CSV for configuration: {exc}")

run = st.sidebar.button("Run Analysis")

def _build_report_html(outputs: List[dict]) -> str:
    parts = [
        "<html><head><meta charset='utf-8'><title>Report</title>",
        "<style>body{font-family:system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;padding:24px} table{border-collapse:collapse} th,td{border:1px solid #ddd;padding:6px 10px} th{background:#f7f7f7}</style>",
        "</head><body>",
        "<h1>Econometrics Toolkit Report</h1>",
    ]
    for item in outputs:
        parts.append(f"<h2>{item.get('title','Output')}</h2>")
        if item["type"] == "table":
            parts.append(item["data"].to_html(index=False))
        elif item["type"] == "image":
            parts.append(f"<img src='data:image/png;base64,{item['data']}' />")
    parts.append("</body></html>")
    return "".join(parts)


if run:
    if uploaded_file is None:
        st.warning("Please upload a CSV first.")
    else:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as exc:
            st.error(f"Failed to read CSV: {exc}")
            st.stop()

        all_outputs: List[dict] = []
        for func in selected:
            try:
                # Check if this function needs configuration
                if func.display_name in configurations:
                    # Pass configuration to the function
                    config = configurations[func.display_name]
                    if hasattr(func, 'apply_with_config'):
                        outputs = func.apply_with_config(df, config)
                    else:
                        # Fallback to regular apply if function doesn't support config yet
                        outputs = func.apply(df)
                else:
                    # No configuration needed
                    outputs = func.apply(df)
                
                all_outputs.extend(outputs)
            except Exception as exc:
                st.error(f"{func.display_name} failed: {exc}")

        st.subheader("Results")
        for item in all_outputs:
            st.markdown(f"### {item.get('title','Output')}")
            if item["type"] == "table":
                st.dataframe(item["data"], use_container_width=True)
            elif item["type"] == "image":
                st.image(base64.b64decode(item["data"]))
            elif item["type"] == "text":
                st.write(item["data"])

        if all_outputs:
            html = _build_report_html(all_outputs)
            st.download_button(
                label="Download HTML report",
                data=html,
                file_name="report.html",
                mime="text/html",
            )

st.markdown("""
Tip: To add new analysis functions, create a new module in `analysis_functions/`
ending with `_func.py` that defines `display_name: str` and `apply(df) -> List[dict]`.
Return items like `{ "type": "table", "title": "Your Title", "data": pandas.DataFrame }`.
""")
