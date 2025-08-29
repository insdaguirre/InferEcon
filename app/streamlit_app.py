from __future__ import annotations

import base64
import io
from typing import List

import pandas as pd
import streamlit as st

# Import should work now since this script is launched by run_app.py
from analysis_functions import load_functions

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

run = st.sidebar.button("Run")

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
