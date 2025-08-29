from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd


display_name = "Summarize"


def _is_numeric_column(series: pd.Series) -> bool:
    # Consider only non-empty cells (after trimming strings)
    mask_non_empty = series.astype(str).str.strip().ne("")
    if mask_non_empty.sum() == 0:
        return False
    coerced = pd.to_numeric(series[mask_non_empty], errors="coerce")
    # Numeric iff all non-empty values convert to numbers
    return coerced.notna().all()


def apply(df: pd.DataFrame) -> List[dict]:
    """Return a list of outputs for Streamlit to render.

    Output item schema:
    {"type": "table", "title": str, "data": pd.DataFrame}
    """
    rows = []
    for col in df.columns:
        series = df[col]
        # Count non-empty observations
        n = series.astype(str).str.strip().ne("").sum()
        if _is_numeric_column(series):
            mask_non_empty = series.astype(str).str.strip().ne("")
            vals = pd.to_numeric(series[mask_non_empty], errors="coerce").astype(float)
            mean = vals.mean()
            std = vals.std(ddof=1) if len(vals) > 1 else np.nan
            vmin = vals.min() if len(vals) else np.nan
            vmax = vals.max() if len(vals) else np.nan
            rows.append([col, int(n), mean, std, vmin, vmax])
        else:
            rows.append([col, int(n), np.nan, np.nan, np.nan, np.nan])

    out_df = pd.DataFrame(rows, columns=["Variable", "N", "Mean", "Std. Dev.", "Min", "Max"])
    return [{"type": "table", "title": "Summarize", "data": out_df}]
