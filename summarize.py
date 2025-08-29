from __future__ import annotations

import csv
import math
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class NumericAccumulator:
    """Online (Welford) accumulator for mean, variance, min, and max."""

    count: int = 0
    mean: float = 0.0
    m2: float = 0.0  # Sum of squares of differences from the current mean
    minimum: Optional[float] = None
    maximum: Optional[float] = None

    def add(self, value: float) -> None:
        self.count += 1
        # Update min/max
        if self.minimum is None or value < self.minimum:
            self.minimum = value
        if self.maximum is None or value > self.maximum:
            self.maximum = value

        # Welford's online algorithm for mean and variance
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2

    def std_dev_sample(self) -> Optional[float]:
        if self.count <= 1:
            return None
        variance = self.m2 / (self.count - 1)
        return math.sqrt(variance)


@dataclass
class ColumnStats:
    non_empty_count: int = 0
    numeric_acc: NumericAccumulator = NumericAccumulator()
    non_numeric_non_empty_count: int = 0

    def is_numeric_column(self) -> bool:
        # Treat as numeric iff all non-empty cells are numeric
        return self.non_empty_count > 0 and self.non_numeric_non_empty_count == 0 and self.numeric_acc.count > 0


def _format_number(value: Optional[float]) -> str:
    if value is None:
        return ""
    # Use general format with up to 6 significant digits, similar to compact Stata output
    try:
        return f"{value:.6g}"
    except Exception:
        return str(value)


def _render_table(rows: List[List[str]], headers: List[str]) -> str:
    # Compute column widths
    columns = list(zip(*([headers] + rows))) if rows else [headers]
    widths: List[int] = [max(len(str(cell)) for cell in col) for col in columns]

    def render_line(cells: List[str]) -> str:
        return "  ".join(str(cell).rjust(width) for cell, width in zip(cells, widths))

    lines: List[str] = []
    lines.append(render_line(headers))
    lines.append(render_line(["-" * w for w in widths]))
    for row in rows:
        lines.append(render_line(row))
    return "\n".join(lines)


def summarize_csv(csv_path: str) -> str:
    """
    Read a CSV and compute, per column, the count of non-empty observations.
    For columns where all non-empty cells are numeric, also compute mean,
    sample standard deviation (n-1), minimum, and maximum. Returns a formatted table string.

    This function does not load the entire file into memory; it streams rows.
    """
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header row.")

        stats: Dict[str, ColumnStats] = {name: ColumnStats() for name in reader.fieldnames}

        for row in reader:
            for col_name in reader.fieldnames:
                raw = row.get(col_name, "")
                if raw is None:
                    raw = ""
                val_str = str(raw).strip()
                if val_str == "":
                    # Empty cell
                    continue
                col_stats = stats[col_name]
                col_stats.non_empty_count += 1
                # Try to parse as float
                try:
                    num = float(val_str)
                except ValueError:
                    col_stats.non_numeric_non_empty_count += 1
                    continue
                col_stats.numeric_acc.add(num)

    # Build rows for the table
    headers = ["Variable", "N", "Mean", "Std. Dev.", "Min", "Max"]
    rows: List[List[str]] = []
    for name, col_stats in stats.items():
        if col_stats.is_numeric_column():
            acc = col_stats.numeric_acc
            mean_str = _format_number(acc.mean)
            std_str = _format_number(acc.std_dev_sample())
            min_str = _format_number(acc.minimum)
            max_str = _format_number(acc.maximum)
            rows.append([
                name,
                str(col_stats.non_empty_count),
                mean_str,
                std_str,
                min_str,
                max_str,
            ])
        else:
            # Non-numeric columns: only show N; others blank
            rows.append([
                name,
                str(col_stats.non_empty_count),
                "",
                "",
                "",
                "",
            ])

    return _render_table(rows, headers)


def _main(argv: List[str]) -> int:
    if len(argv) < 2:
        print("Usage: python summarize.py <path-to-csv>")
        return 2
    path = argv[1]
    try:
        table = summarize_csv(path)
    except Exception as exc:
        print(f"Error: {exc}")
        return 1
    print(table)
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv))
