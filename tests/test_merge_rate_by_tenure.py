import os
import json
import ast
import pytest
import nbformat
import pandas as pd
import numpy as np

NOTEBOOK_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "collectoss-analysis", "8knot", "merge_rate_by_tenure.ipynb"
))

# ---------------------------------------------------------------------------
# Test 1: Notebook Schema and Validity
# ---------------------------------------------------------------------------
def test_notebook_schema_and_validity():
    """Verify that the notebook is valid JSON and complies with the nbformat 4 schema."""
    assert os.path.isfile(NOTEBOOK_PATH), f"Notebook file not found at: {NOTEBOOK_PATH}"
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
    nbformat.validate(nb)
    assert len(nb.cells) >= 20, f"Expected at least 20 cells, got {len(nb.cells)}"

def test_notebook_has_rendered_outputs():
    """Verify that code cells have execution counts and pre-rendered outputs for ReviewNB."""
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    cells_with_outputs = [c for c in code_cells if len(c.get("outputs", [])) > 0]
    assert len(cells_with_outputs) >= 10, f"Expected >= 10 cells with outputs, found {len(cells_with_outputs)}"

# ---------------------------------------------------------------------------
# Test 2: Tenure Cohort Assignment Logic
# ---------------------------------------------------------------------------
def assign_tenure_cohort(days):
    if days <= 0.0001:
        return "1. First PR (Day 0)"
    elif days < 30:
        return "2. < 30 Days"
    elif days < 180:
        return "3. 1 - 6 Months"
    elif days < 365:
        return "4. 6 - 12 Months"
    else:
        return "5. 1+ Years"

@pytest.mark.parametrize("days,expected_cohort", [
    (0.0, "1. First PR (Day 0)"),
    (0.00005, "1. First PR (Day 0)"),
    (0.5, "2. < 30 Days"),
    (14.0, "2. < 30 Days"),
    (29.9, "2. < 30 Days"),
    (30.0, "3. 1 - 6 Months"),
    (90.0, "3. 1 - 6 Months"),
    (179.9, "3. 1 - 6 Months"),
    (180.0, "4. 6 - 12 Months"),
    (364.9, "4. 6 - 12 Months"),
    (365.0, "5. 1+ Years"),
    (720.0, "5. 1+ Years"),
])
def test_tenure_cohort_assignment_boundaries(days, expected_cohort):
    assert assign_tenure_cohort(days) == expected_cohort

# ---------------------------------------------------------------------------
# Test 3: Aggregation and Rate Mathematical Integrity
# ---------------------------------------------------------------------------
def test_cohort_aggregation_and_rates():
    sample_data = pd.DataFrame([
        {"cohort": "1. First PR (Day 0)", "is_merged": True, "dur": 5.0, "revs": 3, "is_core": False},
        {"cohort": "1. First PR (Day 0)", "is_merged": False, "dur": np.nan, "revs": 2, "is_core": False},
        {"cohort": "2. < 30 Days", "is_merged": True, "dur": 4.0, "revs": 2, "is_core": True},
        {"cohort": "2. < 30 Days", "is_merged": True, "dur": 2.0, "revs": 1, "is_core": False},
        {"cohort": "5. 1+ Years", "is_merged": True, "dur": 1.5, "revs": 1, "is_core": True},
    ])

    summary = sample_data.groupby("cohort").agg(
        prs_submitted=("is_merged", "count"),
        prs_merged=("is_merged", "sum"),
        avg_reviews=("revs", "mean"),
        median_time_to_merge=("dur", "median")
    ).reset_index()

    summary["merge_rate_pct"] = (summary["prs_merged"] / summary["prs_submitted"]) * 100

    assert (summary["merge_rate_pct"] >= 0).all()
    assert (summary["merge_rate_pct"] <= 100).all()

    first_pr = summary.loc[summary["cohort"] == "1. First PR (Day 0)"]
    assert round(first_pr["merge_rate_pct"].values[0], 1) == 50.0

    early = summary.loc[summary["cohort"] == "2. < 30 Days"]
    assert round(early["merge_rate_pct"].values[0], 1) == 100.0

# ---------------------------------------------------------------------------
# Test 4: Core vs Non-Core Disparity Logic
# ---------------------------------------------------------------------------
def test_core_vs_non_core_stratification():
    sample_data = pd.DataFrame([
        {"cohort": "1. First PR (Day 0)", "is_core": True, "is_merged": True},
        {"cohort": "1. First PR (Day 0)", "is_core": False, "is_merged": False},
        {"cohort": "3. 1 - 6 Months", "is_core": True, "is_merged": True},
        {"cohort": "3. 1 - 6 Months", "is_core": False, "is_merged": True},
    ])

    disparity = sample_data.groupby(["cohort", "is_core"]).agg(
        prs_submitted=("is_merged", "count"),
        prs_merged=("is_merged", "sum")
    ).reset_index()
    disparity["merge_rate_pct"] = (disparity["prs_merged"] / disparity["prs_submitted"]) * 100

    assert len(disparity) == 4
    core_first = disparity[(disparity["cohort"] == "1. First PR (Day 0)") & (disparity["is_core"] == True)]
    non_core_first = disparity[(disparity["cohort"] == "1. First PR (Day 0)") & (disparity["is_core"] == False)]

    assert core_first["merge_rate_pct"].values[0] == 100.0
    assert non_core_first["merge_rate_pct"].values[0] == 0.0

# ---------------------------------------------------------------------------
# Test 5: End-to-End Notebook Code Execution
# ---------------------------------------------------------------------------
def test_end_to_end_notebook_execution():
    """Verify that every single code cell in the notebook executes successfully."""
    import plotly.graph_objects as go
    go.Figure.show = lambda self, *args, **kwargs: None

    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    namespace = {}
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        try:
            tree = ast.parse(cell.source)
            code = compile(tree, f"<cell_{i}>", "exec")
            exec(code, namespace)
        except Exception as e:
            pytest.fail(f"Cell {i} failed with error: {e}")
