"""
Helper to read the latest prediction artifact from predictions/<date>/daily_report.json.
Used by the Streamlit app to display the last nightly inference results.
"""

import json
import os
from glob import glob
from datetime import datetime


PREDICTIONS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "predictions")


def get_latest_report():
    """
    Load the most recent daily_report.json.
    Returns a dict or None if no artifacts exist yet.
    """
    pattern = os.path.join(PREDICTIONS_DIR, "*", "daily_report.json")
    files = sorted(glob(pattern))   # Sorted alphabetically → chronologically by date prefix
    if not files:
        return None
    with open(files[-1]) as f:
        return json.load(f)


def get_all_reports():
    """
    Load all historical daily reports for trend analysis.
    Returns a list of dicts, sorted oldest-first.
    """
    pattern = os.path.join(PREDICTIONS_DIR, "*", "daily_report.json")
    files = sorted(glob(pattern))
    reports = []
    for fp in files:
        try:
            with open(fp) as f:
                reports.append(json.load(f))
        except Exception:
            continue
    return reports
