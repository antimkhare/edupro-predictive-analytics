"""
features.py
===========
Defines the feature matrix (X) and helpers used by every model module.
"""

import pandas as pd
import numpy as np

# ── Feature columns used in all ML models ─────────────────────────────────────
FEATURE_COLS = [
    "CoursePrice",
    "CourseDuration",
    "CourseRating",
    "YearsOfExperience",
    "TeacherRating",
    "ExpertiseMatch",
    "CourseCategory_enc",
    "CourseType_enc",
    "CourseLevel_enc",
]

# Human-readable labels for charts
FEATURE_LABELS = {
    "CoursePrice":         "Course Price",
    "CourseDuration":      "Course Duration",
    "CourseRating":        "Course Rating",
    "YearsOfExperience":   "Teacher Experience (yrs)",
    "TeacherRating":       "Teacher Rating",
    "ExpertiseMatch":      "Expertise Match",
    "CourseCategory_enc":  "Course Category",
    "CourseType_enc":      "Course Type",
    "CourseLevel_enc":     "Course Level",
}

TARGET_ENROLLMENT = "EnrollmentCount"
TARGET_REVENUE    = "CourseRevenue"


def get_X(merged: pd.DataFrame) -> pd.DataFrame:
    """Return the feature matrix, zero-filling any NaN."""
    return merged[FEATURE_COLS].fillna(0)


def get_y(merged: pd.DataFrame, target: str) -> pd.Series:
    """Return a target vector, zero-filling NaN."""
    return merged[target].fillna(0)


def summarise_features(merged: pd.DataFrame) -> pd.DataFrame:
    """Descriptive statistics for every model feature."""
    X = get_X(merged)
    desc = X.describe().T
    desc.index = [FEATURE_LABELS.get(i, i) for i in desc.index]
    return desc
