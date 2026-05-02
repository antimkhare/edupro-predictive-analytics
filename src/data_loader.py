"""
data_loader.py
==============
Loads and merges the three EduPro sheets (Courses, Teachers, Transactions),
performs feature engineering, and returns a clean analysis-ready DataFrame.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


DATA_PATH = "data/EduPro_Online_Platform.xlsx"


def load_raw_sheets(path: str = DATA_PATH) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all three raw Excel sheets."""
    xl = pd.read_excel(path, sheet_name=None)
    courses      = xl["Courses"].copy()
    teachers     = xl["Teachers"].copy()
    transactions = xl["Transactions"].copy()
    transactions["TransactionDate"] = pd.to_datetime(transactions["TransactionDate"])
    return courses, teachers, transactions


def aggregate_transactions(transactions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate transactions to course level."""
    agg = transactions.groupby("CourseID").agg(
        EnrollmentCount=("TransactionID", "count"),
        CourseRevenue=("Amount", "sum"),
    ).reset_index()
    teacher_per_course = (
        transactions.groupby("CourseID")["TeacherID"].first().reset_index()
    )
    return agg.merge(teacher_per_course, on="CourseID", how="left")


def build_merged(path: str = DATA_PATH) -> pd.DataFrame:
    """
    Build the fully merged, feature-engineered DataFrame.

    Returns
    -------
    pd.DataFrame
        One row per course with all original + engineered features.
    """
    courses, teachers, transactions = load_raw_sheets(path)

    txn_agg = aggregate_transactions(transactions)

    merged = (
        courses
        .merge(txn_agg, on="CourseID", how="left")
        .merge(
            teachers[["TeacherID", "YearsOfExperience", "TeacherRating", "Expertise"]],
            on="TeacherID", how="left",
        )
    )

    # ── Feature Engineering ────────────────────────────────────────────────────
    # Price bands
    merged["PriceBand"] = pd.cut(
        merged["CoursePrice"],
        bins=[-1, 0, 150, 350, 600],
        labels=["Free", "Low", "Medium", "High"],
    )

    # Duration buckets
    merged["DurationBucket"] = pd.cut(
        merged["CourseDuration"],
        bins=[0, 15, 30, 45, 200],
        labels=["Short", "Medium", "Long", "Extended"],
    )

    # Rating tiers
    merged["RatingTier"] = pd.cut(
        merged["CourseRating"],
        bins=[0, 2, 3, 4, 5],
        labels=["Poor", "Average", "Good", "Excellent"],
    )

    # Experience buckets
    merged["ExperienceBucket"] = pd.cut(
        merged["YearsOfExperience"],
        bins=[0, 3, 7, 12, 50],
        labels=["Junior", "Mid", "Senior", "Expert"],
    )

    # Expertise–category match
    merged["ExpertiseMatch"] = (
        merged["Expertise"] == merged["CourseCategory"]
    ).astype(int)

    # Revenue per enrolment
    merged["RevenuePerEnrollment"] = (
        merged["CourseRevenue"] / merged["EnrollmentCount"]
    ).replace([np.inf, -np.inf], 0).fillna(0)

    # Label encoding (for ML)
    le = LabelEncoder()
    for col in ["CourseCategory", "CourseType", "CourseLevel"]:
        merged[f"{col}_enc"] = le.fit_transform(merged[col].astype(str))

    return merged


def get_monthly_trends(transactions: pd.DataFrame) -> pd.DataFrame:
    """Return monthly revenue and enrollment aggregates."""
    t = transactions.copy()
    t["Month"] = t["TransactionDate"].dt.to_period("M").astype(str)
    return (
        t.groupby("Month")
        .agg(Revenue=("Amount", "sum"), Enrollments=("TransactionID", "count"))
        .reset_index()
        .sort_values("Month")
    )


def get_category_summary(merged: pd.DataFrame) -> pd.DataFrame:
    """Return category-level aggregated summary."""
    cat = merged.groupby("CourseCategory").agg(
        TotalRevenue=("CourseRevenue", "sum"),
        TotalEnrollments=("EnrollmentCount", "sum"),
        AvgRating=("CourseRating", "mean"),
        AvgPrice=("CoursePrice", "mean"),
        NumCourses=("CourseID", "count"),
    ).reset_index()
    cat["RevenuePerEnrollment"] = cat["TotalRevenue"] / cat["TotalEnrollments"]
    return cat.sort_values("TotalRevenue", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    merged = build_merged()
    print(f"Merged shape : {merged.shape}")
    print(f"Columns      : {list(merged.columns)}")
    print(merged[["CourseName", "EnrollmentCount", "CourseRevenue"]].head())
