"""
visualizations.py
=================
All Plotly chart functions used by the Streamlit app.
Each function returns a plotly.graph_objects.Figure.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PALETTE = px.colors.qualitative.Bold

LAYOUT_DEFAULTS = dict(
    plot_bgcolor="white",
    paper_bgcolor="white",
    font_family="Inter, Arial, sans-serif",
    font_color="#1e293b",
    margin=dict(t=32, b=16, l=16, r=16),
)


# ── EDA Charts ────────────────────────────────────────────────────────────────

def fig_monthly_trend(monthly: pd.DataFrame) -> go.Figure:
    """Dual-axis bar + line: monthly revenue and enrollments."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=monthly["Month"], y=monthly["Revenue"],
            name="Revenue ($)", marker_color="rgba(59,130,246,0.70)",
            marker_line_width=0,
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=monthly["Month"], y=monthly["Enrollments"],
            name="Enrollments", mode="lines+markers",
            line=dict(color="#f97316", width=2.5),
            marker=dict(size=6),
        ),
        secondary_y=True,
    )
    fig.update_layout(**LAYOUT_DEFAULTS, title="Monthly Revenue & Enrollment Trend",
                      legend=dict(orientation="h", y=1.1))
    fig.update_yaxes(title_text="Revenue ($)", secondary_y=False)
    fig.update_yaxes(title_text="Enrollments",  secondary_y=True)
    return fig


def fig_category_revenue(cat_df: pd.DataFrame) -> go.Figure:
    """Horizontal bar: total revenue by category."""
    fig = px.bar(
        cat_df.sort_values("TotalRevenue"),
        x="TotalRevenue", y="CourseCategory",
        orientation="h",
        color="TotalRevenue",
        color_continuous_scale="Blues",
        labels={"TotalRevenue": "Revenue ($)", "CourseCategory": ""},
        title="Total Revenue by Category",
    )
    fig.update_layout(**LAYOUT_DEFAULTS, coloraxis_showscale=False, showlegend=False)
    return fig


def fig_category_enrollments(cat_df: pd.DataFrame) -> go.Figure:
    """Horizontal bar: total enrollments by category."""
    fig = px.bar(
        cat_df.sort_values("TotalEnrollments"),
        x="TotalEnrollments", y="CourseCategory",
        orientation="h",
        color="TotalEnrollments",
        color_continuous_scale="Greens",
        labels={"TotalEnrollments": "Enrollments", "CourseCategory": ""},
        title="Total Enrollments by Category",
    )
    fig.update_layout(**LAYOUT_DEFAULTS, coloraxis_showscale=False, showlegend=False)
    return fig


def fig_scatter_price_revenue(merged: pd.DataFrame) -> go.Figure:
    """Bubble scatter: price vs revenue, bubble = enrollment size."""
    paid = merged[merged["CoursePrice"] > 0].copy()
    fig = px.scatter(
        paid,
        x="CoursePrice", y="CourseRevenue",
        size="EnrollmentCount", color="CourseCategory",
        hover_name="CourseName",
        color_discrete_sequence=PALETTE,
        size_max=40,
        labels={
            "CoursePrice":    "Course Price ($)",
            "CourseRevenue":  "Revenue ($)",
            "CourseCategory": "Category",
        },
        title="Course Price vs Revenue  (bubble = enrolment count)",
    )
    fig.update_layout(**LAYOUT_DEFAULTS, legend=dict(orientation="h", y=-0.18, x=0))
    return fig


def fig_rating_distribution(merged: pd.DataFrame) -> go.Figure:
    """Histogram of course ratings, coloured by category."""
    fig = px.histogram(
        merged, x="CourseRating", nbins=20,
        color="CourseCategory",
        color_discrete_sequence=PALETTE,
        barmode="overlay",
        labels={"CourseRating": "Course Rating (1–5)"},
        title="Distribution of Course Ratings",
    )
    fig.update_layout(**LAYOUT_DEFAULTS, showlegend=False)
    return fig


def fig_course_type_pie(merged: pd.DataFrame) -> go.Figure:
    """Donut: revenue split by course type."""
    df = merged.groupby("CourseType")["CourseRevenue"].sum().reset_index()
    fig = px.pie(
        df, names="CourseType", values="CourseRevenue",
        color_discrete_sequence=["#3b82f6", "#f97316"],
        hole=0.45,
        title="Revenue Share by Course Type",
    )
    fig.update_layout(**LAYOUT_DEFAULTS)
    return fig


def fig_level_enrollments(merged: pd.DataFrame) -> go.Figure:
    """Bar: enrollments by course level."""
    df = merged.groupby("CourseLevel")["EnrollmentCount"].sum().reset_index()
    fig = px.bar(
        df, x="CourseLevel", y="EnrollmentCount",
        color="CourseLevel",
        color_discrete_sequence=["#10b981", "#3b82f6", "#f97316"],
        labels={"EnrollmentCount": "Enrollments", "CourseLevel": ""},
        title="Enrollments by Course Level",
    )
    fig.update_layout(**LAYOUT_DEFAULTS, showlegend=False)
    return fig


# ── Model Evaluation Charts ───────────────────────────────────────────────────

def fig_r2_comparison(metrics_df: pd.DataFrame, best_name: str) -> go.Figure:
    """Bar chart of R² scores, highlighting the best model."""
    colors = [
        "#10b981" if m == best_name else "#94a3b8"
        for m in metrics_df["Model"]
    ]
    fig = go.Figure(go.Bar(
        x=metrics_df["Model"], y=metrics_df["R2"],
        marker_color=colors, text=metrics_df["R2"].round(4),
        textposition="outside",
    ))
    fig.update_layout(**LAYOUT_DEFAULTS, title="R² Score — All Models",
                      yaxis_title="R² Score", showlegend=False)
    return fig


def fig_error_comparison(metrics_df: pd.DataFrame) -> go.Figure:
    """Grouped bar: MAE vs RMSE for all models."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="MAE", x=metrics_df["Model"], y=metrics_df["MAE"],
        marker_color="rgba(59,130,246,0.78)",
    ))
    fig.add_trace(go.Bar(
        name="RMSE", x=metrics_df["Model"], y=metrics_df["RMSE"],
        marker_color="rgba(249,115,22,0.78)",
    ))
    fig.update_layout(**LAYOUT_DEFAULTS, barmode="group",
                      title="MAE vs RMSE — All Models",
                      legend=dict(orientation="h", y=1.1))
    return fig


# ── Feature Importance Charts ─────────────────────────────────────────────────

def fig_importance_bar(importances: pd.DataFrame, title: str, color_scale: str = "Blues") -> go.Figure:
    """Horizontal bar: feature importances."""
    fig = px.bar(
        importances,
        x="Importance", y="FeatureLabel",
        orientation="h",
        color="Importance",
        color_continuous_scale=color_scale,
        text=importances["Importance"].map(lambda v: f"{v*100:.1f}%"),
        title=title,
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        **LAYOUT_DEFAULTS,
        coloraxis_showscale=False,
        showlegend=False,
        yaxis={"categoryorder": "total ascending"},
    )
    return fig


def fig_importance_radar(enr_imp: pd.DataFrame, rev_imp: pd.DataFrame) -> go.Figure:
    """Radar chart comparing enrollment vs revenue feature importances."""
    feats  = enr_imp["FeatureLabel"].tolist()
    enr_r  = enr_imp["Importance"].tolist()
    rev_map = dict(zip(rev_imp["FeatureLabel"], rev_imp["Importance"]))
    rev_r  = [rev_map.get(f, 0) for f in feats]

    def close(lst):
        return lst + [lst[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=close([v * 100 for v in enr_r]), theta=close(feats),
        fill="toself", name="Enrollment",
        line_color="#3b82f6", fillcolor="rgba(59,130,246,0.15)",
    ))
    fig.add_trace(go.Scatterpolar(
        r=close([v * 100 for v in rev_r]), theta=close(feats),
        fill="toself", name="Revenue",
        line_color="#f97316", fillcolor="rgba(249,115,22,0.12)",
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, ticksuffix="%")),
        legend=dict(orientation="h", y=1.1),
        font_family="Inter, Arial, sans-serif",
        paper_bgcolor="white",
        title="Enrollment vs Revenue — Feature Importance Radar",
    )
    return fig


# ── Insights Chart ────────────────────────────────────────────────────────────

def fig_revenue_gap(merged: pd.DataFrame) -> go.Figure:
    """Bar: untapped revenue potential by freemium conversion ($50 avg)."""
    df = merged.groupby("CourseCategory").agg(
        CurrentRevenue=("CourseRevenue", "sum"),
        Enrollments=("EnrollmentCount", "sum"),
    ).reset_index()
    df["RevenueGap"] = df["Enrollments"] * 50 - df["CurrentRevenue"]
    df = df[df["RevenueGap"] > 0].sort_values("RevenueGap", ascending=False)

    fig = px.bar(
        df, x="CourseCategory", y="RevenueGap",
        color="RevenueGap", color_continuous_scale="Reds",
        labels={"RevenueGap": "Revenue Gap ($)", "CourseCategory": ""},
        title="Untapped Revenue Potential — Freemium Conversion at $50",
    )
    fig.update_layout(**LAYOUT_DEFAULTS, coloraxis_showscale=False, showlegend=False)
    return fig
